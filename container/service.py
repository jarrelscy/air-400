"""Baby monitor service with respiration rate estimation for Home Assistant."""

import os
import sys
import time
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from threading import Thread, Lock
import signal

import cv2
import numpy as np
import requests
import torch
import yaml
from ultralytics import YOLO

# Add parent directory to path for air-400 imports
sys.path.insert(0, '/app')

from models.vire_net import VIRENet
from processors.post_processor import PostProcessor
from processors.pre_processor import PreProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerspectiveCorrector:
    """Handles perspective correction by detecting non-zero content region."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.src_points: Optional[np.ndarray] = None
        self.dst_points: Optional[np.ndarray] = None
        self.transform_matrix: Optional[np.ndarray] = None
        self.output_size = config.get('output_size', (640, 480))
        self.calibrated = False
        self.calibration_frames = []
        self.calibration_count = config.get('calibration_frames', 10)
        self.last_calibration_time = 0
        self.recalibration_interval = config.get('recalibration_interval', 10)  # seconds
        self.on_recalibrate_callback = lambda: None  # Set by service to clear buffer

    def calibrate_from_frame(self, frame: np.ndarray) -> bool:
        """Auto-detect quadrilateral from non-zero region (0 = background)."""
        # Collect calibration frames for more robust detection
        self.calibration_frames.append(frame.copy())

        if len(self.calibration_frames) < self.calibration_count:
            return False

        # Use median frame to reduce noise
        median_frame = np.median(self.calibration_frames, axis=0).astype(np.uint8)
        self.calibration_frames = []

        # Detect non-zero region (background is 0)
        quad = self._detect_content_quadrilateral(median_frame)

        if quad is not None:
            self.set_quadrilateral(quad)
            self.calibrated = True
            logger.info(f"Auto-calibrated perspective from non-zero region: {quad.tolist()}")
            return True

        logger.warning("Could not detect content quadrilateral")
        return False

    def _detect_content_quadrilateral(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect the quadrilateral boundary of non-zero content."""
        # Convert to grayscale first
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.copy()

        # Set top rows to 0 (black) to ignore text overlay area
        scan_start_y = self.config.get('scan_start_y', 200)
        gray[:scan_start_y, :] = 0  # Mark top rows as background

        # Use Otsu's method to find optimal threshold for background detection
        otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info(f"Otsu threshold for background detection: {otsu_thresh}")

        # Create mask - pixels below Otsu threshold are background
        mask = (gray > otsu_thresh).astype(np.uint8) * 255

        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours of non-zero region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get largest contour (main content area)
        largest_contour = max(contours, key=cv2.contourArea)

        # Try different epsilon values to get a quadrilateral
        for eps_mult in [0.01, 0.02, 0.03, 0.04, 0.05]:
            epsilon = eps_mult * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                logger.info(f"Found quadrilateral with epsilon={eps_mult}")
                return self._order_points(pts)

        # If we can't get exactly 4 points, use convex hull and find 4 extreme points
        hull = cv2.convexHull(largest_contour)

        # Find the 4 corners from the hull
        hull_pts = hull.reshape(-1, 2)

        # Get extreme points
        top_left = hull_pts[np.argmin(hull_pts[:, 0] + hull_pts[:, 1])]
        top_right = hull_pts[np.argmin(-hull_pts[:, 0] + hull_pts[:, 1])]
        bottom_right = hull_pts[np.argmax(hull_pts[:, 0] + hull_pts[:, 1])]
        bottom_left = hull_pts[np.argmax(-hull_pts[:, 0] + hull_pts[:, 1])]

        pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        logger.info(f"Using convex hull extreme points for quadrilateral")
        return pts

    def set_quadrilateral(self, points: np.ndarray) -> None:
        """Set the source quadrilateral points for perspective correction."""
        self.src_points = np.float32(points)

        # Fixed output size 480x640
        w, h = 480, 640
        self.output_size = (w, h)
        logger.info(f"Output size: {w}x{h}")

        # No rotation - direct mapping
        self.dst_points = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        self.transform_matrix = cv2.getPerspectiveTransform(
            self.src_points, self.dst_points
        )

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect

    def correct(self, frame: np.ndarray) -> np.ndarray:
        """Apply perspective correction to frame (rotation is built into the transform)."""
        current_time = time.time()

        # Check if recalibration is needed
        if self.calibrated and (current_time - self.last_calibration_time) >= self.recalibration_interval:
            logger.info("Triggering recalibration...")
            self.calibrated = False
            self.calibration_frames = []
            self.on_recalibrate_callback()  # Clear frame buffer

        # Auto-calibrate if not yet done
        if not self.calibrated:
            self.calibrate_from_frame(frame)
            if not self.calibrated:
                return frame  # Return original while calibrating
            self.last_calibration_time = current_time

        if self.transform_matrix is None:
            return frame

        # Rotation is handled by the destination point mapping in set_quadrilateral
        return cv2.warpPerspective(
            frame, self.transform_matrix, self.output_size
        )


class BabyDetector:
    """Detects baby body and face using YOLO models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        body_path = config.get('body_detector_path', '/app/models_pretrained/detectors/yolov8m.pt')
        face_path = config.get('face_detector_path', '/app/models_pretrained/detectors/yolov8n-face.pt')

        self.body_detector = None
        self.face_detector = None

        try:
            if os.path.exists(body_path):
                self.body_detector = YOLO(body_path)
                logger.info(f"Loaded body detector from {body_path}")
            else:
                # Fall back to default YOLO
                self.body_detector = YOLO('yolov8m.pt')
                logger.info("Using default yolov8m.pt for body detection")
        except Exception as e:
            logger.error(f"Failed to load body detector: {e}")

        try:
            if os.path.exists(face_path):
                self.face_detector = YOLO(face_path)
                logger.info(f"Loaded face detector from {face_path}")
        except Exception as e:
            logger.warning(f"Failed to load face detector: {e}")

        self.body_conf_threshold = config.get('body_conf_threshold', 0.25)
        self.face_conf_threshold = config.get('face_conf_threshold', 0.5)

    def preprocess_for_detection(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess IR/grayscale images to improve detection.

        Applies CLAHE and pseudo-coloring to make IR images more recognizable
        to YOLO models trained on color images.
        """
        # Check if image appears to be IR/grayscale (low color variance)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Calculate color variance to detect IR images
            b, g, r = frame[:,:,0], frame[:,:,1], frame[:,:,2]
            color_diff = np.abs(r.astype(float) - g.astype(float)).mean() + \
                         np.abs(g.astype(float) - b.astype(float)).mean()

            is_ir = color_diff < 10  # Low color difference indicates IR/grayscale

            if is_ir:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # Apply CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)

                # Apply color map to create pseudo-color image
                # COLORMAP_BONE gives skin-like tones which may help person detection
                colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_BONE)
                frame = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                logger.debug("Applied CLAHE + COLORMAP_BONE preprocessing for IR image")

        return frame

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect baby body and face in frame."""
        result = {
            'body_detected': False,
            'face_detected': False,
            'body_box': None,
            'face_box': None,
            'body_confidence': 0.0,
            'face_confidence': 0.0
        }

        # Preprocess frame for better IR detection
        processed_frame = self.preprocess_for_detection(frame)

        # Log frame info for debugging
        logger.info(f"Detection input frame: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")

        # Detect body (person class = 0)
        if self.body_detector:
            try:
                detections = self.body_detector(
                    processed_frame,
                    conf=self.body_conf_threshold,
                    device=self.device,
                    verbose=False
                )[0]

                # Log ALL detections for debugging
                if len(detections.boxes) > 0:
                    for i, box in enumerate(detections.boxes):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        logger.info(f"Body detector found: class={cls_id} ({detections.names[cls_id]}), conf={conf:.3f}")
                else:
                    logger.info(f"Body detector found no objects above threshold {self.body_conf_threshold}")

                for box in detections.boxes:
                    if int(box.cls[0]) == 0:  # Person class
                        conf = float(box.conf[0])
                        if conf > result['body_confidence']:
                            result['body_detected'] = True
                            result['body_confidence'] = conf
                            result['body_box'] = box.xyxy[0].cpu().numpy().tolist()
            except Exception as e:
                logger.error(f"Body detection error: {e}")

        # Detect face
        if self.face_detector:
            try:
                detections = self.face_detector(
                    processed_frame,
                    conf=self.face_conf_threshold,
                    device=self.device,
                    verbose=False
                )[0]

                if len(detections.boxes) > 0:
                    # Log all face detections
                    for i, box in enumerate(detections.boxes):
                        conf = float(box.conf[0])
                        logger.info(f"Face detector found face {i+1}: conf={conf:.3f}")

                    # Get highest confidence face
                    best_idx = detections.boxes.conf.argmax()
                    box = detections.boxes[best_idx]
                    result['face_detected'] = True
                    result['face_confidence'] = float(box.conf[0])
                    result['face_box'] = box.xyxy[0].cpu().numpy().tolist()
                else:
                    logger.info(f"Face detector found no faces above threshold {self.face_conf_threshold}")
            except Exception as e:
                logger.error(f"Face detection error: {e}")

        return result


class RespirationEstimator:
    """Estimates respiratory rate from video frames using VIRENet."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.post_processor = PostProcessor()

        # Model parameters
        self.frame_depth = config.get('frame_depth', 10)
        self.img_size = config.get('img_size', 96)
        self.in_channels = config.get('in_channels', 3)
        self.chunk_length = config.get('chunk_length', 180)
        self.target_fps = config.get('target_fps', 30)

        # Initialize PreProcessor with config matching the training config
        preprocess_config = {
            'DO_DOWNSAMPLE_BEFORE_PREPROCESS': False,
            'DATA_NORMALIZE_TYPE': 'Standardized',
            'LABEL_NORMALIZE_TYPE': 'DiffNormalized',
            'FLOW_NORMALIZE_TYPE': 'Standardized',
            'DO_CHUNK': False,  # We handle chunking ourselves
            'CHUNK_LENGTH': self.chunk_length,
            'DO_CROP_INFANT_REGION': False,  # We handle ROI ourselves
            'DO_GRAYSCALE': False,
            'DO_AUGMENTATION': False,
            'DO_OPTICAL_FLOW': True,
            'OF_METHOD': 'coarse2fine',
            'PYFLOW_ALPHA': 0.012,
            'PYFLOW_RATIO': 0.75,
            'PYFLOW_MIN_WIDTH': 20,
            'PYFLOW_N_OUTER_FP_ITERATIONS': 7,
            'PYFLOW_N_INNER_FP_ITERATIONS': 1,
            'PYFLOW_N_SOR_ITERATIONS': 30,
            'DO_DOWNSAMPLE_BEFORE_TRAINING': True,
            'DOWNSAMPLE_SIZE_BEFORE_TRAINING': [self.img_size, self.img_size],
        }

        body_detector_path = config.get('body_detector_path', '/app/models_pretrained/detectors/yolov8m.pt')
        face_detector_path = config.get('face_detector_path', '/app/models_pretrained/detectors/yolov8n-face.pt')

        self.pre_processor = PreProcessor(preprocess_config, body_detector_path, face_detector_path)
        logger.info(f"Initialized PreProcessor with optical flow method: coarse2fine")

        # Load model
        checkpoint_path = config.get(
            'checkpoint_path',
            '/app/models_pretrained/checkpoint/VIRENet_best.pth'
        )
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> None:
        """Load the VIRENet model."""
        try:
            self.model = VIRENet(
                in_channels=self.in_channels,
                frame_depth=self.frame_depth,
                img_size=self.img_size
            )

            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded model checkpoint from {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path}, using random weights")

            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def preprocess_frames(self, frames: np.ndarray, roi_box: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess frames for inference using repo's PreProcessor with optical flow.

        Returns:
            Tuple of (processed_frames for model, raw_flow for magnitude calculation)
        """
        # Crop to ROI if provided
        if roi_box:
            x1, y1, x2, y2 = map(int, roi_box)
            frames = np.array([f[y1:y2, x1:x2] for f in frames])

        # Resize frames to target size first
        resized = np.array([
            cv2.resize(f, (self.img_size, self.img_size))
            for f in frames
        ], dtype=np.float32)

        # Use PreProcessor's optical flow computation (coarse2fine via pyflow)
        logger.info(f"Computing optical flow for {len(resized)} frames...")
        flow_raw = self.pre_processor.compute_pyflow(resized.astype(np.uint8))  # (T, H, W, 3)
        logger.info(f"Optical flow computed: shape={flow_raw.shape}")

        # Normalize flow (Standardized) for model input
        flow_norm = self.pre_processor._standardize(flow_raw)

        # Normalize frames (Standardized)
        frames_norm = self.pre_processor._standardize(resized)

        # Concatenate: flow (3ch) + frames (3ch) = 6 channels
        combined = np.concatenate([flow_norm, frames_norm], axis=-1)  # (T, H, W, 6)

        logger.info(f"Preprocessed frames: shape={combined.shape}")
        return combined, flow_raw  # Return processed frames AND raw flow

    def compute_movement(self, frames: np.ndarray, num_frames: int = 90) -> Optional[float]:
        """Compute movement (optical flow magnitude) without running respiration model.

        This is a lightweight version that only computes optical flow for movement detection.
        Uses fewer frames (default 90 = 3 seconds at 30fps) for faster computation.

        Args:
            frames: Array of frames (T, H, W, C)
            num_frames: Number of frames to use (default 90 = 3 seconds at 30fps)

        Returns:
            Average optical flow magnitude per pixel, or None on error
        """
        if len(frames) < num_frames:
            logger.debug(f"Not enough frames for movement: {len(frames)}/{num_frames}")
            return None

        try:
            # Use last num_frames
            frames_subset = frames[-num_frames:]

            # Resize frames to target size
            resized = np.array([
                cv2.resize(f, (self.img_size, self.img_size))
                for f in frames_subset
            ], dtype=np.float32)

            # Compute optical flow
            logger.debug(f"Computing movement optical flow for {len(resized)} frames...")
            flow_raw = self.pre_processor.compute_pyflow(resized.astype(np.uint8))  # (T, H, W, 3)

            # Compute magnitude from u, v components
            flow_u = flow_raw[:, :, :, 0]
            flow_v = flow_raw[:, :, :, 1]
            flow_magnitude = np.sqrt(flow_u**2 + flow_v**2)

            # Average magnitude per pixel
            total_optical_flow = float(np.mean(flow_magnitude))
            logger.debug(f"Movement optical flow: {total_optical_flow:.4f}")

            return total_optical_flow

        except Exception as e:
            logger.error(f"Movement computation error: {e}")
            return None

    def estimate(self, frames: np.ndarray, roi_box: Optional[List[float]] = None) -> Tuple[Optional[float], Optional[float]]:
        """Estimate respiratory rate from frames.

        Returns:
            Tuple of (respiratory_rate, total_optical_flow)
        """
        if self.model is None or len(frames) < self.chunk_length:
            return None, None

        try:
            # Compute optical flow on FULL frame (no ROI crop) for movement detection
            processed_full, flow_raw_full = self.preprocess_frames(frames[-self.chunk_length:], roi_box=None)

            # Compute total optical flow magnitude from RAW (unnormalized) flow
            # Flow is in first 2 channels (u, v components)
            flow_u = flow_raw_full[:, :, :, 0]  # (T, H, W)
            flow_v = flow_raw_full[:, :, :, 1]  # (T, H, W)
            flow_magnitude = np.sqrt(flow_u**2 + flow_v**2)  # (T, H, W)

            # Average magnitude per pixel, averaged across frames (normalized by resolution)
            total_optical_flow = float(np.mean(flow_magnitude))
            logger.info(f"Total optical flow (full cot, per-pixel avg): {total_optical_flow:.4f}")

            # Preprocess with body ROI for respiration estimation (if available)
            if roi_box is not None:
                processed, _ = self.preprocess_frames(frames[-self.chunk_length:], roi_box)
                logger.info(f"Using body ROI for respiration: {roi_box}")
            else:
                processed = processed_full
                logger.info("No body ROI available, using full frame for respiration")

            # Convert to tensor: (T, H, W, 6) -> (1, T, 6, H, W)
            tensor = torch.from_numpy(processed).permute(0, 3, 1, 2).unsqueeze(0)
            tensor = tensor.to(self.device)

            # VIRENet expects optical flow (first 3 channels) when trained with do_optical_flow=True
            # Extract only flow channels: (1, T, 6, H, W) -> (1, T, 3, H, W)
            tensor = tensor[:, :, :3, :, :]
            logger.info(f"Using optical flow input: shape={tensor.shape}")

            # Reshape for model: (N*D, C, H, W)
            N, D, C, H, W = tensor.shape
            tensor = tensor.reshape(N * D, C, H, W).contiguous()

            # Ensure divisible by frame_depth
            valid_len = (tensor.shape[0] // self.frame_depth) * self.frame_depth
            tensor = tensor[:valid_len].contiguous()

            # Inference
            with torch.no_grad():
                pred = self.model(tensor)

            # Debug: log model output stats
            logger.info(f"Model output: shape={pred.shape}, min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}, std={pred.std():.4f}")

            # Post-process to get respiratory rate
            pred_seq = pred.unsqueeze(0)
            pred_wave, pred_rr = self.post_processor.post_process(
                pred_seq,
                fs=self.target_fps,
                diff_flag=True,
                infant_flag=True,
                use_bandpass=True,
                eval_method='FFT'
            )

            # Save waveform plot
            self._save_waveform(pred_wave, pred_rr)

            return float(pred_rr), total_optical_flow

        except Exception as e:
            logger.error(f"Respiration estimation error: {e}")
            return None, None

    def _save_waveform(self, waveform: np.ndarray, resp_rate: float) -> None:
        """Save respiratory waveform plot to PNG and data to CSV."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            timestamp_file = datetime.now().strftime('%Y%m%d_%H%M%S')

            fig, ax = plt.subplots(figsize=(10, 4))
            time_axis = np.arange(len(waveform)) / self.target_fps
            ax.plot(time_axis, waveform, 'b-', linewidth=1)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Respiratory Waveform - Rate: {resp_rate:.1f} bpm\n{timestamp}')
            ax.grid(True, alpha=0.3)

            output_path = os.environ.get('OUTPUT_IMAGE_PATH', '/app/output/output.png')
            waveform_path = output_path.replace('output.png', 'waveform.png')
            plt.savefig(waveform_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved waveform to {waveform_path}")

            # Save waveform data to CSV in subfolder
            csv_dir = os.path.join(os.path.dirname(output_path), 'waveforms')
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f'waveform_{timestamp_file}.csv')
            waveform_flat = waveform.flatten()
            with open(csv_path, 'w') as f:
                f.write(f"# timestamp: {timestamp}\n")
                f.write(f"# resp_rate_bpm: {resp_rate}\n")
                f.write(f"# fps: {self.target_fps}\n")
                f.write("time_s,amplitude\n")
                for i, val in enumerate(waveform_flat):
                    f.write(f"{i/self.target_fps:.4f},{val:.6f}\n")
            logger.info(f"Saved waveform CSV to {csv_path}")

        except Exception as e:
            logger.error(f"Failed to save waveform: {e}")


class RTSPCapture:
    """Captures frames from RTSP stream using PyAV."""

    def __init__(self, url: str, target_fps: int = 30):
        self.url = url
        self.target_fps = target_fps
        self.container = None
        self.frame = None
        self.lock = Lock()
        self.running = False
        self.thread = None
        self.last_frame_time = 0
        self.frame_interval = 1.0 / target_fps

    def start(self) -> bool:
        """Start capturing from RTSP stream."""
        try:
            import av
            import ssl
            import os

            # Disable SSL verification globally
            ssl._create_default_https_context = ssl._create_unverified_context
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            os.environ['CURL_CA_BUNDLE'] = ''

            # PyAV options for RTSP streams with full TLS bypass
            options = {
                'rtsp_transport': 'tcp',
                'stimeout': '10000000',
                'fflags': 'nobuffer',
                'flags': 'low_delay',
                'tls_verify': '0',
                'verifyhost': '0',
                'verify': '0',
                'allowed_media_types': 'video',
            }

            logger.info(f"Connecting to RTSP stream: {self.url}")
            self.container = av.open(self.url, options=options, timeout=30.0)

            self.running = True
            self.thread = Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            logger.info(f"Started RTSP capture from {self.url}")
            return True
        except Exception as e:
            logger.error(f"RTSP capture error: {e}")
            # Fallback to OpenCV
            return self._start_opencv()

    def _start_opencv(self) -> bool:
        """Fallback to OpenCV capture."""
        try:
            logger.info("Trying OpenCV fallback...")
            import os
            # Add TLS bypass options for ffmpeg
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000|tls_verify;0|verifyhost;0'
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

            if not self.cap.isOpened():
                logger.error(f"Failed to open RTSP stream: {self.url}")
                return False

            self.running = True
            self.thread = Thread(target=self._capture_loop_opencv, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            logger.error(f"OpenCV fallback error: {e}")
            return False

    def _capture_loop(self) -> None:
        """Continuous frame capture loop using PyAV."""
        import av

        reconnect_delay = 1
        max_reconnect_delay = 30

        while self.running:
            try:
                for frame in self.container.decode(video=0):
                    if not self.running:
                        break

                    # Convert to numpy array (RGB)
                    img = frame.to_ndarray(format='rgb24')

                    with self.lock:
                        self.frame = img
                        self.last_frame_time = time.time()

                    # Reset reconnect delay on successful frame
                    reconnect_delay = 1

                    # No sleep - read frames as fast as possible to stay current
                    # The lock ensures we always have the latest frame available

            except Exception as e:
                logger.warning(f"Stream error, reconnecting in {reconnect_delay}s: {e}")
                time.sleep(reconnect_delay)

                # Exponential backoff
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

                try:
                    if self.container:
                        self.container.close()
                    self.container = av.open(self.url, options={
                        'rtsp_transport': 'tcp',
                        'stimeout': '10000000',
                        'fflags': 'nobuffer',
                        'flags': 'low_delay',
                        'tls_verify': '0',
                        'verifyhost': '0',
                    }, timeout=30.0)
                    logger.info("Reconnected to RTSP stream")
                except Exception as re:
                    logger.error(f"Reconnect failed: {re}")

    def _capture_loop_opencv(self) -> None:
        """Continuous frame capture loop using OpenCV."""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self.lock:
                        self.frame = frame
                        self.last_frame_time = time.time()
                else:
                    logger.warning("Lost RTSP connection, reconnecting...")
                    self.cap.release()
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(1)

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame. Returns None if frame is stale to trigger watchdog."""
        with self.lock:
            if self.frame is not None:
                frame_age = time.time() - self.last_frame_time
                if frame_age > 5.0:
                    # Frame too old - return None so watchdog can trigger
                    logger.warning(f"Frame is {frame_age:.1f}s old - returning None to trigger watchdog")
                    return None
                return self.frame.copy()
            return None

    def stop(self) -> None:
        """Stop capturing and reset state."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.container:
            self.container.close()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        # Reset state for clean restart
        with self.lock:
            self.frame = None
            self.last_frame_time = 0
        self.container = None
        self.thread = None


class HomeAssistantClient:
    """Client for posting sensor data to Home Assistant."""

    def __init__(self, config: Dict[str, Any]):
        self.base_url = os.environ.get('HA_URL', config.get('ha_url', 'http://host.docker.internal:8123'))
        self.token = os.environ.get('HA_TOKEN', config.get('ha_token', ''))
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

    def update_sensor(self, entity_id: str, state: Any, attributes: Dict[str, Any] = None) -> bool:
        """Update a sensor in Home Assistant."""
        url = f"{self.base_url}/api/states/{entity_id}"

        payload = {
            'state': state,
            'attributes': attributes or {}
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            if response.status_code in [200, 201]:
                logger.debug(f"Updated {entity_id} to {state}")
                return True
            else:
                logger.error(f"Failed to update {entity_id}: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"HA API error: {e}")
            return False


class BabyMonitorService:
    """Main service orchestrating all components."""

    def __init__(self, config_path: str = '/app/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.rtsp_url = os.environ.get('RTSP_URL', self.config.get('rtsp_url', ''))
        self.update_interval = int(os.environ.get('UPDATE_INTERVAL', self.config.get('update_interval', 10)))

        # Initialize components
        self.capture = RTSPCapture(self.rtsp_url, target_fps=self.config.get('target_fps', 30))
        self.perspective = PerspectiveCorrector(self.config.get('perspective', {}))
        self.detector = BabyDetector(self.config.get('detection', {}))
        self.estimator = RespirationEstimator(self.config.get('respiration', {}))
        self.ha_client = HomeAssistantClient(self.config.get('homeassistant', {}))

        # Set callback to clear frame buffer on recalibration
        self.perspective.on_recalibrate_callback = self._clear_frame_buffer

        # Frame buffer for respiration estimation
        self.frame_buffer = deque(maxlen=self.config.get('respiration', {}).get('chunk_length', 180))

        self.running = False

        # Entity IDs
        self.entity_prefix = self.config.get('entity_prefix', 'sensor.baby_monitor')

        # Track detection over last N cycles for robust analysis triggering
        self.detection_history = deque(maxlen=5)  # Last 5 detection results

        # Inference interval (how often to run respiration estimation)
        self.inference_interval = self.config.get('respiration', {}).get('inference_interval', 30)
        self.last_inference = 0

        # Movement detection interval (runs always, no detection required)
        self.movement_interval = self.config.get('respiration', {}).get('movement_interval', 3)
        self.movement_frames = self.config.get('respiration', {}).get('movement_frames', 90)
        self.last_movement_check = 0

        # Watchdog settings - restart stream if no frames for X seconds
        self.watchdog_timeout = self.config.get('watchdog_timeout', 120)  # 2 minutes default
        self.last_frame_time = time.time()
        self.watchdog_triggered_count = 0

    def _clear_frame_buffer(self) -> None:
        """Clear frame buffer on recalibration to avoid size mismatch."""
        self.frame_buffer.clear()
        logger.info("Frame buffer cleared due to recalibration")

    def start(self) -> None:
        """Start the service."""
        logger.info("Starting Baby Monitor Service...")

        if not self.capture.start():
            logger.error("Failed to start RTSP capture, exiting")
            return

        self.running = True

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self._main_loop()

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _main_loop(self) -> None:
        """Main processing loop."""
        last_update = 0
        loop_count = 0

        while self.running:
            try:
                loop_count += 1
                current_time = time.time()

                # Log every 600 iterations (~60s at 0.1s sleep) to show main loop is alive
                if loop_count % 600 == 0:
                    logger.info(f"Main loop heartbeat: iteration {loop_count}, time_since_frame={time_since_frame:.1f}s")

                # Watchdog check - force reconnect if no frames for too long
                time_since_frame = current_time - self.last_frame_time
                if time_since_frame > self.watchdog_timeout:
                    self.watchdog_triggered_count += 1
                    logger.warning(f"Watchdog triggered: no frames for {time_since_frame:.1f}s (count={self.watchdog_triggered_count}), forcing reconnect...")
                    self.capture.stop()
                    time.sleep(2)
                    if self.capture.start():
                        logger.info("Watchdog: stream reconnected successfully")
                        self.last_frame_time = time.time()
                        self._clear_frame_buffer()
                    else:
                        logger.error("Watchdog: failed to reconnect, retrying in 10s...")
                        time.sleep(10)
                    continue

                frame = self.capture.get_frame()

                if frame is None:
                    time.sleep(0.1)
                    continue

                # Update watchdog timer on successful frame
                self.last_frame_time = current_time

                # Apply perspective correction
                corrected = self.perspective.correct(frame)

                # Only add to frame buffer after perspective is calibrated (ensures consistent frame sizes)
                if self.perspective.calibrated:
                    self.frame_buffer.append(corrected)

                # Movement check - runs every 3 seconds, always (no detection required)
                # Only run after perspective calibration to ensure consistent frame sizes
                if self.perspective.calibrated and current_time - self.last_movement_check >= self.movement_interval:
                    buffer_size = len(self.frame_buffer)
                    if buffer_size >= self.movement_frames:
                        logger.info(f"Computing movement (interval={self.movement_interval}s)...")
                        frames_array = np.array(list(self.frame_buffer))
                        movement = self.estimator.compute_movement(frames_array, num_frames=self.movement_frames)
                        if movement is not None:
                            logger.info(f"Movement (optical flow): {movement:.6f}")
                            timestamp = datetime.now().isoformat()
                            self.ha_client.update_sensor(
                                f"{self.entity_prefix}_optical_flow",
                                round(movement, 6),
                                {
                                    'unit_of_measurement': 'px/frame',
                                    'friendly_name': 'Baby Cot Movement',
                                    'unique_id': 'baby_monitor_optical_flow',
                                    'icon': 'mdi:motion-sensor',
                                    'last_updated': timestamp
                                }
                            )
                    self.last_movement_check = current_time

                # Check if it's time to update
                if current_time - last_update >= self.update_interval:
                    # Run detection on both raw and corrected frames
                    # Use raw frame for better detection, corrected for display/respiration
                    self._process_and_update(frame, corrected)
                    last_update = current_time

                # Small sleep to prevent CPU overload
                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(1)

        self.capture.stop()
        logger.info("Baby Monitor Service stopped")

    def _process_and_update(self, raw_frame: np.ndarray, corrected_frame: np.ndarray) -> None:
        """Process current frame and update Home Assistant."""
        # Save transformed frame to output (only after perspective calibration)
        if self.perspective.calibrated:
            try:
                output_path = os.environ.get('OUTPUT_IMAGE_PATH', '/app/output/output.png')
                frame_bgr = cv2.cvtColor(corrected_frame, cv2.COLOR_RGB2BGR)

                # Add timestamp overlay
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(frame_bgr, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imwrite(output_path, frame_bgr)
                logger.debug(f"Saved transformed frame to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save output image: {e}")
        else:
            logger.info("Skipping output save - perspective not yet calibrated")

        # Detection on RAW frame (better detection before perspective warp)
        logger.info("Running detection on RAW frame:")
        raw_detection = self.detector.detect(raw_frame)

        # Also try on corrected frame
        logger.info("Running detection on CORRECTED frame:")
        corrected_detection = self.detector.detect(corrected_frame)

        # Combine results - use best detection from either frame
        detection_result = {
            'body_detected': raw_detection['body_detected'] or corrected_detection['body_detected'],
            'face_detected': raw_detection['face_detected'] or corrected_detection['face_detected'],
            'body_confidence': max(raw_detection['body_confidence'], corrected_detection['body_confidence']),
            'face_confidence': max(raw_detection['face_confidence'], corrected_detection['face_confidence']),
            'body_box': raw_detection['body_box'] or corrected_detection['body_box'],
            'face_box': raw_detection['face_box'] or corrected_detection['face_box'],
        }

        logger.info(f"Combined detection: body={detection_result['body_detected']}, face={detection_result['face_detected']}")

        # Track detection history
        baby_detected_now = detection_result['body_detected'] or detection_result['face_detected']
        self.detection_history.append(baby_detected_now)

        # Respiration estimation - trigger if any of last 5 cycles detected baby
        # and sufficient time has passed since last inference
        resp_rate = None
        total_optical_flow = None
        any_recent_detection = any(self.detection_history)
        buffer_size = len(self.frame_buffer)
        current_time = time.time()
        time_since_inference = current_time - self.last_inference
        logger.info(f"Frame buffer: {buffer_size}/{self.estimator.chunk_length}, recent_detections={sum(self.detection_history)}/5, time_since_inference={time_since_inference:.1f}s")

        if any_recent_detection and buffer_size >= self.estimator.chunk_length and time_since_inference >= self.inference_interval:
            logger.info(f"Calculating respiratory rate (interval={self.inference_interval}s)...")
            frames_array = np.array(list(self.frame_buffer))
            # Optical flow uses full cot, respiration uses body ROI if available
            resp_rate, total_optical_flow = self.estimator.estimate(frames_array, roi_box=detection_result['body_box'])
            self.last_inference = current_time
            logger.info(f"Respiratory rate result: {resp_rate}, Total optical flow: {total_optical_flow}")

        # Update Home Assistant sensors
        timestamp = datetime.now().isoformat()

        # Body detection sensor
        self.ha_client.update_sensor(
            f"{self.entity_prefix}_body_detected",
            'on' if detection_result['body_detected'] else 'off',
            {
                'device_class': 'occupancy',
                'friendly_name': 'Baby Body Detected',
                'unique_id': 'baby_monitor_body_detected',
                'confidence': detection_result['body_confidence'],
                'last_updated': timestamp
            }
        )

        # Face detection sensor
        self.ha_client.update_sensor(
            f"{self.entity_prefix}_face_detected",
            'on' if detection_result['face_detected'] else 'off',
            {
                'device_class': 'occupancy',
                'friendly_name': 'Baby Face Detected',
                'unique_id': 'baby_monitor_face_detected',
                'confidence': detection_result['face_confidence'],
                'last_updated': timestamp
            }
        )

        # Respiratory rate sensor
        if resp_rate is not None:
            self.ha_client.update_sensor(
                f"{self.entity_prefix}_respiratory_rate",
                round(resp_rate, 1),
                {
                    'unit_of_measurement': 'bpm',
                    'friendly_name': 'Baby Respiratory Rate',
                    'unique_id': 'baby_monitor_respiratory_rate',
                    'icon': 'mdi:lungs',
                    'last_updated': timestamp,
                    'body_detected': detection_result['body_detected'],
                    'face_detected': detection_result['face_detected']
                }
            )
        else:
            self.ha_client.update_sensor(
                f"{self.entity_prefix}_respiratory_rate",
                None,
                {
                    'unit_of_measurement': 'bpm',
                    'friendly_name': 'Baby Respiratory Rate',
                    'unique_id': 'baby_monitor_respiratory_rate',
                    'icon': 'mdi:lungs',
                    'last_updated': timestamp,
                    'reason': 'Insufficient data or baby not detected'
                }
            )

        # Note: optical flow (movement) is now updated separately every 3 seconds in the main loop

        logger.info(
            f"Updated sensors - Body: {detection_result['body_detected']}, "
            f"Face: {detection_result['face_detected']}, "
            f"Resp Rate: {resp_rate}, Optical Flow: {total_optical_flow}"
        )


def main():
    config_path = os.environ.get('CONFIG_PATH', '/app/config.yaml')
    service = BabyMonitorService(config_path)
    service.start()


if __name__ == '__main__':
    main()
