from dataclasses import dataclass
from typing import Optional, Any
import time
import random

import cv2







# ==========================================================
# FRAME CONTAINER
# ==========================================================

@dataclass
class Frame:
    """Container for a captured frame and its metadata."""
    
    data: Any
    timestamp: float
    read_idx: int
    processed_idx: Optional[int] = None


# ==========================================================
# ==========================================================
# FRAME GRABBER
# ==========================================================

class FrameGrabber:
    """
    Efficient video source wrapper.

    Supports:
    - webcam
    - RTSP streams
    - prerecorded videos
    """

    def __init__(self, source, fallback_fps: float = 30.0):

        self.source = source
        self.capture_source = self._normalize_source(source)

        self.cap = None

        self.read_idx = 0

        self._fallback_fps = fallback_fps

        self._fps = fallback_fps

        self._sleep = 1 / self._fps if self._fps > 0 else 0

        self._frame_width = None

        self._frame_height = None

    # ======================================================
    # OPEN SOURCE
    # ======================================================

    def open(self):

        self.cap = cv2.VideoCapture(self.capture_source)

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Failed to open source: {self.source}"
            )

        self._refresh_metadata()

    def _refresh_metadata(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps) if fps and fps > 0 else float(self._fallback_fps)
        self._sleep = 1 / self._fps if self._fps > 0 else 0
        self._frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ======================================================
    # GRAB FRAME
    # ======================================================

    def grab(self) -> Optional[Frame]:

        # --------------------------------------------------
        # Regulate prerecorded video playback speed
        # --------------------------------------------------

        if self._should_regulate_source():
            time.sleep(self._sleep)

        # --------------------------------------------------
        # Read frame
        # --------------------------------------------------

        success, frame = self.cap.read()

        if not success:
            return None

        # --------------------------------------------------
        # Build frame object
        # --------------------------------------------------

        frame_obj = Frame(
            data=frame,
            timestamp=time.time(),
            read_idx=self.read_idx
        )

        self.read_idx += 1

        return frame_obj

    # ======================================================
    # RELEASE SOURCE
    # ======================================================

    def release(self):

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ======================================================
    # SOURCE METADATA
    # ======================================================
    def _should_regulate_source(self) -> bool:
        if not isinstance(self.source, str):
            return False

        source = self.source.strip().lower()
        if source.isdigit() or source.startswith("/dev/video"):
            return False

        return True

    @staticmethod
    def _normalize_source(source):
        if isinstance(source, str) and source.strip().isdigit():
            return int(source.strip())

        return source

    # ======================================================
    # SOURCE METADATA
    # ======================================================

    @property
    def fps(self):

        if self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                self._fps = float(fps)
                self._sleep = 1 / self._fps

        return self._fps

    @property
    def frame_width(self):

        if self.cap is not None:
            self._frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        return self._frame_width

    @property
    def frame_height(self):

        if self.cap is not None:
            self._frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return self._frame_height










class FrameSampler:
    """
    Lightweight high-performance frame sampler.

    Supported methods:
    - periodic_sampling
    - burst_sampling
    - random_sampling
    """

    def __init__(
        self,
        sampling_method: str = "periodic_sampling",
        sampling_factor: int | float = 1,
        window_size: int = 30
    ):

        self.sampling_method = sampling_method

        self.sampling_factor = sampling_factor

        self.window_size = window_size

        # --------------------------------------------------
        # Random sampling state
        # --------------------------------------------------

        self._keep_indices = set()

        # --------------------------------------------------
        # Resolve strategy ONCE
        # --------------------------------------------------

        if self.sampling_method == "periodic_sampling":

            self._should_keep_fn = (
                self._periodic_sampling
            )

        elif self.sampling_method == "burst_sampling":

            self._should_keep_fn = (
                self._burst_sampling
            )

        elif self.sampling_method == "random_sampling":

            self._generate_random_keep_indices()

            self._should_keep_fn = (
                self._random_sampling
            )

        else:

            raise ValueError(
                f"Unsupported sampling method: "
                f"{self.sampling_method}"
            )

    # ======================================================
    # PUBLIC API
    # ======================================================

    def should_keep(self, frame_idx: int) -> bool:

        return self._should_keep_fn(frame_idx)

    # ======================================================
    # SAMPLING METHODS
    # ======================================================

    def _periodic_sampling(
        self,
        frame_idx: int
    ) -> bool:
        """
        Keep 1 frame every N frames.
        """

        return (
            frame_idx % self.sampling_factor
        ) == 0

    def _burst_sampling(
        self,
        frame_idx: int
    ) -> bool:
        """
        Keep burst frames between
        periodic anchor skips.
        """

        return (
            frame_idx % self.sampling_factor
        ) != 0

    def _random_sampling(
        self,
        frame_idx: int
    ) -> bool:
        """
        Random sampling inside windows.
        """

        idx_in_window = (
            frame_idx % self.window_size
        )

        # regenerate once per window
        if idx_in_window == 0:

            self._generate_random_keep_indices()

        return (
            idx_in_window
            in self._keep_indices
        )

    # ======================================================
    # INTERNAL
    # ======================================================

    def _generate_random_keep_indices(self):

        keep_n = int(
            round(
                self.sampling_factor
                * self.window_size
            )
        )

        keep_n = max(
            1,
            min(
                self.window_size,
                keep_n
            )
        )

        self._keep_indices = set(
            random.sample(
                range(self.window_size),
                keep_n
            )
        )
        

    






# ==========================================================
# REAL-TIME SIMULATION PRODUCER
# ==========================================================

class RealTimeSimulationProducer:
    """
    Simulates real-time AI video processing
    on prerecorded videos.

    Behavior:
    - sequential decoding
    - no full-video RAM loading
    - no buffering
    - no sampling/striding
    - frame dropping emerges naturally
      from processing latency

    Designed for scientifically-correct
    real-time benchmarking.
    """

    def __init__(self, source):

        self.source = source

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Failed to open source: "
                f"{source}"
            )

        # --------------------------------------------------
        # Source metadata
        # --------------------------------------------------

        self.fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        self.frame_interval = (
            1.0 / self.fps
        )

        self.total_frames = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        self.video_duration_seconds = (
            self.total_frames / self.fps
        )

        # --------------------------------------------------
        # Runtime state
        # --------------------------------------------------

        self.frame_idx = 0

        self.read_latency = 0

        self.processed_frames = []

    # ======================================================
    # START
    # ======================================================

    def start(self):

        return None
        
    # ======================================================
    # NEXT FRAME
    # ======================================================
    
    def next_frame(
        self,
        processing_latency: float = 0.0
    ) -> Optional[Frame]:
    
        # --------------------------------------------------
        # Compute simulated live frame
        # --------------------------------------------------

        

        if self.frame_idx >= self.total_frames:
            return None
    
        elapsed_time = processing_latency + self.read_latency
        
        if elapsed_time <= self.frame_interval:
            # time.sleep(self.frame_interval - elapsed_time)

            t0 = time.perf_counter()
            success, frame = self.cap.read()
        
            if not success:
        
                return None
            self.read_latency = time.perf_counter() - t0

            self.frame_idx += 1
            
        else:
            
            n_frames_to_skip = int(elapsed_time/self.frame_interval) - 1

            for _ in range(n_frames_to_skip):
        
                success = self.cap.grab()
        
                if not success:
        
                    return None 


            # --------------------------------------------------
            # Retrieve target frame
            # --------------------------------------------------
        
            t0 = time.perf_counter()
        
            success, frame = self.cap.read()
        
            if not success:
        
                return None
        
            self.read_latency = time.perf_counter() - t0
            self.frame_idx += n_frames_to_skip + 1
            

        self.processed_frames.append(self.frame_idx)
        
        # --------------------------------------------------
        # Build frame object
        # --------------------------------------------------
    
        return Frame(
    
            data=frame,
    
            timestamp=time.time(),
    
            read_idx=self.frame_idx
        )
        
    # ======================================================
    # RELEASE
    # ======================================================
    
    def release(self):
    
        self.cap.release()









# ==========================================================
# OFFLINE SAMPLED FRAME PRODUCER
# ==========================================================

class OfflineSampledFrameProducer:
    """
    Sequential offline frame producer.

    Behavior:
    - reads prerecorded video sequentially
    - applies frame sampling
    - returns sampled frames only
    - NO threading
    - NO buffering
    - NO frame dropping due to latency

    Designed for:
    - controlled offline evaluations
    - deterministic benchmark experiments
    """

    def __init__(
        self,
        source,
        fps: float,
        effective_fps: float,
        sampling_type: str = "deterministic",
        window_size: int = 30
    ):

        # --------------------------------------------------
        # FPS configuration
        # --------------------------------------------------

        self.fps = fps

        self.effective_fps = effective_fps

        self.r = (
            self.effective_fps / self.fps
        )

        # --------------------------------------------------
        # Sampling configuration
        # --------------------------------------------------

        self.sampling_type = sampling_type

        self.window_size = window_size

        # --------------------------------------------------
        # Build frame grabber
        # --------------------------------------------------

        self.grabber = FrameGrabber(
            source=source
        )

        # --------------------------------------------------
        # Resolve sampling method
        # --------------------------------------------------

        self.sampling_method = (
            self._resolve_sampling_method()
        )

        # --------------------------------------------------
        # Resolve sampling factor
        # --------------------------------------------------

        self.sampling_factor = (
            self._resolve_sampling_factor()
        )

        # --------------------------------------------------
        # Build sampler
        # --------------------------------------------------

        self.sampler = FrameSampler(

            sampling_method=
            self.sampling_method,

            sampling_factor=
            self.sampling_factor,

            window_size=
            self.window_size
        )

    # ======================================================
    # INTERNAL HELPERS
    # ======================================================

    def _resolve_sampling_method(self):

        # ----------------------------------------------
        # Deterministic sampling
        # ----------------------------------------------

        if self.sampling_type == "deterministic":

            if 0 < self.r <= 0.5:

                return "periodic_sampling"

            elif 0.5 < self.r < 1:

                return "burst_sampling"

            else: 
                
                return "periodic_sampling"
        # ----------------------------------------------
        # Stochastic sampling
        # ----------------------------------------------

        elif self.sampling_type == "stochastic":

            return "random_sampling"

        # ----------------------------------------------
        # Invalid configuration
        # ----------------------------------------------

        raise ValueError(
            "Invalid sampling configuration."
        )

    def _resolve_sampling_factor(self):

        # ----------------------------------------------
        # Periodic sampling
        # ----------------------------------------------

        if (
            self.sampling_method
            == "periodic_sampling"
        ):

            return max(
                1,
                round(1 / self.r)
            )

        # ----------------------------------------------
        # Burst sampling
        # ----------------------------------------------

        elif (
            self.sampling_method
            == "burst_sampling"
        ):

            return max(
                1,
                round(1 / (1 - self.r))
            )

        # ----------------------------------------------
        # Random sampling
        # ----------------------------------------------

        elif (
            self.sampling_method
            == "random_sampling"
        ):

            return self.r

        raise ValueError(
            "Failed to resolve "
            "sampling factor."
        )

    # ======================================================
    # START
    # ======================================================

    def start(self):

        self.grabber.open()

    # ======================================================
    # GET NEXT SAMPLED FRAME
    # ======================================================

    def next_frame(self) -> Optional[Frame]:

        while True:

            # ------------------------------------------------
            # Grab sequential frame
            # ------------------------------------------------

            frame = self.grabber.grab()

            # End of video
            if frame is None:
                return None

            # ------------------------------------------------
            # Apply sampling policy
            # ------------------------------------------------

            if not self.sampler.should_keep(
                frame.read_idx
            ):
                continue

            return frame

    # ======================================================
    # RELEASE
    # ======================================================

    def release(self):

        self.grabber.release()






class DirectFrameProducer:
    """
    Minimal frame producer.

    Behavior:
    - no threading
    - no buffering
    - no sampling
    - no frame dropping logic
    - one OpenCV read per request

    Designed for:
    - offline sequential processing
    - simple frame acquisition pipelines
    """

    def __init__(self, source):

        self.grabber = FrameGrabber(source)

    def start(self):

        self.grabber.open()

    def next_frame(self) -> Optional[Frame]:

        return self.grabber.grab()

    def release(self):

        self.grabber.release()
