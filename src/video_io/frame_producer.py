from dataclasses import dataclass
from typing import Optional, Any
import threading
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
# THREAD-SAFE CIRCULAR BUFFER
# ==========================================================

class CircularFrameBuffer:
    """
    Real-time overwriting circular buffer.

    - Producer overwrites oldest frames when full
    - Consumer reads oldest unread frame
    - Thread-safe
    """

    def __init__(self, capacity: int):

        self.capacity = capacity

        self.buffer = [None] * capacity

        self.head = 0      # next write index
        self.tail = 0      # next read index

        self.size = 0

        self.lock = threading.Lock()

    # ======================================================
    # PRODUCER PUSH
    # ======================================================

    def push(self, frame: Frame):

        with self.lock:

            # Buffer full -> overwrite oldest frame
            if self.size == self.capacity:

                # Move tail forward (drop oldest)
                self.tail = (self.tail + 1) % self.capacity

            else:
                self.size += 1

            # Write new frame
            self.buffer[self.head] = frame

            # Move head forward
            self.head = (self.head + 1) % self.capacity

    # ======================================================
    # CONSUMER POP
    # ======================================================

    def pop(self) -> Optional[Frame]:

        with self.lock:

            if self.size == 0:
                return None

            frame = self.buffer[self.tail]

            # Optional cleanup
            self.buffer[self.tail] = None

            # Move tail forward
            self.tail = (self.tail + 1) % self.capacity

            self.size -= 1

            return frame

    @property
    def is_empty(self) -> bool:

        return self.size == 0

    @property
    def is_full(self) -> bool:

        return self.size == self.capacity

    @property
    def dropped_frames(self) -> int:

        return self._dropped_frames

    @property
    def occupancy_ratio(self) -> float:

        return self.size / self.capacity

    # ======================================================
    # ACCESSORS
    # ======================================================

    def latest(self) -> Optional[Frame]:
        """
        Return newest frame without removing it.
        """

        with self.lock:

            if self.size == 0:
                return None

            latest_idx = (
                self.head - 1
            ) % self.capacity

            return self.buffer[latest_idx]

    def peek(self) -> Optional[Frame]:
        """
        Return oldest unread frame
        without removing it.
        """

        with self.lock:

            if self.size == 0:
                return None

            return self.buffer[self.tail]

    # ======================================================
    # DEBUG / MONITORING
    # ======================================================

    def frame_indices(self) -> list[int]:
        """
        Return frame indices currently
        stored in buffer (ordered oldest->newest).
        """

        with self.lock:

            indices = []

            idx = self.tail

            for _ in range(self.size):

                frame = self.buffer[idx]

                if frame is not None:

                    indices.append(
                        frame.read_idx
                    )

                idx = (
                    idx + 1
                ) % self.capacity

            return indices


    def state(self):

        with self.lock:

            return {
                "head": self.head,
                "tail": self.tail,
                "size": self.size,
                "buffer": [
                    f.data if f is not None else None
                    for f in self.buffer
                ]
            }


    def clear(self):

        with self.lock:

            self.buffer = [None] * self.capacity

            self.head = 0

            self.tail = 0

            self.size = 0

    # ======================================================
    # PYTHON MAGIC
    # ======================================================

    def __len__(self):

        return self.size









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

    def __init__(self, source):

        self.source = source

        self.cap = None

        self.read_idx = 0


        self._fps = self.fps

        self._sleep = 1 / self._fps

        self._frame_width = self.frame_width

        self._frame_height = self.frame_height

    # ======================================================
    # OPEN SOURCE
    # ======================================================

    def open(self):

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Failed to open source: {self.source}"
            )

    # ======================================================
    # GRAB FRAME
    # ======================================================

    def grab(self) -> Optional[Frame]:

        # --------------------------------------------------
        # Regulate prerecorded video playback speed
        # --------------------------------------------------

        if isinstance(self.source, str):
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

    # ======================================================
    # SOURCE METADATA
    # ======================================================
    # ======================================================
    # INTERNAL TEMP OPEN
    # ======================================================

    def _open_temp_capture(self):

        temp_cap = cv2.VideoCapture(self.source)

        if not temp_cap.isOpened():

            raise RuntimeError(
                f"Failed to open source: {self.source}"
            )

        return temp_cap

    # ======================================================
    # SOURCE METADATA
    # ======================================================

    @property
    def fps(self):

        if self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_FPS)) 
            
        temp_cap = self._open_temp_capture()

        fps = temp_cap.get(
            cv2.CAP_PROP_FPS
        )

        temp_cap.release()

        return int(fps)

    @property
    def frame_width(self):

        if self.cap is not None:

            return int(
                self.cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

        temp_cap = self._open_temp_capture()

        width = int(
            temp_cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        temp_cap.release()

        return int(width)

    @property
    def frame_height(self):

        if self.cap is not None:

            return int(
                self.cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

        temp_cap = self._open_temp_capture()

        height = int(
            temp_cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        temp_cap.release()

        return int(height)










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
# FRAME PRODUCER
# ==========================================================

class FrameProducer(threading.Thread):
    """
    Producer thread responsible for:

    1. grabbing frames
    2. applying frame sampling
    3. pushing selected frames to buffer
    """

    def __init__(
        self,
        source,
        buffer: CircularFrameBuffer,
        fps: float,
        effective_fps: float,
        sampling_type: str = "deterministic",
        window_size: int = 30
    ):

        super().__init__(daemon=True)

        # --------------------------------------------------
        # Shared buffer
        # --------------------------------------------------

        self.buffer = buffer

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

        # --------------------------------------------------
        # Runtime state
        # --------------------------------------------------

        self.running = False

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
    # THREAD LOOP
    # ======================================================

    def run(self):

        self.running = True

        while self.running:

            # ------------------------------------------------
            # Grab next frame
            # ------------------------------------------------

            frame = self.grabber.grab()

            # End of stream/video
            if frame is None:
                break

            # ------------------------------------------------
            # Apply sampling policy
            # ------------------------------------------------

            if not self.sampler.should_keep(
                frame.read_idx
            ):
                continue

            # ------------------------------------------------
            # Push selected frame to shared buffer
            # ------------------------------------------------

            self.buffer.push(frame)

    # ======================================================
    # STARTUP
    # ======================================================

    def start_producing(self):
        self.grabber.open()
        self.start()

    # ======================================================
    # SHUTDOWN
    # ======================================================

    def stop_producing(self):

        self.running = False

        self.join()

        self.grabber.release()













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