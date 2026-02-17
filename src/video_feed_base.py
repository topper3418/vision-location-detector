"""Video feed base class with postprocessing pipeline support."""
from typing import Any, Callable, List, Optional, Tuple
import numpy as np

class VideoFeedBase:
    
    """Abstract video feed interface supporting postprocessing pipeline."""
    def __init__(self):
        self.postprocessors: List[Callable[[np.ndarray], Any]] = []

    def initialize(self) -> bool:
        raise NotImplementedError

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def is_opened(self) -> bool:
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    def add_postprocessor(self, func: Callable[[np.ndarray], Any]):
        self.postprocessors.append(func)

    def get_raw_stream(self):
        """Yields raw frames from the video feed."""
        while True:
            success, frame = self.read_frame()
            if not success or frame is None:
                break
            yield frame

    def get_full_stream(self):
        """Yields (data, processed) tuples from the postprocessing pipeline."""
        for frame in self.get_raw_stream():
            processed = frame
            data = frame
            for proc in self.postprocessors:
                result = proc(processed)
                if isinstance(result, tuple):
                    processed, data = result
                else:
                    processed = result
            yield (data, processed)

    def get_processed_stream(self):
        """Yields processed frames (e.g., annotated) from the video feed."""
        for frame in self.get_raw_stream():
            processed = frame
            for proc in self.postprocessors:
                result = proc(processed)
                if isinstance(result, tuple):
                    processed = result[0]
                else:
                    processed = result
            yield processed

    def get_data_stream(self):
        """Yields only the data output from the postprocessing pipeline (headless mode)."""
        for frame in self.get_raw_stream():
            data = frame
            for proc in self.postprocessors:
                result = proc(data)
                if isinstance(result, tuple):
                    data = result[1]
                else:
                    data = result
            yield data
