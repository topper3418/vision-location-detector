# Plan: Refactor Detector to Delegate Pattern

## Goal
Refactor the detection system so that the detector is a delegate object (with a delegate base class) passed to the video feed object, rather than a function being passed. This will improve extensibility, testability, and code clarity.

---

## 1. Design DetectorDelegate Base Class
- Create a new abstract base class `DetectorDelegate` in a new file (e.g., `src/detector_delegate.py`).
- Define the interface for detection delegates (e.g., `detect(frame)`, `draw_detections(frame, detections)`, `initialize()`, `release()`).
- Document the expected contract for subclasses.

## 2. Refactor PedestrianDetector
- Make `PedestrianDetector` inherit from `DetectorDelegate`.
- Ensure it implements all required methods.
- Update any direct usage of `PedestrianDetector` to use the delegate interface.

## 3. Update VideoFeedBase and Subclasses
- Add support for a `delegate` attribute (of type `DetectorDelegate`) in `VideoFeedBase` and its subclasses (`CameraFeed`, `SpoofVideoFeed`).
- Remove or deprecate the `add_postprocessor` method for detection (but keep for other postprocessors if needed).
- Update the streaming and processing pipeline to use the delegate object for detection, not a function.
- Ensure the delegate is called in the correct place in the frame processing pipeline.

## 4. Refactor main.py and Scripts
- Update `src/main.py` to instantiate and pass a `DetectorDelegate` to the video feed, not a function.
- Update `scripts/detect_on_video.py` and any other scripts to use the new delegate pattern.
- Remove any function-based postprocessor code for detection.

## 5. Update and Refactor Tests
- Update unit and integration tests for `PedestrianDetector` to test via the `DetectorDelegate` interface.
- Update tests for `VideoFeedBase`, `CameraFeed`, and `SpoofVideoFeed` to use a delegate object.
- Update main application and integration tests to check correct delegate usage.
- Add/Update mocks for the delegate pattern where needed.

## 6. Documentation
- Document the new delegate pattern in the code and in the README if needed.
- Add migration notes for developers.

---

## Files to Change
- `src/detector.py` (refactor, inherit from delegate)
- `src/detector_delegate.py` (new)
- `src/video_feed_base.py` (delegate support)
- `src/camera.py`, `src/spoof_video_feed.py` (delegate support)
- `src/main.py`, `scripts/detect_on_video.py` (usage)
- All relevant tests in `test/unit/` and `test/integration/`

## Migration Steps
1. Implement and test `DetectorDelegate` base class. **[DONE]**
2. Refactor `PedestrianDetector` to inherit from it. **[DONE]**
3. Update video feed classes to accept and use a delegate. **[DONE]**
4. Refactor main and scripts to use the delegate. **[DONE]**
5. Update all tests. **[DONE]**
6. Document the changes. **[DONE]**

---

# Migration Notes

## Usage
- All detectors must now inherit from `DetectorDelegate` (see `video_feed_base.py`).
- All detection results must be returned as a list of `DetectionResult` objects.
- Video feed classes (`CameraFeed`, `SpoofVideoFeed`) accept a `delegate` argument for detection.
- The main application and scripts must pass a detector delegate to the video feed.
- Tests should use the delegate interface for all detection logic.

## Example

```python
from src.detector import PedestrianDetector
from src.camera import CameraFeed

detector = PedestrianDetector()
detector.initialize()
video_feed = CameraFeed(delegate=detector)
video_feed.initialize()
for detections, frame in video_feed.get_full_stream():
	print(detections)
```

## Summary
- All detection logic is now routed through the delegate interface for consistency and extensibility.
- All tests and scripts have been updated to use the new pattern.

----

## Notes
- Ensure backward compatibility where possible, but prefer the delegate pattern for all detection logic.
- All detection logic should now go through the delegate interface.
- All tests must pass after refactor.
