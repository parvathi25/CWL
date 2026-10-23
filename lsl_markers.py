"""
lsl_markers.py
================
Pushes event marker codes onto an LSL (Lab Streaming Layer) marker stream,
so your eye-tracking recording software (or any other LSL-aware recorder)
can pick them up and sync against gaze data.

Kept in its own file (separate from mot_task.py and eeg_triggers.py) so that:
  - the LSL outlet setup/teardown is isolated and easy to test on its own
    (run this file directly, see __main__ block below)
  - the main task script only ever calls lsl.send(code) -- a single fast,
    non-blocking call -- instead of embedding LSL setup logic inline

Usage from mot_task.py:
    from lsl_markers import LSLMarkers
    lsl = LSLMarkers()
    lsl.send(3)     # pushes marker code 3 (e.g. tracking_onset) onto the stream
    lsl.close()

Requires: pip install pylsl
"""

from pylsl import StreamInfo, StreamOutlet

STREAM_NAME = 'MOT_Task_Markers'
STREAM_TYPE = 'Markers'


class LSLMarkers:
    def __init__(self, stream_name=STREAM_NAME):
        info = StreamInfo(
            name=stream_name,
            type=STREAM_TYPE,
            channel_count=1,
            nominal_srate=0,          # irregular rate -- markers are event-driven, not sampled
            channel_format='int32',
            source_id='mot_task_marker_stream'
        )
        self.outlet = StreamOutlet(info)

    def send(self, code):
        """Push a single integer marker code onto the LSL stream."""
        self.outlet.push_sample([int(code)])

    def close(self):
        pass  # StreamOutlet has no explicit close method; cleaned up on garbage collection


# if __name__ == '__main__':
#     # quick standalone test -- run: python lsl_markers.py
#     # streams marker codes 1-5, one per second; open LabRecorder or your eye
#     # tracker's LSL viewer to confirm the 'MOT_Task_Markers' stream appears
#     # and the codes arrive
#     import time
#     lsl = LSLMarkers()
#     print("Streaming test markers 1-5, one per second. "
#           "Check LabRecorder / your eye tracker's LSL stream viewer.")
#     for code in range(1, 6):
#         lsl.send(code)
#         print("sent marker:", code)
#         time.sleep(1)
#     print("Done.")
