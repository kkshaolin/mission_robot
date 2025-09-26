# marker_detection.py
import time
import threading
from robomaster import robot, vision  # ต้องการ vision

class MarkerDetector:
    def __init__(self, ep_robot):
        self.ep_camera = ep_robot.camera
        self.ep_vision = ep_robot.vision
        self._markers = []
        self._markers_lock = threading.Lock()

        def _on_markers(marker_info):
            now = time.time()
            with self._markers_lock:
                self._markers = [
                    {"x": x, "y": y, "w": w, "h": h, "info": info, "ts": now}
                    for (x, y, w, h, info) in marker_info
                ]

        # ต้องเปิด video stream ก่อน ถึงจะ detect ได้
        self.ep_camera.start_video_stream(display=False)
        self.ep_vision.sub_detect_info(name="marker", callback=_on_markers)
        time.sleep(1.0)

    def get_markers(self, max_age=0.6):
        """คืน list markers (normalized 0..1) ภายในช่วง max_age วินาที"""
        now = time.time()
        with self._markers_lock:
            return [m for m in self._markers if now - m["ts"] <= max_age]

    def close(self):
        try:
            self.ep_vision.unsub_detect_info(name="marker")
        except: pass
        try:
            self.ep_camera.stop_video_stream()
        except: pass