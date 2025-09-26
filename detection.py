# marker_detection.py
import time
import threading
from robomaster import robot, vision  # ต้องการ vision

import time
from robomaster import robot

def read_sensor_IRandTof(ep_sensor):
    latest_distance = [None]   # list so callback can modify

    def sub_data_handler(sub_info):
        distance = sub_info
        latest_distance[0] = float(distance[0])  # update value

    ep_sensor.sub_distance(freq=20, callback=sub_data_handler)

    # wait until callback updates
    for _ in range(10):  # check up to ~1 sec
        if latest_distance[0] is not None:
            return latest_distance[0]
        time.sleep(0.1)

    return None



def find_wall():
    #forward

    #right

    #left

    return 