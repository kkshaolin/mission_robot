# control.py
import time
import math
import numpy as np
import threading  # ใช้สำหรับ lock ใน MarkerDetector แต่ import ที่นี่ด้วยถ้าจำเป็น
from robomaster import robot
from detection import MarkerDetector

def pid_init(Kp, Ki, Kd, setpoint):
    return {
        'Kp': Kp,
        'Ki': Ki,
        'Kd': Kd,
        'setpoint': setpoint,
        'prev_error': 0.0,
        'integral': 0.0,
        'last_time': time.time()
    }

def pid_compute(pid_state, current_value):
    t = time.time()
    dt = t - pid_state['last_time']
    if dt <= 0: return 0.0
    error = pid_state['setpoint'] - current_value
    pid_state['integral'] += error * dt
    derivative = (error - pid_state['prev_error']) / dt
    out = (pid_state['Kp'] * error) + (pid_state['Ki'] * pid_state['integral']) + (pid_state['Kd'] * derivative)
    pid_state['prev_error'] = error
    pid_state['last_time'] = t
    return out
