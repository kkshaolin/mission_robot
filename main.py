from robomaster import robot
import msvcrt
import time
# from control import 
from detection import read_sensor_IRandTof
# from plotting import 

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_sensor = ep_robot.sensor
    
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':  # ESC เพื่อออก
                break
        
        read_sensor_IRandTof(ep_sensor)
        
        # ep_chassis.drive_speed(x=0.3, y=0, z=0, timeout=5)
    
    ep_robot.close()
    