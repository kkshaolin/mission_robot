from robomaster import robot
import msvcrt
import time
import threading

stop_flag = False
tof = 999
last_value_right = 0
last_value_left = 0
ir_right_cm = 999
ir_left_cm = 999

# ตารางแปลง ADC เป็น cm
calibra_table_ir_right = {
    615: 5,
    605: 10,
    415: 15,
    335: 20,
    275: 25,
    255: 30
}
calibra_table_ir_left = {
    680: 5,
    420: 10,
    300: 15,
    235: 20,
    210: 25,
    175: 30
}

# ฟังก์ชันกรองสัญญาณแบบ low-pass
def single_lowpass_filter(new_value, last_value, alpha=0.8):
    return alpha * new_value + (1.0 - alpha) * last_value

# ฟังก์ชันแปลง ADC เป็น cm
def adc_to_cm(adc_value, table):
    points = sorted(table.items(), key=lambda x: x[0], reverse=True)
    if adc_value >= points[0][0]:
        return float(points[0][1])
    if adc_value <= points[-1][0]:
        return float(points[-1][1])
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x2 <= adc_value <= x1:
            return float(y1 + (adc_value - x1) * (y2 - y1) / (x2 - x1))
    return float("nan")

# ฟังก์ชันรับค่าจาก TOF
def sub_data_handler(sub_info):
    global tof
    tof = sub_info[0] / 10

# เทรดสำหรับการเคลื่อนที่
def move_chassis(ep_chassis):
    global tof, ir_right_cm, ir_left_cm
    last_avoid_time = 0  # กันสั่งหลบถี่เกิน
    while not stop_flag:
        if tof <= 30:
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
            break

        now = time.time()
        if ir_right_cm < 10 and now - last_avoid_time > 1:
            ep_chassis.drive_speed(x=0, y=-0.05, z=0, timeout=1)  # วิ่งออกซ้าย
            time.sleep(0.5)
            last_avoid_time = now
        elif ir_left_cm < 10 and now - last_avoid_time > 1:
            ep_chassis.drive_speed(x=0, y=0.05, z=0, timeout=1)  # วิ่งออกขวา
            time.sleep(0.5)
            last_avoid_time = now
        else:
            # เคลื่อนไปข้างหน้า
            ep_chassis.drive_speed(x=0.4, y=0, z=0, timeout=1)

        time.sleep(0.1)

# เทรดสำหรับการอ่านค่าเซ็นเซอร์ IR
def read_ir(ep_sensor_adaptor):
    global ir_right_cm, ir_left_cm, last_value_right, last_value_left
    while not stop_flag:
        ir_right = ep_sensor_adaptor.get_adc(id=2, port=2)
        ir_left = ep_sensor_adaptor.get_adc(id=1, port=2)

        ir_right_filter = single_lowpass_filter(ir_right, last_value_right)
        ir_left_filter = single_lowpass_filter(ir_left, last_value_left)

        last_value_right = ir_right_filter
        last_value_left = ir_left_filter

        ir_right_cm = adc_to_cm(ir_right_filter, calibra_table_ir_right)
        ir_left_cm = adc_to_cm(ir_left_filter, calibra_table_ir_left)

        print(f'TOF: {tof:.1f} cm | IR Right: {ir_right_cm:.1f} cm | IR Left: {ir_left_cm:.1f} cm')

        time.sleep(0.1)

# ส่วนหลัก
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_sensor = ep_robot.sensor
    ep_sensor_adaptor = ep_robot.sensor_adaptor

    ep_sensor.sub_distance(freq=20, callback=sub_data_handler)
    time.sleep(0.2)

    move_thread = threading.Thread(target=move_chassis, args=(ep_chassis,))
    ir_thread = threading.Thread(target=read_ir, args=(ep_sensor_adaptor,))
    move_thread.start()
    ir_thread.start()

    while not stop_flag:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':
                stop_flag = True
                break
        ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=50, yaw_speed=100).wait_for_completed()
        time.sleep(0.5)

    move_thread.join()
    ir_thread.join()
    ep_sensor.unsub_distance()
    ep_robot.close()
