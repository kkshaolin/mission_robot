from robomaster import robot
import msvcrt
import time
import math
import threading # เพิ่ม threading

# --- ค่าคงที่และตัวแปร Global ---
stop_flag = False

# ตัวแปรสำหรับเซ็นเซอร์
tof_distance_cm = 999.0
current_yaw = 0.0

# --- ส่วนที่เพิ่มเข้ามา: ตัวแปรสำหรับ IR Sensors จาก Sensor Adaptor ---
ir_left_cm = 999.0
ir_right_cm = 999.0
last_value_left = 0
last_value_right = 0

# --- ส่วนที่เพิ่มเข้ามา: ตารางเทียบค่า ADC เป็น CM สำหรับ IR Sensors ---
# *** คุณควรทำการ Calibrate ค่าเหล่านี้ใหม่เพื่อให้แม่นยำกับหุ่นยนต์ของคุณ ***
calibra_table_ir_right = {
    615: 5,  415: 15,  275: 25,
    605: 10, 335: 20,  255: 30
}
calibra_table_ir_left = {
    680: 5,  300: 15,  210: 25,
    420: 10, 235: 20,  175: 30
}

# ค่าคงที่สำหรับระบบ DFS และการเคลื่อนที่
NODE_DISTANCE_M = 0.6      # ระยะห่างระหว่างโหนด 60 cm
TOF_WALL_THRESHOLD_CM = 30.0 # ระยะ ToF ที่ถือว่าเป็นทางเปิด (ด้านหน้า)
IR_WALL_THRESHOLD_CM = 25.0  # ระยะ IR ที่ถือว่าเป็นทางเปิด (ด้านข้าง)
WALL_AVOID_THRESHOLD_CM = 10.0 # ระยะ IR ที่ต้องเริ่มขยับหนี
WALL_AVOID_SPEED_Y = 0.05    # ความเร็วในการขยับหนี (m/s)
SCAN_DURATION_S = 0.2
MOVE_SPEED_X = 0.2
TURN_SPEED_Z = 60

# ค่าคงที่สำหรับ PID Controller
Kp_turn = 2.5
Kp_straight = 0.8
Ki_straight = 0.02
Kd_straight = 0.1
integral_straight = 0.0
last_error_straight = 0.0

# โครงสร้างข้อมูลสำหรับ DFS
path_stack = []
visited_nodes = set()
current_pos = (0, 0)
current_heading_degrees = 0

# --- ฟังก์ชัน Callback สำหรับ ToF และ IMU (ยังใช้เหมือนเดิม) ---
def sub_tof_handler(sub_info):
    global tof_distance_cm
    tof_distance_cm = sub_info[0] / 10.0

def sub_imu_handler(attitude_info):
    global current_yaw
    current_yaw = attitude_info[0]

# --- ส่วนที่เพิ่มเข้ามา: ฟังก์ชันสำหรับ IR Sensors ---
def single_lowpass_filter(new_value, last_value, alpha=0.8):
    return alpha * new_value + (1.0 - alpha) * last_value

def adc_to_cm(adc_value, table):
    points = sorted(table.items(), key=lambda x: x[0], reverse=True)
    if adc_value >= points[0][0]: return float(points[0][1])
    if adc_value <= points[-1][0]: return float(points[-1][1])
    for i in range(len(points) - 1):
        x1, y1 = points[i]; x2, y2 = points[i+1]
        if x2 <= adc_value <= x1:
            return float(y1 + (adc_value - x1) * (y2 - y1) / (x2 - x1))
    return 999.0 # คืนค่าระยะไกลถ้าไม่อยู่ในตาราง

# --- ส่วนที่เพิ่มเข้ามา: Thread สำหรับอ่านค่า IR ตลอดเวลา ---
def read_ir_thread(ep_sensor_adaptor):
    """
    Thread ที่ทำงานเบื้องหลังเพื่ออ่านค่า ADC จาก IR sensor,
    กรองสัญญาณ, แปลงเป็น cm, และอัปเดตค่า global ตลอดเวลา
    """
    global ir_right_cm, ir_left_cm, last_value_right, last_value_left
    print("IR reading thread started.")
    while not stop_flag:
        # อ่านค่า ADC ดิบ (โปรดตรวจสอบ id และ port ให้ตรงกับที่ต่อไว้)
        ir_right_adc = ep_sensor_adaptor.get_adc(id=2, port=2)
        ir_left_adc = ep_sensor_adaptor.get_adc(id=1, port=2)

        # กรองสัญญาณ (Low-pass filter)
        ir_right_filtered = single_lowpass_filter(ir_right_adc, last_value_right)
        ir_left_filtered = single_lowpass_filter(ir_left_adc, last_value_left)
        last_value_right = ir_right_filtered
        last_value_left = ir_left_filtered

        # แปลงเป็น cm และอัปเดตตัวแปร global
        ir_right_cm = adc_to_cm(ir_right_filtered, calibra_table_ir_right)
        ir_left_cm = adc_to_cm(ir_left_filtered, calibra_table_ir_left)

        time.sleep(0.05) # อ่านค่าทุก 50ms
    print("IR reading thread stopped.")

# --- ฟังก์ชันช่วย (Helper Functions) ---
def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle <= -180: angle += 360
    return angle

# --- ฟังก์ชันควบคุมการเคลื่อนที่ (ไม่มีการเปลี่ยนแปลง) ---
def turn_to_angle(ep_chassis, ep_gimbal, target_angle):
    global current_yaw
    ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=100, yaw_speed=100)
    target_angle = normalize_angle(target_angle)
    print(f"กำลังหมุนไปที่ {target_angle:.1f}°...")
    while not stop_flag:
        angle_error = normalize_angle(target_angle - current_yaw)
        if abs(angle_error) < 2.0: break
        turn_speed = max(min(angle_error * Kp_turn, TURN_SPEED_Z), -TURN_SPEED_Z)
        ep_chassis.drive_speed(x=0, y=0, z=turn_speed)
        time.sleep(0.02)
    ep_chassis.drive_speed(x=0, y=0, z=0)
    time.sleep(0.5)
    print(f"หมุนสำเร็จ! มุมปัจจุบัน: {current_yaw:.1f}°")

# --- อัปเดต: ฟังก์ชันเดินตรง ให้ใช้ตัวแปร ir_left_cm, ir_right_cm ---
def move_straight_60cm(ep_chassis, target_yaw):
    global integral_straight, last_error_straight, ir_left_cm, ir_right_cm
    print(f"กำลังเคลื่อนที่ไปข้างหน้า 60 cm (PID + Wall Avoidance) ที่มุม {target_yaw:.1f}°")
    integral_straight, last_error_straight = 0.0, 0.0
    start_time = time.time()
    last_time = start_time
    duration = (NODE_DISTANCE_M / MOVE_SPEED_X) * 1.05

    while (time.time() - start_time) < duration and not stop_flag:
        current_time = time.time(); dt = current_time - last_time
        if dt <= 0: time.sleep(0.01); continue
        
        # PID สำหรับรักษาทิศทาง (แกน Z)
        error = normalize_angle(target_yaw - current_yaw)
        integral_straight += error * dt
        derivative = (error - last_error_straight) / dt
        z_correct_speed = (Kp_straight * error) + (Ki_straight * integral_straight) + (Kd_straight * derivative)
        
        # ควบคุมการหลบกำแพง (แกน Y) - **ใช้ตัวแปรใหม่**
        y_correct_speed = 0.0
        if ir_right_cm < WALL_AVOID_THRESHOLD_CM: y_correct_speed -= WALL_AVOID_SPEED_Y
        if ir_left_cm < WALL_AVOID_THRESHOLD_CM: y_correct_speed += WALL_AVOID_SPEED_Y
        
        ep_chassis.drive_speed(x=MOVE_SPEED_X, y=y_correct_speed, z=z_correct_speed)
        
        last_error_straight, last_time = error, current_time
        time.sleep(0.02)
    ep_chassis.drive_speed(x=0, y=0, z=0); time.sleep(0.5)
    print("เคลื่อนที่ 60 cm สำเร็จ")

# --- อัปเดต: ฟังก์ชันสแกน ให้ใช้ตัวแปร ir_left_cm, ir_right_cm ---
def scan_environment():
    global tof_distance_cm, ir_left_cm, ir_right_cm
    print("--- เริ่มการสแกนสภาพแวดล้อม (แบบไม่หมุน) ---")
    open_paths = {'front': False, 'left': False, 'right': False}
    time.sleep(SCAN_DURATION_S) # รอค่าเซ็นเซอร์นิ่ง

    # ตรวจสอบเส้นทางโดยใช้ตัวแปร global ที่อัปเดตจาก thread
    if tof_distance_cm > TOF_WALL_THRESHOLD_CM: open_paths['front'] = True
    if ir_left_cm > IR_WALL_THRESHOLD_CM: open_paths['left'] = True
    if ir_right_cm > IR_WALL_THRESHOLD_CM: open_paths['right'] = True

    print(f"ผลสแกน [หน้า-ToF]: {tof_distance_cm:.1f} cm | [ซ้าย-IR]: {ir_left_cm:.1f} cm | [ขวา-IR]: {ir_right_cm:.1f} cm")
    print(f"--- สแกนเสร็จสิ้น: {open_paths} ---")
    return open_paths

def get_new_pos_and_heading(direction, old_pos, old_heading):
    x, y = old_pos; new_heading = old_heading
    if direction == 'left': new_heading = normalize_angle(old_heading - 90)
    elif direction == 'right': new_heading = normalize_angle(old_heading + 90)
    
    if new_heading == 0: y += 1
    elif new_heading == 90: x += 1
    elif new_heading == -90: x -= 1
    elif abs(new_heading) == 180: y -= 1
    return (x, y), new_heading

# --- ส่วนหลักของโปรแกรม ---
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor
    ep_gimbal = ep_robot.gimbal
    ep_sensor_adaptor = ep_robot.sensor_adaptor # เพิ่ม: สร้าง object ของ sensor_adaptor

    # เริ่ม subscription ของ ToF และ IMU
    ep_sensor.sub_distance(freq=20, callback=sub_tof_handler)
    ep_chassis.sub_attitude(freq=20, callback=sub_imu_handler)
    time.sleep(1)

    # --- ส่วนที่เพิ่มเข้ามา: เริ่ม thread การอ่านค่า IR ---
    # ตั้งเป็น daemon=True เพื่อให้ thread ปิดตัวเองเมื่อโปรแกรมหลักจบ
    ir_reader = threading.Thread(target=read_ir_thread, args=(ep_sensor_adaptor,), daemon=True)
    ir_reader.start()
    time.sleep(1) # รอให้ thread เริ่มอ่านค่าสักครู่

    robot_state = "SCANNING"
    visited_nodes.add(current_pos)
    print(f"เริ่มต้น DFS ที่ตำแหน่ง {current_pos}, ทิศทาง {current_heading_degrees}°")

    try:
        # State machine loop หลักยังคงเหมือนเดิม
        while not stop_flag:
            if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
                stop_flag = True; break

            if robot_state == "SCANNING":
                scan_results = scan_environment()
                robot_state = "DECIDING"
            
            elif robot_state == "DECIDING":
                possible_moves = []
                check_order = ['front', 'left', 'right']
                for direction in check_order:
                    if scan_results[direction]:
                        potential_pos, _ = get_new_pos_and_heading(direction, current_pos, current_heading_degrees)
                        if potential_pos not in visited_nodes:
                            possible_moves.append(direction)
                
                if possible_moves:
                    chosen_direction = possible_moves[0]
                    path_stack.append((current_pos, current_heading_degrees))
                    current_pos, target_heading_degrees = get_new_pos_and_heading(chosen_direction, current_pos, current_heading_degrees)
                    visited_nodes.add(current_pos)
                    print(f"ตัดสินใจเลือกเดินไปทาง: {chosen_direction}")
                    robot_state = "TURNING"
                else:
                    print("เจอทางตัน! เริ่มทำการ Backtrack")
                    robot_state = "BACKTRACKING"

            elif robot_state == "TURNING":
                turn_to_angle(ep_chassis, ep_gimbal, target_heading_degrees)
                current_heading_degrees = target_heading_degrees
                robot_state = "MOVING"

            elif robot_state == "MOVING":
                move_straight_60cm(ep_chassis, current_heading_degrees)
                print(f"ถึงโหนดใหม่ที่ {current_pos}, ทิศทาง {current_heading_degrees}°")
                robot_state = "SCANNING"

            elif robot_state == "BACKTRACKING":
                if not path_stack:
                    print("สำรวจครบทุกเส้นทางแล้ว! จบการทำงาน"); stop_flag = True; break
                
                last_pos, last_heading = path_stack.pop()
                target_x, target_y = last_pos
                current_x, current_y = current_pos
                backtrack_heading = math.degrees(math.atan2(target_x - current_x, target_y - current_y))

                print(f"กำลังย้อนกลับไปยัง {last_pos}")
                turn_to_angle(ep_chassis, ep_gimbal, backtrack_heading)
                move_straight_60cm(ep_chassis, backtrack_heading)
                current_pos = last_pos
                current_heading_degrees = last_heading
                robot_state = "SCANNING"

    finally:
        stop_flag = True # ส่งสัญญาณให้ thread หยุดทำงาน
        time.sleep(0.2) # รอเล็กน้อย
        ep_chassis.drive_speed(x=0, y=0, z=0)
        ep_sensor.unsub_distance()
        ep_chassis.unsub_attitude()
        ep_robot.close()
        print("โปรแกรมปิดตัวลงเรียบร้อย")