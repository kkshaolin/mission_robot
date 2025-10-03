from robomaster import robot  # ไลบรารีสำหรับควบคุมหุ่นยนต์ RoboMaster
import msvcrt  # สำหรับตรวจจับการกดปุ่ม ESC เพื่อหยุดโปรแกรม
import time  # สำหรับจัดการเวลาและ delay
import math  # สำหรับคำนวณมุมและตำแหน่ง
import threading  # สำหรับรันการอ่านค่า IR แบบ concurrent
import matplotlib.pyplot as plt
import numpy as np

stop_flag = False

# ตัวแปรสำหรับเก็บค่าจากเซ็นเซอร์ ToF และ IMU (อัปเดตโดย callback functions)
tof_distance_cm = 999.0  # ระยะทางจาก ToF sensor (เริ่มต้นเป็นค่าไกลมาก)
current_yaw = 0.0  # มุม yaw ปัจจุบันของหุ่นยนต์ (จาก IMU)

# ตัวแปรสำหรับ IR Sensors จาก Sensor Adaptor
ir_left_cm = 999.0  # ระยะทางด้านซ้าย (cm)
ir_right_cm = 999.0  # ระยะทางด้านขวา (cm)

# ตัวแปรเก็บค่าที่ผ่าน low-pass filter แล้ว (สำหรับลดสัญญาณรบกวน)
last_value_left = 0  # ค่า filtered ก่อนหน้าของ IR ซ้าย
last_value_right = 0  # ค่า filtered ก่อนหน้าของ IR ขวา

# --- ส่วนที่เพิ่มเข้ามา: ตารางเทียบค่า ADC เป็น CM สำหรับ IR Sensors ---
calibra_table_ir_right = {
    615: 5,  415: 15,  275: 25, 
    605: 10, 335: 20,  255: 30   
}
calibra_table_ir_left = {
    680: 5,  300: 15,  210: 25, 
    420: 10, 235: 20,  175: 30
}

SCAN_DURATION_S = 0.2  # เวลารอเพื่อให้ค่าเซ็นเซอร์นิ่งก่อนสแกน (วินาที)
TOF_WALL_THRESHOLD_CM = 60  # ถ้า ToF วัดได้มากกว่านี้ = ทางเปิด
IR_WALL_THRESHOLD_CM = 29   # ถ้า IR วัดได้มากกว่านี้ = ทางเปิด


# ค่าคงที่สำหรับระบบ DFS และการเคลื่อนที่
START_CELL = (0, 0)      #  จุดเริ่ม
MAP_BOUNDS = (3, 3)      #  ขนาน map
NODE_DISTANCE = 0.6      # โหนดในเขาวงกต (60 cm)

# --------------------------------------------------------

WALL_AVOID_THRESHOLD_CM = 10.0  # ถ้า IR วัดได้น้อยกว่านี้ = ใกล้กำแพงเกินไป ต้องขยับหนี
WALL_AVOID_SPEED_Y = 0.05    # ความเร็วในการขยับหนีกำแพง (m/s) ในแนวแกน Y (ซ้าย-ขวา)

MOVE_SPEED_X = 0.2  # ความเร็วในการเดินหน้า (m/s)
TURN_SPEED_Z = 60  # ความเร็วสูงสุดในการหมุน (degrees/s)

# ค่าคงที่สำหรับ PID Controller
# PID สำหรับการหมุน (Turn)
Kp_turn = 2.5  # Proportional gain สำหรับการหมุน (ยิ่งสูงยิ่งตอบสนองเร็ว)

# PID สำหรับการเดินตรง (Straight)
Kp_straight = 0.8  # Proportional gain
Ki_straight = 0.02  # Integral gain (แก้ error สะสม)
Kd_straight = 0.1  # Derivative gain (ลดการ overshoot)
integral_straight = 0.0  # ตัวแปรเก็บค่า error สะสม
last_error_straight = 0.0  # ตัวแปรเก็บค่า error ครั้งก่อน

# โครงสร้างข้อมูลสำหรับ DFS (Depth-First Search)
path_stack = []  # Stack เก็บเส้นทางที่เดินมา (สำหรับ backtracking)
visited_nodes = set()  # Set เก็บโหนดที่เคยไปแล้ว (ป้องกันไปซ้ำ)
current_pos = (0, 0)  # ตำแหน่งปัจจุบันในระบบพิกัด (x, y)
current_heading_degrees = 0  # ทิศทางที่หุ่นยนต์หันไป (0=เหนือ, 90=ตะวันออก, -90=ตะวันตก, 180=ใต้)


# WALL_THRESHOLD = 50
# CELL_SIZE = 0.60

# --- ฟังก์ชัน Callback สำหรับ ToF และ IMU (ยังใช้เหมือนเดิม) ---
def sub_tof_handler(sub_info):
    """
    Callback function สำหรับรับค่าจาก ToF sensor
    ถูกเรียกอัตโนมัติเมื่อ ToF sensor ส่งข้อมูลมา
    """
    global tof_distance_cm
    tof_distance_cm = sub_info[0] / 10.0  # แปลงจาก mm เป็น cm (หาร 10)

def sub_imu_handler(attitude_info):
    """
    Callback function สำหรับรับค่ามุมจาก IMU (Inertial Measurement Unit)
    ถูกเรียกอัตโนมัติเมื่อ IMU ส่งข้อมูล attitude มา
    """
    global current_yaw
    current_yaw = attitude_info[0]  # เก็บค่ามุม yaw (หมุนรอบแกนแนวตั้ง)

# --- ส่วนที่เพิ่มเข้ามา: ฟังก์ชันสำหรับ IR Sensors ---
def single_lowpass_filter(new_value, last_value, alpha=0.8):
    """
    Low-pass filter แบบง่าย (Exponential Moving Average)
    ใช้กรองสัญญาณรบกวนจาก sensor
    
    Parameters:
        new_value: ค่าใหม่ที่อ่านได้
        last_value: ค่าที่ filter แล้วจากรอบก่อน
        alpha: น้ำหนักของค่าใหม่ (0-1) ยิ่งใกล้ 1 ยิ่งตอบสนองเร็ว
    
    Returns:
        ค่าที่ผ่าน filter แล้ว
    """
    return alpha * new_value + (1.0 - alpha) * last_value

def adc_to_cm(adc_value, table):
    """
    แปลงค่า ADC (Analog-to-Digital Converter) เป็นระยะทาง (cm)
    โดยใช้ linear interpolation จากตาราง calibration
    
    Parameters:
        adc_value: ค่า ADC ที่อ่านได้จาก IR sensor
        table: ตาราง calibration {ADC: cm}
    
    Returns:
        ระยะทางเป็น cm หรือ 999.0 ถ้าค่าอยู่นอกช่วง
    """
    # เรียงตาราง calibration จากค่า ADC สูงไปต่ำ
    points = sorted(table.items(), key=lambda x: x[0], reverse=True)
    
    # ถ้าค่า ADC สูงกว่าค่าสูงสุดในตาราง = วัตถุอยู่ใกล้มาก
    if adc_value >= points[0][0]: 
        return float(points[0][1])
    
    # ถ้าค่า ADC ต่ำกว่าค่าต่ำสุดในตาราง = วัตถุอยู่ไกลมาก
    if adc_value <= points[-1][0]: 
        return float(points[-1][1])
    
    # หาจุดสองจุดที่อยู่ข้างๆ adc_value แล้วทำ linear interpolation
    for i in range(len(points) - 1):
        x1, y1 = points[i]  # จุดที่มีค่า ADC สูงกว่า
        x2, y2 = points[i+1]  # จุดที่มีค่า ADC ต่ำกว่า
        if x2 <= adc_value <= x1:
            # สูตร linear interpolation: y = y1 + (x-x1) * (y2-y1)/(x2-x1)
            return float(y1 + (adc_value - x1) * (y2 - y1) / (x2 - x1))
    
    return 999.0  # คืนค่าระยะไกลถ้าไม่อยู่ในตาราง

# --- ส่วนที่เพิ่มเข้ามา: Thread สำหรับอ่านค่า IR ตลอดเวลา ---
def read_ir_thread(ep_sensor_adaptor):
    """
    Thread function ที่ทำงานเบื้องหลัง (background)
    อ่านค่า IR sensors ต่อเนื่องและอัปเดตตัวแปร global
    
    หน้าที่:
    1. อ่านค่า ADC จาก IR sensors
    2. กรองสัญญาณด้วย low-pass filter
    3. แปลงค่า ADC เป็น cm
    4. อัปเดตตัวแปร global (ir_left_cm, ir_right_cm)
    5. ทำซ้ำทุก 50ms จนกว่า stop_flag = True
    """
    global ir_right_cm, ir_left_cm, last_value_right, last_value_left
    # print("IR reading thread started.")  # แจ้งว่า thread เริ่มทำงานแล้ว
    
    # Loop ทำงานจนกว่าจะได้รับสัญญาณหยุด
    while not stop_flag:
        # อ่านค่า ADC ดิบจาก sensor adaptor
        ir_right_adc = ep_sensor_adaptor.get_adc(id=2, port=2)
        ir_left_adc = ep_sensor_adaptor.get_adc(id=1, port=2)

        # กรองสัญญาณเพื่อลด noise
        ir_right_filtered = single_lowpass_filter(ir_right_adc, last_value_right)
        ir_left_filtered = single_lowpass_filter(ir_left_adc, last_value_left)
        
        # เก็บค่าที่ filter แล้วไว้สำหรับรอบถัดไป
        last_value_right = ir_right_filtered
        last_value_left = ir_left_filtered

        # แปลงค่า ADC ที่ filter แล้วเป็น cm และอัปเดตตัวแปร global
        ir_right_cm = adc_to_cm(ir_right_filtered, calibra_table_ir_right)
        ir_left_cm = adc_to_cm(ir_left_filtered, calibra_table_ir_left)

        time.sleep(0.05)  # รอ 50ms ก่อนอ่านค่าครั้งถัดไป (อ่าน ~20 ครั้ง/วินาที)
    
    # print("IR reading thread stopped.")  # แจ้งว่า thread หยุดทำงานแล้ว

# --- อัปเดต: ฟังก์ชันสแกน ให้ใช้ตัวแปร ir_left_cm, ir_right_cm ---
def scan_environment():
    """
    สแกนสภาพแวดล้อมรอบตัวโดยไม่ต้องหมุน (Passive Scanning)
    ใช้เซ็นเซอร์ทั้ง 3 ตัวตรวจจับทางเปิด
    """
    global tof_distance_cm, ir_left_cm, ir_right_cm
    
    open_paths = {'front': False, 'left': False, 'right': False} # สร้าง dictionary เก็บผลสแกน
    time.sleep(SCAN_DURATION_S)     # รอให้ค่าเซ็นเซอร์นิ่งสักครู่ (เผื่อหุ่นยนต์เพิ่งหยุด)

    # ตรวจสอบเส้นทางด้านหน้าด้วย ToF
    if tof_distance_cm > TOF_WALL_THRESHOLD_CM: 
        open_paths['front'] = True
    
    # ตรวจสอบเส้นทางด้านซ้ายด้วย IR
    if ir_left_cm > IR_WALL_THRESHOLD_CM: 
        open_paths['left'] = True
    
    # ตรวจสอบเส้นทางด้านขวาด้วย IR
    if ir_right_cm > IR_WALL_THRESHOLD_CM: 
        open_paths['right'] = True

    # แสดงผลการสแกน
    """Returns:
          dict: {'front': bool, 'left': bool, 'right': bool}
               True = ทางเปิด, False = มีกำแพง     """
    # print(f"front: {tof_distance_cm:.1f} cm | left : {ir_left_cm:.1f} cm | right: {ir_right_cm:.1f} cm")
    return open_paths

def get_new_pos_and_heading(direction, old_pos, old_heading):
    """คำนวณตำแหน่งและทิศทางใหม่หลังจากเดินไปทิศทางที่กำหนด
        Parameters:
            direction: 'front', 'left', หรือ 'right'
            old_pos: ตำแหน่งเดิม (x, y)
            old_heading: ทิศทางเดิม (degrees)
        """
    x, y = old_pos  # แยกพิกัด x, y จาก tuple
    new_heading = old_heading  # เริ่มต้นทิศทางใหม่เท่ากับเดิม
    
    # คำนวณทิศทางใหม่ตามทิศที่เลือก
    if direction == 'left': # หมุนซ้าย = ลบมุม 90°
        new_heading = normalize_angle(old_heading - 90)
    elif direction == 'right': # หมุนขวา = บวกมุม 90°
        new_heading = normalize_angle(old_heading + 90)

    # ถ้าเป็น 'front' ทิศทางไม่เปลี่ยน
    
    # คำนวณตำแหน่งใหม่ตามทิศทางที่หัน
    if new_heading == 0: 
        y += 1  # หันไปทางเหนือ → y เพิ่ม
    elif new_heading == 90: 
        x += 1  # หันไปทางตะวันออก → x เพิ่ม
    elif new_heading == -90: 
        x -= 1  # หันไปทางตะวันตก → x ลด
    elif abs(new_heading) == 180: 
        y -= 1  # หันไปทางใต้ → y ลด
    
    """Returns: 
        tuple: (ตำแหน่งใหม่, ทิศทางใหม่) """
    return (x, y), new_heading


def decide_by_dfs(scan_results, current_pos, current_heading):
    """ฟังก์ชันตัดสินใจเลือกทิศทางการเดินโดยใช้หลักการ DFS (Depth-First Search)
       เพิ่มการตรวจสอบ MAZE_BOUNDS เพื่อป้องกันการออกนอกแผนที่"""
    global MAP_BOUNDS
    max_x, max_y = MAP_BOUNDS
    
    # สร้างลิสต์เปล่าสำหรับเก็บทิศทางที่สามารถเดินได้
    possible_moves = []

    # กำหนดลำดับการตรวจสอบทิศทาง: ซ้าย -> หน้า -> ขวา
    check_order = ['left', 'front', 'right']

    # วนลูปตรวจสอบแต่ละทิศทางตามลำดับที่กำหนด
    for direction in check_order:
        if scan_results.get(direction, False):  # ตรวจสอบว่าทิศทางนั้นสามารถเดินได้หรือไม่ (ไม่มีกำแพงขวาง)
            # คำนวณตำแหน่งใหม่และทิศทางหัวหุ่นยนต์หลังจากเดินไปในทิศทางนั้น
            pos, _ = get_new_pos_and_heading(direction, current_pos, current_heading)
            x, y = pos

            # ตรวจสอบว่าอยู่ในขอบเขตแผนที่หรือไม่
            if 0 <= x <= max_x and 0 <= y <= max_y:
                # ตรวจสอบว่าตำแหน่งใหม่นั้นเคยไปแล้วหรือยัง
                if pos not in visited_nodes:
                    possible_moves.append(direction)  # เพิ่มทิศทางนี้เข้าไปในลิสต์ของทางเลือกที่เป็นไปได้

    # คืนค่า: ทิศทางที่เลือก (string) หรือ None ถ้าไม่มีทางเดิน
    if possible_moves:
        return possible_moves[0]  # เลือกทางแรกในลิสต์
    return None  # (ทุกทางถูกปิด เคยไปแล้ว หรืออยู่นอกขอบเขต)


# --- ฟังก์ชันช่วย (Helper Functions) ---
def normalize_angle(angle):
    """
    ปรับมุมให้อยู่ในช่วง (-180, 180] degrees
    เพื่อให้การคำนวณ error ของมุมทำงานถูกต้อง
    
    ตัวอย่าง: 
    - 270° → -90°
    - -200° → 160°
    """
    while angle > 180: 
        angle -= 360  # ถ้ามากกว่า 180 ให้ลบ 360
    while angle <= -180: 
        angle += 360  # ถ้าน้อยกว่าหรือเท่ากับ -180 ให้บวก 360
    return angle

# --- ฟังก์ชันควบคุมการเคลื่อนที่ (ไม่มีการเปลี่ยนแปลง) ---
def turn_to_angle(ep_chassis, ep_gimbal, target_angle):
    """
    หมุนหุ่นยนต์ไปยังมุมที่ต้องการโดยใช้ PID แบบ Proportional
    
    Parameters:
        ep_chassis: object สำหรับควบคุมการเคลื่อนที่
        ep_gimbal: object สำหรับควบคุม gimbal (กล้อง)
        target_angle: มุมเป้าหมาย (degrees)
    """
    global current_yaw
    
    
    # Normalize มุมเป้าหมาย
    target_angle = normalize_angle(target_angle)
    print(f"กำลังหมุนไปที่ {target_angle:.1f}°...")
    
    # Loop จนกว่าจะหมุนถึงมุมเป้าหมาย (ยอมรับ error ±2°)
    while not stop_flag:
        # คำนวณความแตกต่างของมุม (error)
        angle_error = normalize_angle(target_angle - current_yaw)
        
        # ถ้า error น้อยกว่า 2° ถือว่าหมุนสำเร็จแล้ว
        if abs(angle_error) < 2.0: 
            break
        
        # คำนวณความเร็วการหมุนด้วย P-controller
        # คูณ error ด้วย Kp_turn และจำกัดไม่ให้เกิน TURN_SPEED_Z
        turn_speed = max(min(angle_error * Kp_turn, TURN_SPEED_Z), -TURN_SPEED_Z)
        
        # ส่งคำสั่งหมุน (z = angular velocity รอบแกนแนวตั้ง)
        ep_chassis.drive_speed(x=0, y=0, z=turn_speed)
        time.sleep(0.02)  # รอ 20ms ก่อนคำนวณใหม่
    
    # หยุดการหมุน
    ep_chassis.drive_speed(x=0, y=0, z=0)

    # รีเซ็ต gimbal ให้อยู่ตำแหน่งกลาง
    ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=100, yaw_speed=100)
    
    time.sleep(0.5)  # รอให้หุ่นยนต์หยุดนิ่ง
    print(f"หมุนสำเร็จ! มุมปัจจุบัน: {current_yaw:.1f}°")

# --- อัปเดต: ฟังก์ชันเดินตรง ให้ใช้ตัวแปร ir_left_cm, ir_right_cm ---
def move_straight_60cm(ep_chassis, target_yaw):
    """
    เดินหน้าตรงไป 60 cm พร้อมใช้ PID รักษาทิศทางและหลบกำแพง
    
    Features:
    1. PID Controller (แกน Z) - รักษาทิศทางให้ตรง
    2. Wall Avoidance (แกน Y) - ขยับหนีกำแพงถ้าเข้าใกล้เกินไป
    
    Parameters:
        ep_chassis: object สำหรับควบคุมการเคลื่อนที่
        target_yaw: มุมเป้าหมายที่ต้องการรักษา (degrees)
    """
    global integral_straight, last_error_straight, ir_left_cm, ir_right_cm
    
    print(f"กำลังเคลื่อนที่ไปข้างหน้า 60 cm (PID + Wall Avoidance) ที่มุม {target_yaw:.1f}°")
    
    # รีเซ็ตตัวแปร PID
    integral_straight, last_error_straight = 0.0, 0.0
    
    # คำนวณเวลาที่ต้องใช้ในการเดิน 60cm
    # เวลา = ระยะทาง / ความเร็ว, คูณ 1.05 เผื่อเวลาพอเคลื่อนที่ครบ
    start_time = time.time()
    last_time = start_time
    duration = (NODE_DISTANCE_M / MOVE_SPEED_X) * 1.05

    # Loop เดินไปข้างหน้าจนครบระยะทางหรือได้รับสัญญาณหยุด
    while (time.time() - start_time) < duration and not stop_flag:
        current_time = time.time()
        dt = current_time - last_time  # คำนวณช่วงเวลาที่ผ่านไป
        
        # ถ้า dt <= 0 (ไม่น่าจะเกิด แต่ป้องกันการหาร 0)
        if dt <= 0: 
            time.sleep(0.01)
            continue
        
        # === PID Controller สำหรับรักษาทิศทาง (แกน Z) ===
        # คำนวณ error (ความแตกต่างระหว่างมุมเป้าหมายกับมุมปัจจุบัน)
        error = normalize_angle(target_yaw - current_yaw)
        
        # คำนวณ integral (เก็บสะสม error)
        integral_straight += error * dt
        
        # คำนวณ derivative (อัตราการเปลี่ยนแปลงของ error)
        derivative = (error - last_error_straight) / dt
        
        # คำนวณความเร็วการหมุนด้วย PID
        z_correct_speed = (Kp_straight * error) + \
                         (Ki_straight * integral_straight) + \
                         (Kd_straight * derivative)
        
        # === Wall Avoidance สำหรับหลบกำแพง (แกน Y) ===
        y_correct_speed = 0.0  # เริ่มต้นไม่ขยับซ้าย-ขวา
        
        # ถ้าใกล้กำแพงด้านขวาเกินไป → ขยับไปทางซ้าย (y ติดลบ)
        if ir_right_cm < WALL_AVOID_THRESHOLD_CM: 
            y_correct_speed -= WALL_AVOID_SPEED_Y
        
        # ถ้าใกล้กำแพงด้านซ้ายเกินไป → ขยับไปทางขวา (y บวก)
        if ir_left_cm < WALL_AVOID_THRESHOLD_CM: 
            y_correct_speed += WALL_AVOID_SPEED_Y
        
        # ส่งคำสั่งเคลื่อนที่:
        # x = เดินหน้า (ความเร็วคงที่)
        # y = ขยับซ้าย-ขวา (หลบกำแพง)
        # z = หมุน (รักษาทิศทาง)
        ep_chassis.drive_speed(x=MOVE_SPEED_X, y=y_correct_speed, z=z_correct_speed)
        
        # เก็บค่า error และเวลาปัจจุบันไว้สำหรับรอบถัดไป
        last_error_straight, last_time = error, current_time
        time.sleep(0.02)  # รอ 20ms ก่อนคำนวณใหม่
    
    # หยุดการเคลื่อนที่
    ep_chassis.drive_speed(x=0, y=0, z=0)
    time.sleep(0.5)  # รอให้หุ่นยนต์หยุดนิ่ง
    print("เคลื่อนที่ 60 cm สำเร็จ")

def backtrack(ep_chassis, ep_gimbal):
    global current_pos, current_heading_degrees
    print("Backtracking...")
    # pop until we reach a cell that has unexplored neighbour or stack empty
    while path_stack and not stop_flag:
        last_pos, last_heading = path_stack.pop()
        # compute heading from current_pos to last_pos
        target_x, target_y = last_pos
        current_x, current_y = current_pos
        backtrack_heading = math.degrees(math.atan2(target_x - current_x, target_y - current_y))
        backtrack_heading = normalize_angle(backtrack_heading)
        print(f"Backtrack: going to {last_pos} heading {backtrack_heading:.1f}")
        turn_to_angle(ep_chassis, ep_gimbal, backtrack_heading)
        move_straight_60cm(ep_chassis, backtrack_heading)
        # update current pos/heading
        current_pos = last_pos
        current_heading_degrees = last_heading
        # after moving back, check if this cell has unexplored neighbor
        # We'll scan in main loop after returning
        return True
    return False



if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis  
    ep_sensor = ep_robot.sensor 
    ep_gimbal = ep_robot.gimbal
    ep_sensor_adaptor = ep_robot.sensor_adaptor
 
    ep_sensor.sub_distance(freq=20, callback=sub_tof_handler) # Subscribe ToF sensor and IMU (attitude)
    ep_chassis.sub_attitude(freq=20, callback=sub_imu_handler)
    time.sleep(0.5)  # รอให้ subscription เริ่มทำงาน

    # thread การอ่านค่า IR ---
    ir_reader = threading.Thread(target=read_ir_thread, args=(ep_sensor_adaptor,),daemon=True)
    ir_reader.start()
    time.sleep(0.5)

    visited_nodes.add(current_pos)  # เพิ่มตำแหน่งเริ่มต้นเข้า
    
    # main loop
    while not stop_flag:
            # กดปุ่ม ESC เพื่อหยุดโปรแกรม
            if msvcrt.kbhit() and msvcrt.getch() == b'\x1b': 
                stop_flag = True
                break
            
            # reset gimbal
            ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=50, yaw_speed=100).wait_for_completed()
            
            # สแกนสภาพแวดล้อมรอบตัว
            scan_results = scan_environment()
            print(scan_results)
            
            chosen = decide_by_dfs(scan_results, current_pos, current_heading_degrees)
            # if chosen:
            #     path_stack.append((current_pos, current_heading_degrees))
            #     new_pos, new_heading = get_new_pos_and_heading(chosen, current_pos, current_heading_degrees)
                # TURNING
                # turn_to_angle(ep_chassis, ep_gimbal, new_heading)
                # current_heading_degrees = new_heading
                # MOVING
                # move_straight_60cm(ep_chassis, current_heading_degrees)
                # current_pos = new_pos
                # visited_nodes.add(current_pos)
        
                    

            # === State: TURNING ===       
            # turn_to_angle(ep_chassis, ep_gimbal, target_heading_degrees)
            # current_heading_degrees = target_heading_degrees  # อัปเดตทิศทางปัจจุบัน

            # # === State: MOVING ===
            # move_straight_60cm(ep_chassis, current_heading_degrees)
            # print(f"ถึงโหนดใหม่ที่ {current_pos}, ทิศทาง {current_heading_degrees}°")
               
                
        
    stop_flag = True  # ส่งสัญญาณให้ IR thread หยุดทำงาน
    time.sleep(0.2)  # รอให้ thread มีเวลาหยุด
        
    ep_chassis.drive_speed(x=0, y=0, z=0)
    ep_sensor.unsub_distance() 
    ep_chassis.unsub_attitude()  
    ep_robot.close()
    # plot_maze(walls, path_stack[-1], visited, "Final Exploration Map")
    # _fig.savefig("final_maze_map.png", dpi=300)
    # finalize_show()
        
    # print("โปรแกรมปิดตัวลงเรียบร้อย")