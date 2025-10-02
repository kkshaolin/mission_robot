"""
=== สรุปการทำงานของโปรแกรม ===
โปรแกรมนี้ใช้ควบคุมหุ่นยนต์ RoboMaster ให้สำรวจเขาวงกตโดยใช้อัลกอริทึม DFS (Depth-First Search)

หลักการทำงาน:
1. หุ่นยนต์ใช้เซ็นเซอร์ 3 ตัวในการตรวจจับกำแพง:
   - ToF (Time of Flight) ด้านหน้า - วัดระยะทางข้างหน้า
   - IR (Infrared) ซ้าย - วัดระยะทางด้านซ้าย
   - IR (Infrared) ขวา - วัดระยะทางด้านขวา

2. การสำรวจทำงานแบบ State Machine มี 5 สถานะ:
   - SCANNING: สแกนหาทางเปิดรอบตัว (หน้า/ซ้าย/ขวา)
   - DECIDING: ตัดสินใจเลือกทิศทางที่จะไป (ยังไม่เคยไป)
   - TURNING: หมุนหุ่นยนต์ไปยังทิศทางที่เลือก
   - MOVING: เดินตรงไป 60cm (ระยะห่างระหว่างโหนด)
   - BACKTRACKING: ถอยกลับเมื่อเจอทางตัน

3. ระหว่างเดิน หุ่นยนต์จะ:
   - ใช้ PID Controller รักษาทิศทางให้ตรง
   - หลบกำแพงอัตโนมัติถ้าเข้าใกล้เกินไป (< 10cm)

4. ใช้ Threading เพื่ออ่านค่าเซ็นเซอร์ IR แบบ Real-time ในพื้นหลัง

5. บันทึกโหนดที่เคยไปแล้วและเส้นทางที่เดิน เพื่อใช้ใน Backtracking
"""

from robomaster import robot  # ไลบรารีสำหรับควบคุมหุ่นยนต์ RoboMaster
import msvcrt  # สำหรับตรวจจับการกดปุ่ม ESC เพื่อหยุดโปรแกรม
import time  # สำหรับจัดการเวลาและ delay
import math  # สำหรับคำนวณมุมและตำแหน่ง
import threading  # สำหรับรันการอ่านค่า IR แบบ concurrent

# --- ค่าคงที่และตัวแปร Global ---
# ตัวแปรสัญญาณหยุดโปรแกรม (ใช้ร่วมกันระหว่าง main thread และ IR thread)
stop_flag = False

# ตัวแปรสำหรับเก็บค่าจากเซ็นเซอร์ ToF และ IMU (อัปเดตโดย callback functions)
tof_distance_cm = 999.0  # ระยะทางจาก ToF sensor (เริ่มต้นเป็นค่าไกลมาก)
current_yaw = 0.0  # มุม yaw ปัจจุบันของหุ่นยนต์ (จาก IMU)

# --- ส่วนที่เพิ่มเข้ามา: ตัวแปรสำหรับ IR Sensors จาก Sensor Adaptor ---
# ตัวแปรเก็บระยะทางจาก IR sensors (อัปเดตโดย IR reading thread)
ir_left_cm = 999.0  # ระยะทางด้านซ้าย (cm)
ir_right_cm = 999.0  # ระยะทางด้านขวา (cm)
# ตัวแปรเก็บค่าที่ผ่าน low-pass filter แล้ว (สำหรับลดสัญญาณรบกวน)
last_value_left = 0  # ค่า filtered ก่อนหน้าของ IR ซ้าย
last_value_right = 0  # ค่า filtered ก่อนหน้าของ IR ขวา

# --- ส่วนที่เพิ่มเข้ามา: ตารางเทียบค่า ADC เป็น CM สำหรับ IR Sensors ---
# *** คุณควรทำการ Calibrate ค่าเหล่านี้ใหม่เพื่อให้แม่นยำกับหุ่นยนต์ของคุณ ***
# ตาราง lookup: {ค่า_ADC: ระยะทาง_cm} สำหรับแปลงค่า analog เป็นระยะทางจริง
calibra_table_ir_right = {
    615: 5,  415: 15,  275: 25,  # ค่า ADC สูง = ใกล้
    605: 10, 335: 20,  255: 30   # ค่า ADC ต่ำ = ไกล
}
calibra_table_ir_left = {
    680: 5,  300: 15,  210: 25,  # ค่าจะแตกต่างกันเพราะ sensor แต่ละตัวมี characteristic ต่างกัน
    420: 10, 235: 20,  175: 30
}

# ค่าคงที่สำหรับระบบ DFS และการเคลื่อนที่
NODE_DISTANCE_M = 0.6      # ระยะห่างระหว่างโหนดในเขาวงกต (60 cm = 0.6 m)
TOF_WALL_THRESHOLD_CM = 30.0  # ถ้า ToF วัดได้มากกว่านี้ = ทางเปิด (ไม่มีกำแพงด้านหน้า)
IR_WALL_THRESHOLD_CM = 25.0   # ถ้า IR วัดได้มากกว่านี้ = ทางเปิด (ไม่มีกำแพงด้านข้าง)
WALL_AVOID_THRESHOLD_CM = 10.0  # ถ้า IR วัดได้น้อยกว่านี้ = ใกล้กำแพงเกินไป ต้องขยับหนี
WALL_AVOID_SPEED_Y = 0.05    # ความเร็วในการขยับหนีกำแพง (m/s) ในแนวแกน Y (ซ้าย-ขวา)
SCAN_DURATION_S = 0.2  # เวลารอเพื่อให้ค่าเซ็นเซอร์นิ่งก่อนสแกน (วินาที)
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
    print("IR reading thread started.")  # แจ้งว่า thread เริ่มทำงานแล้ว
    
    # Loop ทำงานจนกว่าจะได้รับสัญญาณหยุด
    while not stop_flag:
        # อ่านค่า ADC ดิบจาก sensor adaptor
        # id=2, port=2 = IR sensor ขวา
        ir_right_adc = ep_sensor_adaptor.get_adc(id=2, port=2)
        # id=1, port=2 = IR sensor ซ้าย
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
    
    print("IR reading thread stopped.")  # แจ้งว่า thread หยุดทำงานแล้ว

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
    
    # รีเซ็ต gimbal ให้อยู่ตำแหน่งกลาง
    ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=100, yaw_speed=100)
    
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

# --- อัปเดต: ฟังก์ชันสแกน ให้ใช้ตัวแปร ir_left_cm, ir_right_cm ---
def scan_environment():
    """
    สแกนสภาพแวดล้อมรอบตัวโดยไม่ต้องหมุน (Passive Scanning)
    ใช้เซ็นเซอร์ทั้ง 3 ตัวตรวจจับทางเปิด
    
    Returns:
        dict: {'front': bool, 'left': bool, 'right': bool}
              True = ทางเปิด, False = มีกำแพง
    """
    global tof_distance_cm, ir_left_cm, ir_right_cm
    
    print("--- เริ่มการสแกนสภาพแวดล้อม (แบบไม่หมุน) ---")
    
    # สร้าง dictionary เก็บผลสแกน (เริ่มต้นเป็น False ทั้งหมด)
    open_paths = {'front': False, 'left': False, 'right': False}
    
    # รอให้ค่าเซ็นเซอร์นิ่งสักครู่ (เผื่อหุ่นยนต์เพิ่งหยุด)
    time.sleep(SCAN_DURATION_S)

    # ตรวจสอบเส้นทางด้านหน้าด้วย ToF
    if tof_distance_cm > TOF_WALL_THRESHOLD_CM: 
        open_paths['front'] = True  # ถ้าวัดได้ > 30cm = ทางเปิด
    
    # ตรวจสอบเส้นทางด้านซ้ายด้วย IR
    if ir_left_cm > IR_WALL_THRESHOLD_CM: 
        open_paths['left'] = True  # ถ้าวัดได้ > 25cm = ทางเปิด
    
    # ตรวจสอบเส้นทางด้านขวาด้วย IR
    if ir_right_cm > IR_WALL_THRESHOLD_CM: 
        open_paths['right'] = True  # ถ้าวัดได้ > 25cm = ทางเปิด

    # แสดงผลการสแกน
    print(f"ผลสแกน [หน้า-ToF]: {tof_distance_cm:.1f} cm | [ซ้าย-IR]: {ir_left_cm:.1f} cm | [ขวา-IR]: {ir_right_cm:.1f} cm")
    print(f"--- สแกนเสร็จสิ้น: {open_paths} ---")
    
    return open_paths

def get_new_pos_and_heading(direction, old_pos, old_heading):
    """คำนวณตำแหน่งและทิศทางใหม่หลังจากเดินไปทิศทางที่กำหนด
        Parameters:
            direction: 'front', 'left', หรือ 'right'
            old_pos: ตำแหน่งเดิม (x, y)
            old_heading: ทิศทางเดิม (degrees)
        
        Returns:
            tuple: (ตำแหน่งใหม่, ทิศทางใหม่)
        
        หลักการ:
        - ระบบพิกัด: y+ = เหนือ, x+ = ตะวันออก
        - ทิศทาง: 0°=เหนือ, 90°=ตะวันออก, -90°=ตะวันตก, ±180°=ใต้
        """
    x, y = old_pos  # แยกพิกัด x, y จาก tuple
    new_heading = old_heading  # เริ่มต้นทิศทางใหม่เท่ากับเดิม
    
    # คำนวณทิศทางใหม่ตามทิศที่เลือก
    if direction == 'left': 
        # หมุนซ้าย = ลบมุม 90°
        new_heading = normalize_angle(old_heading - 90)
    elif direction == 'right': 
        # หมุนขวา = บวกมุม 90°
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
    
    return (x, y), new_heading

# --- ส่วนหลักของโปรแกรม (Main Program) ---
if __name__ == '__main__':
    # === การเริ่มต้นหุ่นยนต์ ===
    ep_robot = robot.Robot()  # สร้าง object หุ่นยนต์
    ep_robot.initialize(conn_type="ap")  # เชื่อมต่อแบบ Access Point (WiFi Direct)

    # สร้าง object สำหรับควบคุมส่วนต่างๆ ของหุ่นยนต์
    ep_chassis = ep_robot.chassis  # ควบคุมการเคลื่อนที่
    ep_sensor = ep_robot.sensor  # ควบคุม ToF sensor
    ep_gimbal = ep_robot.gimbal  # ควบคุม gimbal (กล้อง)
    ep_sensor_adaptor = ep_robot.sensor_adaptor  # ควบคุม sensor adaptor (สำหรับ IR)

    # === เริ่ม Subscription สำหรับเซ็นเซอร์ ===
    # Subscribe ToF sensor ให้ส่งข้อมูล 20 ครั้ง/วินาที มายัง callback
    ep_sensor.sub_distance(freq=20, callback=sub_tof_handler)
    
    # Subscribe IMU (attitude) ให้ส่งข้อมูล 20 ครั้ง/วินาที มายัง callback
    ep_chassis.sub_attitude(freq=20, callback=sub_imu_handler)
    
    time.sleep(1)  # รอให้ subscription เริ่มทำงาน

    # --- ส่วนที่เพิ่มเข้ามา: เริ่ม thread การอ่านค่า IR ---
    # สร้าง thread สำหรับอ่านค่า IR sensors แบบ background
    # daemon=True = thread จะปิดตัวอัตโนมัติเมื่อ main program จบ
    ir_reader = threading.Thread(
        target=read_ir_thread,  # ฟังก์ชันที่จะรันใน thread
        args=(ep_sensor_adaptor,),  # argument ส่งให้ฟังก์ชัน
        daemon=True  # ตั้งเป็น daemon thread
    )
    ir_reader.start()  # เริ่มการทำงานของ thread
    time.sleep(1)  # รอให้ thread เริ่มอ่านค่าและอัปเดตตัวแปร global

    # === เริ่มต้นระบบ DFS ===
    robot_state = "SCANNING"  # สถานะเริ่มต้นของ state machine
    visited_nodes.add(current_pos)  # เพิ่มตำแหน่งเริ่มต้นเข้า visited set
    print(f"เริ่มต้น DFS ที่ตำแหน่ง {current_pos}, ทิศทาง {current_heading_degrees}°")

    try:
        # === State Machine Loop หลัก ===
        # Loop นี้จะทำงานเรื่อยๆ จนกว่า stop_flag = True
        while not stop_flag:
            # ตรวจจับการกดปุ่ม ESC เพื่อหยุดโปรแกรม
            if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':  # \x1b = ESC key
                stop_flag = True
                break

            # === State: SCANNING ===
            # สแกนสภาพแวดล้อมรอบตัว
            if robot_state == "SCANNING":
                scan_results = scan_environment()  # เรียกฟังก์ชันสแกน
                robot_state = "DECIDING"  # เปลี่ยนไปสถานะตัดสินใจ
            
            # === State: DECIDING ===
            # ตัดสินใจว่าจะเดินไปทิศทางไหน
            elif robot_state == "DECIDING":
                possible_moves = []  # list เก็บทิศทางที่เป็นไปได้
                check_order = ['front', 'left', 'right']  # ลำดับการตรวจสอบ (DFS ลำดับ)
                
                # วนตรวจสอบแต่ละทิศทาง
                for direction in check_order:
                    # ถ้าทิศทางนั้นเป็นทางเปิด
                    if scan_results[direction]:
                        # คำนวณตำแหน่งที่จะไปถึงถ้าเดินไปทิศทางนั้น
                        potential_pos, _ = get_new_pos_and_heading(
                            direction, current_pos, current_heading_degrees
                        )
                        # ถ้าโหนดนั้นยังไม่เคยไป → เพิ่มเข้า list
                        if potential_pos not in visited_nodes:
                            possible_moves.append(direction)
                
                # ถ้ามีทางเปิดที่ยังไม่เคยไป
                if possible_moves:
                    # เลือกทิศทางแรกใน list (DFS pattern)
                    chosen_direction = possible_moves[0]
                    
                    # บันทึกตำแหน่งและทิศทางปัจจุบันลง stack (สำหรับ backtrack)
                    path_stack.append((current_pos, current_heading_degrees))
                    
                    # คำนวณตำแหน่งและทิศทางใหม่
                    current_pos, target_heading_degrees = get_new_pos_and_heading(
                        chosen_direction, current_pos, current_heading_degrees
                    )
                    
                    # เพิ่มโหนดใหม่เข้า visited set
                    visited_nodes.add(current_pos)
                    
                    print(f"ตัดสินใจเลือกเดินไปทาง: {chosen_direction}")
                    robot_state = "TURNING"  # เปลี่ยนไปสถานะหมุน
                
                # ถ้าไม่มีทางใหม่ให้ไป = เจอทางตัน
                else:
                    print("เจอทางตัน! เริ่มทำการ Backtrack")
                    robot_state = "BACKTRACKING"  # เปลี่ยนไปสถานะย้อนกลับ

            # === State: TURNING ===
            # หมุนหุ่นยนต์ไปยังทิศทางที่เลือก
            elif robot_state == "TURNING":
                turn_to_angle(ep_chassis, ep_gimbal, target_heading_degrees)
                current_heading_degrees = target_heading_degrees  # อัปเดตทิศทางปัจจุบัน
                robot_state = "MOVING"  # เปลี่ยนไปสถานะเคลื่อนที่

            # === State: MOVING ===
            # เดินไปข้างหน้า 60 cm
            elif robot_state == "MOVING":
                move_straight_60cm(ep_chassis, current_heading_degrees)
                print(f"ถึงโหนดใหม่ที่ {current_pos}, ทิศทาง {current_heading_degrees}°")
                robot_state = "SCANNING"  # กลับไปสแกนใหม่

            # === State: BACKTRACKING ===
            # ย้อนกลับไปยังโหนดก่อนหน้า
            elif robot_state == "BACKTRACKING":
                # ถ้า stack ว่าง = สำรวจครบทุกทางแล้ว
                if not path_stack:
                    print("สำรวจครบทุกเส้นทางแล้ว! จบการทำงาน")
                    stop_flag = True  # ตั้ง flag ให้หยุด
                    break
                
                # ดึงตำแหน่งและทิศทางก่อนหน้าออกจาก stack
                last_pos, last_heading = path_stack.pop()
                
                # คำนวณมุมที่ต้องหมุนเพื่อหันไปหา last_pos
                target_x, target_y = last_pos
                current_x, current_y = current_pos
                # ใช้ arctan2 คำนวณมุมจากพิกัด (ระวัง: y, x สลับกัน)
                backtrack_heading = math.degrees(
                    math.atan2(target_x - current_x, target_y - current_y)
                )

                print(f"กำลังย้อนกลับไปยัง {last_pos}")
                
                # หมุนไปหาทิศทางที่ต้องย้อนกลับ
                turn_to_angle(ep_chassis, ep_gimbal, backtrack_heading)
                
                # เดินกลับไป 60 cm
                move_straight_60cm(ep_chassis, backtrack_heading)
                
                # อัปเดตตำแหน่งและทิศทางปัจจุบัน
                current_pos = last_pos
                current_heading_degrees = last_heading
                
                robot_state = "SCANNING"  # กลับไปสแกนใหม่

    # === Exception Handling และ Cleanup ===
    finally:
        # บลอก finally จะทำงานเสมอไม่ว่าจะจบปกติหรือเกิด error
        
        stop_flag = True  # ส่งสัญญาณให้ IR thread หยุดทำงาน
        time.sleep(0.2)  # รอให้ thread มีเวลาหยุด
        
        # หยุดการเคลื่อนที่
        ep_chassis.drive_speed(x=0, y=0, z=0)
        
        # ยกเลิก subscription ทั้งหมด
        ep_sensor.unsub_distance()  # หยุดรับข้อมูล ToF
        ep_chassis.unsub_attitude()  # หยุดรับข้อมูล IMU
        
        # ปิดการเชื่อมต่อกับหุ่นยนต์
        ep_robot.close()
        
        print("โปรแกรมปิดตัวลงเรียบร้อย")