from robomaster import robot  # ไลบรารีสำหรับควบคุมหุ่นยนต์ RoboMaster
import msvcrt  # สำหรับตรวจจับการกดปุ่ม ESC เพื่อหยุดโปรแกรม (Windows)
import time  # สำหรับจัดการเวลาและ delay
import math  # สำหรับคำนวณมุมและตำแหน่ง (atan2, degrees)
import threading  # สำหรับรันการอ่านค่า IR แบบ concurrent (พร้อมกัน)
import matplotlib.pyplot as plt  # สำหรับวาดกราฟแผนที่เขาวงกต
import numpy as np  # สำหรับจัดการอาร์เรย์และช่วงตัวเลข

# ===================== Global State & Constants =====================
# ตัวแปร flag สำหรับหยุดโปรแกรมทั้งหมด (ใช้ร่วมกันทุก thread)
stop_flag = False

# --- ตัวแปรสถานะส่วนกลาง (Global State Variables) ---
# ระยะทางที่วัดได้จากเซ็นเซอร์ ToF (Time of Flight) หน่วยเป็น cm
tof_distance_cm = 999.0
# มุม yaw ปัจจุบันของหุ่นยนต์ (จาก IMU) หน่วยเป็นองศา
current_yaw = 0.0

current_x = 0.0  # ตำแหน่ง x จาก position sensor (เมตร)
current_y = 0.0  # ตำแหน่ง y จาก position sensor (เมตร)

# ระยะทางจากกำแพงด้านซ้ายและขวา (จากเซ็นเซอร์ IR) หน่วยเป็น cm
ir_left_cm = 999.0
ir_right_cm = 999.0

# ค่าล่าสุดของเซ็นเซอร์ IR (ใช้สำหรับ low-pass filter)
last_value_left = 0
last_value_right = 0

# --- Maze State Variables (จัดการสถานะของ DFS) ---
# เก็บข้อมูลแผนที่: key=ตำแหน่ง(x,y), value=set ของทิศทางที่เปิด (degrees)
maze_map = {}
# เก็บตำแหน่งทั้งหมดที่เคยไปแล้ว
visited_nodes = set()
# stack สำหรับเก็บเส้นทางการเดินตาม DFS
path_stack = []
# เก็บพิกัดของกำแพงทั้งหมด (tuple ของคู่ตำแหน่ง)
walls = set()
# ตำแหน่งปัจจุบันของหุ่นยนต์ในเขาวงกต (x, y)
current_pos = (1, 1) # ตำแหน่งเริ่มต้น (x, y)
# ทิศทางหัวหุ่นยนต์ปัจจุบัน: 0=เหนือ, 90=ตะวันออก, -90=ตะวันตก, 180=ใต้
current_heading_degrees = 0 # ทิศทางเริ่มต้น 0=N, 90=E, -90=W, 180=S

# --- ค่าคงที่สำหรับเขาวงกตและการเคลื่อนที่ ---
# ระยะเวลาในการสแกนสภาพแวดล้อม (วินาที)
SCAN_DURATION_S = 0.2
# ระยะทาง ToF ที่ถือว่ามีกำแพง (cm)
TOF_WALL_THRESHOLD_CM = 60
# ระยะทาง IR ที่ถือว่ามีกำแพงด้านข้าง (cm)
IR_WALL_THRESHOLD_CM = 29
# ตำแหน่งเริ่มต้นในเขาวงกต
START_CELL = (1, 1)
# ขอบเขตขั้นต่ำของแผนที่ (x_min, y_min)
MAP_MIN_BOUNDS = (1, 1)
# ขอบเขตสูงสุดของแผนที่ (x_max, y_max)
MAP_MAX_BOUNDS = (3, 3)
# ระยะทางระหว่างช่องในเขาวงกต (เมตร)
NODE_DISTANCE = 0.6
# ระยะที่เริ่มหลีกเลี่ยงกำแพงด้านข้าง (cm)
WALL_AVOID_THRESHOLD_CM = 10.0
# ความเร็วในการขยับหลีกเลี่ยงกำแพง (m/s)
WALL_AVOID_SPEED_Y = 0.05
# ความเร็วการเดินหน้าสูงสุด (m/s)
MOVE_SPEED_X = 2
# ความเร็วการหมุนสูงสุด (degrees/s)
TURN_SPEED_Z = 60

# --- PID Controller Gains ---
# ค่า Proportional gain สำหรับการหมุน
Kp_turn = 2.5

# --- ตารางเทียบค่า ADC เป็น CM สำหรับ IR Sensors ---
# ตาราง calibration สำหรับเซ็นเซอร์ IR ด้านขวา: {ค่า ADC: ระยะทาง cm}
calibra_table_ir_right = {615: 5, 605: 10, 415: 15, 335: 20, 275: 25, 255: 30}
# ตาราง calibration สำหรับเซ็นเซอร์ IR ด้านซ้าย: {ค่า ADC: ระยะทาง cm}
calibra_table_ir_left = {680: 5, 420: 10, 300: 15, 235: 20, 210: 25, 175: 30}

# ตัวแปรสำหรับ Plot กราฟ
# สร้าง figure และ axes สำหรับแสดงแผนที่
_fig, _ax = plt.subplots(figsize=(6, 6))

# ===================== Plotting Functions =====================
def plot_maze(walls_to_plot, visited_to_plot, path_stack_to_plot, current_cell_to_plot, title="Maze Exploration"):
    """วาดสถานะปัจจุบันของการสำรวจเขาวงกต"""
    # ล้างกราฟเดิมออก
    _ax.clear()
    # กำหนดขอบเขตของเขาวงกตที่จะวาด (x_min, x_max, y_min, y_max)
    MAZE_BOUNDS_PLOT = (0, 4, 0, 4)
    x_min, x_max = MAZE_BOUNDS_PLOT[0], MAZE_BOUNDS_PLOT[1]
    y_min, y_max = MAZE_BOUNDS_PLOT[2], MAZE_BOUNDS_PLOT[3]

    # วาดช่องที่เคยไปแล้วเป็นสีฟ้าอ่อน
    for x, y in visited_to_plot:
        # เพิ่มสี่เหลี่ยมขนาด 1x1 กึ่งกลางอยู่ที่ (x, y)
        _ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightcyan', edgecolor='none', zorder=0))

    # วาดกำแพงเป็นเส้นสีดำหนา
    for wall in walls_to_plot:
        # กำแพงแต่ละอันเก็บเป็นคู่ตำแหน่ง ((x1,y1), (x2,y2))
        (x1, y1), (x2, y2) = wall
        # ถ้า y เท่ากัน = กำแพงแนวตั้ง
        if y1 == y2:
            x_mid = (x1 + x2) / 2.0  # จุดกึ่งกลางแกน x
            # วาดเส้นแนวตั้ง
            _ax.plot([x_mid, x_mid], [y1 - 0.5, y1 + 0.5], color='k', linewidth=4)
        # ถ้า x เท่ากัน = กำแพงแนวนอน
        elif x1 == x2:
            y_mid = (y1 + y2) / 2.0  # จุดกึ่งกลางแกน y
            # วาดเส้นแนวนอน
            _ax.plot([x1 - 0.5, x1 + 0.5], [y_mid, y_mid], color='k', linewidth=4)
    
    # วาดเส้นทางที่เดินมา (path stack) เป็นเส้นสีน้ำเงินพร้อมจุด
    if len(path_stack_to_plot) > 1:
        # แยกพิกัด x และ y ออกมา
        path_x, path_y = zip(*path_stack_to_plot)
        # วาดเส้นทางและจุด
        _ax.plot(path_x, path_y, 'b-o', markersize=4, zorder=1)

    # วาดตำแหน่งหุ่นยนต์ปัจจุบันเป็นจุดสีแดง
    cx, cy = current_cell_to_plot
    _ax.plot(cx, cy, 'ro', markersize=12, label='Robot', zorder=2)

    # ตั้งค่าขอบเขตของกราฟ
    _ax.set_xlim(x_min - 0.5, x_max + 0.5)
    _ax.set_ylim(y_min - 0.5, y_max + 0.5)
    # ตั้งค่าให้กราฟเป็นสี่เหลี่ยมจัตุรัส (aspect ratio 1:1)
    _ax.set_aspect('equal', adjustable='box')
    # เพิ่มเส้น grid สีเทาอ่อน
    _ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)
    # กำหนดตำแหน่ง tick marks
    _ax.set_xticks(np.arange(x_min - 0.5, x_max + 1.5, 1))
    _ax.set_yticks(np.arange(y_min - 0.5, y_max + 1.5, 1))
    # ซ่อน tick labels
    _ax.set_xticklabels([])
    _ax.set_yticklabels([])
    # ตั้งชื่อกราฟ
    _ax.set_title(title)
    # อัปเดตการแสดงผล (pause เพื่อให้กราฟ refresh)
    plt.pause(0.1)

def finalize_show():
    """ปิด interactive mode และแสดงกราฟสุดท้าย"""
    # ปิด interactive mode
    plt.ioff()
    # แสดงกราฟและรอให้ผู้ใช้ปิดหน้าต่าง
    plt.show()

# ===================== Sensor Handling Functions =====================
def sub_tof_handler(sub_info):
    """Callback function สำหรับรับข้อมูลจากเซ็นเซอร์ ToF"""
    global tof_distance_cm
    # แปลงค่าจาก mm เป็น cm (หาร 10)
    tof_distance_cm = sub_info[0] / 10.0

def sub_imu_handler(attitude_info):
    """Callback function สำหรับรับข้อมูลมุม yaw จาก IMU"""
    global current_yaw
    # attitude_info[0] คือมุม yaw (องศา)
    current_yaw = attitude_info[0]

def sub_position_handler(position_info):
    """Callback function สำหรับรับข้อมูลตำแหน่งจาก position sensor"""
    global current_x, current_y
    current_x = position_info[0]  # ตำแหน่ง x (เมตร)
    current_y = position_info[1]  # ตำแหน่ง y (เมตร)

def single_lowpass_filter(new_value, last_value, alpha=0.8):
    """Low-pass filter แบบง่าย สำหรับลด noise ของเซ็นเซอร์
    
    Parameters:
        new_value: ค่าใหม่ที่อ่านได้
        last_value: ค่าเดิมที่ filter แล้ว
        alpha: น้ำหนักของค่าใหม่ (0-1), ยิ่งสูงยิ่งตอบสนองเร็ว
    
    Returns:
        ค่าที่ผ่าน filter แล้ว
    """
    # สูตร: output = alpha * new + (1-alpha) * old
    return alpha * new_value + (1.0 - alpha) * last_value

def adc_to_cm(adc_value, table):
    """แปลงค่า ADC จากเซ็นเซอร์ IR เป็นระยะทาง cm โดยใช้ linear interpolation
    
    Parameters:
        adc_value: ค่า ADC ที่อ่านได้จากเซ็นเซอร์
        table: ตาราง calibration {ADC: cm}
    
    Returns:
        ระยะทางเป็น cm
    """
    # เรียงข้อมูลในตารางจาก ADC มากไปน้อย
    points = sorted(table.items(), key=lambda x: x[0], reverse=True)
    # ถ้า ADC มากกว่าค่าสูงสุด คืนระยะใกล้สุด
    if adc_value >= points[0][0]: return float(points[0][1])
    # ถ้า ADC น้อยกว่าค่าต่ำสุด คืนระยะไกลสุด
    if adc_value <= points[-1][0]: return float(points[-1][1])
    # Linear interpolation ระหว่างจุดที่ใกล้เคียง
    for i in range(len(points) - 1):
        x1, y1 = points[i]      # จุดแรก (ADC สูงกว่า, cm น้อยกว่า)
        x2, y2 = points[i+1]    # จุดถัดไป (ADC ต่ำกว่า, cm มากกว่า)
        # ถ้า ADC อยู่ระหว่าง x2 และ x1
        if x2 <= adc_value <= x1:
            # คำนวณค่าระหว่างจุดด้วยสูตร linear interpolation
            return float(y1 + (adc_value - x1) * (y2 - y1) / (x2 - x1))
    # ถ้าไม่อยู่ในช่วงใดๆ คืนค่า error
    return 999.0

def read_ir_thread(ep_sensor_adaptor):
    """Thread function สำหรับอ่านค่าเซ็นเซอร์ IR อย่างต่อเนื่อง
    
    รันแยกจาก main thread เพื่อไม่ให้รบกวนการทำงานหลัก
    """
    global ir_right_cm, ir_left_cm, last_value_right, last_value_left
    # วนลูปจนกว่าจะมีสัญญาณหยุด
    while not stop_flag:
        # อ่านค่า ADC จากเซ็นเซอร์ IR ขวา (sensor id=2, port=2)
        ir_right_adc = ep_sensor_adaptor.get_adc(id=2, port=2)
        # อ่านค่า ADC จากเซ็นเซอร์ IR ซ้าย (sensor id=1, port=2)
        ir_left_adc = ep_sensor_adaptor.get_adc(id=1, port=2)
        # ผ่าน low-pass filter เพื่อลด noise
        ir_right_filtered = single_lowpass_filter(ir_right_adc, last_value_right)
        ir_left_filtered = single_lowpass_filter(ir_left_adc, last_value_left)
        # บันทึกค่า filter แล้วสำหรับรอบถัดไป
        last_value_right, last_value_left = ir_right_filtered, ir_left_filtered
        # แปลงค่า ADC เป็น cm
        ir_right_cm = adc_to_cm(ir_right_filtered, calibra_table_ir_right)
        ir_left_cm = adc_to_cm(ir_left_filtered, calibra_table_ir_left)
        # รอ 50ms ก่อนอ่านค่าใหม่
        time.sleep(0.05)

# ===================== Movement Functions =====================
def normalize_angle(angle):
    """ปรับมุมให้อยู่ในช่วง -180 ถึง 180 องศา
    
    Parameters:
        angle: มุมองศาที่ต้องการปรับ
    
    Returns:
        มุมที่ปรับแล้ว (-180 < angle <= 180)
    """
    # ถ้ามุมมากกว่า 180 ให้ลบ 360 จนกว่าจะอยู่ในช่วง
    while angle > 180: angle -= 360
    # ถ้ามุมน้อยกว่าหรือเท่ากับ -180 ให้บวก 360
    while angle <= -180: angle += 360
    return angle

def turn_to_angle(ep_chassis, ep_gimbal, target_angle):
    """หมุนหุ่นยนต์ไปยังมุมเป้าหมายโดยใช้ PID control
    
    Parameters:
        ep_chassis: object ควบคุมการเคลื่อนที่
        ep_gimbal: object ควบคุม gimbal
        target_angle: มุมเป้าหมาย (degrees)
    """
    global current_yaw
    # ปรับมุมเป้าหมายให้อยู่ในช่วงมาตรฐาน
    target_angle = normalize_angle(target_angle)
    print(f"กำลังหมุนไปที่ {target_angle}°")
    # วนลูปจนกว่าจะหมุนถึงมุมเป้าหมาย
    while not stop_flag:
        # คำนวณความต่างของมุม (error)
        angle_error = normalize_angle(target_angle - current_yaw)
        # ถ้า error น้อยกว่า 2 องศา ถือว่าถึงเป้าหมายแล้ว
        if abs(angle_error) < 2.0: break
        # คำนวณความเร็วการหมุนด้วย Proportional control
        turn_speed = max(min(angle_error * Kp_turn, TURN_SPEED_Z), -TURN_SPEED_Z)
        # สั่งหมุน (z axis)
        ep_chassis.drive_speed(x=0, y=0, z=turn_speed)
        # รอ 20ms
        time.sleep(0.02)
    # หยุดการหมุน
    ep_chassis.drive_speed(x=0, y=0, z=0)
    # รีเซ็ต gimbal กลับสู่ตำแหน่งกลาง
    ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=100, yaw_speed=100)
    # รอให้หุ่นยนต์หยุดนิ่ง
    time.sleep(0.5)

def move_straight_60cm(ep_chassis, target_yaw):
    """เดินหน้าตรง 60 cm โดยใช้ PID ควบคุมทั้งระยะทางและทิศทาง
    พร้อมหลีกเลี่ยงกำแพงด้านข้าง
    
    Parameters:
        ep_chassis: object ควบคุมการเคลื่อนที่
        target_yaw: มุมทิศทางที่ต้องการรักษา (degrees)
    """
    global ir_left_cm, ir_right_cm, current_yaw, current_x, current_y
    
    # บันทึกตำแหน่งเริ่มต้น
    start_x = current_x
    start_y = current_y
    
    # ค่า PID gains สำหรับแกน X (ระยะทาง)
    Kp_x, Ki_x, Kd_x = 3.0, 0.0, 0.05
    # ค่า PID gains สำหรับแกน Z (ทิศทาง)
    Kp_z, Ki_z, Kd_z = 0.8, 0.02, 0.1
    # เริ่มต้นตัวแปร PID สำหรับแกน X
    integral_x, last_error_x = 0.0, 0.0
    # เริ่มต้นตัวแปร PID สำหรับแกน Z
    integral_z, last_error_z = 0.0, 0.0
    
    # บันทึกเวลาเริ่มต้น
    last_time = time.time()

    print(f"กำลังเคลื่อนที่ไปข้างหน้า {NODE_DISTANCE} m")
    
    # วนลูปจนกว่าจะเดินครบระยะทาง
    while not stop_flag:
        # บันทึกเวลาปัจจุบัน
        current_time = time.time()
        # คำนวณ delta time
        dt = current_time - last_time
        # ถ้า dt ไม่ถูกต้อง ข้ามรอบนี้
        if dt <= 0:
            time.sleep(0.01)
            continue
        
        # คำนวณระยะทางที่เดินมาแล้วจากตำแหน่งเริ่มต้น
        traveled_distance = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
        
        # ตรวจสอบว่าถึงเป้าหมายแล้วหรือยัง (ใช้ tolerance 1 cm)
        if traveled_distance >= NODE_DISTANCE - 0.01:
            break
        
        # --- PID แกน X (คุมระยะทาง) ---
        # error = ระยะทางที่เหลือ
        error_x = NODE_DISTANCE - traveled_distance
        # คำนวณ Integral term
        integral_x += error_x * dt
        # คำนวณ Derivative term
        derivative_x = (error_x - last_error_x) / dt
        # คำนวณความเร็ว X จากสูตร PID และจำกัดไม่ให้เกินค่าสูงสุด
        x_speed = max(min((Kp_x * error_x) + (Ki_x * integral_x) + (Kd_x * derivative_x), MOVE_SPEED_X), -MOVE_SPEED_X)
        
        # --- PID แกน Z (คุมทิศทาง) ---
        # error = มุมที่เบี่ยงเบนจากเป้าหมาย
        error_z = normalize_angle(target_yaw - current_yaw)
        # คำนวณ Integral term
        integral_z += error_z * dt
        # คำนวณ Derivative term
        derivative_z = (error_z - last_error_z) / dt
        # คำนวณความเร็วการหมุนจากสูตร PID
        z_speed = (Kp_z * error_z) + (Ki_z * integral_z) + (Kd_z * derivative_z)
        # ถ้า error น้อยกว่า 2 องศา ไม่ต้องหมุน
        if abs(error_z) < 2.0: z_speed = 0.0
        
        # --- Wall Avoidance (แกน Y) ---
        # เริ่มต้นความเร็วแกน Y เป็น 0
        y_speed = 0.0
        # ถ้ากำแพงขวาใกล้เกินไป ขยับไปซ้าย
        if ir_right_cm < WALL_AVOID_THRESHOLD_CM: y_speed -= WALL_AVOID_SPEED_Y
        # ถ้ากำแพงซ้ายใกล้เกินไป ขยับไปขวา
        if ir_left_cm < WALL_AVOID_THRESHOLD_CM: y_speed += WALL_AVOID_SPEED_Y
        
        # สั่งให้หุ่นยนต์เคลื่อนที่ตามความเร็วที่คำนวณ
        ep_chassis.drive_speed(x=x_speed, y=y_speed, z=z_speed)
        
        # บันทึก error สำหรับรอบถัดไป
        last_error_x, last_error_z = error_x, error_z
        # อัปเดตเวลา
        last_time = current_time
        # รอ 20ms
        time.sleep(0.02)
    
    # หยุดการเคลื่อนที่
    ep_chassis.drive_speed(x=0, y=0, z=0)
    # รอให้หุ่นยนต์หยุดนิ่ง
    time.sleep(0.5)
    print(f"เคลื่อนที่สำเร็จ: ระยะทาง {traveled_distance:.3f} m")

# ===================== DFS Logic Functions =====================
def scan_environment():
    """สแกนสภาพแวดล้อมรอบๆ หุ่นยนต์เพื่อหาทิศทางที่เดินได้
    
    Returns:
        dict: {'front': bool, 'left': bool, 'right': bool}
              True = เดินได้, False = มีกำแพง
    """
    global tof_distance_cm, ir_left_cm, ir_right_cm
    # เตรียม dictionary สำหรับเก็บผลสแกน
    open_paths = {'front': False, 'left': False, 'right': False}
    # รอให้เซ็นเซอร์อัปเดตค่า
    time.sleep(SCAN_DURATION_S)
    # ตรวจสอบด้านหน้าด้วย ToF sensor
    if tof_distance_cm > TOF_WALL_THRESHOLD_CM: open_paths['front'] = True
    # ตรวจสอบด้านซ้ายด้วย IR sensor
    if ir_left_cm > IR_WALL_THRESHOLD_CM: open_paths['left'] = True
    # ตรวจสอบด้านขวาด้วย IR sensor
    if ir_right_cm > IR_WALL_THRESHOLD_CM: open_paths['right'] = True
    # แสดงผลการสแกน
    print(f"ผลสแกน: หน้า: {tof_distance_cm:.1f} cm | ซ้าย: {ir_left_cm:.1f} cm | ขวา: {ir_right_cm:.1f} cm")
    return open_paths

def get_target_coordinates(from_pos, heading_deg):
    """คำนวณพิกัดเป้าหมายเมื่อเดินไปในทิศทางที่กำหนด
    
    Parameters:
        from_pos: ตำแหน่งเริ่มต้น (x, y)
        heading_deg: ทิศทางที่ต้องการเดิน (degrees)
    
    Returns:
        tuple: พิกัดเป้าหมาย (x, y)
    """
    # แยกพิกัด x, y
    x, y = from_pos
    # ปรับมุมให้อยู่ในช่วงมาตรฐาน
    heading = normalize_angle(heading_deg)
    # ตรวจสอบทิศทางและคำนวณตำแหน่งใหม่
    if heading == 0: return (x, y + 1)          # เหนือ: y+1
    elif heading == 90: return (x + 1, y)       # ตะวันออก: x+1
    elif heading == -90: return (x - 1, y)      # ตะวันตก: x-1
    elif abs(heading) == 180: return (x, y - 1) # ใต้: y-1
    # ถ้าไม่ตรงกรณีใดๆ คืนตำแหน่งเดิม
    return from_pos

def get_direction_to_neighbor(from_cell, to_cell):
    """คำนวณมุมทิศทางจากช่องหนึ่งไปยังอีกช่องหนึ่ง
    
    Parameters:
        from_cell: ตำแหน่งต้นทาง (x, y)
        to_cell: ตำแหน่งปลายทาง (x, y)
    
    Returns:
        float: มุมทิศทาง (degrees) ที่ normalize แล้ว
    """
    # คำนวณผลต่างของพิกัด
    dx = to_cell[0] - from_cell[0]  # ความต่างแกน x
    dy = to_cell[1] - from_cell[1]  # ความต่างแกน y
    # ใช้ atan2 คำนวณมุมและแปลงเป็นองศา แล้ว normalize
    return normalize_angle(math.degrees(math.atan2(dx, dy)))

def turn_and_move(ep_chassis, ep_gimbal, target_heading):
    """หมุนไปทิศทางเป้าหมาย (ถ้าจำเป็น) แล้วเดินหน้าไป 60 cm
    
    Parameters:
        ep_chassis: object ควบคุมการเคลื่อนที่
        ep_gimbal: object ควบคุม gimbal
        target_heading: ทิศทางเป้าหมาย (degrees)
    """
    global current_heading_degrees
    # ตรวจสอบว่าต้องหมุนหรือไม่ (ถ้าต่างมากกว่า 1 องศา)
    if abs(normalize_angle(target_heading - current_heading_degrees)) > 1:
        # หมุนไปยังทิศทางเป้าหมาย
        turn_to_angle(ep_chassis, ep_gimbal, target_heading)
        # อัปเดตทิศทางปัจจุบัน
        current_heading_degrees = target_heading
    # เดินหน้าไป 60 cm
    move_straight_60cm(ep_chassis, target_heading)

def map_current_cell():
    """สแกนและบันทึกข้อมูลแผนที่ของช่องปัจจุบัน
    
    ทำการสแกนทิศทางที่เปิด และบันทึกกำแพงที่พบ
    """
    global maze_map, walls, current_pos, current_heading_degrees
    print(f"ช่อง {current_pos} ยังไม่ได้สำรวจ กำลังสแกน...")
    # สแกนสภาพแวดล้อม
    scan_results = scan_environment()
    # สร้าง set สำหรับเก็บทิศทางที่เปิด (absolute heading)
    open_headings = set()
    # กำหนดมุมสัมพัทธ์สำหรับแต่ละทิศทาง (relative to current heading)
    relative_moves = {'left': -90, 'front': 0, 'right': 90}
    
    # วนลูปตรวจสอบแต่ละทิศทาง
    for move_key, is_open in scan_results.items():
        # คำนวณมุม relative (เทียบกับทิศทางปัจจุบัน)
        relative_angle = relative_moves[move_key]
        # แปลงเป็นมุม absolute (เทียบกับทิศเหนือ)
        absolute_heading = normalize_angle(current_heading_degrees + relative_angle)
        # ถ้าทิศทางนี้เปิดอยู่
        if is_open:
            # เพิ่มเข้า set ของทิศทางที่เปิด
            open_headings.add(absolute_heading)
        else:
            # ถ้าปิด = มีกำแพง
            # คำนวณตำแหน่งของช่องเพื่อนบ้านในทิศทางนั้น
            neighbor_cell = get_target_coordinates(current_pos, absolute_heading)
            # เพิ่มกำแพงระหว่างช่องปัจจุบันกับช่องเพื่อนบ้าน
            # ใช้ sorted เพื่อให้กำแพง (A,B) และ (B,A) เป็นอันเดียวกัน
            walls.add(tuple(sorted((current_pos, neighbor_cell))))
    
    # บันทึกข้อมูลแผนที่ของช่องนี้
    maze_map[current_pos] = open_headings
    print(f"สร้างแผนที่ช่อง {current_pos} มีทิศทางที่เปิด: {sorted(list(open_headings))}")

def find_and_move_to_next_cell(ep_chassis, ep_gimbal):
    """หาช่องเพื่อนบ้านที่ยังไม่เคยไปและเคลื่อนที่ไปยังช่องนั้น
    
    ตามหลักการ DFS: ตรวจสอบตามลำดับ ซ้าย -> หน้า -> ขวา
    
    Parameters:
        ep_chassis: object ควบคุมการเคลื่อนที่
        ep_gimbal: object ควบคุม gimbal
    
    Returns:
        bool: True = เคลื่อนที่สำเร็จ, False = ไม่มีทางไปต่อ
    """
    global visited_nodes, path_stack, current_pos, current_heading_degrees
    # ลำดับการตรวจสอบ: ซ้าย (-90°), หน้า (0°), ขวา (90°)
    search_order_relative = [-90, 0, 90]
    
    # วนลูปตรวจสอบแต่ละทิศทางตามลำดับ DFS
    for angle in search_order_relative:
        # คำนวณทิศทางเป้าหมาย (absolute heading)
        target_heading = normalize_angle(current_heading_degrees + angle)
        # ตรวจสอบว่าทิศทางนี้เปิดอยู่หรือไม่ (อยู่ใน maze_map)
        if target_heading in maze_map.get(current_pos, set()):
            # คำนวณตำแหน่งของช่องเพื่อนบ้าน
            target_cell = get_target_coordinates(current_pos, target_heading)
            
            # ตรวจสอบว่าช่องเป้าหมายอยู่ในขอบเขตของแผนที่หรือไม่
            min_x, min_y = MAP_MIN_BOUNDS
            max_x, max_y = MAP_MAX_BOUNDS
            # ถ้าอยู่นอกขอบเขต ข้ามทิศทางนี้
            if not (min_x <= target_cell[0] <= max_x and min_y <= target_cell[1] <= max_y):
                continue

            # ตรวจสอบว่าช่องนี้เคยไปแล้วหรือยัง
            if target_cell not in visited_nodes:
                print(f"พบเพื่อนบ้านที่ยังไม่เคยไป {target_cell} กำลังเคลื่อนที่...")
                # หมุนและเคลื่อนที่ไปยังช่องเป้าหมาย
                turn_and_move(ep_chassis, ep_gimbal, target_heading)
                
                # เพิ่มช่องนี้เข้าไปใน visited_nodes
                visited_nodes.add(target_cell)
                # เพิ่มช่องนี้เข้า path stack
                path_stack.append(target_cell)
                # อัปเดตตำแหน่งปัจจุบัน
                current_pos = target_cell
                # คืนค่า True = เคลื่อนที่สำเร็จ
                return True
    # ถ้าไม่มีทิศทางไหนที่ไปได้ คืนค่า False
    return False

def backtrack(ep_chassis, ep_gimbal):
    """ย้อนรอยกลับไปยังช่องก่อนหน้าใน path stack
    
    ใช้เมื่อเจอทางตันและต้อง backtrack ตาม DFS
    
    Parameters:
        ep_chassis: object ควบคุมการเคลื่อนที่
        ep_gimbal: object ควบคุม gimbal
    
    Returns:
        bool: True = ย้อนรอยสำเร็จ, False = กลับถึงจุดเริ่มต้นแล้ว
    """
    global path_stack, current_pos
    print("เจอทางตัน กำลังย้อนรอย (Backtracking)...")
    # ถ้า stack มีแค่จุดเดียว = กลับมาจุดเริ่มต้นแล้ว
    if len(path_stack) <= 1:
        print("กลับมาที่จุดเริ่มต้น การสำรวจสิ้นสุด")
        return False

    # ลบตำแหน่งปัจจุบันออกจาก stack
    path_stack.pop()
    # ดึงตำแหน่งก่อนหน้า (จุดที่จะย้อนรอยไป)
    previous_cell = path_stack[-1]
    
    # คำนวณทิศทางที่ต้องหมุนเพื่อย้อนรอย
    backtrack_heading = get_direction_to_neighbor(current_pos, previous_cell)
    print(f"กำลังย้อนรอยจาก {current_pos} ไปยัง {previous_cell}")
    # หมุนและเคลื่อนที่ย้อนกลับ
    turn_and_move(ep_chassis, ep_gimbal, backtrack_heading)
    # อัปเดตตำแหน่งปัจจุบัน
    current_pos = previous_cell
    # คืนค่า True = ย้อนรอยสำเร็จ
    return True

# ===================== Main Execution Block =====================
if __name__ == '__main__':
    # สร้าง object หุ่นยนต์
    ep_robot = robot.Robot()
    # เชื่อมต่อกับหุ่นยนต์ผ่าน WiFi Access Point
    ep_robot.initialize(conn_type="ap")

    # ดึง object ย่อยสำหรับควบคุมแต่ละส่วน
    ep_chassis = ep_robot.chassis        # ควบคุมการเคลื่อนที่
    ep_sensor = ep_robot.sensor          # เซ็นเซอร์หลัก (ToF, IMU)
    ep_gimbal = ep_robot.gimbal          # ควบคุม gimbal
    ep_sensor_adaptor = ep_robot.sensor_adaptor  # อ่านค่า ADC จาก IR

    # ตั้งชื่อหน้าต่างกราฟ
    _fig.canvas.manager.set_window_title("Maze Map")

    # Subscribe และ เริ่ม Thread
    # Subscribe เซ็นเซอร์ ToF ความถี่ 20 Hz
    ep_sensor.sub_distance(freq=20, callback=sub_tof_handler)
    # Subscribe ข้อมูล attitude (yaw) จาก IMU ความถี่ 20 Hz
    ep_chassis.sub_attitude(freq=20, callback=sub_imu_handler)

    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    # สร้าง thread สำหรับอ่านค่า IR อย่างต่อเนื่อง (daemon=True จะปิดตาม main)
    ir_reader = threading.Thread(target=read_ir_thread, args=(ep_sensor_adaptor,), daemon=True)
    # เริ่มรัน thread
    ir_reader.start()
    # รอ 1 วินาทีให้ทุกอย่างเริ่มต้นเสร็จ
    time.sleep(1)

    # --- Initialize DFS State ---
    # ตั้งตำแหน่งเริ่มต้น
    current_pos = START_CELL
    # เพิ่มตำแหน่งเริ่มต้นเข้า visited_nodes
    visited_nodes.add(current_pos)
    # เพิ่มตำแหน่งเริ่มต้นเข้า path_stack
    path_stack.append(current_pos)
    
    print("เริ่มต้นการสำรวจเขาวงกตแบบ DFS...")
    
    # --- Main Exploration Loop ---
    # วนลูปจนกว่า path_stack จะว่าง (สำรวจเสร็จแล้ว) หรือมีคำสั่งหยุด
    while path_stack and not stop_flag:
        # ตรวจสอบว่ามีการกดปุ่ม ESC หรือไม่
        if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
            print("กดปุ่ม ESC กำลังหยุดการทำงาน...")
            break

        # วาดแผนที่ปัจจุบัน
        plot_maze(walls, visited_nodes, path_stack, current_pos)
        # แสดงสถานะปัจจุบัน
        print(f"\nตำแหน่ง: {current_pos}, ทิศทาง: {current_heading_degrees}°")

        # ถ้าช่องปัจจุบันยังไม่ได้สร้างแผนที่
        if current_pos not in maze_map:
            # สแกนและบันทึกแผนที่ช่องนี้
            map_current_cell()
        
        # พยายามหาและเคลื่อนที่ไปยังช่องถัดไป
        if find_and_move_to_next_cell(ep_chassis, ep_gimbal):
            # ถ้าเคลื่อนที่สำเร็จ วนลูปต่อ
            continue
        
        # ถ้าไม่มีทางไปต่อ ทำการ backtrack
        if not backtrack(ep_chassis, ep_gimbal):
            # ถ้า backtrack ไม่ได้ (กลับถึงจุดเริ่มต้นแล้ว) ออกจากลูป
            break

    # แสดงข้อความเสร็จสิ้น
    print("\nการสำรวจ DFS เสร็จสมบูรณ์")
    # วาดแผนที่สุดท้าย
    plot_maze(walls, visited_nodes, path_stack, current_pos, "Final Map")

    print("กำลังทำความสะอาดและปิดการเชื่อมต่อ...")
    # ตั้งค่า flag เพื่อหยุด thread อ่าน IR
    stop_flag = True
    # รอให้ thread ปิด
    time.sleep(0.2)
    # หยุดการเคลื่อนที่
    ep_chassis.drive_speed(x=0, y=0, z=0)
    # ยกเลิก subscription เซ็นเซอร์ ToF
    ep_sensor.unsub_distance()
    # ยกเลิก subscription attitude (IMU)
    ep_chassis.unsub_attitude()

    ep_chassis.unsub_position()
    # ปิดการเชื่อมต่อกับหุ่นยนต์
    ep_robot.close()
    # แสดงกราฟสุดท้ายและรอให้ผู้ใช้ปิด
    finalize_show()