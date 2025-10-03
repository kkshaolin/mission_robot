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

START_CELL = (1, 1)              #  จุดเริ่ม สามารถเปลี่ยนเป็นพิกัดอื่นได้ เช่น (1, 1) หรือ (-2, -2)
MAP_MIN_BOUNDS = (1, 1)          #  พิกัด (min_x, min_y) ของแผนที่ (มุมซ้ายล่าง)
MAP_MAX_BOUNDS = (3, 3)          #  พิกัด (max_x, max_y) ของแผนที่ (มุมขวาบน)
NODE_DISTANCE = 0.6              # โหนดในเขาวงกต (60 cm)


WALL_AVOID_THRESHOLD_CM = 10.0  # ถ้า IR วัดได้น้อยกว่านี้ = ใกล้กำแพงเกินไป ต้องขยับหนี
WALL_AVOID_SPEED_Y = 0.05    # ความเร็วในการขยับหนีกำแพง (m/s) ในแนวแกน Y (ซ้าย-ขวา)
MOVE_SPEED_X = 1  # ความเร็วในการเดินหน้า (m/s)
TURN_SPEED_Z = 60  # ความเร็วสูงสุดในการหมุน (degrees/s)

# --------------------------------------------------------

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
current_pos = START_CELL  # ตำแหน่งปัจจุบันในระบบพิกัด (x, y)
current_heading_degrees = 0  # ทิศทางที่หุ่นยนต์หันไป (0=เหนือ, 90=ตะวันออก, -90=ตะวันตก, 180=ใต้)
walls = {} # เก็บข้อมูลกำแพงที่ตรวจพบ {(cell1, cell2): 'Occupied'/'Free'}


# WALL_THRESHOLD = 50
# CELL_SIZE = 0.60
"""-----------------------map-------------"""
def plot_maze(walls_to_plot, cell_to_plot, visited_to_plot, title="Maze Exploration"):
    _ax.clear()
    MAZE_BOUNDS_PLOT = (0, 5, 0, 5) # สามารถปรับขนาดตามแผนที่จริงได้
    x_min, x_max = MAZE_BOUNDS_PLOT[0]-1, MAZE_BOUNDS_PLOT[1]+1
    y_min, y_max = MAZE_BOUNDS_PLOT[2]-1, MAZE_BOUNDS_PLOT[3]+1
    for x, y in visited_to_plot:
        _ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightcyan', edgecolor='none', zorder=0))
    for wall in walls_to_plot.keys():
        (x1, y1), (x2, y2) = wall
        if y1 == y2: # Vertical wall
            x_mid = (x1 + x2) / 2.0
            _ax.plot([x_mid, x_mid], [y1 - 0.5, y1 + 0.5], color='k', linewidth=4)
        elif x1 == x2: # Horizontal wall
            y_mid = (y1 + y2) / 2.0
            _ax.plot([x1 - 0.5, x1 + 0.5], [y_mid, y_mid], color='k', linewidth=4)
    cx, cy = cell_to_plot
    _ax.plot(cx, cy, 'bo', markersize=15, label='Robot', zorder=2)
    _ax.set_xlim(x_min - 0.5, x_max + 0.5); _ax.set_ylim(y_min - 0.5, y_max + 0.5)
    _ax.set_aspect('equal', adjustable='box'); _ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)
    _ax.set_xticks(np.arange(x_min - 0.5, x_max + 1.5, 1)); _ax.set_yticks(np.arange(y_min - 0.5, y_max + 1.5, 1))
    _ax.set_xticklabels([]); _ax.set_yticklabels([])
    _ax.set_title(title)

def finalize_show():
    plt.ioff() # ปิดโหมด Interactive
    plt.show() # แสดงผลแบบค้างไว้

def _get_discretized_orientation(yaw_deg):
    """แปลงมุมองศาเป็นทิศทางแบบตัวเลข (0:N, 1:E, 2:S, 3:W)"""
    # 90 คือ East, -90 คือ West
    if -45 <= yaw_deg < 45: return 0      # North (หันหน้าไปทาง +y)
    elif 45 <= yaw_deg < 135: return 1     # East (หันหน้าไปทาง +x)
    elif abs(yaw_deg) >= 135: return 2   # South (หันหน้าไปทาง -y)
    elif -135 < yaw_deg < -45: return 3   # West (หันหน้าไปทาง -x)

def update_map_and_walls(cell, orientation, scan_results, current_walls):
    """
    อัปเดตข้อมูลกำแพง (walls) จากผลการสแกนล่าสุด
    """
    updated_walls = current_walls.copy()
    
    # แปลงทิศทางของหุ่นยนต์ (0-3) ไปเป็นทิศของกำแพง (L, F, R)
    # 0:N -> L=W(3), F=N(0), R=E(1)
    orientation_map = {0:{"left":3,"front":0,"right":1}, 1:{"left":0,"front":1,"right":2}, 2:{"left":1,"front":2,"right":3}, 3:{"left":2,"front":3,"right":0}}
    
    # แปลงทิศทาง (0-3) เป็นการเปลี่ยนแปลงของแกน (dx, dy)
    coord_map = {0:(0,1), 1:(1,0), 2:(0,-1), 3:(-1,0)}
    
    # ตรวจสอบผลสแกนแต่ละทิศทาง
    for move_key in ["left", "front", "right"]:
        # ถ้าผลสแกนคือ False แปลว่ามีกำแพง
        if not scan_results.get(move_key, True):
            direction = orientation_map[orientation][move_key]
            dx, dy = coord_map[direction]
            neighbor_cell = (cell[0] + dx, cell[1] + dy)
            
            # สร้าง key สำหรับ dictionary ของกำแพง โดยเรียงลำดับ tuple เสมอ
            wall_coords = tuple(sorted((cell, neighbor_cell)))
            updated_walls[wall_coords] = 'Wall'
            
    return updated_walls
"""------------------------end map-------------------------"""

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
       ใช้ MAP_MIN_BOUNDS และ MAP_MAX_BOUNDS เพื่อป้องกันการออกนอกแผนที่"""
    global MAP_MIN_BOUNDS, MAP_MAX_BOUNDS  # เรียกใช้ค่าคงที่ตัวใหม่
    min_x, min_y = MAP_MIN_BOUNDS
    max_x, max_y = MAP_MAX_BOUNDS
    
    possible_moves = []
    check_order = ['left', 'front', 'right']

    for direction in check_order:
        if scan_results.get(direction, False):
            pos, _ = get_new_pos_and_heading(direction, current_pos, current_heading)
            x, y = pos

            # ตรวจสอบว่าอยู่ในขอบเขตแผนที่หรือไม่ (ใช้ min และ max)
            if min_x <= x <= max_x and min_y <= y <= max_y:
                if pos not in visited_nodes:
                    possible_moves.append(direction)

    if possible_moves:
        return possible_moves[0]
    return None


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
    """หมุนหุ่นยนต์ไปยังมุมที่ต้องการโดยใช้ PID แบบ Proportional"""
    global current_yaw
    
    # Normalize มุมเป้าหมาย
    target_angle = normalize_angle(target_angle)
    # print(f"หมุน{target_angle}°")
    
    while not stop_flag:    # Loop จนกว่าจะหมุนถึงมุมเป้าหมาย (ยอมรับ error ±2°)
        # คำนวณความแตกต่างของมุม (error)
        angle_error = normalize_angle(target_angle - current_yaw)
        
        if abs(angle_error) < 2.0:  # ถ้า error น้อยกว่า 2° ถือว่าหมุนสำเร็จแล้ว
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

# --- อัปเดต: ฟังก์ชันเดินตรง ให้ใช้ตัวแปร ir_left_cm, ir_right_cm ---
def move_straight_60cm(ep_chassis, target_yaw):
    """
    เดินหน้าตรงไป 60 cm โดยใช้ PID ควบคุม:
      - แกน X: ระยะทาง (PID คุมให้เดิน 60 cm)
      - แกน Z: รักษาทิศทาง (PID คุม yaw)
      - แกน Y: หลีกเลี่ยงกำแพง (Rule-based)

    Parameters:
        ep_chassis: object ควบคุมการเคลื่อนที่
        target_yaw: มุม yaw เป้าหมาย (degrees)
    """
    global ir_left_cm, ir_right_cm, current_yaw, stop_flag

    # ===================== PID Parameters =====================
    # แกน X (เดินไปข้างหน้า)
    Kp_x, Ki_x, Kd_x = 2.0, 0.0, 0.05
    tolerance_x = 0.1      # error ที่ยอมรับได้ (m) ≈ 2 cm

    # แกน Z (ทิศทาง yaw)
    Kp_z, Ki_z, Kd_z = 0.8, 0.02, 0.1
    tolerance_z = 2.0       # error ที่ยอมรับได้ (°)

    # Wall avoidance (แกน Y)
    avoid_speed_y = WALL_AVOID_SPEED_Y
    avoid_threshold_cm = WALL_AVOID_THRESHOLD_CM

    # ===================== Initial Values =====================
    integral_x, last_error_x = 0.0, 0.0
    integral_z, last_error_z = 0.0, 0.0
    traveled_distance = 0.0

    start_time = time.time()
    last_time = start_time

    # ===================== Main Loop =====================
    while traveled_distance < NODE_DISTANCE - tolerance_x and not stop_flag:
        current_time = time.time()
        dt = current_time - last_time
        if dt <= 0:
            time.sleep(0.01)
            continue

        # --- PID แกน X (คุมระยะทาง) ---
        error_x = NODE_DISTANCE - traveled_distance
        integral_x += error_x * dt
        derivative_x = (error_x - last_error_x) / dt
        x_speed = (Kp_x * error_x) + (Ki_x * integral_x) + (Kd_x * derivative_x)
        x_speed = max(min(x_speed, MOVE_SPEED_X), -MOVE_SPEED_X)

        # --- PID แกน Z (คุม yaw) ---
        error_z = normalize_angle(target_yaw - current_yaw)
        integral_z += error_z * dt
        derivative_z = (error_z - last_error_z) / dt
        z_speed = (Kp_z * error_z) + (Ki_z * integral_z) + (Kd_z * derivative_z)

        # ถ้า error_z อยู่ใน tolerance → ไม่ต้องหมุน
        if abs(error_z) < tolerance_z:
            z_speed = 0.0

        # --- Wall Avoidance (แกน Y) ---
        y_speed = 0.0
        if ir_right_cm < avoid_threshold_cm:
            y_speed -= avoid_speed_y
        if ir_left_cm < avoid_threshold_cm:
            y_speed += avoid_speed_y

        # --- ส่งคำสั่งเคลื่อนที่ ---
        ep_chassis.drive_speed(x=x_speed, y=y_speed, z=z_speed)

        # --- Update States ---
        traveled_distance += abs(x_speed) * dt
        last_error_x, last_error_z = error_x, error_z
        last_time = current_time
        time.sleep(0.02)

    # ===================== Stop Motion =====================
    ep_chassis.drive_speed(x=0, y=0, z=0)
    time.sleep(0.5)
    print(f"เคลื่อนที่ 60 cm สำเร็จ (error_x ≤ {tolerance_x*100:.1f} cm, error_z ≤ {tolerance_z:.1f}°)")



def backtrack(ep_chassis, ep_gimbal):
    global current_pos, current_heading_degrees
    print("Backtracking...")
    # pop until we reach a cell that has unexplored neighbour or stack empty
    if path_stack and not stop_flag:
        last_pos, last_heading = path_stack.pop()
        # compute heading from current_pos to last_pos
        target_x, target_y = last_pos
        current_x, current_y = current_pos
        
        # คำนวณมุมที่ต้องหันกลับไปหาโหนดก่อนหน้า
        # atan2(delta_x, delta_y) เพราะ 0 องศาคือแกน Y+ (เหนือ)
        backtrack_heading = math.degrees(math.atan2(target_x - current_x, target_y - current_y))
        backtrack_heading = normalize_angle(backtrack_heading)
        
        print(f"Backtrack: going from {current_pos} to {last_pos} with heading {backtrack_heading:.1f}")
        
        # หันและเคลื่อนที่กลับ
        turn_to_angle(ep_chassis, ep_gimbal, backtrack_heading)
        move_straight_60cm(ep_chassis, backtrack_heading)
        
        # อัปเดตตำแหน่งและทิศทางปัจจุบัน
        current_pos = last_pos
        current_heading_degrees = last_heading # กลับไปใช้ทิศทางเดิม ณ โหนดนั้น
        
        # คืนค่า True เพื่อบอกว่า backtrack สำเร็จ
        return True
    
    # ถ้า path_stack ว่างเปล่า คืนค่า False
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

    _fig, _ax = plt.subplots(figsize=(6, 6))
    _fig.canvas.manager.set_window_title("Maze Map")

    # --- thread การอ่านค่า IR ---
    ir_reader = threading.Thread(target=read_ir_thread, args=(ep_sensor_adaptor,),daemon=True)
    ir_reader.start()
    time.sleep(0.5)

    visited_nodes.add(current_pos)  # เพิ่มตำแหน่งเริ่มต้นเข้า
    print(f"Starting DFS at {current_pos}")

    # --- Main Loop ที่แก้ไขแล้ว ---
    while not stop_flag:
        try:
            # กดปุ่ม ESC เพื่อหยุดโปรแกรม
            if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
                print("ESC pressed. Stopping exploration.")
                stop_flag = True
                break
            
            print(f"\n--- Current Node: {current_pos}, Heading: {current_heading_degrees}° ---")
            visited_nodes.add(current_pos) # ตรวจสอบให้แน่ใจว่าโหนดปัจจุบันถูกเยี่ยมแล้ว
            
            # reset gimbal ให้หันไปข้างหน้าก่อนสแกน
            ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=100, yaw_speed=150).wait_for_completed()
            
            # สแกนสภาพแวดล้อมรอบตัว
            scan_results = scan_environment()
            print(f"Scan results: {scan_results}")

            # อัปเดตและวาดแผนที่
            try:
                discrete_orientation = _get_discretized_orientation(current_heading_degrees)
                walls = update_map_and_walls(current_pos, discrete_orientation, scan_results, walls)
                plot_maze(walls, current_pos, visited_nodes, "Real-time Maze Exploration")
                plt.pause(0.01)
            except Exception as e:
                print(f"Error plotting map: {e}")

            # ตัดสินใจเลือกเส้นทางถัดไป
            chosen_direction = decide_by_dfs(scan_results, current_pos, current_heading_degrees)

            if chosen_direction:
                # === State: สำรวจโหนดใหม่ ===
                print(f"Found new path: {chosen_direction}. Moving forward.")
                
                # เก็บสถานะปัจจุบันลง stack สำหรับการ backtrack
                path_stack.append((current_pos, current_heading_degrees))
                
                # คำนวณตำแหน่งและทิศทางใหม่
                new_pos, new_heading = get_new_pos_and_heading(chosen_direction, current_pos, current_heading_degrees)
                
                # หันไปยังทิศทางใหม่
                turn_to_angle(ep_chassis, ep_gimbal, new_heading)
                current_heading_degrees = new_heading
                
                # เดินไปยังโหนดใหม่
                move_straight_60cm(ep_chassis, current_heading_degrees)
                
                # อัปเดตสถานะปัจจุบัน
                current_pos = new_pos
                print(f"Arrived at new node: {current_pos}")

            else:
                # === State: ทางตัน, เริ่ม Backtracking ===
                print("No new paths. Attempting to backtrack.")
                
                # เรียกฟังก์ชัน backtrack
                backtracked_successfully = backtrack(ep_chassis, ep_gimbal)
                
                if not backtracked_successfully:
                    # ถ้า backtrack ไม่สำเร็จ แสดงว่า stack ว่าง และการสำรวจเสร็จสิ้น
                    print("Exploration complete. No more paths to backtrack to.")
                    stop_flag = True
                    break
                else:
                    # หลังจาก backtrack 1 ก้าว, loop จะวนกลับไปสแกนอีกครั้ง
                    # เพื่อหาเส้นทางที่ยังไม่ได้ไปจากโหนดก่อนหน้านี้
                    print(f"Backtracked to {current_pos}. Rescanning...")

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping...")
            stop_flag = True
            break
            
        # หน่วงเวลาเล็กน้อยเพื่อป้องกันการส่งคำสั่งที่เร็วเกินไป
        time.sleep(0.1)

    # --- ส่วนของการปิดโปรแกรม ---
    print("Cleaning up and shutting down...")
    stop_flag = True  # ส่งสัญญาณให้ IR thread หยุดทำงาน
    if ir_reader.is_alive():
        ir_reader.join(timeout=1)  # รอให้ thread มีเวลาหยุด
        
    ep_chassis.drive_speed(x=0, y=0, z=0)
    ep_sensor.unsub_distance() 
    ep_chassis.unsub_attitude()  
    ep_robot.close()
    
    # แสดงแผนที่ฉบับสมบูรณ์
    print("Displaying final map.")
    plot_maze(walls, current_pos, visited_nodes, "Final Exploration Map")
    # _fig.savefig("final_maze_map.png", dpi=300) # หากต้องการบันทึกเป็นไฟล์
    finalize_show()