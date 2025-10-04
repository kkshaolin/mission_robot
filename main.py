from robomaster import robot  # ไลบรารีสำหรับควบคุมหุ่นยนต์ RoboMaster
import msvcrt  # สำหรับตรวจจับการกดปุ่ม ESC เพื่อหยุดโปรแกรม (Windows)
import time  # สำหรับจัดการเวลาและ delay
import math  # สำหรับคำนวณมุมและตำแหน่ง (atan2, degrees)
import threading  # สำหรับรันการอ่านค่า IR แบบ concurrent (พร้อมกัน)
import matplotlib.pyplot as plt  # สำหรับวาดกราฟแผนที่เขาวงกต
# plt.switch_backend('Agg') # <<< เพิ่มบรรทัดนี้: เพื่อไม่ให้แสดงหน้าต่างกราฟ
import numpy as np  # สำหรับจัดการอาร์เรย์และช่วงตัวเลข
import cv2 # <<< เพิ่มบรรทัดนี้เข้ามา

# ===================== Global State & Constants =====================
# ตัวแปร flag สำหรับหยุดโปรแกรมทั้งหมด (ใช้ร่วมกันทุก thread)
stop_flag = False

# --- ตัวแปรสถานะส่วนกลาง (Global State Variables) ---
tof_distance_cm = 999.0 # ระยะทางที่วัดได้จากเซ็นเซอร์ ToF (Time of Flight) หน่วยเป็น cm
current_yaw = 0.0 # มุม yaw ปัจจุบันของหุ่นยนต์ (จาก IMU) หน่วยเป็นองศา
current_x = 0.0  # ตำแหน่ง x จาก position sensor (เมตร)
current_y = 0.0  # ตำแหน่ง y จาก position sensor (เมตร)
ir_left_cm = 999.0 # ระยะทางจากกำแพงด้านซ้าย (จากเซ็นเซอร์ IR) หน่วยเป็น cm
ir_right_cm = 999.0 # ระยะทางจากกำแพงด้านขวา (จากเซ็นเซอร์ IR) หน่วยเป็น cm
last_value_left = 0 # ค่าล่าสุดของเซ็นเซอร์ IR (ใช้สำหรับ low-pass filter)
last_value_right = 0
FRONT_SAFETY_CM = 12.0

# --- Maze State Variables (จัดการสถานะของ DFS) ---
maze_map = {} # เก็บข้อมูลแผนที่: key=ตำแหน่ง(x,y), value=set ของทิศทางที่เปิด (degrees)
visited_nodes = set() # เก็บตำแหน่งทั้งหมดที่เคยไปแล้ว
path_stack = [] # stack สำหรับเก็บเส้นทางการเดินตาม DFS
walls = set() # เก็บพิกัดของกำแพงทั้งหมด (tuple ของคู่ตำแหน่ง)
current_pos = (1, 1) # ตำแหน่งปัจจุบันของหุ่นยนต์ในเขาวงกต (x, y)
current_heading_degrees = 0 # ทิศทางหัวหุ่นยนต์ปัจจุบัน: 0=เหนือ, 90=ตะวันออก, -90=ตะวันตก, 180=ใต้
markers_found = {} # <<< ใหม่: ตัวแปรสำหรับเก็บข้อมูล Marker ที่เจอ --- รูปแบบ: {(x, y): {'color': 'red', 'shape': 'square'}}

# --- ค่าคงที่สำหรับเขาวงกตและการเคลื่อนที่ ---
SCAN_DURATION_S = 0.2 # ระยะเวลาในการสแกนสภาพแวดล้อม (วินาที)
TOF_WALL_THRESHOLD_CM = 60 # ระยะทาง ToF ที่ถือว่ามีกำแพง (cm)
IR_WALL_THRESHOLD_CM = 29 # ระยะทาง IR ที่ถือว่ามีกำแพงด้านข้าง (cm)
START_CELL = (1, 1) # ตำแหน่งเริ่มต้นในเขาวงกต
MAP_MIN_BOUNDS = (1, 1) # ขอบเขตขั้นต่ำของแผนที่ (x_min, y_min)
MAP_MAX_BOUNDS = (3, 3) # ขอบเขตสูงสุดของแผนที่ (x_max, y_max)
NODE_DISTANCE = 0.6 # ระยะทางระหว่างช่องในเขาวงกต (เมตร)
WALL_AVOID_THRESHOLD_CM = 10.0 # ระยะที่เริ่มหลีกเลี่ยงกำแพงด้านข้าง (cm)
WALL_AVOID_SPEED_Y = 0.05 # ความเร็วในการขยับหลีกเลี่ยงกำแพง (m/s)
MOVE_SPEED_X = 2 # ความเร็วการเดินหน้าสูงสุด (m/s)
TURN_SPEED_Z = 60 # ความเร็วการหมุนสูงสุด (degrees/s)

# --- PID Controller Gains ---
Kp_turn = 2.5 # ค่า Proportional gain สำหรับการหมุน

# --- ตารางเทียบค่า ADC เป็น CM สำหรับ IR Sensors ---
calibra_table_ir_right = {615: 5, 605: 10, 415: 15, 335: 20, 275: 25, 255: 30} # ตาราง calibration สำหรับเซ็นเซอร์ IR ด้านขวา
calibra_table_ir_left = {680: 5, 420: 10, 300: 15, 235: 20, 210: 25, 175: 30} # ตาราง calibration สำหรับเซ็นเซอร์ IR ด้านซ้าย

# --- Marker Detection Constants ---
WIDTH, HEIGHT = 848, 480
CENTER_X, CENTER_Y = WIDTH / 2, HEIGHT / 2
MASK_TOP_Y, MASK_BOTTOM_Y = 140, 400
MASK_LEFT_X, MASK_RIGHT_X = 80, 768
COLOR_RANGES = {
    'red': [
        {'lower': np.array([0, 90, 70]), 'upper': np.array([10, 255, 255])},
        {'lower': np.array([170, 90, 70]), 'upper': np.array([180, 255, 255])}
    ],
    'green': [{'lower': np.array([75, 230, 25]), 'upper': np.array([90, 255, 255])}],
    'blue': [{'lower': np.array([100, 190, 30]), 'upper': np.array([126, 255, 255])}],
    'yellow': [{'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])}]
}
MIN_CONTOUR_AREA = 800
MAX_CONTOUR_AREA = 50000
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0
MIN_SOLIDITY = 0.4
MORPH_KERNEL_SIZE = (7, 7)
MORPH_OPEN_ITER = 2
MORPH_CLOSE_ITER = 2

# --- Plotting Variables ---
_fig, _ax = plt.subplots(figsize=(8, 8)) # สร้าง figure และ axes สำหรับแสดงแผนที่

# ===================== Marker Detection Functions =====================
def identify_shape(contour):
    shape = 'unknown'
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    area = cv2.contourArea(contour)
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h > 0 else 0
        if 0.95 <= aspect_ratio <= 1.05: shape = 'square'
        elif aspect_ratio > 1.05: shape = 'horizontal_rectangle'
        else: shape = 'vertical_rectangle'
    elif len(approx) > 4:
        if peri > 0:
            circularity = 4 * np.pi * (area / (peri * peri))
            if circularity > 0.85: shape = 'circle'
    return shape

def detect_color_mask(frame, color_name):
    m = frame.copy()
    m[0:MASK_TOP_Y, :] = 0
    m[MASK_BOTTOM_Y:, :] = 0
    m[:, 0:MASK_LEFT_X] = 0
    m[:, MASK_RIGHT_X:] = 0
    hsv = cv2.cvtColor(cv2.GaussianBlur(m, (7,7), 1), cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in COLOR_RANGES[color_name]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, r['lower'], r['upper']))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITER)
    return cv2.medianBlur(mask, 5)

def find_largest_target(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    valid_targets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA): continue
        x, y, w, h = cv2.boundingRect(cnt)
        center_x, center_y = x + w/2, y + h/2
        if not (MASK_TOP_Y < center_y < MASK_BOTTOM_Y and MASK_LEFT_X < center_x < MASK_RIGHT_X): continue
        aspect = float(w)/h if h > 0 else 0
        if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO): continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY: continue
        shape = identify_shape(cnt)
        if shape == 'unknown': continue
        valid_targets.append({'contour': cnt, 'area': area, 'shape': shape})
    if not valid_targets: return None
    return max(valid_targets, key=lambda x: x['area'])

def detect_marker_at_current_location(ep_camera, ep_gimbal):
    """
    เปิดกล้อง, ตรวจจับ Marker และปิดกล้อง
    จะพยายามหา Marker ที่ชัดเจนที่สุดในมุมมองปัจจุบัน
    ถ้าเจอ จะบันทึกข้อมูลสี, รูปทรง และตำแหน่งลงใน global dict `markers_found`
    """
    global markers_found, current_pos
    print(f"[{current_pos}] กำลังตรวจจับ Marker...")
    if current_pos in markers_found:
        print(f"[{current_pos}] เคยตรวจพบ Marker แล้ว. ข้ามการตรวจจับซ้ำ")
        return
    try:
        ep_camera.start_video_stream(display=False, resolution='480p')
        ep_gimbal.recenter().wait_for_completed()
        time.sleep(1)
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=2.0)
        if frame is None:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            return
        found_targets = {}
        for color in ['red', 'green', 'blue', 'yellow']:
            mask = detect_color_mask(frame, color)
            target = find_largest_target(mask)
            if target:
                target['color'] = color
                found_targets[f"{color}_{target['shape']}"] = target
        if found_targets:
            best_target = max(found_targets.values(), key=lambda x: x['area'])
            color, shape = best_target['color'], best_target['shape']
            markers_found[current_pos] = {'color': color, 'shape': shape}
            print(f"!!! [{current_pos}] ตรวจพบ Marker: สี {color.upper()}, รูปทรง {shape.upper()} !!!")
        else:
            print(f"[{current_pos}] ไม่พบ Marker ในบริเวณนี้")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดระหว่างการตรวจจับ Marker: {e}")
    finally:
        ep_camera.stop_video_stream()
        print("ปิดการใช้งานกล้อง")

# ===================== Plotting Functions =====================
def plot_maze(walls_to_plot, visited_to_plot, path_stack_to_plot, current_cell_to_plot, markers_to_plot, title="Maze Exploration"):
    """วาดสถานะปัจจุบันของการสำรวจเขาวงกต"""
    _ax.clear() # ล้างกราฟเดิมออก
    MAZE_BOUNDS_PLOT = (0, 4, 0, 4) # กำหนดขอบเขตของเขาวงกตที่จะวาด (x_min, x_max, y_min, y_max)
    x_min, x_max, y_min, y_max = MAZE_BOUNDS_PLOT

    # วาดช่องที่เคยไปแล้วเป็นสีฟ้าอ่อน
    for x, y in visited_to_plot:
        _ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightcyan', edgecolor='none', zorder=0))

    # วาดกำแพงเป็นเส้นสีดำหนา
    for wall in walls_to_plot:
        (x1, y1), (x2, y2) = wall # กำแพงแต่ละอันเก็บเป็นคู่ตำแหน่ง ((x1,y1), (x2,y2))
        if y1 == y2: # ถ้า y เท่ากัน = กำแพงแนวตั้ง
            x_mid = (x1 + x2) / 2.0
            _ax.plot([x_mid, x_mid], [y1 - 0.5, y1 + 0.5], color='k', linewidth=4)
        elif x1 == x2: # ถ้า x เท่ากัน = กำแพงแนวนอน
            y_mid = (y1 + y2) / 2.0
            _ax.plot([x1 - 0.5, x1 + 0.5], [y_mid, y_mid], color='k', linewidth=4)
    
    # วาดเส้นทางที่เดินมา (path stack) เป็นเส้นสีน้ำเงินพร้อมจุด
    if len(path_stack_to_plot) > 1:
        path_x, path_y = zip(*path_stack_to_plot)
        _ax.plot(path_x, path_y, 'b-o', markersize=4, zorder=1)

    # <<< เพิ่มส่วนนี้เข้ามา: ส่วนวาด Marker ---
    marker_symbols = {'circle': 'o', 'square': 's', 'vertical_rectangle': '|', 'horizontal_rectangle': '_'}
    color_map = {'red': 'r', 'green': 'g', 'blue': 'b', 'yellow': 'y'}
    for pos, data in markers_to_plot.items():
        mx, my = pos
        shape_symbol = marker_symbols.get(data['shape'], '*')
        marker_color = color_map.get(data['color'], 'k')
        _ax.plot(mx, my, marker=shape_symbol, color=marker_color, markersize=15, linestyle='None', zorder=3)

    # วาดตำแหน่งหุ่นยนต์ปัจจุบันเป็นจุดสีแดง
    cx, cy = current_cell_to_plot
    _ax.plot(cx, cy, 'ro', markersize=12, label='Robot', zorder=2)

    # ตั้งค่าต่างๆ ของกราฟ
    _ax.set_xlim(x_min - 0.5, x_max + 0.5)
    _ax.set_ylim(y_min - 0.5, y_max + 0.5)
    _ax.set_aspect('equal', adjustable='box')
    _ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)
    _ax.set_xticks(np.arange(x_min - 0.5, x_max + 1.5, 1))
    _ax.set_yticks(np.arange(y_min - 0.5, y_max + 1.5, 1))
    _ax.set_xticklabels([])
    _ax.set_yticklabels([])
    _ax.set_title(title)
    plt.pause(0.1)

def finalize_show():
    """ปิด interactive mode และแสดงกราฟสุดท้าย"""
    plt.ioff()
    plt.show()

# ===================== Sensor Handling Functions =====================
def sub_tof_handler(sub_info):
    """Callback function สำหรับรับข้อมูลจากเซ็นเซอร์ ToF"""
    global tof_distance_cm
    tof_distance_cm = sub_info[0] / 10.0

def sub_imu_handler(attitude_info):
    """Callback function สำหรับรับข้อมูลมุม yaw จาก IMU"""
    global current_yaw
    current_yaw = attitude_info[0]

def sub_position_handler(position_info):
    """Callback function สำหรับรับข้อมูลตำแหน่งจาก position sensor"""
    global current_x, current_y
    current_x = position_info[0]
    current_y = position_info[1]

def single_lowpass_filter(new_value, last_value, alpha=0.8):
    """Low-pass filter แบบง่าย สำหรับลด noise ของเซ็นเซอร์"""
    return alpha * new_value + (1.0 - alpha) * last_value

def adc_to_cm(adc_value, table):
    """แปลงค่า ADC จากเซ็นเซอร์ IR เป็นระยะทาง cm โดยใช้ linear interpolation"""
    points = sorted(table.items(), key=lambda x: x[0], reverse=True)
    if adc_value >= points[0][0]: return float(points[0][1])
    if adc_value <= points[-1][0]: return float(points[-1][1])
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        if x2 <= adc_value <= x1:
            return float(y1 + (adc_value - x1) * (y2 - y1) / (x2 - x1))
    return 999.0

def read_ir_thread(ep_sensor_adaptor):
    """Thread function สำหรับอ่านค่าเซ็นเซอร์ IR อย่างต่อเนื่อง"""
    global ir_right_cm, ir_left_cm, last_value_right, last_value_left
    while not stop_flag:
        ir_right_adc = ep_sensor_adaptor.get_adc(id=2, port=2)
        ir_left_adc = ep_sensor_adaptor.get_adc(id=1, port=2)
        ir_right_filtered = single_lowpass_filter(ir_right_adc, last_value_right)
        ir_left_filtered = single_lowpass_filter(ir_left_adc, last_value_left)
        last_value_right, last_value_left = ir_right_filtered, ir_left_filtered
        ir_right_cm = adc_to_cm(ir_right_filtered, calibra_table_ir_right)
        ir_left_cm = adc_to_cm(ir_left_filtered, calibra_table_ir_left)
        time.sleep(0.05)

# ===================== Movement Functions =====================
def normalize_angle(angle):
    """ปรับมุมให้อยู่ในช่วง -180 ถึง 180 องศา"""
    while angle > 180: angle -= 360
    while angle <= -180: angle += 360
    return angle

def turn_to_angle(ep_chassis, ep_gimbal, target_angle):
    """หมุนหุ่นยนต์ไปยังมุมเป้าหมายโดยใช้ PID control"""
    global current_yaw
    target_angle = normalize_angle(target_angle)
    print(f"กำลังหมุนไปที่ {target_angle}°")
    while not stop_flag:
        angle_error = normalize_angle(target_angle - current_yaw)
        if abs(angle_error) < 2.0: break
        turn_speed = max(min(angle_error * Kp_turn, TURN_SPEED_Z), -TURN_SPEED_Z)
        ep_chassis.drive_speed(x=0, y=0, z=turn_speed)
        time.sleep(0.02)
    ep_chassis.drive_speed(x=0, y=0, z=0)
    ep_gimbal.recenter().wait_for_completed() # ใช้ recenter เพื่อความแน่นอน
    time.sleep(0.5)

def move_straight_60cm(ep_chassis, target_yaw):
    """
    เดินหน้าตรง 60 cm โดยใช้ PID ควบคุมทั้งระยะทางและทิศทาง
    พร้อมหลีกเลี่ยงกำแพงด้านข้างและหยุดฉุกเฉินเมื่อเจอสิ่งกีดขวางหน้า
    
    Parameters:
        ep_chassis: object ควบคุมการเคลื่อนที่
        target_yaw: มุมทิศทางที่ต้องการรักษา (degrees)
    """
    global ir_left_cm, ir_right_cm, current_yaw, current_x, current_y, tof_distance_cm, stop_flag
    # ประกาศใช้ตัวแปร global หลายตัว

    # ---------- การตั้งค่า (CONFIG) ----------
    target_distance_m = 0.6  # ระยะทางเป้าหมาย 60 เซนติเมตร (0.6 เมตร)
    tol_m = 0.01             # ความคลาดเคลื่อนที่ยอมรับได้ 1 เซนติเมตร (0.01 เมตร)
    FRONT_SAFETY_CM = 12.0   # ระยะปลอดภัยด้านหน้า หยุดเมื่อใกล้น้อยกว่า 12 cm

    # ---------- ค่า Gain สำหรับ PID ----------
    # แกน X (ควบคุมระยะทาง - เดินหน้า/ถอยหลัง)
    Kp_x, Ki_x, Kd_x = 3.0, 0.0, 0.1  # P=3.0, I=0, D=0.1
    # แกน Z (ควบคุมทิศทาง - หมุนซ้าย/ขวา)
    Kp_z, Ki_z, Kd_z = 0.8, 0.02, 0.1  # P=0.8, I=0.02, D=0.1
    max_forward_speed = 2.0  # ความเร็วเดินหน้าสูงสุด 2.0 เมตร/วินาที

    # ---------- ตัวแปรสถานะ PID ----------
    # แกน X (ระยะทาง)
    integral_x, last_error_x = 0.0, 0.0  # เริ่มต้น integral และ error ก่อนหน้า
    # แกน Z (ทิศทาง)
    integral_z, last_error_z = 0.0, 0.0  # เริ่มต้น integral และ error ก่อนหน้า

    # ---------- เริ่มต้น ----------
    start_x, start_y = current_x, current_y  # บันทึกตำแหน่งเริ่มต้น
    last_time = time.time()  # บันทึกเวลาเริ่มต้นสำหรับคำนวณ dt
    print(f"กำลังเคลื่อนที่ไปข้างหน้า {target_distance_m} m (target_yaw={target_yaw}°)")

    # ---------- MAIN LOOP (ลูปหลักสำหรับควบคุม) ----------
    while not stop_flag:  # วนลูปจนกว่าจะมีสัญญาณหยุด
        current_time = time.time()  # เวลาปัจจุบัน
        dt = current_time - last_time  # คำนวณเวลาที่ผ่านไป (delta time)
        
        # ถ้า dt <= 0 (ไม่ควรเกิด) ข้ามรอบนี้
        if dt <= 0:
            time.sleep(0.01)  # รอสักหน่อย
            continue  # กลับไปเริ่มลูปใหม่

        # === คำนวณระยะทางที่เดินไปแล้ว ===
        dx = current_x - start_x  # ผลต่างแกน x
        dy = current_y - start_y  # ผลต่างแกน y
        traveled = math.hypot(dx, dy)  # ระยะทางแบบ Euclidean = sqrt(dx² + dy²)
        remaining = target_distance_m - traveled  # ระยะที่เหลือ
        err = remaining  # error = ระยะที่เหลือ

        # ถ้าถึงเป้าหมายแล้ว (error น้อยกว่า tolerance)
        if abs(err) <= tol_m:
            print(f"ถึงเป้าหมาย: เดินแล้ว {traveled:.3f} m (target 0.6 m)")
            break  # ออกจากลูป

        # ตรวจสอบสิ่งกีดขวางด้านหน้า (ระบบหยุดฉุกเฉิน)
        if tof_distance_cm <= FRONT_SAFETY_CM:
            print(f"หยุดฉุกเฉิน! พบสิ่งกีดขวางหน้า ToF={tof_distance_cm:.1f} cm")
            break  # ออกจากลูปทันที

        # ---------- PID Controller แกน X (ควบคุมระยะทาง) ----------
        error_x = target_distance_m - traveled  # error = เป้าหมาย - ที่เดินไปแล้ว
        integral_x += error_x * dt  # สะสม integral (ผลรวมของ error ตามเวลา)
        derivative_x = (error_x - last_error_x) / dt  # อนุพันธ์ (อัตราการเปลี่ยนแปลงของ error)
        
        # คำนวณความเร็วเดินหน้า = P*error + I*integral + D*derivative
        forward_speed = (Kp_x * error_x) + (Ki_x * integral_x) + (Kd_x * derivative_x)
        
        # จำกัดความเร็วไม่ให้ติดลบและไม่เกินค่าสูงสุด
        x_speed = max(0.0, min(forward_speed, max_forward_speed))
        # max(0.0, ...) = ไม่ให้ถอยหลัง, min(..., 2.0) = ไม่เกิน 2.0 m/s

        # ---------- PID Controller แกน Z (ควบคุมทิศทาง - หมุน) ----------
        print(target_yaw, '-', current_yaw)  # แสดงมุมเป้าหมายและมุมปัจจุบัน
        error_z = normalize_angle(target_yaw - current_yaw)  # error มุม = เป้าหมาย - ปัจจุบัน
        integral_z += error_z * dt  # สะสม integral
        derivative_z = (error_z - last_error_z) / dt  # อนุพันธ์
        
        # คำนวณความเร็วหมุน = P*error + I*integral + D*derivative
        z_speed = (Kp_z * error_z) + (Ki_z * integral_z) + (Kd_z * derivative_z)
        
        # ถ้า error มุมน้อยกว่า 2 องศา ไม่ต้องหมุน
        if abs(error_z) < 2.0:
            z_speed = 0.0  # หยุดหมุน

        # ---------- ระบบหลีกเลี่ยงกำแพงด้านข้าง (Y axis) ----------
        y_speed = 0.0  # เริ่มต้นไม่เคลื่อนที่ด้านข้าง
        
        # ถ้าใกล้กำแพงขวามากเกินไป ขยับไปทางซ้าย (y ติดลบ)
        if ir_right_cm < WALL_AVOID_THRESHOLD_CM:
            y_speed -= WALL_AVOID_SPEED_Y  # เคลื่อนที่ไปทางซ้าย
        
        # ถ้าใกล้กำแพงซ้ายมากเกินไป ขยับไปทางขวา (y บวก)
        if ir_left_cm < WALL_AVOID_THRESHOLD_CM:
            y_speed += WALL_AVOID_SPEED_Y  # เคลื่อนที่ไปทางขวา

        # ---------- ส่งคำสั่งเคลื่อนที่ไปยังหุ่นยนต์ ----------
        ep_chassis.drive_speed(x=x_speed, y=y_speed, z=z_speed)
        # x = เดินหน้า/ถอยหลัง, y = เดินข้าง, z = หมุน (ตอนนี้หมุนด้วย PID แล้ว)

        # อัปเดตค่า error และเวลาสำหรับรอบถัดไป
        last_error_x, last_error_z = error_x, error_z  # บันทึก error สำหรับคำนวณ D
        last_time = current_time  # บันทึกเวลาสำหรับคำนวณ dt ในรอบถัดไป

        # รอ 20 มิลลิวินาที (ความถี่ควบคุม 50 Hz)
        time.sleep(0.02)

    # ---------- หยุดการเคลื่อนที่ ----------
    ep_chassis.drive_speed(x=0, y=0, z=0)  # ตั้งความเร็วทุกแกนเป็น 0
    time.sleep(0.5)  # พักเพื่อให้หุ่นยนต์หยุดนิ่ง
    print("เคลื่อนที่สำเร็จ: ระยะทาง 60 cm")  # แสดงข้อความยืนยัน

# ===================== DFS Logic Functions =====================
def scan_environment():
    """สแกนสภาพแวดล้อมรอบๆ หุ่นยนต์เพื่อหาทิศทางที่เดินได้"""
    global tof_distance_cm, ir_left_cm, ir_right_cm
    open_paths = {'front': False, 'left': False, 'right': False}
    time.sleep(SCAN_DURATION_S)
    if tof_distance_cm > TOF_WALL_THRESHOLD_CM: open_paths['front'] = True
    if ir_left_cm > IR_WALL_THRESHOLD_CM: open_paths['left'] = True
    if ir_right_cm > IR_WALL_THRESHOLD_CM: open_paths['right'] = True
    print(f"ผลสแกน: หน้า: {tof_distance_cm:.1f} cm | ซ้าย: {ir_left_cm:.1f} cm | ขวา: {ir_right_cm:.1f} cm")
    return open_paths

def get_target_coordinates(from_pos, heading_deg):
    """คำนวณพิกัดเป้าหมายเมื่อเดินไปในทิศทางที่กำหนด"""
    x, y = from_pos
    heading = normalize_angle(heading_deg)
    if heading == 0: return (x, y + 1)      # เหนือ
    elif heading == 90: return (x + 1, y)     # ตะวันออก
    elif heading == -90: return (x - 1, y)    # ตะวันตก
    elif abs(heading) == 180: return (x, y - 1) # ใต้
    return from_pos

def get_direction_to_neighbor(from_cell, to_cell):
    """คำนวณมุมทิศทางจากช่องหนึ่งไปยังอีกช่องหนึ่ง"""
    dx = to_cell[0] - from_cell[0]
    dy = to_cell[1] - from_cell[1]
    return normalize_angle(math.degrees(math.atan2(dx, dy)))

def turn_and_move(ep_chassis, ep_gimbal, target_heading):
    """หมุนไปทิศทางเป้าหมาย (ถ้าจำเป็น) แล้วเดินหน้าไป 60 cm"""
    global current_heading_degrees
    if abs(normalize_angle(target_heading - current_heading_degrees)) > 1:
        turn_to_angle(ep_chassis, ep_gimbal, target_heading)
        current_heading_degrees = target_heading
    move_straight_60cm(ep_chassis, target_heading)

def map_current_cell():
    """สแกนและบันทึกข้อมูลแผนที่ของช่องปัจจุบัน"""
    global maze_map, walls, current_pos, current_heading_degrees
    print(f"ช่อง {current_pos} ยังไม่ได้สำรวจ กำลังสแกน...")
    scan_results = scan_environment()
    open_headings = set()
    relative_moves = {'left': -90, 'front': 0, 'right': 90}
    for move_key, is_open in scan_results.items():
        relative_angle = relative_moves[move_key]
        absolute_heading = normalize_angle(current_heading_degrees + relative_angle)
        if is_open:
            open_headings.add(absolute_heading)
        else:
            neighbor_cell = get_target_coordinates(current_pos, absolute_heading)
            walls.add(tuple(sorted((current_pos, neighbor_cell))))
    maze_map[current_pos] = open_headings
    print(f"สร้างแผนที่ช่อง {current_pos} มีทิศทางที่เปิด: {sorted(list(open_headings))}")

def find_and_move_to_next_cell(ep_chassis, ep_gimbal):
    """หาช่องเพื่อนบ้านที่ยังไม่เคยไปและเคลื่อนที่ไปยังช่องนั้น (DFS: ซ้าย -> หน้า -> ขวา)"""
    global visited_nodes, path_stack, current_pos, current_heading_degrees
    search_order_relative = [-90, 0, 90]
    for angle in search_order_relative:
        target_heading = normalize_angle(current_heading_degrees + angle)
        if target_heading in maze_map.get(current_pos, set()):
            target_cell = get_target_coordinates(current_pos, target_heading)
            min_x, min_y = MAP_MIN_BOUNDS
            max_x, max_y = MAP_MAX_BOUNDS
            if not (min_x <= target_cell[0] <= max_x and min_y <= target_cell[1] <= max_y):
                continue
            if target_cell not in visited_nodes:
                print(f"พบเพื่อนบ้านที่ยังไม่เคยไป {target_cell} กำลังเคลื่อนที่...")
                turn_and_move(ep_chassis, ep_gimbal, target_heading)
                visited_nodes.add(target_cell)
                path_stack.append(target_cell)
                current_pos = target_cell
                return True
    return False

def backtrack(ep_chassis, ep_gimbal):
    """ย้อนรอยกลับไปยังช่องก่อนหน้าใน path stack เมื่อเจอทางตัน"""
    global path_stack, current_pos
    print("เจอทางตัน กำลังย้อนรอย (Backtracking)...")
    if len(path_stack) <= 1:
        print("กลับมาที่จุดเริ่มต้น การสำรวจสิ้นสุด")
        return False
    path_stack.pop()
    previous_cell = path_stack[-1]
    backtrack_heading = get_direction_to_neighbor(current_pos, previous_cell)
    print(f"กำลังย้อนรอยจาก {current_pos} ไปยัง {previous_cell}")
    turn_and_move(ep_chassis, ep_gimbal, backtrack_heading)
    current_pos = previous_cell
    return True

# ===================== Main Execution Block =====================
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    # ดึง object ย่อยสำหรับควบคุมแต่ละส่วน
    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor
    ep_gimbal = ep_robot.gimbal
    ep_sensor_adaptor = ep_robot.sensor_adaptor
    ep_camera = ep_robot.camera

    _fig.canvas.manager.set_window_title("Maze & Marker Map")

    # Subscribe และ เริ่ม Thread
    ep_sensor.sub_distance(freq=20, callback=sub_tof_handler)
    ep_chassis.sub_attitude(freq=20, callback=sub_imu_handler)
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    ir_reader = threading.Thread(target=read_ir_thread, args=(ep_sensor_adaptor,), daemon=True)
    ir_reader.start()
    time.sleep(1)

    # --- Initialize DFS State ---
    current_pos = START_CELL
    visited_nodes.add(current_pos)
    path_stack.append(current_pos)
    
    print("เริ่มต้นการสำรวจเขาวงกต และค้นหา Marker...")
    
    try:
        # --- Main Exploration Loop ---
        while path_stack and not stop_flag:
            if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
                print("กดปุ่ม ESC กำลังหยุดการทำงาน...")
                break

            # วาดแผนที่ปัจจุบัน
            # plot_maze(walls, visited_nodes, path_stack, current_pos, markers_found)           # ไม่ต้องการแบบเรียลไทม์
            print(f"\nตำแหน่ง: {current_pos}, ทิศทาง: {current_heading_degrees}°")

            # ถ้าช่องปัจจุบันยังไม่ได้สร้างแผนที่
            if current_pos not in maze_map:
                map_current_cell() # สแกนและบันทึกแผนที่ช่องนี้
                detect_marker_at_current_location(ep_camera, ep_gimbal) # ตรวจหา Marker
            
            # พยายามหาและเคลื่อนที่ไปยังช่องถัดไป
            if find_and_move_to_next_cell(ep_chassis, ep_gimbal):
                continue
            
            # ถ้าไม่มีทางไปต่อ ทำการ backtrack
            if not backtrack(ep_chassis, ep_gimbal):
                break
    finally:
        # --- สิ้นสุดการทำงานและบันทึกแผนที่ ---
        print("\nการสำรวจ DFS เสร็จสมบูรณ์")
        print("กำลังสร้างแผนที่สุดท้าย...")
        plot_maze(walls, visited_nodes, path_stack, current_pos, markers_found, "Final Maze & Marker Map")

        try:
            file_name = 'final_maze_and_marker_map.png' # แก้ไขชื่อไฟล์ให้ตรงกัน
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            print(f"บันทึกแผนที่ '{file_name}' เรียบร้อยแล้ว")
            plt.show() # <<< เพิ่มบรรทัดนี้เข้ามา
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการบันทึกไฟล์: {e}")

        # --- ทำความสะอาดและปิดการเชื่อมต่อ ---
        print("กำลังทำความสะอาดและปิดการเชื่อมต่อ...")
        stop_flag = True
        time.sleep(0.2)
        ep_chassis.drive_speed(x=0, y=0, z=0)

        # Unsubscribe ทุกอย่าง
        ep_sensor.unsub_distance()
        ep_chassis.unsub_attitude()
        ep_chassis.unsub_position()
        
        ep_robot.close()
        print("โปรแกรมจบการทำงาน")