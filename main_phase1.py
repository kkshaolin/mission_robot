from robomaster import robot
import msvcrt
import time
import math
import threading
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json

START_CELL = (0, 0)             # ตำแหน่งเริ่มต้นของหุ่นยนต์ในแผนที่ (x, y)
MAP_MIN_BOUNDS = (0, 0)         # พิกัดซ้าย-ล่างสุดของแผนที่
MAP_MAX_BOUNDS = (3, 3)         # พิกัดขวา-บนสุดของแผนที่
NODE_DISTANCE = 0.6             # ระยะห่างระหว่าง Node (เมตร)

# ===================== PID Controller Class (คงเดิม) =====================
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self._prev_error, self._integral = 0.0, 0.0
        self._last_time = None

    def compute(self, current_value):
        if self._last_time is None:
            self._last_time = time.time()
        
        t, dt = time.time(), time.time() - self._last_time
        if dt <= 0.001:
            return 0.0

        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt
        out = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        self._prev_error, self._last_time = error, t
        return out

# ===================== [ใหม่] Control Class สำหรับการเคลื่อนที่ (คงเดิม) =====================
class Control:
    def __init__(self, ep_chassis):
        self.ep_chassis = ep_chassis

    def stop(self):
        self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
        time.sleep(0.2)

    def turn(self, angle_deg):
        print(f"Action: Turning {angle_deg:.1f} degrees")
        self.ep_chassis.move(x=0, y=0, z=-angle_deg, z_speed=45).wait_for_completed()
        time.sleep(0.5)

    def move_forward_pid(self, cell_size_m, Kp=3, Ki=0.0001, Kd=0.001, v_clip=0.4, tol_m=0.02):                    # ปรับความเร็วเดินหน้า
        global current_x, current_y
        print(f"Action: Moving forward {cell_size_m} m using PID")
        pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=cell_size_m)
        sx, sy = current_x, current_y
        t0 = time.time()
        max_duration_s = 15
        while time.time() - t0 < max_duration_s:
            dist = math.hypot(current_x - sx, current_y - sy)
            speed = float(np.clip(pid.compute(dist), -v_clip, v_clip))
            self.ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=0.1)
            if abs(cell_size_m - dist) < tol_m:
                print("Movement complete.")
                self.stop()
                return
            time.sleep(0.02)
        print("[WARNING] move_forward_pid timed out. Stopping robot.")
        self.stop()
    # ===================== [ฟังก์ชันใหม่ - ปรับแก้สำหรับเซนเซอร์ซ้าย] =====================
    def follow_wall_to_next_node(self, cell_size_m):
        """
        เคลื่อนที่ตามกำแพง "ด้านซ้าย" ไปยัง Node ถัดไปโดยใช้ PID 2 ตัว (เวอร์ชันสมบูรณ์)
        """
        global current_x, current_y, ir_left_cm, ir_right_cm 
        
        print(f"Action: Following LEFT wall for {cell_size_m} m")

        # === [สำคัญ] สร้าง PID controllers โดยใช้ค่า GAINS ที่ดีบักมา ===
        # หมายเหตุ: ค่า Gains เหล่านี้มาจากไฟล์ debug_wall_align.py ของคุณ
        pid_angle = PIDController(Kp=14.0, Ki=0.0001, Kd=0.0002, setpoint=0)
        pid_dist = PIDController(Kp=0.04, Ki=0.0001, Kd=0.0002, setpoint=TARGET_WALL_DISTANCE_CM)

        sx, sy = current_x, current_y
        t0 = time.time()
        max_duration_s = 15

        while time.time() - t0 < max_duration_s:
            # 1. อ่านค่าเซนเซอร์ล่าสุดจากตัวแปร global
            ir_front = ir_left_cm 
            ir_rear = ir_right_cm
            
            # 2. คำนวณ Error (ตรรกะเดียวกับในไฟล์ดีบัก)
            angle_error = ir_front - ir_rear # Error สำหรับการปรับมุม
            
            current_dist_avg = (ir_front + ir_rear) / 2.0
            dist_error = current_dist_avg - TARGET_WALL_DISTANCE_CM # Error สำหรับการปรับระยะห่าง
            # หมายเหตุ: สูตร dist_error นี้ถูกต้องสำหรับแกน y ของหุ่น (ค่าลบ -> เลื่อนซ้าย)

            # 3. คำนวณค่าการปรับแก้จาก PID
            z_speed = pid_angle.compute(angle_error)
            y_speed = pid_dist.compute(dist_error)

            # 4. จำกัดค่า Output
            z_speed = float(np.clip(z_speed, -MAX_Z_SPEED, MAX_Z_SPEED))
            y_speed = float(np.clip(y_speed, -MAX_Y_SPEED, MAX_Y_SPEED))
            
            # 5. สั่งการ Chassis
            self.ep_chassis.drive_speed(x=BASE_FORWARD_SPEED_WF, y=y_speed, z=z_speed, timeout=0.1)

            # 6. ตรวจสอบเงื่อนไขการหยุด
            dist_traveled = math.hypot(current_x - sx, current_y - sy)
            if dist_traveled >= cell_size_m:
                print("Movement complete.")
                self.stop()
                return
            
            time.sleep(0.02)
        
        print("[WARNING] follow_wall_to_next_node timed out. Stopping robot.")
        self.stop()

    def dash_forward(self):
        """
        โหมดวิ่งทางตรง: เดินหน้าต่อเนื่องด้วยความเร็วสูง
        จะหยุดเมื่อ: 1. เจทางแยก 2. เจอ Marker 3. เจอทางตัน
        """
        global current_x, current_y, ir_left_digital, ir_right_digital, marker_sighted_flag, tof_distance_cm

        print("Action (Dash): Starting high-speed forward movement.")
        # ใช้ค่า PID ที่ตอบสนองเร็วขึ้นสำหรับความเร็วสูง
        pid_angle = PIDController(Kp=18.0, Ki=0.0001, Kd=0.0005, setpoint=0)
        pid_dist = PIDController(Kp=0.04, Ki=0.0001, Kd=0.0002, setpoint=TARGET_WALL_DISTANCE_CM)
        sx, sy = current_x, current_y
        
        marker_sighted_flag.clear() # รีเซ็ตสัญญาณ Marker ก่อนเริ่มวิ่ง

        while not stop_flag:
            dist_traveled = math.hypot(current_x - sx, current_y - sy)

            # --- ตรวจสอบเงื่อนไขการหยุด ---
            # 1. หยุดเมื่อเจอทางแยก
            if ir_left_digital == 0 or ir_right_digital == 0:
                print(f"\n[!] Junction detected at {dist_traveled:.2f} m. Stopping.")
                target_dist = math.ceil(dist_traveled / NODE_DISTANCE) * NODE_DISTANCE
                if target_dist - dist_traveled < 0.1 and target_dist > 0: target_dist += NODE_DISTANCE
                self.follow_wall_to_next_node(target_dist - dist_traveled) # เดินส่วนที่เหลือให้ถึงโหนด
                return "JUNCTION"

            # 2. หยุดเมื่อ Marker Spotter ส่งสัญญาณมา
            if marker_sighted_flag.is_set():
                print(f"\n[!] Marker sighted at {dist_traveled:.2f} m. Stopping.")
                target_dist = round(dist_traveled / NODE_DISTANCE) * NODE_DISTANCE
                self.follow_wall_to_next_node(target_dist - dist_traveled, speed=0.15) # เดินช้าๆ ให้ถึงโหนด
                return "MARKER_SIGHTED"

            # 3. หยุดเมื่อเจอทางตัน (ToF)
            if tof_distance_cm < TOF_WALL_THRESHOLD_CM:
                 print(f"\n[!] Dead end detected at {dist_traveled:.2f} m. Stopping.")
                 target_dist = round(dist_traveled / NODE_DISTANCE) * NODE_DISTANCE
                 self.follow_wall_to_next_node(target_dist - dist_traveled, speed=0.15)
                 return "DEAD_END"

            # --- การควบคุมการเคลื่อนที่ (ถ้าไม่เจออะไร) ---
            # --- การควบคุมการเคลื่อนที่ (ถ้าไม่เจออะไร) ---
            ir_front, ir_rear = ir_left_cm, ir_right_cm
            angle_error = ir_front - ir_rear
            
            # [แก้ไข] เปลี่ยนสูตรให้ตรงกับ debug_wall_align.py
            current_dist_avg = (ir_front + ir_rear) / 2.0
            dist_error = current_dist_avg - TARGET_WALL_DISTANCE_CM

            z_speed = np.clip(pid_angle.compute(angle_error), -MAX_Z_SPEED, MAX_Z_SPEED)
            y_speed = np.clip(pid_dist.compute(dist_error), -MAX_Y_SPEED, MAX_Y_SPEED)
            self.ep_chassis.drive_speed(x=DASH_SPEED_WF, y=y_speed, z=z_speed, timeout=0.1)
            time.sleep(0.02)
        
        self.stop()
        return "STOPPED"
    
    def align_with_left_wall(self, duration_s=2.0):
        """
        [ฟังก์ชันใหม่] จัดตำแหน่งหุ่นยนต์ให้ขนานกับกำแพงด้านซ้าย โดยไม่เคลื่อนที่ไปข้างหน้า
        จะทำงานเป็นเวลาสั้นๆ เพื่อปรับมุมและระยะห่างให้คงที่
        """
        print("Action: Aligning with left wall...")
        
        # ใช้ PID Gains ชุดเดียวกับ follow_wall_to_next_node
        pid_angle = PIDController(Kp=14.0, Ki=0.0001, Kd=0.0002, setpoint=0)
        pid_dist = PIDController(Kp=0.04, Ki=0.0001, Kd=0.0002, setpoint=TARGET_WALL_DISTANCE_CM)

        start_time = time.time()
        while time.time() - start_time < duration_s:
            ir_front = ir_left_cm 
            ir_rear = ir_right_cm
            
            # คำนวณ Error เหมือนเดิม
            angle_error = ir_front - ir_rear
            current_dist_avg = (ir_front + ir_rear) / 2.0
            dist_error = current_dist_avg - TARGET_WALL_DISTANCE_CM

            # คำนวณ Speed เหมือนเดิม
            z_speed = pid_angle.compute(angle_error)
            y_speed = pid_dist.compute(dist_error)
            
            z_speed = float(np.clip(z_speed, -MAX_Z_SPEED, MAX_Z_SPEED))
            y_speed = float(np.clip(y_speed, -MAX_Y_SPEED, MAX_Y_SPEED))
            
            # [สำคัญ] สั่งเคลื่อนที่เฉพาะแกน y (ด้านข้าง) และ z (หมุน) โดย x = 0
            self.ep_chassis.drive_speed(x=0, y=y_speed, z=z_speed, timeout=0.1)
            time.sleep(0.02)
        
        print("Alignment complete.")
        self.stop() # หยุดให้สนิทหลังจัดตำแหน่งเสร็จ

# ===================== Global State & Constants [แก้ไข] =====================
stop_flag = False
tof_distance_cm = 999.0
current_yaw = 0.0
current_x = 0.0
current_y = 0.0

# [ใหม่] เพิ่มตัวแปรสำหรับ IR แบบดิจิทัล
ir_left_digital = 0  # 0 = ไม่มีกำแพง, 1 = เจอกำแพง
ir_right_digital = 0 # 0 = ไม่มีกำแพง, 1 = เจอกำแพง

ir_left_cm = 999.0
ir_right_cm = 999.0
last_value_left = 0
last_value_right = 0

# ตาราง Calibrate (ตัวอย่าง - ควรใช้ค่าที่ได้จากการทดลองจริงของคุณ)
CALIBRA_TABLE_IR_FRONT = {536: 10, 471: 15, 333: 20, 299: 25}
CALIBRA_TABLE_IR_REAR = {249: 10, 216: 15, 139: 20, 117: 25}

TARGET_WALL_DISTANCE_CM = 8.0  # ระยะห่างที่ต้องการจากกำแพง
BASE_FORWARD_SPEED_WF = 0.25    # ความเร็วเดินหน้าพื้นฐานตอนตามกำแพง
MAX_Y_SPEED = 0.3               # ความเร็วสูงสุดในการเคลื่อนที่ด้านข้าง
MAX_Z_SPEED = 32.0              # ความเร็วสูงสุดในการหมุนตัว

maze_map = {}
visited_nodes = set()
path_stack = []
walls = set()
current_pos = (1, 1)
current_heading_degrees = 0
markers_found = {}

# ==========================================================
# <<< 1. เพิ่มส่วนนี้เข้ามาใหม่ >>>
# ==========================================================
# --- ตัวแปรสำหรับโหมดอัจฉริยะ ---
ROBOT_MODE = "EXPLORE"  # โหมดเริ่มต้น: EXPLORE, DASH
marker_sighted_flag = threading.Event() # ใช้เป็นสัญญาณว่าเจอ Marker
DASH_SPEED_WF = 0.45 # ความเร็วสูงขึ้นสำหรับ Dash Mode

SCAN_DURATION_S = 0.1
TOF_WALL_THRESHOLD_CM = 50
TOF_DASH_THRESHOLD_CM = 120.0 # ระยะ (cm) ที่จะถือว่าเป็นทางตรงยาว
START_CELL = (0, 0)
MAP_MIN_BOUNDS = (0, 0)
MAP_MAX_BOUNDS = (3, 3)
NODE_DISTANCE = 0.6

# --- Marker Detection Constants (คงเดิม) ---
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

_fig, _ax = plt.subplots(figsize=(8, 8))

# ===================== Marker Detection & Plotting Functions (คงเดิมทั้งหมด) =====================
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

def _process_frame_for_markers(frame, all_found_targets):
    """
    ฟังก์ชันย่อย: ทำหน้าที่วิเคราะห์ภาพ 1 frame เพื่อหา Marker ทุกสี
    และอัปเดตผลลัพธ์ลงใน all_found_targets
    """
    if frame is None:
        return

    # วนลูปหา Marker ทุกสีในภาพนั้น
    for color in ['red', 'green', 'blue', 'yellow']:
        mask = detect_color_mask(frame, color)
        target = find_largest_target(mask)
        if target:
            # เก็บ target ที่ดีที่สุด โดยเทียบกับของเดิมที่เคยเจอ
            target_id = f"{color}_{target['shape']}"
            if target_id not in all_found_targets or target['area'] > all_found_targets[target_id]['area']:
                target['color'] = color
                all_found_targets[target_id] = target

def detect_marker_optimized_scan(ep_camera, ep_gimbal):
    """
    [เวอร์ชัน Final] สแกนหา Marker แบบ กลาง -> ซ้าย -> ขวา (ปรับเป็น 90 องศา พร้อมก้ม)
    และสามารถบันทึก Marker ได้หลายอันพร้อมระบุตำแหน่งที่เจอ (side)
    """
    global markers_found, current_pos
    print(f"[{current_pos}] กำลังสแกนหา Marker แบบ Multi-Detection (กลาง->ซ้าย->ขวา)...")
    if current_pos in markers_found:
        print(f"[{current_pos}] เคยสแกนตำแหน่งนี้แล้ว. ข้ามการสแกนซ้ำ")
        return

    found_this_scan = {} 
    
    try:
        ep_camera.start_video_stream(display=True, resolution='480p')
        time.sleep(0.5)

        # --- สร้างฟังก์ชันย่อยเพื่อลดการเขียนโค้ดซ้ำ ---
        def scan_and_process(side_name, angle):
            print(f"  -> กำลังสแกนด้าน {side_name} ({angle}°)...")
            if angle == 0:
                ep_gimbal.recenter().wait_for_completed()
            else:
                # <<< [แก้ไข] สั่งให้หมุน (yaw) ไปยังมุมที่ต้องการ พร้อมกับก้ม (pitch) ลง 20 องศา
                ep_gimbal.move(yaw=angle, pitch=-20, yaw_speed=240).wait_for_completed()
            
            time.sleep(0.7)
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=2.0)
            if frame is None: return

            # วนลูปหา Marker ทุกสีในภาพ (ส่วนนี้เหมือนเดิม)
            for color in ['red', 'green', 'blue', 'yellow']:
                mask = detect_color_mask(frame, color)
                target = find_largest_target(mask)
                if target:
                    target_id = f"{color}_{target['shape']}"
                    if target_id not in found_this_scan or target['area'] > found_this_scan[target_id]['area']:
                        found_this_scan[target_id] = {
                            'color': color,
                            'shape': target['shape'],
                            'side': side_name,
                            'area': target['area']
                        }
        
        # --- เริ่มกระบวนการสแกน ---
        scan_and_process('center', 0)
        scan_and_process('left', -90)   # <<< [แก้ไข] เปลี่ยนจาก -45 เป็น -90
        scan_and_process('right', 90)  # <<< [แก้ไข] เปลี่ยนจาก 45 เป็น 90

        # --- สรุปผลการสแกน (ส่วนนี้เหมือนเดิม) ---
        if found_this_scan:
            final_markers = [dict(list(v.items())[:-1]) for v in found_this_scan.values()]
            markers_found[current_pos] = final_markers
            print(f"!!! [{current_pos}] ตรวจพบ Marker ทั้งหมด {len(final_markers)} ชิ้น:")
            for m in final_markers:
                print(f"    - สี {m['color'].upper()}, รูปทรง {m['shape'].upper()} (ที่ด้าน {m['side']})")
        else:
            print(f"[{current_pos}] ไม่พบ Marker ใดๆ จากการสแกน")

    except Exception as e:
        print(f"เกิดข้อผิดพลาดระหว่างการสแกน Marker: {e}")
    finally:
        ep_gimbal.recenter().wait_for_completed()
        ep_camera.stop_video_stream()
        print("ปิดการใช้งานกล้อง")

def plot_maze(walls_to_plot, visited_to_plot, path_stack_to_plot, current_cell_to_plot, markers_to_plot, title="Maze Exploration"):
    _ax.clear()
    MAZE_BOUNDS_PLOT = (0, 3, 0, 3) # ควรปรับตามขนาดแผนที่จริง
    x_min, x_max, y_min, y_max = MAZE_BOUNDS_PLOT
    
    # ... (ส่วนของการวาด maze, path, robot เหมือนเดิม) ...
    for x, y in visited_to_plot:
        _ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightcyan', edgecolor='none', zorder=0))
    for wall in walls_to_plot:
        (x1, y1), (x2, y2) = wall
        if y1 == y2:
            x_mid = (x1 + x2) / 2.0
            _ax.plot([x_mid, x_mid], [y1 - 0.5, y1 + 0.5], color='k', linewidth=4)
        elif x1 == x2:
            y_mid = (y1 + y2) / 2.0
            _ax.plot([x1 - 0.5, x1 + 0.5], [y_mid, y_mid], color='k', linewidth=4)
    if len(path_stack_to_plot) > 1:
        path_x, path_y = zip(*path_stack_to_plot)
        _ax.plot(path_x, path_y, 'b-o', markersize=4, zorder=1)
    
    # --- [ส่วนที่แก้ไข] ---
    marker_symbols = {'circle': 'o', 'square': 's', 'vertical_rectangle': '|', 'horizontal_rectangle': '_'}
    color_map = {'red': 'r', 'green': 'g', 'blue': 'b', 'yellow': 'y'}
    
    # วนลูปสำหรับแต่ละตำแหน่ง (cell) ที่มี Marker
    for pos, marker_list in markers_to_plot.items():
        cell_x, cell_y = pos
        # วนลูปสำหรับ Marker แต่ละอันใน List ของตำแหน่งนั้น
        for marker_data in marker_list:
            shape_symbol = marker_symbols.get(marker_data['shape'], '*')
            marker_color = color_map.get(marker_data['color'], 'k')
            side = marker_data.get('side', 'center') # get side, default to 'center'
            
            # กำหนดตำแหน่ง offset ในการวาด
            x_offset = 0
            if side == 'left':
                x_offset = -0.25
            elif side == 'right':
                x_offset = 0.25
            
            # วาด Marker ตามตำแหน่งที่คำนวณได้
            _ax.plot(cell_x + x_offset, cell_y, marker=shape_symbol, color=marker_color, markersize=12, linestyle='None', zorder=3)

    cx, cy = current_cell_to_plot
    _ax.plot(cx, cy, 'ro', markersize=12, label='Robot', zorder=2)
    # ... (ส่วนของการตั้งค่า plot ที่เหลือเหมือนเดิม) ...
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
    plt.ioff()
    plt.show()

# ===================== Sensor Handling Functions =====================
def sub_tof_handler(sub_info):
    global tof_distance_cm
    tof_distance_cm = sub_info[0] / 10.0

def sub_imu_handler(attitude_info):
    global current_yaw
    current_yaw = attitude_info[0]

def sub_position_handler(position_info):
    global current_x, current_y
    current_x = position_info[0]
    current_y = position_info[1]


# [แก้ไข] เขียนฟังก์ชัน read_ir_thread ใหม่ทั้งหมด ✨
def single_lowpass_filter(new_value, last_value, alpha=0.6):
    return alpha * new_value + (1 - alpha) * last_value

def adc_to_cm(adc_value, calibration_table):
    sorted_adc = sorted(calibration_table.keys())
    if adc_value >= sorted_adc[-1]:
        return calibration_table[sorted_adc[-1]]
    if adc_value <= sorted_adc[0]:
        return calibration_table[sorted_adc[0]]
    for i in range(len(sorted_adc) - 1):
        adc1, adc2 = sorted_adc[i], sorted_adc[i+1]
        if adc1 <= adc_value <= adc2:
            dist1, dist2 = calibration_table[adc1], calibration_table[adc2]
            ratio = (adc_value - adc1) / (adc2 - adc1)
            return dist1 + ratio * (dist2 - dist1)
    return 999.0

# [แก้ไข] เขียนฟังก์ชัน read_ir_thread ให้กลับมาอ่านค่า Analog
def read_analog_ir_thread(ep_sensor_adaptor):
    global ir_left_cm, ir_right_cm, last_value_left, last_value_right
    while not stop_flag:
        try:
            # สมมติฐาน: id=1 คือ ซ้าย-หน้า, id=2 คือ ซ้าย-หลัง
            adc_front_left = ep_sensor_adaptor.get_adc(id=1, port=2)
            adc_rear_left = ep_sensor_adaptor.get_adc(id=2, port=2) 
            
            filtered_front = single_lowpass_filter(adc_front_left, last_value_left)
            filtered_rear = single_lowpass_filter(adc_rear_left, last_value_right)
            last_value_left, last_value_right = filtered_front, filtered_rear
            
            # แปลงเป็น cm และอัปเดตตัวแปร global
            ir_left_cm = adc_to_cm(filtered_front, CALIBRA_TABLE_IR_FRONT)
            ir_right_cm = adc_to_cm(filtered_rear, CALIBRA_TABLE_IR_REAR)

        except Exception as e:
            print(f"[ERROR] in IR thread: {e}", end='\r')
            ir_left_cm, ir_right_cm = 999.0, 999.0
        time.sleep(0.02)
    
def read_digital_ir_thread(ep_sensor_adaptor):
    global ir_left_digital, ir_right_digital
    while not stop_flag:
        try:
            # การอ่านค่าเซนเซอร์ Digital มักจะใช้ get_io() เพื่ออ่านแรงดันไฟฟ้า
            # สมมติฐาน:
            # id=3 คือ เซนเซอร์ Digital ด้านซ้าย
            # id=4 คือ เซนเซอร์ Digital ด้านขวา
            voltage_left = ep_sensor_adaptor.get_io(id=1, port=1)
            voltage_right = ep_sensor_adaptor.get_io(id=2, port=1)

            # แปลงแรงดันไฟฟ้าเป็น 0 หรือ 1
            # โดยปกติถ้าเจอวัตถุ (มีกำแพง) แรงดันจะต่ำ, ถ้าไม่เจอจะสูง
            ir_left_digital = 1 if voltage_left < 0.5 else 0
            ir_right_digital = 1 if voltage_right < 0.5 else 0

        except Exception as e:
            print(f"[ERROR] in read_digital_ir_thread: {e}")
            # กรณี error ให้ตั้งเป็น 1 (เจอกำแพง) เพื่อความปลอดภัย
            ir_left_digital, ir_right_digital = 1, 1
        time.sleep(0.05)

def marker_spotter_thread(ep_camera):
    """
    Thread ที่ทำงานเบื้องหลังเพื่อส่องหา Marker อย่างรวดเร็วระหว่างที่หุ่นวิ่ง
    """
    global marker_sighted_flag, ROBOT_MODE, stop_flag
    
    while not stop_flag:
        if ROBOT_MODE != "DASH":
            time.sleep(0.5)
            continue
        
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is None: continue
            
            # ตรวจสอบแบบเร็วๆ แค่ว่ามีสีที่ต้องการในภาพหรือไม่
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for color in COLOR_RANGES.keys():
                mask = cv2.inRange(hsv, COLOR_RANGES[color][0]['lower'], COLOR_RANGES[color][0]['upper'])
                # สำหรับสีแดง ต้องรวม 2 ช่วง
                if color == 'red' and len(COLOR_RANGES[color]) > 1:
                     mask2 = cv2.inRange(hsv, COLOR_RANGES[color][1]['lower'], COLOR_RANGES[color][1]['upper'])
                     mask = cv2.bitwise_or(mask, mask2)
                
                # ถ้ามีพื้นที่สีมากกว่า threshold ให้ส่งสัญญาณ
                if cv2.countNonZero(mask) > 2000: # Threshold พื้นที่สี
                    print(f"\nSpotter: Potential {color} marker detected!")
                    marker_sighted_flag.set() # ส่งสัญญาณให้ dash_forward หยุด
                    time.sleep(1) # หยุดชั่วคราวหลังเจอ
                    break 
        except Exception as e:
            print(f"Spotter Error: {e}")
        
        time.sleep(0.1)


# ===================== Movement & DFS Logic Functions =====================
def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle <= -180: angle += 360
    return angle

# [แก้ไข] อัปเดตฟังก์ชัน scan_environment ให้ใช้ค่าดิจิทัล ✅
def scan_environment():
    global tof_distance_cm, ir_left_digital, ir_right_digital
    open_paths = {'front': False, 'left': False, 'right': False}
    time.sleep(SCAN_DURATION_S)
    
    # การตัดสินใจยังคงเหมือนเดิมสำหรับ ToF
    if tof_distance_cm > TOF_WALL_THRESHOLD_CM: open_paths['front'] = True
    
    # เปลี่ยนเงื่อนไขสำหรับ IR: ทางจะเปิดเมื่อค่าดิจิทัลเป็น 0
    if ir_left_digital == 0: open_paths['left'] = True
    if ir_right_digital == 0: open_paths['right'] = True
        
    print(f"ผลสแกน: หน้า: {tof_distance_cm:.1f} cm | ซ้าย (digital): {ir_left_digital} | ขวา (digital): {ir_right_digital}")
    return open_paths

def get_target_coordinates(from_pos, heading_deg):
    x, y = from_pos
    heading = normalize_angle(heading_deg)
    if heading == 0: return (x, y + 1)
    elif heading == 90: return (x + 1, y)
    elif heading == -90: return (x - 1, y)
    elif abs(heading) == 180: return (x, y - 1)
    return from_pos

def get_direction_to_neighbor(from_cell, to_cell):
    dx = to_cell[0] - from_cell[0]
    dy = to_cell[1] - from_cell[1]
    return normalize_angle(math.degrees(math.atan2(dx, dy)))

def turn_and_move(controller, target_heading):
    global current_heading_degrees, ir_left_digital # เพิ่ม ir_left_digital เข้ามาใน global

    # ส่วนของการหมุนตัวยังคงเหมือนเดิม
    turn_angle = normalize_angle(target_heading - current_heading_degrees)
    if abs(turn_angle) > 2.0:
        controller.turn(turn_angle)
    current_heading_degrees = normalize_angle(target_heading)

    # --- <<< [ส่วนที่แก้ไขหลัก] เพิ่มเงื่อนไขตรวจสอบกำแพง >>> ---
    
    # 1. ตรวจสอบว่ามีกำแพงด้านซ้ายหรือไม่ (ir_left_digital == 1)
    if ir_left_digital == 1:
        # ถ้ามีกำแพง: ให้ใช้โหมดเดินตามกำแพง ซึ่งจะทั้งจัดตำแหน่งและเคลื่อนที่ไปพร้อมกัน
        print("Path has a left wall. Engaging wall-following mode...")
        controller.follow_wall_to_next_node(NODE_DISTANCE)
    else:
        # ถ้าไม่มีกำแพง (ที่โล่ง): ให้ใช้โหมดเดินหน้าตรงๆ แบบปกติ
        print("Path is open on the left. Using standard PID forward movement...")
        controller.move_forward_pid(NODE_DISTANCE)

def map_current_cell():
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

def find_and_move_to_next_cell(controller, ep_camera, ep_gimbal):
    global visited_nodes, current_pos, current_heading_degrees

    # <<< [ส่วนที่แก้ไข] ลบการเรียก Dash ออกไป ให้เคลื่อนที่ทีละช่องเท่านั้น >>>
    # จัดลำดับความสำคัญให้ไปข้างหน้าก่อน ถ้าทำได้
    for angle in [0, -90, 90]: # เช็ค หน้า -> ซ้าย -> ขวา
        target_heading = normalize_angle(current_heading_degrees + angle)
        if target_heading in maze_map.get(current_pos, set()):
            target_cell = get_target_coordinates(current_pos, target_heading)
            if target_cell not in visited_nodes:
                print(f"Found unvisited neighbor at {target_cell}, moving one node...")
                turn_and_move(controller, target_heading)
                visited_nodes.add(target_cell)
                path_stack.append(target_cell)
                current_pos = target_cell
                return True # บอก main loop ว่ามีการเคลื่อนที่แล้ว
    return False

def _execute_dash_and_update_map(controller, ep_camera, ep_gimbal):
    """
    ฟังก์ชันสำหรับเริ่มการ Dash และอัปเดตแผนที่ตามผลลัพธ์ที่ได้
    """
    global ROBOT_MODE, current_pos, visited_nodes, path_stack, current_heading_degrees, current_x, current_y
    
    ROBOT_MODE = "DASH"
    ep_camera.start_video_stream(display=False, resolution='480p')
    ep_gimbal.recenter().wait_for_completed()
    time.sleep(0.5)

    stop_reason = controller.dash_forward() # เริ่มวิ่ง
    
    ep_camera.stop_video_stream()
    ROBOT_MODE = "EXPLORE" # กลับสู่โหมดปกติเสมอ

    # อัปเดตแผนที่ตามระยะทางที่วิ่งได้จริง
    # (โค้ดส่วนนี้ยกมาจาก find_and_move_to_next_cell ของเดิม)
    start_pos_coords = (path_stack[-1][0] * NODE_DISTANCE, path_stack[-1][1] * NODE_DISTANCE)
    dist_traveled = math.hypot(current_x - start_pos_coords[0], current_y - start_pos_coords[1])
    nodes_traveled = int(round(dist_traveled / NODE_DISTANCE))
    
    print(f"Dash moved {nodes_traveled} nodes.")
    # อัปเดต path stack และ visited nodes ตามจำนวนช่องที่เคลื่อนที่ได้
    for i in range(nodes_traveled):
        last_pos = path_stack[-1]
        next_pos = get_target_coordinates(last_pos, current_heading_degrees)
        if next_pos in visited_nodes: break
        visited_nodes.add(next_pos)
        path_stack.append(next_pos)
    
    current_pos = path_stack[-1] # อัปเดตตำแหน่งปัจจุบัน

def backtrack(controller):
    global path_stack, current_pos, previous_cell # แก้ไข global เล็กน้อย
    print("เจอทางตัน กำลังย้อนรอย (Backtracking)...")
    if len(path_stack) <= 1:
        print("กลับมาที่จุดเริ่มต้น การสำรวจสิ้นสุด")
        return False
    path_stack.pop()
    previous_cell = path_stack[-1]
    backtrack_heading = get_direction_to_neighbor(current_pos, previous_cell)
    print(f"กำลังย้อนรอยจาก {current_pos} ไปยัง {previous_cell}")
    
    # [เปลี่ยน] เรียก turn_and_move แบบไม่มี follow_side
    turn_and_move(controller, backtrack_heading) 

    current_pos = previous_cell
    return True

# ===================== Main Execution Block (คงเดิม) =====================
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor
    ep_gimbal = ep_robot.gimbal
    ep_sensor_adaptor = ep_robot.sensor_adaptor
    ep_camera = ep_robot.camera

    ep_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)
    ep_gimbal.recenter().wait_for_completed()
    controller = Control(ep_chassis)

    # Subscribe to sensors
    ep_sensor.sub_distance(freq=20, callback=sub_tof_handler)
    ep_chassis.sub_attitude(freq=20, callback=sub_imu_handler)
    ep_chassis.sub_position(freq=5, callback=sub_position_handler)

    # Start sensor threads
    analog_ir_reader = threading.Thread(target=read_analog_ir_thread, args=(ep_sensor_adaptor,), daemon=True)
    digital_ir_reader = threading.Thread(target=read_digital_ir_thread, args=(ep_sensor_adaptor,), daemon=True)
    analog_ir_reader.start()
    digital_ir_reader.start()

    # <<< เพิ่มเข้ามาใหม่ >>>
    # เริ่ม Thread สำหรับส่องหา Marker
    marker_spotter = threading.Thread(target=marker_spotter_thread, args=(ep_camera,), daemon=True)
    marker_spotter.start()

    time.sleep(1)

    # Initialize DFS
    current_pos = START_CELL
    visited_nodes.add(current_pos)
    path_stack.append(current_pos)
    
    print("Starting Smart Maze Exploration...")
    
    try:
        while path_stack and not stop_flag:
            if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
                print("ESC pressed, stopping...")
                stop_flag = True
                break

            print(f"\n--- Current Position: {current_pos}, Heading: {current_heading_degrees}° ---")

            if current_pos not in maze_map:
                map_current_cell()
                detect_marker_optimized_scan(ep_camera, ep_gimbal)
                # วาดแผนที่ทุกครั้งหลังสแกนเสร็จ
                #plot_maze(walls, visited_nodes, path_stack, current_pos, markers_found)

            # --- [LOGIC การตัดสินใจที่ปรับปรุงใหม่] ---
            
            # 1. เช็คเงื่อนไข Dash ด้วย ToF ก่อนเป็นอันดับแรก
            can_dash_forward = normalize_angle(current_heading_degrees) in maze_map.get(current_pos, set())
            next_cell_is_unvisited = get_target_coordinates(current_pos, current_heading_degrees) not in visited_nodes

            # <<< [ส่วนที่แก้ไข] เพิ่มเงื่อนไขบังคับว่าต้องมีกำแพงสองข้าง >>>
            is_in_corridor = (ir_left_digital == 1 and ir_right_digital == 1)

            if can_dash_forward and next_cell_is_unvisited and tof_distance_cm > TOF_DASH_THRESHOLD_CM and is_in_corridor:
                print(f"\n[!] Perfect corridor detected (ToF: {tof_distance_cm:.1f} cm, Walls L/R). Entering DASH mode.")
                _execute_dash_and_update_map(controller, ep_camera, ep_gimbal)
                continue # ข้ามไปรอบถัดไปเลย

            # 2. ถ้า Dash ไม่ได้ ให้ใช้ตรรกะการหาเส้นทางปกติ (ซึ่งจะไม่มีการ Dash แล้ว)
            elif find_and_move_to_next_cell(controller, ep_camera, ep_gimbal):
                continue
            
            # 3. ถ้าไปต่อไม่ได้จริงๆ ให้ Backtrack
            elif not backtrack(controller):
                break

    except (KeyboardInterrupt, Exception) as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nExploration finished or stopped.")
        stop_flag = True # <<< เพิ่มเข้ามาใหม่ เพื่อให้ Thread หยุดทำงาน
        controller.stop()

        # ===================== [แก้ไข] เพิ่มส่วนนี้เข้ามา =====================
        print("Generating final map...")
        # 1. วาดแผนที่ครั้งสุดท้ายด้วยข้อมูลทั้งหมดที่รวบรวมมา
        plot_maze(walls, visited_nodes, path_stack, current_pos, markers_found, title="Final Maze Map")
        
        # 2. แสดงหน้าต่างแผนที่ค้างไว้
        finalize_show() 
        # =================================================================

        # ... (ส่วนบันทึกไฟล์ JSON และ Plotting ของเดิม) ...
        print("Saving map data to maze_map.json...")
        try:
            # แปลงข้อมูลให้อยู่ในรูปแบบที่ JSON จัดเก็บได้ (เช่น set -> list, tuple -> list)
            map_data = {
                'walls': [list(sorted(wall)) for wall in walls],
                'markers': {str(pos): data for pos, data in markers_found.items()},
                'visited_path': list(path_stack)
            }

            # เขียนไฟล์ JSON
            with open('maze_map.json', 'w', encoding='utf-8') as f:
                json.dump(map_data, f, indent=4, ensure_ascii=False)
            
            print("Successfully saved map to maze_map.json")

        except Exception as e:
            print(f"Error saving JSON file: {e}")
        
        ep_sensor.unsub_distance()
        ep_chassis.unsub_attitude()
        ep_chassis.unsub_position()
        ep_robot.close()
        print("Program terminated.")