from robomaster import robot
import msvcrt
import time
import math
import threading
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json # เพิ่ม import json สำหรับการบันทึกไฟล์

# ===================== PID Controller Class =====================
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

# ===================== [ใหม่] Control Class สำหรับการเคลื่อนที่ =====================
class Control:
    def __init__(self, ep_chassis):
        self.ep_chassis = ep_chassis

    def stop(self):
        self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
        time.sleep(0.2)

    def turn(self, angle_deg):
        print(f"Action: Turning {angle_deg:.1f} degrees")
        # ใช้ z_speed เป็นค่าบวกเสมอ และให้ค่า angle_deg เป็นตัวกำหนดทิศทาง
        self.ep_chassis.move(x=0, y=0, z=-angle_deg, z_speed=45).wait_for_completed()
        time.sleep(0.5)

    def move_forward_pid(self, cell_size_m, Kp=3, Ki=0.0001, Kd=0.001, v_clip=0.4, tol_m=0.02):
        global current_x, current_y # เข้าถึง global state ของตำแหน่ง
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

# ===================== Global State & Constants =====================
stop_flag = False
tof_distance_cm = 999.0
current_yaw = 0.0
current_x = 0.0
current_y = 0.0
ir_left_cm = 999.0
ir_right_cm = 999.0
last_value_left = 0
last_value_right = 0

maze_map = {}
visited_nodes = set()
path_stack = []
walls = set()
current_pos = (1, 1)
current_heading_degrees = 0
markers_found = {}

SCAN_DURATION_S = 0.2
TOF_WALL_THRESHOLD_CM = 50
IR_WALL_THRESHOLD_CM = 30
START_CELL = (0, 0)
MAP_MIN_BOUNDS = (0, 0)
MAP_MAX_BOUNDS = (3, 3)
NODE_DISTANCE = 0.6

calibra_table_ir_right = {580: 5, 430: 17.5, 257: 32, 210: 999}
calibra_table_ir_left = {700:5, 200: 17.5, 107: 33, 100: 999}

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

def detect_marker_at_current_location(ep_camera, ep_gimbal):
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

def plot_maze(walls_to_plot, visited_to_plot, path_stack_to_plot, current_cell_to_plot, markers_to_plot, title="Maze Exploration"):
    _ax.clear()
    MAZE_BOUNDS_PLOT = (0, 3, 0, 3)
    x_min, x_max, y_min, y_max = MAZE_BOUNDS_PLOT
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
    marker_symbols = {'circle': 'o', 'square': 's', 'vertical_rectangle': '|', 'horizontal_rectangle': '_'}
    color_map = {'red': 'r', 'green': 'g', 'blue': 'b', 'yellow': 'y'}
    for pos, data in markers_to_plot.items():
        mx, my = pos
        shape_symbol = marker_symbols.get(data['shape'], '*')
        marker_color = color_map.get(data['color'], 'k')
        _ax.plot(mx, my, marker=shape_symbol, color=marker_color, markersize=15, linestyle='None', zorder=3)
    cx, cy = current_cell_to_plot
    _ax.plot(cx, cy, 'ro', markersize=12, label='Robot', zorder=2)
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

# ===================== Sensor Handling Functions (คงเดิมทั้งหมด) =====================
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

def single_lowpass_filter(new_value, last_value, alpha=0.8):
    return alpha * new_value + (1.0 - alpha) * last_value

def adc_to_cm(adc_value, table):
    """
    [แก้ไขแล้ว] แปลงค่า ADC จากเซ็นเซอร์ IR เป็นระยะทาง cm 
    โดยจะคืนค่า 999.0 เมื่อระยะทางไกลเกินกว่าค่าในตาราง
    """
    points = sorted(table.items(), key=lambda x: x[0], reverse=True)
    
    # ถ้าค่า ADC สูงกว่าค่าสูงสุดในตาราง -> ระยะใกล้มาก คืนค่าระยะทางต่ำสุด
    if adc_value >= points[0][0]: 
        return float(points[0][1])
        
    # ถ้าค่า ADC ต่ำกว่าค่าต่ำสุดในตาราง -> ระยะไกลมาก (ไม่มีกำแพง) คืนค่า 999.0
    if adc_value <= points[-1][0]: 
        return 999.0  # <--- จุดที่แก้ไข
        
    # ถ้าค่าอยู่ระหว่างจุดในตาราง ให้คำนวณแบบ linear interpolation
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        if x2 <= adc_value <= x1:
            return float(y1 + (adc_value - x1) * (y2 - y1) / (x2 - x1))
            
    # กรณีอื่นๆ ที่ไม่เข้าเงื่อนไข ให้ถือว่าไกล
    return 999.0

def read_ir_thread(ep_sensor_adaptor):
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

# ===================== Movement & DFS Logic Functions (ปรับปรุงแล้ว) =====================
def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle <= -180: angle += 360
    return angle

def scan_environment():
    global tof_distance_cm, ir_left_cm, ir_right_cm
    open_paths = {'front': False, 'left': False, 'right': False}
    time.sleep(SCAN_DURATION_S)
    if tof_distance_cm > TOF_WALL_THRESHOLD_CM: open_paths['front'] = True
    if ir_left_cm > IR_WALL_THRESHOLD_CM: open_paths['left'] = True
    if ir_right_cm > IR_WALL_THRESHOLD_CM: open_paths['right'] = True
    print(f"ผลสแกน: หน้า: {tof_distance_cm:.1f} cm | ซ้าย: {ir_left_cm:.1f} cm | ขวา: {ir_right_cm:.1f} cm")
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
    """[แก้ไข] หมุนและเคลื่อนที่โดยใช้ Control class"""
    global current_heading_degrees
    
    # คำนวณมุมที่ต้องเลี้ยว
    turn_angle = normalize_angle(target_heading - current_heading_degrees)
    if abs(turn_angle) > 2.0: # ถ้ามุมต่างกันเกิน 2 องศา ให้ทำการเลี้ยว
        controller.turn(turn_angle)
    
    current_heading_degrees = normalize_angle(target_heading) # อัปเดตทิศทางปัจจุบัน
    
    # เดินหน้า
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

def find_and_move_to_next_cell(controller):
    """[แก้ไข] รับ controller object"""
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
                turn_and_move(controller, target_heading)
                visited_nodes.add(target_cell)
                path_stack.append(target_cell)
                current_pos = target_cell
                return True
    return False

def backtrack(controller):
    """[แก้ไข] รับ controller object"""
    global path_stack, current_pos
    print("เจอทางตัน กำลังย้อนรอย (Backtracking)...")
    if len(path_stack) <= 1:
        print("กลับมาที่จุดเริ่มต้น การสำรวจสิ้นสุด")
        return False
    path_stack.pop()
    previous_cell = path_stack[-1]
    backtrack_heading = get_direction_to_neighbor(current_pos, previous_cell)
    print(f"กำลังย้อนรอยจาก {current_pos} ไปยัง {previous_cell}")
    turn_and_move(controller, backtrack_heading)
    current_pos = previous_cell
    return True

# ===================== Main Execution Block (ปรับปรุงแล้ว) =====================
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)
    print("ตั้งค่า Robot Mode เป็น CHASSIS_LEAD เรียบร้อยแล้ว")

    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor
    ep_gimbal = ep_robot.gimbal
    ep_sensor_adaptor = ep_robot.sensor_adaptor
    ep_camera = ep_robot.camera

    ep_gimbal.recenter().wait_for_completed()

    # --- [ใหม่] สร้าง Instance ของ Control class ---
    controller = Control(ep_chassis)

    _fig.canvas.manager.set_window_title("Maze & Marker Map")

    ep_sensor.sub_distance(freq=20, callback=sub_tof_handler)
    ep_chassis.sub_attitude(freq=20, callback=sub_imu_handler)
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    ir_reader = threading.Thread(target=read_ir_thread, args=(ep_sensor_adaptor,), daemon=True)
    ir_reader.start()
    time.sleep(1)

    current_pos = START_CELL
    visited_nodes.add(current_pos)
    path_stack.append(current_pos)
    
    print("เริ่มต้นการสำรวจเขาวงกต และค้นหา Marker...")
    
    try:
        while path_stack and not stop_flag:
            if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
                print("กดปุ่ม ESC กำลังหยุดการทำงาน...")
                break

            print(f"\nตำแหน่ง: {current_pos}, ทิศทาง: {current_heading_degrees}°")

            if current_pos not in maze_map:
                map_current_cell()
                detect_marker_at_current_location(ep_camera, ep_gimbal)
            
            # --- [แก้ไข] ส่ง controller เข้าไปในฟังก์ชัน ---
            if find_and_move_to_next_cell(controller):
                continue
            
            # --- [แก้ไข] ส่ง controller เข้าไปในฟังก์ชัน ---
            if not backtrack(controller):
                break
    finally:
        print("\nการสำรวจ DFS เสร็จสมบูรณ์")

        print("กำลังบันทึกข้อมูลแผนที่และ Marker ลงไฟล์ JSON...")
        maze_map_serializable = {str(k): list(v) for k, v in maze_map.items()}
        markers_found_serializable = {str(k): v for k, v in markers_found.items()}

        output_data = {
            "maze_map": maze_map_serializable,
            "markers_found": markers_found_serializable,
            "start_node": START_CELL
        }
        try:
            with open('map_data.json', 'w') as f:
                json.dump(output_data, f, indent=4)
            print("บันทึกไฟล์ 'map_data.json' เรียบร้อยแล้ว")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการบันทึกไฟล์ JSON: {e}")

        print("กำลังสร้างแผนที่สุดท้าย...")
        plot_maze(walls, visited_nodes, path_stack, current_pos, markers_found, "Final Maze & Marker Map")

        try:
            file_name = 'final_maze_and_marker_map.png'
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            print(f"บันทึกแผนที่ '{file_name}' เรียบร้อยแล้ว")
            plt.show()
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการบันทึกไฟล์: {e}")

        print("กำลังทำความสะอาดและปิดการเชื่อมต่อ...")
        stop_flag = True
        time.sleep(0.2)
        controller.stop() # ใช้ controller.stop() แทน

        ep_sensor.unsub_distance()
        ep_chassis.unsub_attitude()
        ep_chassis.unsub_position()
        
        ep_robot.close()
        print("โปรแกรมจบการทำงาน")