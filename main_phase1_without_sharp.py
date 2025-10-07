from robomaster import robot
import msvcrt
import time
import math
import threading
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json

START_CELL = (0, 2)          # ตำแหน่งเริ่มต้นของหุ่นยนต์ในแผนที่ (x, y)
MAP_MIN_BOUNDS = (0, 0)      # พิกัดซ้าย-ล่างสุดของแผนที่
MAP_MAX_BOUNDS = (5, 5)      # พิกัดขวา-บนสุดของแผนที่
NODE_DISTANCE = 0.6          # ระยะห่างระหว่าง Node (เมตร)

# --- (ลบ) ตาราง Calibrate สำหรับ Analog IR ---

TARGET_WALL_DISTANCE_CM = 8.0  # (ไม่ถูกใช้งานแล้ว)
BASE_FORWARD_SPEED_WF = 0.25   # (ไม่ถูกใช้งานแล้ว)
MAX_Y_SPEED = 0.3              # (ไม่ถูกใช้งานแล้ว)
MAX_Z_SPEED = 32.0             # (ไม่ถูกใช้งานแล้ว)
SCAN_DURATION_S = 0.1
TOF_WALL_THRESHOLD_CM = 50

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

# ===================== Control Class =====================
class Control:
    def __init__(self, ep_chassis):
        self.ep_chassis = ep_chassis

    def stop(self):
        self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
        time.sleep(0.2)

    def turn(self, angle_deg):
        print(f"Action: Turning {angle_deg:.1f} degrees")
        self.ep_chassis.move(x=0, y=0, z=-angle_deg, z_speed=80).wait_for_completed()
        time.sleep(0.2)

    def move_forward_pid(self, cell_size_m, Kp=3, Ki=0.0001, Kd=0.001, v_clip=0.6, tol_m=0.02):
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
    
    # --- (ลบ) ฟังก์ชัน align_with_left_wall ทั้งหมด ---

# ===================== Global State & Constants =====================
stop_flag = False
tof_distance_cm = 999.0
current_yaw = 0.0
current_x = 0.0
current_y = 0.0

# ตัวแปรสำหรับ Digital IR (Analog IR ถูกลบออก)
ir_left_digital = 0   # 0 = ไม่มีกำแพง, 1 = เจอกำแพง
ir_right_digital = 0  # 0 = ไม่มีกำแพง, 1 = เจอกำแพง

# ตัวแปรสำหรับ DFS & Mapping
maze_map = {}
visited_nodes = set()
path_stack = []
walls = set()
current_pos = (1, 1)
current_heading_degrees = 0
markers_found = {}

# --- Marker Detection Constants ---
FRAME_WIDTH = 1400
FRAME_HEIGHT = 480
ROI_TOP = 150
ROI_BOTTOM = FRAME_HEIGHT
ROI_LEFT = 0
ROI_RIGHT = FRAME_WIDTH

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

# ===================== Marker Detection & Plotting Functions =====================
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
    m[0:ROI_TOP, :] = 0
    m[ROI_BOTTOM:, :] = 0
    m[:, 0:ROI_LEFT] = 0
    m[:, ROI_RIGHT:] = 0
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
        
        if not (ROI_TOP < center_y < ROI_BOTTOM and ROI_LEFT < center_x < ROI_RIGHT): continue
        
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

def detect_marker_optimized_scan(ep_camera, ep_gimbal):
    global markers_found, current_pos
    print(f"[{current_pos}] กำลังสแกนหา Marker แบบ Multi-Detection (กลาง->ซ้าย->ขวา)...")
    if current_pos in markers_found:
        print(f"[{current_pos}] เคยสแกนตำแหน่งนี้แล้ว. ข้ามการสแกนซ้ำ")
        return

    found_this_scan = {} 
    
    try:
        ep_camera.start_video_stream(display=True, resolution='480p')
        time.sleep(0.5)

        def scan_and_process(side_name, angle):
            print(f"  -> กำลังสแกนด้าน {side_name} ({angle}°)...")
            if angle == 0:
                ep_gimbal.recenter().wait_for_completed()
            else:
                ep_gimbal.move(yaw=angle, pitch=-3, yaw_speed=240).wait_for_completed()
            
            time.sleep(0.7)
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=2.0)
            if frame is None: return

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
        
        scan_and_process('center', 0)
        scan_and_process('left', -90)
        scan_and_process('right', 90)

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
    MAZE_BOUNDS_PLOT = (0, 5, 0, 5)
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
    
    for pos, marker_list in markers_to_plot.items():
        cell_x, cell_y = pos
        for marker_data in marker_list:
            shape_symbol = marker_symbols.get(marker_data['shape'], '*')
            marker_color = color_map.get(marker_data['color'], 'k')
            side = marker_data.get('side', 'center')
            
            x_offset = 0
            if side == 'left': x_offset = -0.25
            elif side == 'right': x_offset = 0.25
            
            _ax.plot(cell_x + x_offset, cell_y, marker=shape_symbol, color=marker_color, markersize=12, linestyle='None', zorder=3)

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

# --- (ลบ) ฟังก์ชัน single_lowpass_filter ---
# --- (ลบ) ฟังก์ชัน adc_to_cm ---
# --- (ลบ) ฟังก์ชัน read_analog_ir_thread ---
    
def read_digital_ir_thread(ep_sensor_adaptor):
    global ir_left_digital, ir_right_digital
    while not stop_flag:
        try:
            voltage_left = ep_sensor_adaptor.get_io(id=1, port=1)
            voltage_right = ep_sensor_adaptor.get_io(id=1, port=2)

            ir_left_digital = 1 if voltage_left < 0.5 else 0
            ir_right_digital = 1 if voltage_right < 0.5 else 0

        except Exception as e:
            print(f"[ERROR] in read_digital_ir_thread: {e}")
            ir_left_digital, ir_right_digital = 1, 1
        time.sleep(0.05)

# ===================== Movement & DFS Logic Functions =====================
def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle <= -180: angle += 360
    return angle

def scan_environment():
    global tof_distance_cm, ir_left_digital, ir_right_digital
    open_paths = {'front': False, 'left': False, 'right': False}
    time.sleep(SCAN_DURATION_S)
    
    if tof_distance_cm > TOF_WALL_THRESHOLD_CM: open_paths['front'] = True
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
    global current_heading_degrees

    turn_angle = normalize_angle(target_heading - current_heading_degrees)
    if abs(turn_angle) > 2.0:
        controller.turn(turn_angle)
    current_heading_degrees = normalize_angle(target_heading)

    print("\nMoving one node forward...") 
    controller.move_forward_pid(NODE_DISTANCE)

    time.sleep(0.3) 
    
    # --- (ลบ) ตรรกะการเรียก align_with_left_wall ออกไป ---
    print("\nStopped at node.")

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
    global visited_nodes, current_pos, current_heading_degrees

    for angle in [-90, 0, 90]: 
        target_heading = normalize_angle(current_heading_degrees + angle)
        if target_heading in maze_map.get(current_pos, set()):
            target_cell = get_target_coordinates(current_pos, target_heading)
            if target_cell not in visited_nodes:
                print(f"Found unvisited neighbor at {target_cell}, moving one node...")
                turn_and_move(controller, target_heading)
                visited_nodes.add(target_cell)
                path_stack.append(target_cell)
                current_pos = target_cell
                return True
    return False

def backtrack(controller):
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

# ===================== Main Execution Block =====================
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
    # --- (ลบ) การสร้างและเริ่ม analog_ir_reader ---
    digital_ir_reader = threading.Thread(target=read_digital_ir_thread, args=(ep_sensor_adaptor,), daemon=True)
    digital_ir_reader.start()

    time.sleep(1)

    # Initialize DFS
    current_pos = START_CELL
    visited_nodes.add(current_pos)
    path_stack.append(current_pos)
    
    print("Starting Maze Exploration (Node-by-Node)...")
    
    try:
        while path_stack and not stop_flag:
            if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
                print("ESC pressed, stopping...")
                stop_flag = True
                break

            print(f"\n--- Current Position: {current_pos}, Heading: {current_heading_degrees}° ---")

            if current_pos not in maze_map:
                map_current_cell()

                num_open_paths = len(maze_map[current_pos])
                
                if num_open_paths != 2:
                    print(f"[{current_pos}] เป็นทางแยก/ทางตัน ({num_open_paths} paths), กำลังสแกน Marker...")
                    detect_marker_optimized_scan(ep_camera, ep_gimbal)
                else:
                    print(f"[{current_pos}] เป็นทางตรง, ข้ามการสแกน Marker.")

                # plot_maze(walls, visited_nodes, path_stack, current_pos, markers_found)

            if find_and_move_to_next_cell(controller):
                continue
            
            elif not backtrack(controller):
                break

    except (KeyboardInterrupt, Exception) as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nExploration finished or stopped.")
        stop_flag = True
        controller.stop()

        print("Generating final map...")
        plot_maze(walls, visited_nodes, path_stack, current_pos, markers_found, title="Final Maze Map")
        plt.savefig('maze_map.png', dpi=300)
        finalize_show() 
        
        print("Saving map data to maze_map.json...")
        try:
            map_data = {
                'walls': [list(sorted(wall)) for wall in walls],
                'markers': {str(pos): data for pos, data in markers_found.items()},
                'visited_path': list(path_stack)
            }
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