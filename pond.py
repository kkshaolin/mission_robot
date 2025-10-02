# ==============================================================================
#                 RoboMaster EP - DFS Maze Solver with Live Plotting
# ==============================================================================
#
# ผู้พัฒนา: Gemini AI (รวมโค้ดและปรับปรุงจากโค้ดต้นฉบับของผู้ใช้)
# วันที่: 2 ตุลาคม 2025
#
# คำอธิบาย:
# โค้ดนี้ควบคุมหุ่นยนต์ Robomaster EP ให้สำรวจและสร้างแผนที่ของเขาวงกต
# โดยใช้อัลกอริทึม Depth-First Search (DFS) และเดินทางไปยังเป้าหมายด้วย
# อัลกอริทึม Breadth-First Search (BFS)
#
# จุดเด่น:
# - ใช้ Thread แยกสำหรับการอ่านค่า IR Sensor แบบ Real-time เพื่อความแม่นยำ
# - ใช้ PID Control ในการเคลื่อนที่ตรงและ Wall Avoidance เพื่อความเสถียร
# - ใช้ Class 'MazeSolver' ในการจัดการ Logic การสำรวจและสร้างแผนที่
# - แสดงผลแผนที่การสำรวจแบบสดๆ ด้วย Matplotlib
# - สามารถคำนวณเส้นทางที่สั้นที่สุดและเดินทางไปยังเป้าหมายได้หลังสำรวจเสร็จ
#
# ==============================================================================

# --- 1. Imports and Setup ---
import time
import math
import threading
import msvcrt
import numpy as np
import matplotlib.pyplot as plt
from robomaster import robot
from collections import deque # << ADDED: นำเข้า deque สำหรับอัลกอริทึมหาเส้นทาง

# --- 2. Global Constants and Variables ---

# -- Flags & Sensor Globals --
stop_flag = False
tof_distance_cm = 999.0
current_yaw = 0.0
ir_left_cm = 999.0
ir_right_cm = 999.0
last_value_left = 0
last_value_right = 0

# -- IR Sensor Calibration Tables (สำคัญ: ควร Calibrate ใหม่) --
calibra_table_ir_right = {
    615: 5,  415: 15, 275: 25,
    605: 10, 335: 20, 255: 30
}
calibra_table_ir_left = {
    680: 5,  300: 15, 210: 25,
    420: 10, 235: 20, 175: 30
}

# -- Movement & Maze Constants --
NODE_DISTANCE_M = 0.6          # ระยะห่างระหว่างโหนด (60 cm)
TOF_WALL_THRESHOLD_CM = 35.0   # ระยะ ToF ที่ถือว่าเป็นทางเปิด (ด้านหน้า)
IR_WALL_THRESHOLD_CM = 25.0    # ระยะ IR ที่ถือว่าเป็นทางเปิด (ด้านข้าง)
WALL_AVOID_THRESHOLD_CM = 15.0 # ระยะ IR ที่ต้องเริ่มขยับหนีกำแพง
WALL_AVOID_SPEED_Y = 0.05      # ความเร็วในการขยับหนี (m/s)
MOVE_SPEED_X = 0.2             # ความเร็วในการเดินหน้า
TURN_SPEED_Z = 60              # ความเร็วในการหมุน

# -- PID Controller Constants --
Kp_turn = 2.5
Kp_straight = 0.8
Ki_straight = 0.02
Kd_straight = 0.1
integral_straight = 0.0
last_error_straight = 0.0

# -- Matplotlib Visualization Setup --
plt.ion() # เปิด interactive mode
_fig, _ax = plt.subplots(figsize=(8, 8))


# --- 3. Low-Level Robot Control Functions ---

# -- Sensor Callbacks & Helpers --
def sub_tof_handler(sub_info):
    global tof_distance_cm
    tof_distance_cm = sub_info[0] / 10.0

def sub_imu_handler(attitude_info):
    global current_yaw
    current_yaw = attitude_info[0]

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
    return 999.0

def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle <= -180: angle += 360
    return angle

# -- IR Sensor Reading Thread --
def read_ir_thread(ep_sensor_adaptor):
    global ir_right_cm, ir_left_cm, last_value_right, last_value_left
    print("IR reading thread started.")
    while not stop_flag:
        ir_right_adc = ep_sensor_adaptor.get_adc(id=2, port=2)
        ir_left_adc = ep_sensor_adaptor.get_adc(id=1, port=2)
        ir_right_filtered = single_lowpass_filter(ir_right_adc, last_value_right)
        ir_left_filtered = single_lowpass_filter(ir_left_adc, last_value_left)
        last_value_right = ir_right_filtered
        last_value_left = ir_left_filtered
        ir_right_cm = adc_to_cm(ir_right_filtered, calibra_table_ir_right)
        ir_left_cm = adc_to_cm(ir_left_filtered, calibra_table_ir_left)
        time.sleep(0.05)
    print("IR reading thread stopped.")

# -- Core Movement Functions --
def turn_to_angle(ep_chassis, ep_gimbal, target_angle):
    global current_yaw
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
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

def move_straight_60cm(ep_chassis, target_yaw):
    global integral_straight, last_error_straight, ir_left_cm, ir_right_cm
    print(f"กำลังเคลื่อนที่ไปข้างหน้า {NODE_DISTANCE_M*100} cm ที่มุม {target_yaw:.1f}°")
    integral_straight, last_error_straight = 0.0, 0.0
    start_time = time.time()
    last_time = start_time
    duration = (NODE_DISTANCE_M / MOVE_SPEED_X) * 1.1
    while (time.time() - start_time) < duration and not stop_flag:
        current_time = time.time(); dt = current_time - last_time
        if dt <= 0: time.sleep(0.01); continue
        error = normalize_angle(target_yaw - current_yaw)
        integral_straight += error * dt
        derivative = (error - last_error_straight) / dt
        z_correct_speed = (Kp_straight * error) + (Ki_straight * integral_straight) + (Kd_straight * derivative)
        y_correct_speed = 0.0
        if ir_right_cm < WALL_AVOID_THRESHOLD_CM: y_correct_speed -= WALL_AVOID_SPEED_Y
        if ir_left_cm < WALL_AVOID_THRESHOLD_CM: y_correct_speed += WALL_AVOID_SPEED_Y
        ep_chassis.drive_speed(x=MOVE_SPEED_X, y=y_correct_speed, z=z_correct_speed)
        last_error_straight, last_time = error, current_time
        time.sleep(0.02)
    ep_chassis.drive_speed(x=0, y=0, z=0); time.sleep(0.5)
    print("เคลื่อนที่สำเร็จ")

# -- Environment Scanning --
def scan_environment():
    global tof_distance_cm, ir_left_cm, ir_right_cm
    print("--- เริ่มการสแกนสภาพแวดล้อม ---")
    open_paths = {'front': False, 'left': False, 'right': False}
    time.sleep(0.2)
    if tof_distance_cm > TOF_WALL_THRESHOLD_CM: open_paths['front'] = True
    if ir_left_cm > IR_WALL_THRESHOLD_CM: open_paths['left'] = True
    if ir_right_cm > IR_WALL_THRESHOLD_CM: open_paths['right'] = True
    print(f"ผลสแกน [หน้า-ToF]: {tof_distance_cm:.1f} cm | [ซ้าย-IR]: {ir_left_cm:.1f} cm | [ขวา-IR]: {ir_right_cm:.1f} cm")
    print(f"--- สแกนเสร็จสิ้น: {open_paths} ---")
    return open_paths


# --- 4. Visualization Functions ---

def plot_maze(walls, current_cell, visited, title="Maze Exploration"):
    ax = _ax
    ax.clear()
    all_cells = list(visited)
    if not all_cells: all_cells = [(0, 0)]
    all_x = [c[0] for c in all_cells]; all_y = [c[1] for c in all_cells]
    x_min, x_max = min(all_x) - 1, max(all_x) + 1
    y_min, y_max = min(all_y) - 1, max(all_y) + 1
    for x, y in visited:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightcyan', edgecolor='none', zorder=0))
    for wall in walls.keys():
        (x1, y1), (x2, y2) = wall
        if y1 == y2:
            x_mid = (x1 + x2) / 2.0
            ax.plot([x_mid, x_mid], [y1 - 0.5, y1 + 0.5], color='k', linewidth=4)
        elif x1 == x2:
            y_mid = (y1 + y2) / 2.0
            ax.plot([x1 - 0.5, x1 + 0.5], [y_mid, y_mid], color='k', linewidth=4)
    cx, cy = current_cell
    ax.plot(cx, cy, 'bo', markersize=15, label='Robot', zorder=2)
    ax.set_xlim(x_min - 0.5, x_max + 0.5); ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.set_aspect('equal', adjustable='box'); ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(x_min - 0.5, x_max + 1.5, 1)); ax.set_yticks(np.arange(y_min - 0.5, y_max + 1.5, 1))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_title(title)
    _fig.canvas.draw()
    _fig.canvas.flush_events()

def finalize_show():
    plt.ioff()
    plt.show()

# --- 5. Main MazeSolver Class ---

class MazeSolver:
    def __init__(self, ep_chassis, ep_gimbal, start_cell=(0, 0), exploration_zone=None):
        self.ep_chassis = ep_chassis
        self.ep_gimbal = ep_gimbal
        self.current_cell = start_cell
        self.current_heading_degrees = 0
        self.visited = set([start_cell])
        self.path_stack = [start_cell]
        self.walls = {}
        self.exploration_zone = exploration_zone
        if self.exploration_zone:
            x_min, x_max, y_min, y_max = self.exploration_zone
            print(f"--- Maze boundary is pre-defined: x({x_min}-{x_max}), y({y_min}-{y_max}) ---")

    def explore(self):
        """Main exploration loop using DFS."""
        plot_maze(self.walls, self.current_cell, self.visited, "Starting Exploration")
        while self.path_stack and not stop_flag:
            print(f"\n--- Position: {self.current_cell}, Heading: {self.current_heading_degrees}° ---")
            scan_results = scan_environment()
            self._scan_and_map(scan_results)
            plot_maze(self.walls, self.current_cell, self.visited, f"Exploring at {self.current_cell}")
            if self._find_and_move_to_next_cell(scan_results):
                continue
            if not self._backtrack():
                break
        print("\nDFS exploration complete.")
        plot_maze(self.walls, self.current_cell, self.visited, "Final Exploration Map")
    
    def _scan_and_map(self, scan_results):
        relative_to_global = {
            0:   {'front': (0, 1), 'left': (-1, 0), 'right': (1, 0)},
            90:  {'front': (1, 0), 'left': (0, 1), 'right': (0, -1)},
            -90: {'front': (-1, 0), 'left': (0, -1), 'right': (0, 1)},
            180: {'front': (0, -1), 'left': (1, 0), 'right': (-1, 0)},
        }
        for direction, is_open in scan_results.items():
            if not is_open:
                dx, dy = relative_to_global[self.current_heading_degrees][direction]
                neighbor_cell = (self.current_cell[0] + dx, self.current_cell[1] + dy)
                self.walls[tuple(sorted((self.current_cell, neighbor_cell)))] = 'Wall'

    def _find_and_move_to_next_cell(self, scan_results):
        check_order = ['front', 'right', 'left']
        relative_to_heading = {
            'front': self.current_heading_degrees,
            'left': self.current_heading_degrees - 90,
            'right': self.current_heading_degrees + 90,
        }
        heading_to_coord = {0: (0, 1), 90: (1, 0), -90: (-1, 0), 180: (0, -1)}
        for direction in check_order:
            if scan_results[direction]:
                target_heading = normalize_angle(relative_to_heading[direction])
                dx, dy = heading_to_coord[target_heading]
                target_cell = (self.current_cell[0] + dx, self.current_cell[1] + dy)
                is_valid_move = target_cell not in self.visited
                if is_valid_move and self.exploration_zone:
                    x_min, x_max, y_min, y_max = self.exploration_zone
                    if not (x_min <= target_cell[0] <= x_max and y_min <= target_cell[1] <= y_max):
                        print(f"Move to {target_cell} is outside the defined zone. Skipping.")
                        continue
                if is_valid_move:
                    print(f"Found valid neighbor {target_cell}. Moving {direction}...")
                    turn_to_angle(self.ep_chassis, self.ep_gimbal, target_heading)
                    self.current_heading_degrees = target_heading
                    move_straight_60cm(self.ep_chassis, self.current_heading_degrees)
                    self.path_stack.append(target_cell)
                    self.visited.add(target_cell)
                    self.current_cell = target_cell
                    return True
        return False

    def _backtrack(self):
        print("No unvisited cells found. Backtracking...")
        if len(self.path_stack) <= 1:
            print("Returned to start. Exploration finished.")
            return False
        self.path_stack.pop()
        previous_cell = self.path_stack[-1]
        dx = previous_cell[0] - self.current_cell[0]
        dy = previous_cell[1] - self.current_cell[1]
        backtrack_heading = normalize_angle(math.degrees(math.atan2(dx, dy)))
        print(f"Backtracking from {self.current_cell} to {previous_cell}")
        turn_to_angle(self.ep_chassis, self.ep_gimbal, backtrack_heading)
        self.current_heading_degrees = backtrack_heading
        move_straight_60cm(self.ep_chassis, self.current_heading_degrees)
        self.current_cell = previous_cell
        return True
        
    # << ADDED >>: ฟังก์ชันเสริมสำหรับหาทิศทางไปยังช่องข้างๆ
    @staticmethod
    def _get_direction_to_neighbor(current_cell, target_cell):
        dx, dy = target_cell[0] - current_cell[0], target_cell[1] - current_cell[1]
        # Map coordinate change (dx, dy) to heading in degrees
        if dx == 0 and dy == 1: return 0    # North
        if dx == 1 and dy == 0: return 90   # East
        if dx == 0 and dy == -1: return 180  # South
        if dx == -1 and dy == 0: return -90  # West
        return None

    # << ADDED >>: ฟังก์ชันหลักสำหรับเดินทางไปยังเป้าหมาย
    def go_to_goal(self, target_cell):
        if target_cell not in self.visited:
            print(f"[ERROR] Target cell {target_cell} has not been explored. Cannot navigate.")
            return
        print(f"\n--- Pathfinding from current position {self.current_cell} to goal {target_cell} ---")
        path = self._find_path_bfs(self.current_cell, target_cell)
        if path:
            print(f"Path found: {path}")
            self._navigate_path(path)
        else:
            print(f"Could not find a path from {self.current_cell} to {target_cell}.")
            
    # << ADDED >>: อัลกอริทึมหาเส้นทางที่สั้นที่สุด (BFS)
    def _find_path_bfs(self, start, end):
        if start == end: return [start]
        queue = deque([(start, [start])])
        visited_for_pathfinding = {start}
        coord_map = [(0, 1), (1, 0), (0, -1), (-1, 0)] # N, E, S, W
        while queue:
            current_pos, path = queue.popleft()
            for dx, dy in coord_map:
                neighbor = (current_pos[0] + dx, current_pos[1] + dy)
                if neighbor == end:
                    # Check for wall before declaring success
                    if not tuple(sorted((current_pos, neighbor))) in self.walls:
                        return path + [neighbor]
                is_wall = tuple(sorted((current_pos, neighbor))) in self.walls
                if neighbor in self.visited and neighbor not in visited_for_pathfinding and not is_wall:
                    visited_for_pathfinding.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append((neighbor, new_path))
        return None

    # << ADDED >>: ฟังก์ชันสำหรับสั่งให้หุ่นยนต์เคลื่อนที่ตามเส้นทางที่คำนวณได้
    def _navigate_path(self, path):
        print("\n--- Starting Navigation Along Path ---")
        for i in range(len(path) - 1):
            current_cell = path[i]
            next_cell = path[i+1]
            print(f"Navigating from {current_cell} to {next_cell}")
            plot_maze(self.walls, current_cell, self.visited, title=f"Navigating to {path[-1]}")
            
            target_heading = self._get_direction_to_neighbor(current_cell, next_cell)
            if target_heading is not None:
                turn_to_angle(self.ep_chassis, self.ep_gimbal, target_heading)
                self.current_heading_degrees = target_heading
                move_straight_60cm(self.ep_chassis, self.current_heading_degrees)
                self.current_cell = next_cell # Update current cell after moving
            else:
                print(f"[ERROR] Cannot determine direction from {current_cell} to {next_cell}.")
                return
        
        plot_maze(self.walls, self.current_cell, self.visited, title=f"Arrived at Goal {path[-1]}!")
        print(f"--- Arrived at Goal: {self.current_cell} ---")


# --- 6. Main Execution Block ---

if __name__ == "__main__":
    
    # ++++++++++++++ USER SETTINGS ++++++++++++++
    START_CELL = (1, 0) 
    MAZE_BOUNDS = (0, 2, 0, 2) 
    # << REVISED >>: กำหนดเป้าหมายที่ต้องการไปหลังจากสำรวจเสร็จ
    TARGET_GOAL = (1, 0) 
    # +++++++++++++++++++++++++++++++++++++++++++++

    ep_robot = None
    try:
        # -- Initialize Robot --
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")

        ep_chassis = ep_robot.chassis
        ep_sensor = ep_robot.sensor
        ep_gimbal = ep_robot.gimbal
        ep_sensor_adaptor = ep_robot.sensor_adaptor

        # -- Start Sensor Subscriptions --
        ep_sensor.sub_distance(freq=20, callback=sub_tof_handler)
        ep_chassis.sub_attitude(freq=20, callback=sub_imu_handler)
        time.sleep(1)

        # -- Start IR Sensor Thread --
        ir_reader = threading.Thread(target=read_ir_thread, args=(ep_sensor_adaptor,), daemon=True)
        ir_reader.start()
        time.sleep(1)
        
        print("Robot connected and sensors are running.")
        print("Press 'ESC' in the console to stop the program.")

        # -- Initialize Solver --
        solver = MazeSolver(ep_chassis, ep_gimbal, start_cell=START_CELL, exploration_zone=MAZE_BOUNDS)
        
        # << REVISED >>: แบ่งการทำงานเป็น 2 เฟส
        # Phase 1: Exploration
        print("\n--- PHASE 1: EXPLORING THE MAZE ---")
        solver.explore()
        
        # Phase 2: Navigation to Goal
        if not stop_flag:
            print(f"\n--- PHASE 2: NAVIGATING TO GOAL {TARGET_GOAL} ---")
            time.sleep(2) # หยุดพักเล็กน้อย
            solver.go_to_goal(TARGET_GOAL)

        print("\n--- Mission Complete ---")

    except Exception as e:
        print(f"\n--- An error occurred: {e} ---")
        import traceback
        traceback.print_exc()

    finally:
        stop_flag = True
        time.sleep(0.2)
        
        if ep_robot:
            print("Cleaning up and closing connection.")
            # ส่ง drive_speed อีกครั้งเพื่อให้แน่ใจว่าหยุดสนิทก่อนปิด
            try: ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
            except: pass
            ep_sensor.unsub_distance()
            ep_chassis.unsub_attitude()
            ep_robot.close()
        
        print("Saving final map to 'final_maze_map.png'")
        _fig.savefig("final_maze_map.png", dpi=300)
        finalize_show()
        
        print("Program closed successfully.")