# exploration_phase2.py
import time
import json
import math
import heapq  # ใช้สำหรับ Priority Queue ใน A*
from robomaster import robot
import numpy as np

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

    # ===================== [ฟังก์ชันใหม่ - ปรับแก้สำหรับเซนเซอร์ซ้าย] =====================
    def follow_wall_to_next_node(self, cell_size_m):
        """
        เคลื่อนที่ตามกำแพง "ด้านซ้าย" ไปยัง Node ถัดไปโดยใช้ PID 2 ตัว (เวอร์ชันสมบูรณ์)
        """
        global current_x, current_y, ir_left_cm, ir_right_cm, tof_distance_cm 
        
        print(f"Action: Following LEFT wall for {cell_size_m} m")

        # === [สำคัญ] สร้าง PID controllers โดยใช้ค่า GAINS ที่ดีบักมา ===
        # หมายเหตุ: ค่า Gains เหล่านี้มาจากไฟล์ debug_wall_align.py ของคุณ
        pid_angle = PIDController(Kp=14.0, Ki=0.0001, Kd=0.0002, setpoint=0)
        pid_dist = PIDController(Kp=0.01, Ki=0.0, Kd=0.002, setpoint=TARGET_WALL_DISTANCE_CM)

        sx, sy = current_x, current_y
        t0 = time.time()
        max_duration_s = 15

        while time.time() - t0 < max_duration_s:
            # 1. Safety Stop: ตรวจสอบกำแพงด้านหน้าก่อนเสมอ
            if tof_distance_cm < 20: # ถ้าเจอวัตถุในระยะ 20 cm
                print("\n[!!!] SAFETY STOP: ตรวจพบสิ่งกีดขวางด้านหน้า!")
                self.stop()
                return # ออกจากฟังก์ชันทันที
            # 2. อ่านค่าเซนเซอร์ล่าสุดจากตัวแปร global
            ir_front = ir_left_cm 
            ir_rear = ir_right_cm
            
            # 3. คำนวณ Error (ตรรกะเดียวกับในไฟล์ดีบัก)
            angle_error = ir_front - ir_rear # Error สำหรับการปรับมุม
            
            current_dist_avg = (ir_front + ir_rear) / 2.0
            dist_error = TARGET_WALL_DISTANCE_CM - current_dist_avg # Error สำหรับการปรับระยะห่าง
            # หมายเหตุ: สูตร dist_error นี้ถูกต้องสำหรับแกน y ของหุ่น (ค่าลบ -> เลื่อนซ้าย)

            # 4. คำนวณค่าการปรับแก้จาก PID
            z_speed = pid_angle.compute(angle_error)
            y_speed = pid_dist.compute(dist_error)

            # 5. จำกัดค่า Output
            z_speed = float(np.clip(z_speed, -MAX_Z_SPEED, MAX_Z_SPEED))
            y_speed = float(np.clip(y_speed, -MAX_Y_SPEED, MAX_Y_SPEED))
            
            # 6. สั่งการ Chassis
            self.ep_chassis.drive_speed(x=BASE_FORWARD_SPEED_WF, y=y_speed, z=z_speed, timeout=0.1)

            # 7. ตรวจสอบเงื่อนไขการหยุด
            dist_traveled = math.hypot(current_x - sx, current_y - sy)
            if dist_traveled >= cell_size_m:
                print("Movement complete.")
                self.stop()
                return
            
            time.sleep(0.02)
        
        print("[WARNING] follow_wall_to_next_node timed out. Stopping robot.")
        self.stop()


# ===================== Global State & Constants =====================
# ตัวแปรสถานะที่ได้จากการ subscribe
current_yaw = 0.0
current_x = 0.0
current_y = 0.0

# ตัวแปรและค่าคงที่สำหรับ Wall Following (จากเฟส 1)
ir_left_cm = 999.0
ir_right_cm = 999.0
last_value_left = 0
last_value_right = 0

CALIBRA_TABLE_IR_FRONT = {249: 10, 216: 15, 139: 20, 117: 25}
CALIBRA_TABLE_IR_REAR = {536: 10, 471: 15, 333: 20, 299: 25}

TARGET_WALL_DISTANCE_CM = 8.0
BASE_FORWARD_SPEED_WF = 0.3  # อาจต้องปรับความเร็วให้เหมาะกับเฟส 2
MAX_Y_SPEED = 0.3
MAX_Z_SPEED = 32.0
NODE_DISTANCE = 0.6


# ===================== Sensor Handling Functions =====================
def sub_imu_handler(attitude_info):
    """Callback function สำหรับรับข้อมูลมุม yaw จาก IMU"""
    global current_yaw
    current_yaw = attitude_info[0]

def sub_position_handler(position_info):
    """Callback function สำหรับรับข้อมูลตำแหน่ง"""
    global current_x, current_y
    current_x = position_info[0]
    current_y = position_info[1]

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

def read_analog_ir_thread(ep_sensor_adaptor):
    global ir_left_cm, ir_right_cm, last_value_left, last_value_right
    while True: # เปลี่ยนเป็น True เพราะเราจะปิดตอนจบโปรแกรม
        try:
            adc_front_left = ep_sensor_adaptor.get_adc(id=1, port=2)
            adc_rear_left = ep_sensor_adaptor.get_adc(id=2, port=2)
            
            filtered_front = single_lowpass_filter(adc_front_left, last_value_left)
            filtered_rear = single_lowpass_filter(adc_rear_left, last_value_right)
            last_value_left, last_value_right = filtered_front, filtered_rear
            
            ir_left_cm = adc_to_cm(filtered_front, CALIBRA_TABLE_IR_FRONT)
            ir_right_cm = adc_to_cm(filtered_rear, CALIBRA_TABLE_IR_REAR)
        except Exception as e:
            print(f"[ERROR] in IR thread: {e}", end='\r')
            ir_left_cm, ir_right_cm = 999.0, 999.0
        time.sleep(0.02)

# ===================== Movement Functions (จากเฟส 1) =====================
def normalize_angle(angle):
    """ปรับมุมให้อยู่ในช่วง -180 ถึง 180 องศา"""
    while angle > 180: angle -= 360
    while angle <= -180: angle += 360
    return angle


# ===================== A* Pathfinding Algorithm =====================
def heuristic(a, b):
    """คำนวณ Heuristic (Manhattan distance) สำหรับ A*"""
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def get_neighbors(node, maze_map_data):
    """หาเพื่อนบ้านที่สามารถไปได้จากแผนที่"""
    x, y = node
    neighbors = []
    # maze_map_data ถูกแปลง key เป็น string ตอน save, ต้องใช้ str(node)
    open_headings = maze_map_data.get(str(node), [])
    
    for heading in open_headings:
        if heading == 0:   neighbors.append((x, y + 1))  # เหนือ
        elif heading == 90:  neighbors.append((x + 1, y))  # ตะวันออก
        elif heading == -90: neighbors.append((x - 1, y))  # ตะวันตก
        elif abs(heading) == 180: neighbors.append((x, y - 1))  # ใต้
    return neighbors

def a_star_search(maze_map_data, start, goal):
    """
    A* Algorithm เพื่อหาเส้นทางที่สั้นที่สุด
    :param maze_map_data: ข้อมูลแผนที่ที่โหลดมาจาก JSON
    :param start: tuple (x, y) ของจุดเริ่มต้น
    :param goal: tuple (x, y) ของจุดหมาย
    :return: list ของ tuples ที่เป็นเส้นทาง, หรือ None ถ้าไม่พบ
    """
    print(f"\n[A*] กำลังคำนวณเส้นทางจาก {start} -> {goal}...")
    
    frontier = []
    heapq.heappush(frontier, (0, start)) # (priority, item)
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not len(frontier) == 0:
        current_priority, current_node = heapq.heappop(frontier)

        if current_node == goal:
            # Reconstruct path
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()
            print(f"[A*] พบเส้นทาง: {path}")
            return path

        for neighbor in get_neighbors(current_node, maze_map_data):
            new_cost = cost_so_far[current_node] + 1 # cost = 1 for each step
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current_node
    
    print(f"[A*] ไม่พบเส้นทางจาก {start} ไปยัง {goal}")
    return None # Path not found

# ===================== Navigation Logic =====================
def get_heading_to_next_step(current_pos, next_pos):
    """คำนวณมุมที่หุ่นต้องหันไปเพื่อเดินไปยังช่องถัดไป"""
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    # atan2(dx, dy) เพราะแกน y ของหุ่นคือทิศเหนือ (เหมือนในเฟส 1)
    return normalize_angle(math.degrees(math.atan2(dx, dy)))

def follow_path_optimized(controller, path, initial_heading):
    """
    ควบคุมหุ่นยนต์ให้เดินตามเส้นทาง (เวอร์ชันใหม่ที่รวมทางตรงเป็นเส้นเดียว)
    """
    if not path or len(path) < 2:
        print("เส้นทางไม่ถูกต้อง ไม่สามารถเดินได้")
        return initial_heading

    robot_current_heading = initial_heading
    path_index = 0

    while path_index < len(path) - 1:
        current_pos = path[path_index]
        
        # 1. คำนวณทิศทางสำหรับก้าวแรก
        initial_target_heading = get_heading_to_next_step(current_pos, path[path_index + 1])
        
        # 2. มองไปข้างหน้าเพื่อหาว่ามีทางตรงยาวแค่ไหน
        straight_steps = 1
        for i in range(path_index + 1, len(path) - 1):
            # คำนวณทิศทางของก้าวถัดๆ ไป
            next_heading = get_heading_to_next_step(path[i], path[i+1])
            if next_heading == initial_target_heading:
                # ถ้าทิศทางเหมือนเดิม แสดงว่าเป็นทางตรง
                straight_steps += 1
            else:
                # ถ้าทิศทางเปลี่ยน แสดงว่าทางตรงสิ้นสุดแล้ว
                break
        
        # 3. คำนวณระยะทางรวมที่ต้องเดิน
        total_distance = straight_steps * NODE_DISTANCE
        print(f"\nพบทางตรง! จะเดินจาก {current_pos} เป็นระยะทาง {total_distance:.2f} m ({straight_steps} ช่อง)")

        # 4. หันไปยังทิศทางที่ถูกต้อง
        turn_angle = normalize_angle(initial_target_heading - robot_current_heading)
        if abs(turn_angle) > 5.0:
            controller.turn(turn_angle)
        robot_current_heading = normalize_angle(initial_target_heading)

        # 5. สั่งเดินยาวรวดเดียวด้วย Wall Following
        controller.follow_wall_to_next_node(total_distance)
        
        # 6. อัปเดตตำแหน่งใน path ไปยังจุดสุดท้ายของทางตรง
        path_index += straight_steps

    return robot_current_heading

def perform_shooting_action(ep_gimbal, marker_info):
    """จำลองการยิง Marker"""
    print(f"!!! ถึงเป้าหมาย Marker {marker_info['color']} {marker_info['shape']} !!!")
    print("...กำลังเล็ง...")
    ep_gimbal.drive_speed(pitch_speed=50)
    time.sleep(0.5)
    ep_gimbal.drive_speed(pitch_speed=-50)
    time.sleep(0.5)
    ep_gimbal.recenter().wait_for_completed()
    print("ยิงเรียบร้อย!")
    time.sleep(1)


# ===================== Main Execution Block =====================
if __name__ == '__main__':
    # --- 1. โหลดข้อมูลแผนที่ ---
    try:
        with open('map_data.json', 'r') as f:
            map_data = json.load(f)
        
        # แปลง key ที่เป็น string กลับเป็น tuple
        maze_map = map_data['maze_map']
        markers_found = {eval(k): v for k, v in map_data['markers_found'].items()}
        start_node = tuple(map_data['start_node'])

        if not markers_found:
            print("ไม่พบ Marker ในไฟล์ข้อมูล จบการทำงาน")
            exit()
            
        print("โหลดข้อมูลแผนที่และ Marker สำเร็จ!")
        print(f"จุดเริ่มต้น: {start_node}")
        print(f"พบ Marker {len(markers_found)} จุด: {list(markers_found.keys())}")

    except FileNotFoundError:
        print("ไม่พบไฟล์ 'map_data.json'. กรุณารันโค้ดเฟส 1 (exploration.py) ก่อน")
        exit()
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
        exit()

    # --- 2. เชื่อมต่อหุ่นยนต์ ---
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_sensor_adaptor = ep_robot.sensor_adaptor # <<< เพิ่มบรรทัดนี้

    # สร้าง instance ของ Control class
    controller = Control(ep_chassis)

    # Subscribe เพื่อรับค่า yaw และ position
    ep_chassis.sub_attitude(freq=20, callback=sub_imu_handler)
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)

    analog_ir_reader = threading.Thread(target=read_analog_ir_thread, args=(ep_sensor_adaptor,), daemon=True)
    analog_ir_reader.start()

    time.sleep(1) # รอให้ subscriber เริ่มทำงาน
    
    # --- 3. เริ่มกระบวนการนำทาง ---
    robot_current_pos = start_node
    robot_current_heading = 0.0 # เริ่มต้นหันหน้าไปทิศเหนือ (0 องศา)

    # วนลูปสำหรับ Marker แต่ละตัว
    for marker_pos, marker_info in markers_found.items():
        # คำนวณเส้นทางด้วย A*
        path_to_marker = a_star_search(maze_map, robot_current_pos, marker_pos)
    
        if path_to_marker:
            # <<< แก้ไขบรรทัดนี้ ให้เรียกใช้ฟังก์ชันใหม่
            robot_current_heading = follow_path_optimized(controller, path_to_marker, robot_current_heading)
            
            # อัปเดตตำแหน่งปัจจุบันของหุ่น
            robot_current_pos = marker_pos
            # ทำ Action ยิง
            perform_shooting_action(ep_gimbal, marker_info)
        else:
            print(f"ไม่สามารถหาเส้นทางไปยัง Marker ที่ {marker_pos} ได้")

    print("\nภารกิจเสร็จสิ้น! ไปเยือนครบทุก Marker แล้ว")

    # --- 4. ปิดการเชื่อมต่อ ---
    ep_chassis.unsub_attitude()
    ep_chassis.unsub_position()
    ep_robot.close()
    print("โปรแกรมจบการทำงาน")