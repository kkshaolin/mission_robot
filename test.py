import time                  # ใช้สำหรับหน่วงเวลา time.sleep()
import threading             # ใช้สำหรับสร้าง thread ทำงานพร้อมกัน
import msvcrt                # ใช้สำหรับตรวจสอบปุ่มกดบน Windows
import matplotlib.pyplot as plt  # ใช้สำหรับวาดกราฟ/แผนที่
import numpy as np           # ใช้สำหรับคำนวณเชิงตัวเลข
from robomaster import robot # นำเข้าโมดูลควบคุมหุ่นยนต์ RoboMaster

# ===================== Plotting =====================
plt.ion()                    # เปิด interactive mode ของ matplotlib ให้วาดกราฟแบบเรียลไทม์
_fig, _ax = plt.subplots(figsize=(8, 8))  # สร้าง figure และ axes สำหรับวาดแผนที่
CELL_SIZE = 0.6              # ขนาดของแต่ละช่องในแผนที่ (เมตร)

# ฟังก์ชันวาดแผนที่ maze แบบเรียลไทม์
def plot_maze(current_cell, visited, path_history, title="Real-time Maze Exploration"):
    ax = _ax              # ใช้ axes ที่สร้างไว้
    ax.clear()             # ล้างกราฟเดิมก่อนวาดใหม่
    # วาดช่องที่เคยไปแล้ว
    for (x, y) in visited:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                   facecolor='lightgray', edgecolor='gray'))
    # วาดเส้นทางที่หุ่นยนต์เดิน
    if len(path_history) > 1:
        path_x, path_y = zip(*path_history)   # แยก x, y
        ax.plot(path_x, path_y, 'b-o', markersize=4, label='Path')
    # วาดตำแหน่งปัจจุบันของหุ่นยนต์
    cx, cy = current_cell
    ax.plot(cx, cy, 'ro', markersize=12, label='Robot')
    # ตั้งขอบเขตของแผนที่ให้อัตโนมัติ
    all_x = [c[0] for c in visited] or [0]
    all_y = [c[1] for c in visited] or [0]
    ax.set_xlim(min(all_x) - 1.5, max(all_x) + 1.5)
    ax.set_ylim(min(all_y) - 1.5, max(all_y) + 1.5)
    ax.set_aspect('equal', adjustable='box')  # ให้แกน x,y เท่ากัน
    ax.grid(True)                             # เปิด grid
    ax.set_title(title)                        # ตั้งชื่อแผนที่
    ax.legend()                                # แสดง legend
    plt.pause(0.05)                            # อัปเดตกราฟแบบเรียลไทม์

# ปิด interactive mode และแสดงกราฟแบบเต็ม
def finalize_plot():
    plt.ioff()
    plt.show()

# ===================== Globals =====================
tof_cm = 999.0             # ระยะหน้า (TOF sensor) เริ่มต้น
ir_right_cm = 999.0        # ระยะด้านขวา (IR sensor) เริ่มต้น
ir_left_cm = 999.0         # ระยะด้านซ้าย (IR sensor) เริ่มต้น
last_value_right = 0.0     # ค่า IR ขวาล่าสุด ใช้กรองค่า
last_value_left = 0.0      # ค่า IR ซ้ายล่าสุด ใช้กรองค่า

sensor_lock = threading.Lock()  # ล็อกสำหรับ access sensor variables
stop_event = threading.Event()  # event สำหรับหยุดโปรแกรม

# Thresholds และค่าคงที่
FRONT_WALL_THRESHOLD = 35.0   # ระยะหน้าเป็นทางตัน
SIDE_WALL_THRESHOLD = 30.0    # ระยะด้านข้างสำหรับเช็คทาง
TARGET_WALL_DIST = 15.0       # ระยะที่ต้องการให้หุ่นยนต์อยู่จากผนัง
FORWARD_SPEED = 0.3           # ความเร็วเดินหน้า
TURN_SPEED = 50               # ความเร็วหมุน

# ตารางแปลงค่า ADC ของ IR เป็นระยะเซนติเมตร
CALIBRA_TABLE_IR_RIGHT = {615:5,605:10,415:15,335:20,275:25,255:30}
CALIBRA_TABLE_IR_LEFT = {680:5,420:10,300:15,235:20,210:25,175:30}

# ฟังก์ชัน low-pass filter ใช้กรองสัญญาณ IR
def single_lowpass_filter(new_value, last_value, alpha=0.8):
    return alpha * new_value + (1-alpha) * last_value

# แปลงค่า ADC เป็น cm โดยใช้ table
def adc_to_cm(adc_value, table):
    points = sorted(table.items(), key=lambda x: x[0], reverse=True)  # เรียงจากสูงไปต่ำ
    if adc_value >= points[0][0]: return float(points[0][1])         # ถ้าเกินสูงสุด
    if adc_value <= points[-1][0]: return float(points[-1][1])       # ถ้าน้อยสุด
    for i in range(len(points)-1):
        x1,y1 = points[i]
        x2,y2 = points[i+1]
        if x2 <= adc_value <= x1:                                     # หา segment
            return float(y1 + (adc_value-x1)*(y2-y1)/(x2-x1))        # linear interpolation
    return float("nan")                                               # ถ้าไม่เจอ

# ===================== MazeRunner =====================
class MazeRunner:
    def __init__(self, conn_type="ap"):
        print("🔌 Connecting to robot...")
        self.ep_robot = robot.Robot()             # สร้าง object หุ่นยนต์
        self.ep_robot.initialize(conn_type=conn_type)  # initialize หุ่นยนต์
        self.ep_chassis = self.ep_robot.chassis   # chassis สำหรับควบคุมการเคลื่อนที่
        self.ep_sensor = self.ep_robot.sensor     # sensor object
        self.ep_sensor_adaptor = self.ep_robot.sensor_adaptor  # adaptor สำหรับอ่าน ADC

        # Map tracking
        self.current_x_m = 0.0                      # ตำแหน่ง x จริง (เมตร)
        self.current_y_m = 0.0                      # ตำแหน่ง y จริง (เมตร)
        self.visited_cells = set([(0,0)])           # เซตเก็บช่องที่เคยไป
        self.path_history = [(0,0)]                 # ลำดับเส้นทาง
        print("✅ Robot connected!")

        # Subscribe TOF sensor
        self.ep_sensor.sub_distance(freq=20, callback=self._tof_cb)

        # Start threads สำหรับ sensor และ status
        threading.Thread(target=self._sensor_loop, daemon=True).start()
        threading.Thread(target=self._status_loop, daemon=True).start()

    # Callback TOF
    def _tof_cb(self, sub_info):
        global tof_cm
        with sensor_lock:
            tof_cm = sub_info[0]/10.0  # แปลง mm -> cm

    # Loop อ่านค่า IR sensor
    def _sensor_loop(self):
        global ir_right_cm, ir_left_cm, last_value_right, last_value_left
        while not stop_event.is_set():
            ir_right_raw = self.ep_sensor_adaptor.get_adc(id=1, port=1)  # อ่าน IR ขวา
            ir_left_raw = self.ep_sensor_adaptor.get_adc(id=2, port=1)   # อ่าน IR ซ้าย
            ir_right_filtered = single_lowpass_filter(ir_right_raw, last_value_right)  # กรองค่า
            ir_left_filtered = single_lowpass_filter(ir_left_raw, last_value_left)
            last_value_right, last_value_left = ir_right_filtered, ir_left_filtered
            with sensor_lock:
                ir_right_cm = adc_to_cm(ir_right_filtered, CALIBRA_TABLE_IR_RIGHT)  # แปลง cm
                ir_left_cm = adc_to_cm(ir_left_filtered, CALIBRA_TABLE_IR_LEFT)
            time.sleep(0.05)  # หน่วง loop 50ms

    # Loop แสดงสถานะ
    def _status_loop(self):
        while not stop_event.is_set():
            with sensor_lock:
                t, r, l = tof_cm, ir_right_cm, ir_left_cm
            print(f"TOF:{t:5.1f} L:{l:5.1f} R:{r:5.1f} Pos:({self.current_x_m:5.2f},{self.current_y_m:5.2f}) \r", end="")
            time.sleep(0.1)

    # ===================== Movement =====================
    # หมุนหุ่นยนต์
    def turn(self, angle_deg):
        self.ep_chassis.move(x=0, y=0, z=-angle_deg, z_speed=TURN_SPEED).wait_for_completed()
        time.sleep(0.2)

    # เดินตามผนังซ้าย
    def follow_left_wall(self):
        with sensor_lock:
            l_dist, f_dist = ir_left_cm, tof_cm
        error = (l_dist - TARGET_WALL_DIST)/100.0          # คำนวณความต่างจากระยะเป้าหมาย
        turn_rate = max(min(20*error, 20), -20)            # ปรับ turn rate
        self.ep_chassis.drive_speed(x=FORWARD_SPEED, y=0, z=turn_rate, timeout=0.2)

    # ===================== Mapping =====================
    def update_map(self):
        grid_x = round(self.current_x_m / CELL_SIZE)       # แปลงเมตร -> grid
        grid_y = round(self.current_y_m / CELL_SIZE)
        current_cell = (grid_x, grid_y)
        self.visited_cells.add(current_cell)               # เพิ่มช่องที่เคยไป
        if self.path_history[-1] != current_cell:
            self.path_history.append(current_cell)
        plot_maze(current_cell, self.visited_cells, self.path_history)

    # ===================== Maze Solving =====================
    def solve_maze(self):
        print("\n--- 🏁 Start Maze ---")
        try:
            while not stop_event.is_set():
                self.update_map()                           # อัปเดตแผนที่
                with sensor_lock:
                    r_dist, f_dist = ir_right_cm, tof_cm
                    l_dist = ir_left_cm

                if r_dist > SIDE_WALL_THRESHOLD:           # เจอทางขวา
                    print("💡 Right open -> turn right")
                    self.ep_chassis.move(x=0.2, y=0, z=0, xy_speed=0.5).wait_for_completed()
                    self.turn(90)
                elif f_dist < FRONT_WALL_THRESHOLD:        # เจอทางตันหน้า
                    if l_dist > SIDE_WALL_THRESHOLD:      # มีทางซ้าย
                        print("💡 Left open -> turn left")
                        self.turn(-90)
                    else:                                 # ทางตันทุกด้าน
                        print("🛑 Dead end -> turn 180")
                        self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
                        time.sleep(0.3)
                        self.turn(180)
                else:
                    self.follow_left_wall()               # เดินหน้าตามผนังซ้าย
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received")
        finally:
            self.stop_all()

    # ===================== Stop Robot =====================
    def stop_all(self):
        stop_event.set()                                    # ส่งสัญญาณหยุด
        self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
        try: self.ep_sensor.unsub_distance()               # ยกเลิก subscription TOF
        except: pass
        self.ep_robot.close()                               # ปิดหุ่นยนต์
        finalize_plot()                                    # แสดง plot สุดท้าย
        print("Stopped.")

# ===================== Main =====================
if __name__ == "__main__":
    runner = None
    try:
        runner = MazeRunner(conn_type="ap")               # สร้าง MazeRunner

        # Thread ตรวจสอบ ESC key
        def check_exit():
            while not stop_event.is_set():
                if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
                    print("\n[INFO] ESC pressed -> stopping")
                    stop_event.set()
                    break
                time.sleep(0.1)

        threading.Thread(target=check_exit, daemon=True).start()
        runner.solve_maze()                               # เริ่มแก้ Maze
    except Exception as e:
        print(f"\n--- Error: {e} ---")
        import traceback
        traceback.print_exc()
    finally:
        if runner:
            runner.stop_all()                              # ปิดหุ่นยนต์และ plot
