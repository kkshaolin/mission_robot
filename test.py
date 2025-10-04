from robomaster import robot  # นำเข้าไลบรารีสำหรับควบคุมหุ่นยนต์ RoboMaster
import msvcrt  # นำเข้าโมดูลสำหรับตรวจจับการกดปุ่มบน Windows (ใช้สำหรับปุ่ม ESC)
import time  # นำเข้าโมดูลสำหรับจัดการเวลา เช่น delay และ timestamp
import math  # นำเข้าโมดูลคณิตศาสตร์สำหรับคำนวณมุม (atan2, degrees) และระยะทาง
import threading  # นำเข้าโมดูลสำหรับสร้าง thread เพื่อรันงานหลายอย่างพร้อมกัน
import matplotlib.pyplot as plt  # นำเข้าไลบรารีสำหรับวาดกราฟและแสดงแผนที่เขาวงกต
import numpy as np  # นำเข้าไลบรารีสำหรับจัดการอาร์เรย์และการคำนวณเชิงตัวเลข

# ===================== ตัวแปรสถานะและค่าคงที่ระดับ Global =====================
# ตัวแปร flag สำหรับหยุดการทำงานของโปรแกรมทั้งหมด (ใช้ร่วมกันทุก thread)
stop_flag = False  # เริ่มต้นเป็น False (โปรแกรมยังทำงานอยู่)

# --- ตัวแปรสถานะส่วนกลางสำหรับเก็บค่าเซ็นเซอร์และตำแหน่ง ---
# ระยะทางที่วัดได้จากเซ็นเซอร์ ToF (Time of Flight) หน่วยเป็นเซนติเมตร
tof_distance_cm = 999.0  # เริ่มต้นเป็นค่าสูงมาก (หมายถึงไม่มีสิ่งกีดขวาง)
# มุม yaw ปัจจุบันของหุ่นยนต์ (จาก IMU sensor) หน่วยเป็นองศา
current_yaw = 0.0  # เริ่มต้นที่ 0 องศา (หันหน้าไปทางเหนือ)

current_x = 0.0  # ตำแหน่งพิกัด x ปัจจุบันจาก position sensor (หน่วยเมตร)
current_y = 0.0  # ตำแหน่งพิกัด y ปัจจุบันจาก position sensor (หน่วยเมตร)

# ระยะทางจากกำแพงด้านซ้ายและขวา (จากเซ็นเซอร์ IR) หน่วยเป็นเซนติเมตร
ir_left_cm = 999.0  # ระยะจากกำแพงซ้าย เริ่มต้นเป็นค่าสูงมาก
ir_right_cm = 999.0  # ระยะจากกำแพงขวา เริ่มต้นเป็นค่าสูงมาก

# ค่าล่าสุดของเซ็นเซอร์ IR หลังผ่าน filter (ใช้สำหรับ low-pass filter)
last_value_left = 0  # ค่าล่าสุดของ IR ซ้าย
last_value_right = 0  # ค่าล่าสุดของ IR ขวา

# --- ค่าคงที่สำหรับการสำรวจเขาวงกตและการเคลื่อนที่ ---
# ระยะเวลาในการสแกนสภาพแวดล้อม (หน่วยวินาที)
SCAN_DURATION_S = 0.2  # รอ 0.2 วินาที เพื่อให้เซ็นเซอร์อัปเดตค่า
# ระยะทาง ToF ที่ถือว่ามีกำแพงอยู่ด้านหน้า (เซนติเมตร)
TOF_WALL_THRESHOLD_CM = 60  # ถ้าระยะน้อยกว่า 60 cm = มีกำแพง
# ระยะทาง IR ที่ถือว่ามีกำแพงอยู่ด้านข้าง (เซนติเมตร)
IR_WALL_THRESHOLD_CM = 29  # ถ้าระยะน้อยกว่า 29 cm = มีกำแพง
# ตำแหน่งเริ่มต้นของหุ่นยนต์ในเขาวงกต (grid cell)
START_CELL = (1, 1)  # เริ่มที่ช่อง (1,1)
# ขอบเขตขั้นต่ำของแผนที่เขาวงกต (x_min, y_min)
MAP_MIN_BOUNDS = (1, 1)  # พิกัดต่ำสุดที่ยอมให้เดินได้
# ขอบเขตสูงสุดของแผนที่เขาวงกต (x_max, y_max)
MAP_MAX_BOUNDS = (4, 4)  # พิกัดสูงสุดที่ยอมให้เดินได้
# ระยะทางระหว่างช่องในเขาวงกต (หน่วยเมตร)
NODE_DISTANCE = 0.6  # แต่ละช่องห่างกัน 60 เซนติเมตร
# ระยะที่เริ่มหลีกเลี่ยงกำแพงด้านข้าง (เซนติเมตร)
WALL_AVOID_THRESHOLD_CM = 10.0  # ถ้าใกล้กำแพงน้อยกว่า 10 cm ให้หลีกเลี่ยง
# ความเร็วในการขยับหลีกเลี่ยงกำแพงด้านข้าง (เมตร/วินาที)
WALL_AVOID_SPEED_Y = 0.05  # ขยับด้านข้างช้าๆ ที่ 0.05 m/s
# ความเร็วในการหมุนสูงสุด (องศา/วินาที)
TURN_SPEED_Z = 60  # หมุนได้สูงสุด 60 องศา/วินาที

# --- ตัวแปรสถานะสำหรับ DFS Maze Exploration ---
# เก็บข้อมูลแผนที่: key=ตำแหน่ง(x,y), value=set ของทิศทางที่เปิด (degrees)
maze_map = {}  # dictionary เปล่าสำหรับเริ่มต้น
# เก็บตำแหน่งทั้งหมดที่หุ่นยนต์เคยไปแล้ว
visited_nodes = set()  # set เปล่าสำหรับเริ่มต้น
# stack สำหรับเก็บเส้นทางการเดินตาม DFS (Last In First Out)
path_stack = []  # list เปล่าสำหรับเริ่มต้น
# เก็บพิกัดของกำแพงทั้งหมด (tuple ของคู่ตำแหน่ง)
walls = set()  # set เปล่าสำหรับเก็บกำแพง
# ตำแหน่งปัจจุบันของหุ่นยนต์ในเขาวงกต (x, y)
current_pos = START_CELL  # เริ่มต้นที่ตำแหน่งเริ่มต้น (1, 1)
# ทิศทางหัวหุ่นยนต์ปัจจุบัน: 0=เหนือ, 90=ตะวันออก, -90=ตะวันตก, 180/-180=ใต้
current_heading_degrees = 0  # เริ่มต้นหันหน้าไปทางเหนือ (0 องศา)

# --- ตัวแปรสำหรับ A* Algorithm (ยังไม่ได้ใช้ในโค้ดนี้) ---
# รายการจุดเป้าหมาย (checkpoint) ที่ต้องการไป — เพิ่มพิกัด (x,y) ลงใน list นี้
target_marks = []  # list เปล่า (ยังไม่มีเป้าหมาย)
# ติดตามเส้นทางจริงที่หุ่นยนต์เดิน (เพื่อบันทึกหรือวิเคราะห์)
traveled_path = []  # list เปล่าสำหรับเก็บประวัติการเดิน

# --- ค่า Gain สำหรับ PID Controller ---
# ค่า Proportional gain สำหรับการควบคุมการหมุน (turning)
Kp_turn = 2.5  # ยิ่งสูงยิ่งตอบสนองเร็ว แต่อาจเกิด overshoot

# --- ตารางเทียบค่า ADC เป็นระยะทาง CM สำหรับ IR Sensors ---
# ตาราง calibration สำหรับเซ็นเซอร์ IR ด้านขวา: {ค่า ADC: ระยะทาง cm}
calibra_table_ir_right = {615: 5, 605: 10, 415: 15, 335: 20, 275: 25, 255: 30}
# ค่า ADC สูง = ระยะใกล้, ค่า ADC ต่ำ = ระยะไกล
# ตาราง calibration สำหรับเซ็นเซอร์ IR ด้านซ้าย: {ค่า ADC: ระยะทาง cm}
calibra_table_ir_left = {680: 5, 420: 10, 300: 15, 235: 20, 210: 25, 175: 30}
# แต่ละเซ็นเซอร์มี characteristic curve ต่างกัน จึงต้อง calibrate แยก

# --- ตัวแปรสำหรับวาดกราฟด้วย Matplotlib ---
# สร้าง figure และ axes สำหรับแสดงแผนที่เขาวงกต
_fig, _ax = plt.subplots(figsize=(6, 6))  # สร้างกราฟขนาด 6x6 นิ้ว

# ===================== ฟังก์ชันสำหรับวาดกราฟแผนที่ =====================
def plot_maze(walls_to_plot, visited_to_plot, path_stack_to_plot, current_cell_to_plot, title="Maze Exploration"):
    """
    วาดสถานะปัจจุบันของการสำรวจเขาวงกต
    
    Parameters:
        walls_to_plot: set ของกำแพงที่จะวาด
        visited_to_plot: set ของช่องที่เคยไปแล้ว
        path_stack_to_plot: list ของเส้นทาง DFS
        current_cell_to_plot: tuple ตำแหน่งปัจจุบันของหุ่นยนต์
        title: ชื่อกราฟ
    """
    # ล้างกราฟเดิมออกเพื่อเตรียมวาดใหม่
    _ax.clear()
    
    # กำหนดขอบเขตของเขาวงกตที่จะวาด (x_min, x_max, y_min, y_max)
    MAZE_BOUNDS_PLOT = (0, 4, 0, 4)  # แสดงตั้งแต่ 0 ถึง 4 ทั้ง x และ y
    x_min, x_max = MAZE_BOUNDS_PLOT[0], MAZE_BOUNDS_PLOT[1]  # แยกค่า x
    y_min, y_max = MAZE_BOUNDS_PLOT[2], MAZE_BOUNDS_PLOT[3]  # แยกค่า y

    # วนลูปวาดช่องที่เคยไปแล้วเป็นสีฟ้าอ่อน
    for x, y in visited_to_plot:  # วนลูปทุกช่องใน visited_nodes
        # เพิ่มสี่เหลี่ยมจัตุรัสขนาด 1x1 ที่มีจุดกึ่งกลางอยู่ที่ (x, y)
        _ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightcyan', edgecolor='none', zorder=0))
        # (x-0.5, y-0.5) = มุมล่างซ้าย, (1, 1) = ขนาด, lightcyan = สีฟ้าอ่อน, zorder=0 = วาดที่ชั้นล่างสุด

    # วนลูปวาดกำแพงเป็นเส้นสีดำหนา
    for wall in walls_to_plot:  # วนลูปทุกกำแพงใน set
        # กำแพงแต่ละอันเก็บเป็นคู่ตำแหน่ง ((x1,y1), (x2,y2))
        (x1, y1), (x2, y2) = wall  # แยกพิกัดของทั้งสองจุด
        
        # ถ้า y1 == y2 แสดงว่ากำแพงอยู่แนวตั้ง (แกน x ต่างกัน)
        if y1 == y2:
            x_mid = (x1 + x2) / 2.0  # หาจุดกึ่งกลางแกน x
            # วาดเส้นแนวตั้งจากล่างถึงบนของช่อง
            _ax.plot([x_mid, x_mid], [y1 - 0.5, y1 + 0.5], color='k', linewidth=4)
            # [x_mid, x_mid] = พิกัด x เริ่มและจบเท่ากัน (เส้นตั้ง), color='k' = สีดำ
        
        # ถ้า x1 == x2 แสดงว่ากำแพงอยู่แนวนอน (แกน y ต่างกัน)
        elif x1 == x2:
            y_mid = (y1 + y2) / 2.0  # หาจุดกึ่งกลางแกน y
            # วาดเส้นแนวนอนจากซ้ายถึงขวาของช่อง
            _ax.plot([x1 - 0.5, x1 + 0.5], [y_mid, y_mid], color='k', linewidth=4)
            # [y_mid, y_mid] = พิกัด y เริ่มและจบเท่ากัน (เส้นนอน)
    
    # วาดเส้นทางที่เดินมา (path stack) เป็นเส้นสีน้ำเงินพร้อมจุด
    if len(path_stack_to_plot) > 1:  # ต้องมีอย่างน้อย 2 จุดถึงจะวาดเส้นได้
        # แยกพิกัด x และ y ออกมาเป็น tuple แยก
        path_x, path_y = zip(*path_stack_to_plot)  # zip(*) = unpack list of tuples
        # วาดเส้นทางและจุดด้วยสีน้ำเงิน
        _ax.plot(path_x, path_y, 'b-o', markersize=4, zorder=1)
        # 'b-o' = เส้นสีน้ำเงิน (-) พร้อมจุดกลม (o), zorder=1 = วาดบนช่องที่เคยไป

    # วาดตำแหน่งหุ่นยนต์ปัจจุบันเป็นจุดสีแดงขนาดใหญ่
    cx, cy = current_cell_to_plot  # แยกพิกัด x, y ของหุ่นยนต์
    _ax.plot(cx, cy, 'ro', markersize=12, label='Robot', zorder=2)
    # 'ro' = จุดสีแดง (red, circle), markersize=12 = ขนาดใหญ่, zorder=2 = วาดบนสุด

    # ตั้งค่าขอบเขตของกราฟ
    _ax.set_xlim(x_min - 0.5, x_max + 0.5)  # แกน x จาก -0.5 ถึง 4.5
    _ax.set_ylim(y_min - 0.5, y_max + 0.5)  # แกน y จาก -0.5 ถึง 4.5
    
    # ตั้งค่าให้กราฟเป็นสี่เหลี่ยมจัตุรัส (aspect ratio 1:1)
    _ax.set_aspect('equal', adjustable='box')  # 1 หน่วย x = 1 หน่วย y
    
    # เพิ่มเส้น grid สีเทาอ่อนเพื่อให้เห็นช่องชัดเจน
    _ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)
    # which='both' = แสดง grid ทั้ง major และ minor
    
    # กำหนดตำแหน่ง tick marks บนแกน x และ y
    _ax.set_xticks(np.arange(x_min - 0.5, x_max + 1.5, 1))  # ทุก 1 หน่วย
    _ax.set_yticks(np.arange(y_min - 0.5, y_max + 1.5, 1))  # ทุก 1 หน่วย
    
    # ซ่อน tick labels (ตัวเลขบนแกน) เพราะไม่จำเป็น
    _ax.set_xticklabels([])  # ไม่แสดงตัวเลขแกน x
    _ax.set_yticklabels([])  # ไม่แสดงตัวเลขแกน y
    
    # ตั้งชื่อกราฟด้านบน
    _ax.set_title(title)  # ใช้ title ที่ส่งเข้ามา
    
    # อัปเดตการแสดงผล (pause เพื่อให้กราฟ refresh)
    plt.pause(0.1)  # หยุด 0.1 วินาที เพื่อให้ matplotlib วาดกราฟ

def finalize_show():
    """
    ปิด interactive mode และแสดงกราฟสุดท้ายแบบ blocking
    เรียกใช้เมื่อการสำรวจเสร็จสิ้นแล้ว
    """
    # ปิด interactive mode ของ matplotlib
    plt.ioff()  # หยุดการ auto-refresh
    
    # แสดงกราฟและรอให้ผู้ใช้ปิดหน้าต่าง (blocking call)
    plt.show()  # หน้าต่างจะเปิดค้างจนกว่าจะปิดด้วยตนเอง

# ===================== ฟังก์ชันสำหรับจัดการเซ็นเซอร์ =====================
def sub_tof_handler(sub_info):
    """
    Callback function สำหรับรับข้อมูลจากเซ็นเซอร์ ToF (Time of Flight)
    ฟังก์ชันนี้จะถูกเรียกอัตโนมัติเมื่อมีข้อมูลใหม่จากเซ็นเซอร์
    
    Parameters:
        sub_info: list ของข้อมูลเซ็นเซอร์ [ระยะทาง_mm]
    """
    global tof_distance_cm  # ประกาศใช้ตัวแปร global
    # แปลงค่าจาก mm เป็น cm (หารด้วย 10)
    tof_distance_cm = sub_info[0] / 10.0  # sub_info[0] = ระยะทางหน่วย mm

def sub_imu_handler(attitude_info):
    """
    Callback function สำหรับรับข้อมูลมุม yaw จาก IMU (Inertial Measurement Unit)
    ฟังก์ชันนี้จะถูกเรียกอัตโนมัติเมื่อมีข้อมูลใหม่จาก IMU
    
    Parameters:
        attitude_info: list ของข้อมูลทิศทาง [yaw, pitch, roll]
    """
    global current_yaw  # ประกาศใช้ตัวแปร global
    # attitude_info[0] คือมุม yaw (หมุนรอบแกนตั้ง) หน่วยองศา
    current_yaw = attitude_info[0]  # บันทึกมุม yaw ปัจจุบัน

def sub_position_handler(position_info):
    """
    Callback function สำหรับรับข้อมูลตำแหน่งจาก position sensor
    ฟังก์ชันนี้จะถูกเรียกอัตโนมัติเมื่อมีข้อมูลใหม่จากเซ็นเซอร์ตำแหน่ง
    
    Parameters:
        position_info: list ของข้อมูลตำแหน่ง [x, y, z]
    """
    global current_x, current_y  # ประกาศใช้ตัวแปร global
    current_x = position_info[0]  # ตำแหน่ง x (เมตร)
    current_y = position_info[1]  # ตำแหน่ง y (เมตร)
    # ไม่ใช้ position_info[2] (z) เพราะหุ่นยนต์เดินบนพื้นราบ

def single_lowpass_filter(new_value, last_value, alpha=0.8):
    """
    Low-pass filter แบบง่าย (Exponential Moving Average) สำหรับลด noise ของเซ็นเซอร์
    ช่วยให้ค่าที่อ่านได้มีความเสถียรมากขึ้น ไม่กระเด้งไปมา
    
    Parameters:
        new_value: ค่าใหม่ที่อ่านได้จากเซ็นเซอร์
        last_value: ค่าเดิมที่ผ่าน filter แล้ว
        alpha: น้ำหนักของค่าใหม่ (0-1), ยิ่งสูงยิ่งตอบสนองเร็ว แต่ filter น้อย
    
    Returns:
        float: ค่าที่ผ่าน filter แล้ว
    """
    # สูตร Exponential Moving Average: output = alpha * new + (1-alpha) * old
    return alpha * new_value + (1.0 - alpha) * last_value
    # alpha=0.8 หมายถึงให้น้ำหนักค่าใหม่ 80%, ค่าเก่า 20%

def adc_to_cm(adc_value, table):
    """
    แปลงค่า ADC จากเซ็นเซอร์ IR เป็นระยะทาง cm โดยใช้ linear interpolation
    เนื่องจากความสัมพันธ์ระหว่าง ADC และระยะทางไม่เป็นเส้นตรง จึงต้องใช้ตาราง
    
    Parameters:
        adc_value: ค่า ADC ที่อ่านได้จากเซ็นเซอร์ (ตัวเลขดิบ)
        table: ตาราง calibration ในรูปแบบ {ADC: cm}
    
    Returns:
        float: ระยะทางเป็นเซนติเมตร
    """
    # เรียงข้อมูลในตารางจาก ADC มากไปน้อย (ระยะใกล้ไปไกล)
    points = sorted(table.items(), key=lambda x: x[0], reverse=True)
    # sorted(..., reverse=True) = เรียงจากมากไปน้อย
    
    # ถ้า ADC มากกว่าหรือเท่ากับค่าสูงสุดในตาราง = วัตถุใกล้มากที่สุด
    if adc_value >= points[0][0]: 
        return float(points[0][1])  # คืนระยะใกล้สุดในตาราง
    
    # ถ้า ADC น้อยกว่าหรือเท่ากับค่าต่ำสุดในตาราง = วัตถุไกลมากที่สุด
    if adc_value <= points[-1][0]: 
        return float(points[-1][1])  # คืนระยะไกลสุดในตาราง
    
    # กรณีทั่วไป: ทำ Linear interpolation ระหว่างจุดที่ใกล้เคียง
    for i in range(len(points) - 1):  # วนลูปทุกคู่จุดติดกัน
        x1, y1 = points[i]      # จุดแรก (ADC สูงกว่า = ระยะน้อยกว่า)
        x2, y2 = points[i+1]    # จุดถัดไป (ADC ต่ำกว่า = ระยะมากกว่า)
        
        # ถ้า ADC อยู่ระหว่าง x2 และ x1
        if x2 <= adc_value <= x1:
            # คำนวณค่าระหว่างจุดด้วยสูตร linear interpolation
            # y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            return float(y1 + (adc_value - x1) * (y2 - y1) / (x2 - x1))
            # (adc_value - x1) / (x2 - x1) = ตำแหน่งสัมพัทธ์ระหว่าง x1 และ x2
    
    # ถ้าไม่อยู่ในช่วงใดๆ เลย (ไม่ควรเกิดขึ้น) คืนค่า error
    return 999.0  # 999.0 = ค่า error หมายถึงไม่สามารถแปลงได้

def read_ir_thread(ep_sensor_adaptor):
    """
    Thread function สำหรับอ่านค่าเซ็นเซอร์ IR อย่างต่อเนื่อง
    รันแยกจาก main thread เพื่อไม่ให้รบกวนการทำงานหลัก
    อ่านค่า IR ทุก 50ms และอัปเดตตัวแปร global
    
    Parameters:
        ep_sensor_adaptor: object สำหรับอ่านค่า ADC จากเซ็นเซอร์
    """
    global ir_right_cm, ir_left_cm, last_value_right, last_value_left  # ประกาศใช้ตัวแปร global
    
    # วนลูปจนกว่าจะมีสัญญาณหยุด (stop_flag = True)
    while not stop_flag:  # ตราบใดที่ stop_flag ยังเป็น False
        # อ่านค่า ADC จากเซ็นเซอร์ IR ขวา (id=2, port=2)
        ir_right_adc = ep_sensor_adaptor.get_adc(id=2, port=2)
        # id=2 = เซ็นเซอร์ที่ 2, port=2 = พอร์ต ADC
        
        # อ่านค่า ADC จากเซ็นเซอร์ IR ซ้าย (id=1, port=2)
        ir_left_adc = ep_sensor_adaptor.get_adc(id=1, port=2)
        # id=1 = เซ็นเซอร์ที่ 1
        
        # ผ่าน low-pass filter เพื่อลด noise และทำให้ค่าเสถียร
        ir_right_filtered = single_lowpass_filter(ir_right_adc, last_value_right)
        ir_left_filtered = single_lowpass_filter(ir_left_adc, last_value_left)
        
        # บันทึกค่าที่ผ่าน filter แล้วสำหรับรอบถัดไป
        last_value_right, last_value_left = ir_right_filtered, ir_left_filtered
        
        # แปลงค่า ADC ที่ผ่าน filter แล้วเป็นระยะทาง cm
        ir_right_cm = adc_to_cm(ir_right_filtered, calibra_table_ir_right)
        ir_left_cm = adc_to_cm(ir_left_filtered, calibra_table_ir_left)
        
        # รอ 50 มิลลิวินาที (0.05 วินาที) ก่อนอ่านค่าครั้งถัดไป
        time.sleep(0.05)  # ความถี่การอ่าน = 20 Hz

# ===================== ฟังก์ชันสำหรับการเคลื่อนที่ =====================
def normalize_angle(angle):
    """
    ปรับมุมให้อยู่ในช่วง -180 ถึง 180 องศา
    ช่วยให้การคำนวณมุมและการเปรียบเทียบทำได้ง่ายขึ้น
    
    Parameters:
        angle: มุมองศาที่ต้องการปรับ (สามารถเป็นค่าอะไรก็ได้)
    
    Returns:
        float: มุมที่ปรับแล้ว (-180 < angle <= 180)
    """
    # ถ้ามุมมากกว่า 180 องศา ให้ลบ 360 ซ้ำๆ จนกว่าจะอยู่ในช่วง
    while angle > 180: 
        angle -= 360  # เช่น 270° → -90°, 450° → 90°
    
    # ถ้ามุมน้อยกว่าหรือเท่ากับ -180 องศา ให้บวก 360 ซ้ำๆ จนกว่าจะอยู่ในช่วง
    while angle <= -180: 
        angle += 360  # เช่น -270° → 90°, -450° → -90°
    
    return angle  # คืนมุมที่อยู่ในช่วง -180 ถึง 180

def turn_to_angle(ep_chassis, ep_gimbal, target_angle):
    """
    หมุนหุ่นยนต์ไปยังมุมเป้าหมายโดยใช้ PID control (แบบ P-only)
    หมุนช้าๆ เมื่อใกล้เป้าหมาย และเร็วเมื่อห่างจากเป้าหมาย
    
    Parameters:
        ep_chassis: object ควบคุมการเคลื่อนที่ของหุ่นยนต์
        ep_gimbal: object ควบคุม gimbal (กล้อง)
        target_angle: มุมเป้าหมายที่ต้องการหมุนไป (degrees)
    """
    global current_yaw  # ประกาศใช้ตัวแปร global
    
    # ปรับมุมเป้าหมายให้อยู่ในช่วงมาตรฐาน (-180 ถึง 180)
    target_angle = normalize_angle(target_angle)
    
    print(f"กำลังหมุนไปที่ {target_angle}°")  # แสดงข้อความแจ้งเตือน
    
    # วนลูปจนกว่าจะหมุนถึงมุมเป้าหมาย
    while not stop_flag:  # ตราบใดที่ยังไม่มีสัญญาณหยุด
        # คำนวณความต่างของมุม (error) = เป้าหมาย - ปัจจุบัน
        angle_error = normalize_angle(target_angle - current_yaw)
        
        # ถ้า error น้อยกว่า 2 องศา ถือว่าถึงเป้าหมายแล้ว
        if abs(angle_error) < 2.0: 
            break  # ออกจาก loop
        
        # คำนวณความเร็วการหมุนด้วย Proportional control
        # turn_speed = Kp * error (แต่จำกัดไม่ให้เกิน ±TURN_SPEED_Z)
        turn_speed = max(min(angle_error * Kp_turn, TURN_SPEED_Z), -TURN_SPEED_Z)
        # max(..., -TURN_SPEED_Z) = ไม่ให้ติดลบเกิน -60
        # min(..., TURN_SPEED_Z) = ไม่ให้เกิน +60
        
        # สั่งหมุนหุ่นยนต์ (z = แกนหมุน)
        ep_chassis.drive_speed(x=0, y=0, z=turn_speed)
        # x=0 = ไม่เดินหน้า, y=0 = ไม่เดินข้าง, z=turn_speed = หมุน
        
        # รอ 20 มิลลิวินาที ก่อนคำนวณใหม่ (ความถี่ควบคุม 50 Hz)
        time.sleep(0.02)
    
    # หยุดการหมุน (ตั้งความเร็วทุกแกนเป็น 0)
    ep_chassis.drive_speed(x=0, y=0, z=0)
    
    # รีเซ็ต gimbal กลับสู่ตำแหน่งกลาง (pitch=0, yaw=0)
    ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=100, yaw_speed=100)
    # pitch=0 = ไม่เงยหรือก้ม, yaw=0 = กลับตรงกลาง
    
    # รอให้หุ่นยนต์หยุดนิ่ง (stabilize)
    time.sleep(0.5)  # พัก 0.5 วินาที

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
        ep_chassis.drive_speed(x=x_speed, y=y_speed, z=0)
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


# ===================== ฟังก์ชันสำหรับตรรกะ DFS =====================
def scan_environment():
    """
    สแกนสภาพแวดล้อมรอบๆ หุ่นยนต์เพื่อหาทิศทางที่เดินได้
    ตรวจสอบด้านหน้า (ToF), ด้านซ้าย (IR), และด้านขวา (IR)
    
    Returns:
        dict: {'front': bool, 'left': bool, 'right': bool}
              True = เดินได้ (ไม่มีกำแพง), False = มีกำแพง
    """
    global tof_distance_cm, ir_left_cm, ir_right_cm  # ประกาศใช้ตัวแปร global
    
    # เตรียม dictionary สำหรับเก็บผลการสแกน
    open_paths = {'front': False, 'left': False, 'right': False}
    # เริ่มต้นทุกทิศทางเป็น False (สมมติว่ามีกำแพงก่อน)
    
    # รอให้เซ็นเซอร์อัปเดตค่า (ให้เวลาเซ็นเซอร์อ่านค่าใหม่)
    time.sleep(SCAN_DURATION_S)  # หยุด 0.2 วินาที
    
    # ตรวจสอบด้านหน้าด้วย ToF sensor
    if tof_distance_cm > TOF_WALL_THRESHOLD_CM:  # ถ้าระยะมากกว่า 60 cm
        open_paths['front'] = True  # ด้านหน้าเปิด (เดินได้)
    
    # ตรวจสอบด้านซ้ายด้วย IR sensor
    if ir_left_cm > IR_WALL_THRESHOLD_CM:  # ถ้าระยะมากกว่า 29 cm
        open_paths['left'] = True  # ด้านซ้ายเปิด (เดินได้)
    
    # ตรวจสอบด้านขวาด้วย IR sensor
    if ir_right_cm > IR_WALL_THRESHOLD_CM:  # ถ้าระยะมากกว่า 29 cm
        open_paths['right'] = True  # ด้านขวาเปิด (เดินได้)
    
    # แสดงผลการสแกนบน console
    print(f"ผลสแกน: หน้า: {tof_distance_cm:.1f} cm | ซ้าย: {ir_left_cm:.1f} cm | ขวา: {ir_right_cm:.1f} cm")
    
    return open_paths  # คืนค่า dictionary ผลการสแกน

def get_target_coordinates(from_pos, heading_deg):
    """
    คำนวณพิกัดเป้าหมายเมื่อเดินไปในทิศทางที่กำหนด (1 ช่อง)
    ใช้สำหรับหาตำแหน่งของช่องเพื่อนบ้าน
    
    Parameters:
        from_pos: ตำแหน่งเริ่มต้น tuple (x, y)
        heading_deg: ทิศทางที่ต้องการเดิน (degrees)
                     0=เหนือ, 90=ตะวันออก, -90=ตะวันตก, ±180=ใต้
    
    Returns:
        tuple: พิกัดเป้าหมาย (x, y)
    """
    # แยกพิกัด x, y จาก tuple
    x, y = from_pos
    
    # ปรับมุมให้อยู่ในช่วงมาตรฐาน -180 ถึง 180
    heading = normalize_angle(heading_deg)
    
    # ตรวจสอบทิศทางและคำนวณตำแหน่งใหม่
    if heading == 0:  # เหนือ (North)
        return (x, y + 1)  # เดินไปทาง y+1
    elif heading == 90:  # ตะวันออก (East)
        return (x + 1, y)  # เดินไปทาง x+1
    elif heading == -90:  # ตะวันตก (West)
        return (x - 1, y)  # เดินไปทาง x-1
    elif abs(heading) == 180:  # ใต้ (South) - อาจเป็น 180 หรือ -180
        return (x, y - 1)  # เดินไปทาง y-1
    
    # ถ้าไม่ตรงกรณีใดๆ (ไม่ควรเกิดขึ้น) คืนตำแหน่งเดิม
    return from_pos

def get_direction_to_neighbor(from_cell, to_cell):
    """
    คำนวณมุมทิศทางจากช่องหนึ่งไปยังอีกช่องหนึ่ง
    ใช้สำหรับหาทิศทางที่ต้องหมุนเพื่อเดินไปยังช่องเพื่อนบ้าน
    
    Parameters:
        from_cell: ตำแหน่งต้นทาง tuple (x, y)
        to_cell: ตำแหน่งปลายทาง tuple (x, y)
    
    Returns:
        float: มุมทิศทาง (degrees) ที่ normalize แล้ว
    """
    # คำนวณผลต่างของพิกัด
    dx = to_cell[0] - from_cell[0]  # ความต่างแกน x (ปลายทาง - ต้นทาง)
    dy = to_cell[1] - from_cell[1]  # ความต่างแกน y (ปลายทาง - ต้นทาง)
    
    # ใช้ atan2 คำนวณมุม (radian) แล้วแปลงเป็นองศา
    # atan2(dx, dy) จะให้มุมจากแกน y (เหนือ = 0°)
    # degrees(...) แปลง radian เป็นองศา
    # normalize_angle(...) ปรับให้อยู่ในช่วง -180 ถึง 180
    return normalize_angle(math.degrees(math.atan2(dx, dy)))

def turn_and_move(ep_chassis, ep_gimbal, target_heading):
    """
    หมุนไปทิศทางเป้าหมาย (ถ้าจำเป็น) แล้วเดินหน้าไป 60 cm
    เป็นฟังก์ชันรวมสำหรับการเคลื่อนที่ไปยังช่องถัดไป
    
    Parameters:
        ep_chassis: object ควบคุมการเคลื่อนที่
        ep_gimbal: object ควบคุม gimbal
        target_heading: ทิศทางเป้าหมายที่ต้องการหัน (degrees)
    """
    global current_heading_degrees  # ประกาศใช้ตัวแปร global
    
    # ตรวจสอบว่าต้องหมุนหรือไม่ (ถ้าต่างมากกว่า 1 องศา)
    if abs(normalize_angle(target_heading - current_heading_degrees)) > 1:
        # ถ้าต่างมากกว่า 1 องศา = ต้องหมุน
        # หมุนไปยังทิศทางเป้าหมาย
        turn_to_angle(ep_chassis, ep_gimbal, target_heading)
        
        # อัปเดตทิศทางปัจจุบันหลังหมุนเสร็จ
        current_heading_degrees = target_heading
    
    # เดินหน้าไป 60 cm พร้อมรักษาทิศทาง
    move_straight_60cm(ep_chassis, target_heading)

def map_current_cell():
    """
    สแกนและบันทึกข้อมูลแผนที่ของช่องปัจจุบัน
    ทำการสแกนทิศทางที่เปิด และบันทึกกำแพงที่พบลงใน maze_map และ walls
    """
    global maze_map, walls, current_pos, current_heading_degrees
    # ประกาศใช้ตัวแปร global หลายตัว
    
    print(f"ช่อง {current_pos} ยังไม่ได้สำรวจ กำลังสแกน...")  # แจ้งเตือน
    
    # สแกนสภาพแวดล้อมรอบๆ
    scan_results = scan_environment()
    # scan_results = {'front': bool, 'left': bool, 'right': bool}
    
    # สร้าง set สำหรับเก็บทิศทางที่เปิด (absolute heading)
    open_headings = set()  # set เปล่าสำหรับเก็บมุมทิศทาง
    
    # กำหนดมุมสัมพัทธ์สำหรับแต่ละทิศทาง (relative to current heading)
    relative_moves = {'left': -90, 'front': 0, 'right': 90}
    # ซ้าย = -90° (หมุนซ้าย), หน้า = 0° (ตรงไป), ขวา = 90° (หมุนขวา)
    
    # วนลูปตรวจสอบแต่ละทิศทาง (left, front, right)
    for move_key, is_open in scan_results.items():  # วนทุกคู่ key-value
        # คำนวณมุม relative (เทียบกับทิศทางปัจจุบันของหุ่นยนต์)
        relative_angle = relative_moves[move_key]  # ดึงมุมสัมพัทธ์
        
        # แปลงเป็นมุม absolute (เทียบกับทิศเหนือ)
        absolute_heading = normalize_angle(current_heading_degrees + relative_angle)
        # เช่น ถ้าหุ่นยนต์หันหน้าไป 90° (ตะวันออก) และเลี้ยวซ้าย (-90°)
        # absolute = 90 + (-90) = 0° (เหนือ)
        
        # ถ้าทิศทางนี้เปิดอยู่ (ไม่มีกำแพง)
        if is_open:
            # เพิ่มทิศทางนี้เข้า set ของทิศทางที่เปิด
            open_headings.add(absolute_heading)
        else:
            # ถ้าปิด (มีกำแพง)
            # คำนวณตำแหน่งของช่องเพื่อนบ้านในทิศทางนั้น
            neighbor_cell = get_target_coordinates(current_pos, absolute_heading)
            
            # เพิ่มกำแพงระหว่างช่องปัจจุบันกับช่องเพื่อนบ้าน
            # ใช้ sorted เพื่อให้กำแพง (A,B) และ (B,A) เป็นอันเดียวกัน
            walls.add(tuple(sorted((current_pos, neighbor_cell))))
            # sorted(...) ทำให้ ((1,1), (2,1)) และ ((2,1), (1,1)) เป็นอันเดียวกัน
    
    # บันทึกข้อมูลแผนที่ของช่องนี้ลงใน maze_map
    maze_map[current_pos] = open_headings
    # maze_map[(1,1)] = {0, 90} หมายถึงช่อง (1,1) เปิดไปทางเหนือและตะวันออก
    
    print(f"สร้างแผนที่ช่อง {current_pos} มีทิศทางที่เปิด: {sorted(list(open_headings))}")
    # แสดงทิศทางที่เปิดแบบเรียงลำดับ

def find_and_move_to_next_cell(ep_chassis, ep_gimbal):
    """
    หาช่องเพื่อนบ้านที่ยังไม่เคยไปและเคลื่อนที่ไปยังช่องนั้น
    ตามหลักการ DFS: ตรวจสอบตามลำดับ ซ้าย -> หน้า -> ขวา
    
    Parameters:
        ep_chassis: object ควบคุมการเคลื่อนที่
        ep_gimbal: object ควบคุม gimbal
    
    Returns:
        bool: True = เคลื่อนที่สำเร็จไปยังช่องใหม่, False = ไม่มีทางไปต่อ (ทางตัน)
    """
    global visited_nodes, path_stack, current_pos, current_heading_degrees
    # ประกาศใช้ตัวแปร global
    
    # ลำดับการตรวจสอบตาม DFS: ซ้าย (-90°), หน้า (0°), ขวา (90°)
    search_order_relative = [-90, 0, 90]
    # ตรวจสอบซ้ายก่อน แล้วหน้า แล้วขวา (Left-First DFS)
    
    # วนลูปตรวจสอบแต่ละทิศทางตามลำดับ DFS
    for angle in search_order_relative:  # วนลูป -90, 0, 90
        # คำนวณทิศทางเป้าหมาย (absolute heading)
        target_heading = normalize_angle(current_heading_degrees + angle)
        # แปลงมุมสัมพัทธ์เป็นมุมสัมบูรณ์
        
        # ตรวจสอบว่าทิศทางนี้เปิดอยู่หรือไม่ (อยู่ใน maze_map)
        if target_heading in maze_map.get(current_pos, set()):
            # maze_map.get(current_pos, set()) คืน set ของทิศทางที่เปิด
            # ถ้าไม่มีข้อมูล คืน set เปล่า
            
            # คำนวณตำแหน่งของช่องเพื่อนบ้านในทิศทางนั้น
            target_cell = get_target_coordinates(current_pos, target_heading)
            
            # ตรวจสอบว่าช่องเป้าหมายอยู่ในขอบเขตของแผนที่หรือไม่
            min_x, min_y = MAP_MIN_BOUNDS  # ขอบเขตต่ำสุด (1, 1)
            max_x, max_y = MAP_MAX_BOUNDS  # ขอบเขตสูงสุด (4, 4)
            
            # ถ้าช่องเป้าหมายอยู่นอกขอบเขต ข้ามทิศทางนี้
            if not (min_x <= target_cell[0] <= max_x and min_y <= target_cell[1] <= max_y):
                continue  # ข้ามไปทิศทางถัดไป

            # ตรวจสอบว่าช่องนี้เคยไปแล้วหรือยัง
            if target_cell not in visited_nodes:
                # ถ้ายังไม่เคยไป = พบช่องใหม่
                print(f"พบเพื่อนบ้านที่ยังไม่เคยไป {target_cell} กำลังเคลื่อนที่...")
                
                # หมุนและเคลื่อนที่ไปยังช่องเป้าหมาย
                turn_and_move(ep_chassis, ep_gimbal, target_heading)
                
                # เพิ่มช่องนี้เข้าไปใน visited_nodes (บันทึกว่าเคยมาแล้ว)
                visited_nodes.add(target_cell)
                
                # เพิ่มช่องนี้เข้า path stack (เส้นทาง DFS)
                path_stack.append(target_cell)
                
                # อัปเดตตำแหน่งปัจจุบัน
                current_pos = target_cell
                
                # คืนค่า True = เคลื่อนที่สำเร็จ
                return True
    
    # ถ้าไม่มีทิศทางไหนที่ไปได้ (ทางตัน) คืนค่า False
    return False

def backtrack(ep_chassis, ep_gimbal):
    """
    ย้อนรอยกลับไปยังช่องก่อนหน้าใน path stack
    ใช้เมื่อเจอทางตันและต้อง backtrack ตามหลักการ DFS
    
    Parameters:
        ep_chassis: object ควบคุมการเคลื่อนที่
        ep_gimbal: object ควบคุม gimbal
    
    Returns:
        bool: True = ย้อนรอยสำเร็จ, False = กลับถึงจุดเริ่มต้นแล้ว (ไม่มีที่ให้ย้อนอีก)
    """
    global path_stack, current_pos  # ประกาศใช้ตัวแปร global
    
    print("เจอทางตัน กำลังย้อนรอย (Backtracking)...")  # แจ้งเตือน
    
    # ถ้า stack มีแค่จุดเดียว (จุดเริ่มต้น) = กลับมาที่จุดเริ่มต้นแล้ว
    if len(path_stack) <= 1:
        print("กลับมาที่จุดเริ่มต้น การสำรวจสิ้นสุด")
        return False  # คืนค่า False = ไม่สามารถย้อนรอยได้อีก

    # ลบตำแหน่งปัจจุบันออกจาก stack (pop)
    path_stack.pop()  # ลบตัวสุดท้ายออก
    
    # ดึงตำแหน่งก่อนหน้า (จุดที่จะย้อนรอยไป)
    previous_cell = path_stack[-1]  # [-1] = ตัวสุดท้ายใน list
    
    # คำนวณทิศทางที่ต้องหมุนเพื่อย้อนรอยกลับ
    backtrack_heading = get_direction_to_neighbor(current_pos, previous_cell)
    # หาทิศทางจากตำแหน่งปัจจุบันไปยังตำแหน่งก่อนหน้า
    
    print(f"กำลังย้อนรอยจาก {current_pos} ไปยัง {previous_cell}")  # แสดงข้อความ
    
    # หมุนและเคลื่อนที่ย้อนกลับ
    turn_and_move(ep_chassis, ep_gimbal, backtrack_heading)
    
    # อัปเดตตำแหน่งปัจจุบัน
    current_pos = previous_cell
    
    # คืนค่า True = ย้อนรอยสำเร็จ
    return True

# ===================== Main Execution Block (โปรแกรมหลัก) =====================
if __name__ == '__main__':
    # บล็อกนี้จะทำงานเมื่อรันไฟล์นี้โดยตรง (ไม่ใช่ import)
    
    # สร้าง object หุ่นยนต์
    ep_robot = robot.Robot()  # สร้าง instance ของ RoboMaster
    
    # เชื่อมต่อกับหุ่นยนต์ผ่าน WiFi Access Point
    ep_robot.initialize(conn_type="ap")
    # "ap" = Access Point mode (หุ่นยนต์เป็น WiFi hotspot)

    # ดึง object ย่อยสำหรับควบคุมแต่ละส่วนของหุ่นยนต์
    ep_chassis = ep_robot.chassis        # ควบคุมการเคลื่อนที่ (ล้อ)
    ep_sensor = ep_robot.sensor          # เซ็นเซอร์หลัก (ToF, IMU)
    ep_gimbal = ep_robot.gimbal          # ควบคุม gimbal (แกนกล้อง)
    ep_sensor_adaptor = ep_robot.sensor_adaptor  # อ่านค่า ADC จาก IR sensors

    # ตั้งชื่อหน้าต่างกราฟ matplotlib
    _fig.canvas.manager.set_window_title("Maze Map")  # ชื่อหน้าต่าง

    # --- Subscribe เซ็นเซอร์และเริ่ม Thread ---
    # Subscribe เซ็นเซอร์ ToF ความถี่ 20 Hz (20 ครั้งต่อวินาที)
    ep_sensor.sub_distance(freq=20, callback=sub_tof_handler)
    # callback=sub_tof_handler = เรียกฟังก์ชันนี้เมื่อมีข้อมูลใหม่
    
    # Subscribe ข้อมูล attitude (yaw, pitch, roll) จาก IMU ความถี่ 20 Hz
    ep_chassis.sub_attitude(freq=20, callback=sub_imu_handler)
    # callback=sub_imu_handler = เรียกฟังก์ชันนี้เมื่อมีข้อมูลใหม่

    # Subscribe ข้อมูลตำแหน่ง (x, y, z) จาก position sensor ความถี่ 20 Hz
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    # callback=sub_position_handler = เรียกฟังก์ชันนี้เมื่อมีข้อมูลใหม่
    
    # สร้าง thread สำหรับอ่านค่า IR อย่างต่อเนื่อง
    ir_reader = threading.Thread(target=read_ir_thread, args=(ep_sensor_adaptor,), daemon=True)
    # target=read_ir_thread = ฟังก์ชันที่จะรันใน thread
    # args=(ep_sensor_adaptor,) = ส่งพารามิเตอร์เข้าฟังก์ชัน
    # daemon=True = thread จะปิดตามเมื่อโปรแกรมหลักปิด
    
    # เริ่มรัน thread
    ir_reader.start()  # เริ่มการทำงานของ thread
    
    # รอ 1 วินาทีให้ทุกอย่างเริ่มต้นเสร็จสิ้น
    time.sleep(1)  # ให้เวลาเซ็นเซอร์เริ่มส่งข้อมูล

    # --- Initialize DFS State (เตรียมสถานะเริ่มต้นสำหรับ DFS) ---
    # ตั้งตำแหน่งเริ่มต้น
    current_pos = START_CELL  # ตั้งเป็น (1, 1)
    
    # เพิ่มตำแหน่งเริ่มต้นเข้า visited_nodes (บันทึกว่ามาแล้ว)
    visited_nodes.add(current_pos)
    
    # เพิ่มตำแหน่งเริ่มต้นเข้า path_stack (เริ่มต้น stack)
    path_stack.append(current_pos)
    
    print("เริ่มต้นการสำรวจเขาวงกตแบบ DFS...")  # แจ้งเริ่มต้น
    
    # --- Main Exploration Loop (ลูปหลักสำหรับสำรวจเขาวงกต) ---
    # วนลูปจนกว่า path_stack จะว่าง (สำรวจเสร็จแล้ว) หรือมีคำสั่งหยุด
    # while path_stack and not stop_flag:  # เงื่อนไข: stack ไม่ว่าง และ ไม่มีสัญญาณหยุด
    for i in range(1):  # ทดสอบแค่ 1 รอบ (ปกติควรใช้ while loop ด้านบน)
        # ตรวจสอบว่ามีการกดปุ่ม ESC หรือไม่ (สำหรับหยุดฉุกเฉิน)
        if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
            # kbhit() = ตรวจสอบว่ามีการกดปุ่ม, getch() = อ่านปุ่มที่กด
            # b'\x1b' = รหัส ASCII ของปุ่ม ESC
            print("กดปุ่ม ESC กำลังหยุดการทำงาน...")
            break  # ออกจากลูป

        # วาดแผนที่ปัจจุบัน (อัปเดตกราฟ)
        plot_maze(walls, visited_nodes, path_stack, current_pos)
        
        # แสดงสถานะปัจจุบันบน console
        print(f"\nตำแหน่ง: {current_pos}, ทิศทาง: {current_heading_degrees}°")

        # ถ้าช่องปัจจุบันยังไม่ได้สร้างแผนที่ (ยังไม่เคยสแกน)
        if current_pos not in maze_map:
            # สแกนและบันทึกแผนที่ช่องนี้
            map_current_cell()
        
        # พยายามหาและเคลื่อนที่ไปยังช่องถัดไป (ตาม DFS)
        if find_and_move_to_next_cell(ep_chassis, ep_gimbal):
            # ถ้า True = เคลื่อนที่ไปยังช่องใหม่สำเร็จ
            continue  # กลับไปเริ่มลูปใหม่ (สำรวจช่องใหม่)
        
        # ถ้าไม่มีทางไปต่อ (ทางตัน) ทำการ backtrack
        if not backtrack(ep_chassis, ep_gimbal):
            # ถ้า False = ไม่สามารถ backtrack ได้ (กลับถึงจุดเริ่มต้นแล้ว)
            break  # ออกจากลูป (การสำรวจเสร็จสิ้น)

    # แสดงข้อความเสร็จสิ้น
    print("\nการสำรวจ DFS เสร็จสมบูรณ์")  # แจ้งเตือน
    
    # วาดแผนที่สุดท้าย
    plot_maze(walls, visited_nodes, path_stack, current_pos, "Final Map")
    # ใช้ชื่อ "Final Map" สำหรับกราฟสุดท้าย

    print("กำลังทำความสะอาดและปิดการเชื่อมต่อ...")  # แจ้งเตือน
    
    # ตั้งค่า flag เพื่อหยุด thread อ่าน IR
    stop_flag = True  # บอกให้ thread หยุดทำงาน
    
    # รอให้ thread ปิดอย่างสมบูรณ์
    time.sleep(0.2)  # ให้เวลา thread ออกจาก loop
    
    # หยุดการเคลื่อนที่ของหุ่นยนต์
    ep_chassis.drive_speed(x=0, y=0, z=0)  # ตั้งความเร็วทุกแกนเป็น 0
    
    # ยกเลิก subscription เซ็นเซอร์ ToF
    ep_sensor.unsub_distance()  # หยุดรับข้อมูลจาก ToF
    
    # ยกเลิก subscription attitude (IMU)
    ep_chassis.unsub_attitude()  # หยุดรับข้อมูลจาก IMU

    # ยกเลิก subscription position sensor
    ep_chassis.unsub_position()  # หยุดรับข้อมูลตำแหน่ง
    
    # ปิดการเชื่อมต่อกับหุ่นยนต์
    ep_robot.close()  # ตัดการเชื่อมต่อและปล่อยทรัพยากร
    
    # แสดงกราฟสุดท้ายและรอให้ผู้ใช้ปิดหน้าต่าง
    finalize_show()  # เรียกฟังก์ชันแสดงกราฟแบบ blocking