import numpy as np
import matplotlib.pyplot as plt
from robomaster import robot

# ++++++++++++++ USER SETTINGS: แก้ไขค่าตรงนี้ได้เลย ++++++++++++++
    # กำหนดจุดเริ่มต้น (x, y)
START_CELL = (0, 0) 
    
    # กำหนดขอบเขตแผนที่ 6x6 (x_min, x_max, y_min, y_max)
    # เช่น (0, 5, 0, 5) หมายถึง x จาก 0 ถึง 5 และ y จาก 0 ถึง 5
MAZE_BOUNDS = (0, 5, 0, 5)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ===================== Plotting & MazeSolver Classes =====================
def mapping_and_localization():
    _fig, _ax = plt.subplots(figsize=(6, 6)) # ขนาดเเมพ 6x6

    def plot_maze(walls, current_cell, visited, title="Maze Exploration"): # ส่วนของการกำหนดเเมพ 
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


    def finalize_show():  # ส่วนของการโชว์เเมพ เมื่อทำงานเสร็จเเล้ว
        plt.ioff()
        plt.show()

        @staticmethod
        def _get_discretized_orientation(yaw_deg): # ต้องการค่า Yaw เพื่อใช้ในการสร้างเเมพว่า หันไปทางไหน
            yaw = (yaw_deg + 360) % 360
            if yaw >= 315 or yaw < 45: return 0
            elif 45 <= yaw < 135: return 3
            elif 135 <= yaw < 225: return 2
            else: return 1

        @staticmethod
        def _get_direction_to_neighbor(current_cell, target_cell): # ใช้หาทิศทางของเซลล์ต่อไป
            dx, dy = target_cell[0] - current_cell[0], target_cell[1] - current_cell[1]
            if dy == 1: return 0
            if dx == 1: return 1
            if dy == -1: return 2
            if dx == -1: return 3
            return None
        
    def _scan_and_map(self, cell, sensor_data): # ส่วนของการสเเกนเเละสร้างเเมพ
        orientation_map = {0:{"L":3,"F":0,"R":1}, 1:{"L":0,"F":1,"R":2}, 2:{"L":1,"F":2,"R":3}, 3:{"L":2,"F":3,"R":0}}
        coord_map = {0:(0,1), 1:(1,0), 2:(0,-1), 3:(-1,0)}
        map_statuses = {'F':'Occupied' if sensor_data['F_cm']<self.WALL_THRESHOLD else 'Free', 'L':'Occupied' if sensor_data['L_digital']==1 else 'Free', 'R':'Occupied' if sensor_data['R_digital']==1 else 'Free'}
        for move_key in ["L", "F", "R"]:
            direction = orientation_map[self.current_orientation][move_key]
            dx, dy = coord_map[direction]
            neighbor_cell = (cell[0] + dx, cell[1] + dy)
            if neighbor_cell not in self.visited:
                    self.occupancy_grid[neighbor_cell] = map_statuses[move_key]
            if map_statuses[move_key] == 'Occupied':
                    self.walls[tuple(sorted((cell, neighbor_cell)))] = 'Wall'
            return map_statuses

mapping_and_localization()
            
    


        
                 