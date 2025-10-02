import numpy as np
import matplotlib.pyplot as plt
from robomaster import robot

# ===================== Plotting & MazeSolver Classes =====================
_fig, _ax = plt.subplots(figsize=(6, 6))

def plot_maze(walls, current_cell, visited, title="Maze Exploration"):
    # (โค้ดเดิม ไม่มีการเปลี่ยนแปลง)
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


def finalize_show():
    plt.ioff()
    plt.show()

class MazeSolver:
    WALL_THRESHOLD = 50
    CELL_SIZE = 0.60

    # << REVISED: รับ start_cell และ exploration_zone เข้ามาได้ >>
    def __init__(self, ctrl: Control, start_cell=(0, 0), exploration_zone=None):
        self.ctrl = ctrl
        self.visited = set([start_cell])
        self.path_stack = [start_cell]
        self.occupancy_grid = {start_cell: 'Free'}
        self.current_orientation = self._get_discretized_orientation(self.ctrl.get_yaw_deg())
        self.walls = {}
        self.exploration_zone = exploration_zone
        if self.exploration_zone:
            x_min, x_max, y_min, y_max = self.exploration_zone
            print(f"--- Maze boundary is pre-defined: x({x_min}-{x_max}), y({y_min}-{y_max}) ---")

    # (ฟังก์ชัน _get_discretized_orientation และ _get_direction_to_neighbor เหมือนเดิม)
    @staticmethod
    def _get_discretized_orientation(yaw_deg):
        yaw = (yaw_deg + 360) % 360
        if yaw >= 315 or yaw < 45: return 0
        elif 45 <= yaw < 135: return 3
        elif 135 <= yaw < 225: return 2
        else: return 1

    @staticmethod
    def _get_direction_to_neighbor(current_cell, target_cell):
        dx, dy = target_cell[0] - current_cell[0], target_cell[1] - current_cell[1]
        if dy == 1: return 0
        if dx == 1: return 1
        if dy == -1: return 2
        if dx == -1: return 3
        return None


    def explore(self):
        while self.path_stack:
            # << REMOVED: Logic การค้นหาโซนอัตโนมัติถูกเอาออกไปแล้ว >>
            
            current_cell = self.path_stack[-1]
            print(f"\n--- Position: {current_cell}, Orientation: {self.current_orientation} ---")
            sensor_data = self.ctrl.read_all_sensors()
            self._scan_and_map(current_cell, sensor_data)
            
            if self._find_and_move_to_next_cell(current_cell):
                continue

            if not self._backtrack():
                break
        
        print("\nDFS exploration complete.")
        plot_maze(self.walls, self.path_stack[-1], self.visited, "Final Exploration Map")
        
    def _scan_and_map(self, cell, sensor_data):
        # (โค้ดเดิม ไม่มีการเปลี่ยนแปลง)
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


    def _find_and_move_to_next_cell(self, cell):
        # (โค้ดส่วนใหญ่เหมือนเดิม)
        search_order = [(self.current_orientation + 1)%4, self.current_orientation, (self.current_orientation-1+4)%4]
        coord_map = {0:(0,1), 1:(1,0), 2:(0,-1), 3:(-1,0)}
        for direction in search_order:
            dx, dy = coord_map[direction]
            target_cell = (cell[0] + dx, cell[1] + dy)
            
            is_valid_move = (self.occupancy_grid.get(target_cell) == 'Free' and 
                                 target_cell not in self.visited)
            
            if is_valid_move:
                if self.exploration_zone:
                    x_min, x_max, y_min, y_max = self.exploration_zone
                    if not (x_min <= target_cell[0] <= x_max and y_min <= target_cell[1] <= y_max):
                        print(f"Move to {target_cell} is outside the defined zone. Skipping.")
                        continue 

                print(f"Found valid neighbor {target_cell}. Moving...")
                self._turn_to(direction)
                self.ctrl.move_forward_pid(self.CELL_SIZE)
                self.visited.add(target_cell)
                self.path_stack.append(target_cell)
                return True
        return False

    def _backtrack(self):
        # (โค้ดเดิม ไม่มีการเปลี่ยนแปลง)
        print("No unvisited cells found. Backtracking...")
        if len(self.path_stack) <= 1:
            print("Returned to start. Exploration finished.")
            return False
        current_cell = self.path_stack.pop()
        previous_cell = self.path_stack[-1]
        backtrack_direction = self._get_direction_to_neighbor(current_cell, previous_cell)
        if backtrack_direction is None:
            print("Error: Could not determine backtrack direction.")
            self.path_stack.append(current_cell)
            return False
        print(f"Backtracking from {current_cell} to {previous_cell}")
        self._turn_to(backtrack_direction)
        self.ctrl.move_forward_pid(self.CELL_SIZE)
        return True
        
    def _turn_to(self, target_direction):
        # (โค้ดเดิม ไม่มีการเปลี่ยนแปลง)
        turn_angle = (target_direction - self.current_orientation) * 90
        if turn_angle > 180: turn_angle -= 360
        if turn_angle < -180: turn_angle += 360
        if abs(turn_angle) > 5:
            self.ctrl.turn(turn_angle)
        self.current_orientation = target_direction
        
# ===================== Main =====================
if __name__ == "__main__":
    
    # ++++++++++++++ USER SETTINGS: แก้ไขค่าตรงนี้ได้เลย ++++++++++++++
    # กำหนดจุดเริ่มต้น (x, y)
    START_CELL = (0, 0) 
    
    # กำหนดขอบเขตแผนที่ 6x6 (x_min, x_max, y_min, y_max)
    # เช่น (0, 5, 0, 5) หมายถึง x จาก 0 ถึง 5 และ y จาก 0 ถึง 5
    MAZE_BOUNDS = (0, 5, 0, 5)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ctrl = None
    solver = None
    try:
        print("Connecting to robot...")
        ctrl = Control(conn_type="ap") 
        print("Robot connected. Initializing solver...")
        
        solver = MazeSolver(ctrl, start_cell=START_CELL, exploration_zone=MAZE_BOUNDS)
        
        solver.explore()
        print("\n--- Mapping Mission Complete ---")

    except Exception as e:
        print(f"\n--- An error occurred in the main loop: {e} ---")
        import traceback
        traceback.print_exc()
    finally:
        if solver:
            print("Saving final map to 'final_maze_map.png'")
            _fig.savefig("final_maze_map.png", dpi=300)
            finalize_show()

        if ctrl:
            print("Cleaning up and closing connection.")
            ctrl.close()