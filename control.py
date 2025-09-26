# control.py
import time
import math
import numpy as np
import threading  # ใช้สำหรับ lock ใน MarkerDetector แต่ import ที่นี่ด้วยถ้าจำเป็น
from robomaster import robot
from detection import MarkerDetector

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self._prev_error, self._integral = 0.0, 0.0
        self._last_time = time.time()

    def compute(self, current_value):
        t, dt = time.time(), time.time() - self._last_time
        if dt <= 0: return 0.0
        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt
        out = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        self._prev_error, self._last_time = error, t
        return out

class Control:
    def __init__(self, conn_type="ap"):
        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type=conn_type)
        self.ep_chassis = self.ep_robot.chassis
        self.ep_gimbal = self.ep_robot.gimbal
        self.ep_sensor = self.ep_robot.sensor
        # ----- State -----
        self.last_distance_cm = None
        self.current_x, self.current_y, self.current_yaw = 0.0, 0.0, 0.0
        # ----- Callbacks -----
        def _dist_cb(sub_info):
            try:
                mm = int(sub_info[0])
                if mm > 0: self.last_distance_cm = mm / 10.0
            except: pass

        def _att_cb(attitude_info):
            self.current_yaw = float(attitude_info[0])

        def _pos_cb(position_info):
            self.current_x, self.current_y = float(position_info[0]), float(position_info[1])

        self.ep_gimbal.recenter().wait_for_completed()
        self.ep_chassis.sub_attitude(freq=10, callback=_att_cb)
        self.ep_chassis.sub_position(freq=50, callback=_pos_cb)
        self._dist_subscribed = False
        self._dist_cb = _dist_cb
        # ---------- VISION (Marker Detection) ----------
        self.marker_detector = MarkerDetector(self.ep_robot)
        # ----- Chassis/Gimbal helpers -----
        self._sub_distance(freq=20)

    def get_yaw_deg(self):
        return self.current_yaw

    def get_xy_m(self):
        return self.current_x, self.current_y

    def _sub_distance(self, freq=20):
        if not self._dist_subscribed:
            self.ep_sensor.sub_distance(freq=freq, callback=self._dist_cb)
            self._dist_subscribed = True

    def _unsub_distance(self):
        if self._dist_subscribed:
            self.ep_sensor.unsub_distance()
            self._dist_subscribed = False

    def read_distance_at(self, yaw_angle_deg, samples=5, timeout_s=1.0):
        self.last_distance_cm = None
        self.ep_gimbal.moveto(pitch=0, yaw=yaw_angle_deg, yaw_speed=180).wait_for_completed()
        distances = []
        self._sub_distance()
        t0 = time.time()
        while len(distances) < samples and (time.time() - t0) < timeout_s:
            if self.last_distance_cm is not None:
                distances.append(self.last_distance_cm)
                self.last_distance_cm = None
            time.sleep(0.05)
        self._unsub_distance()
        if not distances: return None
        med = float(np.median(distances))
        print(f"Median distance at yaw {yaw_angle_deg}: {med:.1f} cm")
        return med

    def eye(self):
        print("Scanning: [L, F, R]")
        dist = {
            "L": self.read_distance_at(-90),
            "F": self.read_distance_at(0),
            "R": self.read_distance_at(90)
        }
        self.ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=180).wait_for_completed()
        return dist

    def stop(self):
        self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
        time.sleep(0.2)

    def turn(self, angle_deg):
        print(f"Action: Turning {angle_deg:.1f} degrees")
        self.ep_chassis.move(x=0, y=0, z=-angle_deg, z_speed=45).wait_for_completed()
        time.sleep(0.5)

    def move_forward_pid(self, cell_size_m, Kp=4, Ki=0.1, Kd=0.5, v_clip=0.7, tol_m=0.01):
        print(f"Action: Moving forward {cell_size_m} m")
        pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=cell_size_m)
        sx, sy = self.get_xy_m()
        while True:
            dist = math.hypot(self.current_x - sx, self.current_y - sy)
            speed = float(np.clip(pid.compute(dist), -v_clip, v_clip))
            self.ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=0.1)
            if abs(cell_size_m - dist) < tol_m: break
            time.sleep(0.02)
        self.stop()
        print("Movement complete.")

    def get_markers(self, max_age=0.6):
        return self.marker_detector.get_markers(max_age)

    def close(self):
        try: self.ep_sensor.unsub_distance()
        except: pass
        try: self.ep_chassis.unsub_attitude()
        except: pass
        try: self.ep_chassis.unsub_position()
        except: pass
        self.marker_detector.close()
        try: self.ep_robot.close()
        except Exception as e: print(f"Error during cleanup: {e}")

# maze_solver.py
from plotting import plot_maze, finalize_show

class MazeSolver:
    WALL_THRESHOLD = 60
    CELL_SIZE = 0.6

    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.maze_map = {}
        self.visited = set([(0, 0)])
        self.path_stack = [(0, 0)]
        self.walls = set()
        self.current_orientation = self._get_discretized_orientation(self.ctrl.get_yaw_deg())

    @staticmethod
    def _get_discretized_orientation(yaw_deg):
        yaw = (yaw_deg + 360) % 360
        if yaw >= 315 or yaw < 45: return 0
        elif 45 <= yaw < 135: return 3
        elif 135 <= yaw < 225: return 2
        else: return 1

    @staticmethod
    def _get_target_coordinates(grid_x, grid_y, direction):
        if direction == 0: return (grid_x, grid_y + 1)
        elif direction == 1: return (grid_x + 1, grid_y)
        elif direction == 2: return (grid_x, grid_y - 1)
        elif direction == 3: return (grid_x - 1, grid_y)

    @staticmethod
    def _get_relative_directions(orientation):
        return {
            "L": (orientation - 1 + 4) % 4,
            "F": orientation,
            "R": (orientation + 1) % 4
        }

    @staticmethod
    def _get_direction_to_neighbor(current_cell, target_cell):
        dx = target_cell[0] - current_cell[0]
        dy = target_cell[1] - current_cell[1]
        if dx == 1: return 1
        if dx == -1: return 3
        if dy == 1: return 0
        if dy == -1: return 2
        return None

    def explore(self):
        print("Starting DFS Maze Solver (marker: detect-only, no action)...")
        while self.path_stack:
            current_cell = self.path_stack[-1]
            plot_maze(current_cell, self.visited, self.walls, self.path_stack)
            print(f"\nPosition: {current_cell}, Orientation: {self.current_orientation} "
                  f"(Yaw: {self.ctrl.get_yaw_deg():.1f}°)")
            # Log marker detection
            markers = self.ctrl.get_markers(max_age=0.6)
            if markers:
                ids = [str(m["info"]) for m in markers]
                xs = [round(m["x"], 3) for m in markers]
                print(f"[Marker] seen={len(markers)} ids={ids} x={xs}")
            if current_cell not in self.maze_map:
                self._scan_and_map(current_cell)
            if self._find_and_move_to_next_cell(current_cell):
                continue
            if not self._backtrack():
                break
        print("\nDFS exploration complete.")
        plot_maze(self.path_stack[-1], self.visited, self.walls, self.path_stack, "Final Map")
        finalize_show()

    def _scan_and_map(self, cell):
        print(f"Cell {cell} is unmapped. Scanning...")
        distances = self.ctrl.eye()
        relative_dirs = self._get_relative_directions(self.current_orientation)
        open_directions = set()
        for move_key in ["L", "F", "R"]:
            direction = relative_dirs[move_key]
            dist_cm = distances.get(move_key)
            if dist_cm is not None and dist_cm > self.WALL_THRESHOLD:
                open_directions.add(direction)
            else:
                neighbor = self._get_target_coordinates(cell[0], cell[1], direction)
                self.walls.add(tuple(sorted((cell, neighbor))))
        self.maze_map[cell] = open_directions
        print(f"Mapped {cell} with open directions: {sorted(list(open_directions))}")

    def _find_and_move_to_next_cell(self, cell):
        relative_dirs = self._get_relative_directions(self.current_orientation)
        search_order = [relative_dirs["L"], relative_dirs["F"], relative_dirs["R"]]
        for direction in search_order:
            if direction in self.maze_map.get(cell, set()):
                target_cell = self._get_target_coordinates(cell[0], cell[1], direction)
                if target_cell not in self.visited:
                    print(f"Found unvisited neighbor {target_cell}. Moving...")
                    self._turn_to(direction)
                    self.ctrl.move_forward_pid(self.CELL_SIZE)
                    self.visited.add(target_cell)
                    self.path_stack.append(target_cell)
                    return True
        return False

    def _backtrack(self):
        print("Dead end. Backtracking...")
        if len(self.path_stack) <= 1:
            print("Returned to start. Exploration finished.")
            return False
        current_cell = self.path_stack.pop()
        previous_cell = self.path_stack[-1]
        backtrack_direction = self._get_direction_to_neighbor(current_cell, previous_cell)
        if backtrack_direction is None:
            print("Error: Could not determine backtrack direction.")
            return False
        print(f"Backtracking from {current_cell} to {previous_cell}")
        self._turn_to(backtrack_direction)
        self.ctrl.move_forward_pid(self.CELL_SIZE)
        return True

    def _turn_to(self, target_direction):
        turn_angle = (target_direction - self.current_orientation) * 90
        if turn_angle > 180: turn_angle -= 360
        if turn_angle < -180: turn_angle += 360
        if abs(turn_angle) > 1:
            self.ctrl.stop()
            self.ctrl.turn(turn_angle)
        self.current_orientation = target_direction