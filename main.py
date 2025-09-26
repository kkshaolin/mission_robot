# main.py
import time
import sys
import traceback
from control import Control ,MazeSolver

if __name__ == "__main__":
    ctrl = None
    try:
        print("Connecting to robot...")
        ctrl = Control(conn_type="ap")
        # ctrl.move_forward_pid(cell_size_m=0.6)
        print("Robot connected. Initializing solver...")
        solver = MazeSolver(ctrl)
        solver.explore()
    except Exception as e:
        print(f"\n--- An error occurred in the main loop: {e} ---")
        traceback.print_exc()
    finally:
        if ctrl:
            print("Cleaning up and closing connection.")
            ctrl.close()
    sys.exit(0)  # หยุดโปรแกรมเมื่อแสดงผลเสร็จ