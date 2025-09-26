# plotting.py
import matplotlib.pyplot as plt
import sys

plt.ion()
_fig, _ax = plt.subplots(figsize=(8, 8))

def plot_maze(current_cell, visited, walls, path_stack, title="Real-time Maze Exploration"):
    ax = _ax
    ax.clear()
    for (x, y) in visited:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightgray', edgecolor='gray'))
    for wall in walls:
        (x1, y1), (x2, y2) = wall
        if x1 == x2:
            ax.plot([x1 - 0.5, x1 + 0.5], [max(y1, y2) - 0.5, max(y1, y2) - 0.5], 'k-', lw=4)
        else:
            ax.plot([max(x1, x2) - 0.5, max(x1, x2) - 0.5], [y1 - 0.5, y1 + 0.5], 'k-', lw=4)
    if len(path_stack) > 1:
        path_x, path_y = zip(*path_stack)
        ax.plot(path_x, path_y, 'b-o', markersize=5)
    cx, cy = current_cell
    ax.plot(cx, cy, 'ro', markersize=12, label='Robot')
    all_x = [c[0] for c in visited] or [0]
    all_y = [c[1] for c in visited] or [0]
    ax.set_xlim(min(all_x) - 1.5, max(all_x) + 1.5)
    ax.set_ylim(min(all_y) - 1.5, max(all_y) + 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title(title)
    plt.pause(0.1)

def finalize_show():
    plt.ioff()
    plt.show(block=True)  # ใช้ block=True เพื่อให้หน้าต่าง plot ค้างไว้
    sys.exit(0)  # หยุดโปรแกรมเมื่อแสดงผลเสร็จ