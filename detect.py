import cv2
import numpy as np

WIDTH = 848
HEIGHT = 480
RESOLUTION = '480p'
CENTER_X = WIDTH / 2
CENTER_Y = HEIGHT / 2

# HSV Color Ranges (calibrated from field)
COLOR_RANGES = {
    'red': [
        {'lower': np.array([0, 90, 70]), 'upper': np.array([10, 255, 255])},
        {'lower': np.array([170, 90, 70]), 'upper': np.array([180, 255, 255])}
    ],
    'green': [
        {'lower': np.array([75, 230, 25]), 'upper': np.array([90, 255, 255])}
    ],
    'blue': [
        {'lower': np.array([100, 190, 30]), 'upper': np.array([126, 255, 255])}
    ],
    'yellow': [
        {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])}
    ]
}

# Detection Parameters (adjusted for 480p)
MIN_CONTOUR_AREA = 800
MAX_CONTOUR_AREA = 50000
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0
MIN_SOLIDITY = 0.4

# Morphology
MORPH_KERNEL_SIZE = (7, 7)
MORPH_OPEN_ITER = 2
MORPH_CLOSE_ITER = 2

# ROI - Region of Interest
MASK_TOP_Y = 140
MASK_BOTTOM_Y = 400
MASK_LEFT_X = 80
MASK_RIGHT_X = 768

# PID Parameters
KP_YAW, KI_YAW, KD_YAW = 0.22, 0.0002, 0.018
KP_PITCH, KI_PITCH, KD_PITCH = 0.22, 0.0002, 0.018
SIGN_YAW, SIGN_PITCH = +1, -1
MAX_SPEED = 250.0
INT_CLAMP = 30000.0

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clamp(x, lo, hi):
    """Clamp value between lo and hi"""
    return lo if x < lo else hi if x > hi else x

def create_kalman_state():
    """Create new Kalman filter state"""
    return {
        'kf': cv2.KalmanFilter(4, 2),
        'initialized': False
    }

def init_kalman_filter(state):
    """Initialize Kalman filter matrices"""
    state['kf'].measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    state['kf'].transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    state['kf'].processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    state['kf'].measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

def kalman_update(state, x, y):
    """Update Kalman filter and return predicted position"""
    measurement = np.array([[np.float32(x)], [np.float32(y)]])
    
    if not state['initialized']:
        init_kalman_filter(state)
        state['kf'].statePre = np.array([[x], [y], [0], [0]], np.float32)
        state['kf'].statePost = np.array([[x], [y], [0], [0]], np.float32)
        state['initialized'] = True
    
    state['kf'].correct(measurement)
    prediction = state['kf'].predict()
    return float(prediction[0]), float(prediction[1])

def kalman_reset(state):
    """Reset Kalman filter"""
    state['initialized'] = False

def create_pid_state():
    """Create new PID controller state"""
    return {
        'int': 0.0,
        'prev_e': 0.0,
        'prev_t': None,
        'd_ema': 0.0
    }

def pid_step(state, kp, ki, kd, error, t_now):
    """Calculate PID control output"""
    if state['prev_t'] is None:
        state['prev_t'] = t_now
        state['prev_e'] = error
        return kp * error
    
    dt = max(1e-3, t_now - state['prev_t'])
    de = (error - state['prev_e']) / dt
    state['d_ema'] = 0.5 * de + 0.5 * state['d_ema']
    state['int'] = clamp(state['int'] + error*dt, -INT_CLAMP, INT_CLAMP)
    
    u = kp*error + ki*state['int'] + kd*state['d_ema']
    
    state['prev_e'] = error
    state['prev_t'] = t_now
    
    return u

def pid_reset(state):
    """Reset PID state"""
    state['int'] = 0.0
    state['prev_e'] = 0.0
    state['prev_t'] = None
    state['d_ema'] = 0.0

# ============================================================================
# COLOR DETECTION FUNCTIONS
# ============================================================================

def detect_color(frame, color_name):
    """
    Detect specific color in frame
    
    Args:
        frame: BGR image
        color_name: 'red', 'green', 'blue', or 'yellow'
    
    Returns:
        Binary mask (numpy array)
    """
    # Apply ROI mask
    m = frame.copy()
    m[0:MASK_TOP_Y, :] = 0
    m[MASK_BOTTOM_Y:, :] = 0
    m[:, 0:MASK_LEFT_X] = 0
    m[:, MASK_RIGHT_X:] = 0
    
    # Preprocess
    blur = cv2.GaussianBlur(m, (7,7), 1)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # Create color mask
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color_range in COLOR_RANGES[color_name]:
        temp_mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        mask = cv2.bitwise_or(mask, temp_mask)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITER)
    mask = cv2.medianBlur(mask, 5)
    
    return mask

def find_largest_target(mask):
    """
    Find largest valid contour in mask
    
    Args:
        mask: Binary mask
    
    Returns:
        Target dict with keys: 'bbox', 'center', 'area', 'contour'
        or None if no valid target found
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter by area
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        center_x = x + w/2
        center_y = y + h/2
        
        # Filter by ROI bounds
        if not (MASK_TOP_Y < center_y < MASK_BOTTOM_Y):
            continue
        if not (MASK_LEFT_X < center_x < MASK_RIGHT_X):
            continue
        
        # Filter by aspect ratio
        aspect = float(w)/h if h > 0 else 0
        if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
            continue
        
        # Filter by solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area/hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY:
            continue
        
        valid.append((cnt, area))
    
    if not valid:
        return None
    
    # Return largest valid contour
    largest_cnt, _ = max(valid, key=lambda x: x[1])
    hull = cv2.convexHull(largest_cnt)
    x, y, w, h = cv2.boundingRect(hull)
    
    return {
        'bbox': (x, y, x+w, y+h),
        'center': (x+w/2, y+h/2),
        'area': cv2.contourArea(largest_cnt),
        'contour': largest_cnt
    }

def detect_all_colors(frame, target_colors=['red', 'green', 'blue', 'yellow']):
    """
    Detect all target colors in frame
    
    Args:
        frame: BGR image
        target_colors: List of color names to detect
    
    Returns:
        Dict mapping color name to target info
    """
    all_targets = {}
    
    for color in target_colors:
        mask = detect_color(frame, color)
        target = find_largest_target(mask)
        if target:
            target['color'] = color
            all_targets[color] = target
    
    return all_targets

def select_target_by_priority(all_targets, priority=['red', 'green', 'blue', 'yellow']):
    """
    Select target based on priority order
    
    Args:
        all_targets: Dict of detected targets
        priority: List of colors in priority order
    
    Returns:
        Selected target dict or None
    """
    for color in priority:
        if color in all_targets:
            return all_targets[color]
    return None

def select_largest_target(all_targets):
    """
    Select largest target by area
    
    Args:
        all_targets: Dict of detected targets
    
    Returns:
        Largest target dict or None
    """
    if not all_targets:
        return None
    return max(all_targets.values(), key=lambda x: x['area'])

# ============================================================================
# TRACKING FUNCTIONS
# ============================================================================

def track_target(gimbal, target, tracker_state, t_now):
    """
    Track target with gimbal using PID control
    
    Args:
        gimbal: Robomaster gimbal object
        target: Target dict with 'center' key, or None
        tracker_state: Dict with 'kalman', 'pid_yaw', 'pid_pitch' states
        t_now: Current timestamp
    """
    if target:
        cx, cy = target['center']
        tgt_x, tgt_y = kalman_update(tracker_state['kalman'], cx, cy)
        
        err_x = tgt_x - CENTER_X
        err_y = tgt_y - CENTER_Y
        
        u_yaw = pid_step(tracker_state['pid_yaw'], KP_YAW, KI_YAW, KD_YAW, err_x, t_now)
        u_pitch = pid_step(tracker_state['pid_pitch'], KP_PITCH, KI_PITCH, KD_PITCH, err_y, t_now)
        
        u_yaw = clamp(SIGN_YAW * u_yaw, -MAX_SPEED, MAX_SPEED)
        u_pitch = clamp(SIGN_PITCH * u_pitch, -MAX_SPEED, MAX_SPEED)
        
        gimbal.drive_speed(pitch_speed=u_pitch, yaw_speed=u_yaw)
    else:
        stop_tracking(gimbal, tracker_state)

def stop_tracking(gimbal, tracker_state):
    """
    Stop gimbal and reset tracking state
    
    Args:
        gimbal: Robomaster gimbal object
        tracker_state: Dict with 'kalman', 'pid_yaw', 'pid_pitch' states
    """
    gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    pid_reset(tracker_state['pid_yaw'])
    pid_reset(tracker_state['pid_pitch'])
    kalman_reset(tracker_state['kalman'])

def create_tracker_state():
    """
    Create new tracker state
    
    Returns:
        Dict with kalman and PID states
    """
    return {
        'kalman': create_kalman_state(),
        'pid_yaw': create_pid_state(),
        'pid_pitch': create_pid_state()
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def draw_targets(frame, selected, all_targets, show_crosshair=True):
    """
    Draw detection visualization on frame
    
    Args:
        frame: BGR image (modified in-place)
        selected: Selected target dict or None
        all_targets: Dict of all detected targets
        show_crosshair: Whether to draw center crosshair
    """
    h, w = frame.shape[:2]
    
    # Draw center crosshair
    if show_crosshair:
        cv2.drawMarker(frame, (int(w/2), int(h/2)), (255,255,255), cv2.MARKER_CROSS, 20, 2)
    
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255)
    }
    
    # Draw non-selected targets
    for color, target in all_targets.items():
        if target == selected:
            continue
        x1, y1, x2, y2 = map(int, target['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[color], 1)
        cv2.putText(frame, color.upper(), (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_map[color], 1)
    
    # Draw selected target
    if selected:
        x1, y1, x2, y2 = map(int, selected['bbox'])
        color = selected['color']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[color], 3)
        cx, cy = map(int, selected['center'])
        cv2.circle(frame, (cx, cy), 6, color_map[color], -1)
        cv2.drawMarker(frame, (cx, cy), (255,255,255), cv2.MARKER_CROSS, 10, 1)
        cv2.putText(frame, color.upper(), (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map[color], 2)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def run_detection_test():
    """Standalone test function"""
    from robomaster import robot
    import time
    
    ep = robot.Robot()
    ep.initialize(conn_type="ap")
    cam = ep.camera
    gim = ep.gimbal
    
    cam.start_video_stream(display=False, resolution=RESOLUTION)
    gim.recenter().wait_for_completed()
    
    tracker_state = create_tracker_state()
    priority = ['red', 'green', 'blue', 'yellow']
    
    try:
        while True:
            t_now = time.time()
            frame = cam.read_cv2_image(strategy="newest", timeout=0.3)
            if frame is None:
                continue
            
            # Detect all colors
            all_targets = detect_all_colors(frame)
            
            # Select target by priority
            selected = select_target_by_priority(all_targets, priority)
            
            # Track target
            track_target(gim, selected, tracker_state, t_now)
            
            # Visualization
            dbg = frame.copy()
            draw_targets(dbg, selected, all_targets)
            
            y = 30
            if selected:
                cv2.putText(dbg, f"Target: {selected['color'].upper()}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(dbg, "No target", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            y += 30
            cv2.putText(dbg, f"Found: {len(all_targets)}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Multi-Color Detection", dbg)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        stop_tracking(gim, tracker_state)
        if ep.is_connected():
            ep.close()
        cv2.destroyAllWindows()
        print("Done")

if __name__ == "__main__":
    run_detection_test()