import cv2
import numpy as np
from datetime import datetime
import os
from ultralytics import YOLO
import torch

# -----------------------
# PARAMETERS
# -----------------------
PIXEL_DIFF_THRESHOLD = 25
MIN_MOTION_AREA = 1500
BLUR_SIZE = (7, 7)
DILATE_ITERATIONS = 3
MAX_DISTANCE = 50
MAX_DISAPPEARED = 15
SCREENSHOT_COOLDOWN = 45  # Minimum frames between screenshots
MIN_CONFIDENCE = 0.5  # Minimum confidence for object detection
DISPLAY_SCALE = 0.8  # Scale down display for better performance

# -----------------------
# PATHS AND DIRECTORIES
# -----------------------
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# -----------------------
# LOAD YOLOv8 MODEL (Most Accurate)
# -----------------------
print("Loading YOLOv8 model...")
try:
    # Load YOLOv8n (nano) - fastest
    # You can change to 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' for better accuracy
    model = YOLO('yolov8n.pt')
    print("YOLOv8 model loaded successfully!")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
except Exception as e:
    print(f"Error loading YOLOv8: {e}")
    print("Downloading YOLOv8n model...")
    try:
        import requests
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        response = requests.get(url)
        with open('yolov8n.pt', 'wb') as f:
            f.write(response.content)
        model = YOLO('yolov8n.pt')
        print("YOLOv8 model downloaded and loaded!")
    except:
        print("Failed to load YOLOv8. Please install ultralytics: pip install ultralytics")
        exit(1)

# Load face detector for better face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------
# ROBUST CENTROID TRACKER
# -----------------------
class CentroidTracker:
    def __init__(self, max_distance=50, max_disappeared=10):
        self.next_object_id = 0
        self.objects = {}           # {id: centroid}
        self.disappeared = {}       # {id: disappeared_frames}
        self.frames_count = {}      # {id: total_frames_seen}
        self.last_screenshot = {}   # {id: last_screenshot_frame}
        self.current_frame = 0      # Frame counter
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def update(self, detections):
        self.current_frame += 1
        updated_objects = {}
        
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.objects.pop(obj_id)
                    self.disappeared.pop(obj_id)
                    self.last_screenshot.pop(obj_id, None)
            return self.objects, self.frames_count

        input_centroids = []
        for x, y, w, h in detections:
            cx = x + w // 2
            cy = y + h // 2
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.objects[self.next_object_id] = centroid
                self.frames_count[self.next_object_id] = 1
                self.disappeared[self.next_object_id] = 0
                self.last_screenshot[self.next_object_id] = 0
                self.next_object_id += 1
            return self.objects, self.frames_count

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        used_objects = set()
        
        for input_c in input_centroids:
            min_dist = float('inf')
            closest_id = None
            for i, obj_c in enumerate(object_centroids):
                obj_id = object_ids[i]
                if obj_id in used_objects:
                    continue
                dist = np.sqrt((input_c[0]-obj_c[0])**2 + (input_c[1]-obj_c[1])**2)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    closest_id = obj_id

            if closest_id is not None:
                updated_objects[closest_id] = input_c
                self.frames_count[closest_id] += 1
                self.disappeared[closest_id] = 0
                used_objects.add(closest_id)
            else:
                updated_objects[self.next_object_id] = input_c
                self.frames_count[self.next_object_id] = 1
                self.disappeared[self.next_object_id] = 0
                self.last_screenshot[self.next_object_id] = 0
                self.next_object_id += 1

        for obj_id in self.objects.keys():
            if obj_id not in used_objects:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] <= self.max_disappeared:
                    updated_objects[obj_id] = self.objects[obj_id]

        self.objects = updated_objects
        return self.objects, self.frames_count
    
    def can_take_screenshot(self, obj_id):
        if obj_id not in self.last_screenshot:
            self.last_screenshot[obj_id] = 0
            return True
        return (self.current_frame - self.last_screenshot[obj_id]) > SCREENSHOT_COOLDOWN
    
    def update_screenshot_time(self, obj_id):
        self.last_screenshot[obj_id] = self.current_frame

# -----------------------
# OBJECT DETECTION FUNCTIONS
# -----------------------
def detect_objects_yolo(frame):
    """Detect objects using YOLOv8 with high accuracy"""
    results = model(frame, verbose=False)  # Set verbose=False to reduce output
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                if confidence > MIN_CONFIDENCE:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    
                    # Get class name
                    class_name = model.names[class_id]
                    
                    detections.append({
                        'bbox': (x1, y1, w, h),
                        'label': class_name,
                        'confidence': float(confidence),
                        'class_id': class_id
                    })
    
    return detections

def detect_faces(frame):
    """Detect faces using Haar Cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    face_detections = []
    for (x, y, w, h) in faces:
        face_detections.append({
            'bbox': (x, y, w, h),
            'label': 'face',
            'confidence': 0.9,
            'class_id': -1
        })
    
    return face_detections

def save_screenshot(frame, obj_type, obj_id, timestamp, is_cropped=False):
    """Save screenshot without bounding boxes"""
    if is_cropped:
        filename = f"{SCREENSHOT_DIR}/cropped_{obj_type}_{obj_id}_{timestamp}.jpg"
    else:
        filename = f"{SCREENSHOT_DIR}/full_{obj_type}_{obj_id}_{timestamp}.jpg"
    
    try:
        cv2.imwrite(filename, frame)
        return True
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return False

def extract_region(frame, bbox, padding=15):
    """Extract region from frame with padding"""
    x, y, w, h = bbox
    height, width = frame.shape[:2]
    
    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    
    # Ensure region is valid
    if y2 > y1 and x2 > x1:
        return frame[y1:y2, x1:x2]
    return None

# -----------------------
# CLEAN DISPLAY FUNCTIONS
# -----------------------
def draw_clean_info_panel(frame, motion_score, objects_count, status, fps):
    """Draw a clean, organized information panel"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay for info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Draw info text
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Title
    cv2.putText(frame, "MOTION & OBJECT DETECTION", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Time
    cv2.putText(frame, f"Time: {timestamp}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Status with color coding
    status_color = (0, 255, 0) if status == "No Motion" else (0, 0, 255)
    cv2.putText(frame, f"Status: {status}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    
    # Motion score
    cv2.putText(frame, f"Motion: {motion_score:,}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
    
    # Objects tracked
    cv2.putText(frame, f"Objects: {objects_count}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)
    
    return frame

def draw_object_labels(frame, detection):
    """Draw clean object labels"""
    x, y, w, h = detection['bbox']
    label = detection['label']
    confidence = detection['confidence']
    
    # Different colors for different object types
    if label == 'person':
        color = (255, 50, 50)  # Red
        text = f"Person: {confidence:.1%}"
    elif label == 'face':
        color = (255, 255, 0)  # Yellow
        text = "Face"
    elif label in ['car', 'truck', 'bus', 'motorcycle']:
        color = (0, 200, 255)  # Orange
        text = f"{label}: {confidence:.1%}"
    else:
        color = (50, 255, 50)  # Green
        text = f"{label}: {confidence:.1%}"
    
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw label background
    label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                  (x + label_size[0] + 10, y), color, -1)
    
    # Draw label text
    cv2.putText(frame, text, (x + 5, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

# -----------------------
# MAIN LOOP
# -----------------------
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
tracker = CentroidTracker(MAX_DISTANCE, MAX_DISAPPEARED)

print("\n" + "="*50)
print("HIGHLY ACCURATE MOTION & OBJECT DETECTION")
print("="*50)
print(f"Screenshots will be saved in: {SCREENSHOT_DIR}")
print("Press 'q' to quit")
print("Press 's' to manually save screenshot")
print("Press 'd' to toggle detection display")
print("="*50)

# FPS calculation
fps_start_time = datetime.now()
fps_frame_count = 0
fps = 0

# Display settings
show_detections = True

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    fps_frame_count += 1
    
    # Calculate FPS every second
    current_time = datetime.now()
    time_diff = (current_time - fps_start_time).total_seconds()
    if time_diff >= 1.0:
        fps = fps_frame_count / time_diff
        fps_frame_count = 0
        fps_start_time = current_time
    
    original_frame = frame.copy()
    display_frame = frame.copy()
    
    # Resize for faster processing if needed
    if DISPLAY_SCALE != 1.0:
        height, width = frame.shape[:2]
        new_width = int(width * DISPLAY_SCALE)
        new_height = int(height * DISPLAY_SCALE)
        frame = cv2.resize(frame, (new_width, new_height))
    
    # Motion detection
    blurred = cv2.GaussianBlur(frame, BLUR_SIZE, 0)
    fgmask = fgbg.apply(blurred)
    _, thresh = cv2.threshold(fgmask, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    motion_score = 0
    motion_regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_MOTION_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        detections.append((x, y, w, h))
        motion_regions.append((x, y, w, h))
        motion_score += int(area)
    
    objects, frames_count = tracker.update(detections)
    
    # Object detection (always run, but only save screenshots on motion)
    detected_objects = detect_objects_yolo(frame)
    detected_faces = detect_faces(frame)
    
    # Combine all detections
    all_detections = detected_objects + detected_faces
    
    # Check for overlap between motion regions and detected objects
    motion_detected = motion_score > MIN_MOTION_AREA
    
    if motion_detected:
        for obj_id, centroid in objects.items():
            if tracker.can_take_screenshot(obj_id):
                # Find motion region near this object
                for motion_region in motion_regions:
                    mx, my, mw, mh = motion_region
                    cx, cy = centroid
                    
                    # Check if centroid is inside motion region
                    if (mx <= cx <= mx + mw) and (my <= cy <= my + mh):
                        # Find objects in this region
                        for detection in all_detections:
                            ox, oy, ow, oh = detection['bbox']
                            
                            # Check if object overlaps with motion region
                            x_overlap = max(0, min(mx + mw, ox + ow) - max(mx, ox))
                            y_overlap = max(0, min(my + mh, oy + oh) - max(my, oy))
                            overlap_area = x_overlap * y_overlap
                            obj_area = ow * oh
                            
                            # If object overlaps with motion region
                            if obj_area > 0 and overlap_area > 0.2 * obj_area:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                
                                # Save full frame
                                if DISPLAY_SCALE != 1.0:
                                    # Scale coordinates back to original size
                                    scale_factor = 1.0 / DISPLAY_SCALE
                                    ox_orig = int(ox * scale_factor)
                                    oy_orig = int(oy * scale_factor)
                                    ow_orig = int(ow * scale_factor)
                                    oh_orig = int(oh * scale_factor)
                                    obj_bbox_orig = (ox_orig, oy_orig, ow_orig, oh_orig)
                                else:
                                    obj_bbox_orig = (ox, oy, ow, oh)
                                
                                # Save full frame
                                save_screenshot(original_frame, detection['label'], 
                                               f"obj{obj_id}", timestamp, False)
                                
                                # Save cropped object
                                cropped = extract_region(original_frame, obj_bbox_orig, 20)
                                if cropped is not None:
                                    save_screenshot(cropped, detection['label'], 
                                                   f"obj{obj_id}", timestamp, True)
                                
                                print(f"[{timestamp}] Saved: {detection['label']} (ID: {obj_id})")
                                tracker.update_screenshot_time(obj_id)
                                break
    
    # Draw on display frame
    if show_detections:
        # Draw motion regions (semi-transparent)
        for (x, y, w, h) in motion_regions:
            if DISPLAY_SCALE != 1.0:
                scale_factor = 1.0 / DISPLAY_SCALE
                x_disp = int(x * scale_factor)
                y_disp = int(y * scale_factor)
                w_disp = int(w * scale_factor)
                h_disp = int(h * scale_factor)
            else:
                x_disp, y_disp, w_disp, h_disp = x, y, w, h
            
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (x_disp, y_disp), 
                         (x_disp + w_disp, y_disp + h_disp), 
                         (0, 0, 255), -1)
            display_frame = cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0)
            cv2.rectangle(display_frame, (x_disp, y_disp), 
                         (x_disp + w_disp, y_disp + h_disp), 
                         (0, 0, 255), 2)
        
        # Draw object detections
        for detection in all_detections:
            if DISPLAY_SCALE != 1.0:
                scale_factor = 1.0 / DISPLAY_SCALE
                ox, oy, ow, oh = detection['bbox']
                ox_disp = int(ox * scale_factor)
                oy_disp = int(oy * scale_factor)
                ow_disp = int(ow * scale_factor)
                oh_disp = int(oh * scale_factor)
                detection_disp = detection.copy()
                detection_disp['bbox'] = (ox_disp, oy_disp, ow_disp, oh_disp)
            else:
                detection_disp = detection
            
            display_frame = draw_object_labels(display_frame, detection_disp)
    
    # Draw tracked object IDs
    for obj_id, centroid in objects.items():
        if DISPLAY_SCALE != 1.0:
            scale_factor = 1.0 / DISPLAY_SCALE
            cx, cy = centroid
            cx_disp = int(cx * scale_factor)
            cy_disp = int(cy * scale_factor)
        else:
            cx_disp, cy_disp = centroid
        
        cv2.putText(display_frame, f"ID:{obj_id}", 
                   (cx_disp - 20, cy_disp - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.circle(display_frame, (cx_disp, cy_disp), 4, (255, 255, 0), -1)
    
    # Status text
    status = "Motion Detected" if motion_detected else "No Motion"
    
    # Draw clean info panel
    display_frame = draw_clean_info_panel(display_frame, motion_score, 
                                         len(objects), status, fps)
    
    # Display model info
    cv2.putText(display_frame, "Model: YOLOv8 (High Accuracy)", 
                (display_frame.shape[1] - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    
    # Show display
    cv2.imshow("Motion & Object Detection", display_frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Manual screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_screenshot(original_frame, "manual", "capture", timestamp, False)
        print(f"[{timestamp}] Manual screenshot saved")
    elif key == ord('d'):
        show_detections = not show_detections
        print(f"Detection display: {'ON' if show_detections else 'OFF'}")
    elif key == ord('f'):
        # Force detection on next frame
        print("Forcing object detection...")

print("\n" + "="*50)
print("SESSION SUMMARY")
print("="*50)
print(f"Total objects tracked: {len(frames_count)}")
for obj_id, count in frames_count.items():
    print(f"  Object {obj_id}: {count} frames")
print(f"\nScreenshots saved in: {os.path.abspath(SCREENSHOT_DIR)}")
print("="*50)

cap.release()
cv2.destroyAllWindows()