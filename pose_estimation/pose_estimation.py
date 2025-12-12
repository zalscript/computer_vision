import cv2
import math
import numpy as np
from ultralytics import YOLO

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.
    a, b, c: (x, y) coordinates
    b is the vertex of the angle.
    Returns angle in degrees.
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def main():
    # Load the YOLOv8 pose model
    print("Loading YOLOv8 Pose model...")
    model = YOLO('yolov8n-pose.pt')

    # Open the webcam (index 0)
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution to HD (1280x720) to make it wider/clearer
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Pose Estimation. Press 'q' to exit.")

    # Create a named window that can be resized or set to fullscreen
    window_name = "YOLOv8 Pose Estimation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Uncomment the line below to force fullscreen immediately
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Run pose estimation on the frame
        results = model(frame, conf=0.5, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # ----- Jump detection state (ground plane baseline per detected person) -----
        # We store/update ankle baseline when person is detected as Standing.
        # If ankle Y is significantly above baseline, we mark as 'Jumping'.
        if not hasattr(main, "_ground_baselines"):
            main._ground_baselines = {}
            main._jump_cooldowns = {}
        EMA_ALPHA = 0.1
        JUMP_THRESHOLD_RATIO = 0.08  # fraction of frame height that constitutes a jump
        JUMP_DEBOUNCE_FRAMES = 6
        
        # Process keypoints for action recognition
        try:
            # Check if any person is detected
            if results[0].keypoints is not None and results[0].keypoints.shape[1] > 0:
                keypoints_data = results[0].keypoints.data
                
                # Iterate over each detected person
                for person_idx, kps in enumerate(keypoints_data):
                    # Extract coordinates (x, y) for relevant joints
                    # Ensure keypoints are detected (conf > 0.5 is handled by model conf, but good to check visibility)
                    
                    # COCO Keypoint Map:
                    # 11: Left Hip, 13: Left Knee, 15: Left Ankle
                    # 12: Right Hip, 14: Right Knee, 16: Right Ankle
                    
                    # Convert to cpu numpy for easy handling
                    kps_np = kps.cpu().numpy()
                    
                    # Check confidence or visibility if needed (usually kps has x,y,conf)
                    # Here we assume if they are present in output, we try to use them.
                    
                    # Left Leg
                    l_hip = kps_np[11][:2]
                    l_knee = kps_np[13][:2]
                    l_ankle = kps_np[15][:2]
                    
                    # Right Leg
                    r_hip = kps_np[12][:2]
                    r_knee = kps_np[14][:2]
                    r_ankle = kps_np[16][:2]
                    
                    # Calculate Knee Angles
                    # We check if keypoints are not [0,0] (undetected)
                    if np.count_nonzero(l_hip) == 2 and np.count_nonzero(l_knee) == 2 and np.count_nonzero(l_ankle) == 2:
                        angle_l = calculate_angle(l_hip, l_knee, l_ankle)
                    else:
                        angle_l = 0

                    if np.count_nonzero(r_hip) == 2 and np.count_nonzero(r_knee) == 2 and np.count_nonzero(r_ankle) == 2:
                        angle_r = calculate_angle(r_hip, r_knee, r_ankle)
                    else:
                        angle_r = 0
                    
                    # Heuristics for Standing vs Sitting
                    # Standing: Legs are straight, angle likely > 160 degrees
                    # Sitting: Legs are bent, angle likely < 140 degrees (approx 90)
                    
                    # We average the angles if both legs are visible, or take the one that is visible
                    if angle_l > 0 and angle_r > 0:
                        avg_angle = (angle_l + angle_r) / 2
                    elif angle_l > 0:
                        avg_angle = angle_l
                    elif angle_r > 0:
                        avg_angle = angle_r
                    else:
                        avg_angle = 0
                        
                    stage = "Unknown"
                    if avg_angle > 160:
                        stage = "Standing"
                    elif avg_angle >= 140 and avg_angle <= 160:
                        # Ambiguous states between sitting and standing
                        stage = "Bending" 
                    elif avg_angle > 0 and avg_angle < 140:
                        stage = "Sitting"
                    elif avg_angle == 0:
                        stage = "Legs Not Visible"

                    # Compute ankle Y (image coordinates: y increases downwards)
                    frame_h = frame.shape[0]
                    ankle_y = None
                    if np.count_nonzero(l_ankle) == 2 and np.count_nonzero(r_ankle) == 2:
                        ankle_y = (l_ankle[1] + r_ankle[1]) / 2.0
                    elif np.count_nonzero(l_ankle) == 2:
                        ankle_y = l_ankle[1]
                    elif np.count_nonzero(r_ankle) == 2:
                        ankle_y = r_ankle[1]

                    # Update ground baseline when standing
                    if ankle_y is not None:
                        if stage == "Standing":
                            if person_idx not in main._ground_baselines:
                                main._ground_baselines[person_idx] = float(ankle_y)
                            else:
                                main._ground_baselines[person_idx] = (
                                    (1 - EMA_ALPHA) * main._ground_baselines[person_idx]
                                    + EMA_ALPHA * float(ankle_y)
                                )

                        # Check for jump: baseline - current_y > threshold (smaller y = higher)
                        if person_idx in main._ground_baselines:
                            baseline = main._ground_baselines[person_idx]
                            delta = baseline - float(ankle_y)
                            threshold = JUMP_THRESHOLD_RATIO * frame_h
                            # debounce using cooldown frames to avoid flicker
                            cooldown = main._jump_cooldowns.get(person_idx, 0)
                            if delta > threshold and cooldown == 0:
                                stage = "Jumping"
                                main._jump_cooldowns[person_idx] = JUMP_DEBOUNCE_FRAMES
                            elif cooldown > 0:
                                main._jump_cooldowns[person_idx] = cooldown - 1
                    
                    # Display the status on the image
                    # Get bounding box for text position
                    box = results[0].boxes.xyxy[person_idx].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Choose color based on state
                    color = (0, 255, 0) # Green for Standing
                    if stage == "Sitting":
                        color = (0, 255, 255) # Yellow
                    elif stage == "Bending":
                        color = (255, 165, 0) # Orange
                    elif stage == "Legs Not Visible":
                        color = (0, 0, 255) # Red
                    elif stage == "Jumping":
                        color = (255, 0, 0) # Blue-ish for Jumping
                    
                    cv2.putText(annotated_frame, f"{stage} ({int(avg_angle)})", 
                                (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        except Exception as e:
            # Just print error and continue, don't crash stream
            print(f"Error processing pose: {e}")

        # Display the annotated frame
        cv2.imshow(window_name, annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()