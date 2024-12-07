import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os

# Initialize MediaPipe Pose, Hand and Drawing solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    angle = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    return np.abs(angle)

# Variables for hair twirl detection
index_finger_positions = []  # To store previous index finger positions
twirl_threshold = 90  # Minimum distance for hair twirl motion (adjust as necessary)
time_interval = 3  # Time interval in seconds to detect a twirl
twirl_angle_threshold = 1.5  # Threshold for detecting circular motion (angle change)

# Initialize sound file and alert cooldown
sound_file = "ding_sound"
alert_cooldown = 2
last_alert_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb_frame)
    results_hands = hands.process(rgb_frame)

    # Check if pose landmarks are available
    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        # (Pose detection code can stay the same if needed...)
        
    # Hand detection: Check for hand landmarks and calculate movement
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Extract index finger tip position
            index_finger_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]),
                                int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]))
            
            # Calculate distance between current and previous index finger positions to detect movement
            if len(index_finger_positions) > 1:
                # Check if the index finger is moving in a circular motion
                prev_position = index_finger_positions[-2]
                current_position = index_finger_positions[-1]
                
                # Calculate the angle between the previous, current, and next positions to detect circular motion
                if len(index_finger_positions) > 2:
                    next_position = index_finger_positions[-3]
                    angle = calculate_angle(prev_position, current_position, next_position)
                    
                    # If the angle change is significant, assume circular motion
                    if np.abs(angle) > twirl_angle_threshold:
                        current_time = time.time()
                        if current_time - last_alert_time > alert_cooldown:
                            print("Hair twirling detected!")
                            #playsound("/Users/annalysa/Videos/ding_sound")
                            if os.path.exists(sound_file):
                                playsound(sound_file)  # Play sound when twirling is detected
                            last_alert_time = current_time

            # Store current index finger position
            index_finger_positions.append(index_finger_tip)

            # Keep the list of positions within a limit (e.g., last 10 positions)
            if len(index_finger_positions) > 10:
                index_finger_positions.pop(0)

            # Draw landmarks for hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hair Twirl Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
