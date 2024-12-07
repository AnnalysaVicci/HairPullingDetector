import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os

# Initialize MediaPipe Pose and Hand solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Variables for hair twirl detection
hand_positions = []  # To store previous hand positions
hair_twirling_detected = False
twirl_threshold = 100  # Minimum distance for hair twirl motion (adjust as necessary)
time_interval = 3  # Time interval in seconds to detect a twirl

# Initialize sound file and alert cooldown
sound_file = "alert_sound.mp3"
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
        # (Pose detection code here as in your original code...)
        
    # Hand detection: Check for hand landmarks and calculate movement
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Extract wrist and finger positions
            wrist = (int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1]),
                     int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0]))
            thumb = (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * frame.shape[1]),
                     int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * frame.shape[0]))
            
            # Calculate distance between wrist and thumb to track circular motion
            if len(hand_positions) > 0:
                distance = calculate_distance(hand_positions[-1], wrist)
                if distance > twirl_threshold:
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        print("Hair twirling detected!")
                        if os.path.exists(sound_file):
                            playsound(sound_file)
                        last_alert_time = current_time

            # Store current wrist position
            hand_positions.append(wrist)
            # Keep the list of positions within a limit (e.g., last 10 positions)
            if len(hand_positions) > 10:
                hand_positions.pop(0)

            # Draw landmarks for hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hair Twirl Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
