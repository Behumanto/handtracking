import cv2
import mediapipe as mp
import pyautogui as pag
import numpy as np
import time

# Initialize Mediapipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.1, min_tracking_confidence=0.1)
mp_drawing = mp.solutions.drawing_utils



# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened")
    exit()

# Set screen resolution
screen_width, screen_height = pag.size()

# Variables for managing click actions
mouse_down = False
last_release_time = 0
release_delay = 0.3  # Delay after releasing a click to prevent re-triggering
pinch_start_time = None  # Store the start time of the pinch

# Variables for smoothing cursor movement
last_x, last_y = 0, 0
smoothing_factor = 0.2
 
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            midpoint_x = (index_finger_tip.x + thumb_tip.x) / 2
            midpoint_y = (index_finger_tip.y + thumb_tip.y) / 2

            cursor_x = np.interp(midpoint_x, (0, 1), (0, screen_width))
            cursor_y = np.interp(midpoint_y, (0, 1), (0, screen_height))

            # Apply smoothing
            cursor_x = last_x + (cursor_x - last_x) * smoothing_factor
            cursor_y = last_y + (cursor_y - last_y) * smoothing_factor
            last_x, last_y = cursor_x, cursor_y

            pag.moveTo(cursor_x, cursor_y, duration=0.01)

            distance = np.sqrt((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2)

            if distance >= 0.1:
                cv2.circle(frame, (int(midpoint_x * frame_width), int(midpoint_y * frame_height)), 10, (0, 255, 0), 2)

            if distance < 0.1 and not mouse_down:
                if time.time() - last_release_time > release_delay:
                    pag.mouseDown()
                    mouse_down = True
                    pinch_start_time = time.time()
            elif distance > 0.1 and mouse_down:
                pinch_duration = time.time() - pinch_start_time
                pag.mouseUp()
                mouse_down = False
                last_release_time = time.time()
                if pinch_duration <= 1:
                    pag.click()

    cv2.imshow("Medipipe Hands", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
