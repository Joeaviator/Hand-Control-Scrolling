import cv2
import time
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

filters = [None, 'GRAYSCALE', 'SEPIA', 'NEGATIVE', 'BLUR']
current_filter = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

last_action_time = 0
debounce_time = 1  # seconds

def apply_filter(frame, filter_type):
    if filter_type == 'GRAYSCALE':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'SEPIA':
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_frame = cv2.transform(frame, sepia_filter)
        sepia_frame = np.clip(sepia_frame, 0, 255)
        return sepia_frame.astype(np.uint8)
    elif filter_type == 'NEGATIVE':
        return cv2.bitwise_not(frame)
    elif filter_type == 'BLUR':
        return cv2.GaussianBlur(frame, (15, 15), 0)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read the frame")
        break

    img = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            frame_height, frame_width, _ = img.shape

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
            index_tip_x, index_tip_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
            middle_tip_x, middle_tip_y = int(middle_tip.x * frame_width), int(middle_tip.y * frame_height)
            ring_tip_x, ring_tip_y = int(ring_tip.x * frame_width), int(ring_tip.y * frame_height)
            pinky_tip_x, pinky_tip_y = int(pinky_tip.x * frame_width), int(pinky_tip.y * frame_height)

            current_time = time.time()

            # Gesture 1: Thumb is highest -> Next filter
            if (thumb_tip_y < index_tip_y and thumb_tip_y < middle_tip_y and
                thumb_tip_y < ring_tip_y and thumb_tip_y < pinky_tip_y):
                if current_time - last_action_time > debounce_time:
                    current_filter = (current_filter + 1) % len(filters)
                    last_action_time = current_time

            # Gesture 2: Pinky is highest -> Previous filter
            elif (pinky_tip_y < ring_tip_y and pinky_tip_y < middle_tip_y and
                  pinky_tip_y < index_tip_y and pinky_tip_y < thumb_tip_y):
                if current_time - last_action_time > debounce_time:
                    current_filter = (current_filter - 1) % len(filters)
                    last_action_time = current_time

    filtered_img = apply_filter(img, filters[current_filter])

    if filters[current_filter] == 'GRAYSCALE':
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

    cv2.putText(filtered_img, f'Filter: {filters[current_filter]}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Webcam Filter', filtered_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
