import cv2
import mediapipe as mp
import time
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils

Scroll_speed = 300
scroll_delay = 1
Cam_width, Cam_height = 640, 480

def detect_gesture(landmarks, handedness):
    fingers = []
    tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    for tip in tips:
        if landmarks.landmark[tip].y < landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    if (handedness == "Right" and thumb_tip.x > thumb_ip.x) or \
       (handedness == "Left" and thumb_tip.x < thumb_ip.x):
        fingers.append(1)
    else:
        fingers.append(0)
    total = sum(fingers)
    if total == 5:
        return "scroll_up"
    elif total == 0:
        return "scroll_down"
    else:
        return None

cap = cv2.VideoCapture(0)
cap.set(3, Cam_width)
cap.set(4, Cam_height)

last_scroll_time = 0
p_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)
    results = hands.process(image)
    gestures, handedness = "none", "Unknown"
    if results.multi_hand_landmarks:
        for hand, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            handedness = handedness_info.classification[0].label
            gestures = detect_gesture(hand, handedness)
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
            print("Hand:", handedness, "| Gesture:", gestures)
            if (time.time() - last_scroll_time) > scroll_delay:
                if gestures == "scroll_up":
                    pyautogui.scroll(Scroll_speed)
                elif gestures == "scroll_down":
                    pyautogui.scroll(-Scroll_speed)
                last_scroll_time = time.time()
    fps = 1 / (time.time() - p_time) if (time.time() - p_time) > 0 else 0
    p_time = time.time()
    cv2.putText(image, f"FPS: {int(fps)} | Hand: {handedness} | Gesture: {gestures}",
                (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 155, 30), 2)
    cv2.imshow("Gesture Controlled Scrolling", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
