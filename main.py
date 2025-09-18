import cv2
import mediapipe as mp
import time,pyautogui
#Initialize mediapipe
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.9)
mp_drawing=mp.solutions.drawing_utils
#Configuration
Scroll_speed=300
scroll_delay=1
Cam_width,Cam_height=640, 480
#Function to detect gesture
def detect_gesture(landmarks,handedness):
    fingers=[]
    #tips of the fingers
    tips=[mp_hands.Handlandmark.INDEX_FINGER_TIP,mp_hands.Handlandmark.MIDDLE_FINGER_TIP, mp_hands.Handlandmark.RING_FINGER_TIP, mp_hands.Handlandmark.PINKY_FINGER_TIP]
    #Check for the fingers except the Thumb
    for tip in tips:
        if landmarks[tips].y<landmarks[tips-2].y:
            fingers.append(1)
    #Thumb
    thumb_tip=landmarks.landmark[mp_hands.Handlandmark.THUMB_TIP]
    thumb_ip=landmarks.landmark[mp_hands.Handlandmark.THUMB_IP] 
    if (handedness=="Right" and thumb_tip.x>thumb_ip.x) or (handedness=="Left" and thumb_tip.x<thumb_ip.x):
        fingers.append(1)
    #Scroll up
    return "scroll_up" if sum(fingers)==5 else "scroll_down" if len(fingers)==0 else None
#Initialize Camera
cap=cv2.VideoCapture(0)
cap.set(3,Cam_width)
cap.set(4,Cam_height)
last_scroll_time = p_time=0
print("Gesture Control Scrolling(Yeah We advanced😎😎😂). \nOpen palm: Scroll UP. \nClose pal: Scroll Down \nPress Q to quit")
while cap.isOpened():
    sucess,image=cap.read()
    if not sucess:
        print("Error😐😐😭")
        break
    image=cv2.flip(cv2.cvtColor(image.cv2.COLOR_BGR2RGB),1)
    results=hands.process(image)
    gestures,handedness="none","Unknown"
    if results.multi_hand_landmarks:
        for hand,handedness_info in zip(results.multi_hand_landmarks,results.multi_handedness):
           handedness=handedness_info.classification[0].label
           gestures=detect_gesture(hand,handedness) 
           mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS)
           if (time.time()-last_scroll_time)>scroll_delay: 
               if gestures=="scroll_up":
                   pyautogui.scroll(Scroll_speed)
               elif gestures=="scroll_down":
                pyautogui.scroll(-Scroll_speed)
               last_scroll_time=time.time()
    fps=1/(time.time()-p_time) if(time.time()-p_time)>0 else 0 
    p_time=time.time()
    cv2.putText(image,f"FPS:{int(fps)} | Hand:{handedness} | Gesture: {gestures}", (10,30), cv2.FONT_HERSHEY_DUPLEX,0.7,(200,155,30),3)
    cv2.imshow("Gesture Controlled Scrolling", image)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
