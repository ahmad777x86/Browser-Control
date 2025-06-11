import cv2
import mediapipe as mp
import pyautogui as pt
import time

# Declarations and Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
)

drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

# Helper functions
def is_finger_extended(hand_landmarks,finger_tip_id,finger_pip_id):
    if hand_landmarks:
        tip_y = hand_landmarks.landmark[finger_tip_id].y
        pip_y = hand_landmarks.landmark[finger_pip_id].y
        return tip_y < pip_y - 0.04
    return False

def is_thumb_extended(hand_landmarks):
    if hand_landmarks:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
        return thumb_tip < thumb_mcp - 0.04
    return False

def open_tab_gesture(hand_landmarks):
    if hand_landmarks:
        index = is_finger_extended(hand_landmarks,mp_hands.HandLandmark.INDEX_FINGER_TIP,mp_hands.HandLandmark.INDEX_FINGER_PIP)
        middle = is_finger_extended(hand_landmarks,mp_hands.HandLandmark.MIDDLE_FINGER_TIP,mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
        ring = is_finger_extended(hand_landmarks,mp_hands.HandLandmark.RING_FINGER_TIP,mp_hands.HandLandmark.RING_FINGER_PIP)
        pinky = is_finger_extended(hand_landmarks,mp_hands.HandLandmark.PINKY_TIP,mp_hands.HandLandmark.PINKY_PIP)

        thumb = is_thumb_extended(hand_landmarks)
        return index and middle and ring and pinky and thumb
    return False

def close_tab_gesture(hand_landmarks):
    if hand_landmarks:
        index = not is_finger_extended(hand_landmarks,mp_hands.HandLandmark.INDEX_FINGER_TIP,mp_hands.HandLandmark.INDEX_FINGER_PIP)
        middle = not is_finger_extended(hand_landmarks,mp_hands.HandLandmark.MIDDLE_FINGER_TIP,mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
        ring = not is_finger_extended(hand_landmarks,mp_hands.HandLandmark.RING_FINGER_TIP,mp_hands.HandLandmark.RING_FINGER_PIP)
        pinky = not is_finger_extended(hand_landmarks,mp_hands.HandLandmark.PINKY_TIP,mp_hands.HandLandmark.PINKY_PIP)

        thumb = not is_thumb_extended(hand_landmarks)
        return index and middle and ring and pinky and thumb
    return False

# Parameters
last_action_time = time.time()

while True:
    ret , frame = camera.read()
    
    im_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = hands.process(im_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing.draw_landmarks(
                image = frame,
                landmark_list = hand_landmarks,
                connections = mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec = drawing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius = 2),
                connection_drawing_spec = drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius = 2)
            )
        if(open_tab_gesture(hand_landmarks=hand_landmarks) and time.time() - last_action_time > 2.0):
            last_action_time = time.time()
            pt.hotkey("ctrl","t")
            print("open")
        elif(close_tab_gesture(hand_landmarks=hand_landmarks) and time.time() - last_action_time > 2.0):
            last_action_time = time.time()
            pt.hotkey("ctrl","w")
            print("close")

    cv2.imshow("My webcam",frame)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
hands.close()