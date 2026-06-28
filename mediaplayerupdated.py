import cv2 
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. Keyboard / PyAutoGUI Setup ---
use_keyboard = True
try:
    import keyboard
except ImportError:
    use_keyboard = False
    import pyautogui

def send_keys(seq: str):
    if use_keyboard:
        keyboard.send(seq)
    else:
        parts = seq.split('+')
        pyautogui.hotkey(*parts) if len(parts) > 1 else pyautogui.press(parts[0])


# --- 2. Initialize MediaPipe Tasks API ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)


# --- 3. Custom Drawing Function (Replaces mp.solutions.drawing_utils) ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def draw_hand_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    # Draw connections
    for connection in HAND_CONNECTIONS:
        pt1 = landmarks[connection[0]]
        pt2 = landmarks[connection[1]]
        cx1, cy1 = int(pt1.x * w), int(pt1.y * h)
        cx2, cy2 = int(pt2.x * w), int(pt2.y * h)
        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
    # Draw keypoints
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)


# --- 4. Gesture Detection Logic (Math remains the same) ---
def detect_gestures(landmark):
    thumb_tip = landmark[4]
    index_tip = landmark[8]
    middle_tip = landmark[12]
    ring_tip = landmark[16]
    pinky_tip = landmark[20]
    wrist = landmark[0]

    tolerance = 0.02

    fingers = {
         "index": index_tip.y < landmark[6].y - tolerance, 
         "middle": middle_tip.y < landmark[10].y - tolerance,
         "ring": ring_tip.y < landmark[14].y - tolerance,
         "pinky": pinky_tip.y < landmark[18].y - tolerance,
    }

    if all(fingers.values()) and thumb_tip.y < wrist.y:
         return "Open"
    elif not any(fingers.values()):
         return "Fist"
    elif thumb_tip.y < landmark[2].y:
         return "Thumbs_Up"
    elif thumb_tip.y > landmark[2].y:
         return "Thumbs_Down"
    elif fingers["index"] and not (fingers["middle"] or fingers["pinky"] or fingers["ring"]):
         if index_tip.x < wrist.x:
              return "Pointing_Left"
         else:
              return "Pointing_Right"
    else:
         return "Unknown"

def control_media(gesture):
    if gesture == "Fist":
        print(" ▶️ space")
        send_keys("space")
    elif gesture == "Open":
         print("Shift+n")
         send_keys("shift+n")
    elif gesture == "Thumbs_Up":
         print("🔊 up")
         send_keys("up")
    elif gesture == "Thumbs_Down":
         print("🔈 down")
         send_keys("down")


# --- 5. Main Video Loop ---
cap = cv2.VideoCapture(0)

last_sent = None
last_time = 0.0
cooldown = 10.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # NEW API: Convert to mp.Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # NEW API: Run detection
    res = detector.detect(mp_image)

    gesture = "No_hands"

    # NEW API: Check for hand_landmarks instead of multi_hand_landmarks
    if res.hand_landmarks:
        for hand_landmarks in res.hand_landmarks:
            # Draw the skeleton using our custom function
            draw_hand_skeleton(frame, hand_landmarks)
            
            # Detect the gesture
            gesture = detect_gestures(hand_landmarks)
            print("Detected gesture", gesture)

            # Display gesture on screen
            cv2.putText(frame, gesture, (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Cooldown & Execution Logic
    now = time.time()
    if gesture not in ("Unknown", "No_hands"):
        if (gesture != last_sent) or (now - last_time > cooldown):
             control_media(gesture)
             last_sent = gesture
             last_time = now

    cv2.imshow("Window", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()