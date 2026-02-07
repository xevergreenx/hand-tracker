import cv2
import os 
import mediapipe as mp
import numpy as np 

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
        static_image_mode = False,
        max_num_hands = 2,
        min_detection_confidence = 0.7,
    )

cap = cv2.VideoCapture(0)

X = []
Y = []

print(
'''
      0 - normal 
      1 - fuck 
      q - save and exit
''')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=8),
            mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=5)
        )

        for lm in hand.landmark:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    cv2.imshow("Collect gestures", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('0') and result.multi_hand_landmarks:
        X.append(landmarks)
        Y.append(0)
        print("Saved: normal")

    if key == ord('1') and result.multi_hand_landmarks:
        X.append(landmarks)
        Y.append(1)
        print("Saved: fuck")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

os.makedirs("data", exist_ok=True)
np.save("data/X.npy", np.array(X))
np.save("data/Y.npy", np.array(Y))

print(f"Saved dataset: {len(X)} samples")

