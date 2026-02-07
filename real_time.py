import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np

from model import GestureNet

model = GestureNet()
model.load_state_dict(torch.load("gesture.pth"))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

ok_img = cv2.resize(cv2.imread("pic/Добрый.png"), (450, 450))
angry_img = cv2.resize(cv2.imread("pic/Ахуевший.png"), (450, 450))

cap = cv2.VideoCapture(0)
history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    gesture = 0

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(x)
            gesture = pred.argmax(1).item()

        history.append(gesture)
        if len(history) > 7:
            history.pop(0)
        gesture = max(set(history), key=history.count)

    
    h, w, _ = frame.shape
    monkey = angry_img if gesture == 1 else ok_img
    frame[0:450, w-450:w] = monkey

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()