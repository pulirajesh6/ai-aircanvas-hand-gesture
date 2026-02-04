import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
draw_utils = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0
color = (255, 0, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw color buttons
    cv2.rectangle(frame, (10,10), (110,60), (255,0,0), -1)
    cv2.rectangle(frame, (130,10), (230,60), (0,255,0), -1)
    cv2.rectangle(frame, (250,10), (350,60), (0,0,255), -1)
    cv2.rectangle(frame, (370,10), (470,60), (0,0,0), -1)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            tip = hand.landmark[8]
            pip = hand.landmark[6]

            x = int(tip.x * w)
            y = int(tip.y * h)

            cv2.circle(frame, (x,y), 8, (0,255,255), -1)

            index_up = tip.y < pip.y

            # Color button selection
            if y < 60:
                if 10 < x < 110:
                    color = (255,0,0)
                elif 130 < x < 230:
                    color = (0,255,0)
                elif 250 < x < 350:
                    color = (0,0,255)
                elif 370 < x < 470:
                    canvas = np.zeros_like(frame)

                prev_x, prev_y = 0, 0

            # Draw only when index finger up
            elif index_up:
                if prev_x == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), color, 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0, 0

    frame = cv2.add(frame, canvas)

    cv2.imshow("AI AirCanvas Pro", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
