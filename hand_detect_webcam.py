import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_myhand = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    success, img = webcam.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_myhand.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    x_min, x_max, y_min, y_max = float('inf'), 0, float('inf'), 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', img)
    #print(x_min, y_min,x_max, y_max)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
