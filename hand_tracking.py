# Real-time hand tracking using MediaPipe Hands
import mediapipe
import cv2

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

cap = cv2.VideoCapture(0)

# Live video mode; track up to two hands with 70% detection/tracking confidence
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7,
                       min_tracking_confidence=0.7, max_num_hands=2) as hands:
    while True:
        ret, frame = cap.read()
        # Uncomment if the feed is upside down
        # frame = cv2.flip(frame, flipCode=-1)

        # 640 x 480 balances speed and accuracy
        frame1 = cv2.resize(frame, (640, 480))

        results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

        # Draw the 21-point hand landmarks on every detected hand
        if results.multi_hand_landmarks is not None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)

                # Optional: print the pixel coordinates of the index fingertip (landmark 8)
                # for point in handsModule.HandLandmark:
                #     normalizedLandmark = handLandmarks.landmark[point]
                #     pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(
                #         normalizedLandmark.x, normalizedLandmark.y, 640, 480)
                #     if point == 8:
                #         print(point, pixelCoordinatesLandmark, normalizedLandmark)

        cv2.imshow("Frame", frame1)
        key = cv2.waitKey(1) & 0xFF

        # Press 'q' to stop
        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
