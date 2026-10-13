# Step 3 of face recognition: recognise known faces in the live camera feed
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2

# Print a name only when a new person is identified
currentname = "unknown"

# Encodings created by train_model.py
encodingsP = "encodings.pickle"

print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Start the Pi camera stream (use VideoStream(src=0) for a USB webcam)
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

fps = FPS().start()

while True:
    # Grab a frame and resize to 500px wide for speed
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # Detect faces and compute embeddings
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    for encoding in encodings:
        # Compare each face with the known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            # Count votes for each matched person
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Pick the person with the most votes
            name = max(counts, key=counts.get)

            if currentname != name:
                currentname = name
                print(currentname)

        names.append(name)

    # Draw boxes and names
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit when 'q' is pressed
    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
