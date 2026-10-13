# third-eye-for-blind
SOE Mini Project


Visually impaired people depend on others to recognise objects, people and text around them. We built a low-cost assistive device on a Raspberry Pi with a camera module. It runs several computer vision models on the device and announces what it sees through audio.

AI/ML features built:

Object detection: SSD MobileNet v3, pre-trained on the COCO dataset (80 object classes), runs on live camera frames through the OpenCV DNN module. It uses a 0.45 confidence threshold and non-max suppression to cut duplicate detections.
Face recognition: We captured a custom dataset of family members’ faces. The face_recognition library (built on dlib) turns each face into a 128-d encoding, saved with pickle. In real time, OpenCV’s Haar cascade finds faces and each one is matched against the saved encodings, so the device can say who is in front of the user.
Hand gesture recognition: MediaPipe Hands tracks 21 hand landmarks in real time, giving touch-free gesture commands (forward, back, left, right, stop).
Voice interface and text-to-speech: The SpeechRecognition library (Google Speech API) takes spoken queries. TextBlob translates them and Wikipedia API answers short queries. gTTS reads results aloud in English, Hindi and Spanish. Tesseract OCR reads printed text aloud.
