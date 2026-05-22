import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Using 'with' handles resource allocation and cleanup automatically
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break

        # Convert the BGR image to RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find faces
        results = face_detection.process(rgb)

        # Draw the face detection annotations on the original BGR frame.
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        cv2.imshow("MediaPipe Face Detection", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()