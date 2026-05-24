import os
import ssl
# Bypass SSL issues for model downloading
os.environ["YOLO_VERBOSE"] = "False"
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # 1. We removed classes=[0] so YOLO can detect the object correctly!
    results = model(frame, stream=True, verbose=False)

    best_box = None
    max_conf = 0.0

    for r in results:
        # Loop through all detected objects in the frame
        for box in r.boxes:
            conf = float(box.conf[0])  # Get the confidence score
            
            # 2. Keep track of ONLY the object with the highest confidence
            if conf > max_conf:
                max_conf = conf
                best_box = box
        
        # 3. Only draw a bounding box if the confidence is above 40%
        if best_box is not None and max_conf > 0.40:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            
            # Get the correct name of the object
            class_id = int(best_box.cls[0])
            object_name = model.names[class_id]
            
            # Create the text label (e.g., "bottle 87%")
            label = f"{object_name} {max_conf:.2f}"
            
            # Draw the bounding box (Blue rectangle)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw the background label box and text
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display the final frame with exactly ONE object detected
    cv2.imshow("YOLOv8 - Absolute Best Single Object Only", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()