import cv2
from ultralytics import YOLO

# Load the nano YOLOv8 model (lightweight and great for real-time webcams)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Stream=True uses a generator, dramatically reducing memory overhead for video feeds.
    # Verbose=False keeps your terminal clean so you can see your own print statements.
    results = model(frame, stream=True, verbose=False)

    for r in results:
        # Plot the bounding boxes and labels onto the frame
        annotated_frame = r.plot()
        
        # Display the results
        cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()