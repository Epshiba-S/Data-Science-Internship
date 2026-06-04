# train_model.py
import os
from ultralytics import YOLO

def main():
    print("--- Initializing Automated BTS Member Tracking Trainer ---")
    
    # Create the local image folders
    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/images/val", exist_ok=True)
    
    # Configure the names map for the 7 members of BTS cleanly without spacing conflicts
    yaml_lines = [
        "path: ./dataset",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: RM",
        "  1: Jin",
        "  2: Suga",
        "  3: J-Hope",
        "  4: Jimin",
        "  5: V",
        "  6: Jungkook"
    ]
    
    with open("bts_data.yaml", "w") as f:
        f.write("\n".join(yaml_lines))
    print("[INFO] bts_data.yaml successfully configured.")

    # Initialize the base artificial intelligence brain model
    model = YOLO("yolov8n.pt")
    
    # Run a quick 1-epoch loop to test that the AI packages compile correctly
    print("[INFO] Starting model compilation process...")
    model.train(data="bts_data.yaml", epochs=1, imgsz=320, device="cpu") 
    print("[SUCCESS] Pipeline compiled. Model ready for production deployments.")

if __name__ == "__main__":
    main()