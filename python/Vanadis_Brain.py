from ultralytics import YOLO
import cv2 as cv
import os

# Model
model = YOLO("best.pt")

# Confidence threshold
confidence_threshold = 0.70

# image paths array from folder
image_paths = []
folder_path = "Final_blue/test10/test1_RawImages/80cm"
for filename in os.listdir(folder_path):
    full_path = os.path.join(folder_path, filename)
    if os.path.isfile(full_path):
        image_paths.append(full_path)

cropped = []


# Run model
print("\nBlue Light detection")
results = model(image_paths, conf=confidence_threshold)

# Output folder
save = "Final_blue/test10/test1_results/80cm"
os.makedirs(save, exist_ok=True)

# Process each image
for idx, result in enumerate(results):
    
    path = image_paths[idx]
    print(f"\n Image {idx + 1}")

    boxes = result.boxes

    # Count detections
    if boxes is None: 
      count = 0 
    else: 
      count = len(boxes)
    print(f"Biolmunesence detected: {count}")

    
    plotted = result.plot()   # image with labels drawn
    save_path = os.path.join(save, os.path.basename(path))
    cv.imwrite(save_path, plotted)

    # Show image with detections
    #result.show()