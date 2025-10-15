import cv2
import numpy as np
import math
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import torch
import os
import time
import keyboard
import json
import serial
import time

#capture = cv2.VideoCapture("http://192.168.68.103:8080/video")
capture = cv2.VideoCapture(0)


processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

serial_port = 'COM6'
pico_serial = serial.Serial(serial_port, 115200, timeout=1)
time.sleep(2)

base_servo_angle = 50

def nothing(x):
    return

print("Press 'a' to capture an image and detect objects.")
print("Press 'q' to quit.")

annotated_frame = None

keyboard.on_press_key('a', lambda e: on_a_press(e))
def on_a_press(event):
    global annotated_frame
    print("a pressed")
    ret, frame = capture.read()
    if not ret:
        return

    frame = cv2.resize(frame, (1400, 720))

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    texts = [["pen", "pencil", "eraser", "sharpener"]]
    inputs = processor(text=texts, images=img, return_tensors="pt")
    outputs = model(**inputs)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=[img.size[::-1]], threshold=0.01)[0]
    
    # Check if any objects were detected
    if len(results["scores"]) == 0:
        print("No objects detected.")
        return
    
    highest_score_idx = results["scores"].argmax().item()
    object_label = results["labels"][highest_score_idx].item() #.item() to get value from tensor
    box = results["boxes"][highest_score_idx].tolist()
    print(f"Object: {texts[0][object_label]}, Score: {results['scores'][highest_score_idx].item():.3f}")
    print(f"Box Coordinates: {box}")

    object_center_x = (box[0] + box[2]) / 2
    object_center_y = (box[1] + box[3]) / 2

    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    cv2.circle(frame, (int(object_center_x), int(object_center_y)), 5, (255, 0, 0), -1)
    cv2.putText(frame, texts[0][object_label], (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    height, width = frame.shape[:2]
    frame_center_x = width // 2
    frame_center_y = height // 2

    distance_x = object_center_x - frame_center_x
    distance_y = object_center_y - frame_center_y

    angle_rad = math.atan2(distance_y, distance_x)
    angle_deg = math.degrees(angle_rad)
    global base_servo_angle
    base_servo_angle = 90 - angle_deg
    if base_servo_angle < 0:
        base_servo_angle = 0
    elif base_servo_angle > 180:
        base_servo_angle = 180

    distance = math.hypot(distance_x, distance_y)

    cv2.line(frame, (frame_center_x, frame_center_y), (int(object_center_x), int(object_center_y)), (0, 0, 255), 2)
    cv2.putText(frame, f"Angle: {angle_deg:.2f} degrees", (int(object_center_x), int(object_center_y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    annotated_frame = frame.copy()

# Main loop to keep the program running
while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1400, 720))
    
    # Display live feed
    cv2.imshow("Live Feed", frame)
    
    # Display annotated frame in a separate window if available
    if annotated_frame is not None:
        cv2.imshow("Detected Object", annotated_frame)

    instructions = {"angle": base_servo_angle, "servo": 1}
    pico_serial.write(f"{json.dumps(instructions)}\n".encode())
    pico_serial.flush()

    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
