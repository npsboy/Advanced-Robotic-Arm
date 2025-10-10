import cv2
import numpy as np
import math
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import requests
import torch
import os
import time

capture = cv2.VideoCapture("http://192.168.68.103:8080/video")



def nothing(x):
    return


cv2.namedWindow('Trackbars')
cv2.createTrackbar("H_low", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("H_high", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S_low", "Trackbars", 113, 255, nothing)
cv2.createTrackbar("S_high", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V_low", "Trackbars", 130, 255, nothing)
cv2.createTrackbar("V_high", "Trackbars", 200, 255, nothing)


while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1400, 720))

    height, width = frame.shape[:2]
    frame_center_x = width // 2
    frame_center_y = height // 2

    blurred_frame = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # HSV color: (6.19, 69.29%, 54.9%) -> (6, 177, 140) in OpenCV HSV (H:0-179, S:0-255, V:0-255)
    #lower_color = (3, 140, 100)
    #upper_color = (10, 255, 180)
    h_low = cv2.getTrackbarPos("H_low", "Trackbars")
    h_high = cv2.getTrackbarPos("H_high", "Trackbars")
    s_low = cv2.getTrackbarPos("S_low", "Trackbars")
    s_high = cv2.getTrackbarPos("S_high", "Trackbars")
    v_low = cv2.getTrackbarPos("V_low", "Trackbars")
    v_high = cv2.getTrackbarPos("V_high", "Trackbars")

    lower_color = np.array([h_low, s_low, v_low])
    upper_color = np.array([h_high, s_high, v_high])

    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(largest_contour)
            object_cord_x = x + w // 2
            object_cord_y = y + h // 2
            cv2.circle(frame, (object_cord_x, object_cord_y), 5, (255, 0, 0), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Object Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            distance_x = object_cord_x - frame_center_x
            distance_y = object_cord_y - frame_center_y
            shortest_distance = math.hypot(distance_x, distance_y)
            cv2.line(frame, (frame_center_x, frame_center_y), (object_cord_x, object_cord_y), (0, 0, 255), 2)

            angle_rad = math.atan2(distance_y, distance_x)
            angle_deg = math.degrees(angle_rad)
            cv2.putText(frame, f"Angle: {angle_deg:.2f} degrees", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Original Frame', frame)
    cv2.imshow('Mask', mask)


    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()