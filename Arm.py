import cv2

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    blurred_frame = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # HSV color: (6.19, 69.29%, 54.9%) -> (6, 177, 140) in OpenCV HSV (H:0-179, S:0-255, V:0-255)
    lower_color = (3, 140, 100)
    upper_color = (10, 255, 180)

    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cord_x = x + w // 2
            cord_y = y + h // 2
            cv2.circle(frame, (cord_x, cord_y), 5, (255, 0, 0), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Object Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Original Frame', frame)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()