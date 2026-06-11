import cv2
import numpy as np

# Load face, age, and gender detection models
face_net = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
age_net = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

# Mean values for age and gender model normalization
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Labels for age and gender
AGE_LABELS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LABELS = ['Male', 'Female']

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame width and height
    frame_height, frame_width = frame.shape[:2]

    # Convert frame to a blob for the face detector
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            # Get bounding box coordinates
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)

            # Ensure the face is within the frame
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)

            # Extract face region
            face = frame[y1:y2, x1:x2]

            # Process only if face is detected properly
            if face.shape[0] > 0 and face.shape[1] > 0:
                # Convert face to blob
                face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                # Predict gender
                gender_net.setInput(face_blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LABELS[np.argmax(gender_preds)]

                # Predict age
                age_net.setInput(face_blob)
                age_preds = age_net.forward()
                age = AGE_LABELS[np.argmax(age_preds)]

                # Display the age and gender on the video feed
                label = f"{gender}, {age}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Age & Gender Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
