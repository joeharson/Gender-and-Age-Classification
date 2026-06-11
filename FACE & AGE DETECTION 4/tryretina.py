import cv2
from deepface import DeepFace
from retinaface import RetinaFace

# Initialize webcam
cap = cv2.VideoCapture(0)

# Rescale factor (adjust as needed)
rescale_factor = 0.5  # Resize to 50% of the original size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing
    resized_frame = cv2.resize(frame, (0, 0), fx=rescale_factor, fy=rescale_factor)

    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(resized_frame)

    if isinstance(faces, dict):  # Check if faces are detected
        for face_id, face_data in faces.items():
            facial_area = face_data['facial_area']
            x, y, w, h = facial_area[0], facial_area[1], facial_area[2] - facial_area[0], facial_area[3] - facial_area[1]

            # Crop the face region
            face_img = resized_frame[y:y+h, x:x+w]

            # Analyze the face for age and gender using DeepFace
            try:
                results = DeepFace.analyze(face_img, actions=['age', 'gender'], enforce_detection=False)
                for result in results:
                    age = result['age']
                    gender = result['gender']
                    dominant_gender = max(gender, key=gender.get)
                    confidence = gender[dominant_gender]

                    # Draw bounding box and display results
                    cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(resized_frame, f"Age: {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(resized_frame, f"Gender: {dominant_gender} ({confidence:.2f})", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error in DeepFace analysis: {e}")

    # Show the resized frame
    cv2.imshow("Age and Gender Detection", resized_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()