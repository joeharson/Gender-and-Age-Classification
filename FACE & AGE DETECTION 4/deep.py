import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture('rtsp://admin:Admin@123@103.110.239.201:560/live')

# Frame skipping parameters
frame_skip = 1  # Process every 5th frame
frame_count = 0

# Variables to cache results
last_results = None

# Rescale factor (adjust as needed)
rescale_factor = 0.5  # Resize to 50% of the original size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Perform age and gender detection only on every nth frame
    if frame_count % frame_skip == 0:
        try:
            # Analyze the frame for age and gender, and also get face regions
            results = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False)
            last_results = results  # Cache the results for skipped frames
        except Exception as e:
            print(f"Error: {e}")
            last_results = None

    # Use cached results for skipped frames
    if last_results:
        for result in last_results:
            age = result['age']
            gender = result['gender']  # This is a dictionary with gender probabilities
            face_region = result['region']  # Get the bounding box of the face

            # Extract the gender with the highest confidence score
            dominant_gender = max(gender, key=gender.get)  # Get the key (gender) with the highest value (confidence)
            confidence = gender[dominant_gender]  # Get the confidence score for the dominant gender

            # Draw bounding box around the face
            x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display results on the frame
            cv2.putText(frame, f"Age: {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Gender: {dominant_gender} ({confidence:.2f})", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Resize the frame
    resized_frame = cv2.resize(frame, (0, 0), fx=rescale_factor, fy=rescale_factor)

    # Show the resized frame
    cv2.imshow("Age and Gender Detection", resized_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()