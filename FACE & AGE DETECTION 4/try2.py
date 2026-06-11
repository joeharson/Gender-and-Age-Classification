from flask import Flask, render_template, Response, jsonify
import cv2
from deepface import DeepFace
import json
from datetime import datetime
from collections import defaultdict
import threading
import queue
import time
import numpy as np

app = Flask(__name__)

# Global variables
frame_skip = 3
frame_count = 0
last_results = None
rescale_factor = 0.7
detection_history = []
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue()

statistics = {
    'total_detections': 0,
    'gender_stats': defaultdict(int),
    'age_groups': defaultdict(int),
    'average_confidence': 0,
    'last_updated': None
}

def verify_face_detection(result):
    """Verify if the detection is valid"""
    try:
        # Check if region exists and has valid dimensions
        if 'region' not in result:
            return False
        
        region = result['region']
        if not all(key in region for key in ['x', 'y', 'w', 'h']):
            return False
            
        # Check if the detected region has reasonable dimensions
        if region['w'] < 20 or region['h'] < 20:  # Too small to be a face
            return False
            
        # Check confidence score
        if 'gender' in result and isinstance(result['gender'], dict):
            confidence = max(result['gender'].values())
            if confidence < 0.6:  # Minimum confidence threshold
                return False
                
        return True
    except:
        return False

def update_statistics(result):
    global statistics
    try:
        statistics['total_detections'] += 1
        
        # Handle gender stats
        if 'gender' in result:
            gender = result['gender']
            statistics['gender_stats'][gender] += 1
        
        # Handle age groups
        if 'age' in result:
            age = result['age']
            if age < 18:
                age_group = 'Under 18'
            elif age < 30:
                age_group = '18-29'
            elif age < 50:
                age_group = '30-49'
            else:
                age_group = '50+'
            statistics['age_groups'][age_group] += 1
        
        # Handle confidence
        if 'confidence' in result:
            statistics['average_confidence'] = (
                (statistics['average_confidence'] * (statistics['total_detections'] - 1) + 
                result['confidence']) / statistics['total_detections']
            )
        
        statistics['last_updated'] = datetime.now().strftime('%H:%M:%S')
    except Exception as e:
        print(f"Error updating statistics: {e}")

def process_frame(frame):
    try:
        # Add basic face detection first
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Only proceed with DeepFace if faces are detected by OpenCV
        if len(faces) > 0:
            results = DeepFace.analyze(
                frame, 
                actions=['age', 'gender'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(results, dict):
                results = [results]
                
            # Filter out false positives
            verified_results = []
            for result in results:
                if verify_face_detection(result):
                    verified_results.append(result)
                    
            return verified_results if verified_results else None
        return None
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return None

def analysis_worker():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = process_frame(frame)
            result_queue.put(results)
            time.sleep(0.01)

def generate_frames():
    global frame_count, last_results, detection_history
    
    analysis_thread = threading.Thread(target=analysis_worker, daemon=True)
    analysis_thread.start()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    no_detection_count = 0
    last_process_time = time.time()
    min_process_interval = 0.1
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_count += 1
            current_time = time.time()
            
            processing_frame = cv2.resize(frame, (0, 0), fx=rescale_factor, fy=rescale_factor)
            display_frame = frame.copy()
            
            if frame_count % frame_skip == 0 and (current_time - last_process_time) >= min_process_interval:
                try:
                    if not frame_queue.full():
                        frame_queue.put(processing_frame)
                        last_process_time = current_time
                    
                    if not result_queue.empty():
                        results = result_queue.get()
                        if results and len(results) > 0:
                            last_results = results
                            no_detection_count = 0
                            
                            for result in results:
                                try:
                                    gender_dict = result.get('gender', {})
                                    if isinstance(gender_dict, dict):
                                        dominant_gender = max(gender_dict.items(), key=lambda x: x[1])[0]
                                        confidence = max(gender_dict.values())
                                    else:
                                        dominant_gender = str(gender_dict)
                                        confidence = 1.0
                                    
                                    if confidence >= 0.6:  # Only record confident detections
                                        detection_data = {
                                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                                            'age': result.get('age', 'Unknown'),
                                            'gender': dominant_gender,
                                            'confidence': confidence,
                                            'status': 'detected'
                                        }
                                        
                                        detection_history.append(detection_data)
                                        update_statistics(detection_data)
                                except Exception as e:
                                    print(f"Error processing detection result: {e}")
                        else:
                            no_detection_count += 1
                            last_results = None  # Clear last results if no detection
                            
                except Exception as e:
                    print(f"Error in frame processing: {e}")
                    no_detection_count += 1
                
                detection_history = detection_history[-5:]
            
            # Clear overlay text if no recent detections
            if no_detection_count >= 3:
                last_results = None
            
            # Draw "No face detected" only when we're confident there's no face
            if no_detection_count >= 5:
                cv2.putText(display_frame, "No face detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Draw detection results
            if last_results:
                for result in last_results:
                    try:
                        if verify_face_detection(result):  # Double-check before drawing
                            face_region = result['region']
                            age = result.get('age', 'Unknown')
                            
                            gender_dict = result.get('gender', {})
                            if isinstance(gender_dict, dict):
                                dominant_gender = max(gender_dict.items(), key=lambda x: x[1])[0]
                                confidence = max(gender_dict.values())
                                
                                if confidence >= 0.6:  # Only draw confident detections
                                    scale = 1/rescale_factor
                                    x = int(face_region['x'] * scale)
                                    y = int(face_region['y'] * scale)
                                    w = int(face_region['w'] * scale)
                                    h = int(face_region['h'] * scale)
                                    
                                    # Draw rectangle and text
                                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    
                                    # Add text with background
                                    text_bg_color = (0, 0, 0)
                                    text_color = (0, 255, 0)
                                    
                                    age_text = f"Age: {age}"
                                    text_size = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                                    cv2.rectangle(display_frame, (x, y-text_size[1]-10), 
                                                (x+text_size[0], y), text_bg_color, -1)
                                    cv2.putText(display_frame, age_text, (x, y-5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                                    
                                    gender_text = f"{dominant_gender} ({confidence:.2f})"
                                    text_size = cv2.getTextSize(gender_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                                    cv2.rectangle(display_frame, (x, y+h), 
                                                (x+text_size[0], y+h+text_size[1]+10), text_bg_color, -1)
                                    cv2.putText(display_frame, gender_text, 
                                              (x, y+h+text_size[1]+5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                                    
                    except Exception as e:
                        print(f"Error drawing detection results: {e}")
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            ret, buffer = cv2.imencode('.jpg', display_frame, encode_param)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        except Exception as e:
            print(f"Error in main loop: {e}")
            continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detection_history')
def get_detection_history():
    return jsonify({
        'history': detection_history,
        'statistics': statistics
    })

if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0') 