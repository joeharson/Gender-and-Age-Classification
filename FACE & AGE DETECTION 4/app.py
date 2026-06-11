from flask import Flask, render_template, Response, jsonify
import cv2
from deepface import DeepFace
import json
from datetime import datetime
from collections import defaultdict
import threading
import queue
import time

app = Flask(__name__)

# Global variables
frame_skip = 2
frame_count = 0
last_results = None
rescale_factor = 0.7  # Increased for better quality
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
    """Process frame in separate thread"""
    try:
        results = DeepFace.analyze(
            frame, 
            actions=['age', 'gender'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        if isinstance(results, dict):
            results = [results]
        return results
    except Exception as e:
        print(f"Error in processing: {e}")
        return None

def analysis_worker():
    """Worker thread for face analysis"""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = process_frame(frame)
            result_queue.put(results)
            time.sleep(0.01)

def generate_frames():
    global frame_count, last_results, detection_history
    
    # Start analysis worker thread
    analysis_thread = threading.Thread(target=analysis_worker, daemon=True)
    analysis_thread.start()
    
    # Improved camera settings
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increased resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # HD resolution
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to 30
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Adjust brightness
    cap.set(cv2.CAP_PROP_CONTRAST, 150)  # Adjust contrast
    
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
            
            # Create processing frame (smaller) and display frame (higher quality)
            processing_frame = cv2.resize(frame, (0, 0), fx=rescale_factor, fy=rescale_factor)
            display_frame = frame.copy()  # Keep original quality for display
            
            if frame_count % frame_skip == 0 and (current_time - last_process_time) >= min_process_interval:
                try:
                    if not frame_queue.full():
                        frame_queue.put(processing_frame)
                        last_process_time = current_time
                    
                    if not result_queue.empty():
                        results = result_queue.get()
                        if results:
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
                            
                except Exception as e:
                    print(f"Error in frame processing: {e}")
                    no_detection_count += 1
                
                # Limit detection history size
                detection_history = detection_history[-5:]  # Reduced from 10 to 5
            
            # Draw results on the high-quality display frame
            if no_detection_count >= 5:
                cv2.putText(display_frame, "No face detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            if last_results:
                for result in last_results:
                    try:
                        if 'region' in result:
                            face_region = result['region']
                            age = result.get('age', 'Unknown')
                            
                            gender_dict = result.get('gender', {})
                            if isinstance(gender_dict, dict):
                                dominant_gender = max(gender_dict.items(), key=lambda x: x[1])[0]
                                confidence = max(gender_dict.values())
                            else:
                                dominant_gender = str(gender_dict)
                                confidence = 1.0
                            
                            # Scale up the coordinates for the higher resolution display frame
                            scale = 1/rescale_factor
                            x = int(face_region['x'] * scale)
                            y = int(face_region['y'] * scale)
                            w = int(face_region['w'] * scale)
                            h = int(face_region['h'] * scale)
                            
                            # Draw with improved visuals
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # Add background for text for better visibility
                            text_bg_color = (0, 0, 0)
                            text_color = (0, 255, 0)
                            
                            # Age text
                            age_text = f"Age: {age}"
                            text_size = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            cv2.rectangle(display_frame, (x, y-text_size[1]-10), (x+text_size[0], y), text_bg_color, -1)
                            cv2.putText(display_frame, age_text, (x, y-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                            
                            # Gender text
                            gender_text = f"{dominant_gender} ({confidence:.2f})"
                            text_size = cv2.getTextSize(gender_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            cv2.rectangle(display_frame, (x, y+h), (x+text_size[0], y+h+text_size[1]+10), text_bg_color, -1)
                            cv2.putText(display_frame, gender_text, (x, y+h+text_size[1]+5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                            
                    except Exception as e:
                        print(f"Error drawing detection results: {e}")
            
            # Encode the high-quality frame
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Higher JPEG quality
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