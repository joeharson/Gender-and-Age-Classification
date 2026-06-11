# Face & Age Detection 2 - IP Camera (RTSP) Stream Detection

A Python application that detects faces from an IP camera's RTSP stream and predicts age and gender.

## 📁 Files in This Folder

| File | Purpose |
|------|---------|
| `detect.py` | Main Python script - detects from RTSP stream |
| `opencv_face_detector_uint8.pb` | Pre-trained face detection model |
| `opencv_face_detector.pbtxt` | Face detector configuration file |
| `age_net.caffemodel` | Pre-trained age prediction model |
| `age_deploy.prototxt` | Age model configuration file |
| `gender_net.caffemodel` | Pre-trained gender prediction model |
| `gender_deploy.prototxt` | Gender model configuration file |
| `girl1.jpg, kid2.jpg` | Sample test images |
| `_config.yml` | Configuration file |

## 🎯 What It Does

- Connects to an **IP camera** using RTSP protocol
- Detects **faces** from the camera stream
- Predicts the **age group** and **gender**
- Displays results in a window
- Prints predictions to console

## ⚙️ Requirements

```
opencv-python
numpy
```

Install with:
```bash
pip install opencv-python numpy
```

## 🚀 How to Run

```bash
python detect.py --rtsp "rtsp://username:password@ip_address:port/path"
```

### Example:
```bash
python detect.py --rtsp "rtsp://admin:password@192.168.1.100:554/stream1"
```

### Using Default RTSP URL:
If you just run without arguments, it uses the default URL in the code:
```bash
python detect.py
```

## 📝 How to Use

1. Get your **IP camera's RTSP URL** (usually from camera settings)
2. Run the command with your camera URL
3. The program will connect to the camera
4. Detected faces will show **age and gender**
5. Press **'q'** or close window to stop

## 🔗 How to Find Your RTSP URL

- Check your **IP camera's user manual**
- Usually looks like: `rtsp://username:password@IP:PORT/stream`
- Ask your camera manufacturer or IT team

## 🎥 What You'll See

- **Live feed** from your IP camera
- **Green rectangles** around detected faces
- **Text labels** with age and gender predictions
- **Console output** showing "Gender: Male" and "Age: 28 years"

## 💡 Tips

- Make sure camera is **connected to network**
- Check **username and password** for the camera
- Good **network connection** needed for stable stream
- Camera must **support RTSP protocol**

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Error: Could not open RTSP stream" | Check URL, username, password, IP address |
| No detection happening | Ensure good lighting at camera location |
| Slow/laggy detection | Reduce video quality or use faster network |
