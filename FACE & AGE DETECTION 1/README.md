# Face & Age Detection 1 - Webcam Real-Time Detection

A simple Python application that detects faces from your webcam and predicts age and gender in real-time.

## 📁 Files in This Folder

| File | Purpose |
|------|---------|
| `app.py` | Main Python script - runs real-time detection from webcam |
| `app.ipynb` | Jupyter notebook version of the application |
| `opencv_face_detector_uint8.pb` | Pre-trained face detection model |
| `opencv_face_detector.pbtxt` | Face detector configuration file |
| `age_net.caffemodel` | Pre-trained age prediction model |
| `age_deploy.prototxt` | Age model configuration file |
| `gender_net.caffemodel` | Pre-trained gender prediction model |
| `gender_deploy.prototxt` | Gender model configuration file |
| `1.jpeg, 2.jpg, 3.jpg` | Sample test images |

## 🎯 What It Does

- Opens your computer's **webcam**
- Detects **faces** in real-time
- Predicts the **age group** (0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100)
- Predicts the **gender** (Male or Female)
- Displays results **directly on the video**

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

### Using Python Script:
```bash
python app.py
```

### Using Jupyter Notebook:
```bash
jupyter notebook app.ipynb
```

Then run the cells in order.

## 📝 How to Use

1. Run the script
2. Your webcam will open in a window
3. The program will detect your face
4. Your **age group and gender** will appear on the video
5. Press **'q'** to quit

## ✅ What You'll See

- **Green rectangle** around detected faces
- **Text label** showing: "Male, (25-32)" or "Female, (15-20)"
- Real-time processing at ~30 FPS

## 💡 Tips

- Make sure your webcam is working
- Good lighting helps with better detection
- Keep your face clearly visible in the camera
