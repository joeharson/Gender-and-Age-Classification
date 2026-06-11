# Face & Age Detection 3 - Webcam/Video File Detection

A Python application for detecting faces and predicting age and gender from webcam or video files.

## 📁 Files in This Folder

| File | Purpose |
|------|---------|
| `code.py` | Main Python script - detects from webcam or video |
| `age&genderDetection.ipynb` | Jupyter notebook version |
| `modelNweight/` | Folder containing all pre-trained models |
| `image.jpg, image1.jpg, image2.jpg` | Sample test images |

### Models (in `modelNweight/` folder)

| File | Purpose |
|------|---------|
| `opencv_face_detector_uint8.pb` | Face detection model |
| `opencv_face_detector.pbtxt` | Face detector config |
| `age_net.caffemodel` | Age prediction model |
| `age_deploy.prototxt` | Age model config |
| `gender_net.caffemodel` | Gender prediction model |
| `gender_deploy.prototxt` | Gender model config |

## 🎯 What It Does

- Opens your **webcam** or a **video file**
- Detects **faces** in real-time
- Predicts **age group** and **gender**
- Shows results directly on the video
- Organized model structure for easy management

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

### From Webcam (Default):
```bash
python code.py
```

### From Video File:
Edit line in `code.py`:
```python
# Change this line:
cap = cv2.VideoCapture(0)  # 0 for webcam

# To this (for a video file):
cap = cv2.VideoCapture('your_video.mp4')
```

Then run:
```bash
python code.py
```

### Using Jupyter Notebook:
```bash
jupyter notebook age&genderDetection.ipynb
```

## 📝 How to Use

1. Make sure webcam is connected (or have video file ready)
2. Run the script
3. Video window will open
4. Detected faces will show age and gender
5. Press **'q'** to exit

## 🎬 What You'll See

- **Live video** from webcam (or video file)
- **Green rectangles** around detected faces
- **Yellow text** showing: "Male, (25-32)" or "Female, (15-20)"
- Real-time detection results

## 💡 Tips

- All models are in a separate **`modelNweight/`** folder (organized structure)
- Works with **webcam or video files**
- Good lighting improves detection accuracy
- Keep faces **clearly visible** to camera

## 🎥 Supported Video Formats

- `.mp4` - Most common
- `.avi` - Windows video format
- `.mov` - MacOS video format
- `.mkv` - Matroska format
- Any format that OpenCV supports

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Could not open" error | Check video file exists and path is correct |
| No faces detected | Ensure good lighting, face must be visible |
| Slow performance | Use lower resolution video or hardware upgrade |
| Window won't open | Check OpenCV installation |
