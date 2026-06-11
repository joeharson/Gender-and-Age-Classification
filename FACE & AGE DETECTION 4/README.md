# Face & Age Detection 4 - Web Application with Statistics

A web-based application for real-time face, age, and gender detection with a beautiful web interface and statistics dashboard.

## 📁 Files in This Folder

| File | Purpose |
|------|---------|
| `app.py` | Main Flask web server - runs the web app |
| `deep.py` | Helper module for deep learning operations |
| `requirements.txt` | Python packages needed |
| `templates/index.html` | Web page interface |
| `age_net.caffemodel` | Age prediction model |
| `age_deploy.prototxt` | Age model configuration |
| `gender_net.caffemodel` | Gender prediction model |
| `gender_deploy.prototxt` | Gender model configuration |
| `deploy.prototxt, deploy (1).prototxt` | Model configuration files |
| `try2.py, tryretina.py` | Experimental/test scripts |
| `tct.txt` | Notes/testing file |

## 🎯 What It Does

- Creates a **web server** you can open in your browser
- **Real-time** face detection from your webcam
- **Video streaming** in the web page
- Shows **age and gender** predictions
- Displays **statistics dashboard** with:
  - Total detections
  - Gender distribution
  - Age group breakdown
  - Detection confidence scores

## ⚙️ Requirements

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Required packages:
```
flask              # Web server
opencv-python     # Computer vision
deepface          # Advanced face detection & analysis
```

## 🚀 How to Run

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Run the Server
```bash
python app.py
```

### Step 3: Open in Browser
Open your web browser and go to:
```
http://localhost:5000
```

## 📝 How to Use

1. Run `python app.py`
2. Open `http://localhost:5000` in your web browser
3. Allow camera access when prompted
4. You'll see:
   - **Live video feed** from your webcam
   - **Real-time detection** of faces, age, and gender
   - **Statistics panel** showing analysis data
5. Refresh or check the dashboard for updated statistics

## 🎨 Features

✅ **Real-Time Video Streaming** - See your webcam feed in browser  
✅ **Age & Gender Detection** - Accurate predictions using DeepFace  
✅ **Statistics Dashboard** - Track detection trends  
✅ **Gender Statistics** - Count of Male/Female detections  
✅ **Age Group Analysis** - Categorizes age ranges  
✅ **Confidence Scores** - Shows detection reliability  

## 🌐 Web Interface

The `index.html` provides:
- Video streaming display
- Real-time age/gender labels
- Statistics widget
- Responsive design (works on phone/tablet too)

## 💻 What's Running

```
App Server: http://localhost:5000/
Video Stream: http://localhost:5000/video_feed (internal)
```

## 📊 Statistics Tracked

| Metric | Description |
|--------|-------------|
| Total Detections | How many faces detected |
| Gender Stats | Male vs Female count |
| Age Groups | Under 18, 18-29, 30-49, 50+ |
| Confidence | How sure the model is |
| Last Updated | When stats were last updated |

## 🔧 How to Stop

Press `Ctrl + C` in the terminal running the app.

## 💡 Tips

- Make sure your **webcam is working**
- Use **good lighting** for better accuracy
- First time takes longer to load models
- Keep **browser tab open** for detection to continue
- Can open multiple tabs to see same feed

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot open camera" | Check if webcam is connected and working |
| Browser shows nothing | Wait 10-15 seconds for models to load |
| "Port already in use" | Change port in app.py or close other Flask apps |
| Slow detection | Your computer may not be powerful enough, reduce video resolution |
| Models not downloading | Check internet connection, models download on first run |

## ⚡ Performance Tips

- Close other applications to free up CPU
- Reduce video resolution if too slow
- Use a camera with good quality feed
- Modern CPU recommended for smooth operation

## 📱 Access from Other Devices

If you want to access from another computer on the same network:
1. Find your computer's IP address
2. Use: `http://YOUR_IP:5000` from another device
3. Make sure firewall allows port 5000
