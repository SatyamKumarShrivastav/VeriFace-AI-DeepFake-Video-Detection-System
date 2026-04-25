# VeriFace-AI-DeepFake-Video-Detection-System
VeriFace AI is an AI-powered DeepFake Video Detection System that analyzes videos to identify whether they are real or manipulated. It provides a REAL or FAKE verdict, accuracy percentage, and detailed forensic reasoning through an intelligent and user-friendly interface.


# README.md

# 🎬 DeepFake Video Detection System

A complete AI-powered **DeepFake Video Detection System** built with **Python FastAPI + HTML/CSS/JavaScript**, designed to analyze videos and determine whether they are **REAL** or **FAKE** using intelligent frame-level forensic analysis.

The system provides:

* Personalized welcome screen for **Mr. S. K. Shrivastav**
* Video upload with automatic analysis
* Real/Fake classification
* Accuracy percentage
* Detailed technical reasoning
* Beautiful premium UI with dark glassmorphism design



# 📌 Project Goal

The goal of this project is to build a reliable and user-friendly DeepFake Detection platform that helps users verify whether a video is authentic or manipulated.

This system combines:

* Computer Vision
* Video Forensics
* Feature Extraction
* AI-based Heuristic Detection
* Human-readable Explanation Engine

to make deepfake detection accessible without requiring advanced technical knowledge.



# 🏗️ Project Architecture

```text
New folder/
├── backend/                    # Python FastAPI backend
│   ├── main.py                 # FastAPI app + REST API endpoints
│   ├── core/
│   │   ├── video_processor.py  # Frame extraction + preprocessing
│   │   ├── model_inference.py  # DeepFake detection ML model
│   │   └── report_generator.py # Reason generation
│   └── requirements.txt
│
├── frontend/                   # Vanilla HTML/CSS/JS frontend
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
│
└── run.bat                     # One-click launcher
```



# ⚙️ Backend (Python FastAPI)

## 🔍 Detection Approach

The backend performs intelligent frame-level forensic analysis using:

* OpenCV
* scikit-image
* NumPy
* FastAPI

### Detection Features

The system checks for:

* Facial inconsistencies
* Compression artifacts
* Frequency domain anomalies (FFT)
* Temporal inconsistencies between frames
* Frame manipulation patterns
* Visual distortion detection

### Ensemble Scoring

Multiple heuristic detectors are combined to generate:

* Final REAL / FAKE verdict
* Confidence percentage
* Technical explanation

This improves reliability compared to using a single detection method.



# 🔌 API Endpoints

## POST `/analyze`

Upload a video and receive:

* Detection result
* Confidence score
* Technical indicators
* Human-readable reasoning



## GET `/health`

Simple health check endpoint for backend monitoring.

Example response:

```json
{
  "status": "running"
}
```



## GET `/result/{id}`

Fetch cached analysis results using a result ID.

Useful for:

* Rechecking reports
* Frontend refresh handling
* Async workflows



# 🎨 Frontend (HTML/CSS/JS)

## Main Sections



## 👋 Welcome Hero

Animated personalized greeting:

### Welcome! Mr. S. K. Shrivastav

with premium visual effects and motion design.



## 📁 Upload Section

Features:

* Drag & Drop uploader
* File browser upload
* Video preview before analysis
* Supported formats:

  * MP4
  * AVI
  * MOV
  * WebM
  * MKV


## ⚡ Analysis Dashboard

Displays:

* Live progress animation
* Frame-by-frame status
* Loading indicators
* Processing stages

Examples:

* Extracting frames
* Face detection
* Frequency analysis
* Authenticity verification



## 📊 Results Panel

Displays:

* REAL / FAKE verdict card
* Accuracy percentage gauge
* Confidence meter
* Detailed reasoning list
* Technical findings report



# 🎨 Design System

## UI Theme

### Dark Glassmorphism Interface

Modern premium UI with:

* Blur glass cards
* Soft shadows
* Transparent panels
* Smooth hover animations



## Accent Colors

### Cyan + Purple Gradient Theme

Used for:

* Buttons
* Progress bars
* Result cards
* Animated highlights



## Motion & Effects

* Smooth page transitions
* Upload glow effects
* Animated progress bars
* Live scanning visuals
* Floating UI effects



# 🛠️ Tech Stack

| Layer                | Technology         |
| -------------------- | ------------------ |
| Backend              | Python             |
| API Framework        | FastAPI            |
| Computer Vision      | OpenCV             |
| Image Analysis       | scikit-image       |
| Numerical Processing | NumPy              |
| Frontend             | HTML               |
| Styling              | CSS                |
| Logic                | JavaScript         |
| Deployment           | Localhost / Server |



# 🚀 How to Run



## Step 1: Clone Repository

```bash
git clone https://github.com/your-username/deepfake-video-detection-system.git
cd deepfake-video-detection-system
```



## Step 2: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```



## Step 3: Run Backend Server

```bash
uvicorn main:app --reload
```

Server starts at:

```text
http://127.0.0.1:8000
```



## Step 4: Open Frontend

Open:

```text
frontend/index.html
```

in your browser.



## Step 5: One-Click Launch (Optional)

Use:

```text
run.bat
```

to start the full project quickly.



# ✅ Verification Plan

## Test Case 1

### Upload a Real Video

Expected Result:

```text
REAL
```

with high authenticity score.



## Test Case 2

### Upload a Heavily Edited / Compressed Video

Expected Result:

```text
Suspicious / FAKE
```

with technical warnings.



## Test Case 3

### UI Testing

Verify:

* Responsive design
* Upload flow
* Smooth animations
* Progress dashboard
* Final report rendering



# ⚠️ Disclaimer

This project is intended for:

## Educational and Demonstration Purposes Only

It should not be used as:

* Legal proof
* Official forensic evidence
* Government verification system

For production-grade deployment, integration with advanced deepfake detection datasets and trained ML models is recommended.

Examples:

* FaceForensics++
* Celeb-DF
* DFDC Dataset
* Custom CNN / Transformer pipelines



# 🚀 Future Improvements

Possible upgrades include:

* Deep learning model integration
* Real-time webcam detection
* Admin dashboard
* Cloud deployment
* User authentication
* Report export (PDF)
* Detection history
* Batch video processing
* API access for enterprise use



# 👨‍💻 Author

## Satyam Kumar Shrivastav

B.Tech — Artificial Intelligence and Data Science - 3rd year/6th sem

Focused on building real-world AI systems with practical impact, premium UI/UX, and intelligent automation.

---

# 📬 Connect With Me

* 💼 LinkedIn: [Satyam Kumar Shrivastav](https://www.linkedin.com/in/s-k-shrivastav/))
* 📧 Email: [Satyam Kumar Shrivastav](mailto:23ad101@excelcolleges.com)
* 🐦 X (Twitter): [Satyam Kumar Shrivastav](mailto:satyamkumarshrivastav.ai@gmail.com)

---

# ⭐ If you like this project, give it a star!
