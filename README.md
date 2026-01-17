# Online Proctoring System

**Real-Time AI-Based Exam Monitoring Using WebSockets**

---

## 1. Introduction

With the rapid adoption of online examinations, maintaining exam integrity has become a critical challenge. Conventional online exam systems lack effective invigilation, making them susceptible to impersonation, use of unauthorized devices, and reduced candidate attention.

This project presents a **real-time AI-based online proctoring system** that continuously monitors a candidate through their webcam during an examination. The system detects suspicious activities such as **identity mismatch, multiple faces, mobile phone usage, and loss of attention**, and instantly reports these events to the frontend using **WebSocket-based communication**.

---

## 2. Objectives

The primary objectives of this project are:

* To verify the identity of the candidate using face recognition
* To detect the presence of multiple individuals during an examination
* To identify mobile phone usage in real time
* To monitor and score the candidate’s attention level
* To provide low-latency alerts to the frontend interface

---

## 3. Key Features

* User face registration and authentication
* Continuous face presence monitoring
* Multiple face detection
* Mobile phone detection using YOLO
* Attention and focus scoring
* Real-time alerts with a cooldown mechanism
* Low-latency communication using WebSockets

---

## 4. Project Structure

```
├── frontend/
│   └── index.html             # User interface, webcam capture, WebSocket client
│
├── websocket/
│   ├── socket.py             # WebSocket server (entry point)
│   ├── app.py                 # Core proctoring session logic
│   ├── phone_detector.py      # Mobile phone detection (YOLO)
│   ├── attention_engine.py    # Attention scoring and alert rules
│
└── README.md
```

Only the **frontend** and **websocket** directories are included to maintain a clean and secure repository structure.

---

## 5. System Architecture

```
[ Frontend (Browser) ]
        |
        |  WebSocket (live video frames)
        v
[ WebSocket Server (socket.py) ]
        |
        v
[ ProctoringSession (app.py) ]
        |
        +--> Face Recognition (InsightFace)
        +--> Phone Detection (YOLOv8)
        +--> Attention Engine
```

---

## 6. Working Principle (End-to-End Flow)

1. The frontend establishes a WebSocket connection with the backend.
2. The candidate sends an initial webcam frame for **face registration**.
3. The backend extracts and stores the candidate’s face embedding.
4. Live webcam frames are streamed continuously to the server.
5. For each frame, the backend performs:

   * Face presence and identity verification
   * Multiple face detection
   * Mobile phone detection
   * Head pose estimation and attention scoring
6. Detected events are sent back to the frontend in real time.

---

## 7. Technologies Used

### Backend

* Python
* WebSockets (`websockets` library)
* OpenCV
* InsightFace (face detection and recognition)
* YOLOv8 (mobile phone detection)
* NumPy

### Frontend

* HTML, CSS, JavaScript
* Browser Webcam API
* WebSocket API

---

## 8. Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/brijesh-shetty/focus-detection
cd focus-detection
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 9. How to Run

### Step 1: Start the Backend
```bash
cd websocket
python socket.py
```

### Step 2: Run the Frontend
You can serve the frontend using a local server:
```bash
cd frontend
python -m http.server 8000
```
Then visit `http://localhost:8000` in your browser.

---

## 10. WebSocket Events

The backend sends structured JSON events to the frontend, including:

```json
{ "event": "AUTHORIZED" }
{ "event": "FACE_NOT_FOUND" }
{ "event": "MULTIPLE_FACES" }
{ "event": "PHONE_DETECTED" }
{ "event": "LOOKING_AWAY", "score": 35 }
{ "event": "ATTENTION_UPDATE", "score": 82, "label": "Focused" }
```

---

## 11. Design Considerations

* Threaded AI processing to avoid blocking WebSocket communication
* Frame-based optimization for real-time performance
* Cooldown mechanism to prevent alert flooding
* Clear separation between detection, decision-making, and communication layers

---

## 12. Applications

* Online examinations
* Remote interviews
* Certification and assessment platforms
* Secure remote evaluations

---

## 13. Author

**Brijesh Shetty N**  
Online Proctoring System Project

---

## 14. License

This project is intended for **educational and academic purposes only**.
