# Cognia - AI Memory Assistance System for Dementia

<div align="center"> 

**"From Vision to Voice: Real-Time Support for Dementia Care"**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF?style=flat-square&logo=yolo)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask)
![Firebase](https://img.shields.io/badge/Firebase-Firestore-FFCA28?style=flat-square&logo=firebase)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv)
![AssemblyAI](https://img.shields.io/badge/AssemblyAI-STT-blue?style=flat-square)
![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-5-C51A4A?style=flat-square&logo=raspberrypi)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Target Segment](#target-segment)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [API Endpoints](#-api-endpoints)
- [Achieved Metrics](#-achieved-metrics)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## Overview

**Cognia** is a mobile-first AI-powered memory assistance system designed to support individuals with mild dementia and their caregivers. The platform provides real-time scene recognition, object tracking, and voice-based interaction to help patients maintain independence in daily activities while ensuring caregiver peace of mind.

The system uses computer vision (YOLO) for scene classification and object detection, combined with voice interaction (wake-word detection + speech-to-text) to provide ambient, hands-free assistance.

---

## Target Segment

**Mild Dementia Patients** who are still independent in basic Activities of Daily Living (bADL) and instrumental Activities of Daily Living (iADL).

> *By 2030, 19% of Malaysians will be aged 60 & above. 1 in 10 suffers from dementia (NHMS 2018), and 74% of elderly Malaysians live alone.*

---

## 🌟 Key Features

### ✅ Implemented Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Scene Classification** | YOLO-based kitchen/room detection using trained anchors (Stove, Fridge, Basin, Pot, Kettle) | ✅ Achieved |
| **Object Recognition** | Real-time object detection with 87% accuracy | ✅ Achieved |
| **Scene Understanding** | Context-aware scene classification with 82% accuracy | ✅ Achieved |
| **Object Tracking** | Spectacles/glasses location tracking with persistence | ✅ Achieved |
| **Wake-Word Detection** | "Hey Pico" activation using Porcupine | ✅ Achieved |
| **Speech-to-Text** | Real-time transcription via AssemblyAI streaming | ✅ Achieved |
| **Text-to-Speech** | Voice responses using gTTS + pygame | ✅ Achieved |
| **Location Inquiry** | Voice command: "Where am I?" → spoken location | ✅ Achieved |
| **Object Finder** | Voice command: "Where are my spectacles?" → location announcement | ✅ Achieved |
| **Reminder System** | Voice command: "Remind me in X minutes" | ✅ Achieved |
| **Caregiver Dashboard** | React-based web interface for monitoring | ✅ Achieved |
| **Firebase Integration** | Cloud database for presence and last-seen data | ✅ Achieved |
| **REST API** | Flask endpoints for health, presence, and last-seen data | ✅ Achieved |
| **Real-Time Alerts** | FCM push notification support for caregivers | ✅ Achieved |
| **Live Translation** | EN↔BM translation support | ✅ Achieved |

### Voice Commands Supported

| Command | Response |
|---------|----------|
| *"Hey Pico"* | Activates listening mode |
| *"Where am I?"* | Announces current location |
| *"Where are my spectacles?"* | Announces last known spectacle location |
| *"Remind me in X seconds/minutes"* | Sets a countdown timer |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  INPUT                                       │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│  Pi Camera   │  Microphone  │ Motion Sensor│ Daily Routine│  Caregiver App  │
│  (Pi Cam 3)  │              │              │   Schedule   │                 │
└──────┬───────┴──────┬───────┴──────────────┴──────────────┴────────┬────────┘
       │              │                                               │
       ▼              ▼                                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PROCESSING UNIT                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   Ultralytics   │  │    Firebase     │  │      AssemblyAI             │  │
│  │      YOLO       │  │    Database     │  │   (Speech-to-Text)          │  │
│  │ Scene Detection │  │   (Firestore)   │  │                             │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘  │
│           │                    │                          │                  │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌──────────────▼──────────────┐  │
│  │  ByteTrack      │  │  Flask API      │  │      Porcupine              │  │
│  │ Object Tracking │  │   Endpoints     │  │   Wake-Word Detection       │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘  │
│           │                    │                          │                  │
│           └────────────────────┼──────────────────────────┘                  │
│                                │                                             │
│                    ┌───────────▼───────────┐                                 │
│                    │     NLP Processing    │                                 │
│                    │   (Command Parsing)   │                                 │
│                    └───────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               OUTPUT                                         │
├──────────────────┬──────────────────┬──────────────────┬────────────────────┤
│   Real-Time      │   Conversational │   Caregiver      │   Memory           │
│     Alert        │     Reminder     │   Notification   │   Questioning      │
│    (Speaker)     │     (gTTS)       │   (FCM Push)     │   (Voice)          │
└──────────────────┴──────────────────┴──────────────────┴────────────────────┘
```

---

## 🛠️ Tech Stack

### Hardware
| Component | Model | Purpose |
|-----------|-------|---------|
| Microcontroller | Raspberry Pi 5 | Main processing unit |
| Camera | Pi Cam 3 | Visual input for scene detection |
| Speaker | USB/3.5mm | Voice output |
| Microphone | USB | Voice input |

### Software & Libraries

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core programming language |
| **Ultralytics YOLO** | Object detection & scene classification |
| **OpenCV** | Video capture & frame processing |
| **Flask + CORS** | REST API server |
| **Firebase Admin SDK** | Firestore database & FCM push notifications |
| **Porcupine (Picovoice)** | Wake-word detection ("Hey Pico") |
| **AssemblyAI** | Real-time streaming speech-to-text |
| **gTTS + pygame** | Text-to-speech playback |
| **React 18** | Caregiver dashboard frontend |

---

## 📁 Project Structure

```
Cognia/
├── README.md                      # This file
├── Scene_Prediction.py            # Windows/Desktop scene detection + API server
├── main.py                        # Raspberry Pi headless version with voice
├── my_model.pt                    # YOLO model for kitchen object detection
├── my_model_spec.pt               # YOLO model for spectacles detection
├── index.html                     # React-based caregiver dashboard
├── presence.json                  # Current location state (auto-generated)
├── last_spec_seen.json            # Last spectacles location (auto-generated)
└── firebase_service_account.json  # Firebase credentials (not included)
```

### Component Descriptions

| File | Description |
|------|-------------|
| `Scene_Prediction.py` | Desktop version with OpenCV GUI, Flask API, and Firebase sync |
| `main.py` | Headless Raspberry Pi version with Porcupine + AssemblyAI voice |
| `my_model.pt` | Custom-trained YOLO model for kitchen anchors (Stove, Fridge, Basin, Pot, Kettle) |
| `my_model_spec.pt` | Custom-trained YOLO model for spectacles/glasses detection |
| `index.html` | Single-page React caregiver dashboard with activity timeline, task management |

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10+
- Webcam or Pi Camera
- Microphone (for voice features)
- Firebase project (optional, for cloud sync)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Cognia
   ```

2. **Install Python dependencies**
   ```bash
   pip install ultralytics opencv-python flask flask-cors firebase-admin
   pip install pvporcupine pvrecorder assemblyai gtts pygame
   ```

3. **Configure API Keys** (for voice features)
   ```bash
   export ASSEMBLYAI_API_KEY="your_assemblyai_key"
   export PICOVOICE_ACCESS_KEY="your_picovoice_key"
   ```

4. **Firebase Setup** (optional)
   - Create a Firebase project
   - Download service account JSON
   - Save as `firebase_service_account.json`

---

## ▶️ Running the Application

### Desktop Mode (with GUI)
```bash
python Scene_Prediction.py
```
- Opens camera window with bounding boxes
- Starts Flask API on `http://localhost:5000`
- Press `q` to quit

### Headless Mode (Raspberry Pi)
```bash
python main.py
```
- Runs without GUI
- Wake-word activated voice interaction
- Logs to console

### Caregiver Dashboard
Open `index.html` in a browser, or serve via:
```bash
python -m http.server 8080
# Visit http://localhost:8080
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check API status |
| `/api/presence` | GET | Get current patient location |
| `/api/last_seen` | GET | Get last spectacles sighting |
| `/api/summary` | GET | Combined presence + last_seen data |

### Example Response: `/api/summary`
```json
{
  "ok": true,
  "presence": {
    "location": "Kitchen",
    "is_kitchen": true,
    "reason": "Stove conf=0.89",
    "time_iso": "2025-12-27 14:30:00"
  },
  "last_seen": {
    "place": "Kitchen",
    "label": "Spectacle",
    "conf": 0.82,
    "time_iso": "2025-12-27 14:25:00"
  }
}
```

---

## 📊 Achieved Metrics

| Metric | Target | Achieved (Prototype) |
|--------|--------|----------------------|
| Object Recognition | ≥ 80% | **87%** ✅ |
| Scene Understanding | ≥ 75% | **82%** ✅ |
| Real-Time Processing | Yes | **Yes** ✅ |

---

## 🔮 Future Enhancements

| Feature | Description | Status |
|---------|-------------|--------|
| **Ambient AI Voice Interaction** | Full conversational AI without wake-word | 🔄 Planned |
| **Conversational Memory Capture** | Record and recall conversations | 🔄 Planned |
| **Socratic/Guided Recall** | Memory exercises with prompts | 🔄 Planned |
| **Personalized Cognitive Insights** | AI-driven cognitive health reports | 🔄 Planned |
| **Multi-Room Tracking** | Expand beyond kitchen detection | 🔄 Planned |
| **Medication Reminders** | Scheduled medication alerts | 🔄 Planned |
| **Emergency Fall Detection** | Accelerometer-based fall alerts | 🔄 Planned |
| **Family Member Recognition** | Face recognition for familiar people | 🔄 Planned |

---

## 🏆 Competition

**AI in Medicine Bootcamp & Hackathon 2025**

**Team:** EEyerrr  
**Track:** Senior Care Solutions

**Team Members:**
- Chua Zhu Heng
- Chin Pei Kang
- Lim Zhi Pin
- Low Jia Qi
- Satishrao A/L Dharman
- Chong Rui Shen

---

## 📄 License

This project is developed for AI In Medicine Bootcamp & Hackathon 2025 purposes.

---

<div align="center">

**Built with ❤️ for Dementia Care**

*Enhancing Safety & Cognitive Support • Accessible AI for All • Supporting Healthy Aging Communities*

</div>
