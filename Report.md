# Smart Posture Monitor

## 1. Motivation and Problem Statement

With the increasing use of computers and mobile devices, poor sitting posture has become a major health concern. Prolonged incorrect posture can lead to musculoskeletal disorders, back pain, neck strain, and long-term spinal issues. Most users are unaware of their posture during work or study sessions and correct it only after discomfort occurs.

The motivation behind this project is to develop a **real-time, non-intrusive Smart Posture Monitoring system** that continuously monitors a user’s posture using computer vision and machine learning techniques. The system detects improper posture and provides timely feedback, encouraging users to maintain healthy posture habits without requiring wearable sensors.

---

## 2. Hardware and Software Used

### Hardware Components

| Component | Description |
|--------|-------------|
| Raspberry Pi | Edge computing device for running posture detection |
| Raspberry Pi Camera Module | Captures live video feed |
| Laptop / PC | Used for development, training, and monitoring |
| Power Bank | Powers the Raspberry Pi |

### Software Components

| Software / Library | Purpose |
|------------------|--------|
| Python | Core programming language |
| OpenCV | Video capture and image processing |
| MediaPipe | Human pose estimation and keypoint extraction |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Scikit-learn | Model training and evaluation |
| GitHub | Version control and project hosting |

---

## 3. Methodology and System Design

The Smart Posture Monitor follows a **vision-based posture analysis pipeline**, which operates in real time.

### System Workflow

1. Capture live video feed using webcam
2. Extract body landmarks using pose estimation
3. Compute posture-related angles and distances
4. Classify posture as *good* or *bad*
5. Provide visual or textual feedback to the user
6. Tracks focus time
7. Gives alerts to user if idle time is too long
8. Logs activity, transfers csv files through a WiFi network and is visible on a dashboard on a mobile app

---

## 4. Data Collection and Preparation

### Data Collection

- Video frames were captured from users sitting in front of a webcam.
- Different postures were recorded, including:
  - Upright posture
  - Slouched posture
  - Leaning forward or sideways

### Data Preparation

- MediaPipe Pose was used to extract key body landmarks such as:
  - Shoulders
  - Neck
  - Hips
- Features such as angles between shoulder–neck–hip and relative distances were calculated.
- The extracted features were stored in structured datasets for training.

### Annotation

- Postures were manually labeled as:
  - **0 – Good posture**
  - **1 – Bad posture**

---

## 5. Model Development

- A supervised machine learning approach was used.
- Extracted posture features were used as input vectors.
- A classification model was trained to distinguish between correct and incorrect posture.
- Model performance was evaluated using accuracy and confusion matrices.

The lightweight nature of the model allows it to run efficiently in real time without GPU acceleration.

---

## 6. Model Deployment

- The trained model is integrated into a live video pipeline.
- Each video frame is processed independently.
- Predictions are made in real time and overlaid on the video feed.
- The system alerts the user when poor posture is detected continuously for a threshold duration.

Deployment is fully local, ensuring **privacy and low latency**.

---

## 7. Results and Conclusions

### Results

- The system successfully detects poor posture in real time.
- Achieved high accuracy under controlled lighting and camera positioning.
- Minimal latency observed during live inference.

### Conclusion

The Smart Posture Monitor demonstrates that **computer vision-based posture detection** is an effective and practical solution for everyday posture monitoring. The project eliminates the need for wearable devices while maintaining accuracy and usability.

---

## 8. Future Work

- Support for multiple users in a single frame
- Integration with mobile applications
- Audio or haptic feedback for alerts
- Long-term posture analytics and reporting
- Extension to standing and dynamic postures
- Deployment on edge devices such as Raspberry Pi

---

## 9. System Schematic

<img width="413" height="215" alt="image" src="https://gist.github.com/user-attachments/assets/d7f74479-3c01-454f-bede-fc224d0bbd8e" />
<img width="1606" height="773" alt="image" src="https://gist.github.com/user-attachments/assets/ba6f99c2-53bd-4bd3-92dd-f6146cc3b2e9" />




---

## 10. References

1. MediaPipe Pose Documentation – https://developers.google.com/mediapipe
2. OpenCV Library – https://opencv.org/
3. Scikit-learn Documentation – https://scikit-learn.org/
4. Smart Posture Monitor GitHub Repository – https://github.com/pranav-m-r/Smart-Posture-Monitor
5. AI Helmet Reference Project – https://github.com/samy101/ai-helmet

