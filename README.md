# Face Recognition Security System

## Description
This project is a real-time face recognition based security system built using OpenCV and the face_recognition library. The system detects human faces through a webcam and compares them with a predefined dataset of authorized users.

If a match is found, the system grants access and logs the event. If the face is unknown, it triggers an alert and stores the unknown face image for further analysis. This system improves security by allowing only authorized individuals and monitoring unknown access attempts.

---

## Features
- Real-time face detection and recognition
- Authorized and Unauthorized classification
- Automatic logging with timestamps
- Unknown face capture and storage
- Live system status display (IDLE / DETECTING / AUTHORIZED / UNAUTHORIZED)
- FPS (Frames Per Second) monitoring
- Efficient processing using frame skipping

---

## Technologies Used
- Python
- OpenCV
- face_recognition
- NumPy
- OS & Datetime modules

---

## Project Structure
- main.py → Main program execution
- dataset/ → Folder containing known face images
- unknown_faces/ → Stores captured unknown faces
- log.txt → Stores entry logs with date and time

---

## Working Principle
1. The system loads known face images from the dataset.
2. Face encodings are generated using the face_recognition library.
3. Webcam captures live video frames.
4. Faces are detected and compared with known encodings.
5. If match found → Authorized access granted.
6. If no match → Unauthorized alert triggered.
7. Unknown faces are saved and all events are logged.

---

## How to Run
1. Install required libraries:
   pip install opencv-python face_recognition numpy

2. Make sure dataset folder contains known face images.

3. Run the program:
   python main.py

4. Press 'q' to exit.

---

## Sample Output
- Authorized → Door Opening
- Unauthorized → Alarm Triggered

---

## Future Enhancements
- Automatic learning of new faces
- Integration with IoT-based smart door lock systems
- Cloud-based database for face storage
- Mobile notification alerts

---

## Conclusion
This project demonstrates a real-time AI-based security system that combines computer vision and machine learning to enhance access control and monitoring.
