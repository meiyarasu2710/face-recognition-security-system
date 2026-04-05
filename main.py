import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import time

# ==============================
# SYSTEM START
# ==============================
print("=================================")
print(" Edge AI Face Recognition System ")
print("=================================")
print("Initializing system...")
time.sleep(1)
print("Loading database...")
time.sleep(1)
print("Starting camera...")
time.sleep(1)
print("System Active\n")

# ==============================
# LOAD DATASET
# ==============================
path = 'dataset'
images = []
classNames = []

for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    images.append(img)
    classNames.append(os.path.splitext(file)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img, model='small')
        if enc:
            encodeList.append(enc[0])
    return encodeList

encodeListKnown = findEncodings(images)

# ==============================
# CAMERA SETUP
# ==============================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

process_this_frame = True
last_detected = None
last_unknown_time = 0

prev_time = 0

# ==============================
# LOG FUNCTION
# ==============================
def log_entry(name, status):
    with open("log.txt", "a") as f:
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{time_now} - {name} - {status}\n")

# ==============================
# MAIN LOOP
# ==============================
while True:
    success, img = cap.read()

    imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        facesCurFrame = face_recognition.face_locations(imgSmall)
        encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    process_this_frame = not process_this_frame

    if len(facesCurFrame) == 0:
        state = "IDLE"
        last_detected = None
    else:
        state = "DETECTING"

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        if faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
            state = "AUTHORIZED"
            label = f"{name} - Authorized"
            color = (0, 255, 0)

            if last_detected != name:
                print(f"{name}: Door Opening...")
                log_entry(name, "Authorized")
                last_detected = name
                time.sleep(2)

        else:
            name = "UNKNOWN"
            state = "UNAUTHORIZED"
            label = "Unauthorized"
            color = (0, 0, 255)

            current_time = time.time()

            if last_detected != "UNKNOWN" and current_time - last_unknown_time > 5:
                print("Unknown: Alarm Triggered!")
                log_entry("Unknown", "Unauthorized")

                last_detected = "UNKNOWN"
                last_unknown_time = current_time

                # Save unknown face
                unknown_face = img[y1:y2, x1:x2]
                if unknown_face.size > 0:
                    if not os.path.exists("unknown_faces"):
                        os.makedirs("unknown_faces")
                    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                    cv2.imwrite(f"unknown_faces/{filename}", unknown_face)

                time.sleep(2)

        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, label, (x1, y2+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ==============================
    # DISPLAY INFO
    # ==============================
    cv2.putText(img, f"State: {state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(img, f"Faces: {len(facesCurFrame)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time

    cv2.putText(img, f"FPS: {int(fps)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Status Indicator
    if state == "AUTHORIZED":
        cv2.circle(img, (600, 30), 10, (0,255,0), -1)
    elif state == "UNAUTHORIZED":
        cv2.circle(img, (600, 30), 10, (0,0,255), -1)
    else:
        cv2.circle(img, (600, 30), 10, (255,255,0), -1)

    cv2.imshow('Face Recognition System', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()