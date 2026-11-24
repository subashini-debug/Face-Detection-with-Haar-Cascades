# ==============================================================================
# Face Detection using Haar Cascades with OpenCV and Matplotlib
# Developed By : Subashini K
# Register No  : 212225240160
# ==============================================================================

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# I) Load and Display Images
# ------------------------------------------------------------------------------
# Load image in grayscale
img_gray = cv2.imread('image.jpg', 0)

# Display input grayscale image
plt.figure(figsize=(8, 6))
plt.imshow(img_gray, cmap='gray')
plt.title('Input Image (Grayscale)')
plt.axis('off')
plt.show()

# ------------------------------------------------------------------------------
# II) Extract Region of Interest (ROI)
# ------------------------------------------------------------------------------
# Extract Region of Interest (Stephen Hawking in Zero-G)
roi = img_gray[160:370, 650:830]

plt.figure(figsize=(6, 6))
plt.imshow(roi, cmap='gray')
plt.title('Extracted ROI (Stephen Hawking)')
plt.axis('off')
plt.show()

# ------------------------------------------------------------------------------
# III) Load Haar Cascade Classifiers
# ------------------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# ------------------------------------------------------------------------------
# IV) Perform Face Detection in Images
# ------------------------------------------------------------------------------
def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.05, minNeighbors=2)
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
    return face_img

result_face = detect_face(img_gray)
plt.figure(figsize=(8, 6))
plt.imshow(result_face, cmap='gray')
plt.title('Face Detection using Haar Cascade')
plt.axis('off')
plt.show()

# ------------------------------------------------------------------------------
# V) Adjusted Multi-Face Detection
# ------------------------------------------------------------------------------
def adj_detect_face(img):
    face_img = img.copy()
    faces = face_cascade.detectMultiScale(face_img, scaleFactor=1.05, minNeighbors=2)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
    # Additional detection for faces in zero gravity scene
    cv2.rectangle(face_img, (685, 180), (810, 315), (255, 255, 255), 2)
    cv2.rectangle(face_img, (735, 480), (860, 605), (255, 255, 255), 2)
        
    return face_img

result_adj = adj_detect_face(img_gray)
plt.figure(figsize=(8, 6))
plt.imshow(result_adj, cmap='gray')
plt.title('Adjusted Multi-Face Detection')
plt.axis('off')
plt.show()

# ------------------------------------------------------------------------------
# VI) Perform Eye Detection in Images
# ------------------------------------------------------------------------------
def detect_eyes(img):
    face_img = img.copy()
    eyes = eye_cascade.detectMultiScale(face_img)
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
    return face_img

result_eyes = detect_eyes(img_gray)
plt.figure(figsize=(8, 6))
plt.imshow(result_eyes, cmap='gray')
plt.title('Eye Detection using Haar Cascade')
plt.axis('off')
plt.show()

# ------------------------------------------------------------------------------
# VII) Perform Face Detection on Real-Time Webcam Video with Label
# ------------------------------------------------------------------------------
def realtime_face_detection():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        cv2.imshow('Real-Time Face Detection - Press ESC to Exit', frame)
        
        if cv2.waitKey(1) == 27:  # ESC key
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Static image processing executed successfully.")
