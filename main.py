import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Below is used to Pictures 
# img = cv2.imread('saint-tropez-post-malone-2.jpg')
# img = cv2.imread('post_bieber.jpg')

#Below is used for webcam 
webcam = cv2.VideoCapture(0)

while True:
    frame_read, frame = webcam.read()
    #Convert picture to gray scale for faster result
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x1, y1, x2, y2) in face_coordinates:
        cv2.rectangle(frame, (x1,y1), (x1+x2, y1+ y2), (randrange(256),randrange(256),randrange(256)), 5)

    cv2.imshow('Marcelas Face Detector', frame)
    cv2.waitKey(1)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break


# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Below is a simple piece of code that will be used in a loop and find face 
# (x1, y1, x2, y2) = face_coordinates[0]
# cv2.rectangle(img, (x1,y1), (x1+x2, y1+ y2), (0,255,0), 2)

# for (x1, y1, x2, y2) in face_coordinates:
#     cv2.rectangle(img, (x1,y1), (x1+x2, y1+ y2), (randrange(256),randrange(256),randrange(256)), 2)

# cv2.imshow('Marcelas Face Detector', img)
# cv2.waitKey()
# print("Code Complete")