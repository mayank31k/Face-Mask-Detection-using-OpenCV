import numpy as np
import cv2
import random

#  cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

face_cascade = cv2.CascadeClassifier('haar_Cascade\\xml\\haarcascade_frontalface_default.xml')

mouth_cascade = cv2.CascadeClassifier('haar_Cascade\\xml\\haarcascade_mcs_mouth.xml')




bw_threshold = 80

font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (255, 255, 255)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "!!Thank You for wearing MASK!!"
not_weared_mask = "Please wear MASK to defeat Corona!!"


cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)


    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)


    if(len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img, "No face found...", org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    elif(len(faces) == 0 and len(faces_bw) == 1):
    	cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)

        if(len(mouth_rects) == 0):
            cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
        else:
            for (mx, my, mw, mh) in mouth_rects:

                if(y < my < y + h):
                    cv2.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)

                    break

    cv2.imshow('Mask Detection', img)
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
