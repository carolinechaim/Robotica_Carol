#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

cap = cv2.VideoCapture(0)



hsv1_M = np.array([120,  50,  50])
hsv2_M= np.array([163, 255, 255])

hsv1_B = np.array([10, 50, 50])
hsv2_B = np.array([100, 255, 255])

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    # Our operations on the frame come here
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, hsv1_M, hsv2_M)
    mask2 = cv2.inRange(hsv, hsv1_B, hsv2_B)
    mask = cv2.bitwise_or(mask1, mask2)
    seg = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((4, 4)))

    

    # Display the resulting frame
    cv2.imshow('frame',seg)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

