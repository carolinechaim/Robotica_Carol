#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)



hsv1_M = np.array([125,  50,  50])
hsv2_M= np.array([145, 255, 255])
comprimento = 0

hsv1_B = np.array([5, 50, 50])
hsv2_B = np.array([20, 255, 255])
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


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
    seg = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((1, 1)))

    selecao = cv2.bitwise_and(frame, frame, mask=seg)

    blur = cv2.GaussianBlur(selecao,(5,5),0)
    #blur = gray
    # Detect the edges present in the image
    bordas = auto_canny(blur)
    circles = []
    min_contrast = 50
    max_contrast = 250
    linhas = cv2.Canny(blur, min_contrast, max_contrast )



    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.cvtColor(linhas, cv2.COLOR_RGB2BGR)
    bc = blur

    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    circles = None
    circles=cv2.HoughCircles(linhas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=150)
    lista_circulos = []

    if circles is not None:        
        circles = np.uint16(np.around(circles))
        for i in circles[0]:
            #print(i)
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bc,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(bc,(i[0],i[1]),2,(0,0,255),3)
            if len (lista_circulos) < 4:
                lista_circulos.append(i[0])
                lista_circulos.append(i[1])

    if len (lista_circulos)==4:    
        cv2.line(bc,(lista_circulos[0],lista_circulos[1]),(lista_circulos[2],lista_circulos[3]),(255,0,0),5)
        comprimento =math.sqrt((lista_circulos[2]-lista_circulos[0])^2 + (lista_circulos[3]-lista_circulos[1])^2)
        rad = math.atan((lista_circulos[3]-lista_circulos[1])/(lista_circulos[2]-lista_circulos[0]))
        angulo = rad*180/math.pi
        #print (comprimento)
        print (lista_circulos)

    # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])

    #scv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.rectangle(bordas_color,(384,0),(510,128),(0,255,0),3)

    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bc,'Press q to quit',(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    if comprimento != 0:
        cv2.putText(bc,('Comprimento: {} px '.format(str(comprimento))),(0,100), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(bc,('Angulo: {} graus '.format(str(angulo))),(0,150), font, 1,(255,255,255),2,cv2.LINE_AA)


    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    # Display the resulting frame
    cv2.imshow('Detector de circulos',bc)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

