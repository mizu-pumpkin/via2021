#!/usr/bin/env python
# # ACTIVIDAD. Construye un detector de movimiento en una región de interés
# de la imagen marcada manualmente.
# Guarda 2 ó 3 segundos de la secuencia detectada en un archivo de vídeo.
# TODO: Opcional: muestra el objeto seleccionado anulando el fondo.

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import Video, ROI, putText

video = Video(fps=25)
video.ON = False

mainWindowName = 'input'
cv.namedWindow(mainWindowName)
cv.moveWindow(mainWindowName, 0, 0)

region = ROI(mainWindowName)

bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)
kernel = np.ones((3,3),np.uint8)

for key, frame in autoStream():

    if key == ord('g'):
        video.ON = not video.ON
    
    #frame = cv.flip(frame, 1)
    
    if region.roi:
        # Select ROI
        [x1,y1,x2,y2] = region.roi
        act_roi = frame[y1:y2+1, x1:x2+1]

        # Detect activity
        # Remove noise from ROI
        noiseless = cv.cvtColor(act_roi, cv.COLOR_BGR2GRAY)
        noiseless = cv.GaussianBlur(noiseless, (21, 21), 0)
        # Detected activity mask
        fgmask = bgsub.apply(noiseless)
        fgmask = cv.erode(fgmask,kernel,iterations = 1)
        fgmask = cv.medianBlur(fgmask,3)
        # Detected object
        obj = act_roi.copy()
        obj[fgmask==0] = 0
        cv.imshow('Activity', obj)

        # Write video if any activity is detected
        if fgmask.any():
            video.write(act_roi)
            putText(frame, 'Activity detected', orig=(x1,y1-8))
            if video.ON: cv.circle(act_roi,(15,15),6,(0,0,255),-1)

        # Draw ROI rectangle on frame
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)

        # Reset ROI
        if key == ord('x'):
            region.roi = []
            cv.destroyWindow('Activity')

    putText(frame, 'Video record: '+('ON' if video.ON else 'OFF') )
    cv.imshow(mainWindowName,frame)

cv.destroyAllWindows()
video.release()