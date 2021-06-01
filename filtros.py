#!/usr/bin/env python
# FILTROS. Muestra el efecto de diferentes filtros sobre la imagen en vivo de la
# webcam. Selecciona con el teclado el filtro deseado y modifica sus posibles
# parámetros (p.ej. el nivel de suavizado) con las teclas o con trackbars.
# Aplica el filtro en un ROI para comparar el resultado con el resto de la imagen.
# Opcional: implementa en Python o C "desde cero" algún filtro y compara la
# eficiencia con OpenCV.

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import ROI, putText

mainWindowName = 'input'
cv.namedWindow(mainWindowName)
cv.moveWindow(mainWindowName, 0, 0)

region = ROI(mainWindowName)

for key, frame in autoStream():
    
    frame = cv.flip(frame, 1)
    
    if region.roi:
        # Select ROI
        [x1,y1,x2,y2] = region.roi
        roi = frame[y1:y2+1, x1:x2+1]
        cv.imshow('ROI', roi)

        # Draw ROI rectangle on frame
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)

        # Reset ROI
        if key == ord('x'):
            region.roi = []
            cv.destroyWindow('ROI')

    h,w,_ = frame.shape
    putText(frame, f'({w},{h})' )
    cv.imshow(mainWindowName,frame)

cv.destroyAllWindows()