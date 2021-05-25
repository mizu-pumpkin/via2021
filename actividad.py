#!/usr/bin/env python
# # ACTIVIDAD. Construye un detector de movimiento en una región de interés
# de la imagen marcada manualmente.
# Guarda 2 ó 3 segundos de la secuencia detectada en un archivo de vídeo.
# Opcional: muestra el objeto seleccionado anulando el fondo.

import cv2 as cv
from umucv.stream import autoStream
from umucv.util import Video, ROI, putText

def condition(key):
    return key == ord('g')

video = Video(fps=25)
# Podemos activar o desactivar la grabación con video.ON
#video.ON = True

mainWindowName = 'input'
cv.namedWindow(mainWindowName)
cv.moveWindow(mainWindowName, 0, 0)

region = ROI(mainWindowName)

bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)

for key, frame in autoStream():
    
    if region.roi:
        # Mostrar el ROI seleccionado en una nueva ventana
        [x1,y1,x2,y2] = region.roi
        trozo = frame[y1:y2+1, x1:x2+1]
        cv.imshow('actividad', trozo)
        cv.imshow('mask', bgsub.apply(trozo) )
        # Grabar según condición
        if condition(key):
            video.write(trozo)
        # Resetear el ROI
        if key == ord('x'):
            region.roi = []
            cv.destroyWindow('actividad')
        # Dibujar el ROI en el frame principal
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow(mainWindowName,frame)

cv.destroyAllWindows()
video.release()