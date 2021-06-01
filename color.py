#!/usr/bin/env python
# COLOR. Construye un clasificador de objetos en base a la similitud de los
# histogramas de color del ROI (de los 3 canales por separado).
# see FAQ
# Opcional: Segmentación densa por reproyección de histograma.

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import ROI, putText

mainWindowName = 'input'
cv.namedWindow(mainWindowName)
cv.moveWindow(mainWindowName, 0, 0)

region = ROI(mainWindowName)
models = []

for key, frame in autoStream():
    
    frame = cv.flip(frame, 1)
    
    if region.roi:
        # Select ROI
        [x1,y1,x2,y2] = region.roi
        roi = frame[y1:y2+1, x1:x2+1]

        if key == ord('s'):
            models.append(roi.copy())

        # TODO: Cuando se marca un ROI con el ratón se muestran los
        # histogramas (normalizados) de los 3 canales por separado.

        # TODO: En todo momento (siempre que haya algún modelo guardado)
        # se comparan los histogramas del ROI actual con los de todos los modelos.

        # Las distancias se muestran arriba a la izquierda
        distancias = []
        for i, m in enumerate(models):
            distancias.append(i) #distancia_histogramas(m, roi)
        putText(frame, f'{str(distancias)}')

        # Draw ROI rectangle on frame
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        
        # Reset ROI
        if key == ord('x'):
            region.roi = []
            models = []
            cv.destroyWindow('ROI')
            cv.destroyWindow('models')
            cv.destroyWindow('detected')

    if len(models) > 0:

        # Show saved models as single picture
        max_h = 80
        resized_models = [cv.resize(m, (int(m.shape[1] * max_h / m.shape[0]), max_h)) for m in models]
        cv.imshow('models', cv.hconcat(resized_models))

        # Show most similar model
        # La menor distancia nos indica el modelo más parecido,
        # y se muestra en la ventana "detected".
        detected = resized_models[min(distancias)]
        cv.imshow('detected', detected)

    h,w,_ = frame.shape
    cv.imshow(mainWindowName,frame)

cv.destroyAllWindows()