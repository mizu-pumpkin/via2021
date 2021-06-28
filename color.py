#!/usr/bin/env python
# COLOR. Construye un clasificador de objetos en base a la similitud de los
# histogramas de color del ROI (de los 3 canales por separado).
# see FAQ
# Opcional: Segmentación densa por reproyección de histograma.

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import ROI, putText
from collections import deque

bins = np.arange(256).reshape(256,1)
colors = [ (255,0,0),(0,255,0),(0,0,255) ] #('b','g','r')

# https://github.com/opencv/opencv/blob/master/samples/python/hist.py
def calcHist(im, channel):
    hist_item = cv.calcHist([im],[channel],None,[256],[0,256])
    cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
    return np.int32(np.around(hist_item))

def DrawHistogramOnImage(im, h):
    for channel, color in enumerate(colors):
        hist = calcHist(im, channel)
        pts = np.int32(np.column_stack((bins,hist)))
        cv.polylines(h,[pts],False,color,2)

# La comparación entre histogramas puede hacerse de muchas formas.
# Una muy sencilla es la suma de diferencias absolutas en cada canal
# y quedarnos el máximo de los tres canales.
# Los modelos serán en general rectángulos de tamaños diferentes,
# y que tampoco tienen por qué coincidir con el tamaño del ROI que
# queremos clasificar. Esto implica que los histogramas deben normalizarse.
def distancia_histogramas(model, roi):
    # Suma de diferencias absolutas en cada canal
    B = np.sum( cv.absdiff( calcHist(roi, 0) , calcHist(model, 0) ) )
    G = np.sum( cv.absdiff( calcHist(roi, 1) , calcHist(model, 1) ) )
    R = np.sum( cv.absdiff( calcHist(roi, 2) , calcHist(model, 2) ) )
    # y quedarnos el máximo de los tres canales
    return max(B,G,R)


#   ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
#   █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█


cv.namedWindow('MainWindow')
cv.moveWindow('MainWindow', 0, 0)

region = ROI('MainWindow')
models = deque(maxlen=3) # para limitar los modelos a 3

for key, frame in autoStream():
    
    frame = cv.flip(frame, 1)
    
    if region.roi:
        # Select ROI
        [x1,y1,x2,y2] = region.roi
        roi = frame[y1:y2+1, x1:x2+1]

        roi_copy = roi.copy()

        # Cuando se marca un ROI con el ratón se muestran los
        # histogramas (normalizados) de los 3 canales por separado.
        # TODO: cambiar para que no salga boca abajo
        DrawHistogramOnImage(roi_copy, roi)

        # Si se pulsa una cierta tecla se guarda el recuadro como un modelo
        # más y se muestra en la ventana "models" de abajo a la izquierda
        if key == ord('m'):
            model = roi_copy
            models.append(model)

        # En todo momento (siempre que haya algún modelo guardado)
        # se comparan los histogramas del ROI actual con los de todos los modelos.
        distances = []
        for model in models:
            dist = distancia_histogramas(model, roi_copy)
            distances.append(dist/100)

        # Las distancias se muestran arriba a la izquierda
        putText(frame, f'{str(distances)}')

        # Draw ROI rectangle on frame
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        
        # Reset ROI
        if key == ord('x'):
            region.roi = []
            models = []
            cv.destroyWindow('models')
            cv.destroyWindow('detected')
            cv.destroyWindow('histogram')

    if len(models) > 0:
        X,Y,W,H = cv.getWindowImageRect('MainWindow')
        max_h = 80

        # Show saved models as single picture
        # La ventana de models puede construirse con unas miniaturas
        # reescaladas a una misma altura predeterminada y ancho
        # proporcional al original, o simplemente a un cuadrado fijo.
        resized_models = [cv.resize(m, (max_h, max_h)) for m in models] # width: int(m.shape[1] * max_h / m.shape[0])
        cv.namedWindow('models')
        cv.resizeWindow('models', max_h*3, max_h)
        cv.moveWindow('models', X+W, Y)
        cv.imshow('models', cv.hconcat(resized_models))

        # Show most similar model
        # La menor distancia nos indica el modelo más parecido, y se muestra
        # en la ventana "detected". Si la menor distancia es muy grande se
        # puede rechazar la decisión y y mostrar un recuadro negro.
        detected = resized_models[distances.index(min(distances))]
        cv.namedWindow('detected', cv.WINDOW_NORMAL)
        cv.resizeWindow('detected', max_h*3, max_h*3)
        cv.moveWindow('detected', X+W, 32+Y+max_h)
        cv.imshow('detected', detected)

        # Show histogram
        histogram = np.zeros((300,256,3))
        DrawHistogramOnImage(roi_copy, histogram)
        histogram = np.flipud(histogram)
        cv.namedWindow('histogram', cv.WINDOW_NORMAL)
        cv.resizeWindow('histogram', max_h*4, max_h*4)
        cv.moveWindow('histogram', X+W+max_h*3+9, 32+Y)
        cv.imshow('histogram', histogram)

    cv.imshow('MainWindow',frame)

cv.destroyAllWindows()