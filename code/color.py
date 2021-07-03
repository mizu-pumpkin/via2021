#!/usr/bin/env python
# COLOR. Construye un clasificador de objetos en base a la similitud de los
# histogramas de color del ROI (de los 3 canales por separado).
# see FAQ
# TODO: Opcional: Segmentación densa por reproyección de histograma.
# usar el notebook colorseg como referencia (y puede que codebook también)


# █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
# █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█


import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import ROI, putText
from collections import deque


# ▄▀█ █░█ ▀▄▀
# █▀█ █▄█ █░█


bins = np.arange(256).reshape(256,1)
colors = [ (255,0,0),(0,255,0),(0,0,255) ] #('b','g','r')

# https://github.com/opencv/opencv/blob/master/samples/python/hist.py
def calcHist(im, channel):
    hist_item = cv.calcHist([im],[channel],None,[256],[0,256])
    cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
    return np.int32(np.around(hist_item))

def DrawHistogramOnImage(image, h_im):
    width,height,_ = h_im.shape
    for channel, color in enumerate(colors):
        hist = calcHist(image, channel)
        # Editamos los valores para que no salga boca abajo
        # y para que ocupe todo el ancho
        xs = bins*(width/256)
        ys = height-hist*(height/300)
        pts = np.int32(np.column_stack((xs,ys)))
        cv.polylines(h_im,[pts],False,color,thickness=2)

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

def DrawNamedWindow(name, im, x, y, w, h, normal=True):
    if normal:
        cv.namedWindow(name, cv.WINDOW_NORMAL)
    else:
        cv.namedWindow(name)
    cv.resizeWindow(name, w, h)
    cv.moveWindow(name, x, y)
    cv.imshow(name, im)


# ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
# █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█


maxlen = 6
max_h = 80

cv.namedWindow('MainWindow', cv.WINDOW_NORMAL)
cv.moveWindow('MainWindow', 0, 0)
X,Y,_,_ = cv.getWindowImageRect('MainWindow')

region = ROI('MainWindow')
models = deque(maxlen=maxlen) # para limitar los modelos

for key, frame in autoStream():
    iX,iY,W,H = cv.getWindowImageRect('MainWindow')
    
    if region.roi:
        # Select ROI
        [x1,y1,x2,y2] = region.roi
        roi = frame[y1:y2+1, x1:x2+1]

        roi_copy = roi.copy()

        # Cuando se marca un ROI con el ratón se muestran los
        # histogramas (normalizados) de los 3 canales por separado.
        DrawHistogramOnImage(roi_copy, roi)

        # Show histogram
        histogram = np.zeros((300,256,3))
        DrawHistogramOnImage(roi_copy, histogram)
        DrawNamedWindow('histogram', histogram, iX+W, iY+Y+max_h, max_h*int(maxlen/2), max_h*int(maxlen/2))

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
            distances.append(dist/100000)

        # Las distancias se muestran arriba a la izquierda
        distString = ''
        for dist in distances: distString += '%.2f' % dist + ' '
        putText(frame, distString.strip())

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
        # Show saved models as single picture
        # La ventana de models puede construirse con unas miniaturas
        # reescaladas a una misma altura predeterminada y ancho
        # proporcional al original, o simplemente a un cuadrado fijo.
        resized_models = [cv.resize(m, (max_h, max_h)) for m in models] # width: int(m.shape[1] * max_h / m.shape[0])
        DrawNamedWindow('models', cv.hconcat(resized_models), iX+W, iY, max_h*maxlen, max_h, normal=False)

        # Show most similar model
        # La menor distancia nos indica el modelo más parecido, y se muestra
        # en la ventana "detected". Si la menor distancia es muy grande se
        # puede rechazar la decisión y y mostrar un recuadro negro.
        detected = resized_models[distances.index(min(distances))]
        DrawNamedWindow('detected', detected, iX+W+max_h*int(maxlen/2), iY+Y+max_h, max_h*int(maxlen/2), max_h*int(maxlen/2))

    cv.imshow('MainWindow',frame)

cv.destroyAllWindows()