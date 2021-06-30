#!/usr/bin/env python
# RECTIF. Rectifica la imagen de un plano para medir distancias (tomando
# manualmente referencias conocidas). Por ejemplo, mide la distancia entre las
# monedas en coins.png o la distancia a la que se realiza el disparo en
# gol-eder.png. Verifica los resultados con imágenes originales tomadas por ti.


# █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
# █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█


import cv2 as cv
import numpy as np

from umucv.stream   import autoStream
from umucv.util import putText
from collections import deque


# ▄▀█ █░█ ▀▄▀
# █▀█ █▄█ █░█

W_cm_carnet = 8.5
W_px_carnet = 173
H_cm_carnet = 5.3

refCarnet = np.array([
    [0, 0],
    [0, H_cm_carnet],
    [W_cm_carnet, H_cm_carnet],
    [W_cm_carnet, 0]
]) * 20 + np.array([150, 400]) # scale, desp

viewCarnet = np.array([
    [323, 462],
    [510, 560],
    [626, 311],
    [466, 258]
])


# refRugby = np.array([
#     [0, 0],
#     [0, 100],
#     [69, 100],
#     [69, 0]
# ]) * 3 + np.array([100, 150]) # scale, desp

# viewRugby = np.array([
#     [185, 397],
#     [578, 529],
#     [724, 337],
#     [525, 320]
# ])


# refTennis = np.array([
#     [0, 0],
#     [0, 8.23],
#     [23.77, 8.23],
#     [23.77, 0]
# ]) * 15 + np.array([100, 200]) # scale, desp

# viewTennis = np.array([
#     [74, 253],
#     [376, 257],
#     [369, 110],
#     [166, 109]
# ])


# ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
# █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█


view = viewCarnet
ref = refCarnet
width_cm = W_cm_carnet
width_px = W_px_carnet

points = []
ruler = deque(maxlen=2)
show = False

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append( (x, y) )

def fun2(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        ruler.append( (x, y) )

cv.namedWindow('input')
cv.setMouseCallback('input', fun)

cv.namedWindow('rectif')
cv.setMouseCallback('rectif', fun2)

for key,frame in autoStream():

    if key == ord('x'):
        points = []
        ruler = deque(maxlen=2)
        show = False
    
    if key == ord('a'):# and len(points) >= 4:
        #TODO:view = np.array(points)
        H,_ = cv.findHomography(view, ref)
        rectif = cv.warpPerspective(frame, H, (800,600))
        show = True
    
    if show:
        rectifCopy = rectif.copy()
        for p in ruler:
            cv.circle(rectifCopy, p, 3, (0,0,255), -1)
        # Medimos las distancias marcadas
        if len(ruler) == 2:
            cv.line(rectifCopy, ruler[0],ruler[1],(0,0,255))
            c = np.mean(ruler, axis=0).astype(int)
            d = np.linalg.norm(np.array(ruler[1])-ruler[0])
            d = d * width_cm / width_px
            putText(rectifCopy, f'{d:.1f} cm', c)
        # Mostramos la imagen rectificada
        cv.imshow('rectif', rectifCopy)

    for p in points:
        x,y = p
        cv.circle(frame, (x,y), 3, (0,0,255), -1)
        putText(frame,f'{(x,y)}', (x,y))

    cv.imshow('input', frame)

cv.destroyAllWindows()