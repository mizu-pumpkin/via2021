#!/usr/bin/env python
# RECTIF. Rectifica la imagen de un plano para medir distancias (tomando
# manualmente referencias conocidas). Por ejemplo, mide la distancia entre las
# monedas en coins.png o la distancia a la que se realiza el disparo en
# gol-eder.png. Verifica los resultados con imágenes originales tomadas por ti.


# █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
# █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█


import cv2 as cv
import numpy as np
from collections import deque
import math

from umucv.stream import autoStream
from umucv.util import putText
from umucv.htrans import htrans, desp, scale


# ▄▀█ █░█ ▀▄▀
# █▀█ █▄█ █░█


W_cm_carnet = 8.5
H_cm_carnet = 5.3

refCarnet = np.array([
    [0, 0],
    [W_cm_carnet, 0],
    [W_cm_carnet, H_cm_carnet],
    [0, H_cm_carnet]
]) * 20 # !!!: if you change this 20, change also 'W_px_carnet = 173'

W_px_carnet = 173 # !!!: if you change this 173, change also '* 20'

viewCarnet = np.array([
    [323, 462],
    [466, 258],
    [626, 311],
    [510, 560]
])

def transform_corners(H, img):
    h,w,_ = img.shape
    corners = np.array([ [0,0],[0,h],[w,h],[w,0] ])
    trans_corners = htrans(H, corners)
    xx = [x for x,_ in trans_corners]
    yy = [y for _,y in trans_corners]
    return min(xx), max(xx), min(yy), max(yy)


# ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
# █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█


view = viewCarnet # change this value to change scenario
ref = refCarnet # change this value to change scenario
width_cm = W_cm_carnet # change this value to change scenario
width_px = W_px_carnet # change this value to change scenario

points = []
ruler = deque(maxlen=2)
show = False

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append( (x, y) )

def fun2(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        ruler.append( (x, y) )

cv.namedWindow('MainWindow')
cv.setMouseCallback('MainWindow', fun)

cv.namedWindow('rectif')
cv.setMouseCallback('rectif', fun2)

for key,frame in autoStream():

    if key == ord('x'):
        points = []
        ruler = deque(maxlen=2)
        show = False
    
    if key == ord('a'):# and len(points) >= 4:
        #view = np.array(points)
        H,_ = cv.findHomography(view, ref)

        # Find the size to fit the image nicely
        xmin, xmax, ymin, ymax = transform_corners(H, frame)
        width = int( math.ceil(xmax - xmin) )
        height = int( math.ceil(ymax - ymin) )
        T = desp((-xmin, -ymin))
        size = (width, height)
        
        # Warp the perspective based on the found size
        rectif = cv.warpPerspective(frame, T@H, size)
        show = True
    
    if show:
        # Resize the image to fit nicely on average screen
        fac = max(width/800, height/600)
        resized = cv.resize(rectif, ( int(width/fac), int(height/fac) ))

        # Draw the clicked points
        for p in ruler:
            cv.circle(resized, p, 3, (0,0,255), -1)
        
        # Show measures
        if len(ruler) == 2:
            cv.line(resized, ruler[0],ruler[1],(0,0,255))
            c = np.mean(ruler, axis=0).astype(int)
            d = np.linalg.norm(np.array(ruler[1])-ruler[0])
            d = d * width_cm / (width_px / fac)
            putText(resized, f'{d:.1f} cm', c)
        
        # Show rectified and resized image
        cv.imshow('rectif', resized)

    # Show the chosen points for the transformation
    for p in view:
        x,y = p
        cv.circle(frame, (x,y), 3, (0,0,255), -1)
        putText(frame,f'{(x,y)}', (x,y))

    cv.imshow('MainWindow', frame)

cv.destroyAllWindows()