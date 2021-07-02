# CALIBRACIÓN


# a) Realiza una calibración precisa de tu cámara mediante
#    múltiples imágenes de un chessboard.
#
# RMS: 3.217471748824684
# camera matrix:
#  [[3.33350026e+03 0.00000000e+00 1.37895391e+03]
#  [0.00000000e+00 3.30693258e+03 2.09307524e+03]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# distortion coefficients:  [ 0.21264965 -1.51913124 -0.0067358  -0.02069358  2.10661863]


f = 3333.50026

print(f'Distancia focal precisa: {f:.2f} px')


# b) Haz una calibración aproximada con un objeto de tamaño
#    conocido y compara con el resultado anterior.
#
# u = f (X/Z)
# f = u * Z / X


Z = 60 # distancia cámara en cm
X_book = 18 # altura en cm
u_book = 924 # altura en px  
X_switch = 21 # altura en cm
u_switch = 1031 # altura en px

f_book = u_book * Z / X_book
f_switch = u_switch * Z / X_switch

print(f'Distancia focal aproximada (libro): {f_book:.2f} px')
print(f'Distancia focal aproximada (Nintendo Switch): {f_switch:.2f} px')

width = 3000
height = 4000

import math

hFOV = math.degrees( math.atan(width / 2 / 690) ) * 2
vFOV = math.degrees( math.atan(height / 2 / 690) ) * 2

print(f'FOV horizontal: {hFOV:.1f}°')
print(f'FOV vertical: {vFOV:.1f}°')


# c) Determina a qué altura hay que poner la cámara para
#    obtener una vista cenital completa de un campo de baloncesto.
#
# Z = f * X / u


X_basket = 28 * 100
u_basket = height

Z_basket = f * X_basket / u_basket

print(f'Altura para foto campo basket: {Z_basket/100:.2f} m')


# e) TODO: Opcional: determina la posición aproximada desde la que se ha
#    tomado una foto a partir ángulos observados respecto a puntos
#    de referencia conocidos.


# █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
# █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█


import cv2 as cv
from umucv.stream import autoStream
from collections import deque
import numpy as np
from umucv.util import putText


stream = autoStream()
HEIGHT, WIDTH = next(stream)[1].shape[:2]


def angle(p1, p2):
    u = [ p1[0]-WIDTH/2, p1[1]-HEIGHT/2, f]
    v = [ p2[0]-WIDTH/2, p2[1]-HEIGHT/2, f]

    # cos(alpha) = u . v / (|u| * |v|) 
    cosAlpha = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    return math.degrees( math.acos(cosAlpha) )


# ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
# █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█


# d) Haz una aplicación para medir el ángulo que definen dos puntos
#    marcados con el ratón en el imagen.


points = deque(maxlen=2)

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))

cv.namedWindow('webcam')
cv.setMouseCallback('webcam', fun)

for key, frame in stream:

    for p in points:
        cv.circle(frame, p,3,(0,0,255),-1)
    
    if len(points) == 2:
        cv.line(frame, points[0],points[1],(0,0,255))
        c = np.mean(points, axis=0).astype(int)

        alpha = angle(points[0], points[1])

        putText(frame, f'{alpha:.1f} deg', c)

    cv.imshow('webcam',frame)
    
cv.destroyAllWindows()