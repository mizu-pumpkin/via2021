#!/usr/bin/env python
# FILTROS. Muestra el efecto de diferentes filtros sobre la imagen en vivo de la
# webcam. Selecciona con el teclado el filtro deseado y modifica sus posibles
# parámetros (p.ej. el nivel de suavizado) con las teclas o con trackbars.
# Aplica el filtro en un ROI para comparar el resultado con el resto de la imagen.
# TODO: Opcional: implementa en Python o C "desde cero" algún filtro y compara la
# eficiencia con OpenCV.


# █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
# █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█


import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import ROI, putText
import scipy.signal      as signal


# ▄▀█ █░█ ▀▄▀
# █▀█ █▄█ █░█


def nothing(x):
    pass

def MakeNamedWindow(name, x, y, w, h):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, w, h)
    cv.moveWindow(name, x, y)


# █▀▀ █ █░░ ▀█▀ █▀▀ █▀█ █▀
# █▀░ █ █▄▄ ░█░ ██▄ █▀▄ ▄█


def cconv(im,k):
    return cv.filter2D(im,-1,k)
    #return signal.convolve2d(im, k, boundary='symm', mode='same')

# Podemos combinar los kernerls de derivada en dirección horizontal y
# vertical para conseguir una medida de borde en cualquier orientación
def filter_border(f_im, i=1):
    ker_hor = np.array([ [0,0,0], [-i,0,i], [0,0,0] ])
    ker_ver = ker_hor.T # np.array([ [0,-i,0], [0,0,0], [0,i,0] ])
    gx = cconv(f_im, ker_hor)
    gy = cconv(f_im, ker_ver)
    return abs(gx) + abs(gy)

# El operador Laplaciano es la suma de las segundas derivadas respecto a cada variable.
# El efecto del filtro Laplaciano es amplificar las frecuencias altas. La siguiente
# máscara se aproxima al Laplaciano: [0,-1,0], [-1,4,-1], [0,-1,0]
def filter_laplacian(f_im):
    return cv.Laplacian(f_im,-1)

# La siguiente máscara calcula la media de un entorno de radio 5
kerSize = 11
ker_blur = np.ones([kerSize,kerSize])
ker_blur = ker_blur/np.sum(ker_blur)
# Se consigue exactamente el mismo efecto con un "box filter".
# Lo interesante es que está implementado internamente usando "imágenes integrales",
# por lo que el tiempo de cómputo es constante, independientemente del tamaño de la
# región que se promedia.
def filter_box(f_im, kerSize=11):
    return cv.boxFilter(f_im, -1, (kerSize,kerSize))

# No obstante, promediar un entorno abrupto de cada pixel produce "artefactos".
# La forma correcta de eliminar detalles es usar el filtro gaussiano, donde los
# pixels cercanos tienen más peso en el promedio.
# El filtro gaussiano tiene varias características importantes:
#   - separable
#   - no introduce detalles: espacio de escala
#   - cascading
#   - Fourier
#   - analogía física
def filter_gaussian(f_im, kerSize=3):
    return cv.GaussianBlur(f_im, (0,0), kerSize)

# FILTROS NO LINEALES

# El filtro de mediana es no lineal. Es útil para eliminar ruido de "sal y pimienta",
# suavizando la imagen sin destruir los bordes. (Requiere pixels de tipo byte.)
def filter_medianBlur(g_im, kerSize=17):
    return cv.medianBlur(g_im, kerSize)

# El filtro bilateral solo promedia pixels cercanos que además tienen un valor similar.
def filter_bilateral(im, kerSize=10):
    return cv.bilateralFilter(im, 0, kerSize, kerSize)


# ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
# █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█


W = 640
H = 480
cv.namedWindow('MainWindow')
cv.resizeWindow('MainWindow', W, H)
cv.moveWindow('MainWindow', 0, 0)

cv.createTrackbar('Brightness', 'MainWindow', 1, 100, nothing) # Brightness
cv.createTrackbar('Border', 'MainWindow', 1, 100, nothing) # Border
cv.createTrackbar('Kernel', 'MainWindow', 1, 30, nothing) # Drunk, Box, Gaussian

w = int(W * 0.35)
h = int(H * 0.35)
MakeNamedWindow('Brightness', W, 0, w, h)
MakeNamedWindow('Border', W+w, 0, w, h)
MakeNamedWindow('Drunk', W+w*2, 0, w, h)
MakeNamedWindow('Box', W, 32+h, w, h)
MakeNamedWindow('Gaussian', W+w, 32+h, w, h)
MakeNamedWindow('Laplacian', W+w*2, 32+h, w, h)

region = ROI('MainWindow')
opt = 0
filters = [0] * 7
f_names = ['No filter', 'Brightness', 'Border', 'Drunk', 'Box', 'Gaussian', 'Laplacian']

for key, frame in autoStream():
    frame = cv.flip(frame, 1) # TODO: quitar antes de entregar

    # Get values from trackbars
    brightness = cv.getTrackbarPos('Brightness', 'MainWindow') / 10
    border = cv.getTrackbarPos('Border', 'MainWindow') / 10
    kernel = cv.getTrackbarPos('Kernel', 'MainWindow')
    if kernel == 0: kernel = 1

    # Define kernels for filters
    ker_bright = np.array([ [0,0,0], [0,brightness,0], [0,0,0] ])

    ker_drunk = np.zeros([kernel,kernel])
    ker_drunk[0, 0] = 1
    ker_drunk[kernel-1, kernel-1] = 1
    ker_drunk = ker_drunk/np.sum(ker_drunk)

    # Apply filters
    filters[0] = frame
    filters[1] = cconv(frame, ker_bright)
    filters[2] = 3*filter_border(frame, border)
    filters[3] = cconv(frame, ker_drunk)
    filters[4] = filter_box(frame, kernel)
    filters[5] = filter_gaussian(frame, kernel)
    filters[6] = filter_laplacian(frame)

    # Show filters
    cv.imshow('Brightness', filters[1] )
    cv.imshow('Border',     filters[2] )
    cv.imshow('Drunk',      filters[3] )
    cv.imshow('Box',        filters[4] )
    cv.imshow('Gaussian',   filters[5] )
    cv.imshow('Laplacian',  filters[6] )
    
    if region.roi:
        # Select ROI
        [x1,y1,x2,y2] = region.roi
        roi = frame[y1:y2+1, x1:x2+1]
        
        # Select filter to apply to frame
        if (key == ord('0')): opt = 0
        if (key == ord('1')): opt = 1
        if (key == ord('2')): opt = 2
        if (key == ord('3')): opt = 3
        if (key == ord('4')): opt = 4
        if (key == ord('5')): opt = 5
        if (key == ord('6')): opt = 6
        filter = filters[opt]
        
        # Apply filter to frame
        frame[y1:y2+1, x1:x2+1] = filter[y1:y2+1, x1:x2+1]

        # Draw ROI rectangle on frame
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)

        # Reset ROI
        if key == ord('x'):
            region.roi = []

    h,w,_ = frame.shape
    putText(frame, f'{f_names[opt]}' )
    cv.imshow('MainWindow',frame)

cv.destroyAllWindows()