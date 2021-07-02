# Ejercicio SWAP: Escribe una función copy_quad(p,src,q,dst) que mueve
# un cuadrilátero p (los cuatro puntos almacenados en un array  4×2 )
# de una imagen src a un cuadrilátero q en la imagen dst.
# (Pista: necesitarás un máscara y es conveniente reducir la operación
# al "bounding box" de las regiones.) Utilízala para reproducir el
# experimento anterior marcando los puntos de referencia manualmente.


# █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
# █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█


import cv2 as cv
import numpy as np

from umucv.stream import autoStream
from umucv.util import putText


# ▄▀█ █░█ ▀▄▀
# █▀█ █▄█ █░█


def draw_lines(points):

    color = (0,0,255)

    for p in points:
        cv.circle(frame, p, 3, color, -1)

    i = 1
    while i < 4 and i < len(points):
        cv.line(frame, points[i-1], points[i], color)
        i += 1
    
    if len(points) > 3:
        cv.line(frame, points[3], points[0], color)

    i = 5
    while i < 8 and i < len(points):
        cv.line(frame, points[i-1], points[i], color)
        i += 1
    
    if len(points) > 7:
        cv.line(frame, points[7], points[4], color)


# https://stackoverflow.com/questions/30901019/extracting-polygon-given-coordinates-from-an-image-using-opencv
def mask_zone(points):
    mask = np.zeros((frame.shape[0], frame.shape[1]))
    cv.fillConvexPoly(mask, points, 1)
    mask = mask.astype(np.bool)
    out = np.zeros_like(frame)
    out[mask] = frame[mask]
    return out, mask


def copy_quad(p, q, frame):

    src, mask1 = mask_zone(p)
    dst, mask2 = mask_zone(q)

    Hpq = cv.getPerspectiveTransform(p.astype(np.float32), q.astype(np.float32))
    Hqp = cv.getPerspectiveTransform(q.astype(np.float32), p.astype(np.float32))

    out1 = frame.copy()
    out2 = frame.copy()

    h,w,_ = frame.shape

    cv.warpPerspective(src,Hpq,(w,h),out1,0,cv.BORDER_TRANSPARENT)
    cv.warpPerspective(dst,Hqp,(w,h),out2,0,cv.BORDER_TRANSPARENT)

    frame[mask2] = out1[mask2]
    frame[mask1] = out2[mask1]


# ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
# █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█


points = []

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(points) == 8:
            points.clear()
        points.append((x,y))

cv.namedWindow('input')
cv.setMouseCallback('input', fun)

for key, frame in autoStream():

    if key == ord('x'):
        points = []
    
    if len(points) == 8:
        p = np.array([points[0], points[1], points[2], points[3]])
        q = np.array([points[4], points[5], points[6], points[7]])
        copy_quad(p, q, frame)
        putText(frame, 'Swapped. Press x to reset or select new points.')
    else:
        draw_lines(points)
    
    cv.imshow('input',frame)
    
cv.destroyAllWindows()