#!/usr/bin/env python
# RA. Crea un efecto de realidad aumentada interactivo: esto significa que:
# a) los objetos virtuales deben cambiar de forma, posición o tamaño siguiendo alguna lógica;
# b) el usuario puede observar la escena cambiante desde cualquier punto de vista moviendo la cámara alrededor del marcador;
# c) el usuario puede marcar con el ratón en la imagen puntos del plano de la escena para interactuar con los objetos virtuales.


# █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
# █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█


import cv2          as cv
import numpy        as np

import matplotlib.path as mpltPath

from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose
from umucv.util     import cube, showAxes
from umucv.contours import extractContours, redu


# ▄▀█ █░█ ▀▄▀
# █▀█ █▄█ █░█


# matriz de calibración sencilla dada la resolución de la imagen y el fov
# horizontal en grados
def Kfov(sz,hfovd):
    hfov = np.radians(hfovd)
    f = 1/np.tan(hfov/2)
    # print(f)
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])

# seleccionamos los contornos que pueden reducirse al número de lados deseado
def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]

# generamos todos los posibles puntos de partida
def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]

# probamos todas las asociaciones de puntos imagen - modelo y nos quedamos
# con la que produzca menos error de ajuste
def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p.rms)[0]


# █▀█ ▄▀█   █▀█ █▄▄ ░░█ █▀▀ █▀▀ ▀█▀ █▀
# █▀▄ █▀█   █▄█ █▄█ █▄█ ██▄ █▄▄ ░█░ ▄█


# este es nuestro marcador, define el sistema de coordenadas del mundo 3D
marker = np.array([
    [0,   0,   0],
    [0,   1,   0],
    [0.5, 1,   0],
    [0.5, 0.5, 0],
    [1,   0.5, 0],
    [1,   0,   0] ])

square = np.array([
    [0,   0,   0],
    [0,   1,   0],
    [1,   1,   0],
    [1,   0,   0] ])

# The picture we'll be using as texture
pikachu = cv.imread('images/ra/pikachu.png')
h,w = pikachu.shape[:2]
src = np.array([[0,0],[0,h],[w,h],[w,0]])

def drawPikachu(cube, frame):
    sides = [
        ( cube[5][0], np.array([cube[5], cube[0], cube[1], cube[6]]) ),
        ( cube[6][0], np.array([cube[6], cube[1], cube[2], cube[7]]) ),
        ( cube[7][0], np.array([cube[7], cube[2], cube[3], cube[8]]) ),
        ( cube[8][0], np.array([cube[8], cube[3], cube[0], cube[5]]) )
    ]

    # A max of 3 sides can be seen at any given time and the cube will
    # never appear from below. The 2 sides with the highest value of x,
    # will be the ones that are actually seen. Draw them last.
    sides.sort(key=lambda s : s[0], reverse=True)

    # Top side must be the last to be drawn
    sides.append( (cube[5][0], cube[5:9]) )

    for _, dst in sides:
        H = cv.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        cv.warpPerspective(pikachu,H,(WIDTH,HEIGHT),frame,0,cv.BORDER_TRANSPARENT)


# ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
# █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█


# The clicked point on screen
points = [1]
points[0] = (0,0)

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points[0] = (x,y)

cv.namedWindow('AR')
cv.setMouseCallback('AR', fun)

stream = autoStream()
HEIGHT, WIDTH = next(stream)[1].shape[:2]
K = Kfov( (WIDTH, HEIGHT), 60 )

UD = 0 # up/down movement
LR = 0 # left/right movement
sin = 0
cos = 1
minSize = 0.25 # minimum size of the cube
maxSize = 1    # maximum size of the cube
size = minSize # actual size of the cube

pikapika = False # if True, draw Pikachu

for n, (key, frame) in enumerate(stream):

    # Change the size of the cube
    if key == ord('+') and size < maxSize:
        size += minSize
    if key == ord('-') and size > minSize:
        size -= minSize

    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)

    good = polygons(cs,6,3)
    poses = []
    for g in good:
        p = bestPose(K,g,marker)
        if p.rms < 2:
            poses += [p.M]

    for M in poses:

        # capturamos el color de un punto cerca del marcador para borrarlo
        # dibujando un cuadrado encima
        x,y = htrans(M, (0.7,0.7,0) ).astype(int)
        b,g,r = frame[y,x].astype(int)
        cv.drawContours(frame,[htrans(M,square*1.1+(-0.05,-0.05,0)).astype(int)], -1, (int(b),int(g),int(r)) , -1, cv.LINE_AA)

        # Move the cube along the marker
        oldsin = sin
        sin = np.sin(n/50)

        if oldsin <= sin and sin >= 0 and sin <= 1: # up
            UD = abs(sin)
            LR = 0
        elif oldsin > sin and sin >= 0 and sin <= 1: # right
            UD = 1
            LR = abs(np.cos(n/50))
        elif oldsin >= sin and sin <= 0 and sin >= -1: # down
            UD = abs(np.cos(n/50))
            LR = 1
        elif oldsin < sin and sin <= 0 and sin >= -1: # left
            UD = 0
            LR = abs(sin)

        fig3D = cube*size + ((1-size)*LR, (1-size)*UD, 0)
        fig2D = htrans(M, fig3D).astype(int)

        # Check if the clicked point is inside the figure
        if points[0] != (0,0):
            if mpltPath.Path(fig2D).contains_points(points):
                pikapika = not pikapika
            points[0] = (0,0)
        
        if pikapika:
            # If the cube was clicked, draw surprised Pikachu
            drawPikachu(fig2D, frame)
        else:
            # Otherwise draw a wire cube
            cv.drawContours(frame, [fig2D], -1, (0,128,0), 3, cv.LINE_AA)

    cv.imshow('AR',frame)

cv.destroyAllWindows()