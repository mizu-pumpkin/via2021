#!/usr/bin/env python
# PANO. Crea automáticamente un mosaico a partir de las imágenes en una carpeta.
# Las imágenes no tienen por qué estar ordenadas ni formar una cadena lineal y
# no sabemos el espacio que ocupa el resultado. El usuario debe intervenir lo
# menos posible. Recuerda que debe tratarse de una escena plana o de una escena
# cualquiera vista desde el mismo centro de proyección. Debes usar homografías.
# Compara el resultado con el que obtiene la utilidad de stitching de OpenCV.


# █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
# █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█


import cv2 as cv
import numpy as np
import glob

from umucv.stream import autoStream
from umucv.util import putText
from umucv.htrans import desp, scale


# ▄▀█ █░█ ▀▄▀
# █▀█ █▄█ █░█


# utilidad para devolver el número de correspondencias y la homografía entre dos imágenes

sift = cv.AKAZE_create()
bf = cv.BFMatcher()

def match(query, model):

    x1 = cv.cvtColor(query, cv.COLOR_BGR2GRAY)
    x2 = cv.cvtColor(model, cv.COLOR_BGR2GRAY)

    (k1, d1) = sift.detectAndCompute(x1, None)
    (k2, d2) = sift.detectAndCompute(x2, None)
    
    matches = bf.knnMatch(d1,d2,k=2)

    # Apply ratio test
    good = []
    for m in matches:
        if len(m) == 2:
            best, second = m
            if best.distance < 0.75*second.distance:
                good.append(best)
    
    # findHomography doesn't work with less than 4 points
    if len(good) < 4: return len(good), None
    
    # a partir de los matchings seleccionados construimos los arrays de puntos que necesita findHomography
    src_pts = np.array([ k2[m.trainIdx].pt for m in good ]).astype(np.float32).reshape(-1,2)
    dst_pts = np.array([ k1[m.queryIdx].pt for m in good ]).astype(np.float32).reshape(-1,2)
    
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3) # cv.LMEDS
    
    # mask viene como una array 2D de 0 ó 1, lo convertimos a un array 1D de bool
    return sum(mask.flatten() > 0), H


# ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
# █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█


threshold = 30

# Load the images
pano = [cv.imread(x) for x in sorted( glob.glob('images/pano/*.jpg') )]

# Sort matched images by number of matching points
sortedMatches = sorted([(match(p,q)[0],i,j) for i,p in enumerate(pano) for j,q in enumerate(pano) if i< j],reverse=True)

# Remove those below a certain threshold
sortedMatches = [s for s in sortedMatches if s[0] >= threshold]

# Find the center image (the one that matches with more pictures)
maxMatches = [0] * len(pano)
for sm in sortedMatches:
    matches, fst, snd = sm
    maxMatches[fst] += matches
    maxMatches[snd] += matches

center = maxMatches.index(max(maxMatches))

# Set sizes for the result image
h,w,_ = pano[center].shape
x,y = 300,300
T = desp((x,y)) @ scale([0.2,0.2])
size = (x*2,y*2)

# Put the center image in the result
base = cv.warpPerspective(pano[center], T , size)

Hs = [None] * len(pano) # composite homographies to the center
used = [False] * len(pano) # if the image was already put in the result
used[center] = True

# While there are still images to put in the result
while len(sortedMatches) > 0 and not all(used):

    for sm in sortedMatches:
        _, fst, snd = sm
        
        # Check if one of the images is already in the result
        # and match it with one that is not 
        if used[fst] and not used[snd]:
            x1 = fst
            x2 = snd
        elif used[snd] and not used[fst]:
            x1 = snd
            x2 = fst
        else: continue

        _,H = match(pano[x1],pano[x2])

        # Composite the homography if the image is not directly
        # connected to the center image
        if Hs[x1] is not None:
            H = Hs[x1]@H

        print('Matching '+str(x1)+' and '+str(x2))
        cv.warpPerspective(pano[x2], T@H, size, base, 0, cv.BORDER_TRANSPARENT)
        used[x2] = True

        # Save the homography for future compositions
        Hs[x2] = H
    
    # Remove matches that we don't need anymore
    sortedMatches = [s for s in sortedMatches if not (used[s[1]] and used[s[2]]) ]

for key, frame in autoStream():
    cv.imshow('pano',base)

cv.destroyAllWindows()