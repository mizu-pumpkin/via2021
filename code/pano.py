#!/usr/bin/env python


# █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
# █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import math

from umucv.htrans import htrans, desp


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


def transform_corners(H, img):
    h,w,_ = img.shape
    corners = np.array([ [0,0],[0,h],[w,h],[w,0] ])
    trans_corners = htrans(H, corners)
    xx = [x for x,_ in trans_corners]
    yy = [y for _,y in trans_corners]
    return min(xx), max(xx), min(yy), max(yy)


# ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
# █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█


dirPath = '../images/pano/my_scene/'

threshold = 4

# Load the images
pano = [cv.imread(x) for x in sorted( glob.glob(dirPath+'*.jpg') )]

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

# Homography array to store homographies for composition
homographies = [None] * len(pano)

# Array to check if an image was already used
used = [False] * len(pano)
used[center] = True

# Array to store the min/max corners values of the images
corners = [None] * len(pano)
h,w,_ = pano[center].shape
corners[center] = (0, w, 0, h)

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

        print('Matching '+str(x1)+' and '+str(x2))

        _,H = match(pano[x1],pano[x2])

        # Composite the homography if the image is not directly
        # connected to the center image
        if homographies[x1] is not None:
            H = homographies[x1]@H

        # Mark the image as used
        used[x2] = True

        # Save the homography for future compositions
        homographies[x2] = H

        # Save the corners
        corners[x2] = transform_corners(H, pano[x2])
    
    # Remove matches that we don't need anymore
    sortedMatches = [s for s in sortedMatches if not (used[s[1]] and used[s[2]]) ]


# Set sizes for the result image
xmin = int( math.ceil( min([t[0] for t in corners if t is not None]) ) )
xmax = int( math.ceil( max([t[1] for t in corners if t is not None]) ) )
ymin = int( math.ceil( min([t[2] for t in corners if t is not None]) ) )
ymax = int( math.ceil( max([t[3] for t in corners if t is not None]) ) )

width = xmax - xmin
height = ymax - ymin

T = desp((-xmin, -ymin))
size = (width, height)

# Put the center image in the result
result = cv.warpPerspective(pano[center], T , size)

# Put the rest of the images in the result
for i, H in enumerate(homographies):
    if H is not None:
        cv.warpPerspective(pano[i], T@H, size, result, 0, cv.BORDER_TRANSPARENT)

# Write the result to a file
cv.imwrite(dirPath+'result.jpg', result)

# Show the result on screen
plt.imshow( cv.cvtColor(result, cv.COLOR_BGR2RGB) )
plt.show()