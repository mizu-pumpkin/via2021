#!/usr/bin/env python
# SIFT. Escribe una aplicación de reconocimiento de objetos (p. ej. carátulas
# de CD, portadas de libros, cuadros de pintores, etc.) con la webcam basada en
# el número de coincidencias de keypoints.
# TODO: opcional añadir que se guarden los modelos sobre la marcha


# █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
# █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█


from umucv.stream import autoStream
from umucv.util import putText

import time
import cv2 as cv
from pathlib import Path


# ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
# █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█


sift = cv.AKAZE_create() #sift = cv.xfeatures2d.SIFT_create(nfeatures=500)
matcher = cv.BFMatcher()

models = []
models_names = []
models_keypoints = []
models_descriptors = []

path = Path("../images/sift")
path = path.glob("*.jpg")
for imagepath in path:
    image = cv.imread(str(imagepath))
    keypoints , descriptors = sift.detectAndCompute(image, mask=None)
    models.append(image)
    models_names.append(str(imagepath.stem))
    models_keypoints.append(len(keypoints))
    models_descriptors.append(descriptors)

for key, frame in autoStream():

    t0 = time.time()
    keypoints , descriptors = sift.detectAndCompute(frame, mask=None)
    t1 = time.time()

    # Cuando no se detecta ningún keypoint los descriptores no se devuelven como un array
    # de dimension 0x128, sino como un valor None. Hay que tener cuidado con esto para que
    # no se produzcan errores en tiempo de ejecución.
    if descriptors is not None:

        bestLen = 0
        secondLen = 0
        index = 0
        # Find best matching model
        for i in range(len(models)):

            t2 = time.time()
            # solicitamos las dos mejores coincidencias de cada punto, no solo la mejor
            matches = matcher.knnMatch(descriptors, models_descriptors[i], k=2)
            t3 = time.time()

            # ratio test
            # nos quedamos solo con las coincidencias que son mucho mejores que
            # que la "segunda opción". Es decir, si un punto se parece más o menos lo mismo
            # a dos puntos diferentes del modelo lo eliminamos.
            good = []
            for m in matches:
                if len(m) > 1:
                    best,second = m
                    if best.distance < 0.75*second.distance:
                        good.append(best)
            
            # Si los modelos tienen diferente número de keypoints la comparación debe hacerse teniendo
            # en cuenta el porcentaje de coincidencias, no el valor absoluto.
            if len(good)/models_keypoints[i] > bestLen:
                secondLen = bestLen
                bestLen = len(good)/models_keypoints[i]
                index = i

        # Se puede rechazar la decisión cuando el porcentaje ganador sea pequeño, cuando el segundo
        # mejor sea parecido al primero, o cuando haya pocas coincidencias en la imagen, entre otras
        # situaciones que dependen de la aplicación.
        if bestLen*100 > 1:
            # Se muestra en pequeño el modelo ganador su porcentaje, y la diferencia con el segundo mejor.
            image = cv.resize(models[index], (100, 100))
            img_height, img_width, _ = image.shape
            frame[50:50+img_height, :img_width] = image
            putText(frame, f'{1000*(t1-t0):.0f} ms {1000*(t3-t2):.0f} ms')
            putText(frame ,f'{100*bestLen:.1f} % {models_names[index]} +{100*(bestLen-secondLen):.1f} %', 
                            orig=(5,36), color=(200,255,200))
    
    cv.imshow("SIFT",frame)

cv.destroyAllWindows()