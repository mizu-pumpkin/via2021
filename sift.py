#!/usr/bin/env python
# SIFT. Escribe una aplicación de reconocimiento de objetos (p. ej. carátulas
# de CD, portadas de libros, cuadros de pintores, etc.) con la webcam basada en
# el número de coincidencias de keypoints.


# Cuando no se detecta ningún keypoint los descriptores no se devuelven como un array de
# dimension 0x128, sino como un valor None. Hay que tener cuidado con esto para que no se
# produzcan errores en tiempo de ejecución. Esto puede ocurrir cuando la cámara apunta hacia
# la mesa, o está muy desenfocada.

# Aquí hay un vídeo de ejemplo de lo que se puede conseguir sin muchas complicaciones.
# Se muestra en pequeño el modelo ganador, su porcentaje, y la diferencia con el segundo mejor.
# Observa que los objetos se reconocen aunque no se vean enteros en la imagen, con diferentes
# tamaños, con cualquier rotación en el plano de imagen, y con cierta inclinación de perspectiva.
# Hay también cierta resistencia al desenfoque y a reflejos. (Aunque no se ven en esta breve
# secuencia, pueden producirse clasificaciones erróneas cuando la escena tiene muchos puntos
# y no hay ningún modelo conocido.)

# El algoritmo SIFT está en el repositorio "non-free" de opencv. Si vuestra distribución no lo
# incluye podéis utilizar el método AKAZE, que funciona igual de bien o mejor. Solo hay que cambiar una línea:
#    # sift = cv.xfeatures2d.SIFT_create( ..parámetros.. )
#    sift = cv.AKAZE_create()   # tiene otros parámetros pero la configuración por omisión funciona bastante bien.


# █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
# █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█


import cv2 as cv
import time

from umucv.stream import autoStream
from umucv.util import putText
from pathlib import Path


# ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
# █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█


sift = cv.AKAZE_create() #sift = cv.xfeatures2d.SIFT_create(nfeatures=500)
matcher = cv.BFMatcher()

models = []
models_names = []
models_keypoints = []
models_descriptors = []

path = Path("sift-models")
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

    bestLen = 0
    index = 0
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
            bestLen = len(good)/models_keypoints[i]
            index = i

    # Se puede rechazar la decisión cuando el porcentaje ganador sea pequeño, cuando el segundo
    # mejor sea parecido al primero, o cuando haya pocas coincidencias en la imagen, entre otras
    # situaciones que dependen de la aplicación.
    if bestLen*100 > 1:
        image = cv.resize(models[index], (100, 100))
        img_height, img_width, _ = image.shape
        frame[50:50+img_height, :img_width] = image

        putText(frame, f'{1000*(t1-t0):.0f} ms {1000*(t3-t2):.0f} ms')
        putText(frame ,f'{100*bestLen:.1f} % {models_names[index]}', 
                        orig=(5,36), color=(200,255,200))
    
    cv.imshow("SIFT",frame)