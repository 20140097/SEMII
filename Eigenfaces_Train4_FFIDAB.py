import glob
import cv2
import numpy as np
from sklearn.decomposition import PCA

tamaño_imagenes = 250*250
imagenes = [cv2.imread(file,0) for file in glob.glob("C:/Users/Usuario/PycharmProjects/OpenCV/imagesMOD/*.png")]
imagenes_identidades = glob.glob("C:/Users/Usuario/PycharmProjects/OpenCV/imagesMOD/*.png")
lista = imagenes.copy()

IDENTIDADES = []
for i in imagenes_identidades:
    IDENTIDADES.append("{}{}".format(i[50:53],i[56:58]))

IDENTIDADES_FINAL=[]
for i in IDENTIDADES:
    if i[4] == '.':
        IDENTIDADES_FINAL.append(i[:4])
    elif i[3] == 'p':
        IDENTIDADES_FINAL.append(i[:2])
    else:
        IDENTIDADES_FINAL.append(i)

IDENTIDADES_ENTRENAMIENTO = []
IDENTIDADES_PRUEBA = []

POSCICIONES_ENTRENAMIENTO =[]
POSCICIONES_PRUEBA = []

for index,item in enumerate(IDENTIDADES_FINAL):
    if len(item) == 4 and item[3] == 'F':
        IDENTIDADES_ENTRENAMIENTO.append(item)
        POSCICIONES_ENTRENAMIENTO.append(index)
    elif len(item) == 4 and item[3] == 'D':
        IDENTIDADES_ENTRENAMIENTO.append(item)
        POSCICIONES_ENTRENAMIENTO.append(index)
    elif len(item) == 4 and item[3] == 'I':
        IDENTIDADES_ENTRENAMIENTO.append(item)
        POSCICIONES_ENTRENAMIENTO.append(index)
    elif len(item) == 4 and item[3] == 'A':
        IDENTIDADES_ENTRENAMIENTO.append(item)
        POSCICIONES_ENTRENAMIENTO.append(index)
    elif len(item) == 4 and item[3] == 'B':
        IDENTIDADES_ENTRENAMIENTO.append(item)
        POSCICIONES_ENTRENAMIENTO.append(index)
    elif len(item) != 5:
        IDENTIDADES_PRUEBA.append(item)
        POSCICIONES_PRUEBA.append(index)

imagenes_entrenamiento = []
imagenes_prueba = []

for i in POSCICIONES_ENTRENAMIENTO:
    imagenes_entrenamiento.append(lista[i])

for i in POSCICIONES_PRUEBA:
    imagenes_prueba.append(lista[i])

cantidad_entrenamiento = len(imagenes_entrenamiento)
cantidad_prueba = len(imagenes_prueba)

def convertir_matriz(matriz):
    a = []
    for i in matriz:
        b = []
        for j in i:
            b.extend(j)
        a.append(b)
    return a
array_vectores_entrenamiento = convertir_matriz(imagenes_entrenamiento)
array_vectores_prueba = convertir_matriz(imagenes_prueba)

matriz_entrenamiento = np.reshape(array_vectores_entrenamiento, (cantidad_entrenamiento, tamaño_imagenes))
matriz_prueba = np.reshape(array_vectores_prueba, (cantidad_prueba, tamaño_imagenes))


pca_entrenamiento = PCA(n_components=147)
pca_entrenamiento.fit(matriz_entrenamiento)

transform_entrenamiento = pca_entrenamiento.transform(matriz_entrenamiento)
transform_prueba = pca_entrenamiento.transform(matriz_prueba)

ARREGLO_MENORES_DISTANCIAS = []
for i in transform_prueba:
    arreglo_temp = []
    for j in transform_entrenamiento:
        dist = np.linalg.norm(i-j)
        arreglo_temp.append(dist)
    ARREGLO_MENORES_DISTANCIAS.append(arreglo_temp.index(min(arreglo_temp)))

ARREGLO_IDENTIDADES_MENORES_DISTANCIAS = []
for i in ARREGLO_MENORES_DISTANCIAS:
    ARREGLO_IDENTIDADES_MENORES_DISTANCIAS.append(IDENTIDADES_ENTRENAMIENTO[i])



cant_correctos = 0
for i in range(len(ARREGLO_IDENTIDADES_MENORES_DISTANCIAS)):
    if ARREGLO_IDENTIDADES_MENORES_DISTANCIAS[i][0:2] == IDENTIDADES_PRUEBA[i][0:2]:
        cant_correctos = cant_correctos +1
print("{} de {}".format(cant_correctos, cantidad_prueba))
print("Precisión de {}".format(cant_correctos / cantidad_prueba))

