import glob
import cv2
import os
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,accuracy_score
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

tamaño_imagenes = 250*250
imagenes = [cv2.imread(file,0) for file in glob.glob("C:/Users/Usuario/PycharmProjects/OpenCV/images/*.png")]
imagenes_identidades = glob.glob("C:/Users/Usuario/PycharmProjects/OpenCV/images/*.png")
lista = imagenes.copy()

IDENTIDADES = []
for i in imagenes_identidades:
    IDENTIDADES.append(i[47:49])

random.Random(7).shuffle(lista)
random.Random(7).shuffle(IDENTIDADES)

cantidad_entrenamiento = int(len(lista) * 0.8)          #299
cantidad_prueba = (len(lista) - cantidad_entrenamiento) #75

imagenes_entrenamiento = lista[:cantidad_entrenamiento]
imagenes_prueba = lista[cantidad_entrenamiento:]

IDENTIDADES_ENTRENAMIENTO = IDENTIDADES[:cantidad_entrenamiento]
IDENTIDADES_PRUEBA = IDENTIDADES[cantidad_entrenamiento:]

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

pca_entrenamiento = PCA(n_components=265)
pca_entrenamiento.fit(matriz_entrenamiento)
#print(np.cumsum(pca_entrenamiento.explained_variance_ratio_))

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

prueba_predicho = ARREGLO_IDENTIDADES_MENORES_DISTANCIAS.copy()
prueba_real = IDENTIDADES_PRUEBA.copy()
etiquetas = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42']


cm = confusion_matrix(prueba_real, prueba_predicho, etiquetas)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Matriz de Confusion(Eigenfaces)'+ '\nPrecision:{:.3f}'.format(accuracy_score(prueba_real, prueba_predicho)))
sns.heatmap(cm,annot=True,center=True,xticklabels=etiquetas,yticklabels=etiquetas, annot_kws={"size": 5},linewidths=0.5,linecolor='black',square=True)
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.show()
