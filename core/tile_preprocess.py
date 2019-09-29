# -*- coding: utf-8 -*-
# OS and file sistem Libraries
import subprocess
import os
import json
from qgis.PyQt.QtCore import QVariant
from qgis.core import *

# Images and computer vision Libraries
try:
    import cv2
except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'opencv-python'])
#Math ans statistics
try:
    import pandas as pd
except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'pandas'])
#library for load and save python object including models

try:
    from joblib import dump, load
except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'joblib'])

try:
    import sklearn
    a=sklearn.__version__
    if a=='0.20.1':
        from sklearn import ensemble
        from sklearn import datasets
        from sklearn.utils import shuffle
        from sklearn.metrics import mean_squared_error, explained_variance_score
    else:
        subprocess.check_call(['python3', '-m', 'pip', 'uninstall', 'scikit-learn','-y'])
        subprocess.check_call(['python3', '-m', 'pip', 'install', 'scikit-learn==0.20.1'])
        from sklearn import ensemble
        from sklearn import datasets
        from sklearn.utils import shuffle
        from sklearn.metrics import mean_squared_error, explained_variance_score

except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'scikit-learn==0.20.1'])

# Custum build functions
from . import image_procesing_functions

#Codigo para cargar lista de puntos

def getFeatureVector(fileName):
    """Build a feature vector dictionary for given image """
    im = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
    nim = cv2.normalize(im, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    grayim = cv2.cvtColor(nim,cv2.COLOR_BGR2GRAY)  # Pasamos a escala de grises ya qu nos intersan las texturas y objetos no colores
    fltim = grayim.flatten()  # Convertimos la imagenes de matriz a una serie

    # Building descriptors dictionary
    dic = {}
    # Basic statistical descriptors.
    dic['imgSize'] = os.path.getsize(fileName)
    basics = image_procesing_functions.getBasicFeatures(nim,fltim )
    dic = {**dic, **basics}

    # Border descriptors
    bordes = image_procesing_functions.getBorderFeatures(im,nim)
    dic = {**dic, **bordes}

    #Color histogram descriptors
    colorHistogram = image_procesing_functions.get_color_histogram(nim)
    dic = {**dic, **colorHistogram}

    #FFT descriptors

    fftDescriptors = image_procesing_functions.get_ftt_descriptors(fltim)
    dic = {**dic, **fftDescriptors}

    #Texture descriptors Local Binary pattern (LBP)

    lbpDescriptors = image_procesing_functions.get_lbp_descriptors(im)
    dic = {**dic, **lbpDescriptors}

    return dic

def getCoupleVectors(self,umbralTamano):

    features = self.inVector.getFeatures()

    self.inVector.startEditing()
    if self.inVector.dataProvider().fieldNameIndex("descriptorsB") == -1:
        self.inVector.dataProvider().addAttributes([QgsField("descriptorsB", QVariant.String)])
        self.inVector.dataProvider().addAttributes([QgsField("descriptorsA", QVariant.String)])
    self.inVector.updateFields()

    id_new_col_descriptorsB = self.inVector.dataProvider().fieldNameIndex("descriptorsB")
    id_new_col_descriptorsA = self.inVector.dataProvider().fieldNameIndex("descriptorsA")

    total=0
    existed=0
    for feature in features:
        total = total + 1
        pathB = self.outRaster + '/' + feature['tile_B'] + ".png"
        pathA = self.outRaster + '/' + feature['tile_A'] + ".png"
        self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsB, 'A')
        self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsA, 'A')

        if os.path.exists(pathB) and os.path.exists(pathA):  # Check if images exist
            existed = existed + 1
            b = os.path.getsize(pathB)
            a = os.path.getsize(pathA)

            self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsB, 'B')
            self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsA, 'B')

            if b > umbralTamano and a > umbralTamano:

                dicB = getFeatureVector(pathB)
                dicA = getFeatureVector(pathA)
                #dicB0 = dicB
                #dicA0 = dicA


                Vb = round(aply_regresion_model(dicB, self.path2model_visibilidad), 0)
                dicB['visibilidad'] = Vb
                Bb = round(aply_regresion_model(dicB, self.path2model_boscosidad), 0)
                dicB['boscosidad'] = Bb

                Vb = round(aply_regresion_model(dicA, self.path2model_visibilidad), 0)
                dicA['visibilidad'] = Vb
                Bb = round(aply_regresion_model(dicA, self.path2model_boscosidad), 0)
                dicA['boscosidad'] = Bb

                self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsB, json.dumps(str(dicB)))
                self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsA, json.dumps(str(dicA)))
    self.inVector.commitChanges()

# Funcion para asignar punta de calidad (visibilidad) de 0 a 100%.
def aply_regresion_model(dic_image, model):
    """
    dic_imagen= diccionario con los descriptorsde la imagen
    modelo: modleo sklearn persistido como objeto python3 con la libreria joblib
    """
    diclist = [dic_image]
    dftemp = pd.DataFrame(diclist)
    vector_image = dftemp.values  # dftemp[[]].values

    with open(model, 'rb') as fo:
        clf = load(model)
        y = clf.predict(vector_image)
    return y[0]
