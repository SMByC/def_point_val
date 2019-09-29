# -*- coding: utf-8 -*-
# OS and file sistem Libraries
import subprocess
import os
import json
from qgis.PyQt.QtCore import QVariant
from qgis.core import *

# Images and computer vision Libraries
import cv2

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


        if os.path.exists(pathB) and os.path.exists(pathA):  # Check if iamges exist
            existed = existed + 1
            b = os.path.getsize(pathB)
            a = os.path.getsize(pathA)
            if b > umbralTamano and a > umbralTamano:
                dicB = getFeatureVector(pathB)
                self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsB, json.dumps(str(dicB)))
                dicA = getFeatureVector(pathA)
                self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsA, json.dumps(str(dicA)))