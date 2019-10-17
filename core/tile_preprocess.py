# -*- coding: utf-8 -*-
# OS and file sistem Libraries
import subprocess
import os
import json
from qgis.PyQt.QtCore import QVariant
from qgis.core import *
from qgis.PyQt.QtCore import *
from qgis.PyQt.QtGui import *


# Images and computer vision Libraries
try:
    import cv2
except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'opencv-python'])
#Math ans statistics
try:
    import numpy as np
except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'numpy'])
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

    #Geometric feature descriptos (croners) ORB
    orbDescriptors = image_procesing_functions.get_orb_descriptors(im)
    dic = {**dic, **orbDescriptors}
    return dic

def getCoupleVectors(self, umbralTamano, umbralVisibilidad):
    """get descriptor vector for befor and after images and for change"""


    features = self.inVector.getFeatures()

    self.inVector.startEditing()
    if self.inVector.dataProvider().fieldNameIndex("descriptorsB") == -1:
        self.inVector.dataProvider().addAttributes([QgsField("descriptorsB", QVariant.String)])
        self.inVector.dataProvider().addAttributes([QgsField("descriptorsA", QVariant.String)])
        self.inVector.dataProvider().addAttributes([QgsField("deforest", QVariant.Int)])
    self.inVector.updateFields()

    id_new_col_descriptorsB = self.inVector.dataProvider().fieldNameIndex("descriptorsB")
    id_new_col_descriptorsA = self.inVector.dataProvider().fieldNameIndex("descriptorsA")
    id_new_col_descriptorsD = self.inVector.dataProvider().fieldNameIndex("deforest")

    total=0
    existed=0
    goodones=0
    no_info=0
    defores=0
    no_deforest=0
    for feature in features:
        total = total + 1
        pathB = self.outRaster + '/' + feature['tile_B'] + ".png"
        pathA = self.outRaster + '/' + feature['tile_A'] + ".png"
        self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsB, 'A')
        self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsA, 'A')
        forestChange = -3
        self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsD, forestChange)

        if os.path.exists(pathB) and os.path.exists(pathA):  # Check if images exist

            b = os.path.getsize(pathB)
            a = os.path.getsize(pathA)
            self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsB, 'B')
            self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsA, 'B')
            forestChange=-2
            self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsD, forestChange)
            if b > 2600 and a > 2600:
                existed = existed + 1
            if b > umbralTamano and a > umbralTamano:

                dicB = getFeatureVector(pathB)
                dicA = getFeatureVector(pathA)
                #dicB0 = dicB
                #dicA0 = dicA


                Vb = round(aply_regresion_model(dicB, self.path2model_visibilidad), 0)

                Bb = round(aply_regresion_model(dicB, self.path2model_boscosidad), 0)
                dicB['boscosidad'] = Bb
                dicB['visibilidad'] = Vb

                Va = round(aply_regresion_model(dicA, self.path2model_visibilidad), 0)

                Ba = round(aply_regresion_model(dicA, self.path2model_boscosidad), 0)
                dicA['boscosidad'] = Ba
                dicA['visibilidad'] = Va

                self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsB, 'C')
                self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsA, 'C')
                forestChange=-1
                self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsD, forestChange)

                if Va > umbralVisibilidad and Vb > umbralVisibilidad:
                    goodones = goodones+1
                    #Search for orb feature matching
                    try:
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        # Match descriptors.
                        matches = bf.match(dicB['orbDes'], dicA['orbDes'])
                        good = []
                        tempvar = 1
                    except:
                        tempvar = 0
                        matches = []
                        good = []
                    for m in matches:
                        if m.distance < 63:
                            good.append([m.distance])

                    # Optain de diference or change in every descriptor
                    df = pd.DataFrame([dicB, dicA])
                    df = df.drop(columns=['orbKp', 'orbDes'])
                    dif = df.loc[0].values - df.loc[1].values
                    zipbObj = zip(df.columns, dif)
                    dic = dict(zipbObj)
                    dic['orbMatch'] = len(matches)
                    dic['orbMatchThreshold'] = len(good)
                    #Now we have new change descriptor dictionari including orb matching geatures

                    forestChange = aply_clasification_model(dic, self.path2model_deforest)


                    B = 'Bosocosidad: ' + str(dicB['boscosidad']) + ', Visibilidad: ' + str(dicB['visibilidad'])
                    A = 'Bosocosidad: ' + str(dicA['boscosidad']) + ', Visibilidad: ' + str(dicA['visibilidad'])
                    self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsB, B)
                    self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsA,
                                                       A)  # json.dumps(str(dicA['boscosidad'])))
                    self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsD, int(forestChange))
            if forestChange <0:
                no_info = no_info +1
            elif forestChange == 1:
                defores = defores+1
            elif forestChange == 0:
                no_deforest = no_deforest+1
        resume = {'Total_number_of_points': total, 'Points_with_info': existed, 'Points_with_aceptable_images':goodones,
                 'Points_not_posibel_evaluation': no_info, 'Points_with_deforestation': defores,
                 'Points_without_deforetation': no_deforest}

        with open(self.outRaster+ '/' + 'resume.json', 'w') as json_file:
            json.dump(resume, json_file)

    self.inVector.commitChanges()

# Funcion para asignar punta de calidad (visibilidad) de 0 a 100%.
def aply_regresion_model(dic_image, model):
    """
    dic_imagen= diccionario con los descriptorsde la imagen
    modelo: modleo sklearn persistido como objeto python3 con la libreria joblib
    """
    feature_names = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                     '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28',
                     '29', '3', '30', '4', '5', '6', '7', '8', '9', 'avBlue', 'avGreen',
                     'avRed', 'avWhite', 'bordes', 'bordes32', 'curtWhite',
                     'curtosisImagenBinarizada', 'frecbean0pow0', 'frecbean0pow1',
                     'frecbean0pow2', 'frecbean0pow3', 'frecbean1pow0', 'frecbean1pow1',
                     'frecbean1pow2', 'frecbean1pow3', 'frecbean2pow0', 'frecbean2pow1',
                     'frecbean2pow2', 'frecbean2pow3', 'frecbean3pow0', 'frecbean3pow1',
                     'frecbean3pow2', 'frecbean3pow3', 'frecbean4pow0', 'frecbean4pow1',
                     'frecbean4pow2', 'frecbean4pow3', 'frecbean5pow0', 'frecbean5pow1',
                     'frecbean5pow2', 'frecbean5pow3', 'frecbean6pow0', 'frecbean6pow1',
                     'frecbean6pow2', 'frecbean6pow3', 'frecbean7pow0', 'frecbean7pow1',
                     'frecbean7pow2', 'frecbean7pow3', 'imgSize', 'lines', 'lines32',
                     'maxBlue', 'maxGreen', 'maxRed', 'maxWhite', 'minBlue', 'minGreen',
                     'minRed', 'minWhite', 'num_piks_White', 'r0g0b0', 'r0g0b1',
                     'r0g0b2', 'r0g1b0', 'r0g1b1', 'r0g1b2', 'r0g2b0', 'r0g2b1',
                     'r0g2b2', 'r1g0b0', 'r1g0b1', 'r1g0b2', 'r1g1b0', 'r1g1b1',
                     'r1g1b2', 'r1g2b0', 'r1g2b1', 'r1g2b2', 'r2g0b0', 'r2g0b1',
                     'r2g0b2', 'r2g1b0', 'r2g1b1', 'r2g1b2', 'r2g2b0', 'r2g2b1',
                     'r2g2b2', 'simetriaImagenBinarizada', 'skewWhite', 'stdBlue',
                     'stdGreen', 'stdRed', 'stdWhite', 'sumaUmbral', 'textur68_0',
                     'textur68_1', 'textur68_10', 'textur68_11', 'textur68_12',
                     'textur68_13', 'textur68_14', 'textur68_15', 'textur68_16',
                     'textur68_17', 'textur68_2', 'textur68_3', 'textur68_4',
                     'textur68_5', 'textur68_6', 'textur68_7', 'textur68_8',
                     'textur68_9', 'textur6_0', 'textur6_1', 'textur6_2', 'textur6_3',
                     'textur6_4', 'textur6_5', 'textur6_6', 'textur6_7', 'textur6_8']

    diclist = [dic_image]
    dftemp = pd.DataFrame(diclist)
    vector_image = dftemp[feature_names].values#dftemp.values
    with open(model, 'rb') as fo:
        clf = load(model)
        y = clf.predict(vector_image)
    return y[0]


def aply_clasification_model(dic_change, model):
    """
    dic_imagen= diccionario con descriptores de cambio entre las imagenes
    modelo: modleo sklearn persistido como objeto python3 con la libreria joblib
    """
    feature_names = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                    '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28',
                    '29', '3', '30', '4', '5', '6', '7', '8', '9', 'avBlue', 'avGreen',
                    'avRed', 'avWhite', 'bordes', 'bordes32', 'boscosidad',
                    'curtWhite', 'curtosisImagenBinarizada', 'frecbean0pow0',
                    'frecbean0pow1', 'frecbean0pow2', 'frecbean0pow3', 'frecbean1pow0',
                    'frecbean1pow1', 'frecbean1pow2', 'frecbean1pow3', 'frecbean2pow0',
                    'frecbean2pow1', 'frecbean2pow2', 'frecbean2pow3', 'frecbean3pow0',
                    'frecbean3pow1', 'frecbean3pow2', 'frecbean3pow3', 'frecbean4pow0',
                    'frecbean4pow1', 'frecbean4pow2', 'frecbean4pow3', 'frecbean5pow0',
                    'frecbean5pow1', 'frecbean5pow2', 'frecbean5pow3', 'frecbean6pow0',
                    'frecbean6pow1', 'frecbean6pow2', 'frecbean6pow3', 'frecbean7pow0',
                    'frecbean7pow1', 'frecbean7pow2', 'frecbean7pow3', 'imgSize',
                    'lines', 'lines32', 'maxBlue', 'maxGreen', 'maxRed', 'maxWhite',
                    'minBlue', 'minGreen', 'minRed', 'minWhite', 'num_piks_White',
                    'orbMatch', 'orbMatchThreshold', 'r0g0b0', 'r0g0b1', 'r0g0b2',
                    'r0g1b0', 'r0g1b1', 'r0g1b2', 'r0g2b0', 'r0g2b1', 'r0g2b2',
                    'r1g0b0', 'r1g0b1', 'r1g0b2', 'r1g1b0', 'r1g1b1', 'r1g1b2',
                    'r1g2b0', 'r1g2b1', 'r1g2b2', 'r2g0b0', 'r2g0b1', 'r2g0b2',
                    'r2g1b0', 'r2g1b1', 'r2g1b2', 'r2g2b0', 'r2g2b1', 'r2g2b2',
                    'simetriaImagenBinarizada', 'skewWhite', 'stdBlue', 'stdGreen',
                    'stdRed', 'stdWhite', 'sumaUmbral', 'textur68_0', 'textur68_1',
                    'textur68_10', 'textur68_11', 'textur68_12', 'textur68_13',
                    'textur68_14', 'textur68_15', 'textur68_16', 'textur68_17',
                    'textur68_2', 'textur68_3', 'textur68_4', 'textur68_5',
                    'textur68_6', 'textur68_7', 'textur68_8', 'textur68_9',
                    'textur6_0', 'textur6_1', 'textur6_2', 'textur6_3', 'textur6_4',
                    'textur6_5', 'textur6_6', 'textur6_7', 'textur6_8', 'visibilidad']

    diclist = [dic_change]
    dftemp = pd.DataFrame(diclist)
    vector_image = dftemp[feature_names].values  # dftemp.values
    with open(model, 'rb') as fo:
        clf = load(model)
        y = clf.predict(vector_image)
    return y[0] #Retorna 1 si es deforestacion 0 si es no deforestacion


def changePointColor(self):
    layer = self.inVector

    # define ranges: label, lower value, upper value, color name
    # in the field named 'random' (attributes table)
    values = (
        ('no_info', -10, -1, 'yellow'),
        ('doubtful', 0, 0, 'green'),
        ('def_confirmed', 1, 10, 'red'),
    )

    # create a category for each item in values
    ranges = []
    for label, lower, upper, color in values:
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol.setColor(QColor(color))
        rng = QgsRendererRange(lower, upper, symbol, label)
        ranges.append(rng)

    # create the renderer and assign it to a layer
    expression = 'deforest'  # field name
    renderer = QgsGraduatedSymbolRenderer(expression, ranges)
    layer.setRenderer(renderer)


def pngs2geotifs(self):
    zoom = 15
    pixels = 256
    epsg = 4326
     # r=root, d=directories, f = files
    groupName = "PlanetTiles"
    root = QgsProject.instance().layerTreeRoot()
    groupt = root.addGroup(groupName)
    features = self.inVector.getFeatures()

    for feature in features:
        fileNameA = feature['tile_A'] + ".png"
        fileNameB = feature['tile_B'] + ".png"
        raster_layers = [layer.name() for layer in QgsProject.instance().mapLayers().values()]

        if not ( (feature['tile_A'] + ".tif") in raster_layers):
            pathB = os.path.join(self.outRaster, fileNameB)
            pathA = os.path.join(self.outRaster, fileNameA)
            xA = str(fileNameA).split("_")
            google_xA, google_yA = int(xA[6]), int(xA[7].split(".")[0])
            xB = str(fileNameB).split("_")
            google_xB, google_yB = int(xB[6]), int(xB[7].split(".")[0])
            if google_xA == google_xB and google_yA == google_yB:
                group = groupt.addGroup(xA[6] + '_' + xA[7].split(".")[0])

                pathAtif = os.path.join(self.outRaster, feature['tile_A'] + ".tif")
                pathBtif = os.path.join(self.outRaster, feature['tile_b'] + ".tif")

                image_procesing_functions.png2GeoTif(pathB, pathBtif, google_xB, google_yB, zoom, pixels, epsg)
                rasterlyr = QgsRasterLayer(pathBtif, feature['tile_B'] + ".tif")
                QgsProject.instance().addMapLayer(rasterlyr)
                layer = root.findLayer(rasterlyr.id())
                clone = layer.clone()
                group.insertChildNode(0, clone)
                root.removeChildNode(layer)

                image_procesing_functions.png2GeoTif(pathA, pathAtif, google_xA, google_yA, zoom, pixels, epsg)
                rasterlyr = QgsRasterLayer(pathAtif, feature['tile_A'] + ".tif")
                QgsProject.instance().addMapLayer(rasterlyr)
                layer = root.findLayer(rasterlyr.id())
                clone = layer.clone()
                group.insertChildNode(0, clone)
                root.removeChildNode(layer)


        #openRaster(self, dst_filename, pre)




def openRaster(self, path, pre):
    """Open vector from file"""
    if path is not None:
        self.iface.addRasterLayer(path, pre)