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

try:
    from pygeotile.tile import Tile
except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'pyGeoTile'])

# Custum build functions
from . import  tile_proces_functions

#Codigo para cargar lista de puntos



def identifyDeforest(self, umbralTamano, umbralVisibilidad):
    """ clasify the scene of Tile containing point as deforested no deforested or no info
    """
    #-------------Creatin a vector layer for Tile poligons-----------------------------------

    name = 'poly'
    tile_proces_functions.createVectorLayer(self, name)

    #----------------Get list of pints------------------------------------------------------

    features = self.inVector.getFeatures()

    #-------------Define Attributes to add to points layer-----------------------------------

    self.inVector.startEditing()
    if self.inVector.dataProvider().fieldNameIndex("descriptorsB") == -1:
        self.inVector.dataProvider().addAttributes([QgsField("descriptorsB", QVariant.String)])
        self.inVector.dataProvider().addAttributes([QgsField("descriptorsA", QVariant.String)])
        self.inVector.dataProvider().addAttributes([QgsField("deforest", QVariant.Int)])
        self.inVector.dataProvider().addAttributes([QgsField("forestLost", QVariant.Int)])
    self.inVector.updateFields()

    id_new_col_descriptorsB = self.inVector.dataProvider().fieldNameIndex("descriptorsB")
    id_new_col_descriptorsA = self.inVector.dataProvider().fieldNameIndex("descriptorsA")
    id_new_col_descriptorsD = self.inVector.dataProvider().fieldNameIndex("deforest")
    id_new_col_descriptorsL = self.inVector.dataProvider().fieldNameIndex("forestLost")

    #--------------------Inicialice counters an varaibles to use in loop-----------------------
    total=0  # Counter for total numbre of points
    existed=0 #Counter for points with info
    goodones=0 #Counter for points with good images
    no_info=0  #Conter for points with no image or bad images (cloudy)
    defores=0  #Counter for scenes laveled as deforested
    no_deforest=0 #Conter for scenes laveld as no deforested
    EstimatedDeforestation=0 #Totalicer for deforested area estimated with change in forest persentage
    #---------------------------Loop over points in point layer-----------------------------------
    for feature in features:
        """ for every point get descriptors and aply models to clasify the scene as deforested, no deforeste or without info (cloudy)"""
        tempEstimatedDeforestation=0
        total = total + 1
        #----------------Set path for images of Tile B (Before) a(Aferter)----------------------------------------
        pathB = self.outRaster + '/' + feature['tile_B'] + ".png"
        pathA = self.outRaster + '/' + feature['tile_A'] + ".png"
        #----------------Inicialice atrtibute values with default values----------------------------------------
        self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsB, 'A')
        self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsA, 'A')
        forestChange = -3
        self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsD, forestChange)
        self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsL, 0)
        #-------------------If it's not an image for that point is calsified as no info forestChange <0----------
        if os.path.exists(pathB) and os.path.exists(pathA):  # Check if images exist
            #---------If image file size is small it's a nul (no info)--------------------------------------------
            b = os.path.getsize(pathB)
            a = os.path.getsize(pathA)
            self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsB, 'B')
            self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsA, 'B')
            forestChange=-2
            self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsD, forestChange)
            if b > 2600 and a > 2600:
                existed = existed + 1
            if b > umbralTamano and a > umbralTamano:
                #------Now descriptors are calculated for image that pass the threshold size----------------

                dicB = tile_proces_functions.getFeatureVector(pathB)
                dicA = tile_proces_functions.getFeatureVector(pathA)
                #dicB0 = dicB
                #dicA0 = dicA

                #-------Whit descriptors ready, models area aplyed------------------------------------------
                #-----------Firs Visibility (V) and Fores percentage (B)----------------------------------------------------------------
                Vb = round(tile_proces_functions.aply_regresion_model(dicB, self.path2model_visibilidad), 0)

                Bb = round(tile_proces_functions.aply_foresty_model(dicB, self.path2model_boscosidad), 0)
                dicB['boscosidad'] = Bb
                dicB['visibilidad'] = Vb

                Va = round(tile_proces_functions.aply_regresion_model(dicA, self.path2model_visibilidad), 0)

                Ba = round(tile_proces_functions.aply_foresty_model(dicA, self.path2model_boscosidad), 0)
                dicA['boscosidad'] = Ba
                dicA['visibilidad'] = Va

                self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsB, 'C')
                self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsA, 'C')
                forestChange=-1
                self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsD, forestChange)
                #--------If visibility is grater than threshol (umbralVisibilidad) change area evaluated---------
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

                    forestChange = tile_proces_functions.aply_clasification_model(dic, self.path2model_deforest)
                    tempEstimatedDeforestation = dicB['boscosidad']-dicA['boscosidad']
                    if abs(tempEstimatedDeforestation)<2:
                        tempEstimatedDeforestation = 0
                    if tempEstimatedDeforestation < 0:
                        tempEstimatedDeforestation = 0

                    self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsL, int(tempEstimatedDeforestation))

                    B = 'Bosocosidad: ' + str(dicB['boscosidad']) + ', Visibilidad: ' + str(dicB['visibilidad'])
                    A = 'Bosocosidad: ' + str(dicA['boscosidad']) + ', Visibilidad: ' + str(dicA['visibilidad'])
                    self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsB, B)
                    self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsA,
                                                       A)  # json.dumps(str(dicA['boscosidad'])))
                    self.inVector.changeAttributeValue(feature.id(), id_new_col_descriptorsD, int(forestChange))

        #----------Add polygons to polygon layer--------------------------------------------------------------
        defva=-1
        if forestChange > 0:
            defva=1
            if tempEstimatedDeforestation > 0:
                defva = int(tempEstimatedDeforestation)
                if tempEstimatedDeforestation > 190:
                    defva = 190
        else:
            defva = int(forestChange)

        fileNameA = feature['tile_A'] + ".png"
        corners=getCorners(fileNameA)
        tile_proces_functions.addPolygon(self, name, corners, defva)

        #----------increment counters---------------------------------------------------------------------------
        if forestChange <0:
            no_info = no_info +1
        elif forestChange == 1:
            defores = defores+1
        elif forestChange == 0:
            no_deforest = no_deforest+1

        EstimatedDeforestation = EstimatedDeforestation + tempEstimatedDeforestation # +/- 20%


    #--------------------------------------------------------------------------------------------------------

    # ----------Making resume json file-------------------------------------------------------------
    resume = {'Total_number_of_points': total, 'Points_with_info': existed, 'Points_with_aceptable_images':goodones,
                 'Points_not_posibel_evaluation': no_info, 'Points_with_deforestation': defores,
                 'Points_without_deforetation': no_deforest, 'EstimatedDeforestation':int(EstimatedDeforestation*1.44)}


    with open(self.outRaster+ '/' + 'resume.json', 'w') as json_file:
        json.dump(resume, json_file)

    # Now the color of points are changed by his deforest lavel, gree for no deforest, yellow for no info and red for defores

    layer = self.inVector
    values = (
            ('no_info', -10, -1, 'blue'),
            ('doubtful', 0, 0, 'green'),
            ('def_confirmed', 1, 1000, 'red'),
        )
    expression = 'deforest'  # field name

    tile_proces_functions.setLayerRangeColor(self, layer, values, expression)

    #-- Now the tile polygons color are cahnged ------------------------------------------------------------------

    layers = [layer for layer in QgsProject.instance().mapLayers().values()]
    for layer in layers:
        if layer.type() == QgsMapLayer.VectorLayer:
            if layer.name() == name:
                vl = layer
    values = (
        ('no_info', -10, -1, 'blue'),
        ('doubtful', 0, 0, 'green'),
        ('def_confirmed', 1, 4, 'yellow'),
        ('def_confirmed', 5, 10, 'orange'),
        ('tooMuch_def', 11, 200, 'red')
    )

    tile_proces_functions.setLayerRangeColor(self, vl, values, expression)

    self.inVector.commitChanges()

def getCorners(fileNameA):
    zoom = 15
    pixels = 256
    epsg = 4326
    xA = str(fileNameA).split("_")
    google_xA, google_yA = int(xA[6]), int(xA[7].split(".")[0])
    tile = Tile.from_google(google_xA, google_yA, zoom)
    bounds = tile.bounds  # Retorna el punto sur occidental y el punto nororiental del tile.
    p1 = bounds[0]  # Point(latitude=2.6357885741666065, longitude=-72.388916015625)
    p3 = bounds[1]
    p2 = (p3[0], p1[1])
    p4 = (p1[0], p3[1])

    corners = [QgsPointXY(p1[1], p1[0]), QgsPointXY(p2[1], p2[0]), QgsPointXY(p3[1], p3[0]), QgsPointXY(p4[1], p4[0])]  # Tile Carners
    return corners
