# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 08:45:40 2019

@author: Becerra Gustavo, Castro María, Santacruz Ricardo, Torres Harry
"""

import sys

sys.setrecursionlimit(1000000)
import pandas as pd

import cv2
import cv2 as cv
from imageio import imread
from scipy.linalg import norm
from scipy import sum


##Librerías para graficar
import matplotlib

import numpy as np

from scipy.stats import kurtosis
from scipy.stats import skew
from itertools import islice
import math

from sklearn.externals import joblib




#############################################################
###Funciones para obtener bordes, lineas y colores
###Estas funciones se utlizarán para generar los descriptores
##############################################################

#################################
##Funcion window (es igual a la que está en el notebook)
#################################


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


######################################################
##Funcion count maxima (es igual a la que está en el notebook)
#######################################################

def count_maxima(l):
    # b, the middle value, is larger than both a and c
    return sum(a < b > c for a, b, c in window(l, 3))


###############################################################################
##Función para remover transparencia  (es igual a la que está en el notebook)
###############################################################################

def remove_transparency(source, background_color):
    source_img = cv2.cvtColor(source[:, :, :3], cv2.COLOR_BGR2GRAY)
    source_mask = source[:, :, 3] * (1 / 255.0)

    background_mask = 1.0 - source_mask

    bg_part = (background_color * (1 / 255.0)) * (background_mask)
    source_part = (source_img * (1 / 255.0)) * (source_mask)

    return np.uint8(cv2.addWeighted(bg_part, 255.0, source_part, 255.0, 0.0))


##################################################################################
##Función para obtener bordes  (hice un par de modificaciones sobre esta función)
###################################################################################

def getBorderFeatures(img):
    """
    genear descriptores basados en bordes
    No de lineas rectas, cantidad de bordes
    img es una imagen png 256x256 cargada con cv2
    """
    # primero para la imagen si reducir

    imgColor = cv2.normalize(img, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img2 = remove_transparency(imgColor, 0)

    # Se detectan bordes usando Canny
    img_edged = cv2.Canny(img2, 100, 200)

    bordes = img_edged.sum()
    # Se cuentan las linea rectas usando HoughLines
    lines = cv2.HoughLines(img_edged, 1, np.pi / 180,
                           64)  # nomrmalmente deben se entre la curata y la midad de loz pixeles W.
    try:
        if lines.any():
            lines = len(lines)
        else:
            lines = 0

    except:
        lines = 0

    # Luego para la imagen encogida

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    imgColor = cv2.normalize(img, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    imgGray = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
    imgGray = imgGray.astype('uint8')

    std = imgGray.std(axis=0).std(axis=0)
    media = imgGray.mean(axis=0).std(axis=0)
    ventana = int(math.sqrt(imgGray.size) / 4 + 1)

    # Se detectan los bordes usando Canny
    img2 = remove_transparency(imgColor, 0)
    img_edged = cv2.Canny(img2, 100, 200)

    bordes32 = img_edged.sum()

    # Se binariza la imagen usando umbral adaptativo queda ub "borde de la imagen con una textura simplificada"
    umbral = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ventana, 2 * std)

    # Binarizacion de la imagen usando Otsu
    blur = cv2.GaussianBlur(imgGray, (3, 3), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detectar y contar lineas rectas
    lines32 = cv2.HoughLines(img_edged, 1, np.pi / 180, 16)
    try:
        if lines32.any():
            lines32 = len(lines32)
        else:
            lines32 = 0

    except:
        lines32 = 0

    simetriaImagenBinarizada = skew(th3).sum()
    curtosisImagenBinarizada = kurtosis(th3).sum()
    sumaUmbral = umbral.sum()

    return {'lines': lines, 'bordes': bordes, 'lines32': lines32,
            'bordes32': bordes32, 'simetriaImagenBinarizada': simetriaImagenBinarizada,
            'curtosisImagenBinarizada': curtosisImagenBinarizada, 'sumaUmbral': sumaUmbral}


###################################################################################
##Función para obtener colores (hice un par de modificaciones sobre esta función
####################################################################################

def get_color(img):
    nim = cv2.normalize(img, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    grayim = cv2.cvtColor(nim,
                          cv2.COLOR_BGR2GRAY)  # Pasamos a escala de grises ya qu nos intersan las texturas y objetos no colores
    fltim = grayim.flatten()  # Convertimos la imagenes de matriz a una serie
    hist_grayim = np.histogram(fltim, bins=32)[0]
    average = nim.mean(axis=0).mean(axis=0)
    skewness = skew(fltim)
    curtosis = kurtosis(fltim)
    avWhite = average[0] + average[1] + average[2]
    skewWhite = skewness
    curtWhite = curtosis
    num_piks_White = count_maxima(hist_grayim)
    avRed = average[0]
    avGreen = average[1]
    avBlue = average[2]
    std = nim.std(axis=0).std(axis=0)
    stdWhite = std[0] + std[1] + std[2]
    stdRed = std[0]
    stdGreen = std[1]
    stdBlue = std[2]
    minimo = nim.min(axis=0).min(axis=0)
    minWhite = minimo[0] + minimo[1] + minimo[2]
    minRed = minimo[0]
    minGreen = minimo[1]
    minBlue = minimo[2]
    maximo = nim.max(axis=0).max(axis=0)
    maxWhite = maximo[0] + maximo[1] + maximo[2]
    maxRed = maximo[0]
    maxGreen = maximo[1]
    maxBlue = maximo[2]
    return {'avWhite': avWhite,
            'skewWhite': skewWhite,
            'curtWhite': curtWhite,
            'num_piks_White': num_piks_White,
            'avRed': avRed,
            'avGreen': avGreen,
            'avBlue': avBlue,
            'stdWhite': stdWhite,
            'stdRed': stdRed,
            'stdGreen': stdGreen,
            'stdBlue': stdBlue,
            'minWhite': minWhite,
            'minRed': minRed,
            'minGreen': minGreen,
            'minBlue': minBlue,
            'maxWhite': maxWhite,
            'maxRed': maxRed,
            'maxGreen': maxGreen,
            'maxBlue': maxBlue}


##############################################################d##################
##Función para obtener diferencias de bordes, lineas, colores, ORB y distancias,
##Esta función genera los descriptores necesarios para el modelo
################################################################################

def get_diff(img1, img2):
    dic_color_img1 = get_color(img1)
    dic_color_img2 = get_color(img2)
    diff_avWhite = abs(dic_color_img1['avWhite'] - dic_color_img2['avWhite'])
    diff_skewWhite = abs(dic_color_img1['skewWhite'] - dic_color_img2['skewWhite'])
    diff_curtWhite = abs(dic_color_img1['curtWhite'] - dic_color_img2['curtWhite'])
    diff_num_piks_White = abs(dic_color_img1['num_piks_White'] - dic_color_img2['num_piks_White'])
    diff_avRed = abs(dic_color_img1['avRed'] - dic_color_img2['avRed'])
    diff_avGreen = abs(dic_color_img1['avGreen'] - dic_color_img2['avGreen'])
    diff_avBlue = abs(dic_color_img1['avBlue'] - dic_color_img2['avBlue'])
    diff_stdWhite = abs(dic_color_img1['stdWhite'] - dic_color_img2['stdWhite'])
    diff_stdRed = abs(dic_color_img1['stdRed'] - dic_color_img2['stdRed'])
    diff_stdGreen = abs(dic_color_img1['stdGreen'] - dic_color_img2['stdGreen'])
    diff_stdBlue = abs(dic_color_img1['stdBlue'] - dic_color_img2['stdBlue'])
    diff_minWhite = abs(dic_color_img1['minWhite'] - dic_color_img2['minWhite'])
    diff_minRed = abs(dic_color_img1['minRed'] - dic_color_img2['minRed'])
    diff_minGreen = abs(dic_color_img1['minGreen'] - dic_color_img2['minGreen'])
    diff_minBlue = abs(dic_color_img1['minBlue'] - dic_color_img2['minBlue'])
    diff_maxWhite = abs(dic_color_img1['maxWhite'] - dic_color_img2['maxWhite'])
    diff_maxRed = abs(dic_color_img1['maxRed'] - dic_color_img2['maxRed'])
    diff_maxGreen = abs(dic_color_img1['maxGreen'] - dic_color_img2['maxGreen'])
    diff_maxBlue = abs(dic_color_img1['maxBlue'] - dic_color_img2['maxBlue'])
    dic_bordes_img1 = getBorderFeatures(img1)
    dic_bordes_img2 = getBorderFeatures(img2)
    diff_lines = abs(dic_bordes_img1['lines'] - dic_bordes_img2['lines'])
    diff_bordes = abs(dic_bordes_img1['bordes'] - dic_bordes_img2['bordes'])
    diff_lines32 = abs(dic_bordes_img1['lines32'] - dic_bordes_img2['lines32'])
    diff_bordes32 = abs(dic_bordes_img1['bordes32'] - dic_bordes_img2['bordes32'])
    diff_simetriaImagenBinarizada = abs(
        dic_bordes_img1['simetriaImagenBinarizada'] - dic_bordes_img2['simetriaImagenBinarizada'])
    diff_curtosisImagenBinarizada = abs(
        dic_bordes_img1['curtosisImagenBinarizada'] - dic_bordes_img2['curtosisImagenBinarizada'])
    diff_sumaUmbral = abs(dic_bordes_img1['sumaUmbral'] - dic_bordes_img2['sumaUmbral'])
    ###Para detectar diferencias
    diff = img2 - img1
    ##Para detectar matches
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    ##longitud de match orb destriptors
    longitud_matches = len(matches)
    ##Distancia de Manhathan
    dif_manhathan = sum(abs(diff))  # Manhattan norm
    ##Norma cero del vector
    dif_z_norm = norm(diff.ravel(), 0)  # Zero norm
    return {"diff_avWhite": diff_avWhite,
            "diff_skewWhite": diff_skewWhite,
            "diff_curtWhite": diff_curtWhite,
            "diff_num_piks_White": diff_num_piks_White,
            "diff_avRed": diff_avRed,
            "diff_avGreen": diff_avGreen,
            "diff_avBlue": diff_avBlue,
            "diff_stdWhite": diff_stdWhite,
            "diff_stdRed": diff_stdRed,
            "diff_stdGreen": diff_stdGreen,
            "diff_stdBlue": diff_stdBlue,
            "diff_minWhite": diff_minWhite,
            "diff_minRed": diff_minRed,
            "diff_minGreen": diff_minGreen,
            "diff_minBlue": diff_minBlue,
            "diff_maxWhite": diff_maxWhite,
            "diff_maxRed": diff_maxRed,
            "diff_maxGreen": diff_maxGreen,
            "diff_maxBlue": diff_maxBlue,
            "diff_lines": diff_lines,
            "diff_bordes": diff_bordes,
            "diff_lines32": diff_lines32,
            "diff_bordes32": diff_bordes32,
            "diff_simetriaImagenBinarizada": diff_simetriaImagenBinarizada,
            "diff_curtosisImagenBinarizada": diff_curtosisImagenBinarizada,
            "diff_sumaUmbral": diff_sumaUmbral,
            "longitud_matches": longitud_matches,
            "dif_manhathan": dif_manhathan,
            "dif_z_norm": dif_z_norm}


#################################################################
##Función negative, para obtener el "negativo" de un número
#################################################################

def negative(vector):
    for i in vector:
        vector[i] = vector[i] * (-1)
    return (vector)


#############################################################################################################
##Función para aplicar el modelo, los parámetros son las rutas de la imágen de antes y la imagen del después
##############################################################################################################


def aplicar_modelo(self,path_before_img, path_after_img):
    """
    Esta función se utiliza para aplicar el modelo gradient boosting machine para clasificar
    si un par de imágenes presentan cambios o no.
    Los parámetros son:
        path_before_img: Ruta de la imágen de un área x en un tiempo t-x
        path_afer_img: Ruta de la imágen de un área x en un tiempo t
    Resultado de la función: 1 si presenta cambios y 0 en caso contrario.
    """
    vector = list(range(0, 1))
    ##vectores donde se almacenaran las diferencias de los descriptores
    vecttor_diff_avWhite_sd = list(range(0, 1))
    vecttor_diff_skewWhite_sd = list(range(0, 1))
    vecttor_diff_curtWhite_sd = list(range(0, 1))
    vecttor_diff_num_piks_White_sd = list(range(0, 1))
    vecttor_diff_avRed_sd = list(range(0, 1))
    vecttor_diff_avGreen_sd = list(range(0, 1))
    vecttor_diff_avBlue_sd = list(range(0, 1))
    vecttor_diff_stdWhite_sd = list(range(0, 1))
    vecttor_diff_stdRed_sd = list(range(0, 1))
    vecttor_diff_stdGreen_sd = list(range(0, 1))
    vecttor_diff_stdBlue_sd = list(range(0, 1))
    vecttor_diff_minWhite_sd = list(range(0, 1))
    vecttor_diff_minRed_sd = list(range(0, 1))
    vecttor_diff_minGreen_sd = list(range(0, 1))
    vecttor_diff_minBlue_sd = list(range(0, 1))
    vecttor_diff_maxWhite_sd = list(range(0, 1))
    vecttor_diff_maxRed_sd = list(range(0, 1))
    vecttor_diff_maxGreen_sd = list(range(0, 1))
    vecttor_diff_maxBlue_sd = list(range(0, 1))
    vecttor_diff_lines_sd = list(range(0, 1))
    vecttor_diff_bordes_sd = list(range(0, 1))
    vecttor_diff_lines32_sd = list(range(0, 1))
    vecttor_diff_bordes32_sd = list(range(0, 1))
    vecttor_diff_simetriaImagenBinarizada_sd = list(range(0, 1))
    vecttor_diff_curtosisImagenBinarizada_sd = list(range(0, 1))
    vecttor_diff_sumaUmbral_sd = list(range(0, 1))
    vecttor_longitud_matches_sd = list(range(0, 1))
    vecttor_dif_manhathan_sd = list(range(0, 1))
    vecttor_dif_z_norm_sd = list(range(0, 1))
    vector_nomb_imagenes_sd = list(range(0, 1))
    ##Para obtener las diferencias de los descriptores y almacenarla.
    for i in vector:
        ##Vector que contiene el nombre de las imágenes
        ruta_imagen_0_carpeta_k = path_before_img
        ruta_imagen_1_carpeta_k = path_after_img
        try:
            img_0 = imread(ruta_imagen_0_carpeta_k)
            img_1 = imread(ruta_imagen_1_carpeta_k)  ###Considerar si se hace una normalizacion
            ##Aplicar la función de obtener diferencias entre las dos imágenes
            diff = get_diff(img_0, img_1)
            ##Almacenar en cada uno de los vectores
            vecttor_diff_avWhite_sd[i] = diff["diff_avWhite"]
            vecttor_diff_skewWhite_sd[i] = diff["diff_skewWhite"]
            vecttor_diff_curtWhite_sd[i] = diff["diff_curtWhite"]
            vecttor_diff_num_piks_White_sd[i] = diff["diff_num_piks_White"]
            vecttor_diff_avRed_sd[i] = diff["diff_avRed"]
            vecttor_diff_avGreen_sd[i] = diff["diff_avGreen"]
            vecttor_diff_avBlue_sd[i] = diff["diff_avBlue"]
            vecttor_diff_stdWhite_sd[i] = diff["diff_stdWhite"]
            vecttor_diff_stdRed_sd[i] = diff["diff_stdRed"]
            vecttor_diff_stdGreen_sd[i] = diff["diff_stdGreen"]
            vecttor_diff_stdBlue_sd[i] = diff["diff_stdBlue"]
            vecttor_diff_minWhite_sd[i] = diff["diff_minWhite"]
            vecttor_diff_minRed_sd[i] = diff["diff_minRed"]
            vecttor_diff_minGreen_sd[i] = diff["diff_minGreen"]
            vecttor_diff_minBlue_sd[i] = diff["diff_minBlue"]
            vecttor_diff_maxWhite_sd[i] = diff["diff_maxWhite"]
            vecttor_diff_maxRed_sd[i] = diff["diff_maxRed"]
            vecttor_diff_maxGreen_sd[i] = diff["diff_maxGreen"]
            vecttor_diff_maxBlue_sd[i] = diff["diff_maxBlue"]
            vecttor_diff_lines_sd[i] = diff["diff_lines"]
            vecttor_diff_bordes_sd[i] = diff["diff_bordes"]
            vecttor_diff_lines32_sd[i] = diff["diff_lines32"]
            vecttor_diff_bordes32_sd[i] = diff["diff_bordes32"]
            vecttor_diff_simetriaImagenBinarizada_sd[i] = diff["diff_simetriaImagenBinarizada"]
            vecttor_diff_curtosisImagenBinarizada_sd[i] = diff["diff_curtosisImagenBinarizada"]
            vecttor_diff_sumaUmbral_sd[i] = diff["diff_sumaUmbral"]
            vecttor_longitud_matches_sd[i] = diff["longitud_matches"]
            vecttor_dif_manhathan_sd[i] = diff["dif_manhathan"]
            vecttor_dif_z_norm_sd[i] = diff["dif_z_norm"]
        except Exception:
            pass
    ##Data frame para el modelo
    data_frame_si_deforestacion = pd.DataFrame(vecttor_diff_avWhite_sd, columns=['diff_avWhite'])
    data_frame_si_deforestacion["diff_avWhite"] = vecttor_diff_skewWhite_sd
    data_frame_si_deforestacion["diff_curtWhite"] = vecttor_diff_curtWhite_sd
    data_frame_si_deforestacion["diff_num_piks_White"] = vecttor_diff_num_piks_White_sd
    data_frame_si_deforestacion["diff_avRed"] = vecttor_diff_avRed_sd
    data_frame_si_deforestacion["diff_avGreen"] = vecttor_diff_avGreen_sd
    data_frame_si_deforestacion["diff_avBlue"] = vecttor_diff_avBlue_sd
    data_frame_si_deforestacion["diff_stdWhite"] = vecttor_diff_stdWhite_sd
    data_frame_si_deforestacion["diff_stdRed"] = vecttor_diff_stdRed_sd
    data_frame_si_deforestacion["diff_stdGreen"] = vecttor_diff_stdGreen_sd
    data_frame_si_deforestacion["diff_stdBlue"] = vecttor_diff_stdBlue_sd
    data_frame_si_deforestacion["diff_minWhite"] = vecttor_diff_minWhite_sd
    data_frame_si_deforestacion["diff_minRed"] = vecttor_diff_minRed_sd
    data_frame_si_deforestacion["diff_minGreen"] = vecttor_diff_minGreen_sd
    data_frame_si_deforestacion["diff_minBlue"] = vecttor_diff_minBlue_sd
    data_frame_si_deforestacion["diff_maxWhite"] = vecttor_diff_maxWhite_sd
    data_frame_si_deforestacion["diff_maxRed"] = vecttor_diff_maxRed_sd
    data_frame_si_deforestacion["diff_maxGreen"] = vecttor_diff_maxGreen_sd
    data_frame_si_deforestacion["diff_maxBlue"] = vecttor_diff_maxBlue_sd
    data_frame_si_deforestacion["diff_lines"] = vecttor_diff_lines_sd
    data_frame_si_deforestacion["diff_bordes"] = vecttor_diff_bordes_sd
    data_frame_si_deforestacion["diff_lines32"] = vecttor_diff_lines32_sd
    data_frame_si_deforestacion["diff_bordes32"] = vecttor_diff_bordes32_sd
    data_frame_si_deforestacion["diff_simetriaImagenBinarizada"] = vecttor_diff_simetriaImagenBinarizada_sd
    data_frame_si_deforestacion["diff_curtosisImagenBinarizada"] = vecttor_diff_curtosisImagenBinarizada_sd
    data_frame_si_deforestacion["diff_sumaUmbral"] = vecttor_diff_sumaUmbral_sd
    data_frame_si_deforestacion["longitud_matches"] = vecttor_longitud_matches_sd
    data_frame_si_deforestacion["dif_manhathan"] = vecttor_dif_manhathan_sd
    data_frame_si_deforestacion["dif_z_norm"] = vecttor_dif_z_norm_sd
    data_frame_si_deforestacion['y'] = 1  ###Asignamos y = 1, pero esto no se utilizará
    data_frame_si_deforestacion['name'] = "xxxx"  ##Asignamos name xxxxx pero no se utilizará
    ##Dataframe definitivo
    data_frame2 = data_frame_si_deforestacion
    ##Para definir la matriz X de variales explicativas y el vector de variable respuesta
    X_test, Y_test = data_frame2.iloc[:, 0:28], data_frame2.iloc[:, 28:29]
    ##Directorio donde está el modelo, importante!!!!!!!!

    ##Nombre del modelo
    filename = self.path2model_harry
    ##Para cargar el modelo
    alg = joblib.load(filename)
    ##Para aplicar el modelo
    predictions = alg.predict(X_test)
    pred_proba = alg.predict_proba(X_test)[:, 1]
    clasificacion = predictions[0]
    return (clasificacion)




