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
import subprocess

try:
    from pygeotile.tile import Tile
except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'pyGeoTile'])

from itertools import islice

try:
    import cv2
except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'opencv-python'])

try:
    import numpy as np
except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'numpy'])

try:
    import math
except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'math'])

try:

    from scipy.stats import kurtosis, skew
    from scipy.spatial.distance import cdist
except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'scipy'])

try:
    import pandas as pd
except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'pandas'])

try:
    from skimage.feature import local_binary_pattern

except:
    subprocess.check_call(['python3', '-m', 'pip', 'install', 'scikit-image'])


from osgeo import gdal, osr

#------------------------------------------------------------------------------------------------------------------------

def matrixDistance(img, img2):
    img = cv2.normalize(img, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img2 = cv2.normalize(img2, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    imgDist = cdist(img, img2, 'cosine')
    mt = np.transpose(imgDist)

    imgDist = round(np.sum(np.sum(abs(imgDist - mt), axis=0)) / (256), 3)

    return imgDist


#Codigo para cargar lista de puntos
def loadDefPoints(defPointsGeojsonFile):#Falta especificar fecha o arvhivo en los argumentos.
    gdf=gpd.GeoDataFrame.from_file(defPointsGeojsonFile)
    gdf["x"] = gdf.centroid.y
    gdf["y"] = gdf.centroid.x
    df=gdf[["x","y"]]
    return df


#Codigo para optener el tile correspondiente a una coordenada.
def point2Tile(p,z):
    latitude, longitude, zoom = p[0],p[1], z
    tile = Tile.for_latitude_longitude(latitude, longitude, zoom)
    tile= tile.google
    return tile


def directo(imbefore, imafter, path2model_directo):
    imgt1 = imafter
    imgt0 = imbefore
    normalizedImg = np.zeros((256, 256))
    normalizedImgt1 = cv2.normalize(imgt1, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
    normalizedImgt0 = cv2.normalize(imgt0, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
    img0 = imgt0[:, :, ::-1]
    img1 = imgt1[:, :, ::-1]
    normalizedImgt0 = normalizedImgt0[:, :, ::-1]
    normalizedImgt1 = normalizedImgt1[:, :, ::-1]
    # imgDiff=normalizedImgt1-normalizedImgt0
    dst = cv2.addWeighted(normalizedImgt1, 0.5, normalizedImgt0, -0.5, 0)
    dst = cv2.normalize(dst, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    dst = dst[:, :, ::-1]
    diffImage = dst
    direct = 'si'
    dicD = getFeatureVector(diffImage, 'no_file', direct)
    direct = round(aply_directClasification_model(dicD, path2model_directo), 0)
    return direct


def window(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable

    :param seq: secuentia list or vector

    :param n: windows size

    :return: iterable of windows
    """
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def count_maxima(l):
    """find local maximus in a one dimentional vector

    :param l: one dimentional vector

    :return: the count of local maxima finded
    """
    # b, the middle value, is larger than both a and c
    return sum(a < b > c for a, b, c in window(l, 3))



def remove_transparency(source, background_color):
    """remove 4th layer of .png images

    :param source:

    :param background_color: layer ypu want to remove

    :return: RGB image.
    """
    try:
        source_img = cv2.cvtColor(source[:,:,:3], cv2.COLOR_BGR2GRAY)
        source_mask = source[:,:,3]  * (1 / 255.0)

        background_mask = 1.0 - source_mask

        bg_part = (background_color * (1 / 255.0)) * (background_mask)
        source_part = (source_img * (1 / 255.0)) * (source_mask)
        r=np.uint8(cv2.addWeighted(bg_part, 255.0, source_part, 255.0, 0.0))
    except:
        r=source

    return r


def getBorderFeatures(img, nim):
    """genear descriptores basados en bordes No de lineas rectas, cantidad de bordes img es una imagen png 256x256 cargada con cv2

    :param img: 256x256 RGB intensity normalice image readed with OCV

    :param nim: image flattern in a one dimention vecto

    :return: dictionary with some border features: number of lines, numbre of border pixels in image
    """
    # primero para la imagen si reducir

    imgColor = nim
    img2 = remove_transparency(imgColor, 0)

    # Se detectan bordes usando Canny
    img2 = img2.astype(np.uint8)
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
    imgColor = imgColor.astype(np.uint8)
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




def getBasicFeatures(nim,fltim):
    """Calculates basic statistical descriptor or features

    :param nim:  256x256 RGB intensity normaliced image readed with OCV

    :param fltim: image flattern in a one dimention vecto

    :return: dictionary with some basic descritive staticstis
    """
    hist_grayim = np.histogram(fltim, bins=32)[0]
    dictemp = {}

    average = nim.mean(axis=0).mean(axis=0)
    skewness = skew(fltim)
    curtosis = kurtosis(fltim)

    dictemp['avWhite'] = average[0] + average[1] + average[2]
    dictemp['skewWhite'] = skewness
    dictemp['curtWhite'] = curtosis
    dictemp['num_piks_White'] = count_maxima(hist_grayim)
    dictemp['avRed'] = average[0]
    dictemp['avGreen'] = average[1]
    dictemp['avBlue'] = average[2]
    std = nim.std(axis=0).std(axis=0)
    dictemp['stdWhite'] = std[0] + std[1] + std[2]
    dictemp['stdRed'] = std[0]
    dictemp['stdGreen'] = std[1]
    dictemp['stdBlue'] = std[2]
    minimo = nim.min(axis=0).min(axis=0)
    dictemp['minWhite'] = minimo[0] + minimo[1] + minimo[2]
    dictemp['minRed'] = minimo[0]
    dictemp['minGreen'] = minimo[1]
    dictemp['minBlue'] = minimo[2]
    maximo = nim.max(axis=0).max(axis=0)
    dictemp['maxWhite'] = maximo[0] + maximo[1] + maximo[2]
    dictemp['maxRed'] = maximo[0]
    dictemp['maxGreen'] = maximo[1]
    dictemp['maxBlue'] = maximo[2]

    return dictemp


def get_color_histogram(nim):
    """Calculates a histogram of colors in 3x3x3 color combinations over min-max normaliced image as a resumed color descriptor

    :param nim: 256x256 RGB intensity normalice image readed with OCV

    :return: 3X3 RGB histogram
    """
    # Aqui se optinen el histogramade colores
    hist = cv2.calcHist([nim], [0, 1, 2], None, [3, 3, 3], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    hist = hist.astype(int)
    n = 3
    dictemp = {}
    l = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                dictemp[('r' + str(i) + 'g' + str(j) + 'b' + str(k))] = hist[l]
                l = l + 1
    return dictemp


def get_ftt_descriptors(fltim):
    """Calculates FFT2 transfor of image flattered and construct related resum descriptors

    :param fltim: image flattern in a one dimention vector

    :return: dictionary with 32 bean sumiriced FTT and some main FTT componet for ebery bean.

    """
    dictemp={}
    df2 = pd.DataFrame()
    nn = fltim.shape[-1]
    sp = np.fft.fft(fltim, n=nn)  # Optenenos la trasformada de fourier de la seri
    freq = np.fft.fftfreq(nn)
    # Determinamos las frecuencias de muestreo.
    # como la imagen esta compuesa por 65mil pixeles la frecuencia maxima es 65mil muestras o una muestra por cada 5m^2
    df2 = pd.DataFrame()
    df2['frec'] = freq
    # Como se entiende mas facil por pixeles escalmos las frecuencias a 256, donde 256 significa que se muestrea pixel a pixel
    # 128 Significaria que semuestrea cada 2 pixeles, 64 cada 4 pixeles... y 0 seria una sola muestra
    df2['frec'] = df2['frec'] * 256 * 2
    df2['real'] = (sp.real / max(sp.real)) ** 2  # Tomamos unicamente la componente real no usamos la imaginaria
    df2 = df2.sort_values(by=['frec'])
    df2 = df2[df2['frec'] > 0]
    n = int(df2.shape[0] / 8)
    frecus = []
    for i in range(8):
        x = n * i
        y = n * (1 + i)
        temp = df2.loc[x:y]
        temp = temp.sort_values(by=['real'])
        frecus = frecus + temp.tail(4)['frec'].tolist()
    l = 0
    for i in range(8):
        for j in range(4):
            dictemp[('frecbean' + str(i) + 'pow' + str(j))] = frecus[l]
            l = l + 1

    # Ahora vamos a estimar un uspectrograma reducido a 32 beans
    # es decir frecuencia de muestreo enrangos que van aumentando de a 8 pixeles y sacamos sa sumatoria en cada bean
    frecbeans = []
    n = int(df2.shape[0] / 32)
    for i in range(32):
        x = n * i
        y = n * (1 + i)
        temp = df2.loc[x:y]
        temp = temp.real.sum()
        frecbeans.append(temp)
    l = 0
    for i in range(len(frecbeans) - 1):
        dictemp[str(i)] = frecbeans[l]
        l = l + 1

    return dictemp


def getTextureFeatures(im, imgSize, no_points, radius, hlen):
    """Build LBP histogram for give image resampled aath imgSize

    :param im: 256x256 RGB image readed with ocv

    :param imgSize: new ssize for resampling image

    :param no_points: parameter or LBP
    :type no_points: int

    :param radius: paremeter of LBP

    :param hlen: len of histogram

    :return: lbp histogram
    """
    img = cv2.resize(im, (imgSize, imgSize), interpolation=cv2.INTER_AREA)
    imgColor = cv2.normalize(img, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    imgGray = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
    # Se utiliza el algorithmo local binary patterns con el metodo unicorme
    lbp = local_binary_pattern(imgGray, no_points, radius, method='uniform')
    # Calculate the histogram
    hist = np.unique(lbp, return_counts=True)
    x = hist[1]  # itemfreq(lbp.ravel())
    # Normalize the histogram
    histB = (hist[0], x[:] / sum(x[:]))
    # Se asegura una longitud del histograma
    a = len(histB[1])
    if a > hlen:
        histBv = (histB[0][a - hlen:], histB[1][a - hlen:])
    elif a < hlen:
        histBv = (range(hlen), range(hlen))
    else:
        # print('ok')
        histBv = histB
    return (histBv)

def get_lbp_descriptors(im):
    """Build 2 Local Binary Patter histogram for give image one in 68 pixel resample and other in 6 pix resample

    :param im: 256x256 RGB image readed with ocv

    :return: 2 lbp histograms for resampling image at 68X68 and 6X6 pixels
    """
    dictemp = {}
    imgSize = 68
    radius = 2
    no_points = 8 * radius
    hlen = no_points + radius
    hist68 = getTextureFeatures(im, imgSize, no_points, radius, hlen)
    # luego baja
    imgSize = 6
    radius = 1
    no_points = 8 * radius
    hlen = no_points + radius
    hist6 = getTextureFeatures(im, imgSize, no_points, radius, hlen)

    for i in range(9):  # Se encontro que las texturas entre deforestacion y no deforestacion cambian mas en 68 y6 pix
        dictemp['textur6_' + str(i)] = hist6[1][i]
    for i in range(18):  # Se encontro que las texturas entre deforestacion y no deforestacion cambian mas en 68 y6 pix
        dictemp['textur68_' + str(i)] = hist68[1][i]

    return dictemp

def get_orb_descriptors(im):
    """Gets descriptor usinr ORB algorithem from CSV

    :param im:

    :return:
    """
    dictemp = {}
    orb = cv2.ORB_create()
    im0 = remove_transparency(im, 0)
    kp, des = orb.detectAndCompute(im0, None)
    dictemp['orbKp'] = kp
    dictemp['orbDes'] = des
    try:
        dictemp['orbDesLen']= len(des)
    except:
        dictemp['orbDesLen']= 0

    return dictemp

def aply_foresty_model(dic_image, model):
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
       'minRed', 'minWhite', 'num_piks_White', 'r0g0b0',
       'r0g0b1', 'r0g0b2', 'r0g1b0', 'r0g1b1', 'r0g1b2', 'r0g2b0',
       'r0g2b1', 'r0g2b2', 'r1g0b0', 'r1g0b1', 'r1g0b2', 'r1g1b0',
       'r1g1b1', 'r1g1b2', 'r1g2b0', 'r1g2b1', 'r1g2b2', 'r2g0b0',
       'r2g0b1', 'r2g0b2', 'r2g1b0', 'r2g1b1', 'r2g1b2', 'r2g2b0',
       'r2g2b1', 'r2g2b2', 'simetriaImagenBinarizada', 'skewWhite',
       'stdBlue', 'stdGreen', 'stdRed', 'stdWhite', 'sumaUmbral',
       'textur68_0', 'textur68_1', 'textur68_10', 'textur68_11',
       'textur68_12', 'textur68_13', 'textur68_14', 'textur68_15',
       'textur68_16', 'textur68_17', 'textur68_2', 'textur68_3',
       'textur68_4', 'textur68_5', 'textur68_6', 'textur68_7',
       'textur68_8', 'textur68_9', 'textur6_0', 'textur6_1', 'textur6_2',
       'textur6_3', 'textur6_4', 'textur6_5', 'textur6_6', 'textur6_7',
       'textur6_8']
    #["orbDesLen","r0g0b0","skewWhite","0","curtWhite","r0g0b1","bordes32","textur68_12","avBlue","textur68_17"]

    diclist = [dic_image]
    dftemp = pd.DataFrame(diclist)
    vector_image = dftemp[feature_names].values#dftemp.values
    with open(model, 'rb') as fo:
        clf = load(model)
        y = clf.predict(vector_image)
    return y[0]



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
       'minRed', 'minWhite', 'num_piks_White', 'orbDesLen', 'r0g0b0',
       'r0g0b1', 'r0g0b2', 'r0g1b0', 'r0g1b1', 'r0g1b2', 'r0g2b0',
       'r0g2b1', 'r0g2b2', 'r1g0b0', 'r1g0b1', 'r1g0b2', 'r1g1b0',
       'r1g1b1', 'r1g1b2', 'r1g2b0', 'r1g2b1', 'r1g2b2', 'r2g0b0',
       'r2g0b1', 'r2g0b2', 'r2g1b0', 'r2g1b1', 'r2g1b2', 'r2g2b0',
       'r2g2b1', 'r2g2b2', 'simetriaImagenBinarizada', 'skewWhite',
       'stdBlue', 'stdGreen', 'stdRed', 'stdWhite', 'sumaUmbral',
       'textur68_0', 'textur68_1', 'textur68_10', 'textur68_11',
       'textur68_12', 'textur68_13', 'textur68_14', 'textur68_15',
       'textur68_16', 'textur68_17', 'textur68_2', 'textur68_3',
       'textur68_4', 'textur68_5', 'textur68_6', 'textur68_7',
       'textur68_8', 'textur68_9', 'textur6_0', 'textur6_1', 'textur6_2',
       'textur6_3', 'textur6_4', 'textur6_5', 'textur6_6', 'textur6_7',
       'textur6_8']

    diclist = [dic_image]
    dftemp = pd.DataFrame(diclist)
    vector_image = dftemp[feature_names].values#dftemp.values
    with open(model, 'rb') as fo:
        clf = load(model)
        y = clf.predict(vector_image)
    return y[0]


def aply_directClasification_model(dic_change, model):
    """
    dic_imagen= diccionario con descriptores de cambio entre las imagenes
    modelo: modleo sklearn persistido como objeto python3 con la libreria joblib
    """
    feature_names = ["sumaUmbral","21","r0g1b2","26","30","textur68_9","r0g0b0","0","textur68_7",
"r1g2b2","avGreen","textur68_0","r0g2b0","textur68_6","r1g0b1","textur68_4","textur68_13","textur68_12",
"r0g2b2","r2g0b2","r2g2b1","textur68_10","textur68_16","r1g1b2","textur68_17","lines","textur68_8",
"r1g1b1","bordes","textur68_11","r2g0b0","r0g1b1","avRed","r0g0b2","r2g2b0","r0g0b1"]
    diclist = [dic_change]
    dftemp = pd.DataFrame(diclist)
    vector_image = dftemp[feature_names].values  # dftemp.values
    with open(model, 'rb') as fo:
        clf = load(model)
        y = clf.predict(vector_image)
    return y[0] #Retorna 1 si es deforestacion 0 si es no deforestacion




def getFeatureVector(im,fileName,direct):
    """Build a feature vector dictionary for given image

    :param fileName: fileName of tile png image

    :return: dictionary with featurevectors
    """
    #im = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
    if direct=='no':
        nim = cv2.normalize(im, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        nim =im
    grayim = cv2.cvtColor(nim,cv2.COLOR_BGR2GRAY)  # Pasamos a escala de grises ya qu nos intersan las texturas y objetos no colores
    fltim = grayim.flatten()  # Convertimos la imagenes de matriz a una serie

    # Building descriptors dictionary
    dic = {}
    # Basic statistical descriptors.
    if fileName=='no_file':
        dic['imgSize'] = 100000
    else:
        dic['imgSize'] = os.path.getsize(fileName)
    basics = getBasicFeatures(nim,fltim )
    dic = {**dic, **basics}

    # Border descriptors
    bordes = getBorderFeatures(im,nim)
    dic = {**dic, **bordes}

    #Color histogram descriptors
    colorHistogram = get_color_histogram(nim)
    dic = {**dic, **colorHistogram}

    #FFT descriptors

    fftDescriptors = get_ftt_descriptors(fltim)
    dic = {**dic, **fftDescriptors}

    #Texture descriptors Local Binary pattern (LBP)

    lbpDescriptors = get_lbp_descriptors(im)
    dic = {**dic, **lbpDescriptors}

    #Geometric feature descriptos (croners) ORB
    orbDescriptors = get_orb_descriptors(im)
    dic = {**dic, **orbDescriptors}
    return dic


def contruirVecoresQA(path):
    """
    Construye un dataframe apartir de las imagenes clasificadas
    Para cada variable debe haber una carperperta nombrada como la variables y subcarpetas con el nombre de
    cada posible valor de la variable si es numerico solo puede ser un entero
    """
    diclist = []  # Lista con los vectores de caracteristicas de las imagenes de antes

    files = [path]

    for f in files:
        b = os.path.getsize(f)
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        direct = 'no'
        dic = getFeatureVector(im, f, direct)
        base = os.path.basename(f)
        dic['ID'] = base
        df = pd.DataFrame(diclist)
    return df


def contruirVectoresCambio(pathA,pathB,path2model_visibilidad,path2model_boscosidad,path2model_directo):
    path_one = pathB
    path_two = pathA
    imbefore = cv2.imread(path_one, cv2.IMREAD_UNCHANGED)
    b = os.path.getsize(path_one)
    dicB = getFeatureVector(imbefore, path_one, 'no')
    Vb = round(aply_regresion_model(dicB, path2model_visibilidad), 0)
    Bb = round(aply_foresty_model(dicB, path2model_boscosidad), 0)
    dicB['boscosidad'] = Bb
    dicB['visibilidad'] = Vb
    imafter = cv2.imread(path_two, cv2.IMREAD_UNCHANGED)
    bb = os.path.getsize(path_two)
    dicA = getFeatureVector(imafter, path_two, 'no')
    Va = round(aply_regresion_model(dicA, path2model_visibilidad), 0)
    Ba = round(aply_foresty_model(dicA, path2model_boscosidad), 0)
    dicA['boscosidad'] = Ba
    dicA['visibilidad'] = Va

    # Search for orb feature matching
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(dicB['orbDes'], dicA['orbDes'])
        good = []
    except:
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
    dic['visibilidad'] = min(dicB['visibilidad'], dicA['visibilidad'])
    dic['orbMatch'] = len(matches)
    dic['orbMatchThreshold'] = len(good)
    dic['eucli_dist'] = matrixDistance(imbefore, imafter)
    dic['directo'] = directo(imbefore, imafter, path2model_directo)


    return dic



def aply_clasification_model(dic_change, model):
    """
    dic_imagen= diccionario con descriptores de cambio entre las imagenes
    modelo: modleo sklearn persistido como objeto python3 con la libreria joblib
    """
    feature_names =['r0g0b0', 'stdRed', 'curtWhite', 'sumaUmbral', 'r1g1b1', 'r0g1b0', 'avRed', '0',
                    'textur68_10', 'maxRed', 'avWhite', 'orbMatch', 'textur68_9', 'r1g1b2', 'minWhite',
                    'simetriaImagenBinarizada', 'minGreen', 'skewWhite', 'textur68_0', 'textur68_17',
                    'visibilidad', '8', 'textur68_7', '28', '27', 'r0g0b1', 'minBlue', '5',
                    'orbMatchThreshold', '4', 'r1g2b2', 'avBlue', '6', 'avGreen', '7', 'r0g1b2',
                    'boscosidad', 'eucli_dist', 'directo']
    diclist = [dic_change]
    dftemp = pd.DataFrame(diclist)
    vector_image = dftemp[feature_names].values  # dftemp.values
    with open(model, 'rb') as fo:
        clf = load(model)
        y = clf.predict(vector_image)
    return y[0] #Retorna 1 si es deforestacion 0 si es no deforestacion


def evaluateChange(pathA,pathB,path2model_visibilidad,path2model_boscosidad,
                   path2model_directo,path2model_change, directo):
    dicCh=contruirVectoresCambio(pathA, pathB, path2model_visibilidad, path2model_boscosidad, path2model_directo)
    if directo == 'si':
        y= round(dicCh['directo'],0)
    else:
        y=round(aply_clasification_model(dicCh, path2model_change), 0)
    return y

