# -*- coding: utf-8 -*-
#-------------------Import Lybraries------------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------------------------------

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
    source_img = cv2.cvtColor(source[:,:,:3], cv2.COLOR_BGR2GRAY)
    source_mask = source[:,:,3]  * (1 / 255.0)

    background_mask = 1.0 - source_mask

    bg_part = (background_color * (1 / 255.0)) * (background_mask)
    source_part = (source_img * (1 / 255.0)) * (source_mask)

    return np.uint8(cv2.addWeighted(bg_part, 255.0, source_part, 255.0, 0.0))

def getBorderFeatures(img,nim):
    """genear descriptores basados en bordes No de lineas rectas, cantidad de bordes img es una imagen png 256x256 cargada con cv2

    :param img: 256x256 RGB intensity normalice image readed with OCV

    :param nim: image flattern in a one dimention vecto

    :return: dictionary with some border features: number of lines, numbre of border pixels in image
    """
    # primero para la imagen si reducir

    imgColor = nim
    img2 = remove_transparency(imgColor, 0)

    # Se detectan bordes usando Canny
    img_edged = cv2.Canny(img2, 100, 200)

    bordes = img_edged.sum()
    # Se cuentan las linea rectas usando HoughLines
    lines = cv2.HoughLines(img_edged, 1, np.pi / 180,64)  # nomrmalmente deben se entre la curata y la midad de loz pixeles W.
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
        dictemp['orbDesLen'] = len(des)
    except:
        dictemp['orbDesLen'] = 0

    return dictemp





def png2GeoTif(src_filename,dst_filename,google_x,google_y,zoom,pixels,epsg):
    """
    This function make image georeference adn convert to .tif format to be able to load it in qgis

    :param src_filename: '/path/to/source.tif'

    :param dst_filename: '/path/to/destination.tif'

    :param google_x:

    :param google_y:

    :param zoom: 15

    :param pixels:

    :param epsg: 4326 #Es el eps de los puntos del IDEAM y el que retorna pygeotile

    :return:
    """
    # Opens source dataset
    src_ds = gdal.Open(src_filename)
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    # Open destination dataset
    dst_ds = driver.CreateCopy(dst_filename, src_ds, 0)
    #Optener los puntos de referencia
    tile = Tile.from_google(google_x, google_y, zoom)
    bounds = tile.bounds #Retorna el punto sur occidental y el punto nororiental del tile.
    p1 = bounds[0] #Point(latitude=2.6357885741666065, longitude=-72.388916015625)
    p2 = bounds[1]
    uperleftx = min([p1[1], p2[1]])
    uperlefty = max([p1[0], p2[0]])
    scalex = abs(p1[1]-p2[1])/pixels
    scaley = abs(p1[0]-p2[0])/pixels
    # Specify raster location through geotransform array
    # (uperleftx, scalex, skewx, uperlefty, skewy, scaley)
    # Scale = size of one pixel in units of raster projection
    # this example below assumes 100x100
    gt = [uperleftx, scalex, 0, uperlefty, 0, -scaley]
    #print(gt)
    # Set location
    dst_ds.SetGeoTransform(gt)
    # Get raster projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dest_wkt = srs.ExportToWkt()
    # Set projection
    dst_ds.SetProjection(dest_wkt)
    # Close files
    dst_ds = None
    src_ds = None