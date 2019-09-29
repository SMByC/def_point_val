# -*- coding: utf-8 -*-
import subprocess

def checkPackages():
    try:
        import shutil
    except:
        subprocess.check_call(['python3', '-m', 'pip', 'install', 'shutil'])


    # GIS Libraries
    try:
        import geojson
    except:
        subprocess.check_call(['python3', '-m', 'pip', 'install', 'geojson'])

    try:
        import pygeotile
    except:
        subprocess.check_call(['python3', '-m', 'pip', 'install', 'pyGeoTile'])

    # Math and statistics  Libraries

    try:
        import math
    except:
        subprocess.check_call(['python3', '-m', 'pip', 'install', 'math'])

    try:
        import pandas as pd
    except:
        subprocess.check_call(['python3', '-m', 'pip', 'install', 'pandas'])

    try:
        import numpy as np
    except:
        subprocess.check_call(['python3', '-m', 'pip', 'install', 'numpy'])

    try:
        from scipy import stats
        from scipy.stats import kurtosis, skew
    except:
        subprocess.check_call(['python3', '-m', 'pip', 'install', 'scipy'])

    try:
        from matplotlib import image
        from matplotlib import pyplot
        import matplotlib.pyplot as plt
    except:
        subprocess.check_call(['python3', '-m', 'pip', 'install', 'matplotlib'])

    # Images and computer vision Libraries
    try:
        from PIL import Image, ImageChops
    except:
        subprocess.check_call(['python3', '-m', 'pip', 'install', 'pil'])

    try:
        import cv2
    except:
        subprocess.check_call(['python3', '-m', 'pip', 'install', 'opencv-python'])

    try:
        from skimage.feature import local_binary_pattern
    except:
        subprocess.check_call(['python3', '-m', 'pip', 'install', 'scikit-image'])
