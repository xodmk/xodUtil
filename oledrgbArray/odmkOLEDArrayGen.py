# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header begin-----------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

# __::((odmkOLEDArrayGen.py))::__

# Python ODMK img processing research
# ffmpeg experimental


# https://docs.scipy.org/doc/scipy-0.18.1/reference/ndimage.html
# https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/ndimage.html

# http://www.pythonware.com/media/data/pil-handbook.pdf


# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header end-------------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os, sys
import glob, shutil
import string
from math import atan2, floor, ceil
import random
import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import misc

from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

rootDir = 'C:\\odmkDev\\odmkCode\\odmkPython\\'


#sys.path.insert(0, 'C:/odmkDev/odmkCode/odmkPython/util')
sys.path.insert(0, rootDir+'util')
from odmkClear import *
# optional
#from odmkPlotUtil import *
import odmkPlotUtil as odmkplt

#sys.path.insert(1, 'C:/odmkDev/odmkCode/odmkPython/DSP')
sys.path.insert(1, rootDir+'eye')
import odmkEYEuObj as eye

#sys.path.insert(1, 'C:/odmkDev/odmkCode/odmkPython/DSP')
sys.path.insert(2, rootDir+'DSP')
import odmkClocks as clks
import odmkSigGen1 as sigGen

# temp python debugger - use >>>pdb.set_trace() to set break
import pdb



# // *---------------------------------------------------------------------* //
plt.close('all')
clear_all()

# // *---------------------------------------------------------------------* //

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# -----------------------------------------------------------------------------
# utility func
# -----------------------------------------------------------------------------

def cyclicZn(n):
    ''' calculates the Zn roots of unity '''
    cZn = np.zeros((n, 1))*(0+0j)    # column vector of zero complex values
    for k in range(n):
        # z(k) = e^(((k)*2*pi*1j)/n)        # Define cyclic group Zn points
        cZn[k] = np.cos(((k)*2*np.pi)/n) + np.sin(((k)*2*np.pi)/n)*1j   # Euler's identity

    return cZn


def quarterCos(n):
    ''' returns a quarter period cosine wav of length n '''
    t = np.linspace(0, np.pi/4, n)
    qtrcos = np.zeros((n))
    for j in range(n):
        qtrcos[j] = sp.cos(t[j])
    return qtrcos


def randomIdx(k):
    '''returns a random index in range 0 - k-1'''
    randIdx = round(random.random()*(k-1))
    return randIdx

def randomIdxArray(n, k):
    '''for an array of k elements, returns a list of random elements
       of length n (n integers rangin from 0:k-1)'''
    randIdxArray = []
    for i in range(n):
        randIdxArray.append(round(random.random()*(k-1)))
    return randIdxArray
      
def randomPxlLoc(SzX, SzY):
    '''Returns a random location in a 2D array of SzX by SzY'''
    randPxlLoc = np.zeros(2)
    randPxlLoc[0] = int(round(random.random()*(SzX-1)))
    randPxlLoc[1] = int(round(random.random()*(SzY-1)))
    return randPxlLoc


class odmkWeightRnd:
    def __init__(self, weights):
        self.__max = .0
        self.__weights = []
        for value, weight in weights.items():
            self.__max += weight
            self.__weights.append((self.__max, value))

    def random(self):
        r = random.random() * self.__max
        for wtceil, value in self.__weights:
            if wtceil > r:
                return value
                
                
                
class odmkFibonacci:
    ''' Iterator that generates numbers in the Fibonacci sequence '''

    def __init__(self, max):
        self.max = max

    def __iter__(self):
        self.a = 0
        self.b = 1
        return self

    def __next__(self):
        fib = self.a
        if fib > self.max:
            raise StopIteration
        self.a, self.b = self.b, self.a + self.b
        return fib                
                

# a random 0 or 1
# rndIdx = round(random.random())
# -----------------------------------------------------------------------------
# color func ?????
# -----------------------------------------------------------------------------

def rgb_to_hsv(rgb):
    ''' Translated from source of colorsys.rgb_to_hsv
      r,g,b should be a numpy arrays with values between 0 and 255
      rgb_to_hsv returns an array of floats between 0.0 and 1.0. '''
      
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    ''' Translated from source of colorsys.hsv_to_rgb
      h,s should be a numpy arrays with values between 0.0 and 1.0
      v should be a numpy array with values between 0.0 and 255.0
      hsv_to_rgb returns an array of uints between 0 and 255. '''
      
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def shift_hue(img,hout):
    hsv = rgb_to_hsv(img)
    hsv[...,0] = hout
    rgb = hsv_to_rgb(hsv)
    return rgb



# ???check???
def circleMask(SzX, SzY, pxlLoc, radius):
    a, b = pxlLoc
    y, x = np.ogrid[-a:SzX-a, -b:SzY-b]
    cmask = x*x + y*y <= radius*radius
    
    return cmask



# -----------------------------------------------------------------------------
# concat directory functions
# -----------------------------------------------------------------------------



def concatAllDir(dirList, concatDir, reName):
    ''' Takes a list of directories containing .jpg images,
        renames and concatenates all files into new arrays.
        The new file names are formatted for processing with ffmpeg
        if w = 1, write files into output directory.
        if concatDir specified, change output dir name.
        if concatName specified, change output file names'''


    imgFileList = []
    for i in range(len(dirList)):
        imgFileList.extend(sorted(glob.glob(dirList[i]+'*.jpg')))

    #pdb.set_trace()
        
    # normalize image index n_digits to allow for growth
    #newFileList = glob.glob(concatDir+'*.jpg')        
        
        
    imgCount = len(imgFileList)
    n_digits = int(ceil(np.log10(imgCount))) + 2
    nextInc = 0
    for i in range(imgCount):
        nextInc += 1
        zr = ''
        for j in range(n_digits - len(str(nextInc))):
            zr += '0'
        strInc = zr+str(nextInc)
        # print('strInc = '+str(strInc))
        imgNormalizeNm = reName+strInc+'.jpg'
        #imgConcatNmFull = concatDir+imgNormalizeNm
        currentNm = os.path.split(imgFileList[i])[1]
        shutil.copy(imgFileList[i], concatDir)
        currentFile = os.path.join(concatDir+currentNm)
        
        imgConcatNmFull = os.path.join(concatDir+imgNormalizeNm)
        os.rename(currentFile, imgConcatNmFull)
        #pdb.set_trace()

    return



# -----------------------------------------------------------------------------
# img scrolling functions
# -----------------------------------------------------------------------------


def scrollH(image, delta, ctrl):
    ''' Scroll an image sideways
        delta = number of pixels to roll image
        ctrl: 0 = left; 1 = right '''
    xsize, ysize = image.size
    delta = delta % xsize
    image2 = image.copy()
    if delta == 0: return image
    if ctrl==0:    # left scroll
        part1 = image.crop((0, 0, delta, ysize))
        part2 = image2.crop((delta, 0, xsize, ysize))
        image.paste(part1, (xsize-delta, 0, xsize, ysize))
        image.paste(part2, (0, 0, xsize-delta, ysize))
    elif ctrl==1:    # right scroll
        part1 = image.crop((0, 0, xsize-delta, ysize))
        part2 = image2.crop((xsize-delta, 0, xsize, ysize))
        image.paste(part1, (delta, 0, xsize, ysize))
        image.paste(part2, (0, 0, delta, ysize))       
    return image


def scrollV(image, delta, ctrl):
    ''' Scroll an image vertically
        delta = number of pixels to scroll image
        ctrl: 0 = down; 1 = up '''
    xsize, ysize = image.size
    delta = delta % xsize
    image2 = image.copy()
    if delta == 0: return image
    if ctrl==0:    # down scroll
        part1 = image.crop((0, delta, xsize, ysize))
        part2 = image2.crop((0, 0, xsize, delta))
        image.paste(part1, (0, 0, xsize, ysize-delta))
        image.paste(part2, (0, ysize-delta, xsize, ysize))
    elif ctrl==1:    # up scroll
        part1 = image.crop((0, ysize-delta, xsize, ysize))
        part2 = image2.crop((0, 0, xsize, ysize-delta))
        image.paste(part1, (0, 0, xsize, delta))
        image.paste(part2, (0, delta, xsize, ysize))
    return image


def scrollDiag(image, delta, ctrl):
    ''' Scroll an image diagonally
        delta = number of pixels to scroll image (horizontally - vertical follows proportionally)
        ctrl: 0 = up left; 1 = down left; 2 = up right; 3 = down right '''
    if not(ctrl==0 or ctrl==1 or ctrl==2 or ctrl==3):
        print('ERROR: ctrl must be {0, 1, 2, 3}')
    xsize, ysize = image.size
    delta = delta % xsize
    image2 = image.copy()
    image3 = image.copy()
    image4 = image.copy()
    if delta == 0: return image
    if ctrl==0:    # down scroll
        part1 = image.crop((0, delta, xsize, ysize))
        part2 = image2.crop((0, 0, xsize, delta))
        image.paste(part1, (0, 0, xsize, ysize-delta))
        image.paste(part2, (0, ysize-delta, xsize, ysize))
    elif ctrl==1:    # up scroll
        part1 = image.crop((0, ysize-delta, xsize, ysize))
        part2 = image2.crop((0, 0, xsize, ysize-delta))
        image.paste(part1, (0, 0, xsize, delta))
        image.paste(part2, (0, delta, xsize, ysize))
    if ctrl==2:    # down scroll
        part1 = image.crop((0, delta, xsize, ysize))
        part2 = image2.crop((0, 0, xsize, delta))
        image.paste(part1, (0, 0, xsize, ysize-delta))
        image.paste(part2, (0, ysize-delta, xsize, ysize))
    elif ctrl==3:    # up scroll
        part1 = image.crop((0, ysize-delta, xsize, ysize))
        part2 = image2.crop((0, 0, xsize, ysize-delta))
        image.paste(part1, (0, 0, xsize, delta))
        image.paste(part2, (0, delta, xsize, ysize))        
    return image


# -----------------------------------------------------------------------------
# img scaling / zooming / cropping functions
# -----------------------------------------------------------------------------


def odmkEyeRescale(self, img, SzX, SzY):
    # Rescale Image to SzW width and SzY height:
    # *****misc.imresize returns img with [Y-dim, X-dim] format instead of X,Y*****
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]
    imgRescale_r = misc.imresize(img_r, [SzY, SzX], interp='cubic')
    imgRescale_g = misc.imresize(img_g, [SzY, SzX], interp='cubic')
    imgRescale_b = misc.imresize(img_b, [SzY, SzX], interp='cubic')
    imgRescale = np.dstack((imgRescale_r, imgRescale_g, imgRescale_b))
    return imgRescale


def odmkEyeZoom(self, img, Z):
    # Zoom Image by factor Z:
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]
    imgZoom_r = ndimage.interpolation.zoom(img_r, Z)
    imgZoom_g = ndimage.interpolation.zoom(img_g, Z)
    imgZoom_b = ndimage.interpolation.zoom(img_b, Z)
    imgZoom = np.dstack((imgZoom_r, imgZoom_g, imgZoom_b))
    return imgZoom


def odmkEyeCrop(self, img, SzX, SzY, high):
    ''' crop img to SzX width x SzY height '''

    imgWidth = img.shape[1]
    imgHeight = img.shape[0]

    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]

    imgCrop_r = np.zeros((SzY, SzX))
    imgCrop_g = np.zeros((SzY, SzX))
    imgCrop_b = np.zeros((SzY, SzX))

    wdiff_lhs = 0
    hdiff_ths = 0

    if imgWidth > SzX:
        wdiff = floor(imgWidth - SzX)
        if (wdiff % 2) == 1:    # wdiff is odd
            wdiff_lhs = floor(wdiff / 2)
            # wdiff_rhs = ceil(wdiff / 2)
        else:
            wdiff_lhs = wdiff / 2
            # wdiff_rhs = wdiff / 2

    if imgHeight > SzY:
        hdiff = floor(imgHeight - SzY)
        if (hdiff % 2) == 1:    # wdiff is odd
            hdiff_ths = floor(hdiff / 2)
            # hdiff_bhs = ceil(wdiff / 2)
        else:
            hdiff_ths = hdiff / 2
            # hdiff_bhs = hdiff / 2

    for j in range(SzY):
        for k in range(SzX):
            if high == 1:
                imgCrop_r[j, k] = img_r[j, int(k + wdiff_lhs)]
                imgCrop_g[j, k] = img_g[j, int(k + wdiff_lhs)]
                imgCrop_b[j, k] = img_b[j, int(k + wdiff_lhs)]
            else:
                # pdb.set_trace()
                imgCrop_r[j, k] = img_r[int(j + hdiff_ths), int(k + wdiff_lhs)]
                imgCrop_g[j, k] = img_g[int(j + hdiff_ths), int(k + wdiff_lhs)]
                imgCrop_b[j, k] = img_b[int(j + hdiff_ths), int(k + wdiff_lhs)]
        # if j == SzY - 1:
            # print('imgItr ++')
    imgCrop = np.dstack((imgCrop_r, imgCrop_g, imgCrop_b))
    return imgCrop


def odmkEyeDim(self, img, SzX, SzY, high):
    ''' Zoom and Crop image to match source dimensions '''

    imgWidth = img.shape[1]
    imgHeight = img.shape[0]

    #pdb.set_trace()

    if (imgWidth == SzX and imgHeight == SzY):
        # no processing
        imgScaled = img

    elif (imgWidth == SzX and imgHeight > SzY):
        # cropping only
        imgScaled = self.odmkEyeCrop(img, SzX, SzY, high)

    elif (imgWidth > SzX and imgHeight == SzY):
        # cropping only
        imgScaled = self.odmkEyeCrop(img, SzX, SzY, high)

    elif (imgWidth > SzX and imgHeight > SzY):
        # downscaling and cropping
        wRatio = SzX / imgWidth
        hRatio = SzY / imgHeight
        if wRatio >= hRatio:
            zoomFactor = wRatio
        else:
            zoomFactor = hRatio
        imgScaled = self.odmkEyeZoom(img, zoomFactor)
        imgScaled = self.odmkEyeCrop(imgScaled, SzX, SzY, high)

    elif (imgWidth < SzX and imgHeight == SzY):
        # upscaling and cropping
        wdiff = SzX - imgWidth
        zoomFactor = (imgWidth + wdiff) / imgWidth
        imgScaled = self.odmkEyeZoom(img, zoomFactor)
        imgScaled = self.odmkEyeCrop(imgScaled, SzX, SzY, high)

    elif (imgWidth == SzX and imgHeight < SzY):
        # upscaling and cropping
        hdiff = SzY - imgHeight
        zoomFactor = (imgHeight + hdiff) / imgHeight
        imgScaled = self.odmkEyeZoom(img, zoomFactor)
        imgScaled = self.odmkEyeCrop(imgScaled, SzX, SzY, high)

    elif (imgWidth < SzX and imgHeight < SzY):
        # upscaling and cropping (same aspect ratio -> upscaling only)
        wRatio = SzX / imgWidth
        hRatio = SzY / imgHeight
        if wRatio >= hRatio:
            zoomFactor = wRatio
        else:
            zoomFactor = hRatio
        imgScaled = self.odmkEyeZoom(img, zoomFactor)
        imgScaled = self.odmkEyeCrop(imgScaled, SzX, SzY, high)

    elif (imgWidth > SzX and imgHeight < SzY):
        # upscaling and cropping
        # wdiff = imgWidth - SzX
        hdiff = SzY - imgHeight
        zoomFactor = (imgHeight + hdiff) / imgHeight
        imgScaled = self.odmkEyeZoom(img, zoomFactor)
        imgScaled = self.odmkEyeCrop(imgScaled, SzX, SzY, high)

    elif (imgWidth < SzX and imgHeight > SzY):
        # upscaling and cropping
        wdiff = SzX - imgWidth
        # hdiff = imgHeight - SzY
        zoomFactor = (imgWidth + wdiff) / imgWidth
        imgScaled = self.odmkEyeZoom(img, zoomFactor)
        imgScaled = self.odmkEyeCrop(imgScaled, SzX, SzY, high)

    return imgScaled


def odmkScaleAll(self, srcObjList, SzX, SzY, w=0, high=0, outDir='None', outName='None'):
    ''' rescales and normalizes all .jpg files in image object list.
        scales, zooms, & crops images to dimensions SzX, SzY
        Assumes srcObjList of ndimage objects exists in program memory
        Typically importAllJpg is run first.
        if w == 1, write processed images to output directory'''

    # check write condition, if w=0 ignore outDir
    if w == 0 and outDir != 'None':
        print('Warning: write = 0, outDir ignored')
    if w == 1:
        if outDir == 'None':
            print('Error: outDir must be specified; processed img will not be saved')
            w = 0
        else:
            # If Dir does not exist, makedir:
            os.makedirs(outDir, exist_ok=True)
    if outName != 'None':
        imgScaledOutNm = outName
    else:
        imgScaledOutNm = 'odmkScaleAllOut'

    imgScaledNmArray = []
    imgScaledArray = []

    imgCount = len(srcObjList)
    # Find num digits required to represent max index
    n_digits = int(ceil(np.log10(imgCount))) + 2
    nextInc = 0
    for k in range(imgCount):
        imgScaled = self.odmkEyeDim(srcObjList[k], SzX, SzY, high)
        # auto increment output file name
        nextInc += 1
        zr = ''    # reset lead-zero count to zero each itr
        for j in range(n_digits - len(str(nextInc))):
            zr += '0'
        strInc = zr+str(nextInc)
        imgScaledNm = imgScaledOutNm+strInc+'.jpg'
        if w == 1:
            imgScaledFull = outDir+imgScaledNm
            misc.imsave(imgScaledFull, imgScaled)
        imgScaledNmArray.append(imgScaledNm)
        imgScaledArray.append(imgScaled)

    if w == 1:    
        print('Saved Scaled images to the following location:')
        print(imgScaledFull)

    print('\nScaled all images in the source directory\n')
    print('Renamed files: "processScaled00X.jpg"')

    print('\nCreated numpy arrays:')
    print('<<imgScaledArray>> (img data objects)')
    print('<<imgScaledNmArray>> (img names)\n')

    print('\n')

    print('// *--------------------------------------------------------------* //')

    return [imgScaledArray, imgScaledNmArray]



# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


print('// //////////////////////////////////////////////////////////////// //')
print('// *--------------------------------------------------------------* //')
print('// *---::ODMK OLED RGB Array Generator::---*')
print('// *--------------------------------------------------------------* //')
print('// \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ //')


#xLength = 19.492
xLength = 0.5
fs = 48000.0        # audio sample rate:
framesPerSec = 30.0 # video frames per second:
# ub313 bpm = 178 , 267
bpm = 78            # 39 - 78 - 156
timeSig = 4         # time signature: 4 = 4/4; 3 = 3/4


# Digilent Pmod OLED: 96x64 pixels
SzX = 96
SzY = 64


# // *********************************************************************** //
# // Directory Definitions ************************************************* //
# // *********************************************************************** //

# *** hardcoded paths
oledrgbOutputDir = rootDir+'util\\oledrgbArray\\'
oledrgbSourceDir = rootDir+'util\\sourceArray\\'


jpgSrcDir = oledrgbSourceDir

eyeOutDirName = 'gorgulanPalace\\'


# output dir where processed img files are stored:
# eyeOutDirName = 'myOutputDirName'    # defined above
imgOutDir = oledrgbOutputDir+eyeOutDirName
# If Dir does not exist, makedir:
os.makedirs(imgOutDir, exist_ok=True)



# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


if (0):
    arrayName = 'oledrgbPATTERN1'
    rgbTmpData = np.linspace(1,255,12288)
    
    
    rgbPixels = [];
    rgbPixelString = ""
    for pxl in rgbTmpData:
        rgbPixels.append(int(pxl))
        #print(str(int(pxl)))
        #rgbPixelString.join(","+str(pxl))
    
    print("const uint8_t "+arrayName+"[12288] = {")
    charTmpData = str(rgbPixels)
    charTmpData = charTmpData[1:-1]  # Stips off brackets and final comma
    print(charTmpData)
    print("};")

#print("max of charTmpData = "+str(max(rgbPixels)))


rasterLength = 64*96*3

if (1):
    arrayName = 'oledrgbPATTERN2'
    rgbTmpData = 128*np.sin(np.linspace(0,2*np.pi,rasterLength))+128
    
    
    rgbPixels = [];
    rgbPixelString = ""
    Xindex = 0;
    Yindex = 0;
    for Yindex in range(1,64+1):
        print("GORGULAN check Y range: "+str(Yindex))
        for Xindex in range(1,96*3+1):
            if (Xindex%3==1):
                # ** Process RED Pixel
                rgbPixels.append(255)
            elif (Xindex%3==2):
                rgbPixels.append(0)
            elif (Xindex%3==0):
                rgbPixels.append(0)
        #print(str(int(pxl)))
        #rgbPixelString.join(","+str(pxl))
    
    print("const uint8_t "+arrayName+"["+str(rasterLength)+"] = {")
    charTmpData = str(rgbPixels)
    charTmpData = charTmpData[1:-1]  # Stips off brackets and final comma
    print(charTmpData)
    print("};")


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

## Load input image:
#gorgulan11 = misc.imread(srcDir+'keiLHC.jpeg')
#
#numImg = 33
#
#hRotVal = np.linspace(0, 6*np.pi, numImg)
#hRotVal = np.sin(hRotVal)
#
#
##green_hue = (180-78)/360.0
##red_hue = (180-180)/360.0
##rot_hue = (50)/360.0
#
#
#
##gorgulanRed = shift_hue(gorgulan11, rot_hue)
#
##eyeHueFull = outDir+'eyeHue.jpg'
##misc.imsave(eyeHueFull, gorgulanRed)
#
#
#n_digits = int(ceil(np.log10(numImg))) + 2
#nextInc = 0
#for i in range(numImg):
#    
#    gorgulanShift = shift_hue(gorgulan11, hRotVal[i])    
#    
#    nextInc += 1
#    zr = ''
#    for j in range(n_digits - len(str(nextInc))):
#        zr += '0'
#    strInc = zr+str(nextInc)
#    eyeHueIdxNm = eyeHueNm+strInc+'.jpg'
#    eyeHueFull = outDir+eyeHueIdxNm
#    misc.imsave(eyeHueFull, gorgulanShift)



# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


# ***** EYE horizontal / vertical / diagonal scroll tests *****

if (0):

    eyeIn5 = jpgSrcDir+'bananasloth0036.jpg'
    eyeIn6 = jpgSrcDir+'microscopic0042.jpg'
    
    eyeOutFileNameL = 'eyeScrollTestL'
    eyeScrollFullL = imgOutDir+eyeOutFileNameL+'.jpg'
    eyeOutFileNameR = 'eyeScrollTestR'
    eyeScrollFullR = imgOutDir+eyeOutFileNameR+'.jpg'
    eyeOutFileNameD = 'eyeScrollTestD'
    eyeScrollFullD = imgOutDir+eyeOutFileNameD+'.jpg'
    eyeOutFileNameU = 'eyeScrollTestU'
    eyeScrollFullU = imgOutDir+eyeOutFileNameU+'.jpg'
    
    eyeScrollL = misc.imread(eyeIn6)
    eyeScrollR = misc.imread(eyeIn6)
    eyeScrollD = misc.imread(eyeIn6)
    eyeScrollU = misc.imread(eyeIn6)
    
    eyeImgL = Image.fromarray(eyeScrollL)
    eyeImgR = Image.fromarray(eyeScrollR)
    eyeImgD = Image.fromarray(eyeScrollD)
    eyeImgU = Image.fromarray(eyeScrollU)
    
    
    delta = 200
    
    eyeScrollL = scrollH(eyeImgL, delta, 0)
    misc.imsave(eyeScrollFullL, eyeScrollL)
    
    eyeScrollR = scrollH(eyeImgR, delta, 1)
    misc.imsave(eyeScrollFullR, eyeScrollR)
    
    eyeScrollD = scrollV(eyeImgD, delta, 0)
    misc.imsave(eyeScrollFullD, eyeScrollD)
    
    eyeScrollU = scrollV(eyeImgU, delta, 1)
    misc.imsave(eyeScrollFullU, eyeScrollU)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



def horizTrails(img, SzX, SzY, nTrail, ctrl):
    ''' reconstructs img with left & right trails
        nTrail controls number of trails
        ctrl controls x-axis coordinates '''
    
    if ctrl == 0:    # linear
        trails = np.linspace(0, SzX/2, nTrail+2)
        trails = trails[1:len(trails)-1]
    
    else:    # log spaced    
        trails = np.log(np.linspace(1,10,nTrail+2))
        sc = 1/max(trails)
        trails = trails[1:len(trails)-1]
        trails = trails*sc*int(SzX/2)
    
    alpha = np.linspace(0, 1, nTrail)
    
    # coordinates = (left, lower, right, upper)
    boxL = (0, 0, int(SzX / 2), SzY)
    eyeSubL = img.crop(boxL)
    
    boxR = (int(SzX / 2), 0, SzX, SzY)
    eyeSubR = img.crop(boxR)
    
    
    alphaCnt = 0
    for trail in trails:
    
        itrail = int(trail)
    
        boxL_itr1 = (itrail, 0, int(SzX / 2), SzY)
        eyeSubL_itr1 = img.crop(boxL_itr1)
        #print('\nboxL_itr1 = '+str(boxL_itr1))    
        
        boxR_itr1 = (int(SzX / 2), 0, SzX - itrail, SzY)
        eyeSubR_itr1 = img.crop(boxR_itr1)
        #print('\nboxR_itr1 = '+str(boxR_itr1))
        # if tcnt == 2:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        #pdb.set_trace()  
        
        boxL_base1 = (0, 0, int(SzX / 2) - itrail, SzY)
        eyeSubL_base1 = img.crop(boxL_base1)
        
        boxR_base1 = (int((SzX / 2) + itrail), 0, SzX, SzY)
        eyeSubR_base1 = img.crop(boxR_base1)
        
        eyeFadeL_itr1 = Image.blend(eyeSubL_itr1, eyeSubL_base1, alpha[alphaCnt])
        eyeFadeR_itr1 = Image.blend(eyeSubR_itr1, eyeSubR_base1, alpha[alphaCnt])
        
        #pdb.set_trace()
        
        boxL_sb1 = (0, 0, int(SzX / 2) - itrail, SzY)
        boxR_sb1 = (itrail, 0, int(SzX / 2), SzY)    
       
        
        #eye1subL.paste(eye1subL_itr1, boxL_sb1)
        #eye2subR.paste(eye2subR_itr1, boxR_sb1)
        eyeSubL.paste(eyeFadeL_itr1, boxL_sb1)
        eyeSubR.paste(eyeFadeR_itr1, boxR_sb1)    
        
        
        img.paste(eyeSubL, boxL)
        img.paste(eyeSubR, boxR)
    
        alphaCnt += 1

    eyeOutFileName = 'eyeBoxExp'
    eyeBoxFull = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(eyeBoxFull, img)

    return
    

if (0):

    eyeMirExp4 = jpgSrcDir+'bananasloth0036.jpg'

    #eye1 = misc.imread(eyeIn1)
    eye2 = misc.imread(eyeMirExp4)
    #eye1img = Image.fromarray(eye1)
    eye2img = Image.fromarray(eye2)
    
    #eye1imgCpy = eye2img.copy()
    eyeImgCpy = eye2img.copy()

    
    ctrl = 1
    nTrail = 3
    
    img = eyeImgCpy
    
    horizTrails(img, SzX, SzY, nTrail, ctrl)



# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



def eyeRndFour(img1, img2, img3, img4, SzX, SzY, pxlLoc):
    ''' generates composite 4 frame resized images
        sub dimensions defined by pxlLoc [x, y]
        img1 - img4 PIL images
        pxlLoc: tuple of x and y coordinates '''
    
    
    #alpha = np.linspace(0, 1, nTrail)
    
    # coordinates = (left, lower, right, upper)
    boxUL = (0, int(pxlLoc[1]), int(pxlLoc[0]), SzY)
    newSize = [int(pxlLoc[0]), int(SzY - pxlLoc[1])]
    eyeSubUL = img1.resize(newSize)
    
    #pdb.set_trace()    
    
    boxUR = (int(pxlLoc[0]), int(pxlLoc[1]), SzX, SzY)
    newSize = [int(SzX - pxlLoc[0]), int(SzY - pxlLoc[1])]
    eyeSubUR = img2.resize(newSize)
    
     # coordinates = (left, lower, right, upper)
    boxLL = (0, 0, int(pxlLoc[0]), int(pxlLoc[1]))
    newSize = [int(pxlLoc[0]), int(pxlLoc[1])]
    eyeSubLL = img3.resize(newSize)
    
    boxLR = (int(pxlLoc[0]), 0, SzX,int(pxlLoc[1]))
    newSize = [int(SzX - pxlLoc[0]), int(pxlLoc[1])]
    eyeSubLR = img4.resize(newSize)
    

    img1.paste(eyeSubUL, boxUL)
    img1.paste(eyeSubUR, boxUR)
    img1.paste(eyeSubLL, boxLL)
    img1.paste(eyeSubLR, boxLR)

    
    
    eyeOutFileName = 'rndEyeFourExp'
    rndEyeFourFull = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(rndEyeFourFull, img1)
    
    return



if (0):

    eyeExp1 = jpgSrcDir+'bananasloth0036.jpg'
    eyeExp2 = jpgSrcDir+'bananasloth0039.jpg'
    eyeExp3 = jpgSrcDir+'microscopic00652.jpg'
    eyeExp4 = jpgSrcDir+'microscopic00640.jpg'
    
    
    eye1 = misc.imread(eyeExp1)
    eye2 = misc.imread(eyeExp2)
    eye3 = misc.imread(eyeExp3)
    eye4 = misc.imread(eyeExp4)
    #eye1img = Image.fromarray(eye1)
    eye1img = Image.fromarray(eye1)
    eye2img = Image.fromarray(eye2)
    eye3img = Image.fromarray(eye3)
    eye4img = Image.fromarray(eye4)
    
    #eye1imgCpy = eye2img.copy()
    #eyeImgCpy = eye2img.copy()
    
    
    
    #ctrl = 1
    
    rndPxlLoc = randomPxlLoc(SzX, SzY)
    
    eyeRndFour(eye1img, eye2img, eye3img, eye4img, SzX, SzY, rndPxlLoc)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



def eyeBlendFour(img1, img2, img3, img4, SzX, SzY, pxlLoc):
    ''' generates composite 4 frame resized images
        sub dimensions defined by pxlLoc [x, y]
        img1 - img4 PIL images
        pxlLoc: tuple of x and y coordinates '''
    
    imgDim = (SzX, SzY)
    b4Img = Image.new('RGB', imgDim)  # e.g. ('RGB', (640, 480))    
    #alpha = np.linspace(0, 1, nTrail)
    
    # coordinates = (left, lower, right, upper)
    boxUL = (0, int(pxlLoc[1]), int(pxlLoc[0]), SzY)
    newSize = [int(pxlLoc[0]), int(SzY - pxlLoc[1])]
    
    eyeSubUL1 = img1.resize(newSize)
    eyeSubUL2 = img4.resize(newSize)
    
    eyeSubUL = Image.blend(eyeSubUL1, eyeSubUL2, 0.5)
    eyeSubUL = ImageOps.autocontrast(eyeSubUL, cutoff=0)
    
    #pdb.set_trace()    
    
    boxUR = (int(pxlLoc[0]), int(pxlLoc[1]), SzX, SzY)
    newSize = [int(SzX - pxlLoc[0]), int(SzY - pxlLoc[1])]
    
    eyeSubUR1 = img2.resize(newSize)
    eyeSubUR2 = img4.resize(newSize)
    
    eyeSubUR = Image.blend(eyeSubUR1, eyeSubUR2, 0.5)
    eyeSubUR = ImageOps.autocontrast(eyeSubUR, cutoff=0)


    
     # coordinates = (left, lower, right, upper)
    boxLL = (0, 0, int(pxlLoc[0]), int(pxlLoc[1]))
    newSize = [int(pxlLoc[0]), int(pxlLoc[1])]
    
    eyeSubLL1 = img1.resize(newSize)
    eyeSubLL2 = img3.resize(newSize)
    
    eyeSubLL = Image.blend(eyeSubLL1, eyeSubLL2, 0.5)
    eyeSubLL = ImageOps.autocontrast(eyeSubLL, cutoff=0)    
    
    
    
    boxLR = (int(pxlLoc[0]), 0, SzX,int(pxlLoc[1]))
    newSize = [int(SzX - pxlLoc[0]), int(pxlLoc[1])]
    
    eyeSubLR1 = img2.resize(newSize)
    eyeSubLR2 = img3.resize(newSize)
    
    eyeSubLR = Image.blend(eyeSubLR1, eyeSubLR2, 0.5)
    eyeSubLR = ImageOps.autocontrast(eyeSubLR, cutoff=0)    



    b4Img.paste(eyeSubUL, boxUL)
    b4Img.paste(eyeSubUR, boxUR)
    b4Img.paste(eyeSubLL, boxLL)
    b4Img.paste(eyeSubLR, boxLR)

    
    
    eyeOutFileName = 'rndEyeFourExp'
    rndEyeFourFull = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(rndEyeFourFull, b4Img)
    
    return b4Img



if (0):

    eyeExp1 = jpgSrcDir+'bananasloth0036.jpg'
    eyeExp2 = jpgSrcDir+'bananasloth0039.jpg'
    eyeExp3 = jpgSrcDir+'microscopic00652.jpg'
    eyeExp4 = jpgSrcDir+'microscopic00640.jpg'
    
    
    eye1 = misc.imread(eyeExp1)
    eye2 = misc.imread(eyeExp2)
    eye3 = misc.imread(eyeExp3)
    eye4 = misc.imread(eyeExp4)
    #eye1img = Image.fromarray(eye1)
    eye1img = Image.fromarray(eye1)
    eye2img = Image.fromarray(eye2)
    eye3img = Image.fromarray(eye3)
    eye4img = Image.fromarray(eye4)
    
    #eye1imgCpy = eye2img.copy()
    #eyeImgCpy = eye2img.copy()
    
    
    
    #ctrl = 1
    
    rndPxlLoc = randomPxlLoc(int(SzX/2), int(SzY/2))
    
    eyeB4Img = eyeBlendFour(eye1img, eye2img, eye3img, eye4img, SzX, SzY, rndPxlLoc)
    
    
    eyeOutFileName = 'eyeB4ImgTest'
    eyeB4ImgFull = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(eyeB4ImgFull, eyeB4Img)

# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



#def eyeMirrorHV4(img):
#    ''' generates composite 4 frame resized images
#        sub dimensions defined by pxlLoc [x, y]
#        img1 - img4 PIL images
#        pxlLoc: tuple of x and y coordinates '''
#    
#    
#    SzX = img.size[0]
#    SzY = img.size[1]  
#    
#    #alpha = np.linspace(0, 1, nTrail)
#    
#    # coordinates = (left, lower, right, upper)
#    imgDim = (2*SzX, 2*SzY)
#    
#    newImg = Image.new('RGB', imgDim)  # e.g. ('RGB', (640, 480))
#    
#
#    #boxUL = (0, SzY, SzX-1, 2*SzY-1)
#    boxUL = (0, SzY, SzX, 2*SzY)
#    boxUR = (SzX, SzY, 2*SzX, 2*SzY) 
#    boxLL = (0, 0, SzX, SzY)
#    boxLR = (SzX, 0, 2*SzX, SzY)
#    
#
#    eyeSubUL = ImageOps.flip(img)
#    eyeSubUL = ImageOps.mirror(eyeSubUL)
#    eyeSubUR = ImageOps.flip(img)
#    eyeSubLL = ImageOps.mirror(img)
#
#
#    #pdb.set_trace()
#
#    newImg.paste(eyeSubUL, boxUL)
#    newImg.paste(eyeSubUR, boxUR)
#    newImg.paste(eyeSubLL, boxLL)
#    newImg.paste(img, boxLR)
#
#    
#    eyeOutFileName = 'eyeMirrorHV4'
#    eyeMirrorHV4Full = imgOutDir+eyeOutFileName+'.jpg'
#    #misc.imsave(eyeBoxFull, eye1imgCpy)
#    misc.imsave(eyeMirrorHV4Full, newImg)
#    
#    return
#
#
#
#eyeExp1 = jpgSrcDir+'bananasloth0036.jpg'
#eyeExp2 = jpgSrcDir+'bananasloth0039.jpg'
#eyeExp3 = jpgSrcDir+'microscopic00652.jpg'
#eyeExp4 = jpgSrcDir+'microscopic00640.jpg'
#
#
#eye1 = misc.imread(eyeExp1)
#
##eye1img = Image.fromarray(eye1)
#eye1img = Image.fromarray(eye1)
#
#
##ctrl = 1
#
#eyeMirrorHV4(eye1img)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



def eyeMirrorHV4(img, SzX, SzY, ctrl='UL'):
    ''' generates composite 4 frame resized images
        SzX, SzY: output image dimentions
        img: PIL image object
        ctrl: { UL, UR, LL, LR} - selects base quadrant '''
    
    
    if not(ctrl=='UL' or ctrl=='UR' or ctrl=='LL' or ctrl=='LR'):
        print('ERROR: ctrl must be {UL, UR, LL, LR} <string>')
        return
    
    imgSzX = img.size[0]
    imgSzY = img.size[1]
    
    imgDim = (SzX, SzY)
    
    if (SzX == 2*imgSzX and SzY == 2*imgSzY):
        subDim = (imgSzX, imgSzY)
        imgX = img
    else:
        subDim = (int(SzX/2), int(SzY/2))
        imgX = img.resize(subDim)
    
    
    mirrorImg = Image.new('RGB', imgDim)  # e.g. ('RGB', (640, 480))
    

    #boxUL = (0, SzY, SzX-1, 2*SzY-1)
    boxUL = (0, subDim[1], subDim[0], imgDim[1])
    boxUR = (subDim[0], subDim[1], imgDim[0], imgDim[1]) 
    boxLL = (0, 0, subDim[0], subDim[1])
    boxLR = (subDim[0], 0, imgDim[0], subDim[1])
    

    if ctrl == 'LL':
        eyeSubUL = imgX
        eyeSubUR = ImageOps.mirror(imgX)
        eyeSubLL = ImageOps.flip(imgX)
        eyeSubLR = ImageOps.flip(imgX)
        eyeSubLR = ImageOps.mirror(eyeSubLR)        
        
        
    if ctrl == 'LR':
        eyeSubUL = ImageOps.mirror(imgX)
        eyeSubUR = imgX
        eyeSubLL = ImageOps.flip(imgX)
        eyeSubLL = ImageOps.mirror(eyeSubLL)
        eyeSubLR = ImageOps.flip(imgX)
        
        
    if ctrl == 'UL':
        eyeSubUL = ImageOps.flip(imgX)
        eyeSubUR = ImageOps.flip(imgX)
        eyeSubUR = ImageOps.mirror(eyeSubUR)
        eyeSubLL = imgX
        eyeSubLR = ImageOps.mirror(imgX)


    if ctrl == 'UR':
        eyeSubUL = ImageOps.flip(imgX)
        eyeSubUL = ImageOps.mirror(eyeSubUL)
        eyeSubUR = ImageOps.flip(imgX)
        eyeSubLL = ImageOps.mirror(imgX)
        eyeSubLR = imgX


    #pdb.set_trace()

    mirrorImg.paste(eyeSubUL, boxUL)
    mirrorImg.paste(eyeSubUR, boxUR)
    mirrorImg.paste(eyeSubLL, boxLL)
    mirrorImg.paste(eyeSubLR, boxLR)

    
    #eyeOutFileName = 'eyeMirrorHV4'
    #eyeMirrorHV4Full = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeMirrorHV4Full, mirrorImg)
    
    return mirrorImg


if (0):

    eyeExp1 = jpgSrcDir+'bananasloth0036.jpg'
    eyeExp2 = jpgSrcDir+'bananasloth0039.jpg'
    eyeExp3 = jpgSrcDir+'microscopic00652.jpg'
    eyeExp4 = jpgSrcDir+'microscopic00640.jpg'
    eyeExp5 = jpgSrcDir+'microscopic0042.jpg'
    
    eye1 = misc.imread(eyeExp2)
    eye5 = misc.imread(eyeExp5)
    
    
    #eye1img = Image.fromarray(eye1)
    eye1img = Image.fromarray(eye1)
    eye5img = Image.fromarray(eye5)
    
    #ctrl = 1
    
    mirrorHV4reScl = eyeMirrorHV4(eye5img, SzX, SzY, ctrl='LR')
    mirrorHV4noScl = eyeMirrorHV4(eye5img, 2*SzX, 2*SzY, ctrl='LL')
    
    
    eyeOutFileName = 'eye1img'
    eye1imgFull = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(eye1imgFull, eye1img)
    
    eyeOutFileName = 'mirrorHV4reSclTest'
    mirrorHV4reSclFull = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(mirrorHV4reSclFull, mirrorHV4reScl)
    
    eyeOutFileName = 'mirrorHV4noSclTest'
    mirrorHV4noSclFull = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(mirrorHV4noSclFull, mirrorHV4noScl)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



#def eyeRollFour(img1, SzX, SzY, ctrl):
#    ''' generates composite 4 frame resized images
#        sub dimensions defined by pxlLoc [x, y]
#        img1 - img4 PIL images
#        pxlLoc: tuple of x and y coordinates '''
#    
#    
#    #alpha = np.linspace(0, 1, nTrail)
#    
#    # coordinates = (left, lower, right, upper)
#    boxUL = (0, int(SzY/2), int(SzX/2), SzY)
#    newSize = [int(pxlLoc[0]), int(SzY - pxlLoc[1])]
#    eyeSubUL = img1.resize(newSize)
#    
#    #pdb.set_trace()    
#    
#    boxUR = (int(SzX/2), int(SzY/2), SzX, SzY)
#    newSize = [int(SzX - pxlLoc[0]), int(SzY - pxlLoc[1])]
#    eyeSubUR = img2.resize(newSize)
#    
#     # coordinates = (left, lower, right, upper)
#    boxLL = (0, 0, int(SzX/2), int(SzY/2))
#    newSize = [int(pxlLoc[0]), int(pxlLoc[1])]
#    eyeSubLL = img3.resize(newSize)
#    
#    boxLR = (int(SzX/2), 0, SzX, int(SzY/2))
#    newSize = [int(SzX - pxlLoc[0]), int(pxlLoc[1])]
#    eyeSubLR = img4.resize(newSize)
#    
#
#    img1.paste(eyeSubUL, boxUL)
#    img1.paste(eyeSubUR, boxUR)
#    img1.paste(eyeSubLL, boxLL)
#    img1.paste(eyeSubLR, boxLR)
#
#    
#    
#    eyeOutFileName = 'rndEyeFourExp'
#    rndEyeFourFull = imgOutDir+eyeOutFileName+'.jpg'
#    #misc.imsave(eyeBoxFull, eye1imgCpy)
#    misc.imsave(rndEyeFourFull, img1)
#    
#    return
#
#
#
#eyeExp1 = jpgSrcDir+'bananasloth0036.jpg'
#eyeExp2 = jpgSrcDir+'bananasloth0039.jpg'
#eyeExp3 = jpgSrcDir+'microscopic00652.jpg'
#eyeExp4 = jpgSrcDir+'microscopic00640.jpg'
#
#
#eye1 = misc.imread(eyeExp1)
#eye2 = misc.imread(eyeExp2)
#eye3 = misc.imread(eyeExp3)
#eye4 = misc.imread(eyeExp4)
##eye1img = Image.fromarray(eye1)
#eye1img = Image.fromarray(eye1)
#eye2img = Image.fromarray(eye2)
#eye3img = Image.fromarray(eye3)
#eye4img = Image.fromarray(eye4)
#
##eye1imgCpy = eye2img.copy()
##eyeImgCpy = eye2img.copy()
#
#
#
#ctrl = 100
#
#
#eyeRollFour(eye1img, SzX, SzY, ctrl)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



def eyeBox1(img1, img2, boxSzX, boxSzY, pxlLoc, alpha):
    ''' alpha blends random box from img2 onto img1
        sub dimensions defined by pxlLoc [x, y]
        img1: img4 PIL images
        ctrl: toggle background - 0 = no background img (black), 1 = use img1 '''


    if img1.size != img2.size:
        print('ERROR: img1, img2 must be same Dim')
        return        

   
    SzX = img1.size[0]
    SzY = img1.size[1]
    #imgDim = [SzX, SzY]
        

    
    subDim_L = int(pxlLoc[0] - boxSzX//2)
    if subDim_L < 0:
        subDim_L = 0
    subDim_R = int(pxlLoc[0] + boxSzX//2)
    if subDim_R > SzX:
        subDim_R = SzX
    subDim_B = int(pxlLoc[1] - boxSzY//2)
    if subDim_B < 0:
        subDim_B = 0
    subDim_T = int(pxlLoc[0] + boxSzX//2)
    if subDim_T > SzY:
        subDim_T = SzY
    
    
    # coordinates = (left, lower, right, upper)
    boxCC = (subDim_L, subDim_B, subDim_R, subDim_T)
    
    if alpha == 1:
        eyeSubCC = img2.copy().crop(boxCC)       


    else:        
        eyeSub1 = img1.copy().crop(boxCC)
        eyeSub2 = img2.copy().crop(boxCC)    

        eyeSubCC = Image.blend(eyeSub1, eyeSub2, alpha)
        eyeSubCC = ImageOps.autocontrast(eyeSubCC, cutoff=0)


    #pdb.set_trace()

    # left frame
    img1.paste(eyeSubCC, boxCC)

    
    return img1


if (0):

    eyeExp1 = jpgSrcDir+'bananasloth0036.jpg'
    eyeExp2 = jpgSrcDir+'bananasloth0039.jpg'
    eyeExp3 = jpgSrcDir+'microscopic00652.jpg'
    eyeExp4 = jpgSrcDir+'microscopic00640.jpg'
    eyeExp5 = jpgSrcDir+'microscopic0042.jpg'
    
    eye1 = misc.imread(eyeExp2)
    eye2 = misc.imread(eyeExp5)
    
    
    #eye1img = Image.fromarray(eye1)
    eye1img = Image.fromarray(eye1)
    eye2img = Image.fromarray(eye2)
    
    #eye1img = Image.fromarray(eye1)
    eye1img = Image.fromarray(eye1)
    eye2img = Image.fromarray(eye2)



    rndPxlLoc = randomPxlLoc(SzX, SzY)

    boxSzX = SzX//3
    boxSzY = SzY//3
    alpha = 0.5

    eyeBox1test = eyeBox1(eye1img, eye2img, boxSzX, boxSzY, rndPxlLoc, alpha)    
    
    eyeOutFileName = 'eyeBox1test'
    eyeBox1testFull = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(eyeBox1testFull,eyeBox1test)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



def eyePhiHorizFrame1(img1, img2, img3, ctrl=1):
    ''' generates composite 4 frame resized images
        sub dimensions defined by pxlLoc [x, y]
        img1: img4 PIL images
        ctrl: toggle background - 0 = no background img (black), 1 = use img1 '''


    if (img1.size != img2.size != img3.size):
        print('ERROR: img1, img2, img3 must be same Dim')
        return        

    phi = 1.6180339887
    phiInv = 0.6180339887
    oneMphiInv = 0.38196601129
   
    SzX = img1.size[0]
    SzY = img1.size[1]
    imgDim = [SzX, SzY]
    
    dimYsub1 = int(SzY * oneMphiInv)        # same as horiz2_B
    dimYsub2 = int(SzY * phiInv)    # same as horiz3_B
    
    dimXsub1 = int(SzX * oneMphiInv)        # same as vert2_B
    dimXsub2 = int(SzX - (SzX * oneMphiInv))    # same as vert3_B        



    # toggle background on/off
    if ctrl == 0:
        phiHorizFrameImg = Image.new('RGB', imgDim)  # e.g. ('RGB', (640, 480)) 
    else:    
        phiHorizFrameImg = img1.copy()   
   
    
    
    # coordinates = (left, lower, right, upper)
#    boxL = (0, 0, dimH_L, SzY)
#    boxC = (dimH_L, 0, dimH_R, SzY)
#    boxR = (dimH_R, 0, dimH_R, SzY)    
#    
#    boxC_B = (dimH_L, dimC_T, dimH_R, SzY)
#    boxC_T = (dimH_L, 0, dimH_R, dimC_B)    
  
    boxYsub1 = (0, 0, SzX, dimYsub1)
    
    boxYsub2 = (0, dimYsub1, SzX, dimYsub2)
    
    boxYsub3 = (0, dimYsub2, SzX, SzY)
    
    boxXsub1 = (0, dimYsub2, dimXsub1, SzY)
    boxXsub2 = (dimXsub1, dimYsub2, dimXsub2, SzY)
    boxXsub3 = (dimXsub2, dimYsub2, SzX, SzY)    


    #eyeSub1 = img1.copy().crop(boxYsub1)
    eyeSub2 = img2.copy().crop(boxYsub2)
    #eyeSub3 = img3.copy().crop(boxYsub3)        
    
    eyeXSub1 = img1.copy().crop(boxXsub1)  
    eyeXSub2 = img2.copy().crop(boxXsub2)
    eyeXSub3 = img3.copy().crop(boxXsub3)
    
    
    #subDimCC = (dimCC_R - int(dimH_L/2), dimCC_T - int(dimC_B/2))

    #eyeSubCC = img1.copy().resize(subDimCC)
    #eyeSubCC2 = img2.copy().resize(subDimCC)
    #eyeSubCC = Image.blend(eyeSubCC, eyeSubCC2, 0.5)
    #eyeSubCC = ImageOps.autocontrast(eyeSubCC, cutoff=0)


    #sub1SzX = eyeSub2.size[0]
    #sub1SzY = eyeSub2.size[1]
    
    #print('sub1 size X = '+str(sub1SzX))
    #print('sub1 size Y = '+str(sub1SzY))    

    #print('boxYsub2 = '+str(boxYsub2))



    # left frame
    phiHorizFrameImg.paste(eyeSub2, boxYsub2)
    #phiHorizFrameImg.paste(eyeSub3, boxYsub3)
    
    phiHorizFrameImg.paste(eyeXSub1, boxXsub1)
    phiHorizFrameImg.paste(eyeXSub2, boxXsub2)    
    phiHorizFrameImg.paste(eyeXSub3, boxXsub3)
    

    return phiHorizFrameImg


if (0):

    eyeExp1 = jpgSrcDir+'bananasloth000011.jpg'
    eyeExp2 = jpgSrcDir+'odmkDmtpatra001097.jpg'
    eyeExp3 = jpgSrcDir+'bSlothCultLife000929.jpg'
    eyeExp4 = jpgSrcDir+'bananasloth0036.jpg'
    
    
    eye1 = misc.imread(eyeExp1)
    eye2 = misc.imread(eyeExp2)
    eye3 = misc.imread(eyeExp3)    
    
    #eye1img = Image.fromarray(eye1)
    eye1img = Image.fromarray(eye1)
    eye2img = Image.fromarray(eye2)
    eye3img = Image.fromarray(eye3)    
    
    
    #ctrl = 1
    
    phiHorizFrame = eyePhiHorizFrame1(eye1img, eye2img, eye3img)
    
    
    eyeOutFileName = 'phiHorizFrameTest'
    phiHorizFrameFull = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(phiHorizFrameFull, phiHorizFrame)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



def eyePhiFrame1(img1, img2, ctrl=1):
    ''' generates composite 4 frame resized images
        sub dimensions defined by pxlLoc [x, y]
        img1: img4 PIL images
        ctrl: toggle background - 0 = no background img (black), 1 = use img1 '''


    if img1.size != img2.size:
        print('ERROR: img1, img2 must be same Dim')
        return        

    phi = 1.6180339887
    phiInv = 0.6180339887
    oneMphiInv = 0.38196601129
   
    SzX = img1.size[0]
    SzY = img1.size[1]
    imgDim = [SzX, SzY]
    
    # toggle background on/off
    if ctrl == 0:
        phiFrameImg = Image.new('RGB', imgDim)  # e.g. ('RGB', (640, 480)) 
    else:    
        phiFrameImg = img1.copy()    
    
    dimH_L = int(SzX*oneMphiInv)
    dimH_R = int(SzX*phiInv)
    
    dimC_B = int(SzY*oneMphiInv)
    dimC_T = int(SzY*phiInv)
    
    dimCC_L = int(dimH_L/2)
    dimCC_R = SzX - int(dimH_L/2)
    dimCC_B = int(dimC_B/2)
    dimCC_T = SzY - int(dimC_B/2)    
    
    
    # coordinates = (left, lower, right, upper)
#    boxL = (0, 0, dimH_L, SzY)
#    boxC = (dimH_L, 0, dimH_R, SzY)
#    boxR = (dimH_R, 0, dimH_R, SzY)    
#    
#    boxC_B = (dimH_L, dimC_T, dimH_R, SzY)
#    boxC_T = (dimH_L, 0, dimH_R, dimC_B)    
  
    boxCC = (dimCC_L, dimCC_B, dimCC_R, dimCC_T)
    

    boxL_B = (0, dimC_T, dimH_L, SzY)
    boxL_T = (0, 0, dimH_L, dimC_B)
    
    boxR_B = (dimH_R, dimC_T, SzX, SzY)
    boxR_T = (dimH_R, 0, SzX, dimC_B)
    
    
#    eyeSub = eyeSub.resize(subDim)
#    eyeSub = eyeSub.crop(box)
#    eyeSub = ImageOps.mirror(eyeSub)
#    eyeSub = ImageOps.flip(eyeSub)


    eyeSubLT = img1.copy().crop(boxL_T)
    eyeSubLB = img2.copy().crop(boxL_B)
    
    #eyeSubTC = img2.resize(subDim)
    #eyeSubBC = img2.copy().crop(boxC_B)
    

    eyeSubRT = img2.copy().crop(boxR_T)
    eyeSubRB = img1.copy().crop(boxR_B) 


    subDimCC = (dimCC_R - int(dimH_L/2), dimCC_T - int(dimC_B/2))

    eyeSubCC = img1.copy().resize(subDimCC)
    eyeSubCC2 = img2.copy().resize(subDimCC)
    eyeSubCC = Image.blend(eyeSubCC, eyeSubCC2, 0.5)
    eyeSubCC = ImageOps.autocontrast(eyeSubCC, cutoff=0)


    #pdb.set_trace()

    # left frame
    phiFrameImg.paste(eyeSubLT, boxL_T)
    phiFrameImg.paste(eyeSubLB, boxL_B)
    
    # rigth frame
    phiFrameImg.paste(eyeSubRT, boxR_T)
    phiFrameImg.paste(eyeSubRB, boxR_B)

    # center frame
    #phiFrameImg.paste(eyeSubTC, boxC_T)
    #phiFrameImg.paste(eyeSubBC, boxC_B)
    
    # center center
    phiFrameImg.paste(eyeSubCC, boxCC)

    
    #eyeOutFileName = 'eyeMirrorHV4'
    #eyeMirrorHV4Full = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeMirrorHV4Full, mirrorImg)
    
    return phiFrameImg


if (0):

    eyeExp1 = jpgSrcDir+'bananasloth0036.jpg'
    eyeExp2 = jpgSrcDir+'bananasloth0039.jpg'
    eyeExp3 = jpgSrcDir+'microscopic00652.jpg'
    eyeExp4 = jpgSrcDir+'microscopic00640.jpg'
    
    
    eye1 = misc.imread(eyeExp1)
    eye2 = misc.imread(eyeExp2)
    
    #eye1img = Image.fromarray(eye1)
    eye1img = Image.fromarray(eye1)
    eye2img = Image.fromarray(eye2)
    
    
    #ctrl = 1
    
    phiFrame = eyePhiFrame1(eye1img, eye2img)
    
    
    eyeOutFileName = 'phiFrameTest'
    phiFrameFull = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(phiFrameFull,phiFrame)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



def eyePhiFrame2(img1, img2, ctrl='UL'):
    ''' generates subframes increasing by fibonacci sequence
        sub dimensions defined by pxlLoc [x, y]
        img1: img4 PIL images
        ctrl: toggle background - 0 = no background img (black), 1 = use img1 '''


    if not(ctrl=='UL' or ctrl=='UR' or ctrl=='LL' or ctrl=='LR'):
        print('ERROR: ctrl must be {UL, UR, LL, LR} <string>')
        return

    if img1.size != img2.size:
        print('ERROR: img1, img2 must be same Dim')
        return        

    #phi = 1.6180339887
    #phiInv = 0.6180339887
    #oneMphiInv = 0.38196601129
   
    SzX = img1.size[0]
    SzY = img1.size[1]
    imgDim = [SzX, SzY]
    
    # coordinates = (left, lower, right, upper)

    if ctrl == 'LL':
        boxQTR = (0, int(SzY/2), int(SzX/2), SzY)        
    elif ctrl == 'LR':
        boxQTR = (int(SzX/2), int(SzY/2), SzX, SzY)        
    elif ctrl == 'UL':
        boxQTR = (0, 0, int(SzX/2), int(SzY/2))
    elif ctrl == 'UR':
        boxQTR = (int(SzX/2), 0, SzX, int(SzY/2))
        
        
    phiFrameImg = img1.copy()
    phiSubImg1 = img1.copy().crop(boxQTR)
    phiSubImg1 = phiSubImg1.resize(imgDim)
    phiSubImg2 = img2.copy().crop(boxQTR)
    phiSubImg2 = phiSubImg2.resize(imgDim)
    
    
#    eyeOutFileName = 'phiFrameImg'
#    phiFrameImgFull = imgOutDir+eyeOutFileName+'.jpg'
#    #misc.imsave(eyeBoxFull, eye1imgCpy)
#    misc.imsave(phiFrameImgFull,phiFrameImg)
#    
#    eyeOutFileName = 'phiSubImg1'
#    phiSubImg1Full = imgOutDir+eyeOutFileName+'.jpg'
#    #misc.imsave(eyeBoxFull, eye1imgCpy)
#    misc.imsave(phiSubImg1Full,phiSubImg1)
#
#    eyeOutFileName = 'phiSubImg2'
#    phiSubImg2Full = imgOutDir+eyeOutFileName+'.jpg'
#    #misc.imsave(eyeBoxFull, eye1imgCpy)
#    misc.imsave(phiSubImg2Full,phiSubImg2)
   
      
    dimXarray = []
    for xx in odmkFibonacci(SzX):
        if xx > 8:
            dimXarray.append(int(xx))
            #print(str(int(xx)))
            
    dimYarray = []
    for yy in dimXarray:
        dimYarray.append(int(yy*SzY/SzX))
        #print(str(int(yy*SzY/SzX)))
        
    
    #pdb.set_trace()
   
    for bb in range(len(dimXarray)):
        
        if bb%2==1:
            boxBB = (0, 0, dimXarray[len(dimXarray) - bb - 1], dimYarray[len(dimYarray) - bb - 1])
            eyeSub = phiSubImg1.copy().crop(boxBB)
            phiFrameImg.paste(eyeSub, boxBB)
        else:
            boxBB = (0, 0, dimXarray[len(dimXarray) - bb - 1], dimYarray[len(dimYarray) - bb - 1])
            eyeSub = phiSubImg2.copy().crop(boxBB)
            phiFrameImg.paste(eyeSub, boxBB)
           
    
    return phiFrameImg


if (0): 

    eyeExp1 = jpgSrcDir+'keisolomon0017.jpg'
    eyeExp2 = jpgSrcDir+'ksolomon0024.jpg'
    eyeExp3 = jpgSrcDir+'microscopic0052.jpg'
    eyeExp4 = jpgSrcDir+'odmkFxion0042.jpg'

    
    
    eye1 = misc.imread(eyeExp3)
    eye2 = misc.imread(eyeExp4)
    
    
    #eye1img = Image.fromarray(eye1)
    eye1img = Image.fromarray(eye1)
    eye2img = Image.fromarray(eye2)
    
    ctrl = 'UR'
    
    phiFrame2 = eyePhiFrame2(eye1img, eye2img, ctrl)
    
    
    eyeOutFileName = 'phiFrame2Test'
    phiFrame2Full = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(phiFrame2Full,phiFrame2)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



def eyePhiFrame3(img1, img2, ctrl='UL'):
    ''' generates subframes increasing by fibonacci sequence
        sub dimensions defined by pxlLoc [x, y]
        img1: img4 PIL images
        ctrl: toggle background - 0 = no background img (black), 1 = use img1 '''


    if not(ctrl=='UL' or ctrl=='UR' or ctrl=='LL' or ctrl=='LR'):
        print('ERROR: ctrl must be {UL, UR, LL, LR} <string>')
        return

    if img1.size != img2.size:
        print('ERROR: img1, img2 must be same Dim')
        return        

    #phi = 1.6180339887
    #phiInv = 0.6180339887
    #oneMphiInv = 0.38196601129
   
    SzX = img1.size[0]
    SzY = img1.size[1]
    imgDim = [SzX, SzY]
    
    # coordinates = (left, lower, right, upper)

    if ctrl == 'LL':
        boxQTR = (0, int(SzY/2), int(SzX/2), SzY)        
    elif ctrl == 'LR':
        boxQTR = (int(SzX/2), int(SzY/2), SzX, SzY)        
    elif ctrl == 'UL':
        boxQTR = (0, 0, int(SzX/2), int(SzY/2))
    elif ctrl == 'UR':
        boxQTR = (int(SzX/2), 0, SzX, int(SzY/2))
        
        
    phiFrameImg = img1.copy()
    phiSubImg1 = img1.copy().crop(boxQTR)
    phiSubImg1 = phiSubImg1.resize(imgDim)
    phiSubImg2 = img2.copy().crop(boxQTR)
    phiSubImg2 = phiSubImg2.resize(imgDim)
    
    
#    eyeOutFileName = 'phiFrameImg'
#    phiFrameImgFull = imgOutDir+eyeOutFileName+'.jpg'
#    #misc.imsave(eyeBoxFull, eye1imgCpy)
#    misc.imsave(phiFrameImgFull,phiFrameImg)
#    
#    eyeOutFileName = 'phiSubImg1'
#    phiSubImg1Full = imgOutDir+eyeOutFileName+'.jpg'
#    #misc.imsave(eyeBoxFull, eye1imgCpy)
#    misc.imsave(phiSubImg1Full,phiSubImg1)
#
#    eyeOutFileName = 'phiSubImg2'
#    phiSubImg2Full = imgOutDir+eyeOutFileName+'.jpg'
#    #misc.imsave(eyeBoxFull, eye1imgCpy)
#    misc.imsave(phiSubImg2Full,phiSubImg2)
   
      
    dimXarray = []
    for xx in odmkFibonacci(SzX):
        if xx > 8:
            dimXarray.append(int(xx))
            #print(str(int(xx)))
            
    dimYarray = []
    for yy in dimXarray:
        dimYarray.append(int(yy*SzY/SzX))
        #print(str(int(yy*SzY/SzX)))       
        
    dimZarray = np.linspace(0.0, 0.23, len(dimXarray))
        #print(str(int(yy*SzY/SzX)))        
        
    
    #pdb.set_trace()
   
    for bb in range(len(dimXarray)):
        
        if bb%2==1:
            boxBB = (0, 0, dimXarray[len(dimXarray) - bb - 1], dimYarray[len(dimYarray) - bb - 1])
            eyeSub1 = phiSubImg1.copy().crop(boxBB)
            eyeSub2 = phiSubImg2.copy().crop(boxBB)
            
            eyeSub = Image.blend(eyeSub1, eyeSub2, dimZarray[len(dimZarray) - bb - 1])
            
            phiFrameImg.paste(eyeSub, boxBB)
        else:
            boxBB = (0, 0, dimXarray[len(dimXarray) - bb - 1], dimYarray[len(dimYarray) - bb - 1])
            eyeSub1 = phiSubImg1.copy().crop(boxBB)
            eyeSub2 = phiSubImg2.copy().crop(boxBB)            
            
            eyeSub = Image.blend(eyeSub2, eyeSub1, dimZarray[len(dimZarray) - bb - 1])
            
            phiFrameImg.paste(eyeSub, boxBB)
           
    
    return phiFrameImg


if (0): 

    eyeExp1 = jpgSrcDir+'keisolomon0017.jpg'
    eyeExp2 = jpgSrcDir+'ksolomon0024.jpg'
    eyeExp3 = jpgSrcDir+'microscopic0052.jpg'
    eyeExp4 = jpgSrcDir+'odmkFxion0042.jpg'

    
    
    eye1 = misc.imread(eyeExp3)
    eye2 = misc.imread(eyeExp4)
    
    
    #eye1img = Image.fromarray(eye1)
    eye1img = Image.fromarray(eye1)
    eye2img = Image.fromarray(eye2)
    
    ctrl = 'UR'
    
    phiFrame3 = eyePhiFrame3(eye1img, eye2img, ctrl)
    
    
    eyeOutFileName = 'phiFrame3Test'
    phiFrame3Full = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(phiFrame3Full,phiFrame3)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //




def eyeSinBlend(img1, img2, cycles, ctrl=1):
    ''' generates composite 4 frame resized images
        sub dimensions defined by pxlLoc [x, y]
        img1: img4 PIL images
        ctrl: toggle background - 0 = no background img (black), 1 = use img1 '''


    if (img1.size != img2.size != img3.size):
        print('ERROR: img1, img2, img3 must be same Dim')
        return        

    phi = 1.6180339887
    phiInv = 0.6180339887
    oneMphiInv = 0.38196601129
   
    SzX = img1.size[0]
    SzY = img1.size[1]
    imgDim = [SzX, SzY]
    
    dimYsub1 = int(SzY * oneMphiInv)        # same as horiz2_B
    dimYsub2 = int(SzY * phiInv)    # same as horiz3_B
    
    dimXsub1 = int(SzX * oneMphiInv)        # same as vert2_B
    dimXsub2 = int(SzX - (SzX * oneMphiInv))    # same as vert3_B        



    # toggle background on/off
    if ctrl == 0:
        phiHorizFrameImg = Image.new('RGB', imgDim)  # e.g. ('RGB', (640, 480)) 
    else:    
        phiHorizFrameImg = img1.copy()   
   
    
    
    # coordinates = (left, lower, right, upper)
#    boxL = (0, 0, dimH_L, SzY)
#    boxC = (dimH_L, 0, dimH_R, SzY)
#    boxR = (dimH_R, 0, dimH_R, SzY)    
#    
#    boxC_B = (dimH_L, dimC_T, dimH_R, SzY)
#    boxC_T = (dimH_L, 0, dimH_R, dimC_B)    
  
    boxYsub1 = (0, 0, SzX, dimYsub1)
    
    boxYsub2 = (0, dimYsub1, SzX, dimYsub2)
    
    boxYsub3 = (0, dimYsub2, SzX, SzY)
    
    boxXsub1 = (0, dimYsub2, dimXsub1, SzY)
    boxXsub2 = (dimXsub1, dimYsub2, dimXsub2, SzY)
    boxXsub3 = (dimXsub2, dimYsub2, SzX, SzY)    


    #eyeSub1 = img1.copy().crop(boxYsub1)
    eyeSub2 = img2.copy().crop(boxYsub2)
    #eyeSub3 = img3.copy().crop(boxYsub3)        
    
    eyeXSub1 = img1.copy().crop(boxXsub1)  
    eyeXSub2 = img2.copy().crop(boxXsub2)
    eyeXSub3 = img3.copy().crop(boxXsub3)
    
    
    #subDimCC = (dimCC_R - int(dimH_L/2), dimCC_T - int(dimC_B/2))

    #eyeSubCC = img1.copy().resize(subDimCC)
    #eyeSubCC2 = img2.copy().resize(subDimCC)
    #eyeSubCC = Image.blend(eyeSubCC, eyeSubCC2, 0.5)
    #eyeSubCC = ImageOps.autocontrast(eyeSubCC, cutoff=0)


    #sub1SzX = eyeSub2.size[0]
    #sub1SzY = eyeSub2.size[1]
    
    #print('sub1 size X = '+str(sub1SzX))
    #print('sub1 size Y = '+str(sub1SzY))    

    #print('boxYsub2 = '+str(boxYsub2))



    # left frame
    phiHorizFrameImg.paste(eyeSub2, boxYsub2)
    #phiHorizFrameImg.paste(eyeSub3, boxYsub3)
    
    phiHorizFrameImg.paste(eyeXSub1, boxXsub1)
    phiHorizFrameImg.paste(eyeXSub2, boxXsub2)    
    phiHorizFrameImg.paste(eyeXSub3, boxXsub3)
    

    return phiHorizFrameImg


if (0):

    eyeExp1 = jpgSrcDir+'bananasloth000011.jpg'
    eyeExp2 = jpgSrcDir+'odmkDmtpatra001097.jpg'
    eyeExp3 = jpgSrcDir+'bSlothCultLife000929.jpg'
    eyeExp4 = jpgSrcDir+'bananasloth0036.jpg'
    
    
    eye1 = misc.imread(eyeExp1)
    eye2 = misc.imread(eyeExp2)
    eye3 = misc.imread(eyeExp3)    
    
    #eye1img = Image.fromarray(eye1)
    eye1img = Image.fromarray(eye1)
    eye2img = Image.fromarray(eye2)
    eye3img = Image.fromarray(eye3)    
    
    
    #ctrl = 1
    
    phiHorizFrame = eyePhiHorizFrame1(eye1img, eye2img, eye3img)
    
    
    eyeOutFileName = 'phiHorizFrameTest'
    phiHorizFrameFull = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(phiHorizFrameFull, phiHorizFrame)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


def eyePolyXFade(img1, img2, n):
    ''' generates composite n frame mapped image
        SzX, SzY: output image dimentions
        img: PIL image object
        ?ctrl: { UL, UR, LL, LR} - selects base quadrant ?
        
        center_pxl_coord
        xy_pxl_coord -> polar_pxl_coord
        polar_pxl_angle </> bounded_region ? process a / b
        
        
        '''
        
    
    
    if not(img1.size[0]==img2.size[0] and img1.size[1]==img2.size[1] ):
        print('ERROR: sizeof img1 must equal siseof img2')
        return
    
    SzX = img1.size[0]
    SzY = img1.size[1]
    
    #imgDim = (SzX, SzY)
    
    #nCzn = cyclicZn(n)
       
    #mag = sp.sqrt(specframe[k].real*specframe[k].real + specframe[k].imag*specframe[k].imag)
    #phi = np.arctan2(specframe[k].imag, specframe[k].real)    
    

    eyeOut_r = np.zeros((SzY, SzX))
    eyeOut_g = np.zeros((SzY, SzX))
    eyeOut_b = np.zeros((SzY, SzX))
    
    
    # cZn[k] = np.cos(((k)*2*np.pi)/n) + np.sin(((k)*2*np.pi)/n)*1j   # Euler's identity
    cZn = cyclicZn(n)
            
    for k in range(1,n):
        alphaX = k * 1/n

        imgBlend = Image.blend(img1, img2, alphaX) 
        imgArray = np.array(imgBlend)
        
        #r1, g1, b1 = imgBlend[..., 0], imgBlend[..., 1], imgBlend[..., 2]
        
        for x in range(SzX):
            for y in range(SzY):        
                if ( (np.arctan2(x, y) < np.arctan2(np.real(cZn[k]), np.imag(cZn[k]))) and (np.arctan2(x, y) > np.arctan2(np.real(cZn[k-1]), np.imag(cZn[k-1]))) ):
                    eyeOut_r[y, x] = imgArray[y, x, 0]
                    eyeOut_g[y, x] = imgArray[y, x, 1] 
                    eyeOut_b[y, x] = imgArray[y, x, 2]                    


    eyeOut = np.dstack((eyeOut_r, eyeOut_g, eyeOut_b))
    
    #eyeOutFileName = 'eyeMirrorHV4'
    #eyeMirrorHV4Full = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeMirrorHV4Full, mirrorImg)
    
    return eyeOut


if (0):

    eyeExp1 = jpgSrcDir+'bananasloth000011.jpg'
    eyeExp2 = jpgSrcDir+'odmkDmtpatra001097.jpg'
    eyeExp3 = jpgSrcDir+'odmkExternal000121.jpg'
    eyeExp4 = jpgSrcDir+'bSlothCultLife000929.jpg'
    eyeExp5 = jpgSrcDir+'bananasloth0036.jpg'
    
    eye1 = misc.imread(eyeExp2)
    eye2 = misc.imread(eyeExp5)
    
    
    #eye1img = Image.fromarray(eye1)
    eye1img = Image.fromarray(eye1)
    eye2img = Image.fromarray(eye2)
    
    polyXF_eye1 = eyePolyXFade(eye1img, eye2img, 5)

    
    eyeOutFileName = 'polyXF_eye1'
    eye1imgFull = imgOutDir+eyeOutFileName+'.jpg'
    #misc.imsave(eyeBoxFull, eye1imgCpy)
    misc.imsave(eye1imgFull, polyXF_eye1)
    



# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



if (0):
    
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::Concatenate all images in directory list::---*')
    print('// *--------------------------------------------------------------* //')    
    
    # repeatReverseAllImg - operates on directory
    loc1 = eyeDir+'bSlothCultMirrorHV4_Z1_135bpm\\'
    loc2 = eyeDir+'bSlothCultMirrorHV4_Z2_135bpm\\'
    loc3 = eyeDir+'bSlothCultMirrorHV4_Z3_135bpm\\'
    loc4 = eyeDir+'bSlothCultMirrorHV4_Z4_135bpm\\'
    loc5 = eyeDir+'bSlothCultMirrorHV4_Z5_135bpm\\'
    loc6 = eyeDir+'bSlothCultMirrorHV4_Z6_135bpm/'
    loc7 = eyeDir+'bSlothCultMirrorHV4_Z7_135bpm/'
    loc8 = eyeDir+'bSlothCultMirrorHV4_Z8_135bpm/'
    loc9 = eyeDir+'bSlothCultMirrorHV4_Z9_135bpm/'
    loc10 = eyeDir+'bSlothCultMirrorHV4_Z10_135bpm/'    
    
    #loc6 = eyeDir+'odmkFxionDir_Z6/'
    #    loc6 = eyeDir+'bananaslothParascopeDir_6/'
    #    loc7 = eyeDir+'bananaslothParascopeDir_7/'
    #    loc8 = eyeDir+'bananaslothParascopeDir_8/'    
    
    
    concatDirList = [loc1, loc2, loc3, loc4, loc5, loc6, loc7, loc8, loc9, loc10]
    #concatDirList = [loc1, loc2] 
       
    concatReName = 'bSlothCultMirrorHV4_Z'
    concatOutDir = 'bSlothCultMirrorHV4_Z_135bpm\\'
    
    
    # ex. defined above: concatDirList = [loc1, loc2, loc3, loc4, loc5, loc6, loc7, loc8]
    dirList = concatDirList
    reName = concatReName
    
    eyeConcatDir = eyeDir+concatOutDir
    os.makedirs(eyeConcatDir, exist_ok=True)
    
    
    # function:  concatAllDir(dirList, w=0, concatDir='None', concatName='None')
    # [gorgulanObjList, gorgulanSrcList] = concatAllDir(dirList)
    # [gorgulanObjList, gorgulanSrcList] = concatAllDir(dirList, w=0, concatName=reName)
    # [gorgulanObjList, gorgulanSrcList] = concatAllDir(dirList, w=0, concatDir=gorgulanConcatDir)
    # [gorgulanObjList, gorgulanSrcList] = concatAllDir(dirList, w=1, concatDir=gorgulanConcatDir)
    concatAllDir(dirList, eyeConcatDir, reName)
    
    print('\nOutput all images to: '+eyeConcatDir)
    
    print('// *--------------------------------------------------------------* //')








# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

#eyeIn5 = jpgSrcDir+'bananasloth0036.jpg'
#eyeIn6 = jpgSrcDir+'microscopic0042.jpg'
#
## region = region.transpose(Image.ROTATE_180)
## im.paste(region, box)
##eye1 = misc.imread(eyeIn1)
#eyeSh = misc.imread(eyeIn6)
##eye1img = Image.fromarray(eye1)
#eyeShImg = Image.fromarray(eyeSh)
#
#eyeOutFileName = 'eyeShiftTest'
#eyeShiftFull = imgOutDir+eyeOutFileName+'.jpg'
#
##eyeShift1 = ndimage.shift(eye1, 100, mode='reflect')    
#eyeShift1 = ndimage.shift(eyeSh, 500, mode='wrap')    #  mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). Default is ‘constant’.
#misc.imsave(eyeShiftFull, eyeShift1)



# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



# IMG Coordinates:

#    |  0,0  ... 0,SzX   |
#    |   .         .     |
#    |   .         .     |
#    |   .         .     |
#    | 0,SzY ... SzX,SzY |
#
#



## ndimage map_coordinates example
#aa1 = np.arange(12.).reshape((4, 3))
#
#
#ndimage.map_coordinates(aa1, [[0.5, 2], [0.5, 1]], order=1)
#
#inds1 = np.array([[0.5, 2], [0.5, 4]])
#
#ndimage.map_coordinates(aa1, inds1, order=1, cval=-33.3)
#
#ndimage.map_coordinates(aa1, inds1, order=1, mode='nearest')
#
#ndimage.map_coordinates(aa1, inds1, order=1, cval=0, output=bool)
#
#
#
#
## ndimage geometric_transform example
#a = np.arange(12.).reshape((4, 3))
#
#def shift_func(output_coords):
#    return (output_coords[0] - 0.5, output_coords[1] - 0.5)
#
#ndimage.geometric_transform(a, shift_func)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //




#    img_r = img[:, :, 0]
#    img_g = img[:, :, 1]
#    img_b = img[:, :, 2]
#
#    imgCrop_r = np.zeros((SzY, SzX))
#    imgCrop_g = np.zeros((SzY, SzX))
#    imgCrop_b = np.zeros((SzY, SzX))
#
#
#    for j in range(SzY):
#        for k in range(SzX):
#            if high == 1:
#                imgCrop_r[j, k] = img_r[j, int(k + wdiff_lhs)]
#                imgCrop_g[j, k] = img_g[j, int(k + wdiff_lhs)]
#                imgCrop_b[j, k] = img_b[j, int(k + wdiff_lhs)]
#            else:
#                # pdb.set_trace()
#                imgCrop_r[j, k] = img_r[int(j + hdiff_ths), int(k + wdiff_lhs)]
#                imgCrop_g[j, k] = img_g[int(j + hdiff_ths), int(k + wdiff_lhs)]
#                imgCrop_b[j, k] = img_b[int(j + hdiff_ths), int(k + wdiff_lhs)]
#        # if j == SzY - 1:
#            # print('imgItr ++')
#    imgCrop = np.dstack((imgCrop_r, imgCrop_g, imgCrop_b))





# scipy.ndimage.zoom(input, zoom, output=None, order=3, mode='constant', cval=0.0, prefilter=True)[source]
#
#    Zoom an array.
#
#    The array is zoomed using spline interpolation of the requested order.
#    Parameters:	
#
#    input : ndarray
#        The input array.
#
#    zoom : float or sequence, optional
#        The zoom factor along the axes. If a float, zoom is the same for each axis. If a sequence, zoom should contain one value for each axis.
#
#    output : ndarray or dtype, optional
#        The array in which to place the output, or the dtype of the returned array.
#
#    order : int, optional
#        The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
#
#    mode : str, optional
#        Points outside the boundaries of the input are filled according to the given mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). Default is ‘constant’.
#
#    cval : scalar, optional
#        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0
#
#    prefilter : bool, optional
#        The parameter prefilter determines if the input is pre-filtered with spline_filter before interpolation (necessary for spline interpolation of order > 1). If False, it is assumed that the input is already filtered. Default is True.
#
#    Returns:	
#
#    zoom : ndarray or None
#
#        The zoomed input. If output is given as a parameter, None is returned.




# *****Map the input array to new coordinates by interpolation.*****

# The array of coordinates is used to find, for each point in the output, the corresponding coordinates in the input.
# The value of the input at those coordinates is determined by spline interpolation of the requested order.

# The shape of the output is derived from that of the coordinate array by dropping the first axis.
# The values of the array along the first axis are the coordinates in the input array at which the output value is found.


# scipy.ndimage.map_coordinates(input, coordinates, output=None, order=3, mode='constant', cval=0.0,
# prefilter=True)[source]
#
#    Map the input array to new coordinates by interpolation.
#
#    The array of coordinates is used to find, for each point in the output, the corresponding coordinates in the input. The value of the input at those coordinates is determined by spline interpolation of the requested order.
#
#    The shape of the output is derived from that of the coordinate array by dropping the first axis. The values of the array along the first axis are the coordinates in the input array at which the output value is found.
#    Parameters:	
#
#    input : ndarray
#        The input array.
#
#    coordinates : array_like
#        The coordinates at which input is evaluated.
#
#    output : ndarray or dtype, optional
#        The array in which to place the output, or the dtype of the returned array.
#
#    order : int, optional
#        The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
#
#    mode : str, optional
#        Points outside the boundaries of the input are filled according to the given mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). Default is ‘constant’.
#
#    cval : scalar, optional
#        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0
#
#    prefilter : bool, optional
#        The parameter prefilter determines if the input is pre-filtered with spline_filter before interpolation (necessary for spline interpolation of order > 1). If False, it is assumed that the input is already filtered. Default is True.
#
#    Returns:	
#
#    map_coordinates : ndarray
#        The result of transforming the input. The shape of the output is derived from that of coordinates by dropping the first axis.









# scipy.ndimage.geometric_transform(input, mapping, output_shape=None, output=None, order=3, mode='constant', cval=0.0, prefilter=True, extra_arguments=(), extra_keywords={})[source]
#
#    Apply an arbritrary geometric transform.
#
#    The given mapping function is used to find, for each point in the output, the corresponding coordinates in the input. The value of the input at those coordinates is determined by spline interpolation of the requested order.
#    Parameters:	
#
#    input : array_like
#        The input array.
#
#    mapping : callable
#        A callable object that accepts a tuple of length equal to the output array rank, and returns the corresponding input coordinates as a tuple of length equal to the input array rank.
#
#    output_shape : tuple of ints, optional
#        Shape tuple.
#
#    output : ndarray or dtype, optional
#        The array in which to place the output, or the dtype of the returned array.
#
#    order : int, optional
#        The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
#
#    mode : str, optional
#        Points outside the boundaries of the input are filled according to the given mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). Default is ‘constant’.
#
#    cval : scalar, optional
#        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0
#
#    prefilter : bool, optional
#        The parameter prefilter determines if the input is pre-filtered with spline_filter before interpolation (necessary for spline interpolation of order > 1). If False, it is assumed that the input is already filtered. Default is True.
#
#    extra_arguments : tuple, optional
#        Extra arguments passed to mapping.
#
#    extra_keywords : dict, optional
#        Extra keywords passed to mapping.
#
#    Returns:	
#
#    return_value : ndarray or None
#        The filtered input. If output is given as a parameter, None is returned.



## ***** PIL library functions *****


#    eyeSub = eyeSub.resize(subDim)
#    eyeSub = eyeSub.crop(box)
#    eyeSub = ImageOps.mirror(eyeSub)
#    eyeSub = ImageOps.flip(eyeSub)



#im.resize(size) ⇒ image
#
#im.resize(size, filter) ⇒ image
#
#Returns a resized copy of an image. The size argument gives the requested size in pixels,
#as a 2-tuple: (width, height).
#

#
#autocontrast #
#
#ImageOps.autocontrast(image, cutoff=0) ⇒ image
#
#Maximize (normalize) image contrast. This function calculates a histogram of the input image, removes cutoff
#percent of the lightest and darkest pixels from the histogram, and remaps the image so that the darkest
#remaining pixel becomes black (0), and the lightest becomes white (255).
#
#
#
#ImageOps.flip(image) ⇒ image
#
#Flip the image vertically (top to bottom).
#
#
#
#ImageOps.invert(image) ⇒ image
#
#Invert (negate) the image.
#
#
#
#ImageOps.mirror(image) ⇒ image
#
#Flip image horizontally (left to right).
#
#
#
#ImageOps.solarize(image, threshold=128) ⇒ image
#
#Invert all pixel values above the given threshold.