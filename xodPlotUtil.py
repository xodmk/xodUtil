# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header begin-----------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

# __::((xodPlotUtil.py))::__

# Python XODMK plotting functions

# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header end-------------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation



# temp python debugger - use >>>pdb.set_trace() to set break
import pdb


mplot_black = (0./255., 0./255., 0./255.)
mplot_white = (255./255., 255./255., 255./255.)
mplot_red = (255./255., 0./255., 0./255.)
mplot_orange = (255/255., 165/255., 0./255.)
mplot_darkorange = (255/255., 140/255., 0./255.)
mplot_orangered = (255/255., 69/255., 0./255.)
mplot_yellow = (255./255., 255./255., 0./255.)
mplot_lime = (0./255., 255./255., 0./255.)
mplot_green = (0./255., 128./255., 0./255.)
mplot_darkgreen = (0./255., 100./255., 0./255.)
mplot_cyan = (0./255., 255./255., 255./255.)
mplot_blue = (0./255., 0./255., 255./255.)
mplot_midnightblue = (25./255., 25./255., 112./255.)
mplot_magenta = (255./255., 0./255., 255./255.)
mplot_grey = (128./255., 128./255., 128./255.)
mplot_silver = (192./255., 192./255., 192./255.)
mplot_darkgrey = (64./255., 64./255., 64./255.)
mplot_darkdarkgrey = (32./255., 32./255., 32./255.)
mplot_purple = (128./255., 0./255., 128./255.)
mplot_maroon = (128./255., 0./255., 0./255.)
mplot_olive = (128./255., 128./255., 0./255.)
mplot_teal = (0./255., 128./255., 128./255.)


# // *---------------------------------------------------------------------* //
# // *--Plot Functions--*
# // *---------------------------------------------------------------------* //

def xodPlot1D(fnum, sig, xLin, pltTitle, pltXlabel, pltYlabel, fsizeX=9, fsizeY=7,
               lncolor='red', lnstyle='-', lnwidth=1.00, pltGrid=False, pltBgColor='black'):
    ''' 1D Matplotlib plot
        required inputs:
            fnum => unique plot number
            sig => signal to plot
            xLin => linear space to define x-axis (0 to max x-axis length-1)
            pltTitle => text string for plot title
            pltXlabel => text string for x-axis
            pltYlabel => text string for y-axis
        optional inputs:
            fsizeX => figure width
            fsizeY => figure height
            lncolor => line color (default = red ; html color names, html color codes??)
            lnstyle => line style (default = plain line ; * ; o ; etc..)
            lnwidth => line width
            pltGrid => use grid : default = True ; <True;False>
            pltBgColor => backgroud color (default = black) '''


    plt.figure(num=fnum, facecolor='silver', edgecolor='k', figsize=(fsizeX, fsizeY))
    # check if xLin is < than or = to sig
    if len(xLin) > len(sig):
        print('ERROR (xodPlot1D - plot#'+str(fnum)+'): length of xLin x-axis longer than signal length - fnum = '+str(fnum))
        return 1
    elif len(xLin) == len(sig):
        odmkMatPlt = plt.plot(xLin, sig)
    else:
        odmkMatPlt = plt.plot(xLin, sig[0:len(xLin)])

    plt.setp(odmkMatPlt, color=lncolor, ls=lnstyle, linewidth=lnwidth)
    plt.xlabel(pltXlabel)
    plt.ylabel(pltYlabel)
    plt.title(pltTitle)
    plt.grid(color='c', linestyle=':', linewidth=.5)
    plt.grid(pltGrid)
    # plt.xticks(np.linspace(0, Fs/2, 10))
    ax = plt.gca()
    ax.set_facecolor(pltBgColor)

    return 0


def xodMultiPlot1D(fnum, sigArray, xLin, pltTitle, pltXlabel, pltYlabel, fsizeX=9, fsizeY=7,
                    colorMap='gnuplot', lnstyle='-', lnwidth=1.00, pltGrid=False, pltBgColor='black'):
    ''' 1D Matplotlib multi-plot
        required inputs:
            fnum => unique plot number
            sigArray => signal to plot : 2D Numpy array *** (r=nArrays, c=arrayLength)
            xLin => linear space to define x-axis (0 to max x-axis length-1)
            pltTitle => text string for plot title
            pltXlabel => text string for x-axis
            pltYlabel => text string for y-axis
        optional inputs:
            fsizeX => figure width
            fsizeY => figure height
            lncolor => line color (default = red ; html color names, html color codes??)
            lnstyle => line style (default = plain line ; * ; o ; etc..)
            lnwidth => line width
            pltGrid => use grid : default = True ; <True;False>
            pltBgColor => backgroud color (default = black) '''


    try:
        arrShape = sigArray.shape
        if len(arrShape) != 2:
            print('ERROR (xodMultiPlot1D - plot#'+str(fnum)+'): sigArray must be 2D array [r=nArrays, c=arrayLength]')
            return 1
    except:
        print('ERROR (xodMultiPlot1D - plot#'+str(fnum)+'): sigArray must be 2D array [r=nArrays, c=arrayLength]')
        return 1              

    # define the color map
    try:
        cmap = plt.cm.get_cmap(colorMap)
    except ValueError as e:
        print('ValueError: ', e)
    colors = cmap(np.linspace(0.0, 1.0, len(sigArray)))

    plt.figure(num=fnum, facecolor='silver', edgecolor='k', figsize=(fsizeX, fsizeY))
    # check if xLin is < than or = to sig
    if len(xLin) > len(sigArray[0, :]):
        print('ERROR (xodMultiPlot1D - plot#'+str(fnum)+'): length of xLin x-axis longer than signal length')
        return 1
    else:
        if len(xLin) == len(sigArray[0, :]):
            for i in range(len(sigArray[:, 0])):
                plt.plot(xLin, sigArray[i, :], color=colors[i], ls=lnstyle, linewidth=lnwidth)
        else:
            for i in range(len(sigArray[:, 0])):
                plt.plot(xLin, sigArray[i, 0:len(xLin)], color=colors[i], ls=lnstyle, linewidth=lnwidth)

        plt.xlabel(pltXlabel)
        plt.ylabel(pltYlabel)
        plt.title(pltTitle)
        plt.grid(color='c', linestyle=':', linewidth=.5)
        plt.grid(pltGrid)
        # plt.xticks(np.linspace(0, Fs/2, 10))
        ax = plt.gca()
        ax.set_facecolor(pltBgColor)

    return 0


# temp - before adding figure size params..
#def odmkMultiPlot1D(fnum, sigArray, xLin, pltTitle, pltXlabel, pltYlabel, 
#                    colorMap='gnuplot', lnstyle='-', lnwidth=1.00, pltGrid=False, pltBgColor='black'):
#    ''' ODMK 1D Matplotlib multi-plot
#        required inputs:
#        fnum => unique plot number
#        sigArray => signal to plot : 2D Numpy array *** (r=nArrays, c=arrayLength)
#        xLin => linear space to define x-axis (0 to max x-axis length-1)
#        pltTitle => text string for plot title
#        pltXlabel => text string for x-axis
#        pltYlabel => text string for y-axis
#        optional inputs:
#        lncolor => line color (default = red ; html color names, html color codes??)
#        lnstyle => line style (default = plain line ; * ; o ; etc..)
#        lnwidth => line width
#        pltGrid => use grid : default = True ; <True;False>
#        pltBgColor => backgroud color (default = black) '''
#
#
#    try:
#        arrShape = sigArray.shape
#        if len(arrShape) != 2:
#            print('ERROR (odmkMultiPlot1D - plot#'+str(fnum)+'): sigArray must be 2D array [r=nArrays, c=arrayLength]')
#            return 1
#    except:
#        print('ERROR (odmkMultiPlot1D - plot#'+str(fnum)+'): sigArray must be 2D array [r=nArrays, c=arrayLength]')
#        return 1              
#
#    # define the color map
#    try:
#        cmap = plt.cm.get_cmap(colorMap)
#    except ValueError as e:
#        print('ValueError: ', e)
#    colors = cmap(np.linspace(0.0, 1.0, len(sigArray)))
#
#    plt.figure(num=fnum, facecolor='silver', edgecolor='k')
#    # check if xLin is < than or = to sig
#    if len(xLin) > len(sigArray[0, :]):
#        print('ERROR (odmkMultiPlot1D - plot#'+str(fnum)+'): length of xLin x-axis longer than signal length')
#        return 1
#    else:
#        if len(xLin) == len(sigArray[0, :]):
#            for i in range(len(sigArray[:, 0])):
#                plt.plot(xLin, sigArray[i, :], color=colors[i], ls=lnstyle, linewidth=lnwidth)
#        else:
#            for i in range(len(sigArray[:, 0])):
#                plt.plot(xLin, sigArray[i, 0:len(xLin)], color=colors[i], ls=lnstyle, linewidth=lnwidth)
#
#        plt.xlabel(pltXlabel)
#        plt.ylabel(pltYlabel)
#        plt.title(pltTitle)
#        plt.grid(color='c', linestyle=':', linewidth=.5)
#        plt.grid(pltGrid)
#        # plt.xticks(np.linspace(0, Fs/2, 10))
#        ax = plt.gca()
#        ax.set_facecolor(pltBgColor)
#
#    return 0


#//////////////////////////////////////////////////////////////////////////////
# begin: Example 5 - p q modulated---------------------------------------------
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\



def xodPlot3D(fnum, xyzArray, pltTitle, pltXlabel, pltYlabel, pltZlabel, 
               colorMap='gnuplot', pltGrid=False, pltBgColor='black'):
    ''' 3D Matplotlib plot
        required inputs:
        fnum => unique plot number
        sig => signal to plot : 2D Numpy array
        pltTitle => text string for plot title
        pltXlabel => text string for x-axis
        pltYlabel => text string for y-axis
        pltZlabel => text string for z-axis
        optional inputs:
        pltGrid => use grid : default = True ; <True;False>
        pltBgColor => backgroud color (default = black) '''

    p5=2
    q5=15
    
    aLength=666
    dotSize=131
    
    # Generate torus mesh
    angle5 = np.linspace(0, 2 * np.pi, aLength)
    theta5, phi5 = np.meshgrid(angle5, angle5)
    r5 = 0.25
    R5 = 1
    #X = (R + r * np.cos(q*phi)) * np.cos(p*theta)
    #Y = (R + r * np.cos(q*phi)) * np.sin(p*theta)
    #Z = r * np.sin(q*phi)
    
#    X = np.cos(p5*theta5)*(R5 + 0.15*np.cos(3*theta5) + 0.35*np.cos(9*theta5) - 0.4*np.cos(q5*theta5))
#    Y = np.sin(p5*theta5)*(R5 + 0.15*np.cos(3*theta5) + 0.35*np.cos(9*theta5) - 0.4*np.cos(q5*theta5))
#    Z = r5 * np.sin(q5*theta5)
    
    #cmhot5 = plt.get_cmap("BuPu")
    cmhot5 = plt.get_cmap("gist_heat")
    #cmhot5 = plt.get_cmap("RdBu")
    #cmhot5 = plt.get_cmap("gist_earth")
    
    # Display the mesh
    fig3D = plt.figure(num=fnum, figsize=(13,6), facecolor='black', edgecolor='r')
    ax2 = fig3D.add_subplot(1, 2, 1, projection='3d')
    #ax2 = plt.axes(projection='3d')
    ax2.set_xlim3d(-1, 1)
    ax2.set_ylim3d(-1, 1)
    ax2.set_zlim3d(-1, 1)
    
    ax2.plot_surface(X, Y, Z, color = 'purple', rstride = 1, cstride = 1)
    #ax2.plot(X, Y, Z, label='parametric curve')
    
    #ax2.scatter(X5, Y5, Z5, c=Z5, s=dotSize, cmap=cmhot5, marker='o', alpha=0.5)
    plt.xlabel('torusKnot X')
    plt.ylabel('torusKnot Y')
    plt.title('odmk torusKnot P='+str(p5)+', Q='+str(q5))
    plt.grid(False)
    ax2 = plt.gca()
    #ax2.set_axis_bgcolor(mplot_darkdarkgrey)
    ax2.set_facecolor(mplot_orangered)
    ax2.view_init(15, 45)


    #//////////////////////////////////////////////////////////////////////////
    # begin: Ex30 - Cascaded Bar Plot------------------------------------------
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def xodBarPlot3D(fnum, sigArray, pltTitle, pltXlabel, pltYlabel, pltZlabel, colorMap='gnuplot', pltGrid=False, pltBgColor='black'):
    ''' 3D Matplotlib bar plot
        required inputs:
        fnum => unique plot number
        sig => signal to plot : 2D Numpy array
        pltTitle => text string for plot title
        pltXlabel => text string for x-axis
        pltYlabel => text string for y-axis
        pltZlabel => text string for z-axis
        optional inputs:
        pltGrid => use grid : default = True ; <True;False>
        pltBgColor => backgroud color (default = black) '''

    # define the color map
    try:
        cmap = plt.cm.get_cmap(colorMap)
    except ValueError as e:
        print('ERROR (odmkBarPlot3D) ValueError: ', e)
    colors = cmap(np.linspace(0.0, 1.0, len(sigArray[0, :])))     # ?
    # colors = plt.cm.Spectral(np.linspace(0.1, 1.0, len(rows)))
    
    figbp = plt.figure(num=fnum, facecolor='silver', edgecolor='k')    

    ax = figbp.add_subplot(111, projection='3d')
    
    nBars = len(sigArray[:, 0])     # ex: NFFT / 2
    # create an evenly spaced array for num frames
    rows = np.array([])
    for j in range(len(sigArray[0, :])):
        rows = np.append(rows, j*20)     # num of frames
    
    
    #iterator c -> y,r,b,g ; z => 0,20,40,60
    #for c, z in zip(['y', 'r', 'b', 'g'], [0, 20, 40, 60]):
    for i in range(len(rows)):    
        xs = np.arange(nBars)
        ys = np.arange(len(sigArray[i, :]))
        zs = sigArray[:, i]
    
        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        cs = np.repeat(colors[i],len(xs)).reshape(4,len(xs)).T
        #cs[0] = 'c'
        ax.bar(xs, ys, zs, zdir='y', color=cs, alpha=0.9)
    
    ax.set_xlabel('pltXlabel')
    ax.set_ylabel('pltYlabel')
    ax.set_zlabel('pltZlabel')
    
    #pdb.set_trace()
    
    #//////////////////////////////////////////////////////////////////////////
    # end: Ex30 - Cascaded Bar Plot--------------------------------------------
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\