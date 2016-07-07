#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#  
#  SOIF - Sofware for Optical Interferometry fitting
#  Copyright (C) 2016  Guillaume Schworer
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  
#  For any information, bug report, idea, donation, hug, beer, please contact
#    guillaume.schworer@obspm.fr
#
###############################################################################


from scipy.special import j1 as airyJ1
from scipy.signal import fftconvolve
import numpy as np
from time import time as timetime
from multiprocessing import current_process as multiprocessingcurrent_process
from sys import maxsize as sysmaxint
sysmaxint = min(2**32, sysmaxint)
from scipy.interpolate import interp1d as scipyinterpolateinterp1d
from scipy.integrate import cumtrapz as scipyintegratecumtrapz
from matplotlib.cm import get_cmap as cmget_cmap
from matplotlib.cm import ScalarMappable as cmScalarMappable
from matplotlib.pyplot import Normalize as matplotlibpyplotNormalize
from matplotlib.pyplot import subplots as matplotlibpyplotsubplots


DATATYPEEXTNAMES = {'OI_T3':'T3', 'OI_VIS2':'VIS2', 'OI_VIS':'VIS'}
DATAEXTNAMES = {'OI_T3':'T3PHI', 'OI_VIS2':'VIS2DATA', 'OI_VIS':'VISPHI'}
ALLDATAEXTNAMES = {'OI_T3':['T3AMP','T3PHI'], 'OI_VIS2':['VIS2'], 'OI_VIS':['VISPHI','VISAMP']}
DATAKEYSDATATYPE = {'T3AMP':{'data':'T3AMP', 'error':'T3AMPERR', 'mask':'FLAG'},
                    'T3PHI':{'data':'T3PHI', 'error':'T3PHIERR', 'mask':'FLAG'},
                    'VIS2':{'data':'VIS2DATA', 'error':'VIS2ERR', 'mask':'FLAG'},
                    'VISPHI':{'data':'VISPHI', 'error':'VISPHIERR', 'mask':'FLAG'},
                    'VISAMP':{'data':'VISAMP', 'error':'VISAMPERR', 'mask':'FLAG'}
                    }
UVKEYSDATATYPE = {  'T3AMP':{'u1':'U1COORD', 'v1':'V1COORD', 'u2':'U2COORD', 'v2':'V2COORD'},
                    'T3PHI':{'u1':'U1COORD', 'v1':'V1COORD', 'u2':'U2COORD', 'v2':'V2COORD'},
                    'VIS2':{'u':'UCOORD', 'v':'VCOORD'},
                    'VISPHI':{'u':'UCOORD', 'v':'VCOORD'},
                    'VISAMP':{'u':'UCOORD', 'v':'VCOORD'}
                    }
ATTRDATATYPE = {'T3AMP':{'is_t3':True, 'is_angle':False},
                'T3PHI':{'is_t3':True, 'is_angle':True},
                'VIS2':{'is_t3':False, 'is_angle':False},
                'VISPHI':{'is_t3':False, 'is_angle':True},
                'VISAMP':{'is_t3':False, 'is_angle':False}
                }
DATAKEYSLOWER = ['vis2', 't3phi', 't3amp', 'visphi', 'visamp']

def abs2(ar): return np.abs(ar)**2


FCTVISCOMP = {'VIS2':abs2, 'T3PHI':np.angle, 'T3AMP':np.abs, 'VISPHI':np.angle, 'VISAMP':np.abs}

KEYSWL = {'wl':'EFF_WAVE', 'wl_d':'EFF_BAND'}
KEYSDATA = ['data', 'error', 'mask']
KEYSUV = ['v', 'u', 'wl', 'wl_d']
INPUTSAVEKEY = ['src', 'hduidx', 'hduwlidx', 'degree', 'flatten', 'significant_figures', 'indices', 'wlindices']


OLDNUMPY = int(np.__version__.split('.')[1])<10
MAS2RAD = 4.84813681109536e-09 # np.pi/1000./3600./180. # multiply mili-arc-second (mas) with that and you get radian
RAD2MAS = 1/MAS2RAD
ASEC2RAD = 4.84813681109536e-06 # np.pi/3600./180. # multiply arc second with that and you get radian
RAD2ASEC = 1/ASEC2RAD
DEG2RAD = 0.017453292519943295 # np.pi/180. # multiply degrees with that and you get radian
RAD2DEG = 1/DEG2RAD
DEUXPI = 2*np.pi
LNDEUXPI = np.log(2*np.pi)
PISURDEUX = 0.5*np.pi
SECPERDAY = 86400.

def hduToDataType(hdu):
    """
    Give a hdu, return a datatype name as string or None if it is not a data hdu
    """
    return DATATYPEEXTNAMES.get(getattr(hdu, "header", {}).get('EXTNAME', None), None)

def hduWlindex(hdus, returnHdu=False):
    """
    Give a list of hdus, return the index of the OI_WAVELENGTH hdu, or False if not found
    """
    for idx, hdu in enumerate(hdus):
        if getattr(hdu, "header", {}).get('EXTNAME', None) == 'OI_WAVELENGTH':
            if returnHdu:
                return idx, hdu
            else:
                return idx
    else:
        return False

def hduToColNames(hdu):
    """
    Returns a list of the column names
    """
    return [i.name for i in hdu.data.columns.columns]

def replicate(ar, dims):
    left = ()
    right = ()
    hasbeennone = False
    for i in dims:
        if i is None or i==-1:
            hasbeennone = True
            continue
        if hasbeennone:
            right += (i,)
        else:
            left += (i,)
    ar = np.tile(ar, left+tuple(np.ones(ar.ndim).astype(int)))
    if right!=():
        ar = ar.repeat(np.prod(right)).reshape(ar.shape+right)
    return ar


class font:
    white = '\033[97m'
    black = '\033[38;5;16m'
    gray = '\033[90m'
    red = '\033[31m'
    green = '\033[32m'
    yellow = '\033[33m'
    orange = '\033[38;5;166m'
    blue = '\033[34m'
    magenta = '\033[35m'
    nocolor = '\033[39m'
    bold = '\033[1m'
    nobold = '\033[21m'
    underlined = '\033[4m'
    nounderlined = '\033[24m'
    dim = '\033[2m'
    nodim = '\033[22m'
    normal = nodim + nobold + nobold + nocolor
    clear = chr(27)+"[2J"
    def pos(self, line, col): return "\033["+str(int(line))+";"+str(int(col))+"H"


def astroskyplot(radius, polar=False, unit='arcsec', sameaxes=False):
    if polar:
        fig, ax = matplotlibpyplotsubplots(subplot_kw={'projection':'polar'})
        ax.set_theta_zero_location('N')
        ax.set_ylim([0, float(radius)])
        ax.set_xticklabels(['N',u'45°','E',u'135°','S',u'225°','W',u'315°'])
        ax.set_xlabel('['+unit+']')
    else:
        fig, ax = matplotlibpyplotsubplots()
        ax.set_xlim([-float(radius), float(radius)])
        ax.set_ylim([-float(radius), float(radius)])
        ax.invert_xaxis()
        ax.set_ylabel('E, dec ['+unit+']')
        ax.set_xlabel('S, ra ['+unit+']')
    if sameaxes: ax.set_aspect('equal')
    return fig, ax


def gen_seed(): return int(int(str((timetime()%3600)/3600)[2:])*multiprocessingcurrent_process().pid%sysmaxint)


def gen_generator(seed=None):
    if seed is None: return np.random.RandomState(gen_seed())
    return np.random.RandomState(seed)


def random_custom_pdf(x, pdf, size=1, renorm=False):
    if (np.asarray(pdf)<0).any(): raise Exception("Can't compute pdf with negative values") # if any negative value in pdf, raises error
    cdf = integrate_array(x, pdf)
    return random_custom_cdf(x, cdf, size=size, renorm=renorm)


def random_custom_cdf(x, cdf, size=1, renorm=False):
    cdf = np.array(cdf)
    if (np.diff(cdf)<0).any(): raise Exception("cdf must be monotonic increasing")
    if not renorm:
        if np.round(cdf[-1],10)!=1 or np.round(cdf[0],10)!=0: raise Exception("Wrong cdf distribution (first element not 0 or last element not 1)") # if first or last element not close enough from 0 and 1, rejects
    cdf = (cdf-cdf[0])/(cdf[-1]-cdf[0]) # forces first and last elements to be 0 and 1 exactly
    rnd = gen_generator()
    if size is False: return scipyinterpolateinterp1d(cdf, x, kind='linear')
    if size==1: return float(scipyinterpolateinterp1d(cdf, x, kind='linear')(rnd.uniform(size=1)))
    if np.size(size)==1: return scipyinterpolateinterp1d(cdf, x, kind='linear')(rnd.uniform(size=size))
    return scipyinterpolateinterp1d(cdf, x, kind='linear')(rnd.uniform(size=np.prod(size)).reshape(size))


def integrate_array(x, y, axis=0):
    y = np.array(y).swapaxes(axis,-1)
    inty = np.zeros(y.shape)
    myslice = [slice(0,i) for i in aslist(y.shape)[:-1]]+[slice(1,y.shape[-1])]
    inty[myslice] = scipyintegratecumtrapz(y, x)
    return inty.swapaxes(axis,-1)


def aslist(data, numpy=False, integer=False):
    if not hasattr(data, "__iter__") or isinstance(data, dict):
        if integer:
            ret = np.asarray([int(data)])
        else:
            ret = np.asarray([data])
    else:
        if integer:
            ret = np.asarray(data).flatten().astype(int)
        else:
            ret = np.asarray(data).flatten()
    if not numpy:
        return list(ret)
    return ret

def clean_name(txt):
    authorized=range(65,91)+range(48,58)+range(97,123)
    txt = str(txt)
    return "".join([(ord(letter) in authorized)*letter for letter in txt])


def colorbar(cmap="jet", cm_min=0, cm_max=1):
    if isinstance(cmap, str): cmap = cmget_cmap(cmap)
    norm = matplotlibpyplotNormalize(cm_min, cm_max)
    mappable = cmScalarMappable(cmap=cmap, norm=norm)
    mappable._A = []
    return cmap, norm, mappable

def ident(ar, *args, **kwargs): return ar

def _vv(img, masperpx, u, v, wl):
    demifov = 0.5*np.asarray(img.shape)*masperpx
    Y, X = np.meshgrid(np.linspace(demifov[0], -demifov[0], img.shape[0]), np.linspace(demifov[1], -demifov[1], img.shape[1]))
    X = X.reshape(X.shape+(1,))
    X = X.astype(np.float32)
    dum = X*u
    X = None
    Y = Y.reshape(Y.shape+(1,))
    Y = Y.astype(np.float32)
    dum += Y*v
    dum *= DEUXPI*MAS2RAD/wl
    return dum.astype(np.complex64)

def calcImgVis(img, masperpx, u, v, wl):
    """
    img: the image, (x, y) dimension
    masperpx the mas/px values for the images (x, y) in mas/px
    u, v: the coordinates on which to calculate the FT
    wl: the wavelength
    """
    dum = _vv(img, masperpx, u, v, wl)
    dum *= 1j
    dum = np.exp(dum)
    dum = dum.T
    dum *= img
    dum = dum.T
    if OLDNUMPY:
        return dum.sum(axis=0).sum(axis=0)
    else:
        return dum.sum(axis=(0,1))


def getDetails(hdu):
    allstation = hdu.data['STA_INDEX']
    ndata = allstation.shape[0]
    t3unique = 1.
    if allstation.shape[1] == 3:
        t3unique = np.exp(1./allstation[:,2])
    nunique = len(set((allstation[:,0] + 1j*allstation[:,1])*t3unique))
    nholes = np.arange(2, 100)
    if allstation.shape[1] == 3:
        nholes = nholes[np.argmin(np.abs(nholes*(nholes-1)*(nholes-2)/6-nunique))]
    else:
        nholes = nholes[np.argmin(np.abs(nholes*(nholes-1)/2-nunique))]
    nset = ndata/nunique
    if hdu.data[DATAEXTNAMES[hdu.header.get('EXTNAME')]].ndim == 1:
        nwl = 1
    else:
        nwl = hdu.data[DATAEXTNAMES[hdu.header.get('EXTNAME')]].shape[1]
    return ndata, nset, nunique, nholes, nwl

def unique(ar, precision, return_index=False, return_inverse=False, return_counts=False):
    """
    Returns the unique numbers of an array, at a certain precision

    Refer to np.unique
    """
    return np.unique(np.floor(ar/precision).astype(int)*precision, return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)


def gethduMJD(hdu, withdet=False):
    ndata, nset, nunique, nholes, nwl = getDetails(hdu)
    mjd = hdu.data['MJD']
    if mjd.std() == 0.:
        delta = (hdu.data['INT_TIME'][np.arange(0, ndata, nunique)]*nunique)
        mjd = mjd + np.clip(delta, 30., 1e99).cumsum().repeat(nunique)/SECPERDAY
    sortmjd = np.argsort(mjd)
    mjd = mjd[sortmjd]
    if withdet:
        return np.arange(nset).repeat(nunique)[sortmjd], mjd, (ndata, nset, nunique, nholes, nwl)
    else:
        return np.arange(nset).repeat(nunique)[sortmjd], mjd


def maskedshape(shape, maskTrue):
    return "%s - %s%% masked" % (shape, np.round(maskTrue*1./(np.prod(shape) + maskTrue), 3)*100)


def round_fig(x, n=1, retint=False):
    """
    Rounds x at the n-th figure. n must be >1
    ex: 1234.567 with n=3-> 1230.0
    """
    if np.iterable(x):
        x = np.asarray(x).copy()
        ff = (x!=0)
        dd = 10**(np.floor(np.log10(np.abs(x[ff])))-n+1)
        x[ff] = np.round(x[ff]/dd)
        if not retint: x[ff] *= dd
        return x
    elif x != 0:
        dd = 10**(np.floor(np.log10(np.abs(x)))-n+1)
        x = np.round(x/dd)
        if not retint: x *= dd
        return x
    else:
        return x
    

def quantile(x, q, weights=None):
    """
    Like numpy.percentile, but:
    * Values of q are quantiles [0., 1.] rather than percentiles [0., 100.]
    * scalar q not supported (q must be iterable)
    * optional weights on x
    """
    if weights is None:
        return np.percentile(x, [100. * qi for qi in q])
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        cdf = np.add.accumulate(weights[idx])
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()

def psf(lOverd, masperpx):
    """
    lOverd in radian
    masperpx in mas per px
    """
    nbpts = (lOverd*4/(masperpx*MAS2RAD))//2*2+1
    y, x = np.meshgrid(np.linspace(-1,1,nbpts), np.linspace(-1,1,nbpts))
    psf = airy(np.hypot(y,x)*2*lOverd+1e-10, 1/lOverd)**2
    return psf/psf.sum()


def airy(th, B, lam=None):
    """
    Return the visibility value (unsquared), given
    - th the angular separation in radian
    - B the baseline in m (alternatively in m/lambda)
    - lam the wavelength (alternatively None if B is already given as m/lambda)
    """
    if lam is None:
        x = np.pi*th*B
    else:
        x = np.pi*th*B/lam
    return 2*airyJ1(x)/x


def gauss2D(x, y, A, x0, y0, sigmaX, sigmaY, theta):
    sx2 = 1./sigmaX**2
    sy2 = 1./sigmaY**2
    st2 = np.sin(theta)**2
    ct2 = np.cos(theta)**2
    s2t = np.sin(2*theta)
    a = ct2*sx2 + st2*sy2
    b = s2t*(sy2 - sx2)
    c = st2*sx2 + ct2*sy2
    #a = (np.cos(theta)/sigmaX)**2 + (np.sin(theta)/sigmaY)**2
    #b = np.sin(2*theta)/sigmaY**2 - np.sin(2*theta)/sigmaX**2
    #c = (np.sin(theta)/sigmaX)**2 + (np.cos(theta)/sigmaY)**2
    return A*np.exp(-0.5*(a*(x-x0)**2 + b*(x-x0)*(y-y0) + c*(y-y0)**2))


def gauss1D(x, y, A, x0, y0, sigma):
    return A*np.exp(-0.5*((x-x0)**2 + (y-y0)**2)/sigma**2)


def BB(teff, wl, rel=False):
    """
    Give an effective temerature in K and a wavelength in m
    Returns the relative flux emitted (with respect to another temperature)
    """
    # pi*2*h*c**2 = 3.741771524664128e-16
    # h*c/kb = 0.014387769599838155
    return 1/(np.exp(0.014387769599838155/(wl*teff))-1)


class PriorWrapper(object):
    def __init__(self, fct, makenorm=True, makeunlog=False, fct_log=False, npts=1000, *args, **kwargs):
        if not callable(fct): raise Exception("Not callable")
        self.fct = fct
        self.args = args
        self.kwargs = kwargs
        self.makeunlog = makeunlog is True # apply exp on a ln-function at callback or not
        self.makenorm = makenorm is True # renorm at callback or not
        self.fct_log = fct_log is not False and makeunlog is False # whether to add or divide the normalization
        self.prior_log = kwargs.get('prior_log', False) # jeffrey
        if kwargs.get('prior_bounds', None) is None: raise Exception("No prior bounds found")
        if self.prior_log:
            self.x = np.logspace(np.log(kwargs.get('prior_bounds')[0]), np.log(kwargs.get('prior_bounds')[1]), npts, base=np.e)
        else:
            self.x = np.linspace(kwargs.get('prior_bounds')[0], kwargs.get('prior_bounds')[1], npts)
        if makenorm is True:
            if self.fct_log:
                self.normval = np.log(np.trapz(np.exp(self.fct(self.x, *self.args, **self.kwargs)), self.x))
            elif self.makeunlog:
                self.normval = np.trapz(np.exp(self.fct(self.x, *self.args, **self.kwargs)), self.x)
            else:
                self.normval = np.trapz(self.fct(self.x, *self.args, **self.kwargs), self.x)

    def __call__(self, x, *args, **kwargs):
        mergeargs = args + self.args[len(args):]
        mergekwargs = {}
        mergekwargs.update(self.kwargs, **kwargs)
        ret = self.fct(x, *mergeargs, **mergekwargs)
        if self.makeunlog: ret = np.exp(ret)
        if self.makenorm:
            if self.fct_log:
                ret -= self.normval
            else:
                ret /= self.normval
        return ret
