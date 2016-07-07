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


from oimainobject import Oimainobject as _Oimainobject
import _core
np = _core.np


_objects = ['PointSource', 'UniformDisk', 'UniformDisk2D', 'UniformRing', 'UniformRing2D', 'Gauss', 'Gauss2D', 'GaussDiff2D', 'GaussDiff', 'BGimage']

"""
Must return
a dict, with keys:

"""


class PointSource(_Oimainobject):
    _keys = ['sep','pa','cr']

    def __init__(self, name, priors={}, bounds={}, **kwargs):
        super(PointSource, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)

    def _calcCompVis(self, u, v, wl, blwl, oidata=None):
        """
        Calculates the complex visibilities of the object
        """
        oscil = self.oscil(u, v, wl)

        return oscil, 1./self.cr

    def image(self, sepmax, masperpx=None, wl=None, nbpts=101):
        im = np.zeros((nbpts, nbpts))
        ra, dec = self.to_pospx(sepmax=sepmax, nbpts=nbpts, integer=True)
        im[dec, ra] = 1./self.cr
        return im


class UniformRing(_Oimainobject):
    _keys = ['sep','pa','cr','rin','wid']

    def __init__(self, name, priors={}, bounds={}, **kwargs):
        super(UniformRing, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)

    def _calcCompVis(self, u, v, wl, blwl, oidata=None):
        """
        Calculates the complex visibilities of the object
        """
        oscil = self.oscil(u, v, wl)

        rout = self.rin + self.wid

        # analytical fourier transform for a uniform disk minus another uniform disk
        visout = _core.airy(rout*_core.MAS2RAD, blwl)
        visin = _core.airy(self.rin*_core.MAS2RAD, blwl)

        normflux = rout*rout-self.rin*self.rin

        return oscil*(visout*rout*rout-visin*self.rin*self.rin)/normflux, 1./self.cr

    def image(self, sepmax, masperpx=None, wl=None, nbpts=101): # in flux per pixel
        x, y = np.meshgrid(np.arange(nbpts), np.arange(nbpts), sparse=False, indexing='xy')
        ra, dec = self.to_pospx(sepmax=sepmax, nbpts=nbpts)
        masperpx = 2*sepmax/nbpts
        theim = np.zeros((nbpts,nbpts))
        theim[((np.hypot(x-ra, y-dec)<=(self.rin+self.wid)/masperpx) & (np.hypot(x-ra, y-dec)>=self.rin/masperpx))] = 1
        return theim/(theim.sum()*self.cr)


class UniformRing2D(_Oimainobject):
    _keys = ['sep','pa','cr','rin','wid','rat','th']

    def __init__(self, name, priors={}, bounds={}, **kwargs):
        super(UniformRing2D, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)

    def _calcCompVis(self, u, v, wl, blwl, oidata=None):
        """
        Calculates the complex visibilities of the object
        """
        oscil = self.oscil(u, v, wl)
        rout = self.rin + self.wid

        U, V = self._shearCoord(u,v, fourier=True)
        blwl = np.sqrt(U*U+V*V)/wl

        # analytical fourier transform for a uniform disk minus another uniform disk
        visout = _core.airy(rout*_core.MAS2RAD, blwl)
        visin = _core.airy(self.rin*_core.MAS2RAD, blwl)

        normflux = rout*rout-self.rin*self.rin

        return oscil*(visout*rout*rout-visin*self.rin*self.rin)/normflux, 1./self.cr

    def image(self, sepmax, masperpx=None, wl=None, nbpts=101): # in flux per pixel
        x, y = np.meshgrid(np.arange(nbpts), np.arange(nbpts), sparse=False, indexing='xy')
        ra, dec = self.to_pospx(sepmax=sepmax, nbpts=nbpts)
        x, y = self._shearCoord(x-ra, y-dec)
        masperpx = 2*sepmax/nbpts
        theim = np.zeros((nbpts,nbpts))
        theim[((np.hypot(x, y)<=(self.rin+self.wid)/masperpx) & (np.hypot(x, y)>=self.rin/masperpx))] = 1
        return theim/(theim.sum()*self.cr)


class BGimage(_Oimainobject):
    _keys = ['cr']
    _save = ['_img', 'masperpx', 'negRA']

    def __init__(self, name, img=None, masperpx=None, priors={}, bounds={}, negRA=False, totFlux=None, **kwargs):
        super(BGimage, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)
        self.masperpx = float(masperpx)
        self.negRA = bool(negRA)
        if img is None: img = kwargs.get('_img')
        self.totFlux = img.sum() if totFlux is None else float(totFlux)
        self._img = img/self.totFlux
        self._prepared = False
        if kwargs.get('oidata') is not None: self._prepare(oidata=kwargs['oidata'])

    def _calcCompVis(self, *args, **kwargs):
        return self._compvis, 1./self.cr

    def _prepare(self, oidata):
        if self._prepared: return
        self._compvis = _core.calcImgVis(img=self.img, masperpx=self.masperpx, u=oidata.uvwl['u'], v=oidata.uvwl['v'], wl=oidata.uvwl['wl'])
        self._prepared = True

    def prepare(self, oidata, force=False):
        if force: self._prepared = False
        self._prepare(oidata=oidata)

    @property
    def img(self):
        if self.negRA:
            return self._img[:,::-1]
        else:
            return self._img
    @img.setter
    def img(self, value):
        raise Exception("read-only")


    def image(self, sepmax=None, masperpx=None, wl=None, nbpts=101):
        if masperpx is None: masperpx = self.masperpx
        if sepmax is None: sepmax = np.min(self.img.shape)*masperpx
        if nbpts is None:
            nbpts = int(2.*sepmax/masperpx)
            sepmax = 0.5*nbpts*masperpx
        if np.abs(1-masperpx/self._masperpx)*nbpts > 1: raise Exception("different masperpx parameter, can't rescale image")
        if self.img.shape[0] > nbpts:
            deb_x = self.img.shape[0]//2 - nbpts//2
            deb_y = self.img.shape[1]//2 - nbpts//2
            return self.img[deb_x:deb_x+nbpts, deb_y:deb_y+nbpts]/self.cr
        elif self.img.shape[0] < nbpts:
            im = np.zeros((nbpts, nbpts))
            deb_x = nbpts//2 - self.img.shape[0]//2
            deb_y = nbpts//2 - self.img.shape[1]//2
            im[deb_x:deb_x+self.img.shape[0], deb_y:deb_y+self.img.shape[0]] = self.img
            return im/self.cr
        else:
            return self.img/self.cr

    @property
    def _masperpx(self):
        return self.masperpx


class UniformDisk(_Oimainobject):
    _keys = ['sep','pa','cr','diam']

    def __init__(self, name, priors={}, bounds={}, **kwargs):
        super(UniformDisk, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)

    def _calcCompVis(self, u, v, wl, blwl, oidata=None):
        """
        Calculates the complex visibilities of the object
        """
        oscil = self.oscil(u, v, wl)

        # analytical fourier transform for a uniform disk
        vis = _core.airy(self.diam*_core.MAS2RAD, blwl)

        return oscil*vis, 1./self.cr

    def image(self, sepmax, masperpx=None, wl=None, nbpts=101): # in flux per pixel
        x, y = np.meshgrid(np.arange(nbpts), np.arange(nbpts), sparse=False, indexing='xy')
        masperpx = 2*sepmax/nbpts
        ra, dec = self.to_pospx(sepmax=sepmax, nbpts=nbpts)
        theim = np.zeros((nbpts,nbpts))
        theim[np.hypot(x-ra,y-dec)<=self.diam/masperpx] = 1
        return theim/(theim.sum()*self.cr)



class UniformDisk2D(_Oimainobject):
    _keys = ['sep','pa','cr','diam','rat','th']

    def __init__(self, name, priors={}, bounds={}, **kwargs):
        super(UniformDisk2D, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)

    def _calcCompVis(self, u, v, wl, blwl, oidata=None):
        """
        Calculates the complex visibilities of the object
        """
        oscil = self.oscil(u, v, wl)

        U, V = self._shearCoord(u,v, fourier=True)
        blwl = np.sqrt(U*U+V*V)/wl

        # analytical fourier transform for a uniform disk
        vis = _core.airy(self.diam*_core.MAS2RAD, blwl)

        return oscil*vis, 1./self.cr

    def image(self, sepmax, masperpx=None, wl=None, nbpts=101): # in flux per pixel
        x, y = np.meshgrid(np.arange(nbpts), np.arange(nbpts), sparse=False, indexing='xy')
        ra, dec = self.to_pospx(sepmax=sepmax, nbpts=nbpts)
        x, y = self._shearCoord(x-ra, y-dec)
        masperpx = 2*sepmax/nbpts
        theim = np.zeros((nbpts,nbpts))
        theim[np.hypot(x, y)<=self.diam/masperpx] = 1
        return theim/(theim.sum()*self.cr)


class GaussDiff(_Oimainobject):
    _keys = ['sep','pa','cr','sig','dif']
    def __init__(self, name, priors={}, bounds={}, **kwargs):
        super(GaussDiff, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)

    def _calcCompVis(self, u, v, wl, blwl, oidata=None):
        """
        Calculates the complex visibilities of the object
        """
        oscil = self.oscil(u, v, wl)
        sigout = self.sig + self.dif
        
        # analytical fourier transform for a 1D gaussian blob
        ggin = _core.gauss1D(u, v, 1., 0., 0., wl/(_core.DEUXPI*_core.MAS2RAD*self.sig))
        ggout = _core.gauss1D(u, v, 1., 0., 0., wl/(_core.DEUXPI*_core.MAS2RAD*sigout))

        nin = self.sig*self.sig
        nout = sigout*sigout

        return oscil*(ggout*nout-ggin*nin)/(nout-nin), 1./self.cr

    def image(self, sepmax, masperpx=None, wl=None, nbpts=101): # in flux per pixel
        x, y = np.meshgrid(np.arange(nbpts), np.arange(nbpts), sparse=False, indexing='xy')
        ra, dec = self.to_pospx(sepmax=sepmax, nbpts=nbpts)
        theim = _core.gauss1D(x-ra, y-dec, 1., 0., 0., 0.5*nbpts*self.sig/sepmax) - _core.gauss1D(x-ra, y-dec, 1., 0., 0., 0.5*nbpts*(self.sig+self.dif)/sepmax)
        return theim/(theim.sum()*self.cr) # renorm just in case of bad sampling, so the integral is perfectly 1


class GaussDiff2D(_Oimainobject):
    _keys = ['sep','pa','cr','sig','dif','rat','th']
    def __init__(self, name, priors={}, bounds={}, **kwargs):
        super(GaussDiff2D, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)

    def _calcCompVis(self, u, v, wl, blwl, oidata=None):
        """
        Calculates the complex visibilities of the object
        """
        oscil = self.oscil(u, v, wl)
        sigout = self.sig + self.dif

        U, V = self._shearCoord(u,v, fourier=True)

        # analytical fourier transform for a 1D gaussian blob
        ggin = _core.gauss1D(U, V, 1., 0., 0., wl/(_core.DEUXPI*_core.MAS2RAD*self.sig))
        ggout = _core.gauss1D(U, V, 1., 0., 0., wl/(_core.DEUXPI*_core.MAS2RAD*sigout))

        nin = self.sig*self.sig
        nout = sigout*sigout

        return oscil*(ggout*nout-ggin*nin)/(nout-nin), 1./self.cr

    def image(self, sepmax, masperpx=None, wl=None, nbpts=101): # in flux per pixel
        x, y = np.meshgrid(np.arange(nbpts), np.arange(nbpts), sparse=False, indexing='xy')
        ra, dec = self.to_pospx(sepmax=sepmax, nbpts=nbpts)
        x, y = self._shearCoord(x-ra, y-dec)
        theim = _core.gauss1D(x, y, 1., 0., 0., 0.5*nbpts*self.sig/sepmax) - _core.gauss1D(x, y, 1., 0., 0., 0.5*nbpts*(self.sig+self.dif)/sepmax)
        return theim/(theim.sum()*self.cr) # renorm just in case of bad sampling, so the integral is perfectly 1


class Gauss(_Oimainobject):
    _keys = ['sep','pa','cr','sig']
    def __init__(self, name, priors={}, bounds={}, **kwargs):
        super(Gauss, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)

    def _calcCompVis(self, u, v, wl, blwl, oidata=None):
        """
        Calculates the complex visibilities of the object
        """
        oscil = self.oscil(u, v, wl)

        # analytical fourier transform for a 1D gaussian blob
        gg = _core.gauss1D(u, v, 1., 0., 0., wl/(_core.DEUXPI*_core.MAS2RAD*self.sig))

        return oscil*gg, 1./self.cr

    def image(self, sepmax, masperpx=None, wl=None, nbpts=101): # in flux per pixel
        x, y = np.meshgrid(np.arange(nbpts), np.arange(nbpts), sparse=False, indexing='xy')
        ra, dec = self.to_pospx(sepmax=sepmax, nbpts=nbpts)
        theim = _core.gauss1D(x-ra, y-dec, 1., 0., 0., 0.5*nbpts*self.sig/sepmax)
        # amp = 1./(self.cr*_core.DEUXPI*self.sig**2)*(2.*sepmax/(nbpts-1))**2
        return theim/(theim.sum()*self.cr) # renorm just in case of bad sampling, so the integral is perfectly 1


class Gauss2D(_Oimainobject):
    _keys = ['sep','pa','cr','sig','rat','th']
    def __init__(self, name, priors={}, bounds={}, **kwargs):
        super(Gauss2D, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)

    def _calcCompVis(self, u, v, wl, blwl, oidata=None):
        """
        Calculates the complex visibilities of the object
        """
        oscil = self.oscil(u, v, wl)

        U, V = self._shearCoord(u,v, fourier=True)

        # analytical fourier transform for a 1D gaussian blob
        gg = _core.gauss1D(U, V, 1., 0., 0., wl/(_core.DEUXPI*_core.MAS2RAD*self.sig))

        return oscil*gg, 1./self.cr

    def image(self, sepmax, masperpx=None, wl=None, nbpts=101): # in flux per pixel
        x, y = np.meshgrid(np.arange(nbpts), np.arange(nbpts), sparse=False, indexing='xy')
        ra, dec = self.to_pospx(sepmax=sepmax, nbpts=nbpts)
        x, y = self._shearCoord(x-ra, y-dec)
        theim = _core.gauss1D(x, y, 1., 0., 0., 0.5*nbpts*self.sig/sepmax)
        return theim/(theim.sum()*self.cr) # another step of normalization, in case the sum is not really =1 (bad sampling)


"""
class PreObject(_Oimainobject):
    _keys = ['cr']

    def __init__(self, name, data, au, px, keysInd, keysVal={}, priors={}, bounds={}, **kwargs):
        super(PreObject, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)
        self._keys += keysInd.keys()
        self._keysInd = dict(keysInd)
        self._keysVal = dict(keysVal)
        self.au = float(au)
        self.px = float(px)
        self.data = np.asarray(data)
        self._myslice = tuple(slice(0, i) for i in self.data.shape)
        dum = range(self._nparams)
        for k,v in keysInd.items(): dum[v]=-1
        self.dataaxis = np.max(dum)
        self._selInter = tuple(slice(0, 2) if i!=self.dataaxis else slice(0,s) for i,s in enumerate(self.data.shape))

    def _calcCompVis(self, *args, **kwargs):
        myslice = list(self._myslice)
        for k,v in self._keysInd.items():
            myslice[v] = slice(int(getattr(self, k)), int(getattr(self, k))+2)
        dum = self.data[myslice].copy()
        hop = list(self._selInter)
        for k,v in self._keysInd.items():
            hop[v] = slice(0,1)
            dum = dum[hop] + (dum.take([1], axis=v) - dum[hop]) * (getattr(self, k) - int(getattr(self, k)))
        return self._compvis, 1./self.cr



class PointSourceLinCR(_Oimainobject):
    _keys = ['sep','pa','cr1','cr2']

    def __init__(self, name, priors={}, bounds={}, **kwargs):
        super(PointSourceLinCR, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)

    def _calcCompVis(self, u, v, wl, blwl, oidata):
        "Calculates the complex visibilities of the object"
        oscil = self.oscil(u, v, wl)

        flx = oidata._wlspan / ((wl - oidata._wlmin) * (self.cr2 - self.cr1) + self.cr1 * oidata._wlspan)

        return oscil, flx

    def LinCR2Teff(self, wl, cr1, cr2):
        "Out of wl_min, wl_max, cr1 and cr2, determines the 2 effective temperatures of the components"
        pass


class UniformDiskLinCR(_Oimainobject):
    _keys = ['sep','pa','cr1','cr2','diam']

    def __init__(self, name, priors={}, bounds={}, **kwargs):
        super(UniformDiskLinCR, self).__init__(name=name, priors=priors, bounds=bounds, **kwargs)

    def _calcCompVis(self, u, v, wl, blwl, oidata=None):
        "Calculates the complex visibilities of the object"
        oscil = self.oscil(u, v, wl)

        vis = _core.airy(self.diam*_core.MAS2RAD, blwl)

        flx = oidata._wlspan / ((wl - oidata._wlmin) * (self.cr2 - self.cr1) + self.cr1 * oidata._wlspan)

        return oscil*vis, flx


class Spectral(object):
    def __init__(self, allwl=None, spectralKey='cr', **kwargs):
        if str(spectralKey) not in self._initkeys: raise Exception("not such key: %s" % spectralKey)
        self._spectralKey = str(spectralKey)
        self._keys = list(self._initkeys) # makes a copy
        self._keys.remove(self.spectralKey)
        self._wlspectral = np.sort(list(set(np.ravel(allwl))))
        self.nspectral = self._wlspectral.size
        self.wlspectral = {}
        self._wlspectralindex = {}
        for i, wl in enumerate(self._wlspectral):
            self.wlspectral[self.spectralKey+str(i)] = wl
            self._wlspectralindex[self.spectralKey+str(i)] = (allwl==wl)
            self._keys += [self.spectralKey+str(i)]

    @property
    def spectralKey(self):
        return self._spectralKey
    @spectralKey.setter
    def spectralKey(self, value):
        raise Exception("read-only")

    def _single2spectral(self, priors, bounds):
        specval = priors.get(self.spectralKey)
        if specval is not None:
            priors.pop(self.spectralKey)
            specval = np.asarray(specval) # pretty array
            if specval.size!=self.nspectral and specval.size!=1: # if some specval initialization missing
                print(font.orange+"WARNING"+font.normal+": contrast ratio size '%s' should be same size as wavelengths '%s' in '%s'. Parameter set to None." % (specval.size, self.nspectral, name))
                specval = None
            else:
                if specval.size==1: specval = specval*np.ones(self.nspectral) # just one value of specval param, flatten it on all wl
                for i, spectralit in enumerate(specval):
                    if spectralit is not None: # this value of param is not known
                        priors[self.spectralKey+str(i)] = spectralit
        
        specval = bounds.get(self.spectralKey)
        if specval is not None:
            bounds.pop(self.spectralKey)
            specval = np.asarray(specval) # pretty array
            ntotelmt = np.sum([np.size(item) for item in specval])
            if specval.shape[0]!=self.nspectral and specval.shape[0]!=ntotelmt: # if some specval initialization missing
                specval = None
                print(font.orange+"WARNING"+font.normal+": contrast ratio size '%s' should be same size as wavelengths '%s' in '%s'. Bound set to None." % (specval.size, self.nspectral, name))
            else:
                if specval.shape[0]==ntotelmt: specval = np.tile(specval, self.nspectral).reshape((self.nspectral, -1)) # just one value of specval bound, copy it on all wl
                for i, spectralit in enumerate(specval):
                    if spectralit is not None: # this value of param is not known
                        bounds[self.spectralKey+str(i)] = spectralit
        return priors, bounds


class PointSourceSpectral(PointSource, Spectral):
    _initkeys = ['sep','pa', 'cr']

    def __init__(self, name, oidata, priors={}, bounds={}, **kwargs):
        if not isinstance(oidata, Oidata): raise Exception("oidata must be Oidata type")
        Spectral.__init__(self, allwl=oidata.uvwl['wl'], **kwargs)
        priors, bounds = self._single2spectral(priors, bounds)
        PointSource.__init__(self, name=name, priors=priors, bounds=bounds, **kwargs)

    @property
    def cr(self):
        ret = np.ones(self._wlspectralindex['cr0'].shape)
        for i in range(self.nspectral):
            ret[self._wlspectralindex['cr'+str(i)]] = getattr(self, 'cr'+str(i))
        return ret
    @cr.setter
    def cr(self, value):
        raise Exception("Read-only")
    
"""
