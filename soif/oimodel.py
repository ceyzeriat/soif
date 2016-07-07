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


import matplotlib.pyplot as _plt
from matplotlib.gridspec import GridSpec as _matplotlibGridspecGridSpec
from matplotlib.patches import FancyArrowPatch as _matplotlibPatchesFancyArrowPatch
from oiunitmodels import _objects
from _oiunitmodels import *
from oimainobject import Oimainobject as _Oimainobject
from copy import deepcopy as _deepcopy
from time import strftime as _strftime
try:
    import astropy.io.fits as _pf
except ImportError:
    import pyfits as _pf

import _core
_np = _core.np


class Oimodel(object):
    def __init__(self, oidata, objs=[], tweakparams=None):
        self._objs = []
        self.oidata = oidata
        if tweakparams is not None and not callable(tweakparams): raise Exception("tweakparams shall be callable")
        self._tweakparams = tweakparams
        self.nobj = 0
        self._nparamsObj = 0
        if not hasattr(objs, "__iter__"):
            self.add_obj(objs)
        else:
            for item in objs:
                self.add_obj(item)

    def _info(self):
        return _core.font.blue+"<Oifiting Model>%s\n %s objects:\n  %s\n%s"%(_core.font.normal, self.nobj, "\n  ".join(map(str, self._objs)), str(self.oidata))
    def __repr__(self):
        return self._info()
    def __str__(self):
        return self._info()


    @property
    def nparams(self):
        return self._nparamsObj + self.oidata.systematic_fit
    @nparams.setter
    def nparams(self, value):
        raise AttributeError("Read-only")

    @property
    def nparamsObjs(self):
        self._nparamsObj = 0
        for item in self._objs: self._nparamsObj += item._nparams
        return self._nparamsObj
    @nparamsObjs.setter
    def nparamsObjs(self, value):
        raise AttributeError("Read-only")
    

    def add_obj(self, typ, name=None, params={}, prior={}):
        """
        Add an object to the model
        """
        # gets the future new name of the object
        if isinstance(typ, _Oimainobject):
            name = typ.name
        else:
            name = _core.clean_name(name)
        # check for already existing name
        if hasattr(self, "o_"+name): raise NameError("Object name already exists")
        # if all good, proceeds
        if isinstance(typ, _Oimainobject): # if first argument is already a built object
            self._objs.append(_deepcopy(typ))
        elif typ in _objects: # if we need to build the object, and it exists
            self._objs.append(globals()[typ](name=name, params=params, prior=prior))
        else:
            print(_core.font.red+"ERROR: Could not find the object name for '%s', name given: %s%s" % (typ, name, _core.font.normal))
            return
        setattr(self, "o_"+name, self._objs[-1]) # quick access as a class property
        if hasattr(self._objs[-1], '_prepare'): self._objs[-1]._prepare(oidata=self.oidata)
        self.nobj += 1
        dum = self.nparamsObjs
    
    def del_obj(self, idobj):
        """
        Delete an object from the model.
        idobj can be the name of the object or its index in the model list 
        """
        if not isinstance(idobj, int):
            name = _core.clean_name(name)
            if not hasattr(self, "o_"+name): raise NameError("Didn't find object name")
            ind = self._objs.index(getattr(self, "o_"+name))
        else:
            ind = name
            name = self._objs[ind].name
        dummy = self._objs.pop(ind)
        delattr(self, "o_"+name)
        print(_core.font.blue+"Deleted object '%s'." % (name)+_core.font.normal)
        self.nobj -= 1
        dum = self.nparamsObjs

    def getP0(self):
        """
        Return an initialized param list
        """
        ret = []
        for item in self._objs:
            ret += item.getP0()
        if self.oidata.systematic_fit: ret += [self.oidata.systematic_p0()]
        return ret


    @property
    def paramstr(self):
        ret = []
        for item in self._objs:
            for arg in item._pkeys:
                ret.append(item.name+"_"+arg)
        if self.oidata.systematic_fit: ret += ["sys"]
        return ret
    @paramstr.setter
    def paramstr(self, value):
        raise Exception("Read-only")


    @property
    def params(self):
        """
        Return current params
        """
        ret = []
        for item in self._objs:
            ret += getattr(item, "params", [])
        if self.oidata.systematic_fit: ret += [self.oidata.systematic_prior if self.oidata.systematic_prior is not None else self.oidata.systematic_p0()]
        return ret
    @params.setter
    def params(self, value):
        self.setParams(value)


    def reinit(self, params):
        self.setParams(params=params, priors=True)


    def statut(self, params, customlike=None, **kwargs):
        self.setParams(params)
        chi2 = self.likelihood(params=params, customlike=customlike, chi2=True, **kwargs)
        print('Khi-2: %.3f' % chi2)
        print('Khi-2 NULL: %.3f' % self.likelihood(params=params, customlike=customlike, chi2=True, null=True, **kwargs))
        print('ln-like: %.3f' % self.likelihood(params=params, customlike=customlike, chi2=False, **kwargs))
        print('ln-like NULL: %.3f' % self.likelihood(params=params, customlike=customlike, chi2=False, null=True, **kwargs))
        print('Number of parameters (k): %i' % self._nparamsObj)


    def setParams(self, params, priors=False):
        if len(params) != self.nparams: raise Exception("params has not the correct number of parameters")
        parampos = 0
        for item in self._objs:
            item.setParams(params=params[parampos:parampos+item._nparams], priors=priors)
            parampos += item._nparams
        if self.oidata.systematic_fit:
            self.oidata.systematic_prior = params[parampos]
            self.oidata._systematic_prior = self.oidata.systematic_prior


    def compVis(self, params=None):
        """
        Calculate the complex visibility of the model from each separate object
        """
        if params is None: params = self.getP0() # initialize at p0 in case no params is given
        if self._tweakparams is not None: self._tweakparams(self, params)
        parampos = 0
        totflx = 0.
        totviscomp = _np.zeros(self.oidata.uvwl['u'].shape, dtype=complex) # initialize array
        for item in self._objs:
            viscomp, flx = item.compVis(oidata=self.oidata, params=params[parampos:parampos+item._nparams], flat=True)
            totviscomp += viscomp*flx
            totflx += flx
            parampos += item._nparams
        totviscomp /= totflx
        return self.oidata.remorph(totviscomp)

    def compuvimage(self, blmax, wl=None, params=None, nbpts=101):
        parampos = 0
        totFluxvis2 = 0.
        if wl is None: wl = self.oidata._wlmin+self.oidata._wlspan*0.5
        compvis = _np.zeros((nbpts, nbpts), dtype=complex) # initialize array
        u, v = _np.meshgrid(_np.linspace(-blmax, blmax, nbpts), _np.linspace(-blmax, blmax, nbpts))
        if self._tweakparams is not None: self._tweakparams(self, params)
        if params is not None: self.setParams(params)
        for item in self._objs:
            try:
                vis2 = item._calcCompVis(u=u, v=v, wl=wl, blwl=_np.hypot(u, v)/wl)
            except AttributeError:
                raise AttributeError("it looks like some parameters are not initialized. Input your parameters after params=[...]")
            compvis += vis2[0]*vis2[1]
            totFluxvis2 += vis2[1]
            parampos += item._nparams
        return compvis/totFluxvis2

    def compimage(self, params=None, sepmax=None, wl=None, masperpx=None, nbpts=101, psfConvolve=None, **kwargs):
        """
        psfConvolve in mas (lambda/D)
        """
        parampos = 0
        totFluxvis2 = 0.
        # check for set-resolution objects
        masperpxfixed = None
        nbptscheck = nbpts
        if nbpts is None: nbptscheck = int(2.*sepmax/masperpx)
        for item in self._objs:
            if hasattr(item, "_masperpx"):
                if masperpxfixed is None:
                    masperpxfixed = item._masperpx
                elif _np.abs(1-masperpxfixed/item._masperpx)*nbptscheck > 1:
                    raise Exception("can't return an image, it seems that several non-scalable images do not have the save masperpx")
        if masperpxfixed is not None: masperpx = masperpxfixed
        # check resolution parameters
        if sepmax is None and masperpx is None and nbpts is not None:
            sepmax = 1000*nbpts/(2*_np.hypot(self.oidata.uvwl['v'], self.oidata.uvwl['u']).max())
        if masperpx is not None and sepmax is not None:
            nbpts = int(2.*sepmax/masperpx)
        elif masperpx is not None and nbpts is not None:
            pass
        elif sepmax is not None and nbpts is not None:
            masperpx = sepmax*2./nbpts
        else:
            raise Exception("can't determine the resolution, nbpts should not be None")
        sepmax = 0.5*nbpts*masperpx
        # check other parameters
        if wl is None: wl = self.oidata.uvwl['wl'].min()
        # initialize array
        img = _np.zeros((nbpts, nbpts))
        if self._tweakparams is not None: self._tweakparams(self, params)
        if params is not None: self.setParams(params)
        for item in self._objs:
            try:
                img += item.image(sepmax=sepmax, masperpx=masperpx, wl=wl, nbpts=nbpts)
            except AttributeError:
                raise AttributeError("it looks like some parameters are not initialized. Input your parameters after params=[...]")
            parampos += item._nparams
        if psfConvolve is not None:
            if len(_core.aslist(psfConvolve))==1:
                psf = _core.psf(float(psfConvolve)*_core.MAS2RAD, masperpx)
            elif len(_core.aslist(psfConvolve))==3:
                sx, sy, th = map(float, psfConvolve[:3])
                y, x = _np.meshgrid(_np.arange(nbpts)-nbpts//2, _np.arange(nbpts)-nbpts//2)
                psf = _core.gauss2D(x, y, 1, 0, 0, sx*0.5/masperpx, sy*0.5/masperpx, th)
            img = _core.fftconvolve(img, psf, mode='same')
        if kwargs.get('retresol', False): return img, (sepmax, masperpx, nbpts)
        return img


    def image(self, params=None, sepmax=None, wl=None, masperpx=None, nbpts=101, cmap='jet', cm_min=None, cm_max=None, ret=False, visu=None, psfConvolve=None, **visuargs):
        """
        psfConvolve in mas (lambda/D)
        """
        # check other params
        if visu is None: visu = _core.ident
        if not callable(visu):
            print(_core.font.red+"ERROR: visu parameter must be callable"+_core.font.normal)
            return
        toplot, (sepmax, masperpx, nbpts) = self.compimage(sepmax=sepmax, wl=wl, params=params, nbpts=nbpts, psfConvolve=psfConvolve, retresol=True)
        toplot = visu(toplot, **visuargs)
        if cm_min is None: cm_min = toplot.min()
        if cm_max is None: cm_max = toplot.max()
        cmap, norm, mappable = _core.colorbar(cmap=cmap, cm_min=cm_min, cm_max=cm_max)
        thefig, ax = _core.astroskyplot(sepmax, polar=False, unit='mas')
        ax.matshow(toplot, origin='lower', extent=[-sepmax,sepmax,-sepmax,sepmax], cmap=cmap, norm=norm)
        _plt.colorbar(mappable)
        if ret: return toplot


    def uvimage(self, params=None, blmax=None, wl=None, typ='vis2', nbpts=101, cmap='jet', cm_min=None, cm_max=None, ret=False, visu=None, **visuargs):
        """
        typ can be: vis, vis2, phase
        """
        typdic = {'vis2':_core.abs2, 'phase':_np.angle, 'vis':_np.abs}
        # check other params
        if visu is None: visu = _core.ident
        if not callable(visu):
            print(_core.font.red+"ERROR: visu parameter must be callable"+_core.font.normal)
            return
        toplot = self.compuvimage(blmax=blmax, wl=wl, params=params, nbpts=nbpts)
        toplot = visu(typdic.get(typ, typdic['vis2'])(toplot), **visuargs)
        if cm_min is None: cm_min = toplot.min()
        if cm_max is None: cm_max = toplot.max()
        cmap, norm, mappable = _core.colorbar(cmap=cmap, cm_min=cm_min, cm_max=cm_max)
        thefig, ax = _core.astroskyplot(blmax, polar=False, unit='m')
        ax.matshow(toplot, origin='lower', extent=[-blmax,blmax,-blmax,blmax], cmap=cmap, norm=norm)
        _plt.colorbar(mappable)
        if ret: return toplot

    def residual(self, params, c=None, cmap='jet', cm_min=None, cm_max=None, datatype='All'):
        calcindex = {'vis2':0, 't3phi':1, 't3amp':2, 'visphi':3, 'visamp':4}
        fullmodel = self.compVis(params=params)
        cm_min_orig = cm_min
        cm_max_orig = cm_max
        if datatype.lower() == 'all':
            datatype = _core.DATAKEYSLOWER
        elif _np.iterable(datatype)==1 and not isinstance(datatype, str):
            datatype = [str(dum).lower() for dum in datatype if dum.lower() in _core.DATAKEYSLOWER]
        else:
            datatype = [str(dum).lower() for dum in [str(datatype)] if dum.lower() in _core.DATAKEYSLOWER]
        if self.oidata.systematic_fit: self.oidata.systematic_prior = params[self._nparamsObj]
        for index, datatypeloop in enumerate(datatype):
            if not getattr(self.oidata, datatypeloop): continue
            data = getattr(self.oidata, datatypeloop).data
            model = fullmodel[calcindex[datatypeloop]]
            error = getattr(self.oidata, datatypeloop).error
            if self.oidata.systematic_fit: error = _np.sqrt(error*error + self.oidata.systematic_prior*self.oidata.systematic_prior)
            thefig = _plt.figure()
            if c is not None:
                grid = _matplotlibGridspecGridSpec(5, 21)
            else:
                grid = _matplotlibGridspecGridSpec(5, 20)
            theax = thefig.add_subplot(grid[0:4,0:20])
            theotherax = thefig.add_subplot(grid[4,0:20], sharex=theax)
            for k, v in _np.ndenumerate(data):
                theax.add_patch(_matplotlibPatchesFancyArrowPatch((model[k], data[k]),(model[k], data[k]+error[k]), arrowstyle='-['))
                theax.add_patch(_matplotlibPatchesFancyArrowPatch((model[k], data[k]),(model[k], data[k]-error[k]), arrowstyle='-['))
            mini = min(data.min(), model.min())
            maxi = max(data.max(), model.max())
            theax.plot([mini, maxi], [mini, maxi], 'g-', lw=2)
            if c is not None:
                thecb = thefig.add_subplot(grid[:,20])
                thecb.set_xticklabels('')
                thecb.yaxis.tick_right()
                colorkeyattr = str(c)
                if not hasattr(getattr(self.oidata, datatypeloop), colorkeyattr): colorkeyattr = 'wl'
                colorattr = getattr(getattr(self.oidata, datatypeloop), colorkeyattr)
                #if datatypeloop[:2]=='t3':
                #    if colorattr.shape != self.oidata._t3outputshape:
                #        colorattr = colorattr.mean(axis=-1)
                if cm_min_orig is None: cm_min = colorattr.min()
                if cm_max_orig is None: cm_max = colorattr.max()
                cmap, norm, mappable = _core.colorbar(cmap=cmap, cm_min=cm_min, cm_max=cm_max)
                theax.scatter(x=model, y=data, s=30, c=colorattr, cmap=cmap, norm=norm, marker='o', edgecolors='none', alpha=1)
                div = _np.true_divide(model-data, error)
                theotherax.scatter(x=model, y=self.oidata.erb_sigma(_np.abs(div))*_np.sign(div), s=30, c=colorattr, cmap=cmap, norm=norm, marker='o', edgecolors='none', alpha=1)
                thefig.colorbar(mappable=mappable, cax=thecb)
                colortitle = ', color= '+colorkeyattr
            else:
                theax.scatter(x=model, y=data, s=30, marker='o', edgecolors='none', alpha=1)
                div = _np.true_divide(model-data, error)
                theotherax.scatter(x=model, y=self.oidata.erb_sigma(_np.abs(div))*_np.sign(div), s=30, marker='o', edgecolors='none', alpha=1)
                colortitle = ''
            theotherax.grid(True)
            theax.set_title(datatypeloop+colortitle)


    #def grid(self, param_x, param_y, bin_x=50, bin_y=50, mode=_np.max, cmap="jet"):
    #    pass

    def likelihood(self, params, customlike=None, chi2=False, **kwargs):
        kwargs['chi2'] = chi2
        return _likelihood(params=params, model=self, customlike=customlike, kwargs=kwargs)


    def save(self, filename, clobber=False):
        hdulist = _pf.HDUList()
        ext = '.oif.fits'
        if filename.find(ext)==-1: filename += ext
        hdu = _pf.PrimaryHDU()
        hdu.header.set('EXT', 'MODEL', comment='Type of information in the HDU')
        hdu.header.set('DATE', _strftime('%Y%m%dT%H%M%S'), comment='Creation Date')
        hdu.header.set('NOBJ', self.nobj, comment='Number of objects in the model')
        hdu.header.set('NPARAMS', self.nparams, comment='Total number of free parameters')
        for i, item in enumerate(self._objs):
            hdu.header.set('OBJ'+str(i), item.name, comment='Name of object '+str(i))
        allmodes = ['vis2', 't3phi', 't3amp', 'visphi', 'visamp']
        for mode in allmodes:
            number = []
            if getattr(self.oidata, mode):
                number = [len(i) for i in getattr(self.oidata, "_input_"+mode) if i is not None]
            hdu.header.set('N'+mode, int(_np.sum(number)), comment='Total number of '+mode+' measurements')
            hdu.header.set('F_'+mode, len(number), comment='Number of '+mode+' files')

        hdu.header.add_comment('Written by Guillaume SCHWORER')
        hdulist.append(hdu)

        hdulist.writeto(filename, clobber=clobber)

        for item in self._objs:
            item.save(filename, append=True, clobber=clobber)

        self.oidata.save(filename, append=True, clobber=clobber)

        return filename


def _likelihood(params, model, customlike=None, kwargs={}):
    '''
    Calculate the likelihood of the model versus the data.
    Use customlike and kwargs parameters for a custom likelihood function.
    This customlike function must accept input params:
    - params (list of the p values for that step)
    - modeledData (tuple of model vis2, t3phi, t3amp, visphi, visamp given the p value of the parameters). The uncalculated data is set to None.
    - model (oifiting object)

    And must return the log-likelihood of that step
    '''
    null = kwargs.pop('null', False)
    chi2 = kwargs.pop('chi2', False)
    quality = 0.
    ln_prior = 0.
    if model.oidata.systematic_fit:
        model.oidata.systematic_prior = params[model._nparamsObj]
    if null: # if we just want NULL hypothesis
        vis2, t3phi, t3amp, visphi, visamp = 1., 0., 1., 0., 1.
    else:
        parampos = 0
        for item in model._objs:
            for arg in item._pkeys:
                pb = getattr(item, arg+'_bounds')
                if not (pb[0] <= params[parampos] <= pb[1]):
                    return -_np.inf # exit with -inf if params outside bounds
                parampos += 1
        if model.oidata.systematic_fit:
            if not (model.oidata.systematic_bounds[0] <= model.oidata.systematic_prior <= model.oidata.systematic_bounds[1]):
                return -_np.inf # exit with -inf if params outside bounds
            #parampos += 1
        if customlike is not None: return customlike(params=params, model=self, **kwargs)
        if not chi2:
            pass
            #ln_prior += getattr(item, arg+'_prior_lnfunc')(params[parampos], params=params, parampos=parampos, **getattr(item, arg+'_prior_kwargs'))
        if model.nobj==0:
            vis2, t3phi, t3amp, visphi, visamp = 1., 0., 1., 0., 1.
        else:
            vis2, t3phi, t3amp, visphi, visamp = model.compVis(params=params)
    if model.oidata.vis2:
        if model.oidata.systematic_fit:
            invvar = 1./(model.oidata.vis2.error*model.oidata.vis2.error + model.oidata.systematic_prior*model.oidata.systematic_prior)
        else:
            invvar = model.oidata.vis2._invvar
        if chi2:
            quality += ((model.oidata.vis2.data - vis2)**2*invvar).sum()
        else:
            quality += ((model.oidata.vis2.data - vis2)**2*invvar - _np.log(invvar)).sum()
    if model.oidata.visphi:
        if model.oidata.systematic_fit:
            invvar = 1./(model.oidata.visphi.error*model.oidata.visphi.error + model.oidata.systematic_prior*model.oidata.systematic_prior)
        else:
            invvar = model.oidata.visphi._invvar
        if chi2:
            quality += ((model.oidata.visphi.data - visphi)**2*invvar).sum()
        else:
            quality += ((model.oidata.visphi.data - visphi)**2*invvar - _np.log(invvar)).sum()
    if model.oidata.visamp:
        if model.oidata.systematic_fit:
            invvar = 1./(model.oidata.visamp.error*model.oidata.visamp.error + model.oidata.systematic_prior*model.oidata.systematic_prior)
        else:
            invvar = model.oidata.visamp._invvar
        if chi2:
            quality += ((model.oidata.visamp.data - visamp)**2*invvar).sum()
        else:
            quality += ((model.oidata.visamp.data - visamp)**2*invvar - _np.log(invvar)).sum()
    if model.oidata.t3phi:
        if model.oidata.systematic_fit:
            invvar = 1./(model.oidata.t3phi.error*model.oidata.t3phi.error + model.oidata.systematic_prior*model.oidata.systematic_prior)
        else:
            invvar = model.oidata.t3phi._invvar
        if chi2:
            quality += ((model.oidata.t3phi.data - t3phi)**2*invvar).sum()
        else:
            quality += ((model.oidata.t3phi.data - t3phi)**2*invvar - _np.log(invvar)).sum()
    if model.oidata.t3amp:
        if model.oidata.systematic_fit:
            invvar = 1./(model.oidata.t3amp.error*model.oidata.t3amp.error + model.oidata.systematic_prior*model.oidata.systematic_prior)
        else:
            invvar = model.oidata.t3amp._invvar
        if chi2:
            quality += ((model.oidata.t3amp.data - t3amp)**2*invvar).sum()
        else:
            quality += ((model.oidata.t3amp.data - t3amp)**2*invvar - _np.log(invvar)).sum()
    if chi2:
        return quality
    else:
        return ln_prior - 0.5*quality # log like
