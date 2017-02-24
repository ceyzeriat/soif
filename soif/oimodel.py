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


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as _matplotlibGridspecGridSpec
from matplotlib.patches import FancyArrowPatch as _matPatFancyArrowPatch
from copy import deepcopy
from time import strftime
try:
    import astropy.io.fits as pf
except ImportError:
    import pyfits as pf

from .oiunitmodels import *
from .oimainobject import Oimainobject
from . import oiexception as exc
from . import core
from .oifits import Oifits
np = core.np

__all__ = ['Oimodel']


class Oimodel(object):
    def __init__(self, oidata=None, objs=[], tweakparams=None, **kwargs):
        self._objs = []
        self.raiseError = bool(kwargs.pop('raiseError', True))
        if isinstance(oidata, Oifits):
            self.oidata = oidata
        else:
            self.oidata = None
        if tweakparams is not None and not callable(tweakparams):
            if exc.raiseIt(exc.NotCallable, self.raiseError, fct="tweakparams"):
                return False
        self._tweakparams = tweakparams
        self.nobj = 0
        self._nparamsObj = 0
        if not hasattr(objs, "__iter__"):
            self.add_obj(objs)
        else:
            for item in objs:
                self.add_obj(item)

    def _info(self):
        txt = core.font.blue+"<SOIF Model>{}\n {} objects:\n  {}".format(
                            core.font.normal,
                            self.nobj,
                            "\n  ".join(map(str, self._objs)))
        if self._hasdata:
            return "{}\n{}".format(txt, str(self.oidata))
        else:
            return txt

    def __repr__(self):
        return self._info()

    __str__ = __repr__

    @property
    def _hasdata(self):
        return self.oidata is not None

    @property
    def nparams(self):
        if self._hasdata:
            return self.nparamsObjs + int(self.oidata.systematic_fit)
        else:
            return self.nparamsObjs

    @nparams.setter
    def nparams(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="nparams")

    @property
    def nparamsObjs(self):
        self._nparamsObj = 0
        for item in self._objs:
            self._nparamsObj += item._nparams
        return self._nparamsObj

    @nparamsObjs.setter
    def nparamsObjs(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="nparamsObjs")

    def add_obj(self, typ, name=None, params={}, prior={}):
        """
        Adds an object to the model
        """
        # gets the future new name of the object
        if isinstance(typ, Oimainobject):
            name = typ.name
        else:
            name = core.clean_name(name)
        # check for already existing name
        if hasattr(self, "o_"+name):
            if exc.raiseIt(exc.BusyName, self.raiseError, name=name):
                return
        # if all good, proceeds
        if isinstance(typ, Oimainobject):  # if first argument is already a built object
            self._objs.append(deepcopy(typ))
        elif typ in _objects:  # if we need to build the object, and it exists
            self._objs.append(globals()[typ](name=name, params=params, prior=prior))
        else:
            if exc.raiseIt(exc.InalidUnitaryModel, self.raiseError, typ=typ):
                return
        setattr(self, "o_"+name, self._objs[-1])  # quick access as a class property
        if hasattr(self._objs[-1], 'prepare') and self._hasdata:
            self._objs[-1].prepare(oidata=self.oidata)
        self.nobj += 1
        # dum = self.nparamsObjs

    def del_obj(self, idobj):
        """
        Delete an object from the model.
        idobj can be the name of the object or its index in the model
        list.
        """
        if not isinstance(idobj, int):
            name = core.clean_name(idobj)
            if not hasattr(self, "o_"+name):
                if exc.raiseIt(exc.NotFoundName, self.raiseError, name=name):
                    return
            ind = self._objs.index(getattr(self, "o_"+name))
        else:
            ind = idobj
            name = self._objs[ind].name
        self._objs.pop(ind)
        delattr(self, "o_"+name)
        print("{}Deleted object '{}'{}.".format(core.font.blue, name, core.font.normal))
        self.nobj -= 1
        dum = self.nparamsObjs

    def getP0(self):
        """
        Return an initialized param list
        """
        ret = []
        for item in self._objs:
            ret += item.getP0()
        if getattr(self.oidata, "systematic_fit", False):
            ret += [self.oidata.systematic_p0()]
        return ret


    @property
    def paramstr(self):
        ret = []
        for item in self._objs:
            for arg in item._pkeys:
                ret.append(item.name+"_"+arg)
        if getattr(self.oidata, "systematic_fit", False):
            ret += ["sys"]
        return ret
    @paramstr.setter
    def paramstr(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="paramstr")


    @property
    def params(self):
        """
        Return current params
        """
        ret = []
        for item in self._objs:
            ret += getattr(item, "params", [])
        if getattr(self.oidata, "systematic_fit", False):
            fiterrors = self.oidata.systematic_prior is not None
            ret += [self.oidata.systematic_prior if fiterrors else self.oidata.systematic_p0()]
        return ret
    @params.setter
    def params(self, value):
        self.setParams(value)


    def reinit(self, params):
        self.setParams(params=params, priors=True)


    def statut(self, params, customlike=None, **kwargs):
        self.setParams(params)
        chi2 = self.likelihood(params=params, customlike=customlike, chi2=True, **kwargs)
        print('Khi-2: {:.3f}'.format(chi2))
        print('Khi-2 NULL: {:.3f}'
            .format(self.likelihood(params=params, customlike=customlike, chi2=True, null=True, **kwargs)))
        print('ln-like: {:.3f}'.format(self.likelihood(params=params, customlike=customlike, chi2=False, **kwargs)))
        print('ln-like NULL: {:.3f}'
            .format(self.likelihood(params=params, customlike=customlike, chi2=False, null=True, **kwargs)))
        print('Number of parameters (k): {:d}'.format(self._nparamsObj))


    def setParams(self, params, priors=False):
        if len(params) != self.nparams:
            if exc.raiseIt(exc.BadParamsSize, self.raiseError, size=self.nparams):
                return
        parampos = 0
        for item in self._objs:
            item.setParams(params=params[parampos:parampos+item._nparams], priors=priors)
            parampos += item._nparams
        if getattr(self.oidata, "systematic_fit", False):
            self.oidata.systematic_prior = params[parampos]
            self.oidata._systematic_prior = self.oidata.systematic_prior


    def compVis(self, params=None, u=None, v=None, wl=None):
        """
        Calculates the complex visibility of the model from all unitary models
        """
        if self._hasdata:
            totviscomp = self._compVis(u=self.oidata.uvwl['u'], v=self.oidata.uvwl['v'], wl=self.oidata.uvwl['wl'],
                blwl=self.oidata.uvwl['blwl'], params=params)
            return self.oidata.remorph(totviscomp)
        else:
            if u is None or v is None or wl is None:
                if exc.raiseIt(exc.NoDataModel, self.raiseError):
                    return
            else:
                return self._compVis(u=u, v=v, wl=wl, params=params)


    def calcuvimage(self, blmax, wl, params=None, nbpts=101):
        """
        Outputs the complex visibility for a grid a (u,v) in [-blmax,blmax], with nbpts in each dimension
        """
        u, v = np.meshgrid(np.linspace(-blmax, blmax, nbpts), np.linspace(-blmax, blmax, nbpts))
        return self._compVis(u=u, v=v, wl=wl, blwl=np.hypot(u, v)/wl, params=params)


    def _compVis(self, u, v, wl, blwl, params=None):
        parampos = 0
        totFluxvis = 0.
        totviscomp = np.zeros(u.shape, dtype=complex) # initialize array
        if params is None:
            params = self.getP0() # initialize at p0 in case no params is given
        else:
            self.setParams(params)
        if self._tweakparams is not None:
            self._tweakparams(self, params)
        for item in self._objs:
            compvis, flx = item._calcCompVis(u=u, v=v, wl=wl, blwl=blwl)
            compvis *= flx
            totviscomp += compvis
            totFluxvis += flx
            parampos += item._nparams
        totviscomp /= totFluxvis
        return totviscomp


    def calcimage(self, params=None, sepmax=None, wl=None, masperpx=None, nbpts=101, psfConvolve=None, **kwargs):
        """
        psfConvolve in mas (lambda/D)
        """
        parampos = 0
        totFluxvis = 0.
        # check for set-resolution objects
        masperpxfixed = None
        nbptscheck = nbpts
        if nbpts is None: nbptscheck = int(2.*sepmax/masperpx)
        for item in self._objs:
            if hasattr(item, "_masperpx"):
                if masperpxfixed is None:
                    masperpxfixed = item._masperpx
                elif np.abs(1-masperpxfixed/item._masperpx)*nbptscheck > 1:
                    if exc.raiseIt(exc.MasperpxMismatch, self.raiseError, mpp1=masperpxfixed, mpp2=item._masperpx):
                        return
        if masperpxfixed is not None:
            masperpx = masperpxfixed
        # check resolution parameters
        if sepmax is None and masperpx is None and nbpts is not None:
            sepmax = 1000*nbpts/(2*np.hypot(self.oidata.uvwl['v'], self.oidata.uvwl['u']).max())
        if masperpx is not None and sepmax is not None:
            nbpts = int(2.*sepmax/masperpx)
        elif masperpx is not None and nbpts is not None:
            pass
        else:
            masperpx = sepmax*2./nbpts
        #elif sepmax is not None and nbpts is not None:
        #    masperpx = sepmax*2./nbpts
        #else:
        #    raise Exception("can't determine the resolution, nbpts should not be None")
        sepmax = 0.5*nbpts*masperpx
        # check other parameters
        if wl is None: wl = self.oidata.uvwl['wl'].min()
        # initialize array
        img = np.zeros((nbpts, nbpts))
        if self._tweakparams is not None: self._tweakparams(self, params)
        if params is None:
            params = self.getP0() # initialize at p0 in case no params is given
        else:
            self.setParams(params)
        for item in self._objs:
            img += item.image(sepmax=sepmax, masperpx=masperpx, wl=wl, nbpts=nbpts)
            parampos += item._nparams
        if psfConvolve is not None:
            if len(core.aslist(psfConvolve)) == 1:
                psf = core.psf(float(psfConvolve)*core.MAS2RAD, masperpx)
            elif len(core.aslist(psfConvolve)) == 3:
                sx, sy, th = map(float, psfConvolve[:3])
                y, x = np.meshgrid(np.arange(nbpts)-nbpts//2, np.arange(nbpts)-nbpts//2)
                psf = core.gauss2D(x, y, 1, 0, 0, sx*0.5/masperpx, sy*0.5/masperpx, th)
            img = core.fftconvolve(img, psf, mode='same')
        if kwargs.pop('retresol', False):
            return img, (sepmax, masperpx, nbpts)
        return img


    def image(self, params=None, sepmax=None, wl=None, masperpx=None, nbpts=101, cmap='jet', cm_min=None,
              cm_max=None, ret=False, visu=None, psfConvolve=None, **visuargs):
        """
        psfConvolve in mas (lambda/D)
        """
        # check other params
        if visu is None:
            visu = core.ident
        if not callable(visu):
            if exc.raiseIt(exc.NotCallable, self.raiseError, fct='visu'):
                return
        toplot, (sepmax, masperpx, nbpts) = self.calcimage(sepmax=sepmax, wl=wl, params=params, nbpts=nbpts, psfConvolve=psfConvolve, retresol=True)
        toplot = visu(toplot, **visuargs)
        if cm_min is None:
            cm_min = toplot.min()
        if cm_max is None:
            cm_max = toplot.max()
        cmap, norm, mappable = core.colorbar(cmap=cmap, cm_min=cm_min, cm_max=cm_max)
        thefig, ax = core.astroskyplot(sepmax, polar=False, unit='mas')
        ax.matshow(toplot, origin='lower', extent=[-sepmax, sepmax, -sepmax, sepmax], cmap=cmap, norm=norm)
        plt.colorbar(mappable)
        if ret:
            return toplot


    def uvimage(self, params=None, blmax=None, wl=None, typ='vis2', nbpts=101, cmap='jet', cm_min=None,
                cm_max=None, ret=False, visu=None, **visuargs):
        """
        typ can be: 'VIS2', 'T3PHI', 'T3AMP', 'VISPHI', or 'VISAMP'
        """
        # check other params
        if visu is None: visu = core.ident
        if not callable(visu):
            if exc.raiseIt(exc.NotCallable, self.raiseError, fct='visu'):
                return
        toplot = self.calcuvimage(blmax=blmax, wl=wl, params=params, nbpts=nbpts)
        toplot = visu(core.FCTVISCOMP.get(typ.upper(), core.FCTVISCOMP['VIS2'])(toplot), **visuargs)
        if cm_min is None:
            cm_min = toplot.min()
        if cm_max is None:
            cm_max = toplot.max()
        cmap, norm, mappable = core.colorbar(cmap=cmap, cm_min=cm_min, cm_max=cm_max)
        thefig, ax = core.astroskyplot(blmax, polar=False, unit='m')
        ax.matshow(toplot, origin='lower', extent=[-blmax, blmax, -blmax, blmax], cmap=cmap, norm=norm)
        plt.colorbar(mappable)
        if ret:
            return toplot

    def residual(self, params, c=None, cmap='jet', cm_min=None, cm_max=None, datatype='All'):
        if not self._hasdata:
            if exc.raiseIt(exc.NoDataModel, self.raiseError): return
        calcindex = {'vis2':0, 't3phi':1, 't3amp':2, 'visphi':3, 'visamp':4}
        fullmodel = self.compVis(params=params)
        cm_min_orig = cm_min
        cm_max_orig = cm_max
        if datatype.lower() == 'all':
            datatype = core.DATAKEYSLOWER
        elif np.iterable(datatype)==1 and not isinstance(datatype, str):
            datatype = [str(dum).lower() for dum in datatype if dum.lower() in core.DATAKEYSLOWER]
        else:
            datatype = [str(dum).lower() for dum in [str(datatype)] if dum.lower() in core.DATAKEYSLOWER]
        if self.oidata.systematic_fit: self.oidata.systematic_prior = params[self._nparamsObj]
        for index, datatypeloop in enumerate(datatype):
            if not getattr(self.oidata, datatypeloop): continue
            data = getattr(self.oidata, datatypeloop).data
            model = fullmodel[calcindex[datatypeloop]]
            error = getattr(self.oidata, datatypeloop).error
            if self.oidata.systematic_fit: error = np.sqrt(error*error + self.oidata.systematic_prior*self.oidata.systematic_prior)
            thefig = plt.figure()
            if c is not None:
                grid = _matplotlibGridspecGridSpec(5, 21)
            else:
                grid = _matplotlibGridspecGridSpec(5, 20)
            theax = thefig.add_subplot(grid[0:4,0:20])
            theotherax = thefig.add_subplot(grid[4,0:20], sharex=theax)
            for k, v in np.ndenumerate(data):
                theax.add_patch(_matPatFancyArrowPatch((model[k], data[k]),(model[k], data[k]+error[k]), arrowstyle='-['))
                theax.add_patch(_matPatFancyArrowPatch((model[k], data[k]),(model[k], data[k]-error[k]), arrowstyle='-['))
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
                cmap, norm, mappable = core.colorbar(cmap=cmap, cm_min=cm_min, cm_max=cm_max)
                theax.scatter(x=model, y=data, s=30, c=colorattr, cmap=cmap, norm=norm, marker='o', edgecolors='none', alpha=1)
                div = np.true_divide(model-data, error)
                theotherax.scatter(x=model, y=self.oidata.erb_sigma(np.abs(div))*np.sign(div), s=30, c=colorattr, cmap=cmap, norm=norm, marker='o', edgecolors='none', alpha=1)
                thefig.colorbar(mappable=mappable, cax=thecb)
                colortitle = ', color= '+colorkeyattr
            else:
                theax.scatter(x=model, y=data, s=30, marker='o', edgecolors='none', alpha=1)
                div = np.true_divide(model-data, error)
                theotherax.scatter(x=model, y=self.oidata.erb_sigma(np.abs(div))*np.sign(div), s=30, marker='o', edgecolors='none', alpha=1)
                colortitle = ''
            theotherax.grid(True)
            theax.set_title(datatypeloop+colortitle)


    #def grid(self, param_x, param_y, bin_x=50, bin_y=50, mode=np.max, cmap="jet"):
    #    pass

    def likelihood(self, params, customlike=None, chi2=False, **kwargs):
        if not self._hasdata:
            if exc.raiseIt(exc.NoDataModel, self.raiseError): return
        kwargs['chi2'] = chi2
        return standardLikelihood(params=params, model=self, customlike=customlike, kwargs=kwargs)


    def save(self, filename, clobber=False):
        hdulist = pf.HDUList()
        ext = '.oif.fits'
        if filename.find(ext)==-1: filename += ext
        hdu = pf.PrimaryHDU()
        hdu.header.set('EXT', 'MODEL', comment='Type of information in the HDU')
        hdu.header.set('DATE', strftime('%Y%m%dT%H%M%S'), comment='Creation Date')
        hdu.header.set('NOBJ', self.nobj, comment='Number of objects in the model')
        hdu.header.set('NPARAMS', self.nparams, comment='Total number of free parameters')
        for i, item in enumerate(self._objs):
            hdu.header.set('OBJ'+str(i), item.name, comment='Name of object '+str(i))
        allmodes = ['vis2', 't3phi', 't3amp', 'visphi', 'visamp']
        for mode in allmodes:
            number = []
            if getattr(self.oidata, mode):
                number = [len(i) for i in getattr(self.oidata, "_input_"+mode) if i is not None]
            hdu.header.set('N'+mode, int(np.sum(number)), comment='Total number of '+mode+' measurements')
            hdu.header.set('F_'+mode, len(number), comment='Number of '+mode+' files')

        hdu.header.add_comment('Written by Guillaume SCHWORER')
        hdulist.append(hdu)

        hdulist.writeto(filename, clobber=clobber)

        for item in self._objs:
            item.save(filename, append=True, clobber=clobber)

        if self._hasdata: self.oidata.save(filename, append=True, clobber=clobber)

        return filename


def standardLikelihood(params, model, customlike=None, kwargs={}):
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
                    return -np.inf # exit with -inf if params outside bounds
                parampos += 1
        if model.oidata.systematic_fit:
            if not (model.oidata.systematic_bounds[0] <= model.oidata.systematic_prior <= model.oidata.systematic_bounds[1]):
                return -np.inf # exit with -inf if params outside bounds
            #parampos += 1
        if customlike is not None: return customlike(params=params, model=model, **kwargs)
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
            quality += ((model.oidata.vis2.data - vis2)**2*invvar - np.log(invvar)).sum()
    if model.oidata.visphi:
        if model.oidata.systematic_fit:
            invvar = 1./(model.oidata.visphi.error*model.oidata.visphi.error + model.oidata.systematic_prior*model.oidata.systematic_prior)
        else:
            invvar = model.oidata.visphi._invvar
        if chi2:
            quality += ((model.oidata.visphi.data - visphi)**2*invvar).sum()
        else:
            quality += ((model.oidata.visphi.data - visphi)**2*invvar - np.log(invvar)).sum()
    if model.oidata.visamp:
        if model.oidata.systematic_fit:
            invvar = 1./(model.oidata.visamp.error*model.oidata.visamp.error + model.oidata.systematic_prior*model.oidata.systematic_prior)
        else:
            invvar = model.oidata.visamp._invvar
        if chi2:
            quality += ((model.oidata.visamp.data - visamp)**2*invvar).sum()
        else:
            quality += ((model.oidata.visamp.data - visamp)**2*invvar - np.log(invvar)).sum()
    if model.oidata.t3phi:
        if model.oidata.systematic_fit:
            invvar = 1./(model.oidata.t3phi.error*model.oidata.t3phi.error + model.oidata.systematic_prior*model.oidata.systematic_prior)
        else:
            invvar = model.oidata.t3phi._invvar
        if chi2:
            quality += ((model.oidata.t3phi.data - t3phi)**2*invvar).sum()
        else:
            quality += ((model.oidata.t3phi.data - t3phi)**2*invvar - np.log(invvar)).sum()
    if model.oidata.t3amp:
        if model.oidata.systematic_fit:
            invvar = 1./(model.oidata.t3amp.error*model.oidata.t3amp.error + model.oidata.systematic_prior*model.oidata.systematic_prior)
        else:
            invvar = model.oidata.t3amp._invvar
        if chi2:
            quality += ((model.oidata.t3amp.data - t3amp)**2*invvar).sum()
        else:
            quality += ((model.oidata.t3amp.data - t3amp)**2*invvar - np.log(invvar)).sum()

    ### !!! hard-coded shit to fit black-body stuff
    #khi_temp = (core.ratio_bb_flux(1.65*1e-6, params[0], params[4], model._objs[0].diam, model._objs[1].diam) - 1.65)**2*11.111 - 2.40794560
    #khi_temp = (core.ratio_bb_flux(1.65*1e-6, np.random.normal(model._objs[0].temp, 2000), params[3], model._objs[0].diam, model._objs[1].diam) - 1.65)**2*11.111 - 2.40794560
    #quality += khi_temp*model.oidata.vis2.data.size

    if chi2:
        return quality
    else:
        return ln_prior - 0.5*quality # log like


def tweakparams(model, params):
    #Hmag = 5.44-0.62470279
    Rmag = 4.5274 # obtained from V and R mag estimates, extinction corrected
    model._objs[0].diam = soif.core.mag2diam(Rmag - 2.5*np.log10(1./(1. + 1./params[2] + 1./params[4])), 'R', model._objs[0].temp)
    model._objs[1].diam = soif.core.mag2diam(Rmag - 2.5*np.log10((1./params[2])/(1. + 1./params[2] + 1./params[4])), 'R', params[3])
    #model._objs[0].diam = soif.core.mag2diam(Rmag - 2.5*np.log10(1./(1. + 1./params[3] + 1./params[5])), 'R', params[0])
    #model._objs[1].diam = soif.core.mag2diam(Rmag - 2.5*np.log10((1./params[3])/(1. + 1./params[3] + 1./params[5])), 'R', params[4])

