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
try:
    from emcee import EnsembleSampler as _emceeEnsembleSampler
    _EMCEE = True
except ImportError:
    _EMCEE = False
from time import sleep as _sleep
from time import time as _time

from oimodel import _likelihood

import _core
_np = _core.np

from MCres import MCres as _MCres
from patiencebar import Patiencebar as _Patiencebar


class Oifiting(_MCres):
    def __init__(self, model, nwalkers=100, niters=500, burnInIts=100, threads=1, customlike=None, **kwargs):
        if 'sampler' in kwargs.keys():
            self._init(sampler=kwargs.pop('sampler'), paramstr=self.model.paramstr, nwalkers=self.nwalkers, niters=self.niters, burnInIts=self.burnInIts)
        elif _EMCEE:
            self.model = model
            self.nwalkers = int(nwalkers)
            self.niters = 0
            self.burnInIts = int(0 if burnInIts is None else burnInIts)
            self.sampler = _emceeEnsembleSampler(self.nwalkers, self.model.nparams, _likelihood, args=[self.model, customlike, kwargs], threads=max(1, int(threads)))
            self.run(niters=int(niters), burnInIts=self.burnInIts)

    def _init(self, sampler, paramstr=None, nwalkers=None, niters=None, burnInIts=None):
        super(Oifiting, self)._init(sampler=self.sampler, paramstr=self.model.paramstr, nwalkers=self.nwalkers, niters=self.niters, burnInIts=self.burnInIts)

    def _info(self):
        return _core.font.blue+"<Oifiting Fit>%s\n %i walkers, %s burn-in, %s iters\n%s"%(_core.font.normal, self.nwalkers, self.burnInIts, self.niters, str(self.model))


    def run(self, niters, burnInIts=0):
        '''
        Start a MCMC simulation on as many CPUs as threads parameter. Use customlike parameter to use a custom likelihood function. This function must accept modeledData (ndarray) and model (Oifiting) as input parameters. It will also be given the kwargs that you give to this runmc function.
            e.g.: mycustomlike(modeledData, model, **kwargs)
        '''
        if not _EMCEE:
            print(_core.font.red+"Can't run more iterations, you don't have emcee installed"+_core.font.normal)
            return
        if self.model.nparams==0:
            print(_core.font.red+"ERROR: All parameters are set, there is no reason for MCMC"+_core.font.normal)
            return

        self.niters = getattr(self, "niters", 0) + int(niters)

        pb = _Patiencebar()
        print(_core.font.green+'Running emcee'+_core.font.normal)
        t0 = _time()

        if burnInIts>0:
            pb.reset(valmax=int(burnInIts), title="Burn in")
            p0 = [self.model.getP0() for i in range(self.nwalkers)]
            crackit = self.sampler.sample(p0, iterations=burnInIts, storechain=False)
            for res in crackit: pb.update()
        else: # no burn in
            res = [getattr(self, "_p0", None), None, getattr(self, "_rstate0", None)] # try init where we where
            if res[0] is None: res[0] = [self.model.getP0() for i in range(self.nwalkers)] # if first simu, no burn, was requested at all

        pb.reset(valmax=int(niters), title="Sampling")
        self.sampler.reset()

        crackit = self.sampler.sample(res[0], iterations=int(niters), rstate0=res[2])
        for res in crackit: pb.update()

        _sleep(0.1)
        print('Time elapsed = %.2fs' %(_time()-t0))
        print("Mean acceptance rate: %.3f" % (self.sampler.acceptance_fraction.mean()))

        self._p0 = res[0]
        self._rstate0 = res[2]
        self._init(sampler=self.sampler, paramstr=self.model.paramstr, nwalkers=self.nwalkers, niters=self.niters, burnInIts=self.burnInIts)


    def MCmap(self, param_x, param_y, bin_x=50, bin_y=50, cmap="jet", cm_min=None, cm_max=None, axescolor='w', polar=False, showmax=True, radec=False, **kwargs):
        """
        Return a 2D histogram of the MC chain, showing the walker density per bin
        """
        if radec:
            fig, ax = _core.astroskyplot(bin_y[-1], polar=polar, unit='mas')
        else:
            if polar:
                fig, ax = _plt.subplots(subplot_kw={'projection':'polar'})
                ax.set_theta_zero_location('N')
            else:
                fig, ax = _plt.subplots()
        self._map(param_x=param_x, param_y=param_y, fig=fig, ax=ax, bin_x=bin_x, bin_y=bin_y, cmap=cmap, cm_min=cm_min, cm_max=cm_max, axescolor=axescolor, polar=polar, showmax=showmax, **kwargs)


    def Pbmap(self, param_x, param_y, bin_x=50, bin_y=50, cmap="jet", cm_min=None, cm_max=None, axescolor='w', polar=False, showmax=True, radec=False, **kwargs):
        """
        Return a 2D histogram of the MC chain, showing the best loglikelihood per bin
        """
        if radec:
            fig, ax = _core.astroskyplot(bin_y[-1], polar=polar, unit='mas')
        else:
            if polar:
                fig, ax = _plt.subplots(subplot_kw={'projection':'polar'})
                ax.set_theta_zero_location('N')
            else:
                fig, ax = _plt.subplots()
        self._map(param_x=param_x, param_y=param_y, fig=fig, ax=ax, data=self.lnprob.compressed(), method=_np.max, bin_x=bin_x, bin_y=bin_y, cmap=cmap, cm_min=cm_min, cm_max=cm_max, axescolor=axescolor, polar=polar, showmax=showmax, **kwargs)


    def save(self, name, clobber=False):
        name = self.model.save(name, clobber=clobber)
        super(Oifiting, self).save(name=name, clobber=clobber, append=True)
        return name


    def statut(self, params=None, customlike=None, **kwargs):
        if params is None: params = self.best
        self.model.statut(params=params, customlike=customlike)


    def uvimage(self, blmax=None, wl=None, params=None, datatype='Data', nbpts=101, withdata=True, cmap='jet', cm_min=None, cm_max=None, ret=False):
        """
        Datatype in ['vis2', 'phase', 'vis']
        """
        allmodes = ['vis2', 'phase', 'vis']
        calctrick = {'vis2':_core.abs2, 'phase':_np.angle, 'vis':_np.abs}
        cm_min_orig = cm_min
        cm_max_orig = cm_max
        if datatype.lower() == 'all':
            datatype = allmodes
        if datatype.lower() == 'data':
            datatype = []
            if self.model.oidata.vis2: datatype.append('vis2')
            if self.model.oidata.visamp or self.model.oidata.t3amp: datatype.append('vis')
            if self.model.oidata.t3phi or self.model.oidata.visphi: datatype.append('phase')
        elif _np.iterable(datatype)==1 and not isinstance(datatype, str):
            datatype = [str(dum).lower() for dum in datatype if dum.lower() in allmodes]
        else:
            datatype = [str(dum).lower() for dum in [str(datatype)] if dum.lower() in allmodes]
        if blmax is None: blmax = _np.hypot(self.model.oidata.uvwl['v'], self.model.oidata.uvwl['u']).max()
        if wl is None: wl = self.model.oidata.uvwl['wl'].min()
        if params is None: params = self.best
        datashown = self.model.compuvimage(blmax=blmax, wl=wl, params=params, nbpts=nbpts)
        if ret: retval = []
        for index, datatypeloop in enumerate(datatype):
            data = calctrick[datatypeloop](datashown)
            if ret: retval.append(data)
            if cm_min_orig is None: cm_min = data.min()
            if cm_max_orig is None: cm_max = data.max()
            cmap, norm, mappable = _core.funcs.colorbar(cmap=cmap, cm_min=cm_min, cm_max=cm_max)
            thefig, ax = _core.astroskyplot(blmax, polar=False, unit='m')
            ax.matshow(data, origin='lower', extent=[-blmax,blmax,-blmax,blmax], cmap=cmap, norm=norm)
            if withdata:
                thefig.axes.scatter(getattr(self.model.oidata, datatypeloop).u, getattr(self.model.oidata, datatypeloop).v, c=getattr(self.model.oidata, datatypeloop).data, cmap=cmap, norm=norm)
            _plt.colorbar(mappable)
            thefig.axes.set_title(datatypeloop)
        if ret: return retval


    def image(self, params=None, sepmax=None, wl=None, masperpx=None, nbpts=101, cmap='jet', cm_min=None, cm_max=None, ret=False, visu=None, **visuargs):
        if params is None: params = self.best
        return self.model.image(params=params, sepmax=sepmax, wl=wl, masperpx=masperpx, nbpts=nbpts, cmap=cmap, cm_min=cm_min, cm_max=cm_max, ret=ret, visu=visu, **visuargs)


    def imagefft(self, sepmax=None, wl=None, params=None, nbpts=101, cmap='jet', cm_min=None, cm_max=None, ret=False, visu=None, **visuargs):
        if sepmax is None: sepmax = 1000*nbpts/(2*_np.hypot(self.model.oidata.uvwl['v'], self.model.oidata.uvwl['u']).max())
        if wl is None: wl = self.model.oidata.uvwl['wl'].min()
        if params is None: params = self.best
        if visu is None: visu = _np.abs
        #toplot = visu(self.model.compimage(sepmax=sepmax, wl=wl, params=params, nbpts=nbpts), **visuargs)
        toplot = _np.abs(_np.fft.fftshift(_np.fft.ifft2(self.model.compuvimage(blmax=nbpts*2*sepmax*_core.MAS2RAD/(_np.pi*wl), wl=wl, params=params, nbpts=nbpts))))
        if cm_min is None: cm_min = toplot.min()
        if cm_max is None: cm_max = toplot.max()
        cmap, norm, mappable = _core.funcs.colorbar(cmap=cmap, cm_min=cm_min, cm_max=cm_max)
        thefig, ax = _core.astroskyplot(sepmax, polar=False, unit='mas')
        ax.matshow(toplot, origin='lower', extent=[-sepmax,sepmax,-sepmax,sepmax], cmap=cmap, norm=norm)
        _plt.colorbar(mappable)
        if ret: return toplot


    def residual(self, params=None, c=None, cmap='jet', cm_min=None, cm_max=None, datatype='All'):
        if params is None: params = self.best
        self.model.residual(params=params, c=c, cmap=cmap, cm_min=cm_min, cm_max=cm_max, datatype=datatype)
