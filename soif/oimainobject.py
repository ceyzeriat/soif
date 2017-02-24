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

from time import strftime
from time import time

try:
    import astropy.io.fits as pf
except ImportError:
    import pyfits as pf

from . import oipriors
from . import oiexception as exc
from . import core
np = core.np


__all__ = ['Oimainobject']


class Oimainobject(object):
    _keys = []

    def __init__(self, name, priors={}, bounds={}, verbose=False, *args, **kwargs):
        super(Oimainobject, self).__init__(*args, **kwargs)
        self.raiseError = bool(kwargs.pop('raiseError', True))
        self.name = str(name)
        self._nkeys = len(self._keys)
        self._pkeys = []
        self._nparams = 0
        self._vkeys = []
        self._pmask = np.zeros(self._nkeys, dtype=bool)
        self._vmask = np.zeros(self._nkeys, dtype=bool)
        for item in bounds.keys():
            if item not in self._keys:
                print("{}WARNING{}: Ignoring unknown bounds '{}' for object '{}'.".format(
                                core.font.orange,
                                core.font.normal,
                                item,
                                name))
        for item in priors.keys():
            if item not in self._keys:
                print("{}WARNING{}: Ignoring unknown prior '{}' for object '{}'.".format(
                                core.font.orange,
                                core.font.normal,
                                item,
                                name))
        for i, item in enumerate(self._keys):
            found = False
            if item in priors.keys():
                found = True
                setattr(self, item, float(priors[item]))
                setattr(self, "_"+item, float(priors[item]))
            if item in bounds.keys():
                vals = bounds[item]
                found = True
                if len(vals) < 2:  # error if less than 2 bounds
                    if exc.raiseIt(exc.InvalidBound, self.raiseError, name=name, param=item, vv=str(vals)):
                        return
                # set basic parameters
                setattr(self, item+"_bounds", list(map(float, vals[:2])))
                prior_range = np.abs(getattr(self, item+"_bounds")[1]-getattr(self, item+"_bounds")[0])*1.
                allkwargs = {'prior_bounds': getattr(self, item+"_bounds"), 'prior_range': prior_range, 'prior_invrange': 1./prior_range, 'prior_lninvrange': -np.log(prior_range)}
                if len(vals)>=4:
                    if not isinstance(vals[3], dict):
                        raise Exception("For object '{}', the fourth argument '{}' in prior definition must be a dictionary.".format(item, vals[3]))
                    allkwargs.update(vals[3])
                setattr(self, item+"_prior_kwargs", allkwargs)
                if len(vals)==2:
                    if verbose:
                        print("{}WARNING{}: No prior probability function found for parameter '{}', object '{}'. Assuming Uniform.".format(core.font.orange, core.font.normal, item, name))
                    setattr(self, item+"_prior_lnfunc", core.PriorWrapper(oipriors.lnuniform, makenorm=True, makeunlog=False, fct_log=True, **allkwargs)) # default is uniform on the range
                    setattr(self, item+"_prior_func", core.PriorWrapper(oipriors.uniform, makenorm=True, makeunlog=False, fct_log=False, **allkwargs)) # default is uniform on the range
                if len(vals)>=3:
                    if not callable(vals[2]):
                        raise Exception("For object '{}', the third argument '{}' in bounds definition must be callable.".format(item, vals[2]))
                    setattr(self, item+"_prior_lnfunc", core.PriorWrapper(vals[2], makenorm=True, makeunlog=False, fct_log=True, **allkwargs))
                if len(vals) in [3, 4]:
                    if verbose:
                        print("{}WARNING{}: No bounds probability function found for parameter '{}', object '{}'. Initial values for this parameter will be obtained from the exponentiation of the log-probability function.".format(core.font.orange, core.font.normal, item, name))
                    setattr(self, item+"_prior_func", core.PriorWrapper(vals[2], makenorm=True, makeunlog=True, fct_log=False, **allkwargs))
                if len(vals)==5:
                    if not callable(vals[4]):
                        raise Exception("For object '{}', the fifth argument '{}' in prior definition must be callable.".format(item, vals[4]))
                    setattr(self, item+"_prior_func", core.PriorWrapper(vals[4], makenorm=True, makeunlog=False, fct_log=False, **allkwargs))
                # pre-computing of P0 stuff
                x = getattr(getattr(self, item+"_prior_func"), 'x')
                setattr(self, "_"+item+"_P0", core.random_custom_pdf(x, getattr(self, item+"_prior_func")(x), size=False, renorm=True))
                self._pkeys.append(item)
                self._pmask[i] = True
                self._nparams += 1
            else:
                self._vkeys.append(item)
                self._vmask[i] = True
            if not found:
                exc.raiseIt(exc.InvalidBound, self.raiseError, name=name, param=item, vv="None")
                # force 0.0 to prior
                setattr(self, item, 0.0)
                setattr(self, "_"+item, 0.0)
                return

    @property
    def typ(self):
        return self.__class__.__name__
    @typ.setter
    def typ(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="typ")

    def _info(self):
        anyparam = len(self._pkeys)>0
        anyvalue = len(self._vkeys)>0
        ret = "{}<{}>{} '{}': {:d} params".format(core.font.yellow, self.typ, core.font.normal, self.name,
            self._nparams)
        if anyparam:
            ret += " (" + ", ".join([item for item in self._pkeys]) + ")"
        if anyvalue:
            ret += "; " + ", ".join([item+"="+str(getattr(self, item)) for item in self._vkeys])
        return ret
    def __repr__(self):
        return self._info()
    def __str__(self):
        return self._info()

    def show(self):
        ret = "{}<{}>{} '{}': {:d} params".format(core.font.yellow, self.typ, core.font.normal, self.name, self._nparams)
        if len(self._vkeys)>0:
            ret += "\nValues:"
            for item in self._vkeys:
                ret += "\n  {}{}{}={}".format(core.font.orange, item, core.font.normal, getattr(self, item))
        if len(self._pkeys)>0:
            ret += "\nParams:"
            for item in self._pkeys:
                ret += "\n  {}{}{}={} in [{}, {}]".format(core.font.orange, item, core.font.normal, getattr(self, item, "."), getattr(self, item+"_bounds")[0], getattr(self, item+"_bounds")[1])
        print(ret)

    @classmethod
    def keys(cls):
        return getattr(cls, '_keys', [])

    def to_radec(self):
        """
        Returns (ra, dec) in radian from sep and pa
        """
        seprad = self.sep*core.MAS2RAD
        return seprad*np.sin(self.pa), seprad*np.cos(self.pa)

    def to_pospx(self, sepmax, nbpts, integer=False):
        norm = nbpts/(2.*sepmax)
        ra, dec = self.sep*np.asarray([np.sin(self.pa), np.cos(self.pa)])*norm+0.5*(nbpts-1)
        if integer: ra, dec = np.round([ra, dec]).astype(int)
        return ra, dec

    def _shearCoord(self, x, y, fourier=False):
        if fourier:
            rat = 1./self.rat
        else:
            rat = self.rat
        cth = np.cos(self.th)
        sth = np.sin(self.th)
        return (x*cth-y*sth)*rat, x*sth+y*cth

    def oscil(self, u, v, wl):
        if self.sep!=0:
            ret = np.zeros(u.shape, dtype=complex)
            dra, ddec = self.to_radec()
            # analytical fourier transform for a point source
            oscil = -core.DEUXPI*(u*dra + v*ddec) / wl
            ret.real = np.cos(oscil)
            ret.imag = np.sin(oscil)
            return ret
        else:
            return np.ones(u.shape, dtype=complex)

    @property
    def params(self):
        """
        Returns a list of the parameters values, according to the parameter keys order
        """
        return [getattr(self, item) if hasattr(self, item) else np.mean(getattr(self, item+"_bounds")) for item in self._pkeys]
    @params.setter
    def params(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="params")

    def setParams(self, params, priors=False):
        """
        Affects to the object the registered values from params, according the to parameter keys order
        """
        for ind, arg in enumerate(self._pkeys):
            setattr(self, arg, params[ind])
        if priors:
            for ind, arg in enumerate(self._pkeys):
                setattr(self, "_"+arg, params[ind])

    def getP0(self):
        """
        Returns a list of initial values for each parameter in the object, according the to parameter keys order
        """
        ret = []
        randomizer = core.gen_generator()
        for item in self._pkeys:
            ret.append(np.clip(getattr(self, "_"+item+"_P0")(randomizer.uniform()), a_min=getattr(self, item+"_bounds")[0], a_max=getattr(self, item+"_bounds")[1]))
        return ret

    def compVis(self, oidata, params=None, flat=False):
        """
        Calculates the complex visibilities of the object
        """
        if params is not None: self.setParams(params)
        if flat:
            return self._calcCompVis(u=oidata.uvwl['u'], v=oidata.uvwl['v'], wl=oidata.uvwl['wl'], blwl=oidata.uvwl['blwl'], oidata=oidata)
        else:
            return oidata.remorph(self._calcCompVis(u=oidata.uvwl['u'], v=oidata.uvwl['v'], wl=oidata.uvwl['wl'], blwl=oidata.uvwl['blwl'], oidata=oidata))

    def save(self, filename, append=False, clobber=False):
        ext = '.oif.fits'
        if filename.find(ext)==-1: filename += ext
        if append:
            hdulist = pf.open(filename, mode='append')
        else:
            hdulist = pf.HDUList()
        hdu = pf.PrimaryHDU()
        hdu.header.set('DATE', strftime('%Y%m%dT%H%M%S'), comment='Creation Date')
        hdu.header.set('EXT', 'OBJ', comment='Type of information in the HDU')
        hdu.header.set('NAME', str(self.name), comment='Name of the object')
        hdu.header.set('TYP', str(self.typ), comment='Unit-model of the object')
        hdu.header.set('NVALUE', len(self._vkeys), comment='Number of set parameters in the model')
        hdu.header.set('NPARAM', len(self._pkeys), comment='Number of free parameters in the model')
        hdu.header.set('NSAVE', len(getattr(self, '_save', {})), comment='Number of additional parameters in the model')
        for i, k in enumerate(self._vkeys):
            hdu.header.set('VALUE'+str(i), k, comment='Name of the set parameter '+str(i))
            hdu.header.set(k,  getattr(self, "_"+k), comment='Value of the set parameter '+str(i))
        for i, k in enumerate(self._pkeys):
            hdu.header.set('PARAM'+str(i), k, comment='Name of the free parameter '+str(i))
            hdu.header.set(k,  getattr(self, "_"+k, 'NONE'), comment='Prior of the free parameter '+str(i))
            hdu.header.set(k+'L',  getattr(self, k+"_bounds")[0], comment='Value of the lower prior-bound ')
            hdu.header.set(k+'H',  getattr(self, k+"_bounds")[1], comment='Value of the higher prior-bound ')
        for i, k in enumerate(getattr(self, '_save', [])):
            hdu.header.set('SAVE'+str(i), k, comment='Name of the additional parameter '+str(i))
            if not isinstance(getattr(self, k), (np.ndarray, list, tuple)):
                hdu.header.set(k,  getattr(self, k), comment='Value of the additional parameter '+str(i))
            else:
                hdu.header.set(k, 'ARRAY', comment='Value of the additional parameter '+str(i))
                key = 'K'+str(int(time()/100%1*10**7))
                hdu.header.set('KEY'+str(i), key, comment='Unique key to find the parameter value '+str(i))
                paramhdu = pf.PrimaryHDU()
                paramhdu.header.set('DATE', strftime('%Y%m%dT%H%M%S'), comment='Creation Date')
                paramhdu.header.set('NAME', str(self.name), comment='Name of the object')
                paramhdu.header.set('TYP', str(self.typ), comment='Unit-model of the object')
                paramhdu.header.set('SAVE'+str(i), k, comment='Name of the additional parameter '+str(i))
                paramhdu.header.set('EXT', key, comment='Type of information in the HDU : unique key')
                paramhdu.data = np.asarray(getattr(self, k))
                hdulist.append(paramhdu)

        hdu.header.add_comment('Written by Guillaume SCHWORER')
        hdulist.append(hdu)
        
        if append:
            hdulist.flush()
            hdulist.close()
        else:
            hdulist.writeto(filename, clobber=clobber)
        return filename
