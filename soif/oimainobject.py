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


import oipriors as oipriors
import _core
_np = _core.np

from time import strftime as _strftime
from time import time as _time
try:
    import astropy.io.fits as _pf
except ImportError:
    import pyfits as _pf


class Oimainobject(object):
    def __init__(self, name, priors={}, bounds={}, verbose=False, **kwargs):
        self.name = str(name)
        self._nkeys = len(self._keys)
        self._pkeys = []
        self._nparams = 0
        self._vkeys = []
        self._pmask = _np.zeros(self._nkeys, dtype=bool)
        self._vmask = _np.zeros(self._nkeys, dtype=bool)
        for item in bounds.keys():
            if item not in self._keys: print(_core.font.orange+"WARNING"+_core.font.normal+": Ignoring unknown bounds '%s' for object '%s'." % (item, name))
        for item in priors.keys():
            if item not in self._keys: print(_core.font.orange+"WARNING"+_core.font.normal+": Ignoring unknown parameter '%s' for object '%s'." % (item, name))
        for i, item in enumerate(self._keys):
            found = False
            if item in priors.keys():
                found = True
                setattr(self, item, float(priors[item]))
                setattr(self, "_"+item, float(priors[item]))
            if item in bounds.keys():
                found = True
                if len(bounds[item])<2: # error if less than 2 bounds
                    raise Exception("Bounds for object '%s', parameter '%s' has only one bound: %s." % (name, item, str(bounds[item])))
                #set basic parameters
                setattr(self, item+"_bounds", bounds[item][:2])
                prior_range = _np.abs(getattr(self, item+"_bounds")[1]-getattr(self, item+"_bounds")[0])
                allkwargs = {'prior_bounds': getattr(self, item+"_bounds"), 'prior_range': prior_range, 'prior_invrange': 1./prior_range, 'prior_lninvrange':-_np.log(prior_range)}
                if len(bounds[item])>=4:
                    if not isinstance(bounds[item][3], dict):
                        raise Exception("For object '%s', the fourth argument '%s' in prior definition must be a dictionary." %  (item, bounds[item][3]))
                    allkwargs.update(bounds[item][3])
                setattr(self, item+"_prior_kwargs", allkwargs)
                if len(bounds[item])==2:
                    if verbose: print("%sWARNING%s: No prior probability function found for parameter '%s', object '%s'. Assuming Uniform." % (_core.font.orange, _core.font.normal, item, name))
                    setattr(self, item+"_prior_lnfunc", _core.PriorWrapper(oipriors.lnuniform, makenorm=True, makeunlog=False, fct_log=True, **allkwargs)) # default is uniform on the range
                    setattr(self, item+"_prior_func", _core.PriorWrapper(oipriors.uniform, makenorm=True, makeunlog=False, fct_log=False, **allkwargs)) # default is uniform on the range
                if len(bounds[item])>=3:
                    if not callable(bounds[item][2]):
                        raise Exception("For object '%s', the third argument '%s' in bounds definition must be callable." %  (item, bounds[item][2]))
                    setattr(self, item+"_prior_lnfunc", _core.PriorWrapper(bounds[item][2], makenorm=True, makeunlog=False, fct_log=True, **allkwargs))
                if len(bounds[item]) in [3, 4]:
                    if verbose: print(_core.font.orange+"WARNING"+_core.font.normal+": No bounds probability function found for parameter '%s', object '%s'. Initial values for this parameter will be obtained from the exponentiation of the log-probability function." % (item, name))
                    setattr(self, item+"_prior_func", _core.PriorWrapper(bounds[item][2], makenorm=True, makeunlog=True, fct_log=False, **allkwargs))
                if len(bounds[item])==5:
                    if not callable(bounds[item][4]):
                        raise Exception("For object '%s', the fifth argument '%s' in prior definition must be callable." %  (item, bounds[item][4]))
                    setattr(self, item+"_prior_func", _core.PriorWrapper(bounds[item][4], makenorm=True, makeunlog=False, fct_log=False, **allkwargs))
                # pre-computing of P0 stuff
                x = getattr(getattr(self, item+"_prior_func"), 'x')
                setattr(self, "_"+item+"_P0", _core.random_custom_pdf(x, getattr(self, item+"_prior_func")(x), size=False, renorm=True))
                self._pkeys.append(item)
                self._pmask[i] = True
                self._nparams += 1
            else:
                self._vkeys.append(item)
                self._vmask[i] = True
            if not found:
                raise Exception("No prior nor bounds found for parameter '%s' in object '%s'." % (item, name))
                setattr(self, item, 0)
                setattr(self, "_"+item, 0)

    @property
    def typ(self):
        return self.__class__.__name__
    @typ.setter
    def typ(self, value):
        raise AttributeError("Read-only")


    def _info(self):
        anyparam = len(self._pkeys)>0
        anyvalue = len(self._vkeys)>0
        ret = "%s<%s>%s '%s': %i params" % (_core.font.yellow, self.typ, _core.font.normal, self.name, self._nparams)
        if anyparam: ret += " (" + ", ".join([item for item in self._pkeys]) + ")"
        if anyvalue: ret += "; " + ", ".join([item+"="+str(getattr(self, item)) for item in self._vkeys])
        return ret
    def __repr__(self):
        return self._info()
    def __str__(self):
        return self._info()

    def show(self):
        ret = "%s<%s>%s '%s': %i params" % (_core.font.yellow, self.typ, _core.font.normal, self.name, self._nparams)
        if len(self._vkeys)>0:
            ret += "\nValues:"
            for item in self._vkeys:
                ret += "\n  %s%s%s=%s" % (_core.font.orange, item, _core.font.normal, getattr(self, item))
        if len(self._pkeys)>0:
            ret += "\nParams:"
            for item in self._pkeys:
                ret += "\n  %s%s%s=%s in [%s, %s]" % (_core.font.orange, item, _core.font.normal, getattr(self, item, "."), getattr(self, item+"_bounds")[0], getattr(self, item+"_bounds")[1])
        print(ret)

    @property
    def keys(self):
        return self._keys
    @keys.setter
    def keys(self, value):
        raise AttributeError("Read-only")


    def to_radec(self):
        """
        Returns (ra, dec) in radian from sep and pa
        """
        seprad = self.sep*_core.MAS2RAD
        return seprad*_np.sin(self.pa), seprad*_np.cos(self.pa)


    def to_pospx(self, sepmax, nbpts, integer=False):
        norm = nbpts/(2.*sepmax)
        ra, dec = self.sep*_np.asarray([_np.sin(self.pa), _np.cos(self.pa)])*norm+0.5*(nbpts-1)
        if integer: ra, dec = _np.round([ra, dec]).astype(int)
        return ra, dec

    def _shearCoord(self, x, y, fourier=False):
        if fourier:
            rat = 1./self.rat
        else:
            rat = self.rat
        cth = _np.cos(self.th)
        sth = _np.sin(self.th)
        return (x*cth-y*sth)*rat, x*sth+y*cth

    def oscil(self, u, v, wl):
        if self.sep!=0:
            ret = _np.zeros(u.shape, dtype=complex)
            dra, ddec = self.to_radec()
            # analytical fourier transform for a point source
            oscil = -_core.DEUXPI*(u*dra + v*ddec) / wl
            ret.real = _np.cos(oscil)
            ret.imag = _np.sin(oscil)
            return ret
        else:
            return _np.ones(u.shape, dtype=complex)


    @property
    def params(self):
        """
        Returns a list of the parameters values, according to the parameter keys order
        """
        return [getattr(self, item, _np.mean(getattr(self, item+"_bounds"))) for item in self._pkeys]
    @params.setter
    def params(self, value):
        raise AttributeError("Read-only")


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
        randomizer = _core.gen_generator()
        for item in self._pkeys:
            ret.append(_np.clip(getattr(self, "_"+item+"_P0")(randomizer.uniform()), a_min=getattr(self, item+"_bounds")[0], a_max=getattr(self, item+"_bounds")[1]))
        return ret


    def compVis(self, oidata, params=None, flat=False):
        """
        Does the paperwork before calculating the complex visibilities of the object
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
            hdulist = _pf.open(filename, mode='append')
        else:
            hdulist = _pf.HDUList()
        hdu = _pf.PrimaryHDU()
        hdu.header.set('DATE', _strftime('%Y%m%dT%H%M%S'), comment='Creation Date')
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
            if not isinstance(getattr(self, k), (_np.ndarray, list, tuple)):
                hdu.header.set(k,  getattr(self, k), comment='Value of the additional parameter '+str(i))
            else:
                hdu.header.set(k, 'ARRAY', comment='Value of the additional parameter '+str(i))
                key = 'K'+str(int(_time()/100%1*10**7))
                hdu.header.set('KEY'+str(i), key, comment='Unique key to find the parameter value '+str(i))
                paramhdu = _pf.PrimaryHDU()
                paramhdu.header.set('DATE', _strftime('%Y%m%dT%H%M%S'), comment='Creation Date')
                paramhdu.header.set('NAME', str(self.name), comment='Name of the object')
                paramhdu.header.set('TYP', str(self.typ), comment='Unit-model of the object')
                paramhdu.header.set('SAVE'+str(i), k, comment='Name of the additional parameter '+str(i))
                paramhdu.header.set('EXT', key, comment='Type of information in the HDU : unique key')
                paramhdu.data = _np.asarray(getattr(self, k))
                hdulist.append(paramhdu)

        hdu.header.add_comment('Written by Guillaume SCHWORER')
        hdulist.append(hdu)
        
        if append:
            hdulist.flush()
            hdulist.close()
        else:
            hdulist.writeto(filename, clobber=clobber)
        return filename
