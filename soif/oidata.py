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


#import matplotlib.pyplot as _plt
from time import strftime as _strftime
try:
    import astropy.io.fits as _pf
except ImportError:
    import pyfits as _pf

import oiexception as _exc
import _core
_np = _core.np


class Oigrab(object):
    """
    Opens, reads and filters data contained in the OIFITS file ``src``.

    Args:
      * src (str): the path (relative or absolute) to the OIFITS file
    
    Kwargs:
      * raiseError (bool): if ``True``, raises errors, otherwise prints them

    Raises:
      * NoTargetTable: if the OIFITS file has no OI_TARGET table

    >>> import soif.oidata as oidata
    >>> data = oidata.Oigrab('./data/datafile.oifits')
    """
    def __init__(self, src, **kwargs):
        self._init(src=src, **kwargs)

    def _init(self, src, **kwargs):
        self.src = str(src)
        self.raiseError = bool(kwargs.pop('raiseError', True))
        hdus = _pf.open(self.src)
        for item in hdus:
            if item.header.get('EXTNAME')=='OI_TARGET':
                hdutgt = item
                break
        else:
            hdus.close()
            if _exc.raiseIt(_exc.NoTargetTable, self.raiseError, src=self.src): return False
        self._targets = {}
        for ind, tgt in zip(hdutgt.data["TARGET_ID"], hdutgt.data["TARGET"]):
            self._targets[ind] = tgt
        hdus.close()

    def _info(self):
        return "%s<SOIF File>%s\n File: '%s'" % (_core.font.blue, _core.font.normal, self.src)
    def __repr__(self):
        return self._info()
    def __str__(self):
        return self._info()

    @property
    def targets(self):
        return self._targets
    @targets.setter
    def targets(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="targets")

    def show_specs(self, ret=False, **kwargs):
        """
        Gets the target list and the data details from the OIFITS file.

        Args:
          * ret (bool): if ``True``, returns the information, otherwise prints it

        Returns:
          * a dictionary {'hdu index:info'} where info corresponds to a list of (Acquisition index, Target ID, MJD, N(UV), N(wl)) tuples

        >>> import soif.oidata as oidata
        >>> data = oidata.Oigrab('./data/datafile.oifits')
        >>> data.showspecs()
        TARGETS:
        1: IRS_48
        2: Elia_2-15

        VIS2 [hdu=3]:
        Acq. Index | Target ID |      MJD       |  UVs | N wl
        -----------------------------------------------------
                 0 |         1 | 55636.3827746  |   21 |    1
                 1 |         2 | 55636.3827989  |   21 |    1
                 2 |         1 | 55636.3828232  |   21 |    1
        """
        hdus = _pf.open(self.src)
        if not ret:
            print("TARGETS:")
            for ind, tgt in self.targets.items():
                print('%d: %s' % (ind, tgt))
        tgtlist = {}
        for idx, item in enumerate(hdus):
            if _core.hduToDataType(item) is not None:
                targetindexnumber, MJD, (ndata, nset, nunique, nholes, nwl) = _core.gethduMJD(item, withdet=True)
                tgtlist[idx] = []
                if not ret: print("\n%s [hdu=%d]:\nAcq. Index | Target ID |      MJD      |  UVs | N wl\n%s" % (_core.hduToDataType(item), idx, "-"*52))
                for tgtidx, sMJD in zip(targetindexnumber.reshape((-1, nunique))[:,0], MJD.reshape((-1, nunique))[:,0]):
                    tgtfilter = slice(tgtidx*nunique, (tgtidx+1)*nunique)
                    tgtid = item.data['TARGET_ID'][tgtfilter]
                    if not ret: print("%10s | %9s | %13s | %4s | %4s"%(tgtidx, tgtid[0], sMJD, tgtid.size, nwl))
                    tgtlist[idx].append((tgtidx, tgtid[0], sMJD, tgtid.size, nwl))
        hdus.close()
        if ret: return tgtlist

    def show_filtered(self, tgt=None, mjd=[None, None], hduNums=[], vis2=True, t3phi=True, t3amp=True, visphi=True, visamp=True, verbose=False, **kwargs): # inst=None, array=None, wl=None, 
        """
        Given an oifits file 'src' and filtering parameters on the target name (OI_TARGET table), the instrument name (OI_WAVELENGTH table), the array name (OI_ARRAY table), the observation wavelength (OI_WAVELENGTH table) and the acquisition time [t_min, t_max] (OI_VIS2, OI_VIS, OI_T3 tabkes), this function returns the data indices of the data matching all of these different filters.
        These lists are used to load the data within an Oidata object.

        Leave input parameter to 'None' to discard filtering on that particular parameter.

        Returns: VIS2, T3, VIS indeces as a tuple of 3 lists
        """
        hdus = _pf.open(self.src)
        mjd = [float(mjd[0] if mjd[0] is not None else -_np.inf), float(mjd[1] if mjd[1] is not None else _np.inf)]
        datayouwant = {'data':{'VIS2':bool(vis2), 'T3PHI':bool(t3phi), 'T3AMP':bool(t3amp), 'VISPHI':bool(visphi), 'VISAMP':bool(visamp)}}
        for idx, item in enumerate(hdus):
            if _core.aslist(hduNums) != [] and idx not in _core.aslist(hduNums): continue
            if _core.hduToDataType(item) is not None:
                MJD = _core.gethduMJD(item)[1]
                filt = ((MJD>=mjd[0]) & (MJD<mjd[1]))                
                if tgt is not None:
                    filt = (filt & (item.data.field("TARGET_ID") == int(tgt)))
                if verbose: print("%s:\n  %d/%s\n"%(_core.hduToDataType(item), filt.sum(), item.data["TARGET_ID"].size))
                datayouwant[idx] = _np.arange(item.data["TARGET_ID"].size)[filt]
        hdus.close()
        return datayouwant

    def extract(self, tgt=None, mjd=[None, None], wl=[None, None], hduNums=[], vis2=True, t3phi=True, t3amp=True, visphi=True, visamp=True, flatten=False, degree=True, significant_figures=5, erb_sigma=None, sigma_erb=None, systematic_prior=None, systematic_bounds=None, verbose=False, **kwargs):
        datayouwant = self.show_filtered(tgt=tgt, mjd=mjd, vis2=vis2, hduNums=hduNums, t3phi=t3phi, t3amp=t3amp, visphi=visphi, visamp=visamp, verbose=verbose, **kwargs)
        return Oifits(src=self.src, datafilter=datayouwant, flatten=flatten, degree=degree, significant_figures=significant_figures, wl=wl, erb_sigma=erb_sigma, sigma_erb=sigma_erb, systematic_prior=systematic_prior, systematic_bounds=systematic_bounds, **kwargs)


class OidataEmpty(object):
    def __init__(self, datatype, **kwargs):
        self.raiseError = bool(kwargs.pop('raiseError', True))
        self.datatype = str(datatype).upper()
        if self.datatype not in _core.ATTRDATATYPE.keys():
            if _exc.raiseIt(_exc.InvalidDataType, self.raiseError, datatype=self.datatype): return False
        self._has = False
        self._useit = False

    def _info(self):
        return "%s data, None" % (self.datatype)
    def __repr__(self):
        return self._info()
    def __str__(self):
        return self._info()

    @property
    def useit(self):
        return (self._useit and self._has)
    @useit.setter
    def useit(self, value):
        self._useit = bool(value)
    
    def __bool__(self):
        return self._has
    def __nonzero__(self):
        return self.__bool__()

    def __getitem__(self, key):
        return getattr(self, "_"+key)


class Oidata(OidataEmpty):
    def __init__(self, src, hduidx, datatype, hduwlidx, indices=[], wlindices=[], degree=True, flatten=False, significant_figures=5, **kwargs):
        super(Oidata, self).__init__(datatype=datatype, **kwargs)
        self._input_src = [str(src)]
        self._input_hduidx = [int(hduidx)]
        self._input_hduwlidx = [int(hduwlidx)]
        hdus = _pf.open(self._input_src[-1])
        hdu = hdus[self._input_hduidx[-1]]
        hduwl = hdus[self._input_hduwlidx[-1]]

        if self.datatype not in _core.ATTRDATATYPE.keys():
            if _exc.raiseIt(_exc.InvalidDataType, self.raiseError, datatype=self.datatype): return False
        if _core.DATAKEYSDATATYPE[self.datatype]['data'] not in _core.hduToColNames(hdu):
            if _exc.raiseIt(_exc.HduDatatypeMismatch, self.raiseError, hduhead=_core.hduToDataType(hdu), datatype=self.datatype): return False

        self._input_degree = [bool(degree)]
        self._input_flatten = [bool(flatten)]
        self._input_significant_figures = [int(significant_figures)]
        self._input_indices = [list(indices)]
        self._input_wlindices = [list(wlindices)]
        
        self._has = True
        self._useit = True
        wlindices = slice(0, 10000000) if wlindices == [] else _np.asarray(wlindices).ravel()
        indices = slice(0, 10000000) if indices == [] else _np.asarray(indices).ravel()

        # attributes to be copy-pasted
        for key, vl in _core.ATTRDATATYPE[self.datatype].items():
            setattr(self, '_'+key, vl)

        # wavelength attributes to be extracted out of OIFITS
        for key, vl in _core.KEYSWL.items():
            setattr(self, '_'+key, hduwl.data[vl].ravel()[wlindices])
        self._wlsize = self._wl.size
        self._datasize = hdu.data[_core.DATAKEYSDATATYPE[self.datatype]['data']][indices].shape[0]

        # data attributes to be extracted out of OIFITS
        for key, vl in _core.DATAKEYSDATATYPE[self.datatype].items():
            setattr(self, '_'+key, hdu.data[vl][indices].reshape((self._datasize, -1))[:,wlindices])
        self.mask = _np.logical_not(self._mask) # swaps Trues to Falses to have a good mask

        # data attributes to be extracted out of OIFITS and replicated with wl
        for key, vl in _core.UVKEYSDATATYPE[self.datatype].items():
            setattr(self, '_'+key, _core.replicate(hdu.data[vl][indices].ravel(), (None, self._wlsize)))

        # done with the file
        hdus.close()

        # add the data dimension on wl-like attributes
        for key in _core.KEYSWL.keys():
            setattr(self, "_"+key, _core.replicate(getattr(self, "_"+key), (self._datasize, None)))

        # combine the UV coordinates of T3
        if self.is_t3:
            self._u = _np.concatenate((_np.expand_dims(self._u1, -1), _np.expand_dims(self._u2, -1), _np.expand_dims(-self._u1-self._u2, -1)), axis=-1)
            self._v = _np.concatenate((_np.expand_dims(self._v1, -1), _np.expand_dims(self._v2, -1), _np.expand_dims(-self._v1-self._v2, -1)), axis=-1)
            # delete temporary attributes
            for key in _core.UVKEYSDATATYPE[self.datatype].keys():
                delattr(self, "_"+key)
            # add the data dimension on wl-like attributes
            for key in _core.KEYSWL.keys():
                setattr(self, "_"+key, _core.replicate(getattr(self, "_"+key), (None, 3)))

        # convert to radian if needed
        if self.is_angle and degree:
            self._data = self._data*_core.DEG2RAD
            self._error = self._error*_core.DEG2RAD

        self.significant_figures = min(8, max(1, int(significant_figures)))
        self._flat = bool(flatten)
        if self._flat:
            self.flatten()
        else:
            self.update()

    def _info(self):
        return "%s data, shape: %s, wl: %s%s" % (self.datatype, _core.maskedshape(self.shapedata, _np.logical_not(self.mask).sum()), self._wlmin, (" to "+str(self._wlmax))*int(self._wlspan!=0))
    
    @property
    def data(self):
        if self._use_mask:
            return self._data[self.mask]
        else:
            return self._data
    @data.setter
    def data(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="data")

    @property
    def error(self):
        if self._use_mask:
            return self._error[self.mask]
        else:
            return self._error
    @error.setter
    def error(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="error")

    @property
    def mask(self):
        return self._mask
    @mask.setter
    def mask(self, value):
        value = _np.asarray(value, dtype=bool)
        if value.shape != self._data.shape:
            if _exc.raiseIt(_exc.BadMaskShape, self.raiseError, shape=str(self.data.shape)): return False
        self._use_mask = not value.all()
        self._mask = value

    @property
    def u(self):
        if self._use_mask:
            return self._u[self.mask]
        else:
            return self._u
    @u.setter
    def u(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="u")

    @property
    def v(self):
        if self._use_mask:
            return self._v[self.mask]
        else:
            return self._v
    @v.setter
    def v(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="v")

    @property
    def wl(self):
        if self._use_mask:
            return self._wl[self.mask]
        else:
            return self._wl
    @wl.setter
    def wl(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="wl")

    @property
    def wl_d(self):
        if self._use_mask:
            return self._wl_d[self.mask]
        else:
            return self._wl_d
    @wl_d.setter
    def wl_d(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="wl_d")

    @property
    def shapedata(self):
        return self.data.shape
    @shapedata.setter
    def shapedata(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="shapedata")

    @property
    def shapeuv(self):
        return self.u.shape
    @shapeuv.setter
    def shapeuv(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="shapeuv")

    @property
    def is_angle(self):
        return self._is_angle
    @is_angle.setter
    def is_angle(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="is_angle")

    @property
    def is_t3(self):
        return self._is_t3
    @is_t3.setter
    def is_t3(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="is_t3")

    @property
    def flat(self):
        return self._flat
    @flat.setter
    def flat(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="flat")

    def flatten(self, **kwargs):
        if self.is_t3:
            for key in _core.KEYSUV:
                setattr(self, "_"+key, getattr(self, "_"+key).reshape((-1, 3)))
            for key in _core.KEYSDATA:
                setattr(self, "_"+key, getattr(self, "_"+key).ravel())
        else:
            for key in _core.KEYSDATA + _core.KEYSUV:
                setattr(self, "_"+key, getattr(self, "_"+key).ravel())
        self.update()
        self._flat = True

    def _addData(self, data, flatten=True, **kwargs):
        if not isinstance(data, OidataEmpty): return # trivial, nothing to add
        if not isinstance(data, Oidata):
            if _exc.raiseIt(_exc.WrongData, _exc.doraise(self, **kwargs), typ='Oidata'): return False
        if self.datatype != data.datatype:
            if _exc.raiseIt(_exc.IncompatibleData, _exc.doraise(self, **kwargs), typ1=self.datatype, typ2=data.datatype): return False
        # do we flatten it?
        if flatten or self.shapedata[-1] != data.shapedata[-1]:
            self.flatten()
            data.flatten()
        # concatenate data
        for key in _core.KEYSDATA + _core.KEYSUV:
            setattr(self, "_"+key, _np.concatenate((getattr(self, "_"+key), getattr(data, "_"+key)), axis=0))
        # update input keys
        for key in _core.INPUTSAVEKEY:
            setattr(self, "_input_"+key, getattr(self, "_input_"+key) + getattr(data, "_input_"+key))
        # update the data
        self.update()

    def update(self):
        """
        Given u, v, wl and flag information as object properties, this function updates the Oidata object: the data masking (from the new mask property) and the bl, pa, blwl properties (from u, v and wl properties)
        """
        self._u = _core.round_fig(self._u, self.significant_figures)
        self._v = _core.round_fig(self._v, self.significant_figures)
        self._wl = _core.round_fig(self._wl, self.significant_figures)
        self._wl_d = _core.round_fig(self._wl_d, self.significant_figures)
        self._invvar = 1./self.error**2
        self.bl = _core.round_fig(_np.hypot(self.v, self.u), self.significant_figures)
        self.pa = _core.round_fig(_np.arctan2(self.v, self.u), self.significant_figures)
        self.blwl = _core.round_fig(self.bl/self.wl, self.significant_figures)
        self._wlmin = self.wl.min()
        self._wlmax = self.wl.max()
        self._wlspan = self._wlmax - self._wlmin



class Oifits(object):
    """
    This class opens, reads and sorts data contained in the file 'src' (oifits format).

    vis2, vis and t3 are lists of which corresponding data indeces to extract from the file. 
    """
    def __init__(self, src, datafilter, wl=[None, None], erb_sigma=None, sigma_erb=None, systematic_prior=None, systematic_bounds=None, flatten=False, degree=True, significant_figures=5, **kwargs):
        self.raiseError = bool(kwargs.pop('raiseError', True))
        # initialize empty data
        for key in _core.DATAKEYSLOWER:
            setattr(self, key, OidataEmpty(key))

        self.erb_sigma = _core.ident if erb_sigma is None else erb_sigma
        if not callable(self.erb_sigma):
            if _exc.raiseIt(_exc.NotCallable, self.raiseError, fct="erb_sigma"): return False
        self.sigma_erb = _core.ident if sigma_erb is None else sigma_erb
        if not callable(self.sigma_erb):
            if _exc.raiseIt(_exc.NotCallable, self.raiseError, fct="sigma_erb"): return False
        self.systematic_bounds = None if systematic_bounds is None else list(map(float, list(systematic_bounds)[:2]))
        self.systematic_prior = None if systematic_prior is None else float(systematic_prior)
        self._systematic_prior = self.systematic_prior
        # init parameters
        self.addData(src=str(src), datafilter=dict(datafilter), flatten=bool(flatten), degree=bool(degree), significant_figures=int(significant_figures), wl=list(wl)[:2], noupdate=True, **kwargs)
        self.update()

    def _info(self):
        txt = "\n".join([" "+str(getattr(self, key)) for key in _core.DATAKEYSLOWER if getattr(self, key).useit])
        if txt == "": txt = " No data"
        txt = "%s<SOIF Data>%s\n%s" % (_core.font.blue, _core.font.normal, txt)
        return txt
    def __repr__(self):
        return self._info()
    def __str__(self):
        return self._info()

    def addData(self, src, datafilter, flatten=False, degree=True, significant_figures=5, wl=[None, None], **kwargs):
        hdus = _pf.open(src) # open
        hduwlidx = _core.hduWlindex(hdus)
        if not hduwlidx:
            if _exc.raiseIt(_exc.NoWavelengthTable, self.raiseError, src=src): return
        # get wl sorted
        wl = [float(wl[0] if wl[0] is not None else -_np.inf), float(wl[1] if wl[1] is not None else _np.inf)]
        allwl = hdus[hduwlidx].data[_core.KEYSWL['wl']]
        wlindices = _np.arange(allwl.size)[((allwl>=wl[0]) & (allwl<wl[1]))]
        # for each datafilter
        whichdata = datafilter.get('data', {'data':{'VIS2':True, 'T3PHI':True, 'T3AMP':True, 'VISPHI':True, 'VISAMP':True}})
        for idx, indices in datafilter.items():
            if not isinstance(idx, int): continue
            # if real data
            if _core.hduToDataType(hdus[idx]) is not None:
                for datatype in _core.ALLDATAEXTNAMES[hdus[idx].header['EXTNAME']]:
                    if not whichdata.get(datatype.upper(), True): continue
                    thedata = Oidata(src=src, hduidx=idx, datatype=datatype, hduwlidx=hduwlidx, indices=indices, wlindices=wlindices, degree=degree, flatten=flatten, significant_figures=significant_figures, **kwargs)
                    if getattr(self, datatype.lower()):
                        getattr(self, datatype.lower())._addData(thedata, flatten=flatten, raiseError=self.raiseError)
                    else:
                        setattr(self, datatype.lower(), thedata)
            else:
                hdus.close()
                if _exc.raiseIt(_exc.NotADataHdu, self.raiseError, idx=idx, src=str(src)): return
        hdus.close()
        if not kwargs.pop('noupdate', False): self.update()

    def flatten(self):
        """
        Flattens all data contained in the Oidata object. This can be useful in order to add several bits of data that do not have the same shapes
        """
        for key in _core.DATAKEYSLOWER:
            if getattr(self, key): getattr(self, key).flatten()

    def update(self):
        """
        Updates all data contained in the Oidata object
        """
        for key in _core.DATAKEYSLOWER:
            if getattr(self, key): getattr(self, key).update()

        funkydtypeint = [('u', int), ('v', int), ('wl', int)]
        funkydtype = [('u', _np.float32), ('v', _np.float32), ('wl', _np.float32)]

        # get uvwl sets as integer to extract uniques
        unique_uvwl = _np.zeros(0, dtype=funkydtypeint)
        for key in _core.DATAKEYSLOWER:
            thedata = getattr(self, key)
            if thedata:
                dum = _np.zeros(thedata.shapeuv, dtype=funkydtypeint)
                dum['u'] = _core.round_fig(x=thedata.u, n=thedata.significant_figures, retint=True)
                dum['v'] = _core.round_fig(x=thedata.v, n=thedata.significant_figures, retint=True)
                dum['wl'] = _core.round_fig(x=thedata.wl, n=thedata.significant_figures, retint=True)
                # deal with symmetry
                inv = (dum['u']<0)
                dum['u'][inv] *= -1
                dum['v'][inv] *= -1
                # stack
                unique_uvwl = _np.hstack((unique_uvwl, dum.flatten()))
        if _core.OLDNUMPY:
            uvwlind = _np.unique((unique_uvwl['u']+1j*unique_uvwl['v'])/unique_uvwl['wl'], return_index=True)[1]
        else:
            uvwlind = _np.unique(unique_uvwl, return_index=True)[1]
        # get uvwl sets as floats for calculations
        unique_uvwl = _np.zeros(0, dtype=funkydtype)
        for key in _core.DATAKEYSLOWER:
            thedata = getattr(self, key)
            if thedata:
                dum = _np.zeros(thedata.shapeuv, dtype=funkydtype)
                dum['u'] = _core.round_fig(x=thedata.u, n=thedata.significant_figures)
                dum['v'] = _core.round_fig(x=thedata.v, n=thedata.significant_figures)
                dum['wl'] = _core.round_fig(x=thedata.wl, n=thedata.significant_figures)
                # simplify symmetry
                inv = (dum['u']<0)
                dum['u'][inv] *= -1
                dum['v'][inv] *= -1
                # save the phase symmetry
                if thedata.is_angle:
                    thedata._phasesign = 1-inv*2
                # stack
                unique_uvwl = _np.hstack((unique_uvwl, dum.flatten()))
                thedata._ind = dum.copy()
        # extract uniques
        dum = _np.zeros(uvwlind.shape, dtype=funkydtype)
        dum['u'], dum['v'], dum['wl'] = unique_uvwl['u'][uvwlind], unique_uvwl['v'][uvwlind], unique_uvwl['wl'][uvwlind]
        unique_uvwl = dum
        # save indices
        for key in _core.DATAKEYSLOWER:
            thedata = getattr(self, key)
            if thedata:
                dum = _np.zeros(thedata._ind.shape, dtype='int')
                for i, v in enumerate(unique_uvwl):
                    dum[thedata._ind == v] = i
                thedata._ind = dum.copy()
        # prepare pre-processed uniques
        self.uvwl = {'u':unique_uvwl['u'], 'v':unique_uvwl['v'], 'wl':unique_uvwl['wl'], 'blwl':_np.hypot(unique_uvwl['u'], unique_uvwl['v'])/unique_uvwl['wl']}

    @property
    def systematic_fit(self):
        return isinstance(self.systematic_bounds, list)
    @systematic_fit.setter
    def systematic_fit(self, value):
        _exc.raiseIt(_exc.ReadOnly, self.raiseError, attr="systematic_fit")

    def systematic_p0(self):
        if self.systematic_fit:
            randomizer = _core.gen_generator()
            return randomizer.uniform(low=self.systematic_bounds[0], high=self.systematic_bounds[1])
        else:
            _exc.raiseIt(NoSystematicsFit, self.raiseError)

    def remorph(self, viscomp):
        ret = []
        for key in _core.DATAKEYSLOWER:
            thedata = getattr(self, key)
            if thedata:
                retdbl = isinstance(viscomp, (tuple, list))
                if retdbl:
                    viscomp, flx = viscomp
                if thedata.is_t3:
                    if thedata.is_angle: # t3phi
                        dum = (_core.FCTVISCOMP[key.upper()](viscomp)[thedata._ind]*thedata._phasesign).sum(-1)
                    else: # t3amp
                        dum = (_core.FCTVISCOMP[key.upper()](viscomp[thedata._ind])).prod(-1)
                else:
                    dum = _core.FCTVISCOMP[key.upper()](viscomp)[thedata._ind]
                    if thedata.is_angle: # visphi
                        dum *= thedata._phasesign
                if retdbl:
                    ret.append((dum, flx[thedata._ind]))
                else:
                    ret.append(dum)
            else:
                ret.append(None)
        return ret


    def save(self, filename, append=False, clobber=False):
        ext = '.oif.fits'
        if filename.find(ext)==-1: filename += ext
        if append:
            hdulist = _pf.open(filename, mode='append')
        else:
            hdulist = _pf.HDUList()
        allmodes = ['vis2', 't3phi', 't3amp', 'visphi', 'visamp']
        for i in range(len(self._input_src)):
            for mode in allmodes:
                if getattr(self,"_has"+mode):
                    hdu = _pf.PrimaryHDU()
                    hdu.header.set('EXT', 'DATA', comment='Type of information in the HDU')
                    hdu.header.set('DATE', _strftime('%Y%m%dT%H%M%S'), comment='Creation Date')
                    hdu.header.set('DATAFILE', i, comment='Data file number')
                    hdu.header.set('SRC', str(self._input_src[i]), comment='Path to the datafile')
                    hdu.header.set('DATATYPE', mode, comment='Type of data from datafile '+str(i))
                    hdu.header.set('FLATTEN', bool(self._input_flatten[i]), comment='Should the data be flatten')
                    hdu.header.set('SIG_FIG', int(self.significant_figures), comment='Significant figures for u,v coordinates')
                    if mode in ['visphi', 't3phi']:
                        hdu.header.set('DEGREES', bool(self._input_degrees[i]), comment='Is datafile in degrees')
                    hdu.data = _np.ravel(getattr(self, "_input_"+mode)[i])

                    hdu.header.add_comment('Measurement indices to be imported from datafile for the given datatype.')
                    hdu.header.add_comment('Written by Guillaume SCHWORER')
                    hdulist.append(hdu)
        for mode in allmodes:
            if getattr(self,"_has"+mode):
                hdu = _pf.PrimaryHDU()
                hdu.header.set('EXT', 'DATAMASK', comment='Type of information in the HDU')
                hdu.header.set('DATE', _strftime('%Y%m%dT%H%M%S'), comment='Creation Date')
                hdu.header.set('DATATYPE', mode, comment='Type of data from datafile '+str(i))
                hdu.data = getattr(self, mode).mask.astype(_np.uint8)
                hdu.header.add_comment('Data mask for the given datatype for all data contained in the different files.')
                hdu.header.add_comment('Written by Guillaume SCHWORER')
                hdulist.append(hdu)

        if append:
            hdulist.flush()
            hdulist.close()
        else:
            hdulist.writeto(filename, clobber=clobber)
