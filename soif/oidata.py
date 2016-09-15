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


# import matplotlib.pyplot as _plt
try:
    import astropy.io.fits as pf
except ImportError:
    import pyfits as pf

from .oidataempty import OidataEmpty
from . import oiexception as exc
from . import core
np = core.np

__all__ = []


class Oidata(OidataEmpty):
    def __init__(self, src, hduidx, datatype, hduwlidx, indices=(),
                 wlindices=(), degree=True, flatten=False,
                 significant_figures=5, **kwargs):
        super(Oidata, self).__init__(datatype=datatype, **kwargs)
        self._input_src = [str(src)]
        self._input_hduidx = [int(hduidx)]
        self._input_hduwlidx = [int(hduwlidx)]
        hdus = pf.open(self._input_src[-1])
        hdu = hdus[self._input_hduidx[-1]]
        hduwl = hdus[self._input_hduwlidx[-1]]

        if core.DATAKEYSDATATYPE[self.datatype]['data'] \
           not in core.hduToColNames(hdu):
            if exc.raiseIt(exc.HduDatatypeMismatch,
                           self.raiseError,
                           hduhead=core.hduToDataType(hdu),
                           datatype=self.datatype):
                return

        self._input_degree = [bool(degree)]
        self._input_flatten = [bool(flatten)]
        self._input_significant_figures = [int(significant_figures)]
        self._input_indices = [list(indices)]
        self._input_wlindices = [list(wlindices)]

        self._has = True
        self._useit = True
        wlindices = slice(0, 10000000) if wlindices == () \
            else np.asarray(wlindices).ravel()
        indices = slice(0, 10000000) if indices == () \
            else np.asarray(indices).ravel()

        # attributes to be copy-pasted
        for key, vl in core.ATTRDATATYPE[self.datatype].items():
            setattr(self, '_'+key, vl)

        # wavelength attributes to be extracted out of OIFITS
        for key, vl in core.KEYSWL.items():
            setattr(self, '_'+key, hduwl.data[vl].ravel()[wlindices])
        self._wlsize = self._wl.size
        key = core.DATAKEYSDATATYPE[self.datatype]['data']
        self._datasize = hdu.data[key][indices].shape[0]

        # data attributes to be extracted out of OIFITS
        for key, vl in core.DATAKEYSDATATYPE[self.datatype].items():
            setattr(self,
                    '_'+key,
                    hdu.data[vl][indices]
                    .reshape((self._datasize, -1))[:, wlindices])
        # swaps Trues to Falses to have a good mask
        self.mask = np.logical_not(self._mask)

        # data attributes to be extracted out of OIFITS and replicated with wl
        for key, vl in core.UVKEYSDATATYPE[self.datatype].items():
            setattr(self,
                    '_'+key,
                    core.replicate(hdu.data[vl][indices].ravel(),
                                   (None, self._wlsize)))

        # done with the file
        hdus.close()

        # add the data dimension on wl-like attributes
        for key in core.KEYSWL.keys():
            setattr(self,
                    "_"+key,
                    core.replicate(self[key],
                                   (self._datasize, None)))

        # combine the UV coordinates of T3
        if self.is_t3:
            self._u = np.concatenate((np.expand_dims(self._u1, -1),
                                      np.expand_dims(self._u2, -1),
                                      np.expand_dims(-self._u1-self._u2, -1)),
                                     axis=-1)
            self._v = np.concatenate((np.expand_dims(self._v1, -1),
                                      np.expand_dims(self._v2, -1),
                                      np.expand_dims(-self._v1-self._v2, -1)),
                                     axis=-1)
            # delete temporary attributes
            for key in core.UVKEYSDATATYPE[self.datatype].keys():
                delattr(self, "_"+key)
            # add the data dimension on wl-like attributes
            for key in core.KEYSWL.keys():
                setattr(self,
                        "_"+key,
                        core.replicate(self[key], (None, 3)))

        # check shape sanity - might be raised in case there are several wl
        # tables in the oifits. This feature is not covered by this library
        extra_dim = (3,) if self.is_t3 else ()
        if not self._u.shape == self._data.shape + extra_dim:
            if exc.raiseIt(exc.shapeIssue,
                           self.raiseError,
                           uvshape=self._u.shape[:2],
                           datashape=self._data.shape):
                return

        # convert to radian if needed
        if self.is_angle and degree:
            self._data = self._data*core.DEG2RAD
            self._error = self._error*core.DEG2RAD

        self.significant_figures = min(8, max(1, int(significant_figures)))
        self._flat = bool(flatten)
        if self._flat:
            self.flatten()
        else:
            self.update()

    def _info(self):
        return str(u"{} data, shape: {}, wl: {:.2f}{} \xb5m".format(
            self.datatype,
            core.maskedshape(self.shapedata,
                             np.logical_not(self.mask).sum()),
            self._wlmin*1e6,
            u" to {:.2f}".format(self._wlmax*1e6) if self._wlspan != 0 else u""
            ).encode('utf-8'))

    @property
    def data(self):
        if self._use_mask:
            return self._data[self.mask]
        else:
            return self._data

    @data.setter
    def data(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="data")

    @property
    def error(self):
        if self._use_mask:
            return self._error[self.mask]
        else:
            return self._error

    @error.setter
    def error(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="error")

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        if value is True or value is None:  # removes the mask
            self._use_mask = False
            self._mask = np.ones(self._data.shape, dtype=bool)
        elif value is False:  # set up full masking
            self._use_mask = True
            self._mask = np.zeros(self._data.shape, dtype=bool)
        else:  # convert value to proper bool mask
            value = np.asarray(value, dtype=bool)
            if value.shape != self._data.shape:
                if exc.raiseIt(exc.BadMaskShape,
                               self.raiseError,
                               shape=self._data.shape):
                    return False
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
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="u")

    @property
    def v(self):
        if self._use_mask:
            return self._v[self.mask]
        else:
            return self._v

    @v.setter
    def v(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="v")

    @property
    def wl(self):
        if self._use_mask:
            return self._wl[self.mask]
        else:
            return self._wl

    @wl.setter
    def wl(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="wl")

    @property
    def wl_d(self):
        if self._use_mask:
            return self._wl_d[self.mask]
        else:
            return self._wl_d

    @wl_d.setter
    def wl_d(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="wl_d")

    @property
    def bl(self):
        if self._use_mask:
            return self._bl[self.mask]
        else:
            return self._bl

    @bl.setter
    def bl(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="bl")

    @property
    def pa(self):
        if self._use_mask:
            return self._pa[self.mask]
        else:
            return self._pa

    @pa.setter
    def pa(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="pa")

    @property
    def blwl(self):
        if self._use_mask:
            return self._blwl[self.mask]
        else:
            return self._blwl

    @blwl.setter
    def blwl(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="blwl")

    @property
    def shapedata(self):
        return self.data.shape

    @shapedata.setter
    def shapedata(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="shapedata")

    @property
    def shapeuv(self):
        return self.u.shape

    @shapeuv.setter
    def shapeuv(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="shapeuv")

    @property
    def is_angle(self):
        return self._is_angle

    @is_angle.setter
    def is_angle(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="is_angle")

    @property
    def is_t3(self):
        return self._is_t3

    @is_t3.setter
    def is_t3(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="is_t3")

    @property
    def flat(self):
        return self._flat

    @flat.setter
    def flat(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="flat")

    def flatten(self, **kwargs):
        if self.is_t3:
            for key in core.KEYSUV:
                setattr(self, "_"+key, self[key].reshape((-1, 3)))
            for key in core.KEYSDATA:
                setattr(self, "_"+key, self[key].ravel())
        else:
            for key in core.KEYSDATA + core.KEYSUV:
                setattr(self, "_"+key, self[key].ravel())
        self.update()
        self._flat = True

    def _addData(self, data, flatten=True, **kwargs):
        if isinstance(data, OidataEmpty):
            return  # trivial, nothing to add
        if not isinstance(data, Oidata):
            if exc.raiseIt(exc.WrongData,
                           exc.doraise(self, **kwargs),
                           typ='Oidata'):
                return False
        if self.datatype != data.datatype:
            if exc.raiseIt(exc.IncompatibleData,
                           exc.doraise(self, **kwargs),
                           typ1=self.datatype,
                           typ2=data.datatype):
                return False
        # do we flatten it?
        if flatten or self.shapedata[-1] != data.shapedata[-1]:
            self.flatten()
            data.flatten()
        # concatenate data
        for key in core.KEYSDATA + core.KEYSUV:
            setattr(self,
                    "_"+key,
                    np.concatenate((self[key], data[key]), axis=0))
        # update input keys
        for key in core.INPUTSAVEKEY:
            setattr(self,
                    "_input_"+key,
                    self["input_"+key] + data["input_"+key])
        # update the data
        self.update()

    def update(self, **kwargs):
        """
        Given u, v, wl and flag information as object properties, this
        function updates the Oidata object: the data masking (from the
        new mask property) and the bl, pa, blwl properties (from u, v
        and wl properties)
        """
        self._u = core.round_fig(self._u,
                                 self.significant_figures)
        self._v = core.round_fig(self._v,
                                 self.significant_figures)
        self._wl = core.round_fig(self._wl,
                                  self.significant_figures)
        self._wl_d = core.round_fig(self._wl_d,
                                    self.significant_figures)
        if (self.error == 0).any():
            if exc.raiseIt(exc.ZeroErrorbars, exc.doraise(self, **kwargs)):
                return False
        self._invvar = 1./self.error**2
        self._bl = core.round_fig(np.hypot(self['v'], self['u']),
                                  self.significant_figures)
        self._pa = core.round_fig(np.arctan2(self['v'], self['u']),
                                  self.significant_figures)
        self._blwl = core.round_fig(self['bl']/self['wl'],
                                    self.significant_figures)
        self._wlmin = self.wl.min()
        self._wlmax = self.wl.max()
        self._wlspan = self._wlmax - self._wlmin
