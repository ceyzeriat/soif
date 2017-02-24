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
try:
    import astropy.io.fits as pf
except ImportError:
    import pyfits as pf

from . import oiexception as exc
from .oidataempty import OidataEmpty
from .oidata import Oidata
from . import core
np = core.np

__all__ = ['Oifits']


class Oifits(object):
    """
    This class opens, reads and sorts data contained in the file
    'src' (oifits format).
    """
    def __init__(self, src, datafilter, wl=(None, None), erb_sigma=None,
                 sigma_erb=None, systematic_prior=None, systematic_bounds=(),
                 flatten=False, degree=True, significant_figures=5, **kwargs):
        self.raiseError = bool(kwargs.pop('raiseError', True))
        # initialize empty data
        for key in core.DATAKEYSLOWER:
            setattr(self, key, OidataEmpty(key))

        self.erb_sigma = core.ident if erb_sigma is None else erb_sigma
        if not callable(self.erb_sigma):
            if exc.raiseIt(exc.NotCallable, self.raiseError, fct="erb_sigma"):
                return
        self.sigma_erb = core.ident if sigma_erb is None else sigma_erb
        if not callable(self.sigma_erb):
            if exc.raiseIt(exc.NotCallable, self.raiseError, fct="sigma_erb"):
                return
        self.systematic_bounds = None \
            if len(systematic_bounds) <= 2  \
            else list(map(float, list(systematic_bounds)[:2]))
        self.systematic_prior = None \
            if systematic_prior is None \
            else float(systematic_prior)
        self._systematic_prior = self.systematic_prior
        # init parameters
        self.addData(src=str(src), datafilter=dict(datafilter),
                     flatten=bool(flatten), degree=bool(degree),
                     significant_figures=int(significant_figures),
                     wl=list(wl)[:2], noupdate=True, **kwargs)
        self.update()

    def _info(self):
        txt = "\n".join([" {}".format(getattr(self, key))
                         for key in core.DATAKEYSLOWER
                         if getattr(self, key).useit])
        txt = " No data" if txt == "" else txt
        txt = "{}<SOIF Data>{}\n{}".format(core.font.blue,
                                           core.font.normal,
                                           txt)
        return txt

    def __repr__(self):
        return self._info()

    __str__ = __repr__

    def addData(self, src, datafilter=None, flatten=False, degree=True,
                significant_figures=5, wl=(None, None), **kwargs):
        if datafilter is None:
            datafilter = {}
        if isinstance(src, Oifits):
            thedata = src
            for datatype in core.DATAKEYSLOWER:
                if not getattr(thedata, datatype):
                    continue
                if getattr(self, datatype):
                    getattr(self, datatype)._addData(
                                                getattr(thedata, datatype),
                                                flatten=flatten,
                                                **kwargs)
                else:
                    setattr(self, datatype, getattr(thedata, datatype))
        else:
            whichdata = datafilter.get('data',
                                       {'data': {'VIS2': True,
                                                 'T3PHI': True,
                                                 'T3AMP': True,
                                                 'VISPHI': True,
                                                 'VISAMP': True}})
            hdus = pf.open(src)  # open
            hduwlidx = core.hduWlindex(hdus)
            if not hduwlidx:
                if exc.raiseIt(exc.NoWavelengthTable,
                               self.raiseError,
                               src=src):
                    return
            # get wl sorted
            wl = [float(wl[0] if wl[0] is not None else -np.inf),
                  float(wl[1] if wl[1] is not None else np.inf)]
            allwl = hdus[hduwlidx].data[core.KEYSWL['wl']]
            wlindices = np.arange(allwl.size)[((allwl >= wl[0])
                                               & (allwl < wl[1]))]
            # for each datafilter
            for idx, indices in datafilter.items():
                if not isinstance(idx, int) or np.size(indices) == 0:
                    continue
                # if real data
                if core.hduToDataType(hdus[idx]) is not None:
                    hduextname = hdus[idx].header['EXTNAME']
                    for datatype in core.ALLDATAEXTNAMES[hduextname]:
                        if not whichdata.get(datatype.upper(), True):
                            continue
                        thedata = Oidata(
                            src=src, hduidx=idx, datatype=datatype,
                            hduwlidx=hduwlidx, indices=indices,
                            wlindices=wlindices, degree=degree,
                            flatten=flatten,
                            significant_figures=significant_figures, **kwargs)
                        if getattr(self, datatype.lower()):
                            getattr(self, datatype.lower())._addData(
                                                            thedata,
                                                            flatten=flatten,
                                                            **kwargs)
                        else:
                            setattr(self, datatype.lower(), thedata)
                else:
                    hdus.close()
                    if exc.raiseIt(exc.NotADataHdu,
                                   self.raiseError,
                                   idx=idx,
                                   src=str(src)):
                        return
            hdus.close()
        if not kwargs.pop('noupdate', False):
            self.update()

    def flatten(self):
        """
        Flattens all data contained in the Oidata object. This can be
        useful in order to add several bits of data that do not have
        the same shapes
        """
        for key in core.DATAKEYSLOWER:
            if getattr(self, key):
                getattr(self, key).flatten()
        self.update()

    def update(self):
        """
        Updates all data contained in the Oidata object
        """
        for key in core.DATAKEYSLOWER:
            if getattr(self, key):
                getattr(self, key).update()

        funkydtypeint = [('u', int), ('v', int), ('wl', int)]
        funkydtype = [('u', np.float32), ('v', np.float32), ('wl', np.float32)]

        # get uvwl sets as integer to extract uniques
        unique_uvwl = np.zeros(0, dtype=funkydtypeint)
        for key in core.DATAKEYSLOWER:
            thedata = getattr(self, key)
            if thedata:
                dum = np.zeros(thedata.shapeuv, dtype=funkydtypeint)
                dum['u'] = core.round_fig(x=thedata.u,
                                          n=thedata.significant_figures,
                                          retint=True)
                dum['v'] = core.round_fig(x=thedata.v,
                                          n=thedata.significant_figures,
                                          retint=True)
                dum['wl'] = core.round_fig(x=thedata.wl,
                                           n=thedata.significant_figures,
                                           retint=True)
                # deal with symmetry
                inv = (dum['u'] < 0)
                dum['u'][inv] *= -1
                dum['v'][inv] *= -1
                # stack
                unique_uvwl = np.hstack((unique_uvwl, dum.flatten()))
        if core.OLDNUMPY:
            uvwlind = np.unique((unique_uvwl['u']+1j*unique_uvwl['v'])
                                / unique_uvwl['wl'],
                                return_index=True)[1]
        else:
            uvwlind = np.unique(unique_uvwl, return_index=True)[1]
        # get uvwl sets as floats for calculations
        unique_uvwl = np.zeros(0, dtype=funkydtype)
        for key in core.DATAKEYSLOWER:
            thedata = getattr(self, key)
            if thedata:
                dum = np.zeros(thedata.shapeuv, dtype=funkydtype)
                dum['u'] = core.round_fig(x=thedata.u,
                                          n=thedata.significant_figures)
                dum['v'] = core.round_fig(x=thedata.v,
                                          n=thedata.significant_figures)
                dum['wl'] = core.round_fig(x=thedata.wl,
                                           n=thedata.significant_figures)
                # simplify symmetry
                inv = (dum['u'] < 0)
                dum['u'][inv] *= -1
                dum['v'][inv] *= -1
                # save the phase symmetry
                if thedata.is_angle:
                    thedata._phasesign = 1-inv*2
                # stack
                unique_uvwl = np.hstack((unique_uvwl, dum.flatten()))
                thedata._ind = dum.copy()
        # extract uniques
        dum = np.zeros(uvwlind.shape, dtype=funkydtype)
        dum['u'] = unique_uvwl['u'][uvwlind]
        dum['v'] = unique_uvwl['v'][uvwlind]
        dum['wl'] = unique_uvwl['wl'][uvwlind]
        unique_uvwl = dum
        # save indices
        for key in core.DATAKEYSLOWER:
            thedata = getattr(self, key)
            if thedata:
                dum = np.zeros(thedata._ind.shape, dtype='int')
                for i, v in enumerate(unique_uvwl):
                    dum[thedata._ind == v] = i
                thedata._ind = dum.copy()
        # prepare pre-processed uniques
        self.uvwl = {'u': unique_uvwl['u'],
                     'v': unique_uvwl['v'],
                     'wl': unique_uvwl['wl'],
                     'blwl': np.hypot(unique_uvwl['u'], unique_uvwl['v'])
                             / unique_uvwl['wl']}
        self._wlmin = self.uvwl['wl'].min()
        self._wlmax = self.uvwl['wl'].max()
        self._wlspan = self._wlmax - self._wlmin

    @property
    def systematic_fit(self):
        return isinstance(self.systematic_bounds, list)

    @systematic_fit.setter
    def systematic_fit(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="systematic_fit")

    def systematic_p0(self):
        if self.systematic_fit:
            randomizer = core.gen_generator()
            return randomizer.uniform(low=self.systematic_bounds[0],
                                      high=self.systematic_bounds[1])
        else:
            exc.raiseIt(exc.NoSystematicsFit, self.raiseError)

    def remorph(self, viscomp):
        ret = []
        for key in core.DATAKEYSLOWER:
            thedata = getattr(self, key)
            if thedata.useit:
                retdbl = isinstance(viscomp, (tuple, list))
                if retdbl:
                    viscomp, flx = viscomp
                if thedata.is_t3:
                    fct = core.FCTVISCOMP[key.upper()]
                    if thedata.is_angle:  # t3phi
                        dum = (fct(viscomp)[thedata._ind]
                               * thedata._phasesign).sum(-1)
                    else:  # t3amp
                        dum = (fct(viscomp[thedata._ind])).prod(-1)
                else:
                    dum = core.FCTVISCOMP[key.upper()](viscomp)[thedata._ind]
                    if thedata.is_angle:  # visphi
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
        if filename.find(ext) == -1:
            filename += ext
        if append:
            hdulist = pf.open(filename, mode='append')
        else:
            hdulist = pf.HDUList()
        allmodes = ['vis2', 't3phi', 't3amp', 'visphi', 'visamp']
        for i in range(len(self._input_src)):
            for mode in allmodes:
                if getattr(self, "_has"+mode):
                    hdu = pf.PrimaryHDU()
                    hdu.header.set('EXT',
                                   'DATA',
                                   comment='Type of information in the HDU')
                    hdu.header.set('DATE',
                                   strftime('%Y%m%dT%H%M%S'),
                                   comment='Creation Date')
                    hdu.header.set('DATAFILE', i, comment='Data file number')
                    hdu.header.set('SRC',
                                   str(self._input_src[i]),
                                   comment='Path to the datafile')
                    hdu.header.set('DATATYPE',
                                   mode,
                                   comment='Type of data from'
                                           ' datafile {:d}'.format(i))
                    hdu.header.set('FLATTEN',
                                   bool(self._input_flatten[i]),
                                   comment='Should the data be flatten')
                    hdu.header.set('SIG_FIG',
                                   int(self.significant_figures),
                                   comment='Significant figures for'
                                           'u,v coordinates')
                    if mode in ['visphi', 't3phi']:
                        hdu.header.set('DEGREES',
                                       bool(self._input_degrees[i]),
                                       comment='Is datafile in degrees')
                    hdu.data = np.ravel(getattr(self, "_input_"+mode)[i])

                    hdu.header.add_comment('Measurement indices to be imported'
                                           ' from datafile for the given'
                                           ' datatype.')
                    hdu.header.add_comment('Written by Guillaume SCHWORER')
                    hdulist.append(hdu)
        for mode in allmodes:
            if getattr(self, "_has"+mode):
                hdu = pf.PrimaryHDU()
                hdu.header.set('EXT',
                               'DATAMASK',
                               comment='Type of information in the HDU')
                hdu.header.set('DATE',
                               strftime('%Y%m%dT%H%M%S'),
                               comment='Creation Date')
                hdu.header.set('DATATYPE',
                               mode,
                               comment='Type of data from datafile '+str(i))
                hdu.data = getattr(self, mode).mask.astype(np.uint8)
                hdu.header.add_comment('Data mask for the given datatype for'
                                       ' all data contained in the different'
                                       ' files.')
                hdu.header.add_comment('Written by Guillaume SCHWORER')
                hdulist.append(hdu)

        if append:
            hdulist.flush()
            hdulist.close()
        else:
            hdulist.writeto(filename, clobber=clobber)
