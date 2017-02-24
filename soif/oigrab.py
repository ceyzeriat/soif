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


try:
    import astropy.io.fits as pf
except ImportError:
    import pyfits as pf

from .oifits import Oifits
from . import oiexception as exc
from . import core
np = core.np

__all__ = ['Oigrab']


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
        hdus = pf.open(self.src)
        for item in hdus:
            if item.header.get('EXTNAME') == 'OI_TARGET':
                hdutgt = item
                break
        else:
            hdus.close()
            if exc.raiseIt(exc.NoTargetTable, self.raiseError, src=self.src):
                return
        self._targets = {}
        for ind, tgt in zip(hdutgt.data["TARGET_ID"], hdutgt.data["TARGET"]):
            self._targets[ind] = tgt
        if not core.hduWlindex(hdus):
            if exc.raiseIt(exc.NoWavelengthTable,
                           self.raiseError,
                           src=self.src):
                return
        # allwl = hdus[self._hduwlidx].data[core.KEYSWL['wl']]
        # self._wlmin = allwl.min()
        # self._wlmax = allwl.max()
        hdus.close()

    def _info(self):
        return "{}<SOIF File>{}\n File: '{}'".format(core.font.blue,
                                                     core.font.normal,
                                                     self.src)

    def __repr__(self):
        return self._info()

    __str__ = __repr__

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        exc.raiseIt(exc.ReadOnly, self.raiseError, attr="targets")

    def show_specs(self, ret=False, **kwargs):
        """
        Gets the target list and the data details from the OIFITS
        file.

        Args:
          * ret (bool): if ``True``, returns the information,
          otherwise prints it

        Returns:
          * a dictionary {'hdu index:info'} where info corresponds to
          a list of (Acquisition index, Target ID, MJD, N(UV), N(wl))
          tuples

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
        hdus = pf.open(self.src)
        if not ret:
            print("TARGETS:")
            for ind, tgt in self.targets.items():
                print("{}: {}".format(ind, tgt))
        tgtlist = {}
        # formattitle = u"\n{} [hdu={:d}]: ({}{} µm)\n{}\n{}"
        formattitle = u"\n{} [hdu={:d}]:\n{}\n{}"
        formathead = u"{:^10} | {:^9} | {:^13} | {:^4} | {:^4}"
        formatline = u"{:>10} | {:>9} | {:>13} | {:>4} | {:>4}"
        for idx, item in enumerate(hdus):
            # if we have data in this header
            if core.hduToDataType(item) is not None:
                targetsortmjd, MJD, (ndata, nset, nunique, nholes, nwl) = \
                    core.gethduMJD(item, withdet=True)
                tgtlist[idx] = []
                if not ret:
                    print(formattitle.format(
                        core.hduToDataType(item),
                        idx,
                        # "{:.3f}".format(self._wlmin*1e6),
                        # "" if self._wlmax == self._wlmin \
                        #    else "-{:.3f}".format(self._wlmax*1e6),
                        formathead.format('Acq. Index', 'Target ID', 'MJD',
                                          'UVs', 'N wl'),
                        '-'*52))
                for tgtidx, sMJD in zip(targetsortmjd, MJD[targetsortmjd, 0]):
                    tgtfilter = slice(tgtidx*nunique, (tgtidx+1)*nunique)
                    tgtid = item.data['TARGET_ID'][tgtfilter]
                    info = (tgtidx, tgtid[0], sMJD, tgtid.size, nwl)
                    if not ret:
                        print(formatline.format(*info))
                    tgtlist[idx].append(info)
        hdus.close()
        if ret:
            return tgtlist

    def filtered(self, tgt=None, mjd=(None, None), hdus=(),
                      vis2=True, t3phi=True, t3amp=True, visphi=True,
                      visamp=True, verbose=False, **kwargs):
        """
        Give filtering parameters on the target name (OI_TARGET table),
        the observation wavelength (OI_WAVELENGTH table), the
        acquisition time [mjd_min, mjd_max], or the hdu index.
        Returns the data indices of the data matching all of these
        different filters. These lists are used to load the data within
        an Oidata object.
        Leave input parameter to 'None' to discard filtering on this
        particular parameter.
        """
        allhdus = pf.open(self.src)
        mjd = (float(mjd[0] if mjd[0] is not None else -np.inf),
               float(mjd[1] if mjd[1] is not None else np.inf))
        datayouwant = {'data': {'VIS2': bool(vis2),
                                'T3PHI': bool(t3phi),
                                'T3AMP': bool(t3amp),
                                'VISPHI': bool(visphi),
                                'VISAMP': bool(visamp)
                                }
                       }
        hdus = core.aslist(hdus)
        for idx, item in enumerate(allhdus):
            # do we want this hdu?
            if len(hdus) > 0 and idx not in hdus:
                    continue
            # is this hdu actual data?
            if core.hduToDataType(item) is not None:
                mjditem = core.gethduMJD(item)[1].ravel()
                filt = ((mjditem >= mjd[0]) & (mjditem <= mjd[1]))
                if tgt is not None:
                    filt = (filt & (item.data.field("TARGET_ID") == int(tgt)))
                if verbose:
                    print("hdu {:d}: {}:\n  {}/{}\n".format(
                                idx,
                                core.hduToDataType(item),
                                filt.sum(),
                                item.data["TARGET_ID"].size))
                if filt.any():
                    datayouwant[idx] = np.arange(item.data["TARGET_ID"].size)[filt]
        allhdus.close()
        return datayouwant

    def extract(self, tgt=None, mjd=(None, None), wl=(None, None), hdus=(),
                vis2=True, t3phi=True, t3amp=True, visphi=True, visamp=True,
                flatten=False, degree=True, significant_figures=5,
                erb_sigma=None, sigma_erb=None, systematic_prior=None,
                systematic_bounds=(), verbose=False, **kwargs):
        datayouwant = self.filtered(
                        tgt=tgt, mjd=mjd, vis2=vis2, hdus=hdus,
                        t3phi=t3phi, t3amp=t3amp, visphi=visphi, visamp=visamp,
                        verbose=verbose, **kwargs)
        return Oifits(src=self.src, datafilter=datayouwant, flatten=flatten,
                      degree=degree, significant_figures=significant_figures,
                      wl=wl, erb_sigma=erb_sigma, sigma_erb=sigma_erb,
                      systematic_prior=systematic_prior,
                      systematic_bounds=systematic_bounds, **kwargs)
