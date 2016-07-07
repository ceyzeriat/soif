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

raise ImportError

import numpy as np
from oidata import Oidata
from oimodel import Oimodel
from oifiting import Oifiting
from _oiunitmodels import *
try:
    import astropy.io.fits as pf
except ImportError:
    import pyfits as pf

from MCres import FakeSampler

# known bug, doesn't keep the dataflag mask correct at loading
# doesn't save the prior type, just the lower and higher bounds
ext = '.oif.fits'
card = 'EXT'


def loaddata(filename):
    if filename.find(ext)==-1: filename += ext
    data = None
    hdulist = pf.open(filename)
    # extract data hdu per hdu
    for item in hdulist:
        if item.header.get(card)=='DATA':
            if data is None: # first set
                data = Oidata(src=item.header.get('SRC'), flatten=item.header.get('FLATTEN'), degrees=item.header.get('DEGREES'), significant_figures=item.header.get('SIG_FIG'), **{item.header.get('DATATYPE'):item.data})
            else:
                data.addData(src=item.header.get('SRC'), flatten=item.header.get('FLATTEN'), degrees=item.header.get('DEGREES'), significant_figures=item.header.get('SIG_FIG'), **{item.header.get('DATATYPE'):item.data})
    # apply data mode per mode
    for item in hdulist:
        if item.header.get(card)=='DATAMASK':
            theoldmask = getattr(data, item.header.get('DATATYPE')).mask
            if theoldmask.shape != item.data.shape:
                print("Error while applying data mask on "+item.header.get('DATATYPE'))
            else:
                getattr(data, item.header.get('DATATYPE')).mask = np.logical_or(theoldmask, item.data.astype(bool))
    return data


def loadmodel(filename):
    if filename.find(ext)==-1: filename += ext
    return Oimodel(loaddata(filename), loadobj(filename))


def load(filename):
    cardres = 'MCRES'
    try:
        hdulist = pf.open(filename)
    except IOError:
        if filename.find(ext)==-1: filename += ext
        hdulist = pf.open(filename)
    for item in hdulist:
        if item.header.get(card)==cardres:
            sampler = FakeSampler(item.data[:,:-1], item.data[:,-1])
    return Oifiting(sampler=sampler, model=loadmodel(filename))


def loadobj(filename, name=None):
    if filename.find(ext)==-1: filename += ext
    data = []
    hdulist = pf.open(filename)

    for item in hdulist:
        if item.header.get(card) == 'OBJ':
            if name is None or str(name).upper()==item.header.get('NAME').upper():
                priors = {}
                kwargs = {}
                for i in range(item.header.get('NVALUE')):
                    key = item.header.get('VALUE'+str(i))
                    if item.header.get(key)!='NONE':
                        priors[key] = item.header.get(key)
                bounds = {}
                for i in range(item.header.get('NPARAM')):
                    key = item.header.get('PARAM'+str(i))
                    L = item.header.get(key+'L')
                    H = item.header.get(key+'H')
                    if item.header.get(key)!='NONE':
                        priors[key] = item.header.get(key)
                    bounds[key] = [L, H]
                for i in range(item.header.get('NSAVE')):
                    key = item.header.get('SAVE'+str(i))
                    vv = item.header.get(key)
                    if vv != 'ARRAY':
                        kwargs[key] = vv
                    else:
                        uniq = item.header.get('KEY'+str(i))
                        for itemuniq in hdulist:
                            if itemuniq.header.get(card) == uniq:
                                kwargs[key] = itemuniq.data

                data.append(globals()[item.header.get('TYP')](name=item.header.get('NAME'), priors=priors, bounds=bounds, **kwargs))
    return data
