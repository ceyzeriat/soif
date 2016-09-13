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


import numpy as np
import os
from nose.tools import raises

from ..oidata import Oidata
from ..oidataempty import OidataEmpty
from ..oifits import Oifits
from ..oigrab import Oigrab
from .. import oiexception as exc

FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MWC361.oifits')
FILENAME_NOTARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MWC361_notarget.oifits')
FILENAME_NOWL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MWC361_nowl.oifits')
FILENAME_FULL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MWC361_full.oifits')
VALIDHDU = 4
DATASETSIZE = 12
VALIDTGT = 1

def test():
    oig = Oigrab(FILENAME)
    assert len(oig.targets) == 3
    assert oig.targets[0] == 'HD_204770'
    assert str(oig) == repr(oig)

@raises(exc.NoTargetTable)
def test_NoTargetTable():
    oig = Oigrab(FILENAME_NOTARGET)

@raises(exc.NoWavelengthTable)
def test_NoWavelengthTable():
    oig = Oigrab(FILENAME_NOWL)

@raises(exc.ReadOnly)
def test_NoTargetTable():
    oig = Oigrab(FILENAME)
    oig.targets = []

def test_show_specs():
    oig = Oigrab(FILENAME)
    ans = oig.show_specs(ret=True)
    for item in range(10):
        if item != VALIDHDU:
            assert ans.get(item) is None
    assert len(ans[VALIDHDU]) == DATASETSIZE
    assert np.allclose(ans[VALIDHDU][0], (0, 0, 57190.4437, 1, 38))
    assert (np.diff([item[2] for item in ans[VALIDHDU]]) >= 0).all()
    

def test_show_filtered():
    oig = Oigrab(FILENAME)
    for item in range(10):
        if item != VALIDHDU:
            assert oig.show_filtered(tgt=VALIDTGT).get(item) is None
        else:
            assert oig.show_filtered(tgt=VALIDTGT).get(item).tolist() == [ 2,  5,  8, 11]
    oig = Oigrab(FILENAME_FULL)

def test_extract():
    oig = Oigrab(FILENAME)
    ans1 = oig.extract(tgt=VALIDTGT)
    filt = np.asarray([item[1] for item in oig.show_specs(ret=True)[VALIDHDU]]) == VALIDTGT
    ans2 = Oifits(oig.src, datafilter={VALIDHDU: np.arange(DATASETSIZE)[filt]+1})
    assert np.allclose(ans1.vis2.data, ans2.vis2.data)
