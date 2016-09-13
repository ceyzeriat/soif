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
from ..oigrab import Oigrab
from ..oidataempty import OidataEmpty
from ..oifits import Oifits
from .. import oiexception as exc

FILENAME = os.path.dirname(os.path.abspath(__file__)) + '/MWC361.oifits'
FILENAME_NOTARGET = os.path.dirname(os.path.abspath(__file__)) + '/MWC361_notarget.oifits'
FILENAME_NOWL = os.path.dirname(os.path.abspath(__file__)) + '/MWC361_nowl.oifits'
VALIDHDU = 4
DATASETSIZE = 12

def test_oigrab():
    oig = Oigrab(FILENAME)
    assert len(oig.targets) == 3
    assert oig.targets[0] == 'HD_204770'
    assert str(oig) == repr(oig)

@raises(exc.NoTargetTable)
def test_oigrab_NoTargetTable():
    oig = Oigrab(FILENAME_NOTARGET)
