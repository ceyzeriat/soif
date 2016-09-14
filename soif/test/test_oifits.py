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




"""def test_extract():
    oig = Oigrab(FILENAME)
    ans1 = oig.extract(tgt=VALIDTGT)
    filt = np.asarray([item[1] for item in oig.show_specs(ret=True)[VALIDHDU]]) == VALIDTGT
    ans2 = Oifits(oig.src, datafilter={VALIDHDU: np.arange(DATASETSIZE)[filt]+1})
    assert np.allclose(ans1.vis2.data, ans2.vis2.data)

"""