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


import pytest

import numpy as np

from multi import M

from .. import _core

def test_abs2():
	assert _core.abs2(-2) == 4
	assert np.abs(_core.abs2(3.1) - 9.61) < 1e-13

def test_gen_seed():
	seeds = [_core.gen_seed() for i in range(100)]
	assert len(seeds) == len(set(seeds))
	seeds = M(_core.gen_seed, n=100)()
	assert len(seeds) == len(set(seeds))

