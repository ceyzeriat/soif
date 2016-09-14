#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
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
################################################################################


import numpy as np
import os
from nose.tools import raises

from ..oidata import Oidata
from ..oigrab import Oigrab
from ..oidataempty import OidataEmpty
from ..oifits import Oifits
from .. import oiexception as exc

FILENAME = os.path.dirname(os.path.abspath(__file__)) + '/test.oifits'
FILENAME2 = os.path.dirname(os.path.abspath(__file__)) + '/test2.oifits'
FILENAME_FULL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_full.oifits')
VALIDHDU = 4
VALIDHDUT3 = 4
VALIDHDUFAKET3 = 6
WLHDUT3 = 2
WLHDU = 3
VALIDTGT = 1


def test_create():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    assert str(oid) == repr(oid)
    assert oid
    assert bool(oid)
    oid.useit = False
    assert oid
    assert bool(oid)
    assert not oid.useit
    assert oid.shapedata == (4,38)
    assert oid.shapedata == oid.shapeuv
    assert oid.data.shape == oid.shapedata
    assert oid.error.shape == oid.shapedata
    assert oid.wl.shape == oid.shapedata
    assert oid.wl_d.shape == oid.shapedata
    assert oid.u.shape == oid.shapedata
    assert oid.v.shape == oid.shapedata
    assert oid.pa.shape == oid.shapedata
    assert oid.bl.shape == oid.shapedata
    assert oid.blwl.shape == oid.shapedata
    oid.flatten()
    assert oid.shapedata == (152,)
    assert oid.shapedata == oid.shapeuv
    assert oid.data.shape == oid.shapedata
    assert oid.error.shape == oid.shapedata
    assert oid.wl.shape == oid.shapedata
    assert oid.wl_d.shape == oid.shapedata
    assert oid.u.shape == oid.shapedata
    assert oid.v.shape == oid.shapedata
    assert oid.pa.shape == oid.shapedata
    assert oid.bl.shape == oid.shapedata
    assert oid.blwl.shape == oid.shapedata

def test_create_T3():
    oig = Oigrab(FILENAME2)
    datafilter = oig.filtered(tgt=VALIDTGT, t3amp=False)
    oid = Oidata(src=FILENAME2, hduidx=VALIDHDUT3, datatype="T3PHI", hduwlidx=WLHDUT3, indices=datafilter[VALIDHDUT3])
    assert str(oid) == repr(oid)
    assert oid
    assert bool(oid)
    oid.useit = False
    assert oid
    assert bool(oid)
    assert not oid.useit
    assert len(oid.shapedata) == 2
    assert len(oid.shapeuv) == 3
    assert oid.shapeuv[-1] == 3
    assert oid.data.shape == oid.shapedata
    assert oid.error.shape == oid.shapedata
    assert oid.wl.shape == oid.shapeuv
    assert oid.wl_d.shape == oid.shapeuv
    assert oid.u.shape == oid.shapeuv
    assert oid.v.shape == oid.shapeuv
    assert oid.pa.shape == oid.shapeuv
    assert oid.bl.shape == oid.shapeuv
    assert oid.blwl.shape == oid.shapeuv
    oid.flatten()
    assert len(oid.shapedata) == 1
    assert len(oid.shapeuv) == 2
    assert oid.shapeuv[-1] == 3
    assert oid.data.shape == oid.shapedata
    assert oid.error.shape == oid.shapedata
    assert oid.wl.shape == oid.shapeuv
    assert oid.wl_d.shape == oid.shapeuv
    assert oid.u.shape == oid.shapeuv
    assert oid.v.shape == oid.shapeuv
    assert oid.pa.shape == oid.shapeuv
    assert oid.bl.shape == oid.shapeuv
    assert oid.blwl.shape == oid.shapeuv

@raises(exc.shapeIssue)
def test_shapeIssue():
    oig = Oigrab(FILENAME_FULL)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME_FULL, hduidx=VALIDHDUFAKET3, datatype="T3PHI", hduwlidx=WLHDU, indices=datafilter[VALIDHDUFAKET3])
    assert oid.shapedata + (3,) == oid.shapeuv


"""def test_masking():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])

def test_oigrab():
    oig = Oigrab(FILENAME)
    assert len(oig.targets) == 3
    assert oig.targets[0] == 'HD_204770'
    assert str(oig) == repr(oig)"""


@raises(exc.HduDatatypeMismatch)
def test_HduDatatypeMismatch():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="T3AMP", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])

def test_HduDatatypeMismatch_noraise():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="T3AMP", hduwlidx=WLHDU, indices=datafilter[VALIDHDU], raiseError=False)

@raises(exc.ReadOnly)
def test_data_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.data = 'random'

@raises(exc.ReadOnly)
def test_error_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.error = 'random'

@raises(exc.ReadOnly)
def test_u_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.u = 'random'

@raises(exc.ReadOnly)
def test_v_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.v = 'random'

@raises(exc.ReadOnly)
def test_wl_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.wl = 'random'

@raises(exc.ReadOnly)
def test_wl_d_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.wl_d = 'random'

@raises(exc.ReadOnly)
def test_bl_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.bl = 'random'

@raises(exc.ReadOnly)
def test_paa_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.pa = 'random'

@raises(exc.ReadOnly)
def test_blwl_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.blwl = 'random'

@raises(exc.ReadOnly)
def test_shapedata_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.shapedata = 'random'

@raises(exc.ReadOnly)
def test_shapeuv_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.shapeuv = 'random'

@raises(exc.ReadOnly)
def test_is_angle_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.is_angle = 'random'

@raises(exc.ReadOnly)
def test_is_t3_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.is_t3 = 'random'

@raises(exc.ReadOnly)
def test_flat_readonly():
    oig = Oigrab(FILENAME)
    datafilter = oig.filtered(tgt=VALIDTGT)
    oid = Oidata(src=FILENAME, hduidx=VALIDHDU, datatype="VIS2", hduwlidx=WLHDU, indices=datafilter[VALIDHDU])
    oid.flat = 'random'
