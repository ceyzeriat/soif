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
from . import oiexception as exc
from . import core

__all__ = []


class OidataEmpty(object):
    def __init__(self, datatype, **kwargs):
        self.raiseError = bool(kwargs.pop('raiseError', True))
        self.datatype = str(datatype).upper()
        if self.datatype not in core.DATAKEYSUPPER:
            if exc.raiseIt(exc.InvalidDataType,
                           self.raiseError,
                           datatype=self.datatype):
                return
        self._has = False
        self._useit = False

    def _info(self):
        return "{} data, None".format(self.datatype)

    def __repr__(self):
        return self._info()

    __str__ = __repr__

    @property
    def useit(self):
        """
        Turn this True or False whether you wan't the fitting to
        include this datatype
        """
        return self._useit and self._has

    @useit.setter
    def useit(self, value):
        self._useit = bool(value)

    def __bool__(self):
        return self._has

    __nonzero__ = __bool__

    def __getitem__(self, key):
        return getattr(self, "_"+key)
