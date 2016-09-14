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

"""
A package that includes oifits reading and fitting tools
"""

from time import gmtime as _gmtime


from .oidata import *
from .oifits import *
from .oigrab import *
from .oidataempty import *
from .oigrab import *
from .oipriors import *
from .oimodel import *
try:
    from .oifiting import *
except ImportError:
    pass
from .oiunitmodels import *
try:
    from .oiload import *
except ImportError:
    pass
try:
    from .oipriors import *
except:
    pass

from ._version import __version__, __major__, __minor__, __micro__

_disclaimer = """SOIF  Copyright (C) 2015-%s  Guillaume Schworer
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions.""" % _gmtime()[0]

print(_disclaimer)
