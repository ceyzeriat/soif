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

from time import gmtime as _gmtime

_disclaimer = """OIF  Copyright (C) 2015-%s  Guillaume Schworer
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions.""" % _gmtime()[0]

print(_disclaimer)


"""
A package that includes oifits reading and fitting tools
"""

from . import oidata
from . import oipriors
from . import oimodel
try:
	from . import oifiting
except ImportError:
	pass
from . import oiunitmodels
try:
	from . import oiload
except ImportError:
	pass
from ._version import __version__, __major__, __minor__

#import oipriors

