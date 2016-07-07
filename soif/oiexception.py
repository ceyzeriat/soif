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

def doraise(obj, **kwargs):
    return bool(kwargs.pop('raiseError', getattr(obj, 'raiseError', True)))

def raiseIt(exc, raiseoupas, *args, **kwargs):
    exc = exc(*args, **kwargs)
    if raiseoupas:
        raise exc
    else:
        print("\033[31m"+exc.message+"\033[39m")
        return True
    return False

class OIException(Exception):
    """
    Root for SOIF Exceptions, only used to trigger any soif errors, never raised
    """
    def __init__(self, *args, **kwargs):
        self.args = [a for a in args] + [a for a in kwargs.values()]
    def __str__(self):
        return repr(self.message)
    def __repr__(self):
        return repr(self.message)

class NoTargetTable(OIException):
    """
    If the file has no OITARGET table
    """
    def __init__(self, src="", *args, **kwargs):
        super(NoTargetTable, self).__init__(src, *args, **kwargs)
        self.message = "There's no OI_TARGET table in the oifits file '%s'! Go get some coffee!" % (src)

class NoWavelengthTable(OIException):
    """
    If the file has no OITARGET table
    """
    def __init__(self, src="", *args, **kwargs):
        super(NoWavelengthTable, self).__init__(src, *args, **kwargs)
        self.message = "There's no OI_WAVELENGTH table in the oifits file '%s'! You're pretty much screwed!" % (src)

class ReadOnly(OIException):
    """
    If the parameter is read-only
    """
    def __init__(self, attr, *args, **kwargs):
        super(ReadOnly, self).__init__(attr, *args, **kwargs)
        self.message = "Attribute '%s' is read-only" % (attr)

class InvalidDataType(OIException):
    """
    If the data type provided does not exist
    """
    def __init__(self, datatype, *args, **kwargs):
        super(InvalidDataType, self).__init__(datatype, *args, **kwargs)
        self.message = "Data type '%s' does not exist" % (datatype)

class HduDatatypeMismatch(OIException):
    """
    If the data type and the hdu provided do not match
    """
    def __init__(self, hduhead, datatype, *args, **kwargs):
        super(HduDatatypeMismatch, self).__init__(hduhead, datatype, *args, **kwargs)
        self.message = "Data type '%s' and hdu with '%s' data do not match" % (datatype, hduhead)

class BadMaskShape(OIException):
    """
    If the mask shape does not match the data shape
    """
    def __init__(self, shape, *args, **kwargs):
        super(BadMaskShape, self).__init__(shape, *args, **kwargs)
        self.message = "Bad mask shape. Should be '%s'" % shape

class WrongData(OIException):
    """
    If the data provided has the wrong data type
    """
    def __init__(self, typ, *args, **kwargs):
        super(WrongData, self).__init__(typ, *args, **kwargs)
        self.message = "Wrong data given, should be '%s'" % typ

class IncompatibleData(OIException):
    """
    If the data type and the hdu provided do not match
    """
    def __init__(self, typ1, typ2, *args, **kwargs):
        super(IncompatibleData, self).__init__(typ1, typ2, *args, **kwargs)
        self.message = "Can't merge '%s' and '%s'" % (typ1, typ2)

class NotADataHdu(OIException):
    """
    If the hdu provided does not contain data
    """
    def __init__(self, idx, src, *args, **kwargs):
        super(NotADataHdu, self).__init__(idx, src, *args, **kwargs)
        self.message = "Hdu index '%s' in file '%s' does not contain data" % (idx, src)

class NoSystematicsFit(OIException):
    """
    If the user did not set on the fit of systematics
    """
    def __init__(self, *args, **kwargs):
        super(NoSystematicsFit, self).__init__(*args, **kwargs)
        self.message = "You are not fitting systematics"

class NotCallable(OIException):
    """
    If the function is callable
    """
    def __init__(self, fct, *args, **kwargs):
        super(NotCallable, self).__init__(fct, *args, **kwargs)
        self.message = "'%s' should be callable" % fct
