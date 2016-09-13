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
        print("\033[31m{}\033[39m".format(exc.message))
        return True
    return False


class OIException(Exception):
    """
    Root for SOIF Exceptions, only used to trigger any soif
    errors, never raised
    """
    def __init__(self, *args, **kwargs):
        self.args = [a for a in args] + [a for a in kwargs.values()]

    def __repr__(self):
        return repr(self.message)

    __str__ = __repr__


class NoTargetTable(OIException):
    """
    If the file has no OITARGET table
    """
    def __init__(self, src="", *args, **kwargs):
        super(NoTargetTable, self).__init__(src, *args, **kwargs)
        self.message = "There's no OI_TARGET table in the oifits file '{}'! Oups".format(src)


class NoWavelengthTable(OIException):
    """
    If the file has no OITARGET table
    """
    def __init__(self, src="", *args, **kwargs):
        super(NoWavelengthTable, self).__init__(src, *args, **kwargs)
        self.message = "There's no OI_WAVELENGTH table in the oifits file '{}'! Haha, you're screwed!".format(src)


class ReadOnly(OIException):
    """
    If the parameter is read-only
    """
    def __init__(self, attr, *args, **kwargs):
        super(ReadOnly, self).__init__(attr, *args, **kwargs)
        self.message = "Attribute '{}' is read-only. What did you think?".format(attr)


class InvalidDataType(OIException):
    """
    If the data type provided does not exist
    """
    def __init__(self, datatype, *args, **kwargs):
        super(InvalidDataType, self).__init__(datatype, *args, **kwargs)
        self.message = "Surprise! Data type '{}' does not exist".format(datatype)


class HduDatatypeMismatch(OIException):
    """
    If the data type and the hdu provided do not match
    """
    def __init__(self, hduhead, datatype, *args, **kwargs):
        super(HduDatatypeMismatch, self).__init__(hduhead, datatype,
                                                  *args, **kwargs)
        self.message = "Data type '{}' and hdu with '{}' data do not match. Don't mix peas with poos".format(datatype, hduhead)


class BadMaskShape(OIException):
    """
    If the mask shape does not match the data shape
    """
    def __init__(self, shape, *args, **kwargs):
        super(BadMaskShape, self).__init__(shape, *args, **kwargs)
        self.message = "Bad mask shape; it should be '{}'. Nudge it and try again".format(shape)


class WrongData(OIException):
    """
    If the data provided has the wrong data type
    """
    def __init__(self, typ, *args, **kwargs):
        super(WrongData, self).__init__(typ, *args, **kwargs)
        self.message = "Wrong data given, should be '{}'. Already told you, don't mix apples and camembert".format(typ)


class IncompatibleData(OIException):
    """
    If the data type and the hdu provided do not match
    """
    def __init__(self, typ1, typ2, *args, **kwargs):
        super(IncompatibleData, self).__init__(typ1, typ2, *args, **kwargs)
        self.message = "Can't merge '{}' and '{}'. You looked very optimistic until now".format(typ1, typ2)


class NotADataHdu(OIException):
    """
    If the hdu provided does not contain data
    """
    def __init__(self, idx, src, *args, **kwargs):
        super(NotADataHdu, self).__init__(idx, src, *args, **kwargs)
        self.message = "Hdu index '{}' in file '{}' does not contain data. Yep. Too bad".format(idx, src)


class NoSystematicsFit(OIException):
    """
    If the user did not set on the fit of systematics
    """
    def __init__(self, *args, **kwargs):
        super(NoSystematicsFit, self).__init__(*args, **kwargs)
        self.message = "You are not fitting systematics. But maybe you should"


class NotCallable(OIException):
    """
    If the function is callable
    """
    def __init__(self, fct, *args, **kwargs):
        super(NotCallable, self).__init__(fct, *args, **kwargs)
        self.message = "'{}' should be callable. Get up and go get it".format(fct)


class NoEMCEE(OIException):
    """
    EMCEE is not installed
    """
    def __init__(self, *args, **kwargs):
        super(NoEMCEE, self).__init__(*args, **kwargs)
        self.message = "EMCEE is not installed. How come it is not?"


class AllParamsSet(OIException):
    """
    When all parameters are set - nothing to run EMCEE on
    """
    def __init__(self, *args, **kwargs):
        super(AllParamsSet, self).__init__(*args, **kwargs)
        self.message = "All parameters are set, there is no reason for MCMC. That, was a successful simulation!"


class InvalidBound(OIException):
    """
    Unitary model with invalid bounds
    """
    def __init__(self, name, param, vv, *args, **kwargs):
        super(InvalidBound, self).__init__(name, param, vv, *args, **kwargs)
        self.message = "Object '{}', parameter '{}' has invalid bounds: {}.".format(name, param, vv)


class NotFoundName(OIException):
    """
    Did not find the unitary model from its name
    """
    def __init__(self, name, *args, **kwargs):
        super(NotFoundName, self).__init__(name, *args, **kwargs)
        self.message = "Object name '{}' not found.".format(name)


class InalidUnitaryModel(OIException):
    """
    Did not find unitary model
    """
    def __init__(self, typ, *args, **kwargs):
        super(InalidUnitaryModel, self).__init__(typ, *args, **kwargs)
        self.message = "Unitary model '{}' does not exist.".format(typ)


class BusyName(OIException):
    """
    The unitary model name already exists
    """
    def __init__(self, name, *args, **kwargs):
        super(BusyName, self).__init__(name, *args, **kwargs)
        self.message = "Object name '{}' already exists.".format(name)


class BadParamsSize(OIException):
    """
    The input parameter 'params' has the wrong size
    """
    def __init__(self, size, *args, **kwargs):
        super(BadParamsSize, self).__init__(size, *args, **kwargs)
        self.message = "'params' size mismatch, should be '{:d}'".format(size)


class MasperpxMismatch(OIException):
    """
    The unitary model name already exists
    """
    def __init__(self, mpp1, mpp2, *args, **kwargs):
        super(MasperpxMismatch, self).__init__(mpp1, mpp2, *args, **kwargs)
        self.message = "Cannot rescale image, masperpx mismatch '{:.3f}' and '{:.3f}'".format(mpp1, mpp2)


class NoDataModel(OIException):
    """
    If there is no Oidata provided with the model and one tries to do
    stuff that is possible only with Oidata
    """
    def __init__(self, *args, **kwargs):
        super(NoDataModel, self).__init__(*args, **kwargs)
        self.message = "There is no data in this model. You cannot do that you unless specifically authorized"


class ZeroErrorbars(OIException):
    """
    If there is some data errors are zero
    """
    def __init__(self, *args, **kwargs):
        super(ZeroErrorbars, self).__init__(*args, **kwargs)
        self.message = "Some data-errors are 0. That's gonna be an issue very soon"
