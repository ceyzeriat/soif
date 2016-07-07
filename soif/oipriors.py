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


def lnuniform(x, prior_lninvrange=0, *args, **kwargs):
    """
    Returns the log-probability of drawing any value in the range of the prior
    """
    if np.size(x)==1: return prior_lninvrange
    return np.ones(np.shape(x))*prior_lninvrange

def uniform(x, prior_invrange=0, *args, **kwargs):
    """
    Returns the probability of drawing any value in the range of the prior
    """
    if np.size(x)==1: return prior_invrange
    return np.ones(np.shape(x))*prior_invrange


def lnnormal(x, x0=0., sigma=1., *args, **kwargs):
    """
    Returns the probability of drawing x
    """
    return -0.5*((x-x0)/sigma)**2

def normal(x, x0=0., sigma=1., *args, **kwargs):
    """
    Returns the probability of drawing x
    """
    return np.exp(-0.5*((x-x0)/sigma)**2)


def lnnormalbumpy(x, x0=0., rangeshrink=10, *args, **kwargs):
    """
    Returns the probability of drawing x
    """
    return np.exp(-0.5*((x-x0)*pctrange*kwargs['prior_invrange'])**2)

def normalbumpy(x, x0=0., rangeshrink=10, *args, **kwargs):
    """
    Returns the probability of drawing x
    """
    return np.exp(np.exp(-0.5*((x-x0)*pctrange*kwargs['prior_invrange'])**2))


def lntriangle(x, x0=0, slope=4, *args, **kwargs):
    """
    Returns the log-probability of drawing any value in the range of the prior
    """
    return -slope*np.abs(x-x0)*kwargs['prior_invrange']

def triangle(x, x0=0, slope=4, *args, **kwargs):
    """
    Returns the probability of drawing any value in the range of the prior
    """
    return np.exp(-slope*np.abs(x-x0)*kwargs['prior_invrange'])
