#!/usr/bin/env python
"""
Foufun module
==============

.. moduleauthor :: Chris Swierczewski <cswiercz@gmail.com>
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>
.. moduleauthor :: Gregory Potter <ghpotter@gmail.com>
.. moduleauthor :: Migael Strydom <migael.strydom+git@gmail.com>

"""
# coding: UTF-8
from __future__ import division

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import sys
from functools import wraps

from scipy.interpolate import BarycentricInterpolator as Bary

from pointfun import *

class Foufun(Pointfun):
    """
    Represent a function as equally spaced points and then use Fourier
    methods to operate on it.

    For a Foufun object, the following properties are always defined:
    numpy.array x: The points on which the Foufun is defined. 
    numpy.array f: The values of the Foufun at points x.
    numpy.array ai: The Fourier coefficients of the Foufun.
    BarycentricInterpolator p: A polynomial interpolation through points
                               f on domain x.
    """
