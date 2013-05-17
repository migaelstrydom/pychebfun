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

# Use linear interpolation for speed.
from scipy.interpolate import interp1d as FInterp

from pointfun import *

def cast_scalar(method):
    """
    Used to cast scalar to the Pointfuns
    """
    @wraps(method)
    def new_method(self, other):
        if np.isscalar(other):
            other = self.__class__([float(other)]*len(self), 
                                   interval=self.interval)
        return method(self, other)
    return new_method

class Foufun(Pointfun):
    """
    Represent a periodic function as equally spaced points and then use Fourier
    methods to operate on it.

    For a Foufun object, the following properties are always defined:
    numpy.array x: The points on which the Foufun is defined. 
    numpy.array f: The values of the Foufun at points x. N points are stored,
       and f(x[N+1]) == f(x[0]). 
    numpy.array interval: The interval on which the function is periodic,
       f(interval[0]) == f(interval[1]).
    numpy.array ai: The Fourier coefficients of the Foufun.
    BarycentricInterpolator p: A polynomial interpolation through points
                               f on domain x.
    """

    class UnmatchingDimensions(Exception):
        """
        Raised when performing operations on two Foufuns that are
        not defined on the same interval or have differing numbers 
        of points.
        """

    def __init__(self, f=None, N=0, coeffs=None, interval=[0., 2*np.pi]):
        """
        Create a Fourier series approximation of the function $f$ on
        the interval :math:`[0, 2*pi]`. 

        :param callable f: Python, Numpy, or Sage function
        :param int N: (default = None) specify number of interpolating points
               If none are specified, the optimal number will be determined.
        :param np.array coeffs: (default = None) specify the Fourier
               coefficients
        :param list interval: The domain on which the function is defined.
        """
        if self.record:
            self.intermediate = []
            self.bnds = []

        if np.isscalar(f):
            f = [f]

        self.interval = np.array(interval)

        f_is_list = False

        # Check if f is iterable
        try:
            i = iter(f) # interpolation values provided
        except TypeError:
            pass
        else:
            f_is_list = True

        if f_is_list:
            vals = np.array(f)

            N = len(vals)

            self.ai = self.get_coeffs_from_array(vals)

            self.f = vals.copy()
            self.x = self.interpolation_points(N, self.interval)

        elif isinstance(f, Foufun): # copy if f is another Chebfun
            self.ai = f.ai.copy()
            self.x = f.x
            self.f = f.f
            self.p = f.p
            self.interval = f.interval
            return None

        elif coeffs is not None: # if the coefficients of a Chebfun are given

            N = 2*(len(coeffs)-1)
            self.ai = np.array(coeffs).astype(complex)

            if N == 0:
                self.f = np.array([self.ai[0]])
                self.x = np.array([self.interval[0]])
            else:
                self.f = np.fft.irfft(self.ai)
                self.x = self.interpolation_points(N, self.interval)

        else: # f must be a function
            if not N: # N is not provided
                # Choose some value
                N = 16

            self.x  = self.interpolation_points(N, self.interval)
            self.f  = f(self.x)
            self.ai = self.get_coeffs_from_array(self.f)

        self.p  = FInterp(np.concatenate((self.x, [self.interval[1]])), 
                       np.concatenate((self.f,[self.f[0]])))

    @classmethod
    def interpolation_points(self, N, interval):
        """
        Returns N+1 evenly spaced points on the interval.
        """
        return np.linspace(interval[0], interval[1], N+1)[:N]
        

    def get_coeffs_from_array(self, f):
        if len(f) == 1:
            return np.array(f).astype(complex)

        return np.fft.rfft(f)

    def get_coeffs_from_function(self, f, N, interval):
        return self.get_coeffs_from_array(f(
                self.interpolation_points(N, interval)))

    @cast_scalar
    def __add__(self, other):
        """
        Addition
        """
        if not np.allclose(self.interval, other.interval) or \
                len(self) != len(other):
            raise self.UnmatchingDimensions(self.interval, other.interval, 
                                            len(self), len(other))

        return Foufun(self.f+other.f, interval=self.interval)

    __radd__ = __add__


    @cast_scalar
    def __sub__(self, other):
        """
        Subtraction.
        """
        if not np.allclose(self.interval, other.interval) or \
                len(self) != len(other):
            raise self.UnmatchingDimensions(self.interval, other.interval, 
                                            len(self), len(other))

        return Foufun(self.f-other.f, interval=self.interval)

    def __rsub__(self, other):
        return -(self - other)


    @cast_scalar
    def __mul__(self, other):
        """
        Chebfun multiplication.
        """
        if not np.allclose(self.interval, other.interval) or \
                len(self) != len(other):
            raise self.UnmatchingDimensions(self.interval, other.interval, 
                                            len(self), len(other))

        return Foufun(self.f*other.f, interval=self.interval)

    __rmul__ = __mul__

    @cast_scalar
    def __div__(self, other):
        """
        Chebfun division
        """
        if not np.allclose(self.interval, other.interval) or \
                len(self) != len(other):
            raise self.UnmatchingDimensions(self.interval, other.interval, 
                                            len(self), len(other))

        return Foufun(self.f/other.f, interval=self.interval)

    __truediv__ = __div__

    @cast_scalar
    def __rdiv__(self, other):
        if not np.allclose(self.interval, other.interval) or \
                len(self) != len(other):
            raise self.UnmatchingDimensions(self.interval, other.interval, 
                                            len(self), len(other))

        return Foufun(other.f/self.f, interval=self.interval)

    __rtruediv__ = __rdiv__

    def __neg__(self):
        """
        Negation.
        """
        return self.__class__(-self.f, interval=self.interval)

    def __pow__(self, other):
        return self.__class__(self.f**other, interval=self.interval)


    def sqrt(self):
        """
        Square root of Pointfun.
        """
        return self.__class__(np.sqrt(self.f), 
                              interval=self.interval)


    def differentiate(self):
        new_coeffs = self.ai.copy()
        scale = 2*np.pi/(self.interval[1]-self.interval[0])
        for k in xrange(len(self.ai)-1):
            new_coeffs[k] = complex(0,k)*new_coeffs[k]*scale
        new_coeffs[-1] = 0.

        return Foufun(coeffs=new_coeffs, interval=self.interval)

    def compare(self, f, *args, **kwds):
        """
        Plots the give function f against the current Foufun interpolant.
        Also plots the errors in the points stored compared to f.

        INPUTS:

            -- f: Python, Numpy, or Sage function
        """
        fig = plt.figure()
        ax  = fig.add_subplot(211)

        ax.plot(self.x, f(self.x), 
                '#dddddd', linewidth=10, label='Actual', *args, **kwds)
        label = 'Interpolant (d={0})'.format(len(self))
        self.plot(color='red', label=label, *args, **kwds)
        ax.legend(loc='best')

        ax  = fig.add_subplot(212)
        ax.plot(self.x, abs(f(self.x)-self.f), 'k')

        return ax
