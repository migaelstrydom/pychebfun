#!/usr/bin/env python
"""
Common base class for Chebfun and Foufun.
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

def cast_scalar(method):
    """
    Used to cast scalar to the Genfuns
    """
    @wraps(method)
    def new_method(self, other):
        if np.isscalar(other):
            other = self.__class__([float(other)], interval=self.interval)
        return method(self, other)
    return new_method

emach     = sys.float_info.epsilon                        # machine epsilon

class Pointfun(object):
    """
    Construct a function from points. This serves as the base class for
    Chebfun and Foufun, providing the common operations between the two.

    For a Pointfun object, the following properties are always defined:
    numpy.array x: The points on which the Pointfun is defined. These
        are Chebyshev points for Chebfun or equally spaced points for 
        Foufun.
    numpy.array f: The values of the Pointfun at points x.
    numpy.array ai: The coefficients of the basis functions that make
        up the Pointfun. These are Chebyshev coefficients for Chebfun 
        or Fourier coefficients for Foufun.
    BarycentricInterpolator p: A polynomial interpolation through points
                               f on domain x.
    """
    max_nb_dichotomy = 12 # maximum number of dichotomy of the interval
    plot_res = 1000 # Number of points to use when plotting
    record = False # whether to record convergence information

    class NoConvergence(Exception):
        """
        Raised when dichotomy does not converge.
        """

    @classmethod
    def intersect_intervals(self, inta, intb):
        intersection = np.array([max(inta[0], intb[0]),
                                 min(inta[1], intb[1])])

        if intersection[0] > intersection[1]:
            return np.zeros(2)

        return intersection

    def coefficients(self):
        """
        Return the coefficients of the function.
        """
        return self.ai

    def __repr__(self):
        return "<Pointfun({0})>".format(len(self))

    #
    # Basic Operator Overloads
    #
    def __call__(self, x):
        return self.p(x)

    def __len__(self):
        return self.p.n

    def __nonzero__(self):
        """
        Test for difference from zero (up to tolerance)
        """
        return not np.allclose(self.coefficients(), 0.)

    @cast_scalar
    def __eq__(self, other):
        return np.allclose(self.interval, other.interval) and \
                not(self - other)

    def __neq__(self, other):
        return not (self == other)

    @cast_scalar
    def __add__(self, other):
        """
        Addition
        """
        sum_interval = self.intersect_intervals(self.interval, other.interval)
        if sum_interval[0] == sum_interval[1]:
            sum_interval = self.interval

        return self.__class__(None, 0,
                       self.get_optimal_coefficients(
                lambda x: self(x) + other(x),
                max(len(self), len(other)),
                sum_interval),
                       interval=sum_interval)

    __radd__ = __add__


    @cast_scalar
    def __sub__(self, other):
        """
        Subtraction.
        """
        sum_interval = self.intersect_intervals(self.interval, other.interval)
        if sum_interval[0] == sum_interval[1]:
            sum_interval = self.interval

        return self.__class__(None, 0,
                       self.get_optimal_coefficients(
                lambda x: self(x) - other(x),
                max(len(self), len(other)),
                sum_interval),
                       interval=sum_interval)

    def __rsub__(self, other):
        return -(self - other)


    @cast_scalar
    def __mul__(self, other):
        """
        Chebfun multiplication.
        """
        sum_interval = self.intersect_intervals(self.interval, other.interval)
        if sum_interval[0] == sum_interval[1]:
            sum_interval = self.interval
        return self.__class__(lambda x: self(x) * other(x),
                       interval=sum_interval)

    __rmul__ = __mul__

    @cast_scalar
    def __div__(self, other):
        """
        Chebfun division
        """
        sum_interval = self.intersect_intervals(self.interval, other.interval)
        if sum_interval[0] == sum_interval[1]:
            sum_interval = self.interval
        return self.__class__(lambda x: self(x) / other(x),
                       interval=sum_interval)

    __truediv__ = __div__

    @cast_scalar
    def __rdiv__(self, other):
        sum_interval = self.intersect_intervals(self.interval, other.interval)
        if sum_interval[0] == sum_interval[1]:
            sum_interval = self.interval
        return self.__class__(lambda x: other(x)/self(x),
                       interval=sum_interval)

    __rtruediv__ = __rdiv__

    def __neg__(self):
        """
        Negation.
        """
        return self.__class__(lambda x: -self(x), interval=self.interval)

    def __pow__(self, other):
        return self.__class__(lambda x: self(x)**other, interval=self.interval)


    def sqrt(self):
        """
        Square root of Pointfun.
        """
        return self.__class__(lambda x: np.sqrt(self(x)), 
                              interval=self.interval)

    def __abs__(self):
        """
        Absolute value of Pointfun. (Python)

        (Coerces to NumPy absolute value.)
        """
        return self.__class__(lambda x: np.abs(self(x)),
                              interval=self.interval)

    def abs(self):
        """
        Absolute value of Pointfun. (NumPy)
        """
        return self.__abs__()

    def sin(self):
        """
        Sine of Pointfun
        """
        return self.__class__(lambda x: np.sin(self(x)),interval=self.interval)

    #
    # Numpy / Scipy Operator Overloads
    #

    def plot(self, interpolation_points=True, *args, **kwargs):
        xs = np.linspace(self.interval[0], self.interval[1],
                         self.plot_res)

        axis = plt.gca()
        axis.plot(xs, self(xs), *args, **kwargs)
        if interpolation_points:
            # figure out current colour
            current_color = axis.lines[-1].get_color()
            axis.plot(self.x, self.f,
                      marker='.', linestyle='', color=current_color)
        plt.plot()

    def coefficient_plot(self, *args, **kwds):
        """
        Plot the coefficients.
        """
        fig = plt.figure()
        ax  = fig.add_subplot(111)

        data = np.log10(np.abs(self.ai))
        ax.plot(data, 'r' , *args, **kwds)
        ax.plot(data, 'r.', *args, **kwds)

        return ax

    def plot_interpolating_points(self):
        plt.plot(self.x, self.f)

    def compare(self, f, *args, **kwds):
        """
        Plots the original function against its interpolant.

        INPUTS:

            -- f: Python, Numpy, or Sage function
        """
        x   = np.linspace(self.interval[0], self.interval[1], 10000)
        fig = plt.figure()
        ax  = fig.add_subplot(211)

        ax.plot(x, f(x), '#dddddd', linewidth=10, label='Actual', *args, **kwds)
        label = 'Interpolant (d={0})'.format(len(self))
        self.plot(color='red', label=label, *args, **kwds)
        ax.legend(loc='best')

        ax  = fig.add_subplot(212)
        ax.plot(x, abs(f(x)-self(x)), 'k')

        return ax
