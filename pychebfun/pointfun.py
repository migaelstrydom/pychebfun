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

    def __call__(self, x):
        return self.p(x)

    def __len__(self):
        return len(self.x)

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
