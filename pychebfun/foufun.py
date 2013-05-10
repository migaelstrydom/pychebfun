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

    def __init__(self, f=None, N=0, coeffs=None, interval=[-np.pi, np.pi]):
        """
        Create a Fourier series approximation of the function $f$ on
        the interval :math:`[-pi, pi]`. 

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

        try:
            i = iter(f) # interpolation values provided
        except TypeError:
            pass
        else:
            vals = np.array(f)

            N = len(vals)

            self.ai = self.get_coeffs_from_array(vals)

            self.f = vals.copy()
            self.x = self.interpolation_points(N, self.interval)

            self.p  = Bary(self.x, self.f)

            return None

        if isinstance(f, Foufun): # copy if f is another Chebfun
            self.ai = f.ai.copy()
            self.x = f.x
            self.f = f.f
            self.p = f.p
            self.interval = f.interval

        if coeffs is not None: # if the coefficients of a Chebfun are given

            N = len(coeffs)
            self.ai = np.array(coeffs)
            self.x = self.interpolation_points(N, self.interval)

            if N == 1:
                self.f = np.array([self.ai[0]])
            else:
                self.f = np.fft.irfft(self.ai, len(self.ai))
            self.p = Bary(self.x, self.f)

        else: # if the coefficients of a Foufun are not given
            if not N: # N is not provided
                # Find the right number of coefficients to keep
                coeffs = \
                    self.get_optimal_coefficients(f,
                                                  pow(2, self.max_nb_dichotomy),
                                                  self.interval)
                N = len(coeffs)-1

            else:
                nextpow2 = int(np.log2(N))+1
                coeffs = self.cheb_poly_fit_function(f, pow(2, nextpow2),
                                                     self.interval)

            self.ai = coeffs[:N+1]
            if N == 0:
                self.x = self.interpolation_points(1, self.interval)
            else:
                self.x  = self.interpolation_points(N, self.interval)
            self.f  = f(self.x)
            self.p  = Bary(self.x, self.f)

    @classmethod
    def interpolation_points(self, N, interval):
        """
        Returns N+1 evenly spaced points on the interval.
        """
        return np.linspace(interval[0], interval[1], N+1)[:N]
        

    def get_coeffs_from_array(self, f):
        if len(f) == 1:
            return np.array(f)

        return np.fft.rfft(f)

    def get_coeffs_from_function(self, f, N, interval):
        return self.get_coeffs_from_array(f(
                self.interpolation_points(N, interval)))

    #@classmethod  # Can't use this because of self.record
    def get_optimal_coefficients(self, f, maxN, interval):
        N = 2
        for k in xrange(2, self.max_nb_dichotomy):
            N = N*2

            coeffs = self.get_coeffs_from_function(f, N, interval)
            abs_max_coeff = np.max(np.abs(coeffs))
            # Special case: check for the zero function
            if abs_max_coeff < 2*emach:
                return np.array([0.])
            # Check for negligible coefficients
            # If within bound: get negligible coeffs and break
            bnd = 128*emach*abs_max_coeff
            if self.record:
                self.bnds.append(bnd)
                self.intermediate.append(coeffs)

            last = abs(coeffs[-2:])
            print 'coeffs:', coeffs
            if np.all(last <= bnd) or N >= maxN:
                break

        # End of convergence loop: construct polynomial
        [inds]  = np.where(abs(coeffs) >= bnd)
        N = min(inds[-1], maxN)

        if self.record:
            self.bnds.append(bnd)
            self.intermediate.append(coeffs)

        if N == pow(2, self.max_nb_dichotomy-1):
            raise self.NoConvergence(last, bnd)

        return coeffs[:N+1]
