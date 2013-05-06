#!/usr/bin/env python
"""
Chebfun module
==============

.. moduleauthor :: Chris Swierczewski <cswiercz@gmail.com>
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>
.. moduleauthor :: Gregory Potter <ghpotter@gmail.com>


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
    Used to cast scalar to Chebfuns
    """
    @wraps(method)
    def new_method(self, other):
        if np.isscalar(other):
            other = Chebfun([float(other)], interval=self.interval)
        return method(self, other)
    return new_method

emach     = sys.float_info.epsilon                        # machine epsilon

class Chebfun(object):
    """
    Construct a Lagrange interpolating polynomial over the Chebyshev points.

    For a Chebfun object, the following properties are always defined:
    numpy.array x: The Chebyshev points on which the Chebfun is defined.
    numpy.array f: The values of the Chebfun at points x.
    numpy.array ai: The Chebyshev polynomial coefficients of the function
                    when rescaled to be defined on [-1, 1]
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

    def __init__(self, f=None, N=0, chebcoeff=None, interval=[-1.,1.]):
        """
        Create a Chebyshev polynomial approximation of the function $f$ on
        the interval :math:`[-1, 1]`.

        :param callable f: Python, Numpy, or Sage function
        :param int N: (default = None)  specify number of interpolating points
        :param np.array chebcoeff: (default = np.array(0)) specify the
               coefficients of a Chebfun
        :param list interval: The domain on which the Chebfun is defined.
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

            self.ai = self.cheb_poly_fit_array(vals)

            if N == 1:
                self.f = np.array([vals[0], vals[0]])
                self.x = self.interpolation_points(1, self.interval)
            else:
                self.f = vals.copy()
                self.x = self.interpolation_points(N-1, self.interval)

            self.p  = Bary(self.x, self.f)

            return None

        if isinstance(f, Chebfun): # copy if f is another Chebfun
            self.ai = f.ai.copy()
            self.x = f.x
            self.f = f.f
            self.p = f.p

        if chebcoeff is not None: # if the coefficients of a Chebfun are given

            self.N = N = len(chebcoeff) - 1
            self.ai = np.array(chebcoeff)
            if N == 0:
                self.x = self.interpolation_points(1, self.interval)
                self.f = np.array([self.ai[0], self.ai[0]])
            else:
                self.f = idct(self.ai)
                self.x = self.interpolation_points(N, self.interval)
            self.p = Bary(self.x, self.f)

        else: # if the coefficients of a Chebfun are not given
            if not N: # N is not provided
                # Find out the right number of coefficients to keep
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
        N+1 Chebyshev points in [-1, 1], boundaries included
        """
        midp = 0.5*(interval[0]+interval[1])
        return 0.5*(interval[1]-interval[0])* \
            np.cos(np.arange(N+1)*np.pi/N) + midp

    @classmethod
    def intersect_intervals(self, inta, intb):
        intersection = np.array([max(inta[0], intb[0]),
                                 min(inta[1], intb[1])])

        if intersection[0] > intersection[1]:
            return np.zeros(2)

        return intersection

    #@classmethod  # Can't use this because of self.record
    def get_optimal_coefficients(self, f, maxN, interval):
        N = 2
        for k in xrange(2, self.max_nb_dichotomy):
            N = N*2

            coeffs = self.cheb_poly_fit_function(f, N, interval)
            absMaxCoeff = np.max(np.abs(coeffs))
            # Special case: check for the zero function
            if absMaxCoeff < 2*emach:
                return np.array([0.])
            # Check for negligible coefficients
            # If within bound: get negligible coeffs and break
            bnd = 128*emach*absMaxCoeff
            if self.record:
                self.bnds.append(bnd)
                self.intermediate.append(coeffs)

            last = abs(coeffs[-2:])
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

    def sample(self, f, N):
        x = self.interpolation_points(N, self.interval)
        return f(x)

    @classmethod
    def cheb_poly_fit_function(self, f, N, interval):
        """
        Compute Chebyshev coefficients of a function f on N points.
        @return: The the first N Chebyshev coefficients in the expansion of f
        """
        evened = even_data(f(self.interpolation_points(N, interval)))
        coeffs = dct(evened)
        return coeffs

    @classmethod
    def cheb_poly_fit_array(self, f):
        """
        Compute Chebyshev coefficients of a function defined on an array f.
        @return: The first N Chebyshev coefficients in the expansion of f,
                 where N = len(f)
        """
        if len(f) == 1:
            return np.array(f)
        evened = even_data(f)
        coeffs = dct(evened)
        return coeffs

    def __repr__(self):
        return "<Chebfun({0})>".format(len(self))

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
        return not np.allclose(self.chebyshev_coefficients(), 0.)

    def __eq__(self, other):
        if np.isscalar(other):
            other = Chebfun(other)
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

        return Chebfun(None, 0,
                       self.get_optimal_coefficients(
                lambda x: self(x) + other(x),
                max(len(self), len(other)),
                sum_interval),
                       interval=sum_interval)

    __radd__ = __add__


    @cast_scalar
    def __sub__(self, other):
        """
        Chebfun subtraction.
        """
        sum_interval = self.intersect_intervals(self.interval, other.interval)
        if sum_interval[0] == sum_interval[1]:
            sum_interval = self.interval

        return Chebfun(None, 0,
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
        return Chebfun(lambda x: self(x) * other(x),
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
        return Chebfun(lambda x: self(x) / other(x),
                       interval=sum_interval)

    __truediv__ = __div__

    @cast_scalar
    def __rdiv__(self, other):
        sum_interval = self.intersect_intervals(self.interval, other.interval)
        if sum_interval[0] == sum_interval[1]:
            sum_interval = self.interval
        return Chebfun(lambda x: other(x)/self(x),
                       interval=sum_interval)

    __rtruediv__ = __rdiv__

    def __neg__(self):
        """
        Chebfun negation.
        """
        return Chebfun(lambda x: -self(x), interval=self.interval)

    def __pow__(self, other):
        return Chebfun(lambda x: self(x)**other, interval=self.interval)


    def sqrt(self):
        """
        Square root of Chebfun.
        """
        return Chebfun(lambda x: np.sqrt(self(x)), interval=self.interval)

    def __abs__(self):
        """
        Absolute value of Chebfun. (Python)

        (Coerces to NumPy absolute value.)
        """
        return Chebfun(lambda x: np.abs(self(x)),interval=self.interval)

    def abs(self):
        """
        Absolute value of Chebfun. (NumPy)
        """
        return self.__abs__()

    def sin(self):
        """
        Sine of Chebfun
        """
        return Chebfun(lambda x: np.sin(self(x)),interval=self.interval)


    #
    # Numpy / Scipy Operator Overloads
    #

    def chebyshev_coefficients(self):
        return self.ai

    # Not working yet
    def integral(self):
        """
        Evaluate the integral of the Chebfun over the given interval using
        Clenshaw-Curtis quadrature.
        """
        ai2 = self.ai[::2]
        n = len(ai2)
        Tints = 2/(1-(2*np.arange(n))**2)
        val = np.sum(Tints*ai2)

        return val


    def integrate(self):
        """
        Return the Chebfun representing the integral of self over the domain.

        (Simply numerically integrates the underlying Barcentric polynomial.)
        """
        return Chebfun(self.p.integrate)


    def derivative(self):
        return self.differentiate()

    def differentiate(self):
        """
        Return the Chebfun representing the derivative of self. Uses spectral
        methods for accurately constructing the derivative.
        """
        # Compute new ai by doing a backsolve

        # If a_i and b_i are the kth Chebyshev polynomial expansion coefficient
        # Then b_{i-1} = b_{i+1} + 2ia_i; b_N = b_{N+1} = 0; b_0 = b_2/2 + a_1

        N = len(self.ai) - 1
        if N == 0: # Special case: derivative = const
            deriv = (self.f[1]-self.f[0])/(self.x[1]-self.x[0])
            return Chebfun(deriv)

        bi = np.zeros(N+1)
        bi[N-1] = 2*N*self.ai[N]
        for i in xrange(N-2, 0, -1):
            bi[i] = bi[i+2] + 2*(i+1)*self.ai[i+1]

        bi[0] = bi[2]*0.5 + self.ai[1]
        bi = bi*(2./(self.interval[1]-self.interval[0]))

        return Chebfun(self, chebcoeff=bi, interval=self.interval)

    def roots(self):
        """
        Return the roots of the chebfun.
        """
        N = len(self.ai)
        coeffs = np.hstack([self.ai[-1::-1], self.ai[1:]])
        coeffs[N-1] *= 2
        zNq = np.poly1d(coeffs)
        roots = np.array([np.real(r) for r in zNq.roots if np.allclose(abs(r), 1.)])
        return np.unique(roots)

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

    def chebcoeffplot(self, *args, **kwds):
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
        Plots the original function against its chebfun interpolant.

        INPUTS:

            -- f: Python, Numpy, or Sage function
        """
        x   = np.linspace(self.interval[0], self.interval[1], 10000)
        fig = plt.figure()
        ax  = fig.add_subplot(211)

        ax.plot(x, f(x), '#dddddd', linewidth=10, label='Actual', *args, **kwds)
        label = 'Chebfun Interpolant (d={0})'.format(len(self))
        self.plot(color='red', label=label, *args, **kwds)
        ax.legend(loc='best')

        ax  = fig.add_subplot(212)
        ax.plot(x, abs(f(x)-self(x)), 'k')

        return ax

def chebpoly(n):
    if not n:
        return Chebfun(np.array([1.]))
    vals = np.ones(n+1)
    vals[-1::-2] = -1
    if len(vals) % 2 == 1:
        return Chebfun(-vals)
    else:
        return Chebfun(vals)

def even_data(data):
    """
    Construct Extended Data Vector (equivalent to creating an
    even extension of the original function)
    """
    return np.hstack([data, data[-2:0:-1]])

import scipy.fftpack as fftpack

def dct(data):
    """
    Compute DCT
    """
    N = len(data)//2
    dctdata     = fftpack.dct(data[:N+1], 1)
    dctdata     /= N
    dctdata[0]  /= 2.
    dctdata[-1] /= 2.
    return dctdata

def idct(chebcoeff):
    """
    Compute the inverse DCT
    """
    N = len(chebcoeff)

    data = 2.*chebcoeff
    data[0] *= 2
    data[-1] *= 2
    data *= N

    idctdata = fftpack.dct(data, 1)/(4*N)
    return idctdata


