#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

# to prevent plots from popping up
import matplotlib
matplotlib.use('agg')

import os

import sys
testdir = os.path.dirname(__file__)
moduledir = os.path.join(testdir, os.path.pardir)
sys.path.insert(0, moduledir)
from pychebfun import *

import numpy as np
np.seterr(all='raise')
import numpy.testing as npt

import unittest

@np.vectorize
def zero(x):
    return 0.
@np.vectorize
def const_func(val, x):
    return val

xs = np.linspace(0,2*np.pi,1000)

class Test_Foufun_Init(unittest.TestCase):
    def test_const_init(self):
        ff = Foufun(2.71828)
        self.assertEqual(len(ff.x), 1)
        self.assertEqual(len(ff.f), 1)
        npt.assert_equal(ff.f, [2.71828])
        npt.assert_almost_equal(ff(xs), np.array([2.71828]*1000))
        npt.assert_almost_equal(ff.coefficients(), [2.71828])

    def test_array_init(self):
        ff = Foufun([0., 2.])
        self.assertEqual(len(ff.x), 2)
        self.assertEqual(len(ff.f), 2)
        npt.assert_almost_equal(ff.f, 2./np.pi*ff.x)

    def test_coeffs_init(self):
        ff = Foufun(coeffs=[1.15])
        npt.assert_almost_equal(ff(xs), const_func(1.15, xs))

    def test_func_init(self):
        ff = Foufun(np.sin, 32)
        self.assertEqual(len(ff), 32)
        npt.assert_almost_equal(ff.f, np.sin(ff.x))

    def test_len(self):
        ff = Foufun(2.71828)
        self.assertEqual(len(ff), len(ff.coefficients()))

class Test_Foufun_Differentiate(unittest.TestCase):
    def test_differentiate1(self):
        ff = Foufun(lambda x: np.sin(2*x), 256)
        dff = ff.differentiate()
        dffexact = lambda x: 2*np.cos(2*x)
        npt.assert_allclose(dffexact(dff.x), dff.f, atol=1e-12)

    def test_differentiate2(self):
        ff = Foufun(lambda x: np.exp(3*np.sin(2*x)), 256)
        dff = ff.differentiate()
        dffexact = lambda x: 6*np.cos(2*x)*np.exp(3*np.sin(2*x))
        npt.assert_allclose(dffexact(dff.x), dff.f, atol=1e-12)

    def test_differentiate3(self):
        ff = Foufun(lambda x: np.exp(3*np.sin(2*x)), 768,
                    interval=[-np.pi, 7*np.pi])
        dff = ff.differentiate()
        dffexact = lambda x: 6*np.cos(2*x)*np.exp(3*np.sin(2*x))
        npt.assert_allclose(dffexact(dff.x), dff.f, atol=1e-12)

    def test_differentiate4(self):
        Bc = 5.15
        ff = Foufun(lambda x: np.exp(-0.5*Bc*x*x), 150,
                    interval=[-15., 5.])
        dff = ff.differentiate()
        dffexact = lambda x: -Bc*x*np.exp(-0.5*Bc*x*x)
        npt.assert_allclose(dffexact(dff.x), dff.f, atol=1e-12)

    def test_differentiate5(self):
        Bc = 5.15
        ff = Foufun(lambda x: np.exp(-0.5*Bc*x*x), 151,
                    interval=[-15., 5.])
        dff = ff.differentiate()
        dffexact = lambda x: -Bc*x*np.exp(-0.5*Bc*x*x)
        npt.assert_allclose(dffexact(dff.x), dff.f, atol=1e-12)

class Test_Foufun_Arithmetic(unittest.TestCase):

    def test_add_scalar(self):
        Bc = 37.
        ff = Foufun(lambda x: np.exp(-0.5*Bc*x*x), 100,
                    interval=[-15., 5.])
        res = ff + 3.141
        resexact = lambda x: np.exp(-0.5*Bc*x*x)+3.141
        npt.assert_allclose(resexact(ff.x), res.f)

    def test_add(self):
        Bc = 10.
        ff = Foufun(lambda x: np.exp(-0.5*Bc*x*x), 100,
                    interval=[-15., 5.])
        res = ff + Foufun(np.sin, 100, interval=[-15., 5.])
        resexact = lambda x: np.exp(-0.5*Bc*x*x)+np.sin(x)
        npt.assert_allclose(resexact(ff.x), res.f)


if __name__ == '__main__':
    unittest.main()


