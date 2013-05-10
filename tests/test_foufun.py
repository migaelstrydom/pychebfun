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

xs = np.linspace(-np.pi,np.pi,1000)

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
        npt.assert_almost_equal(ff(xs), 2./np.pi*xs+2.)
        
    def test_coeffs_init(self):
        ff = Foufun(coeffs=[1.15])
        npt.assert_almost_equal(ff(xs), const_func(1.15, xs))

    def test_len(self):
        ff = Foufun(2.71828)
        self.assertEqual(len(ff), len(ff.coefficients()))

if __name__ == '__main__':
    unittest.main()


