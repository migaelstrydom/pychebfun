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


def f(x):
    return np.sin(6*x) + np.sin(30*np.exp(x))


def fd(x):
    """
    Derivative of f
    """
    return 6*np.cos(6*x) + np.cos(30*np.exp(x))*30*np.exp(x)

def piecewise_continuous(x):
    """
    The function is on the verge of being discontinuous at many points
    """
    return np.exp(x)*np.sin(3*x)*np.tanh(5*np.cos(30*x))

def gaussian_function(Bc):
    """
    Gaussian function.
    """
    return Chebfun(lambda x : np.exp(-0.5*Bc*x*x))

def problem_functions(N, Bc):
    """
    Returns two functions such that when the second is subtracted from
    the first, Chebfun does not converge.
    """
    ustart = 0.99999

    H = Chebfun([Bc*0.25])*gaussian_function(Bc)
    U = gaussian_function(Bc)

    return ((Bc*ustart)*U), ((1.+ 3.*ustart**4)*H)

def runge(x):
    return 1./(1+25*x**2)

@np.vectorize
def zero(x):
    return 0.
@np.vectorize
def const_func(val, x):
    return val

xs = np.linspace(-1,1,1000)

class Test_Chebfun(unittest.TestCase):
    def setUp(self):
        # Constuct the O(dx^-16) "spectrally accurate" chebfun p
        Chebfun.record = True
        self.p = Chebfun(f,)

    def test_initialise(self):
        # Init with array
        self.assertEqual(len(Chebfun(1.).x), 2)
        self.assertEqual(len(Chebfun(1.).f), 2)
        self.assertEqual(len(Chebfun([1.]).x), 2)
        self.assertEqual(len(Chebfun([1.]).f), 2)
        self.assertEqual(len(Chebfun([1., 2.]).x), 2)
        self.assertEqual(len(Chebfun([1., 2.]).f), 2)
        self.assertEqual(len(Chebfun([1., 2., 3.]).x), 3)
        self.assertEqual(len(Chebfun([1., 2., 3.]).f), 3)
        # Init with function
        cfrung = Chebfun(runge)
        self.assertEqual(len(cfrung.x), len(cfrung.f))
        cfrung = Chebfun(cfrung)
        self.assertEqual(len(cfrung.x), len(cfrung.f))
        # Init with chebyshev polynomial coefficients
        cfcoeff = Chebfun(None, 0, [1., 0.])
        self.assertEqual(len(cfcoeff.x), len(cfcoeff.f))

    def test_constant(self):
        self.assertEqual(Chebfun(1.), Chebfun([1.]))
        self.assertEqual(Chebfun(1.), Chebfun([1., 1.]))
        self.assertEqual(Chebfun(1.), Chebfun([1., 1., 1.]))
        self.assertEqual(Chebfun(1.), Chebfun([1., 1., 1., 1.]))
        self.assertEqual(Chebfun(1.), Chebfun([1., 1., 1., 1., 1.]))
        self.assertEqual(Chebfun(1.), Chebfun([1.]*137))

    def test_len(self):
        self.assertEqual(len(self.p), len(self.p.chebyshev_coefficients()))

    def test_error(self):
        x = xs
        err = abs(f(x)-self.p(x))
        npt.assert_array_almost_equal(self.p(x),f(x),decimal=13)

    def test_root(self):
        roots = self.p.roots()
        npt.assert_array_almost_equal(f(roots),0)

    def test_all_roots(self):
        roots = self.p.roots()
        self.assertEqual(len(roots),22)

    def test_plot(self):
        self.p.plot()

    def test_plot_interpolation_points(self):
        plt.clf()
        self.p.plot()
        a = plt.gca()
        self.assertEqual(len(a.lines),2)
        plt.clf()
        self.p.plot(interpolation_points=False)
        a = plt.gca()
        self.assertEqual(len(a.lines),1)

    def test_chebcoeff(self):
        new = Chebfun(chebcoeff=self.p.ai)
        npt.assert_allclose(self.p(xs), new(xs))

    def test_cheb_plot(self):
        self.p.compare(f)

    def test_chebcoeffplot(self):
        self.p.chebcoeffplot()

    def test_prod(self):
        pp = self.p*self.p
        npt.assert_array_almost_equal(self.p(xs)*self.p(xs),pp(xs))

    def test_square(self):
        def square(x):
            return self.p(x)*self.p(x)
        sq = Chebfun(square)
        npt.assert_array_less(0, sq(xs))
        self.sq = sq

    def test_chebyshev_points(self):
        N = pow(2,5)
        pts = self.p.interpolation_points(N, [-1., 1.])
        npt.assert_array_almost_equal(pts[[0,-1]],np.array([1.,-1]))

    def test_N(self):
        N = len(self.p) - 1
        pN = Chebfun(f, N)
        self.assertEqual(len(pN.chebyshev_coefficients()), N+1)
        self.assertEqual(len(pN.chebyshev_coefficients()),len(pN))
        npt.assert_array_almost_equal(pN(xs), self.p(xs))
        npt.assert_array_almost_equal(pN.chebyshev_coefficients(),self.p.chebyshev_coefficients())

    def test_record(self):
        p = Chebfun(f)
        self.assertEqual(len(p.bnds), 7)

    def test_zero(self):
        p = Chebfun(zero)
        self.assertEqual(len(p),2)

    def test_nonzero(self):
        self.assertTrue(self.p)
        mp = Chebfun(zero)
        self.assertFalse(mp)

    def test_integral(self):
        def q(x):
            return x*x
        p = Chebfun(q)
        i = p.integral()
        self.assertAlmostEqual(i,2/3)

    @unittest.expectedFailure
    def test_integrate(self):
        q = self.p.integrate()

    def test_differentiate(self):
        computed = self.p.differentiate()
        expected = Chebfun(fd)
        npt.assert_allclose(computed(xs), expected(xs),)

        npt.assert_allclose(
            Chebfun(lambda x: const_func(-3.141, x)).differentiate()(xs),
            Chebfun(0.)(xs))
        for n in xrange(2, 11):
            cf = Chebfun(lambda x: x**n)
            self.assertEqual(np.max(np.abs(
                Chebfun(lambda x: n*x**(n-1))(xs)-cf.differentiate()(xs))) <
                128*emach,
                True)
        npt.assert_allclose(
            Chebfun(lambda x: 3.141*x*x).differentiate()(xs),
            Chebfun(lambda x: 6.282*x)(xs))
        npt.assert_allclose(
            Chebfun(np.sin).differentiate()(xs),
            Chebfun(np.cos)(xs))
        npt.assert_allclose(
            Chebfun(lambda x: np.exp(7.*np.sin(3.*x))).differentiate()(xs),
            Chebfun(lambda x: 21.*np.cos(3.*x)*np.exp(7.*np.sin(3.*x)))(xs),
            1e-07, 1e-07)
        npt.assert_allclose(
            Chebfun(lambda x: np.exp(-0.5*x*x)).differentiate()(xs),
            Chebfun(lambda x: -x*np.exp(-0.5*x*x))(xs))

    def test_interp_values(self):
        """
        Instanciate Chebfun from interpolation values.
        """
        p2 = Chebfun(self.p.f)
        npt.assert_almost_equal(self.p.ai, p2.ai)
        npt.assert_array_almost_equal(self.p(xs), p2(xs))

    def test_equal(self):
        self.assertEqual(self.p, Chebfun(self.p))


class Test_Misc(unittest.TestCase):
    def test_truncate(self, N=17):
        """
        Check that the Chebyshev coefficients are properly truncated.
        """
        small = Chebfun(f, N=N)
        new = Chebfun(small)
        self.assertEqual(len(new), len(small),)

    def test_error(self):
        chebpolyplot(f)

    def test_vectorized(self):
        fv = np.vectorize(f)
        p = Chebfun(fv)

    def test_examples(self):
        """
        Check that the examples can be executed.
        """
        here = os.path.dirname(__file__)
        example_folder = os.path.join(here,os.path.pardir,'examples')
        files = os.listdir(example_folder)
        for example in files:
            file_name = os.path.join(example_folder,example)
            try:
                execfile(file_name, {})
            except Exception as e:
                raise Exception('Error in {0}: {0}'.format(example), e)

    def test_chebpoly(self, ns=[1,2,3,4,5,6,7,8,9,10]):
        """
        Check that chebpoly really returns the chebyshev polynomials.
        """
        for n in ns:
            c = chebpoly(n)
            npt.assert_array_almost_equal(c.chebyshev_coefficients(), [0]*n+[1.])

    def test_list_init(self):
        c = Chebfun([1.])
        npt.assert_array_almost_equal(c.chebyshev_coefficients(),[1.])

    def test_scalar_init(self):
        one = Chebfun(1.)
        npt.assert_array_almost_equal(one(xs), 1.)

    def test_no_convergence(self):
        with self.assertRaises(Chebfun.NoConvergence):
            Chebfun(np.sign)

    def test_runge(self):
        """
        Test some of the capabilities of operator overloading.
        """
        r = Chebfun(runge)
        x = chebpoly(1)
        rr = 1./(1+25*x**2)
        npt.assert_almost_equal(r(xs),rr(xs), decimal=13)

    def test_idct(self, N=64):
        data = np.random.rand(N-1)
        computed = idct(dct(data))
        npt.assert_allclose(computed, data[:N//2])

    def test_even_data(self):
        """
        even_data on vector of length N+1 returns a vector of size 2*N
        """
        N = 32
        data = np.random.rand(N+1)
        even = even_data(data)
        self.assertEqual(len(even), 2*N)


    @unittest.expectedFailure
    def test_underflow(self):
        Chebfun.max_nb_dichotomy = 13
        p = Chebfun(piecewise_continuous)

class Test_Arithmetic(unittest.TestCase):
    def setUp(self):
        self.p1 = Chebfun(f)
        self.p2 = Chebfun(runge)

    def test_constant_functions(self):
        self.assertEqual(0, Chebfun(zero))
        cf = Chebfun(2.)
        self.assertEqual(3*cf, 6)
        self.assertEqual(3.14159*cf, 6.28318)
        self.assertEqual(len(cf), 2)
        self.assertEqual(len(Chebfun(lambda x: const_func(1.,x))), 2)
        self.assertEqual(len(Chebfun(lambda x: const_func(-2.71828,x))), 2)
        self.assertEqual(len(Chebfun(lambda x: const_func(0.,x))), 2)
        self.assertEqual(len(Chebfun(Chebfun(zero))), 2)

    def test_mul(self):
        self.assertEqual(self.p1*Chebfun(zero), Chebfun(zero))
        self.assertEqual(self.p1*Chebfun(lambda x: const_func(1., x)), self.p1)

    def test_scalar_mul(self):
        self.assertEqual(self.p1, self.p1)
        self.assertEqual(self.p1*1, 1*self.p1)
        self.assertEqual(self.p1*1, self.p1)
        self.assertEqual(0*self.p1, Chebfun(zero))

    def test_commutativity(self):
        self.assertEqual(self.p1*self.p2, self.p2*self.p1)
        self.assertEqual(self.p1+self.p2, self.p2+self.p1)

    def test_minus(self):
        a = self.p1 - self.p2
        b = self.p2 - self.p1
        self.assertEqual(a+b,0)

    def test_add_coeffs(self):
        cfa = Chebfun(lambda x: np.sin(x)**2)
        cfb = Chebfun(lambda x: np.cos(x)**2)
        cfs = cfa + cfb
        self.assertEqual(cfs, Chebfun(lambda x: const_func(1., x)))
        self.assertEqual(len(cfs), 2)

    def test_add_sub_convergence(self):
        pa, pb = problem_functions(30, 125.13)
        # Does not converge: Chebfun(lambda x: pa(x)-pb(x))
        # But should:
        cf = pa - pb
        self.assertEqual(len(cf), max(len(pa), len(pb))+1)
        cf = pa + (-pb)
        self.assertEqual(len(cf), max(len(pa), len(pb))+1)

    def test_neg(self):
        cfa = Chebfun(np.cos)
        cfb = Chebfun(lambda x: -np.cos(x))
        self.assertEqual(-cfa, cfb)



class Test_Interval(unittest.TestCase):

    def test_intersect(self):
        npt.assert_equal(
            Chebfun.intersect_intervals([-5., 0.], [-2., 3.]),
            np.array([-2., 0.]))
        npt.assert_equal(
            Chebfun.intersect_intervals([-5., 5.], [-5., 5.]),
            np.array([-5., 5.]))
        npt.assert_equal(
            Chebfun.intersect_intervals([-5., 4.], [4., 5.]),
            np.array([4., 4.]))
        npt.assert_equal(
            Chebfun.intersect_intervals([-5., -4.], [4., 5.]),
            np.array([0., 0.]))

    def test_init_with_lambda(self):
        cf = Chebfun(lambda x: x*x, interval=[0., 2.])
        cfm1 = Chebfun(lambda x: (x+1.)**2)
        self.assertEqual(len(cf), 3)
        npt.assert_allclose(cf.x, np.linspace(2., 0., 3))
        npt.assert_allclose(cf.f, np.linspace(2., 0., 3)**2)
        npt.assert_equal(cf.chebyshev_coefficients(),
                         cfm1.chebyshev_coefficients())

    def test_init_with_array(self):
        cf = Chebfun([0., 0., 1., 0., 0.], interval=[-3.141, 6.282])
        cfi1 = Chebfun([0., 0., 1., 0., 0.], interval=[-1., 1.])
        self.assertEqual(cf((-3.141+6.282)*0.5), 1.0)
        npt.assert_equal(cf.f, np.array([0., 0., 1., 0., 0.]))
        npt.assert_equal(cf.chebyshev_coefficients(),
                         cfi1.chebyshev_coefficients())

    def test_init_with_coeffs(self):
        xs1 = np.linspace(-1., 1., 1000)
        xs2 = np.linspace(3., 7., 1000)
        cf = Chebfun(chebcoeff=[0., 0., 1.], interval=[3., 7.])
        cfp = chebpoly(2)
        npt.assert_allclose(cf(xs2), cfp(xs1), rtol=1e-11)

    def test_add(self):
        cfa = Chebfun(lambda x: np.sin(x)**2, interval=[-10., 10.])
        cfb = Chebfun(lambda x: np.cos(x)**2, interval=[1., 9.])
        cfs = cfa + cfb
        npt.assert_equal(cfs.interval, cfb.interval)
        self.assertEqual(Chebfun(1., interval=cfb.interval), cfs)
        cfa2 = Chebfun(lambda x: np.sin(x)**2, interval=[-10., 2.])
        cfb2 = Chebfun(lambda x: np.cos(x)**2, interval=[1.1, 9.])
        cfs2 = cfa2 + cfb2
        npt.assert_equal(cfs2.interval, [1.1, 2.])
        self.assertEqual(Chebfun(1., interval=[1.1, 2.]), cfs2)

    def test_subtract(self):
        cfa = Chebfun(lambda x: np.sin(x)**2, interval=[-10., 10.])
        cfb = Chebfun(lambda x: -np.cos(x)**2, interval=[1., 9.])
        cfs = cfa - cfb
        npt.assert_equal(cfs.interval, cfb.interval)
        self.assertEqual(Chebfun(1., interval=cfb.interval), cfs)
        cfa2 = Chebfun(lambda x: np.sin(x)**2, interval=[-10., 2.])
        cfb2 = Chebfun(lambda x: -np.cos(x)**2, interval=[1.1, 9.])
        cfs2 = cfa2 - cfb2
        npt.assert_equal(cfs2.interval, [1.1, 2.])
        self.assertEqual(Chebfun(1., interval=[1.1, 2.]), cfs2)

    def test_multiply(self):
        cfa = Chebfun(lambda x: np.exp(x), interval=[-10., 10.])
        cfb = Chebfun(lambda x: np.exp(-x), interval=[1., 9.])
        cfs = cfa * cfb
        npt.assert_equal(cfs.interval, cfb.interval)
        self.assertEqual(Chebfun(1., interval=cfb.interval), cfs)
        cfa2 = Chebfun(lambda x: np.exp(x), interval=[-10., 2.])
        cfb2 = Chebfun(lambda x: np.exp(-x), interval=[1.1, 9.])
        cfs2 = cfa2 * cfb2
        npt.assert_equal(cfs2.interval, [1.1, 2.])
        self.assertEqual(Chebfun(1., interval=[1.1, 2.]), cfs2)

    def test_divide(self):
        cfa = Chebfun(lambda x: np.sin(x), interval=[-1.56, 1.5])
        cfb = Chebfun(lambda x: np.cos(x), interval=[1., 1.3])
        cfs = cfa / cfb
        npt.assert_equal(cfs.interval, cfb.interval)
        self.assertEqual(Chebfun(lambda x: np.tan(x), interval=cfb.interval), cfs)
        cfa2 = Chebfun(lambda x: np.sin(x), interval=[-1.55, 1.1])
        cfb2 = Chebfun(lambda x: np.cos(x), interval=[1., 1.3])
        cfs2 = cfa2 / cfb2
        npt.assert_equal(cfs2.interval, [1., 1.1])
        self.assertEqual(Chebfun(np.tan, interval=[1., 1.1]), cfs2)

    def test_differentiate(self):
        inter = [-3.141, 2.71828]
        xs = np.linspace(inter[0], inter[1], 500)

        npt.assert_allclose(
            Chebfun(lambda x: const_func(-3.141, x),
                interval=inter).differentiate()(xs),
            Chebfun(0., interval=inter)(xs))
        for n in xrange(2, 11):
            cf = Chebfun(lambda x: x**n, interval=inter)
            self.assertEqual(np.max(np.abs(
                Chebfun(lambda x: n*x**(n-1), interval=inter)(xs)
                    - cf.differentiate()(xs))) <
                1e-9,
                True)
        npt.assert_allclose(
            Chebfun(lambda x: 3.141*x*x, interval=inter).differentiate()(xs),
            Chebfun(lambda x: 6.282*x, interval=inter)(xs))
        npt.assert_allclose(
            Chebfun(np.sin, interval=inter).differentiate()(xs),
            Chebfun(np.cos, interval=inter)(xs))
        npt.assert_allclose(
            Chebfun(lambda x: np.exp(7.*np.sin(3.*x)),
                interval=inter).differentiate()(xs),
            Chebfun(lambda x: 21.*np.cos(3.*x)*np.exp(7.*np.sin(3.*x)),
                interval=inter)(xs),
            1e-07, 1e-07)
        npt.assert_allclose(
            Chebfun(lambda x: np.exp(-0.5*x*x),
                interval=inter).differentiate()(xs),
            Chebfun(lambda x: -x*np.exp(-0.5*x*x),
                interval=inter)(xs))

    def test_arithmetic_with_scalar(self):
        inter = [-3.141, 2.71828]
        npt.assert_equal((2. + Chebfun(1., interval=inter)).interval, inter)
        npt.assert_equal((2. - Chebfun(1., interval=inter)).interval, inter)
        npt.assert_equal((2. * Chebfun(1., interval=inter)).interval, inter)
        npt.assert_equal((2. / Chebfun(1., interval=inter)).interval, inter)

if __name__ == '__main__':
    unittest.main()
    ## suite = unittest.TestLoader().loadTestsFromTestCase(Test_Chebfun)
    ## unittest.TextTestRunner(verbosity=2).run(suite)

