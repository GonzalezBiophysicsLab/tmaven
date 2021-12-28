import numpy as np
import numba as nb
from math import lgamma
from math import erf as erff

@nb.vectorize
def erf(x):
	return erff(x)

@nb.vectorize
def gammaln(x):
	return lgamma(x)

@nb.vectorize
def psi(x):
	''' This is the Cephes version used in Scipy, but I rewrote it in Python'''
	A = [
		8.33333333333333333333E-2,
		-2.10927960927960927961E-2,
		7.57575757575757575758E-3,
		-4.16666666666666666667E-3,
		3.96825396825396825397E-3,
		-8.33333333333333333333E-3,
		8.33333333333333333333E-2
	]

	# check for positive integer up to 10
	if x <= 10. and x==np.floor(x):
		y = 0.0
		for i in range(1,np.floor(x)):
			y += 1./i
		y -= 0.577215664901532860606512090082402431 #Euler
		return y
	else:
		s = x
		w = 0.0
		while s < 10.:
			w += 1./s
			s += 1.
		z = 1.0 / (s * s);


		poly = A[0]
		for aa in A[1:]:
			poly = poly*z +aa
		y = z * poly
		y = np.log(s) - (0.5 / s) - y - w;

		return y


### Test
# psi(1.) # initialize - don't time the jit process
# from scipy import special
#
# a = np.random.rand(1000,10)*20
#
# %timeit special.psi(a)
# %timeit psi(a)
