import numpy as np
import numba as nb
from math import lgamma
from math import erf as erff

@nb.vectorize(cache=True)
def erf(x):
        return erff(x)

@nb.vectorize(cache=True)
def gammaln(x):
	return lgamma(x)

@nb.vectorize(cache=True)
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

@nb.vectorize([nb.float64(nb.float64)],cache=True)
def trigamma(q):
	''' This is the Cephes version of zeta used in Scipy, but I rewrote it in Python for x = 2'''

	A = [
		12.0,
		-720.0,
		30240.0,
		-1209600.0,
		47900160.0,
		-1.8924375803183791606e9,
		7.47242496e10,
		-2.950130727918164224e12,
		1.1646782814350067249e14,
		-4.5979787224074726105e15,
		1.8152105401943546773e17,
		-7.1661652561756670113e18
	]
	macheps =  2.22044604925e-16 ## double for numpy

	if q <= 0.0:
		if q == np.floor(q):
			return np.nan
		return np.inf

	# /* Asymptotic expansion
	#  * https://dlmf.nist.gov/25.11#E43
	#  */
	if (q > 1e8):
		return (1. + .5/q) /q

	# /* Euler-Maclaurin summation formula */
	# /* Permit negative q but continue sum until n+q > +9 .
	#  * This case should be handled by a reflection formula.
	#  * If q<0 and x is an integer, there is a relation to
	#  * the polyGamma function.
	#  */
	s = q**(-2.)
	a = q
	i = 0
	b = 0.0

	while ((i<9) or (a<=9.0)):
		i+= 1
		a += 1.0
		b = a**(-2.)
		s += b
		if np.abs(b/s) < macheps:
			return s

	w = a
	s += b*w
	s -= .5*b
	a = 1.
	k = 0.
	for i in range(12):
		a *= 2+k
		b /= w
		t = a*b / A[i]
		s = s +t
		t = np.abs(t/s)
		if t < macheps:
			return s
		k += 1.0
		a *= 2 + k
		b /= w
		k += 1.
	return s
### Test
# psi(1.) # initialize - don't time the jit process
# from scipy import special
#
# a = np.random.rand(1000,10)*20
#
# %timeit special.psi(a)
# %timeit psi(a)


@nb.vectorize(cache=True)
def invert_psi(y):
	## Minka appendix C -- doesn't really work if x is negative....

	## initial guess (Minka 149)
	if y >= -2.22:
		x = np.exp(y)+.5
	else:
		x = -1./(y - psi(1.))

	## iterations
	for i in range(5): ## Minka says only 5 to get 14 bit accuracy
		x = x - (psi(x)-y)/trigamma(x) ## Minka 146
	return x

# @nb.vectorize(cache=True)
# def invpsi(x):
# # Y = INVPSI(X)
# #
# # Inverse digamma (psi) function.  The digamma function is the
# # derivative of the log gamma function.  This calculates the value
# # Y > 0 for a value X such that digamma(Y) = X.
# #
# # This algorithm is from Paul Fackler: http://www4.ncsu.edu/~pfackler/

#     L = 1;
#     y = np.exp(x)
#     while L > 10e-8:
#         y += L*np.sign(x-psi(y))
#         L = L / 2

#     return y


invpsi = invert_psi

@nb.njit
def rev_eye(N):
	return np.ones((N,N)) - np.eye(N)