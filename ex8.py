import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def psi(d, x):
	vec = np.zeros(d + 1)
	for i in xrange(d + 1):
		vec[i] = float(x)**i
	return vec


def is_invertible(a):
	return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def leastSquare(X, Y, d):
	tempA = np.transpose(np.array((map(partial(psi, d), X))))
	tempB = np.transpose(np.array((map(partial(psi, d), X))))
	A = np.dot(tempA, tempA.transpose())
	B = np.dot(tempB, Y)

	if is_invertible(A):
		return np.dot(np.linalg.inv(A), B)
	else:
		return np.dot(np.dot(np.linalg.pinv(A), tempA), Y)


def trainPolynomials(trainingX, trainingY):
	polynomials = []

	for i in xrange(15):
		polynomials.append(leastSquare(trainingX, trainingY, i + 1))
		print "Training error on polynomial {} is {}".format(i + 1, calculateError(trainingX, trainingY, polynomials[-1]))

	return polynomials


def calculateError(X, Y, polynomial):
	psiD = partial(psi, len(polynomial)-1)

	error = 0.0
	for i in xrange(len(X)):
		error += ((np.inner(polynomial, psiD(X[i]))) - Y[i]) ** 2

	return error / len(X)


def choosePolynomial(polynomials, validationX, validationY):
	minError = calculateError(validationX, validationY, polynomials[0])
	minErrorPolynomial = polynomials[0]

	for polynomial in polynomials:
		error = calculateError(validationX, validationY, polynomial)
		print "Validation error on polynomial {} is {}".format(len(polynomial)-1, error)
		if error < minError:
			minError = error
			minErrorPolynomial = polynomial

	return minErrorPolynomial

if __name__ == "__main__":
	X = np.loadtxt("X.txt");
	Y = np.loadtxt("Y.txt");

	trainingX = X[:20]
	trainingY = Y[:20]
	validationX = X[20:121]
	validationY = Y[20:121]
	testX = X[121:]
	testY = Y[121:]

	polynomials = trainPolynomials(trainingX, trainingY)
	bestPolynomial = choosePolynomial(polynomials, validationX, validationY)
	print "Polynomial of degree {} fits best.".format(len(bestPolynomial) - 1)
	print bestPolynomial
	print "Error on test set with best polynomial is {}".format(calculateError(testX, testY, bestPolynomial))


	plt.plot(map(lambda x: len(x) - 1, polynomials), map(partial(calculateError, trainingX, trainingY), polynomials), '-b', label='Training error')
	plt.plot(map(lambda x: len(x) - 1, polynomials), map(partial(calculateError, validationX, validationY), polynomials), '-r', label='Validation error')
	plt.legend()
	plt.xlabel('Polynomial degree')
	plt.ylabel('Error')
	plt.show()

	# plt.plot(X, Y, 'ob', label='Original data', markersize=2)
	# colors = ['r','b','g','y','b']
	# for i in xrange(10, 11, 1):
	# 	psiD = partial(psi, len(polynomials[i])-1)
	# 	plt.plot(X, map(partial(np.inner, polynomials[i]), map(psiD, X)), 'o'+colors[i%len(colors)], label='Estimate {}'.format(i+1), markersize=1)
	# plt.legend()
	# plt.show()