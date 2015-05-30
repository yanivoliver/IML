import numpy as np

def full_perceptron(X, Y):
	w = np.zeros(X[0].size)
	for i in xrange(len(X)):
		if Y[i] * np.inner(w, X[i]) <= 0:
			w += Y[i] * X[i]
	return w

def calculate_error(X, Y, w):
	errors = 0.0
	for i in xrange(len(X)):
		if Y[i] * np.inner(w, X[i]) <= 0:
			errors += 1
	return errors / len(X)

if __name__ == "__main__":
	Xtrain = np.loadtxt("Xtrain");
	Ytrain = np.loadtxt("Ytrain");
	Xtest = np.loadtxt("Xtest")
	Ytest = np.loadtxt("Ytest")

	w = full_perceptron(Xtrain, Ytrain)
	print "Test average lost: {}%".format(calculate_error(Xtest, Ytest, w))