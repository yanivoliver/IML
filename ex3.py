import numpy as np

# This function computes the empirical 0-1 error of the classifier corresponding
# to the given coordinate.
#
# Input:
#   'coordinate': an int netween 0 to (n-1)
#   'data': a 0-1 martix of size n times m and of type numpy.ndarray.
#   Each column is an example
#   'labels': a 0-1 vector (of type numpy.ndarray) of size m. label[i] is the 
#   label of the i'th example
#
# Output:
#   'zero_one_loss': The empirical 0-1 error of the classifier corresponding
#   to 'coordinate'.
def zeroOneLoss(coordinate,data,labels):
    num_coordinates, num_samples, = data.shape
    #Count the errors of the given coordinate. Store the number in 'num_errors'
    num_errors = float(sum([1 if data[coordinate][i] != labels[i] else 0 for i in xrange(len(data[coordinate]))]))
    zero_one_loss = num_errors/num_samples                         
    return zero_one_loss

# This function computes the empirical 0-1 error of all the classifiers in our
# class.
#
# Input:
#   'data': A 0-1 martix of size n times m and of type numpy.ndarray.
#   Each column is an example
#   'labels': A 0-1 vector (of type numpy.ndarray) of size m. label[i] is the
#   label of the i'th example
#
# Output:
#   'zero_one_loss': A vector (of type numpy.ndarray) of size n such that
#   zero_one_loss[i] is the empirical 0-1 error of the classifier corresponding
#   to i.
def allZeroOneLosses(data, labels):
    num_coordinates, num_samples, = data.shape
    zero_one_loss = np.zeros(num_coordinates)
    # Fill the vector 'zero_one_loss'
    for i in xrange(num_coordinates):
        zero_one_loss[i] = zeroOneLoss(i, data, labels)
    return zero_one_loss

# This function computes the ERM.
#
# Input:
#   'data': A 0-1 martix of size n times m and of type numpy.ndarray.
#   Each column is an example
#   'labels': A 0-1 vector (of type numpy.ndarray) of size m. label[i] is the
#   label of the i'th example
#
# Output:
#   'erm': The coordinate of the ERM classifier
#   'erm_error': The error of the ERM classifier
def ERM(data, labels):
    zero_one_loss = allZeroOneLosses(data,labels)
    # Find the ERM and its empirical error
    erm_error = zero_one_loss.min()
    erm = zero_one_loss.argmin()
    return erm, erm_error

# This function computes a bag-of-words representation of the given text.
#
# Input:
#   'text': A text file. Every line contiant an SMS message and a label.
#   The first word in each line is either 'spam' or 'ham', indicating wheather
#   the message is spam or not. The remaining part of line is the message.
#   'D': A python dictinary, mapping strings (words) to integers.
#   'num_samples': The number of examples. Equals to the number of
#   lines in 'text'
#
# Output:
#   'data': A 0-1 martix of size len(D) times num_examples and of type
#   numpy.ndarray. Each column is an example. The i'th column corresponds
#   to the i'th messege in 'text'. For every word w in D, if w appears in
#   the message, then the D[w]'th coordinate in that vector is 1. The rest of
#   the coordinates are 0.
#   'labels': A 0-1 vector (of type numpy.ndarray) of size m. label[i] is the
#   label of the i'th message (1 if it is "spam" and 0 if it is "ham")
def vectorizeText(text, D, num_samples):
    vector_length = len(D)
    data=np.zeros((vector_length,num_samples))
    labels=np.zeros(num_samples)    
    ln=0
    for line in text:
        parts = line.split()
		# read the line and fill in the returned matrix ('data') and vector
        # ('labels') remember that the first word is not part of the messege,
        # and only indicates whether the messege is spam or not        
        labels[ln] = 1 if parts[0] == "spam" else 0
        for i in xrange(1, len(parts)):
            data[D[parts[i]]][ln] = 1
        ln=ln+1

    return data,labels
