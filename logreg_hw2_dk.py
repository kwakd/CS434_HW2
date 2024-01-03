from math import log
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
step_size=0.0001
max_iters=1000

def main():

    # Load the training data
    logging.info("Loading data")
    X_train, y_train, X_test = loadData()

    logging.info("\n---------------------------------------------------------------------------\n")

    # Fit a logistic regression model on train and plot its losses
    logging.info("Training logistic regression model (No Bias Term)")
    w, losses = trainLogistic(X_train,y_train)
    y_pred_train = X_train@w >= 0

    logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
    logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))

    logging.info("\n---------------------------------------------------------------------------\n")

    X_train_bias = dummyAugment(X_train)

    # Fit a logistic regression model on train and plot its losses
    logging.info("Training logistic regression model (Added Bias Term)")
    w_bias, bias_losses = trainLogistic(X_train_bias,y_train)
    y_pred_train = X_train_bias@w_bias >= 0

    logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w_bias]))
    logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))

    plt.figure(figsize=(16,9))

    # need to do reshape
    reshaped_losses = np.reshape(losses, len(losses))
    reshaped_bias_losses = np.reshape(bias_losses, len(bias_losses))

    plt.plot(range(len(losses)), reshaped_losses, label="No Bias Term Added")
    plt.plot(range(len(bias_losses)), reshaped_bias_losses, label="Bias Term Added")
    plt.title("Logistic Regression Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Likelihood")
    plt.legend()
    plt.show()

    logging.info("\n---------------------------------------------------------------------------\n")

    logging.info("Running cross-fold validation for bias case:")

    # Perform k-fold cross
    for k in [2,3,4, 5, 10, 20, 50]:
        cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k)
        logging.info("{}-fold Cross Val Accuracy -- Mean (stdev): {:.4}% ({:.4}%)".format(k,cv_acc*100, cv_std*100))

    ####################################################
    # Write the code to make your test submission here
    ####################################################

    #raise Exception('Student error: You haven\'t implemented the code in main() to make test predictions.')
    X_test_bias = dummyAugment(X_test)
    y_pred_test = X_test_bias@w_bias >= 0
    y_pred_test = y_pred_test * 1 # convert boolean array to int (0/1) array
    test_out = np.concatenate((np.expand_dims(np.array(range(len(y_pred_test)),dtype=np.int64), axis=1), y_pred_test), axis=1)
    header = np.array([["id", "type"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_logreg.csv', test_out, fmt='%s', delimiter=',')


######################################################################
# Q3.1 logistic
######################################################################
# Given an input vector z, return a vector of the outputs of a logistic
# function applied to each input value
#
# Input:
#   z --   a n-by-1 vector
#
# Output:
#   logit_z --  a n-by-1 vector where logit_z[i] is the result of
#               applying the logistic function to z[i]
######################################################################
def logistic(wx):
	return (1.0 / (1.0 + np.exp(wx)))


######################################################################
# Q3.2 calculateNegativeLogLikelihood
######################################################################
# Given an input data matrix X, label vector y, and weight vector w
# compute the negative log likelihood of a logistic regression model
# using w on the data defined by X and y
#
# Input:
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   y --    a n-by-1 vector representing the labels of the examples in X
#
#   w --    a d-by-1 weight bector
#
# Output:
#   nll --  the value of the negative log-likelihood
######################################################################
def calculateNegativeLogLikelihood(X,y,w):
    # print("X", X)
    # print(X.shape)
    # print(len(X))
    # print("y", y)
    # print("w", w)

    nll = 0
    for i in range(len(X)):
        wx = -np.dot(np.transpose(w), X[i])
        # wx = -np.dot(w.T, X[i]) # same as wx = -np.dot(np.transpose(w), X[i])
        # print(wx)

        # equation (7)
        logisticWX = logistic(wx.reshape(len(wx), 1))
        # print(wx.reshape(len(wx)))
        # print(logisticWX)

        # equation (8)
        nll -= y[i]*np.log(logisticWX) + (1 - y[i])*np.log(1.0 - logisticWX)
        # print(nll)
        # print(y[i])
        # print(X[i])

    # print("nll", nll)
    # print("nll.shape", nll.shape)

    return nll



######################################################################
# Q4 trainLogistic
######################################################################
# Given an input data matrix X, label vector y, maximum number of
# iterations max_iters, and step size step_size -- run max_iters of
# gradient descent with a step size of step_size to optimize a weight
# vector that minimizies negative log-likelihood on the data defined
# by X and y
#
# Input:
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   y --    a n-by-1 vector representing the labels of the examples in X
#
#   max_iters --   the maximum number of gradient descent iterations
#
#   step_size -- the step size (or learning rate) for gradient descent
#
# Output:
#   w --  the d-by-1 weight vector at the end of training
#
#   losses -- a list of negative log-likelihood values for each iteration
######################################################################
def trainLogistic(X,y, max_iters=max_iters, step_size=step_size):

    # Initialize our weights with zeros
    w = np.zeros( (X.shape[1],1) )

    # Keep track of losses for plotting
    losses = [calculateNegativeLogLikelihood(X,y,w)]

    # Take up to max_iters steps of gradient descent
    for i in range(max_iters):

        # Todo: Compute the gradient over the dataset and store in w_grad
        # .
        # . Implement equation 9.
        # .

        # equation (9)
        w_grad = 0
        for i in range(len(X)):
          wx = -np.dot(np.transpose(w), X[i])
          # print(wx)

          logisticWX = logistic(wx)
          # print(logisticWX)

          w_grad += (logisticWX - y[i]) * X[i]
          # print(w_grad)


        # This is here to make sure your gradient is the right shape
        # print(w_grad.shape)
        # print((X.shape[1],))
        # print((X.shape[1],1))
        w_grad = w_grad.reshape(len(w_grad), 1)
        assert(w_grad.shape == (X.shape[1],1))

        # Take the update step in gradient descent
        w = w - step_size*w_grad

        # Calculate the negative log-likelihood with the
        # new weight vector and store it for plotting later
        losses.append(calculateNegativeLogLikelihood(X,y,w))


        # print(losses)
        # print(len(losses))


    return w, losses


######################################################################
# Q5 dummyAugment
######################################################################
# Given an input data matrix X, add a column of ones to the left-hand
# side
#
# Input:
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
# Output:
#   aug_X --  a n-by-(d+1) matrix of examples where each row
#                   corresponds to a single d-dimensional example
#                   where the the first column is all ones
#
######################################################################
def dummyAugment(X):
    # print(X)
    # print(X.shape)
    new_column = np.ones(len(X))
    aug_X = np.insert(X, 0, new_column, axis=1)
    # print(aug_X)
    # print(aug_X.shape)
    return aug_X




##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

# Given a matrix X (n x d) and y (n x 1), perform k fold cross val.
def kFoldCrossVal(X, y, k):
  fold_size = int(np.ceil(len(X)/k))

  rand_inds = np.random.permutation(len(X))
  X = X[rand_inds]
  y = y[rand_inds]

  acc = []
  inds = np.arange(len(X))
  for j in range(k):

      start = min(len(X),fold_size*j)
      end = min(len(X),fold_size*(j+1))
      test_idx = np.arange(start, end)
      train_idx = np.concatenate( [np.arange(0,start), np.arange(end, len(X))] )
      if len(test_idx) < 2:
        break

      X_fold_test = X[test_idx]
      y_fold_test = y[test_idx]

      X_fold_train = X[train_idx]
      y_fold_train = y[train_idx]

      w, losses = trainLogistic(X_fold_train, y_fold_train)

      acc.append(np.mean((X_fold_test@w >= 0) == y_fold_test))

  return np.mean(acc), np.std(acc)


# Loads the train and test splits, passes back x/y for train and just x for test
def loadData():
  train = np.loadtxt("train_cancer.csv", delimiter=",")
  test = np.loadtxt("test_cancer_pub.csv", delimiter=",")

  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test

  return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.


main()
