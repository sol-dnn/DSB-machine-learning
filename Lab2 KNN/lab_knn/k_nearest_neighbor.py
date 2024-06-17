import numpy as np


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
    Train the classifier. For k-nearest neighbors this is just
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
                    
            # X_test: Matrix of test points, shape (num_test, num_features)
            # X_train: Matrix of training points, shape (num_train, num_features)
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        # Initialize the distances matrix with zeros
        dists = np.zeros((num_test, num_train))
        
        for i in range(num_test):
            for j in range(num_train):
            # Compute the L2 distance between the ith test point and the jth training point
            #dists[i, j] = np.sqrt(np.sum(np.square(X[i, :] - self.X_train[j, :])))
                dists[i, j] = np.linalg.norm(X[i] - self.X_train[j],ord=2)
          
        return dists

    def compute_distances_one_loop(self, X):
        """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
        #######################################################################
        # TODO:                                                               #
        # Compute the l2 distance between the ith test point and all training #
        # points, and store the result in dists[i, :].                        #
        #######################################################################
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        # Initialize the distances matrix with zeros
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            # Compute the L2 distance between the ith test point and all training points
            dists[i, :] = np.linalg.norm(X[i] - self.X_train, ord=2, axis=1)
        return dists
        

        '''
        X[i]: This selects the i-th row of the matrix X, representing a single test point.
        self.X_train: This represents the matrix of training points. 
        X[i] - self.X_train: This operation subtracts the i-th test point (a row vector) from each row of the self.X_train               matrix. This is where broadcasting comes into play.
        Broadcasting is a mechanism in NumPy that allows operations between arrays of different shapes and sizes. When operating         on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions and works its way               backward, comparing dimensions. 
        Two dimensions are compatible when they are equal or one of them is 1. If these conditions are not met, NumPy raises a           "ValueError."
        In this case, X[i] is broadcasted to have the same shape as self.X_train by duplicating its row across the new axis.

        The result is a matrix where each row represents the element-wise subtraction between the i-th test point and the               corresponding training point.
        '''


    def compute_distances_no_loops(self, X):
        
        """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.
    Input / Output: Same as compute_distances_two_loops
    """
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        # Use broadcasting to compute differences between all test points and all training points
        differences = X[:, np.newaxis, :] - self.X_train

        # Compute squared distances
        squared_dists = np.sum(differences ** 2, axis=2)

        # Take the square root to get the Euclidean distances
        dists = np.sqrt(squared_dists)
 
        return dists


    def predict_labels(self, dists, k=1):
        
        """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
        #########################################################################
        # TODO:                                                                 #
        # Use the distance matrix to find the k nearest neighbors of the ith    #
        # testing point, and use self.y_train to find the labels of these       #
        # neighbors. Store these labels in a variable (let's say closest_y).    #
        # Hint: Look up the function numpy.argsort.                             #
        # Now that you have found the labels of the k nearest neighbors, you    #
        # need to find the most common label in the list closest_y of labels.   #
        # Store this label in y_pred[i]. Break ties by choosing the smaller     #
        # label.                                                                #
        #########################################################################
                
        num_test = dists.shape[0]  #number of rows
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            # Find the indices of the k-nearest neighbors for the ith test point
            closest_indices = np.argsort(dists[i, :])[:k]
            #This loop iterates over each test point. For each test point, 
            #it finds the indices of the k-nearest neighbors 
            #by sorting the distances for that test point and taking the first k indices.
            
            #closest_indices is an array with the indices of the k smallest elements in the i-th row of dists.
            
            
            # Get the labels of the k-nearest neighbors
            closest_labels = self.y_train[closest_indices]

            # Find the most common label in the list of closest labels
            unique_labels, counts = np.unique(closest_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]

            # Store the predicted label for the ith test point
            y_pred[i] = most_common_label
            
        return y_pred
