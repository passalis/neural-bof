import theano
import numpy as np
from sklearn.preprocessing import normalize as feature_normalizer
import theano.tensor as T
import theano.gradient
import sklearn.cluster as cluster

floatX = theano.config.floatX


class NBoFInputLayer:
    """
    Defines a Neural BoF input layer
    """

    def __init__(self, g=0.1, feature_dimension=89, n_codewords=16):
        """
        Intializes the Neural BoF object
        :param g: defines the softness of the quantization
        :param feature_dimension: dimension of the feature vectors
        :param n_codewords: number of codewords / RBF neurons to be used
        """

        self.Nk = n_codewords
        self.D = feature_dimension

        # RBF-centers / codewords
        V = np.random.rand(self.Nk, self.D).astype(dtype=floatX)
        self.V = theano.shared(value=V, name='V', borrow=True)
        # Input weights for the RBF neurons
        self.W = theano.shared(value=np.ones((self.Nk, self.D), dtype=floatX) / g, name='W')

        # Tensor of input objects (n_objects, n_features, self.D)
        self.X = T.tensor3(name='X', dtype=floatX)

        # Feature matrix of an object (n_features, self.D)
        self.x = T.matrix(name='x', dtype=floatX)

        # Encode a set of objects
        """
        Note that the number of features per object is fixed and same for all objects.
        The code can be easily extended by defining a feature vector mask, allowing for a variable number of feature
        vectors for each object (or alternatively separately encoding each object).
        """
        self.encode_objects_theano = theano.function(inputs=[self.X], outputs=self.sym_histograms(self.X))

        # Encodes only one object with an arbitrary number of features
        self.encode_object_theano = theano.function(inputs=[self.x], outputs=self.sym_histogram(self.x))

    def sym_histogram(self, X):
        """
        Computes a soft-quantized histogram of a set of feature vectors (X is a matrix).
        :param X: matrix of feature vectors
        :return:
        """
        distances = sym_distance_matrix(X, self.V, self.W)
        membership = T.nnet.softmax(-distances)
        histogram = T.mean(membership, axis=0)
        return histogram

    def sym_histograms(self, X):
        """
        Encodes a set of objects (X is a tensor3)
        :param X: tensor3 containing the feature vectors for each object
        :return:
        """
        histograms, updates = theano.map(self.sym_histogram, X)
        return histograms

    def initialize_dictionary(self, X, max_iter=100, redo=5, n_samples=10000, normalize=True):
        """
        Samples some feature vectors from X and learns an initial dictionary
        :param X: list of objects
        :param max_iter: maximum k-means iters
        :param redo: number of times to repeat k-means clustering
        :param n_samples: number of feature vectors to sample from the objects
        :param normalize: use l_2 norm normalization for the feature vectors
        """

        # Sample only a small number of feature vectors from each object
        samples_per_object = np.ceil(n_samples / len(X))

        features = None
        print "Sampling feature vectors..."
        for i in (range(len(X))):
            idx = np.random.permutation(X[i].shape[0])[:samples_per_object+1]
            cur_features = X[i][idx, :]
            if features is None:
                features = cur_features
            else:
                features = np.vstack((features, cur_features))

        print "Clustering feature vectors..."
        features = np.float64(features)
        if normalize:
            features = feature_normalizer(features)

        V = cluster.k_means(features, n_clusters=self.Nk, max_iter=max_iter, n_init=redo)
        self.V.set_value(np.asarray(V[0], dtype=theano.config.floatX))


def sym_distance_matrix(A, V, W):
    """
    Calculates the distances between the feature vectors in A and the codewords in V (weighted by W)
    :param A: the matrix that contains the feature vectors
    :param V: the matrix that contains the codewords / RBF neurons centers
    :param W: weight matrix (if W is set to 1, then the regular distance matrix is computed)
    :return:
    """

    def row_dist(t, w):
        D = (w*(A - t)) ** 2
        D = T.sum(D, axis=1)
        D = T.maximum(D, 0)
        D = T.sqrt(D)
        return D

    D, _ = theano.map(fn=row_dist, sequences=[V, W])
    return D.T

