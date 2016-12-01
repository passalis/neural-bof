import lasagne
import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from nbof import NBoFInputLayer


class NBoF:
    """
    Implements the Neural BoF model
    """

    def __init__(self, n_codewords=256, eta=0.001, eta_V=0.001, eta_W=0.0001, g=0.01, update_V=False, update_W=False,
                 n_hidden=200, n_output=15, activation='sigmoid'):
        """
        :param n_codewords: number of codewords / RBF neurons to be used
        :param eta: learning rate for the output MLP
        :param eta_V: learning rate for the centers of the RBF neurons
        :param eta_W: learning rate for the input weights of the RBF neurons
        :param g: defines the softness of the quantization
        :param update_V: if set to True, the model learns the centers of the RBF neurons using backprop
        :param update_W: if set to True, the model learns the input weights of the RBF neurons using backprop
        :param n_hidden: number of hidden neurons to be used in the output MLP
        :param n_output: number of output neurons (equals to the number of the classes)
        :param activation: type of activation (sigmoid or elu)
        """
        self.n_output = n_output

        # Input variables
        input = T.ftensor3('input_data')
        labels = T.imatrix('labels')

        # Neural BoF input Layer
        self.bow = NBoFInputLayer(g=g, feature_dimension=512, n_codewords=n_codewords)

        # Define the MLP
        network = lasagne.layers.InputLayer(shape=(None, n_codewords), input_var=self.bow.sym_histograms(input))
        if activation == 'sigmoid':
            network = lasagne.layers.DenseLayer(network, n_hidden, nonlinearity=lasagne.nonlinearities.sigmoid,
                                                W=lasagne.init.Normal(std=0.5, mean=0))
            network = lasagne.layers.DenseLayer(network, n_output, nonlinearity=lasagne.nonlinearities.sigmoid,
                                            W=lasagne.init.Normal(std=0.5, mean=0))
        elif activation == 'elu':
            network = lasagne.layers.DenseLayer(network, n_hidden, nonlinearity=lasagne.nonlinearities.elu,
                                                W=lasagne.init.Normal(std=0.5, mean=0))
            network = lasagne.layers.DenseLayer(network, n_output, nonlinearity=lasagne.nonlinearities.softmax,
                                            W=lasagne.init.Normal(std=0.5, mean=0))
        else:
            print "Activation function not supported!"
            assert False


        # Define the used loss function ( a variant of the cross entropy loss is used )
        params = lasagne.layers.get_all_params(network, trainable=True)
        prediction = lasagne.layers.get_output(network)
        prediction_vec = prediction.reshape((-1,))
        labels_vec = labels.reshape((-1,))
        prediction_vec = theano.tensor.clip(prediction_vec, 0.0001, 0.9999)
        loss = labels_vec*T.log(prediction_vec) + (1-labels_vec)*T.log(1-prediction_vec)
        loss = -loss.sum()

        # Compile a function of training the MLP only
        updates = lasagne.updates.adam(loss, params, learning_rate=eta)
        self.finetune = theano.function(inputs=[input, labels], outputs=loss, updates=updates)

        # Compile a function for training the whole network structure
        if update_V:
            dictionary_grad = T.grad(loss, self.bow.V)
            dictionary_grad = T.switch(T.isnan(dictionary_grad), 0, dictionary_grad)
            updates_V = lasagne.updates.adam(loss_or_grads=[dictionary_grad], params=[self.bow.V], learning_rate=eta_V)
            updates.update(updates_V)
        if update_W:
            W_grad = T.grad(loss, self.bow.W)
            W_grad = T.switch(T.isnan(W_grad), 0, W_grad)
            updates_sigma = lasagne.updates.adam(loss_or_grads=[W_grad], params=[self.bow.W], learning_rate=eta_W)
            updates.update(updates_sigma)
        self.train = theano.function(inputs=[input, labels], outputs=loss, updates=updates)

        # Compile a function for testing the network
        self.predict = theano.function(inputs=[input], outputs=T.argmax(prediction, axis=1))

    def do_train_epoch(self, train_idx, train_data, train_labels, batch_size=50, normalize=True, type='full',
                       n_samples=0):
        """
        Trains the Neural BoF model for one epoch (one pass through the data)
        :param train_idx: the indices to be used for training (the training set is not expected to fit in RAM)
        :param train_data: the full data matrix (usually an h5 array)
        :param train_labels: the full labels array
        :param batch_size: batch size
        :param normalize: if True, normalizes the features to have unit l_2 norm
        :param type: either 'full' (trains the whole network) or 'finetune' (trains only the MLP)
        :param n_samples: number of feature vectors to sample from each object
        :return:
        """
        n_batches = int(len(train_idx) / batch_size)
        loss = 0

        for i in tqdm(range(n_batches)):
            cur_idx = train_idx[i * batch_size:(i + 1) * batch_size]
            cur_idx = np.sort(cur_idx)
            cur_data = np.float32(train_data[cur_idx, :, :])

            # Use feature streaming without memory
            # Note that memory is not supported in the current implementation
            if n_samples > 0:
                cur_data_upd = np.zeros((cur_data.shape[0], n_samples, cur_data.shape[2]), dtype='float32')
                for j in range(cur_data.shape[0]):
                    idx = np.random.permutation(cur_data.shape[1])[:n_samples]
                    cur_data_upd[j, :, :] = cur_data[j, idx, :]
                cur_data = cur_data_upd

            cur_labels = to_one_hot(train_labels[cur_idx], n=self.n_output)
            if normalize:
                cur_data = normalize_l2(cur_data)

            if type == 'full':
                cur_loss = self.train(cur_data, cur_labels)
            else:
                cur_loss = self.finetune(cur_data, cur_labels)
            loss += cur_loss

        if n_batches * batch_size < len(train_idx):
            cur_idx = train_idx[n_batches * batch_size:]
            # Sort idx (h5 arrays are used)
            cur_idx = np.sort(cur_idx)
            cur_data = np.float32(train_data[cur_idx, :, :])
            cur_labels = to_one_hot(train_labels[cur_idx], n=self.n_output)

            # Permutate the feature vectors
            if n_samples > 0:
                cur_data_upd = np.zeros((cur_data.shape[0], n_samples, cur_data.shape[2]), dtype='float32')
                for j in range(cur_data.shape[0]):
                    idx = np.random.permutation(cur_data.shape[1])[:n_samples]
                    cur_data_upd[j, :, :] = cur_data[j, idx, :]
                cur_data = cur_data_upd

            if normalize:
                cur_data = normalize_l2(cur_data)
            loss += self.train(cur_data, cur_labels)

        print "Loss = ", loss / (n_batches * batch_size + len(cur_labels))

    def evaluate_model(self, idx, data, target_labels, batch_size=128, normalize=True):
        """
        Evaluates the Neural BoF model
        :param idx: test indices
        :param data: data
        :param target_labels: labels
        :param batch_size: batch size
        :param normalize: if True, normalizes the feature vectors to have unit l_2 norm
        :return:
        """
        idx = np.sort(idx)
        labels = np.zeros((len(target_labels),))
        n_batches = int(len(idx) / batch_size)
        for i in tqdm(range(n_batches)):
            cur_idx = idx[i * batch_size:(i + 1) * batch_size]
            cur_data = data[cur_idx, :, :]
            if normalize:
                cur_data = normalize_l2(cur_data)
            labels[cur_idx] = self.predict(cur_data)
        if n_batches * batch_size < len(idx):
            cur_idx = idx[n_batches * batch_size:]
            cur_data = data[cur_idx, :, :]
            if normalize:
                cur_data = normalize_l2(cur_data)
            labels[cur_idx] = self.predict(cur_data)
        acc = 100 * np.mean(labels[idx] == target_labels[idx])
        precision = 100 * mean_precision(target_labels[idx], labels[idx])
        return acc, precision


def normalize_l2(data):
    from sklearn.preprocessing import normalize
    for i in range(len(data)):
        data[i, :, :] = normalize(data[i, :, :])
    return data


def to_one_hot(labels, n=15):
    labs = np.zeros((len(labels), n))
    for i, label in enumerate(labels):
        labs[i, label] = 1
    return np.int32(labs)


def mean_precision(labels, pred):
    unique_labels = np.unique(labels)
    prec = 0.0
    for label in unique_labels:
        idx = labels == label
        prec += float(np.sum(pred[idx] == label)) / float(np.sum(idx))
    return prec / unique_labels.shape[0]
