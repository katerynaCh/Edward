from __future__ import print_function

import os
import sys
import timeit

import numpy

import math

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_data, get_image_size, get_amount_of_classes
from mlp import HiddenLayer

# Initialize and create variables defined by user
# First, image sizes and amount of classes

dataset = '/Users/Aleksei/Desktop/testing_original'
image_x, image_y = get_image_size(dataset)
amount_classes = get_amount_of_classes(dataset)

# Pooling size
poolsize_x = 2
poolsize_y = 2

# Learning rate
# Epochs to be trained and batch size
user_learning_rate = 0.0025
user_nepochs = 15
user_batch = 20
#
# # Size of the convolution filter windows
user_filter_x = 5
user_filter_y = 5

# Treshhold for model training
user_treshhold = 0.995

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(poolsize_x, poolsize_y)):
        global poolsize_x
        global poolsize_y
        global image_x
        global image_y
        global amount_classes
        global user_learning_rate
        global user_nepochs
        global user_batch
        global user_filter_x
        global user_filter_y
        global user_treshold
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        #reshape bias to a tensor
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # parameters of the current layer
        self.params = [self.W, self.b]

        # model input
        self.input = input





def evaluate_lenet5(learning_rate=user_learning_rate, n_epochs=user_nepochs,
                    dataset='/Users/Aleksei/Desktop/testing_original',
                    nkerns=[20, 50], batch_size=user_batch):
    """ Calculates the model. If you want to modify the variables further, they are of the following format:
    learning_rate: float
    n_epochs: int
    dataset: string
    nkerns: list of ints
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # calculate the mini batches number for the three stages
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    index = T.lscalar()

    x = T.matrix('x')  # the data is presented as a matrix of RGB of pixels
    y = T.ivector('y')  # the labels are presented as 1D vector of integers

    ####################
    #  BUILD THE MODEL #
    ####################
    print('... building the model')

    # Reshape the matrix to appropriate size.
    layer0_input = x.reshape((batch_size, 1, image_x, image_y))

    # Construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, image_x, image_y),
        filter_shape=(nkerns[0], 1, user_filter_x, user_filter_y),
        poolsize=(poolsize_x, poolsize_y)
    )

    # Construct the further convolutional layers of the amount defined by user
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], math.floor((image_x - user_filter_x + 1) / poolsize_x),
                     math.floor((image_y - user_filter_y + 1) / poolsize_y)),
        filter_shape=(nkerns[1], nkerns[0], user_filter_x, user_filter_y),
        poolsize=(poolsize_x, poolsize_y)
    )
    image_x_1 = math.floor((image_x - user_filter_x + 1) / poolsize_x)
    image_y_1 = math.floor((image_y - user_filter_y + 1) / poolsize_y)

    # create the Hidden layer from the output of previous layers
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * int(math.floor((image_x_1 - user_filter_x + 1) / poolsize_x)) * int(
            math.floor((image_y_1 - user_filter_y + 1) / poolsize_y)),
        n_out=batch_size,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=batch_size, n_out=amount_classes)

    # the cost evaluates the accuracy of the model
    cost = layer3.negative_log_likelihood(y)

    # test and validate the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    patience = 10000
    patience_increase = 2  # time between iterations
    improvement_threshold = user_treshhold # result is considered to be better if improved by this K
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                # test if the current validation score is better than the best. save if it is.
                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                        ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)