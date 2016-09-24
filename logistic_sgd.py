from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import sys
import timeit

import numpy

import theano
import theano.tensor as T

from PIL import Image
import os, os.path


class LogisticRegression(object):
    # Logistic regression performs classification based on convolutional pooling layers output

    def __init__(self, input, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def errors(self, y):
        # check if there are any errors

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            # 1 means the error in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def get_image_size(dataset):
    list_fold = os.listdir(dataset)
    for folder in list_fold:
        if os.path.isdir(os.path.join(dataset, folder)) and str.isdigit(folder):
            list_files = os.listdir(os.path.join(dataset, folder))
            for file in list_files:
                if file.endswith("jpg"):
                    wid, hei = Image.open(os.path.join(dataset, folder, file)).size
                    break
            break
    return wid, hei


def resize_data(file, width, height):
    im1 = Image.open(file)
    image_resized = im1.resize((width, height), Image.ANTIALIAS)
    image_resized.save(file)

def get_amount_of_classes(dataset):
    i=0
    list_fold = os.listdir(dataset)
    for folder in list_fold:
        if os.path.isdir(os.path.join(dataset, folder)) and str.isdigit(folder):
            i+=1
    return i

def load_data(dataset):
    global count
    global height
    global width
    # list of folders
    list_fold = []
    list_fold = os.listdir(dataset)

    print('... loading the data')

    # count all files and image size
    count = 0
    for folder in list_fold:
        if os.path.isdir(os.path.join(dataset, folder)):
            list_files = os.listdir(os.path.join(dataset, folder))
            for file in list_files:
                if file.endswith("jpg"):
                    count += 1
                    if count == 1:
                        width, height = get_image_size(os.path.join(dataset))
                    resize_data((os.path.join(dataset, folder, file)), width, height)

    # initialize array of images
    arr_data = numpy.zeros(shape=(count, (width * height)))

    # initialize array of labels
    arr_labels = numpy.zeros(shape=count)
    i = 0
    for folder in list_fold:
        if os.path.isdir(os.path.join(dataset, folder)):
            list_files = os.listdir(os.path.join(dataset, folder))
            for file in list_files:
                if file.endswith("jpg"):
                    pic_mat = numpy.asarray(Image.open(os.path.join(dataset, folder, file)).convert('L'))
                    pic_string = pic_mat.ravel()
                    arr_data[i] = pic_string
                    arr_labels[i] = float(folder)
                    i = i + 1

                    # stick images to labels
    arr_nonrand = numpy.c_[arr_data, arr_labels]

    # randomize strings
    arr_rand = numpy.random.permutation(arr_nonrand)

    # unstick images from labels
    # MIGHT BE PROBLEMS WITH DATA TYPE(FLOAT)!!!!!!!!!!!!!!!!!!!!!!!
    image_arr = numpy.asarray([col[:-1] for col in arr_rand])
    label_arr = numpy.asarray([col[-1] for col in arr_rand])

    # creating for training, valid and test
    train_var = 0.7
    valid_var = 0.2
    test_var = 0.1

    # define sizes of training, valid and test - images
    num_train_rows = int(round(train_var * count, 0))
    num_valid_rows = int(round(valid_var * count, 0))
    num_test_rows = int(count - num_train_rows - num_valid_rows)

    # define sizes of training, valid and test - labels
    num_valid_lab = int(round(valid_var * count, 0))
    num_train_lab = int(round(train_var * count, 0))
    num_test_lab = int(count - num_train_lab - num_valid_lab)

    # define rows of training, valid and test - images
    train_rows = image_arr[0:num_train_rows][:]
    valid_rows = image_arr[0:num_valid_rows][:]
    test_rows = image_arr[0:num_test_rows][:]

    # define labels of training, valid and test
    train_lab = label_arr[0:num_train_lab][:]
    valid_lab = label_arr[0:num_valid_lab][:]
    test_lab = label_arr[0:num_test_lab][:]

    # creating tuples of training, valid and test
    train_set = (train_rows, train_lab)
    valid_set = (valid_rows, valid_lab)
    test_set = (test_rows, test_lab)

    def shared_dataset(data_xy, borrow=True):
        # This function performs the optimization for GPU
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=100,
                           dataset='/Users/Aleksei/Desktop/testing_original',
                           batch_size=45):
    # perform gradient descent
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD THE MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    classifier = LogisticRegression(input=x, n_in=200 * 250, n_out=2)

    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000
    patience_increase = 2
    # found
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


def predict():


    # load the saved model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset = '/Users/Aleksei/Desktop/testing_original'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)


if __name__ == '__main__':
    sgd_optimization_mnist()
