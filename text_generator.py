import math
import os
import pickle
import sys
from collections import OrderedDict

import numpy as np
import theano
from theano import tensor

from data_manipulation import get_text, prepare_data


def save_params(tparams, fp='params.pkl'):
    od = OrderedDict()
    for k, v in tparams.items():
        od[k] = v.get_value()
    with open(fp, 'wb') as f:
        pickle.dump(od, f)


def load_params(fp='params.pkl'):
    with open(fp, 'rb') as f:
        params = pickle.load(f)
    return params


def ortho_weight(ndim):
    """
    ...
    :param ndim:
    :return: array (ndim, ndim)
    """
    # TODO(jav): investigate how the ortho_weight function works theoretically
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


def init_params(options):
    """
    Set up paramaters connecting all layers
    :param options: User defined options dictionary
    :return: OrderedDict of paramater values
    """
    params = OrderedDict()

    lstm_layer_sizes = [options['n_in']] + options['n_hidden']

    for idx, (n_in, n_out) in enumerate(zip(lstm_layer_sizes[:-1], lstm_layer_sizes[1:])):
        W = 0.01 * np.random.rand(n_in, n_out).astype(theano.config.floatX)
        params['W' + str(idx)] = W

        W = np.concatenate([ortho_weight(n_out) for _ in range(4)], axis=1)
        params['lstm_W' + str(idx)] = W

        U = np.concatenate([ortho_weight(n_out) for _ in range(4)], axis=1)
        params['lstm_U' + str(idx)] = U

        b = np.zeros(4 * n_out).astype(theano.config.floatX)
        params['lstm_b' + str(idx)] = b

    params['U'] = 0.01 * np.random.rand(options['n_hidden'][-1], options['n_out']).astype(theano.config.floatX)
    params['b'] = np.zeros(options['n_out']).astype(theano.config.floatX)

    # Each W has dimensionality (n_in)
    # lstm_W has dimensionality (n_hidden, n_hidden*4)
    # lstm_U has dimensionality (n_hidden, n_hidden*4)
    # lstm_b has dimensionality (n_hidden*4)
    # U has dimensionality (n_hidden, n_out)
    # b has dimensionality (n_out)

    return params


def init_theano_params(params):
    """
    Turn numpy array values into symbolic theano shared objects
    :param params: Values
    :return: OrderedDict of symbolic parameters
    """

    theano_params = OrderedDict()

    for name, value in params.items():
        theano_params[name] = theano.shared(value, name)

    return theano_params


def lstm_layer(tparams, state_below, idx=0):
    """
    Representative of an lstm layer
    :param tparams: shared variable theano params (OrderedDictionary)
    :param state_below: weight tensor of dimensionality (n_timesteps, n_sequences, n_hidden)
    :return: output of neurons in hidden layer
    """

    n_sequences = state_below.shape[1]
    n_hidden = state_below.shape[2]

    def _step(x, h_, c_):

        # preact has dimensionality (n_sequences, n_hidden * 4)
        preact = theano.dot(h_, tparams['lstm_U' + str(idx)]) + x

        gate_input = theano.tensor.nnet.sigmoid(preact[:, : n_hidden])
        gate_forget = theano.tensor.nnet.sigmoid(preact[:, n_hidden: 2 * n_hidden])
        gate_output = theano.tensor.nnet.sigmoid(preact[:, 2 * n_hidden: 3 * n_hidden])
        c_candidate = theano.tensor.tanh(preact[:, 3 * n_hidden: 4 * n_hidden])

        c = gate_input * c_candidate + gate_forget * c_

        h = gate_output * theano.tensor.tanh(c)

        return h, c

    state_below = theano.dot(state_below, tparams['lstm_W' + str(idx)]) + tparams['lstm_b' + str(idx)]

    ret, updates = theano.scan(
        _step,
        sequences=[state_below],
        outputs_info=[
            theano.tensor.alloc(np.array(0, dtype=theano.config.floatX), n_sequences, n_hidden),
            theano.tensor.alloc(np.array(0, dtype=theano.config.floatX), n_sequences, n_hidden)
        ],
        name='lstm_layer' + str(idx)
    )

    return ret


def build_model(tparams, options):
    """
    Builds model with one LSTM layer
    :param tparams: Theano parameters
    :param options: Model options
    :return:
    """

    # x_train has dimensionality (n_timesteps, n_sequences, n_out)
    x_train = tensor.tensor3('x_train', dtype=theano.config.floatX)
    # mask_train has dimensionality (n_timesteps, n_sequences)
    mask_train = tensor.matrix('mask_train', dtype=theano.config.floatX)
    # y_train has dimensionality (n_timesteps, n_sequences)
    y_train = tensor.lmatrix('y_train')

    trng = tensor.shared_randomstreams.RandomStreams()
    dropout_noise_swtich = theano.shared(np.array(0, dtype=theano.config.floatX))

    curr_layer = x_train
    n_timesteps, n_sequences, n_out = curr_layer.shape

    # TODO: Turn this into a scan
    for idx in range(len(options['n_hidden'])):

        # get all the weights (as vectors) for each neuron in curr_layer
        # Wi has dimensionality (n_in, n_out) for moving from current to next layer
        W_relevant = tparams['W' + str(idx)][None, None, :, :] * curr_layer[:, :, :, None]
        # W_relevant has dimensionality (n_timesteps, n_sequences, n_in, n_out)
        W_relevant = W_relevant.sum(axis=2)
        # W_relevant has dimensionality (n_timesteps, n_sequences, n_out)

        # curr_layer and c have dimensionality (n_timesteps, n_sequences, n_out)
        curr_layer, c = lstm_layer(tparams, W_relevant, idx=idx)

    # mask_train has dimensionality (n_timesteps, n_sequences)
    # get rid of timesteps that weren't actually used
    curr_layer = curr_layer * mask_train[:, :, None]

    # Note curr_layer needs to be flattened since softmax cannot accept 3D array
    curr_layer_flattened = curr_layer.reshape([n_timesteps * n_sequences, options['n_hidden'][-1]])

    pred_flattened = theano.tensor.nnet.softmax(theano.dot(curr_layer_flattened, tparams['U']) + tparams['b'])

    likelihood = pred_flattened[theano.tensor.arange(0, n_timesteps * n_sequences), y_train.flatten()]
    # cull unused timesteps
    likelihood = likelihood[mask_train.flatten().nonzero()]

    neg_log_likelihood = -tensor.log(likelihood).mean()

    # GENERATION FUNCTIONS

    # Note: if generating, only one sequence is being used
    softmax_logit_last_char = pred_flattened[-1]
    # get char idx to seed generation
    p = softmax_logit_last_char

    f_generate = theano.function(
        inputs=[x_train, mask_train],
        outputs=p
    )

    return dropout_noise_swtich, x_train, mask_train, y_train, neg_log_likelihood, f_generate


def adadelta(tparams, x, mask, y, cost):
    """
    Does adadelta parameter updates

    :param tparams: Iterable of theano parameters
    :param x: Input of dimensionality (n_timesteps, n_sequences, n_out)
    :param mask: Binary mask
    :param y: Output
    :param cost: Cost expression
    :return: Function returning cost expression with adadelta defined updates
    """

    # Note: this differs from the lstm.py implementation b/c I can't figure out why theirs is so convoluted

    grads = tensor.grad(cost, list(tparams.values()))

    # Initialize the running tallies to 0 for RMS(grad2)
    running_param_deltas2 = [theano.shared(p.get_value() * np.array(0, dtype=theano.config.floatX),
                                           name='%s_run_param_deltas2' % k)
                             for k, p in tparams.items()]

    running_grads2 = [theano.shared(p.get_value() * np.array(0, dtype=theano.config.floatX),
                                    name='%s_run_grads2' % k)
                      for k, p in tparams.items()]

    running_grads2_new = [0.95 * run_grad2 + 0.05 * g ** 2 for run_grad2, g in zip(running_grads2, grads)]

    running_grads2_updates = [(run_grad2, run_grad2_new) for run_grad2, run_grad2_new in zip(running_grads2, running_grads2_new)]

    param_deltas = [-tensor.sqrt(run_param_delta2 + 1e-6) / tensor.sqrt(run_grad2_new + 1e-6) * grad
                    for (run_param_delta2, run_grad2_new, grad)
                    in zip(running_param_deltas2, running_grads2_new, grads)]

    running_param_deltas2_updates = [(run_param_delta2, 0.95 * run_param_delta2 + 0.05 * param_delta ** 2)
                                     for (run_param_delta2, param_delta)
                                     in zip(running_param_deltas2, param_deltas)]

    param_updates = [(param, param + update) for param, update in zip(tparams.values(), param_deltas)]

    f_update = theano.function(
        inputs=[x, mask, y],
        outputs=cost,
        updates=running_grads2_updates + running_param_deltas2_updates + param_updates,
    )

    return f_update


def get_kfold(text_array, minibatch_size, shuffle=False, n_folds=100, start_positions=None):
    """
    Get kfold iterator
    :param text_array: array of characters (encoded as integer positions)
    :param minibatch_size: chunk size to break into
    :param shuffle: randomly order the indices
    :param n_folds: number of folds to break data into
    :param start_positions: indices of the starts of the training examples
    :return: iterable of iterables (elements of these are indices)
    """

    roll_offset = np.random.randint(minibatch_size * n_folds)
    text_array = np.roll(text_array, roll_offset)

    if start_positions is None:
        res = np.array_split(text_array, n_folds)
        for fold in res:
            yield np.array_split(fold, minibatch_size)
        return

    # TODO: Figure out merits of the commented code vs. implemented code
    # end_indices = np.random.randint(1, 7, size=start_positions.size)
    # end_positions = np.array([start_positions[i + offset] if i + offset < len(start_positions) else None for i, offset in enumerate(end_indices)])

    offset = np.random.randint(150, 1000)
    end_positions = start_positions + offset

    n = len(start_positions)
    idx_array = np.arange(n)

    if shuffle:
        np.random.shuffle(idx_array)

    res = [
        text_array[start_pos: end_pos]
        for start_pos, end_pos in np.array(list(zip(start_positions, end_positions)))[idx_array]
        ]
    res = np.array(res)

    for i in range(math.ceil(n / minibatch_size)):
        yield res[i * minibatch_size : (i + 1) * minibatch_size]


def train_lstm(
        n_hidden=[128, 128, 128],
        params_save_path='shakespeare/params_128_128_128.pkl',
        model_save_path='shakespeare/model_128_128_128.pkl',
        decay_c=0.,
        max_epochs=5000,
        batch_size_train=16,
        n_predictions=100,
        train=True,
        build=True,
        generate=False,
):
    """
    Train the lstm

    :param n_hidden: Number hidden units in the LSTM layer
    :param params_save_path: Path to file to save parameters
    :param model_save_path: Path to file to save model
    :param decay_c: Multiplier for sum of squares of weight matrix entries
    :param max_epochs: Maximum number of epochs through the data to perform
    :param batch_size_train: Batch size during training
    :param n_predictions: Number of predictions to make after LSTM is seeded
    :param train: Boolean to indicate whether or not to train before generating
    :param build: Boolean to indicate whether or not to build model or load it
    :param generate: Boolean to indicate whether or not to generate new text
    :return:
    """

    model_options = locals().copy()

    print('loading data')
    text_array, start_indices, char_idx_dict, idx_char_dict = get_text(filepath='shakespeare/speare_preproc.txt', regex='[A-Z]+:\n')

    print('building model')

    model_options['n_in'] = len(char_idx_dict)
    model_options['n_out'] = model_options['n_in']

    if os.path.isfile(params_save_path):
        params = load_params(fp=params_save_path)
    else:
        params = init_params(model_options)

    tparams = init_theano_params(params)

    if build:
        # Cost returned is neg_log_likelihood
        use_dropout_noise, x, mask, y, cost, f_generate = build_model(tparams, model_options)
        if decay_c > 0:
            decay_c = theano.shared(np.array(decay_c, dtype=theano.config.floatX), name='decay_c')
            weight_decay = (tparams['U'] ** 2).sum()
            cost += decay_c * weight_decay
        f_update = adadelta(tparams, x, mask, y, cost)
        with open(model_save_path, 'wb') as f:
            pickle.dump([use_dropout_noise, x, mask, y, cost, f_generate, f_update], f)
    else:
        with open(model_save_path, 'rb') as f:
            use_dropout_noise, x, mask, y, cost, f_generate, f_update = pickle.load(f)

    if train:

        print('training')

        try:
            for idx_epoch in range(max_epochs):

                n_sequences_seen = 0
                cost_epoch = []
                for sequences in get_kfold(text_array, batch_size_train, shuffle=True):
                    n_sequences_seen += len(sequences)

                    x, mask, y = prepare_data(sequences, len(char_idx_dict))

                    cost_batch = f_update(x, mask, y)
                    print(cost_batch)
                    cost_epoch.append(cost_batch)

                print('Epoch {} saw {} sequences with avg train cost {}'.format(idx_epoch, n_sequences_seen, np.average(cost_epoch)))
        except KeyboardInterrupt:
            print('On epoch {}'.format(idx_epoch))

        save_params(tparams, fp=params_save_path)

    if generate:
        print('generating')
        seed_text = 'L'

        seed_idx = np.array(list(map(char_idx_dict.__getitem__, seed_text)))[:, None]
        idx1, idx0 = np.meshgrid(np.arange(seed_idx.shape[1]), np.arange(seed_idx.shape[0]))

        n_char_set = len(char_idx_dict)

        # seed has dimensionality (n_timesteps, 1, n_out)
        seed = np.zeros(seed_idx.shape + (n_char_set,), dtype=theano.config.floatX)
        seed[idx0, idx1, seed_idx] = 1

        for _ in range(n_predictions):
            mask = np.ones(seed.shape[:2], dtype=theano.config.floatX)
            res = f_generate(seed, mask)
            # res[-1] += 1 - res.sum()
            res = res.astype(np.float64)
            res /= res.sum()
            res_idx = np.random.choice(len(res), p=res)
            # res_idx = res[0]
            seed2 = np.zeros((1, 1, n_char_set), dtype=theano.config.floatX)
            seed2[0, 0, res_idx] = 1
            seed = np.concatenate((seed, seed2), axis=0)

        res_str = ''.join(map(idx_char_dict.__getitem__, map(np.argmax, seed)))
        print(res_str)


if __name__ == '__main__':

    theano.config.exception_verbosity = 'high'
    sys.setrecursionlimit(10000)
    train_lstm(n_predictions=100)
