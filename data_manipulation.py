import numpy as np
import itertools
import re
import theano


def get_text(filepath='speare_preproc.txt', regex='.+\n'):
    """
    Load text from filepath
    :param filepath: filepath to load from
    :param regex: the regex on which to split into different training samples
    :return: (array (n_folds, n_chars), character to index dict, index to character dict)
    """
    with open(filepath) as f:
        t = f.read()

    # Note: the characters must be sorted since iterating over a set can have different orders (as set order is
    # determined by hashes of their elements, and a hash seed exists for each instance of the Python interpreter)
    char_idx = {char: i for i, char in enumerate(sorted(set(t), key=lambda c: ord(c)))}
    idx_char = {v: k for k, v in char_idx.items()}

    line_start_indices = np.array([m.start() for m in re.finditer(regex, t)])

    a = np.array(list(map(char_idx.__getitem__, t)))

    return a, line_start_indices, char_idx, idx_char


def prepare_data(text_array, num_chars):
    """
    Process inputs, mask, and outputs for each text array
    :param text_array: Array of arrays in which each subarray is an array of character indices
    :param num_chars: Number of characters to represent
    :return: (inputs (n_chars-1, n_text_arrays, n_unique_chars), mask (n_chars-1, n_text_arrays), outputs (n_chars-1, n_text_arrays))
    """

    x = []
    y = []

    for sequence in text_array:
        x.append(sequence[:-1])
        y.append(sequence[1:])

    # x_idxs has dimensionality (n_chars-1, n_text_arrays)
    x_idxs = np.array(list(itertools.zip_longest(*x, fillvalue=-1)))
    mask = x_idxs >= 0
    y = np.array(list(itertools.zip_longest(*y, fillvalue=-1)))

    x = np.zeros(x_idxs.shape + (num_chars,), dtype=theano.config.floatX)
    # TODO: use indexing='ij' for this
    idx1, idx0 = np.meshgrid(np.arange(x_idxs.shape[1]), np.arange(x_idxs.shape[0]))
    x[idx0, idx1, x_idxs] = 1

    return x, mask, y