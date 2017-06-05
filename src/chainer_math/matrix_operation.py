
from chainer import Variable, functions as F
import numpy as np

np.random.seed(46)


def tridiag_matmul(vec, diag_v, upper_v, lower_v):
    """
    :param vec: multiplied vector, shape is (batch size, dim)
    :param diag_v: diagonal elements in tridiagonal matrix
    :param upper_v: upper elements ( this equals to diag(mat, 1) )
    :param lower_v: lower elements ( this equals to diag(mat, -1) )
    :return: res
    """
    _batchsize = vec.shape[0]
    pad = Variable(np.zeros((_batchsize, 1), dtype=np.float32))
    first = vec * diag_v
    forward = vec[:, :-1]
    back = vec[:, 1:]
    second = F.concat([pad, forward * upper_v], axis=1)
    third = F.concat([back * lower_v, pad], axis=1)
    return first + second + third


def tridiag_matmul2(vec, diag_v, non_diag_v):
    raise NotImplementedError


# inserting zeros between every values
def insert_zeros(vec):
    batchsize, dim = vec.get_shape().as_list()
    _vec = tf.reshape(vec, [-1, 1, dim])
    pad = tf.zeros_like(_vec)
    # new_v = tf.reshape(tf.transpose(tf.concat([_vec, pad], axis=1), perm=(0, 2, 1)), [-1, 2*dim])[:, :-1]
    new_v = tf.reshape(tf.transpose(tf.concat([_vec, pad], axis=1), perm=(0, 2, 1)), [-1, 2 * dim])
    new_v, _ = tf.split(new_v, [2*dim-1, 1], 1)
    return new_v


def complex_lookup_initializer(n_word, dim):
    bound = np.sqrt(6) / np.sqrt(2*dim)
    vals = np.ndarray((n_word, dim), dtype=np.complex)
    vals.real = np.random.uniform(-bound, bound, (n_word, dim))
    vals.imag = np.random.uniform(-bound, bound, (n_word, dim))
    return tf.constant_initializer(vals, dtype=tf.complex64)


# def cons
