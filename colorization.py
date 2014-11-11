import os
import sys

import numpy as np
import scipy
from scipy import io, misc, sparse
from scipy.linalg import solve_banded
from scipy.ndimage.filters import uniform_filter
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import normalize

import cartesian
import color_conv


def neighborhood(X, r):
    d = 1
    l1, h1 = max(r[0]-d, 0), min(r[0]+d+1, X.shape[0])
    l2, h2 = max(r[1]-d, 0), min(r[1]+d+1, X.shape[1])
    return X[l1:h1, l2:h2]


def weight(r, V, Y, S):
    return [np.exp(-1 * np.square(Y[r] - Y[v]) / S[r]) if S[r] > 0.0 else 0.0 for v in V]


def find_marked_locations(bw, marked):
    diff = marked - bw
    colored = [set(zip(*np.nonzero(diff[:, :, i]))) for i in [1, 2]]
    return colored[0].union(colored[1])


def std_matrix(A):
    S = np.empty_like(A)
    for i in xrange(A.shape[0]):
        for j in xrange(A.shape[1]):
            S[i, j] = np.square(np.std(neighborhood(A, [i, j])))
    return S


def build_weights_matrix(Y):
    (n, m) = [Y.shape[0], Y.shape[1]]
    S = std_matrix(Y)
    size = n * m
    cart = cartesian.cartesian([xrange(n), xrange(m)])
    cart_r = cart.reshape(n, m, 2)
    xy2idx = np.arange(size).reshape(n, m)  # [x,y] -> index
    W = sparse.lil_matrix((size, size))
    for i in xrange(Y.shape[0]):
        for j in xrange(Y.shape[1]):
            idx = xy2idx[i, j]
            N = neighborhood(cart_r, [i, j]).reshape(-1, 2)
            N = [tuple(neighbor) for neighbor in N]
            N.remove((i, j))
            p_idx = [xy2idx[xy] for xy in N]
            weights = weight((i, j), N, Y, S)
            W[idx, p_idx] = -1 * np.asmatrix(weights)

    Wn = normalize(W, norm='l1', axis=1)
    Wn[np.arange(size), np.arange(size)] = 1

    return Wn


def run():
    extension = 'bmp'
    pic = sys.argv[1] if len(sys.argv) > 1 else 'smiley'
    name = 'samples/' + pic
    bw_filename = name + '.' + extension
    marked_filename = name + '_marked.' + extension
    out_filename = name + '_out.' + extension

    bw_rgb = misc.imread(bw_filename)
    marked_rgb = misc.imread(marked_filename)

    bw = color_conv.rgb2yiq(bw_rgb)
    marked = color_conv.rgb2yiq(marked_rgb)

    Y = np.array(bw[:, :, 0], dtype='float64')

    (n, m) = np.shape(bw)[0:2]  # extract image dimensions
    size = n * m
    Wn = 0
    if (os.path.isfile(pic + '.mtx')):
        Wn = scipy.io.mmread(pic).tocsr()
    else:
        Wn = build_weights_matrix(Y)
        scipy.io.mmwrite(pic, Wn)

    ## once markes are found
    colored = find_marked_locations(bw_rgb, marked_rgb)

    ## set rows in colored indices
    Wn = Wn.tolil()
    xy2idx = np.arange(size).reshape(n, m)  # [x,y] -> index
    for idx in [xy2idx[i, j] for [i, j] in colored]:
        Wn[idx] = sparse.csr_matrix(([1.0], ([0], [idx])), shape=(1, size))

    LU = scipy.sparse.linalg.splu(Wn.tocsc())

    b1 = (marked[:, :, 1]).flatten()
    b2 = (marked[:, :, 2]).flatten()

    x1 = LU.solve(b1)
    x2 = LU.solve(b2)

    sol = np.zeros(np.shape(bw_rgb))
    sol[:, :, 0] = Y
    sol[:, :, 1] = x1.reshape((n, m))
    sol[:, :, 2] = x2.reshape((n, m))
    sol_rgb = color_conv.yiq2rgb(sol)

    misc.imsave(out_filename, sol_rgb)


if __name__ == '__main__':
    run()
