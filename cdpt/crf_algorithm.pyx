#!/usr/bin/env python
# encoding: utf-8

cimport cython

from libc.math cimport exp, log
import numpy as np
cimport numpy as np
np.import_array()

from cpython cimport bool

cdef np.float64_t NEGINF = -np.inf
cdef np.float64_t INF = np.inf
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline np.float64_t float64_max(np.float64_t a, np.float64_t b): return a if a >= b else b
cdef inline np.float64_t float64_min(np.float64_t a, np.float64_t b): return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double logsumexp(double[:] vec, int length):
    cdef double m, ret
    cdef int i
    ret = 0.
    m = NEGINF
    for i in range(length):
        if vec[i] > m:
            m = vec[i]
    for i in range(length):
        ret += exp(vec[i] - m)
    return m + log(ret)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.npy_intp, ndim=2] crf_marginal(double[:, :] scores, double[:, :] transitions):
    cdef int length, nl, i, t, t0
    length, nl = scores.shape[0], scores.shape[1]

    cdef double emission_score, trans_score, s, partition
    cdef double [:] tmp_var = np.zeros((nl,))
    cdef double [:] forward_var = np.zeros((nl,)) + NEGINF
    cdef double [:] backward_var = np.zeros((nl,)) + NEGINF
    # START TAG
    forward_var[nl - 2] = 0.
    backward_var[nl - 1] = 0.
    cdef double [:, :] forward_scores = np.zeros((length, nl))
    cdef double [:, :] backward_scores = np.zeros((length, nl))
    cdef double [:, :] final_scores = np.zeros((length, nl))

    for i in range(length):
        for t in range(nl):
            emission_score = scores[i, t]

            for t0 in range(nl):
                trans_score = transitions[t0, t]
                s = forward_var[t0] + emission_score + trans_score
                tmp_var[t0] = s

            forward_scores[i, t] = logsumexp(tmp_var, nl)

        for t in range(nl):
            forward_var[t] = forward_scores[i, t]

    for t0 in range(nl):
        trans_score = transitions[t0, nl - 1]
        s = forward_var[t0] + trans_score
        tmp_var[t0] = s

    partition = logsumexp(tmp_var, nl)

    for i in range(length - 1, -1, -1):
        for t in range(nl):
            emission_score = scores[i, t]

            for t0 in range(nl):
                trans_score = transitions[t, t0]
                s = backward_var[t0] + emission_score + trans_score
                tmp_var[t0] = s

            backward_scores[i, t] = logsumexp(tmp_var, nl)

        for t in range(nl):
            backward_var[t] = backward_scores[i, t]

    for i in range(length):
        for t in range(nl):
            final_scores[i, t] = forward_scores[i, t] + backward_scores[i, t] - scores[i, t] - partition

    return np.array(final_scores)
