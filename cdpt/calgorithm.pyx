#!/usr/bin/env python
# encoding: utf-8

cimport cython

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
cpdef np.ndarray[np.npy_intp, ndim=1] parse_proj(np.ndarray[np.float64_t, ndim=2] scores):
    cdef int nr, nc, N, i, k, s, t, r, maxidx
    cdef np.float64_t tmp, cand
    cdef np.ndarray[np.float64_t, ndim=2] complete_0
    cdef np.ndarray[np.float64_t, ndim=2] complete_1
    cdef np.ndarray[np.float64_t, ndim=2] incomplete_0
    cdef np.ndarray[np.float64_t, ndim=2] incomplete_1
    cdef np.ndarray[np.npy_intp, ndim=3] complete_backtrack
    cdef np.ndarray[np.npy_intp, ndim=3] incomplete_backtrack
    cdef np.ndarray[np.npy_intp, ndim=1] heads

    nr, nc = np.shape(scores)

    N = nr - 1 # Number of words (excluding root).

    complete_0 = np.zeros((nr, nr)) # s, t, direction (right=1).
    complete_1 = np.zeros((nr, nr)) # s, t, direction (right=1).
    incomplete_0 = np.zeros((nr, nr)) # s, t, direction (right=1).
    incomplete_1 = np.zeros((nr, nr)) # s, t, direction (right=1).

    complete_backtrack = -np.ones((nr, nr, 2), dtype=int) # s, t, direction (right=1).
    incomplete_backtrack = -np.ones((nr, nr, 2), dtype=int) # s, t, direction (right=1).

    for i in range(nr):
        incomplete_0[i, 0] = NEGINF

    for k in range(1, nr):
        for s in range(nr - k):
            t = s + k
            tmp = NEGINF
            maxidx = s
            for r in range(s, t):
                cand = complete_1[s, r] + complete_0[r+1, t]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
                if s == 0 and r == 0:
                    break
            incomplete_0[t, s] = tmp + scores[t, s]
            incomplete_1[s, t] = tmp + scores[s, t]
            incomplete_backtrack[s, t, 0] = maxidx
            incomplete_backtrack[s, t, 1] = maxidx

            tmp = NEGINF
            maxidx = s
            for r in range(s, t):
                cand = complete_0[s, r] + incomplete_0[t, r]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            complete_0[s, t] = tmp
            complete_backtrack[s, t, 0] = maxidx

            tmp = NEGINF
            maxidx = s + 1
            for r in range(s+1, t+1):
                cand = incomplete_1[s, r] + complete_1[r, t]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            complete_1[s, t] = tmp
            complete_backtrack[s, t, 1] = maxidx

    heads = -np.ones(N + 1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    return heads


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack_eisner(np.ndarray[np.npy_intp, ndim=3] incomplete_backtrack,
        np.ndarray[np.npy_intp, ndim=3]complete_backtrack,
        int s, int t, int direction, int complete, np.ndarray[np.npy_intp, ndim=1] heads):
    cdef int r
    if s == t:
        return
    if complete:
        r = complete_backtrack[s, t, direction]
        if direction:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
    else:
        r = incomplete_backtrack[s, t, direction]
        if direction:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return
        else:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_mwe(np.ndarray[np.float64_t, ndim=2] scores,
              np.ndarray[np.float64_t, ndim=3] relscores,
              np.ndarray[np.float64_t, ndim=2] bio,
              int mwe_rel):
    cdef int n, _, nr, i, j, k, d, s, t, r
    cdef np.float64_t cand
    cdef np.ndarray[np.float64_t, ndim=3] tri
    cdef np.ndarray[np.npy_intp, ndim=2] tri_back
    cdef np.ndarray[np.float64_t, ndim=2] tra
    cdef np.ndarray[np.npy_intp, ndim=2] tra_back
    cdef np.ndarray[np.npy_intp, ndim=1] heads
    cdef np.ndarray[np.npy_intp, ndim=1] rels

    cdef np.ndarray[np.npy_intp, ndim=2] best_rel

    n, _ = np.shape(scores)

    tri = np.zeros((2, n, n)) + NEGINF
    tri_back = np.zeros((n, n), dtype=int) - 1
    tra = np.zeros((n, n)) + NEGINF
    tra_back = np.zeros((n, n), dtype=int)

    heads = np.zeros((n,), dtype=int) - 1
    rels = np.zeros((n,), dtype=int) - 1


    for i in range(n):
        tri[1, i, i] = 0.
        tri[0, i, i] = 0.
        if bio[i, 1] - bio[i, 3] > 0:
            tri[1, i, i] = bio[i, 1] - bio[i, 3]
    for i in range(1, n):
        for j in range(i + 1, n):
            tri[1, i, j] = bio[i, 1] - bio[i, 3]
            for k in range(i + 1, j + 1):
                tri[1, i, j] += bio[k, 2] - bio[k, 3] + scores[i, k] + relscores[mwe_rel, i, k]

    relscores[mwe_rel, :, :] += NEGINF
    best_rel = np.argmax(relscores, 0)

    for d in range(1, n):
        for s in range(n - d):
            t = s + d
            for r in range(s, t):
                cand = tri[1, s, r] + tri[0, t, r+1] + scores[s, t] + relscores[best_rel[s, t], s, t]
                if cand > tra[s, t]:
                    tra[s, t] = cand
                    tra_back[s, t] = r
                if s == 0 and r == 0:
                    break

            for r in range(s, t):
                cand = tri[1, s, r] + tri[0, t, r+1] + scores[t, s] + relscores[best_rel[t, s], t, s]
                if cand > tra[t, s]:
                    tra[t, s] = cand
                    tra_back[t, s] = r
                if s == 0 and r == 0:
                    break

            for r in range(s, t):
                cand = tri[0, r, s] + tra[t, r]
                if cand > tri[0, t, s]:
                    tri[0, t, s] = cand
                    tri_back[t, s] = r

            for r in range(s + 1, t + 1):
                cand = tra[s, r] + tri[1, r, t]
                if cand > tri[1, s, t]:
                    tri[1, s, t] = cand
                    tri_back[s, t] = r

    backtrack_mwe(tri_back, tra_back, 1, 0, n - 1, heads, rels, mwe_rel)

    for i in range(1, n):
        if rels[i] < 0:
            rels[i] = best_rel[heads[i], i]

    return heads, rels

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack_mwe(np.ndarray[np.npy_intp, ndim=2] tri_back, np.ndarray[np.npy_intp, ndim=2] tra_back,
                         int tri, int i, int j,
                        np.ndarray[np.npy_intp, ndim=1] heads, np.ndarray[np.npy_intp, ndim=1] rels, mwe_rel):
    cdef int r
    if i == j:
        return
    if tri:
        r = tri_back[i, j]
        if r == -1:
            for r in range(i + 1, j + 1):
                heads[r] = i
                rels[r] = mwe_rel
        else:
            backtrack_mwe(tri_back, tra_back, 0, i, r, heads, rels, mwe_rel)
            backtrack_mwe(tri_back, tra_back, 1, r, j, heads, rels, mwe_rel)
    else:
        r = tra_back[i, j]
        heads[j] = i
        if i < j:
            backtrack_mwe(tri_back, tra_back, 1, i, r, heads, rels, mwe_rel)
            backtrack_mwe(tri_back, tra_back, 1, j, r+1, heads, rels, mwe_rel)
        else:
            backtrack_mwe(tri_back, tra_back, 1, i, r+1, heads, rels, mwe_rel)
            backtrack_mwe(tri_back, tra_back, 1, j, r, heads, rels, mwe_rel)


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_mwe_ud(np.ndarray[np.float64_t, ndim=2] scores,
              np.ndarray[np.float64_t, ndim=3] relscores_,
              np.ndarray[np.float64_t, ndim=2] bio,
              int mwe_rel, int punct_rel):
    cdef int n, _, nr, i, j, k, d, s, t, r
    cdef np.float64_t cand
    cdef np.ndarray[np.float64_t, ndim=3] relscores = np.copy(relscores_)
    cdef np.ndarray[np.float64_t, ndim=3] tri
    cdef np.ndarray[np.npy_intp, ndim=2] tri_back
    cdef np.ndarray[np.float64_t, ndim=2] tra
    cdef np.ndarray[np.npy_intp, ndim=2] tra_back
    cdef np.ndarray[np.npy_intp, ndim=1] heads
    cdef np.ndarray[np.npy_intp, ndim=1] rels

    cdef np.ndarray[np.npy_intp, ndim=2] best_rel

    n, _ = np.shape(scores)

    tri = np.zeros((2, n, n)) + NEGINF
    tri_back = np.zeros((n, n), dtype=int) - 1
    tra = np.zeros((n, n)) + NEGINF
    tra_back = np.zeros((n, n), dtype=int)

    heads = np.zeros((n,), dtype=int) - 1
    rels = np.zeros((n,), dtype=int) - 1


    for i in range(n):
        tri[1, i, i] = 0.
        tri[0, i, i] = 0.
    for i in range(1, n):
        for j in range(i + 1, n):
            tri[1, i, j] = bio[i, 1] - bio[i, 3]
            for k in range(i + 1, j):
                tri[1, i, j] += bio[k, 2] - bio[k, 3] + scores[i, k] + float64_max(relscores[mwe_rel, i, k], relscores[punct_rel, i, k])
                #  tri[1, i, j] += bio[k, 2] - bio[k, 3] + scores[i, k]
            tri[1, i, j] += bio[j, 2] - bio[j, 3] + scores[i, j] + relscores[mwe_rel, i, j]
            #  tri[1, i, j] += bio[j, 2] - bio[j, 3] + scores[i, j]

    relscores[mwe_rel, :, :] += NEGINF
    best_rel = np.argmax(relscores, 0)

    for d in range(1, n):
        for s in range(n - d):
            t = s + d
            for r in range(s, t):
                cand = tri[1, s, r] + tri[0, t, r+1] + scores[s, t] + relscores[best_rel[s, t], s, t]
                #  cand = tri[1, s, r] + tri[0, t, r+1] + scores[s, t]
                if cand > tra[s, t]:
                    tra[s, t] = cand
                    tra_back[s, t] = r
                if s == 0 and r == 0:
                    break

            for r in range(s, t):
                cand = tri[1, s, r] + tri[0, t, r+1] + scores[t, s] + relscores[best_rel[t, s], t, s]
                #  cand = tri[1, s, r] + tri[0, t, r+1] + scores[t, s]
                if cand > tra[t, s]:
                    tra[t, s] = cand
                    tra_back[t, s] = r
                if s == 0 and r == 0:
                    break

            for r in range(s, t):
                cand = tri[0, r, s] + tra[t, r]
                if cand > tri[0, t, s]:
                    tri[0, t, s] = cand
                    tri_back[t, s] = r

            for r in range(s + 1, t + 1):
                cand = tra[s, r] + tri[1, r, t]
                if cand > tri[1, s, t]:
                    tri[1, s, t] = cand
                    tri_back[s, t] = r

    backtrack_mwe_flat(tri_back, tra_back, 1, 0, n - 1, heads, rels, mwe_rel, -2)

    for i in range(1, n):
        if rels[i] == -2:
            if relscores_[mwe_rel, heads[i], i] > relscores_[punct_rel, heads[i], i]:
                rels[i] = mwe_rel
            else:
                rels[i] = punct_rel
        elif rels[i] < 0:
            rels[i] = best_rel[heads[i], i]

    return heads, rels


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack_mwe_flat(np.ndarray[np.npy_intp, ndim=2] tri_back, np.ndarray[np.npy_intp, ndim=2] tra_back,
                         int tri, int i, int j,
                        np.ndarray[np.npy_intp, ndim=1] heads, np.ndarray[np.npy_intp, ndim=1] rels, int mwe_rel, int punct_rel):
    cdef int r
    if i == j:
        return
    if tri:
        r = tri_back[i, j]
        if r == -1:
            for r in range(i + 1, j + 1):
                heads[r] = i
                if r == j:
                    rels[r] = mwe_rel
                else:
                    rels[r] = punct_rel
        else:
            backtrack_mwe_flat(tri_back, tra_back, 0, i, r, heads, rels, mwe_rel, punct_rel)
            backtrack_mwe_flat(tri_back, tra_back, 1, r, j, heads, rels, mwe_rel, punct_rel)
    else:
        r = tra_back[i, j]
        heads[j] = i
        if i < j:
            backtrack_mwe_flat(tri_back, tra_back, 1, i, r, heads, rels, mwe_rel, punct_rel)
            backtrack_mwe_flat(tri_back, tra_back, 1, j, r+1, heads, rels, mwe_rel, punct_rel)
        else:
            backtrack_mwe_flat(tri_back, tra_back, 1, i, r+1, heads, rels, mwe_rel, punct_rel)
            backtrack_mwe_flat(tri_back, tra_back, 1, j, r, heads, rels, mwe_rel, punct_rel)
