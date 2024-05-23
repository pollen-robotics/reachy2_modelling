import numpy as np


def svals(J):
    u, s, v = np.linalg.svd(J)
    return s


def manip(J, eps=1e-6):
    det = np.linalg.det(J.T @ J)
    if det < 0 and det > -eps:
        det = 0
    return np.sqrt(det)
