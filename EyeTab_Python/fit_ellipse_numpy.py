import numpy as np

from math import degrees
from numpy.linalg import eig, inv

#### ADAPTED FROM
#### http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

def fit_ellipse_get_coeffs(x,y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    mat = V[:, n]
    
    return get_coeffs(mat), get_rotated_rect(mat)

def fitEllipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    mat = V[:, n]
    
    # make it look like OpenCV rotated rect
    return get_rotated_rect(mat)

def get_coeffs(mat):
    b, c, d, f, g, a = mat[1] / 2, mat[2], mat[3] / 2, mat[4] / 2, mat[5], mat[0]
    return a, 2*b, c, 2*d, 2*f, g

def get_rotated_rect(mat):
    return ellipse_center(mat), ellipse_axis_length(mat), degrees(ellipse_angle_of_rotation(mat)) + 90

def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return (x0, y0)

def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return 0.5 * np.arctan(2 * b / (a - c))

def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    min_axis = min(res1, res2) * 2
    maj_axis = max(res1, res2) * 2
    return (min_axis, maj_axis)
