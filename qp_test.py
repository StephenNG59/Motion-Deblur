from numpy import array, dot, identity, zeros, float32
from qpsolvers import solve_qp
import cv2 as cv

# M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
M = identity(3)
P = dot(M.T, M)  # quick way to build a symmetric matrix
q = dot(array([3., 2., 3.]), M)
G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = array([3., 2., -2.]).reshape((3,))
A = array([1., 1., 1.])
b = array([1.])
xx = zeros((3, 2))

M = array([[1., 1., 1.], [1., 0., 0.], [0., 0., 1.]])
P = dot(M.T, M)
print(P)

print(M.T)
print(q)
print(dot(M.T, q))

print("QP solution:", solve_qp(P, q, G, h, A, b))

