import numpy
from numba import cuda, vectorize, float32

DIFFERENT_LENGTH_MESSAGE = '''
The length of the vectors differ
len(A) = {}
len(B) = {}
'''

@cuda.reduce
def sum(a, b):
    return a + b

@vectorize([float32(float32, float32)])
def product(a, b):
    return a * b

def dot(A, B):
    assert len(A) == len(B), DIFFERENT_LENGTH_MESSAGE.format(len(A), len(B))
    C = numpy.zeros(len(A), dtype=numpy.float32)
    C = product(A, B)
    return sum(C)

if __name__ == '__main__':
    N = 10 ** 6
    A = numpy.ones(N, dtype=numpy.float32)
    B = numpy.ones(N - 1, dtype=numpy.float32)
    print('Expected {} got {}'.format(N, dot(A, B)))
