import unittest
import numpy as np
import abc


class Matrix(object):
    def __init__(self, array=None):
        if isinstance(array, np.ndarray):
            self._matrix = array
            return

        if array == None:
            array = [[]]
        self._validate(array)
        self._matrix = np.array(array)

    def __call__(self, *args, **kwargs):
        return self._matrix

    def to_list(self):
        return list(self._matrix.tolist())

    def get_transposed(self):
        if self._matrix.size == 0:
            return Matrix()

        return Matrix(list(self._matrix.transpose().tolist()))

    def mul(self, matrix):
        if not isinstance(matrix, Matrix):
            raise ValueError('Argument should be instance of Matrix, %s given' % type(metrix))

        c = self() * matrix()
        return Matrix(c)
        

    def _validate(self, array):
        try:
            array = list(array)
        except Exception as e:
            raise ValueError('Invalid matrix value. Expected list got %s' % type(array))

        if len(array) < 1 or not isinstance(array[0], list):
            raise ValueError('Invalid matrix value. The list be two-dimensional')


if __name__ == '__main__':

    class TestMatrix(unittest.TestCase, metaclass=abc.ABCMeta):
        _1d = [[1, 2, 3]]
        _1dt = [[1], [2], [3]]
        _2d = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        _2dt = [
            [1, 4],
            [2, 5],
            [3, 6],
        ]


    class TestMatrixInit(TestMatrix):

        def test_matrix_init_without_args(self):
            m = Matrix()
            self.assertTrue(isinstance(m, Matrix))

        def test_matrix_init_as_vector(self):
            m = Matrix(self._1d)
            self.assertTrue(isinstance(m, Matrix))

        def test_matrix_init_as_two_dimensional(self):
            m = Matrix(self._2d)
            self.assertTrue(isinstance(m, Matrix))


    class TestMatrixValue(TestMatrix):

        def test_matrix_empty_value(self):
            m = Matrix()
            self.assertEqual(m.to_list(), [[]])

        def test_matrix_vector_value(self):
            m = Matrix(self._1d)
            self.assertEqual(m.to_list(), self._1d)

        def test_two_dimensional_matrix_value(self):
            m = Matrix(self._2d)
            self.assertEqual(m.to_list(), self._2d)


    class TestMatrixValidation(TestMatrix):

        def test_matrix_empty_validation_success(self):
            Matrix()

        def test_matrix_vector_validation_success(self):
            Matrix(self._1d)

        def test_two_dimensional_matrix_validation_success(self):
            Matrix(self._2d)

        def test_empty_one_dimensional_list(self):
            self.assertRaises(ValueError, Matrix, [])


    class TestMatrixTranspose(TestMatrix):
        def test_empty_matrix_transpose(self):
            m = Matrix()
            self.assertEqual(m.get_transposed().to_list(), [[]])

        def test_vector_matrix_transpose(self):
            m = Matrix(self._1d)
            self.assertEqual(m.get_transposed().to_list(), self._1dt)

        def test_two_dimensional_matrix_transpose(self):
            m = Matrix(self._2d)
            self.assertEqual(m.get_transposed().to_list(), self._2dt)


    class TestMatrixMultiplication(TestMatrix):
        def test_empty_matrix_multiplication(self):
            a = Matrix()
            b = Matrix()
            self.assertEqual(a.mul(b).to_list(), [[]])

        def test_vector_matrix_multiplication(self):
            a = Matrix(self._1d)
            b = Matrix(self._1dt)
            self.assertEqual(
            a.mul(b).to_list(),
            [[1, 2, 3],
             [2, 4, 6],
             [3, 6, 9]]
            )

        def test_two_dimensional_matrix_multiplication_a_mul_a(self):
            a = Matrix(self._2d)
            self.assertEqual(
            a.mul(a).to_list(),
            [[ 1,  4,  9],
             [16, 25, 36]]
            )

        def test_two_dimensional_matrix_multiplication_at_mul_at(self):
            a = Matrix(self._2dt)
            self.assertEqual(
            a.mul(a).to_list(),
            [[ 1, 16],
             [ 4, 25],
             [ 9, 36]]
            )

        def test_two_dimensional_matrix_multiplication_a_mul_at(self):
            a = Matrix(self._2d)
            b = Matrix(self._2dt)
            self.assertRaises(ValueError, a.mul, b)

    """
    Run tests...
    """
    unittest.main()
