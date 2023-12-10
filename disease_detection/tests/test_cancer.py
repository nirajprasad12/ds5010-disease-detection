
import unittest
import numpy as np
from disease_detection import cancer


class TestCancer(unittest.TestCase):

    def test_initialization_fail(self):
        with self.assertRaises(Exception) as context:
            g = cancer.cancer(None)
        self.assertEqual(str(context.exception), "Input cannot be None")

        with self.assertRaises(Exception) as context:
            g = cancer([[1, 2, 3]])
        self.assertEqual(str(context.exception), "Input array length must be 30 for all rows")

        with self.assertRaises(Exception) as context:
            g = cancer([[1, 'a', 3]*30])
        self.assertEqual(str(context.exception), "All elements in the input array must be numeric")

    def test_LogisticRegression(self):
        g = cancer(np.array([[1.0]*30, [2.0]*30]))
        result = g.LogisticRegression()
        self.assertEqual(len(result), 2)

    def test_KNearestNeighbours(self):
        g = cancer(np.array([[1.0]*30, [2.0]*30]))
        result = g.KNearestNeighbours()
        self.assertEqual(len(result), 2)

    def test_SupportVectorClassifier(self):
        g = cancer(np.array([[1.0]*30, [2.0]*30]))
        result = g.SupportVectorClassifier()
        self.assertEqual(len(result), 2)

    def test_GNB(self):
        g = cancer(np.array([[1.0]*30, [2.0]*30]))
        result = g.GNB()
        self.assertEqual(len(result), 2)

    def test_RandomForest(self):
        g = cancer(np.array([[1.0]*30, [2.0]*30]))
        result = g.RandomForest()
        self.assertEqual(len(result), 2)

if __name__ == '__main__':
    unittest.main()
