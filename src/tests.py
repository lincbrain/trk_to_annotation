import unittest
import math
import numpy as np

from utils import load_from_file, split_along_grid


class TestConvertMethods(unittest.TestCase):
    def testLength(self):
        streamline_start, streamline_end, streamline_tract, streamline_scalars, scalar_keys, lb, ub = load_from_file()
        points, streamline_tract, streamline_scalars = split_along_grid(
            streamline_start, streamline_end, streamline_tract, streamline_scalars, lb, ub, [32])
        startingDist = 0.0
        for i in range(len(streamline_start)):
            diff = streamline_end[i] - streamline_start[i]
            startingDist += math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
        endDist = 0.0
        for point in points:
            diff = point[1] - point[0]
            endDist += math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
        self.assertEqual(round(endDist, 2), round(startingDist, 2))

    def testSplit1(self):
        streamline_start = np.array([[0.2, 0.3, 0.4]])
        streamline_end = np.array([[0.6, 0.3, 0.4]])
        streamline_tract = np.array([0])
        points, streamline_tract, streamline_scalars = split_along_grid(streamline_start, streamline_end, streamline_tract, np.zeros(
            (1, 0)), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), [2])
        self.assertEqual(points.shape[0], 2)
        self.assertEqual(points[0][0][0], 0.2)
        self.assertEqual(points[1][1][0], 0.6)
        self.assertEqual(points[0][1][0], 0.5)
        self.assertEqual(points[1][0][0], 0.5)

    def testSplit2(self):
        streamline_start = np.array([[0.2, 0.45, 0.4]])
        streamline_end = np.array([[0.6, 0.9, 0.4]])
        streamline_tract = np.array([0])
        points, streamline_tract, streamline_scalars = split_along_grid(streamline_start, streamline_end, streamline_tract, np.zeros(
            (1, 0)), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), [2])
        self.assertEqual(points.shape[0], 3)
        self.assertEqual(points[0][0][0], 0.2)
        self.assertEqual(points[0][0][1], 0.45)
        self.assertEqual(points[1][0][1], 0.5)
        self.assertEqual(points[2][0][0], 0.5)
        self.assertEqual(points[2][1][0], 0.6)
        self.assertEqual(points[2][1][1], 0.9)

    def testNoSplit(self):
        streamline_start = np.array([[0.2, 0.45, 0.4]])
        streamline_end = np.array([[0.3, 0.3, 0.4]])
        streamline_tract = np.array([0])
        points, streamline_tract, streamline_scalars = split_along_grid(streamline_start, streamline_end, streamline_tract, np.zeros(
            (1, 0)), np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), [2])
        self.assertEqual(points.shape[0], 1)
        self.assertEqual(points[0][0][0], 0.2)
        self.assertEqual(points[0][0][1], 0.45)
        self.assertEqual(points[0][1][0], 0.3)
        self.assertEqual(points[0][1][1], 0.3)


if __name__ == '__main__':
    unittest.main()
