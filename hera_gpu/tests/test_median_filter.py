import unittest
import numpy as np 
import scipy.signal as sps
import median_filter as mf

class BasicTest(unittest.TestCase):
	def testOneByOne(self):
		""" Test using a 1x1 window """
		in0 = np.random.rand(1, 1)
		in1 = np.random.rand(29, 29)
		in2 = np.random.rand(93, 93)
		in3 = np.random.rand(7000, 7000)

		check0 = sps.medfilt2d(in0, 1)
		check1 = sps.medfilt2d(in1, 1)
		check2 = sps.medfilt2d(in2, 1)
		check3 = sps.medfilt2d(in3, 1)

		self.assertTrue(np.allclose(check0, mf.MedianFilter(kernel_size=1, input=in0)))
		self.assertTrue(np.allclose(check1, mf.MedianFilter(kernel_size=1, input=in1)))
		self.assertTrue(np.allclose(check2, mf.MedianFilter(kernel_size=1, input=in2)))
		self.assertTrue(np.allclose(check3, mf.MedianFilter(kernel_size=1, input=in3)))

	def testThreeByThree(self):
		""" Test using a 3x3 window """
		in0 = np.random.rand(1, 1)
		in1 = np.random.rand(29, 29)
		in2 = np.random.rand(93, 93)
		in3 = np.random.rand(7000, 7000)

		check0 = sps.medfilt2d(in0, 3)
		check1 = sps.medfilt2d(in1, 3)
		check2 = sps.medfilt2d(in2, 3)
		check3 = sps.medfilt2d(in3, 3)

		self.assertTrue(np.allclose(check0, mf.MedianFilter(kernel_size=3, input=in0)))
		self.assertTrue(np.allclose(check1, mf.MedianFilter(kernel_size=3, input=in1)))
		self.assertTrue(np.allclose(check2, mf.MedianFilter(kernel_size=3, input=in2)))
		self.assertTrue(np.allclose(check3, mf.MedianFilter(kernel_size=3, input=in3)))

class BigTest(unittest.TestCase):
	def testFivebyFive(self):
		""" Test using a 5x5 window """
		in0 = np.random.rand(1, 1)
		in1 = np.random.rand(4000, 4000)

		check0 = sps.medfilt2d(in0, 5)
		check1 = sps.medfilt2d(in1, 5)

		self.assertTrue(np.allclose(check0, mf.MedianFilter(kernel_size=5, input=[in0])[0]))
		self.assertTrue(np.allclose(check1, mf.MedianFilter(kernel_size=5, input=[in1])[0]))
	
	def testNinebyNine(self):
		""" Test using a 9x9 window """
		in0 = np.random.rand(1, 1)
		in1 = np.random.rand(997, 997)

		check0 = sps.medfilt2d(in0, 9)
		check1 = sps.medfilt2d(in1, 9)

		self.assertTrue(np.allclose(check0, mf.MedianFilter(kernel_size=9, input=[in0], bw=16, bh=16)[0]))
		self.assertTrue(np.allclose(check1, mf.MedianFilter(kernel_size=9, input=[in1])[0]))


class NotSquareTest(unittest.TestCase):
	def testLopsidedImage(self):
		""" Test using non-square images """
		in0 = np.random.rand(1, 73)
		in1 = np.random.rand(5, 3)
		in2 = np.random.rand(2, 3)
		in3 = np.random.rand(8013, 700)

		check0 = sps.medfilt2d(in0, 1)
		check1 = sps.medfilt2d(in1, 1)
		check2 = sps.medfilt2d(in2, 3)
		check3 = sps.medfilt2d(in3, 5)

		self.assertTrue(np.allclose(check0, mf.MedianFilter(kernel_size=1, input=[in0])[0]))
		self.assertTrue(np.allclose(check1, mf.MedianFilter(kernel_size=1, input=[in1])[0]))
		self.assertTrue(np.allclose(check2, mf.MedianFilter(kernel_size=3, input=[in2])[0]))
		self.assertTrue(np.allclose(check3, mf.MedianFilter(kernel_size=5, input=[in3])[0]))

	def testLopsidedWindow(self):
		""" Test using non-square windows """
		in0 = np.random.rand(1, 73)
		in1 = np.random.rand(5, 3)
		in2 = np.random.rand(2, 3)
		in3 = np.random.rand(8013, 700)

		check0 = sps.medfilt2d(in0, (1, 11))
		check1 = sps.medfilt2d(in1, (11, 1))
		check2 = sps.medfilt2d(in2, (3, 5))
		check3 = sps.medfilt2d(in3, (9, 5))

		self.assertTrue(np.allclose(check0, mf.MedianFilter(kernel_size=(1, 11), input=[in0])[0]))
		self.assertTrue(np.allclose(check1, mf.MedianFilter(kernel_size=(11, 1), input=[in1])[0]))
		self.assertTrue(np.allclose(check2, mf.MedianFilter(kernel_size=(3, 5), input=[in2])[0]))
		self.assertTrue(np.allclose(check3, mf.MedianFilter(kernel_size=(9, 5), input=[in3])[0]))


if __name__ == '__main__':
	unittest.main()
