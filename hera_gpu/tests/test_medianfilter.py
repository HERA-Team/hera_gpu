# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License
"""Tests for the antenna_metrics module."""

from __future__ import print_function
import unittest
import numpy as np

def fake_data():
    np.random.seed(1)
    a = np.random.rand(60,1024)
    return a

class TestMedianFilter(unittest.TestCase):

    def setUp(self):
        self.radius = 11
        self.data = fake_data()
        from scipy.signal import medfilt2d
        self.solution = medfilt2d(self.data,self.radius)
        
    def test_medfilt2d_pytorch(self):
        from hera_gpu import medfilt2d_pytorch
        ans = medfilt2d_pytorch(self.data,self.radius)
        np.testing.assert_array_equal(ans,self.solution)

    def test_medfilt2d_jack(self):
        from hera_gpu import medfilt2d_jack
        ans = medfilt2d_jack(self.data,self.radius)
        np.testing.assert_array_equal(ans.shape,self.solution.shape)
        np.testing.assert_array_equal(ans,self.solution)

    def test_medfilt2d_suomela(self):
        pass

if __name__ == '__main__':
    unittest.main()
