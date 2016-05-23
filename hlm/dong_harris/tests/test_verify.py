from hlm.dong_harris import verify
import numpy as np
import pysal as ps
import unittest

membership = np.array([1, 3, 4, 1, 4, 0, 4, 4, 4, 0, 
                       0, 2, 4, 3, 0, 0, 3, 3, 1, 0,
                       3, 1, 2, 1, 0])

class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.membership = np.array([1, 3, 4, 1, 4, 0, 4, 4, 4, 0, 
                                    0, 2, 4, 3, 0, 0, 3, 3, 1, 0,
                                    3, 1, 2, 1, 0])
        self.Delta = np.array([[0,1,0,0,0],
                               [0,0,0,1,0],
                               [0,0,0,0,1], 
                               [0,1,0,0,0], 
                               [0,0,0,0,1], 
                               [1,0,0,0,0], 
                               [0,0,0,0,1], 
                               [0,0,0,0,1], 
                               [0,0,0,0,1], 
                               [1,0,0,0,0], 
                               [1,0,0,0,0], 
                               [0,0,1,0,0], 
                               [0,0,0,0,1], 
                               [0,0,0,1,0], 
                               [1,0,0,0,0], 
                               [1,0,0,0,0], 
                               [0,0,0,1,0], 
                               [0,0,0,1,0], 
                               [0,1,0,0,0], 
                               [1,0,0,0,0], 
                               [0,0,0,1,0], 
                               [0,1,0,0,0], 
                               [0,0,1,0,0], 
                               [0,1,0,0,0], 
                               [1,0,0,0,0]])
        self.N,self.J = self.Delta.shape
    
    def test_membership(self):
        dtest, _ = verify.Delta_members(Delta=None, 
                                        membership = self.membership,
                                        N = self.N,
                                        J = self.J)
        np.testing.assert_equal(dtest, self.Delta)
        _, mtest = verify.Delta_members(Delta=self.Delta,
                                        membership = None,
                                        N = self.N,
                                        J = self.J)
        np.testing.assert_equal(mtest.flatten(), self.membership)

if __name__ == '__main__':
    unittest.main()

