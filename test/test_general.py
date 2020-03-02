import unittest
import sys
import numpy as np

sys.path.insert(0, '../')

import functions as fc

class TestGeneral(unittest.TestCase):

    def setUp(self):
        self.my_current_ref_frame = np.array([1, 1, 1])
        self.theta = 45.
        self.phi = 45.
        self.new_ref_frame = fc.rotate_ref_frame(self.my_current_ref_frame, 
                                                self.theta, self.phi)
        self.result = np.array([0.29289321881345254, 1.1102230246251565e-16, 1.7071067811865475])


    def test_rotate_ref_frame(self):
        self.assertEqual(self.new_ref_frame.tolist(), self.result.tolist())

if __name__ == '__main__':
    unittest.main()

