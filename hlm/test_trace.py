import numpy as np
from .trace import NewTrace
from .abstracts import Hashmap
import unittest as ut

class Test_Trace(ut.TestCase):
    def setUp(self):

        self.a = {chr(i+97):list(range(10)) for i in range(5)}
        
        self.t = NewTrace(**self.a)
        self.mt = NewTrace(self.a,self.a,self.a)

    def test_validate_names(self):
        b = self.a.copy()
        try:
            bad_names = NewTrace(self.a,b,self.a,self.a)
        except KeyError:
            pass

    # tests
    def test_slicing(self):
        t = self.t
        mt = self.mt
    #1index
        assert t[1] == {'a':1, 'b':1, 'c':1, 'd':1, 'e':1}
        assert mt[6] == [{k:6 for k in ['a','b', 'c', 'd', 'e']}]*3
        assert t[-1] =={k:9 for k in  ['a','b', 'c', 'd', 'e']}
        assert mt[-1] == [{k:9 for k in ['a','b', 'c', 'd', 'e']}]*3

        assert t[2:5] == {k:list(range(2,5)) for k in  ['a','b', 'c', 'd', 'e']}
        assert mt[8:] == [ {k:list(range(8,10)) for k in  ['a','b', 'c', 'd', 'e'] }] * 3 
        assert t[-4::2] ==  {k:[6,8] for k in  ['a','b', 'c', 'd', 'e']}

        assert t['a'] == list(range(10))
        assert mt['a'] == [list(range(10))]*3
        assert t[['a','b']] == {'a':list(range(10)), 'b':list(range(10))}
        assert mt[['a','b']] == [{'a':list(range(10)), 'b':list(range(10))}]*3

        #2index 
        assert t['a', 1] == 1
        assert t[['a', 'b'], 1] == {'a':1, 'b':1}
        assert mt['e', 5] == [5]*3
        assert mt[['d', 'e'], 8:] == [{'d':[8,9], 'e':[8,9]}]*3

        assert t[0, 'a'] == list(range(10))
        assert t[0, ['a', 'b']] == {'a':list(range(10)), 'b':list(range(10))}
        try:
            t[1, ['a','c']]
            raise Exception('This did not raise an exception within the slicer!')
        except IndexError:
            pass
        assert mt[1:, ['a','c']] == [{'a':list(range(10)), 'c':list(range(10))}] * 2
        assert mt[2, 'a'] == list(range(10))
        assert t[0,-1] == {k:9 for k in ['a', 'b', 'c', 'd', 'e']}
        assert t[0,:] ==  {k:list(range(10)) for k in  ['a', 'b', 'c', 'd', 'e']}
        assert mt[:, -1:-4:-1] ==  [{k:[9,8,7] for k in ['a', 'b', 'c', 'd', 'e']}]*3

        #3index
        assert t[0, 'a', -1] == 9
        assert t[0, ['a','b'],-3::2] == {'a':[7,9], 'b':[7,9]}
        assert t[0, : ,-1] == {k:9 for k in ['a','b','c','d','e']}
        try:
            t[1, 'a', -1]
            raise Exception('this did not raise an exception when it should have')
        except IndexError:
            pass
        assert mt[1:, 'a', -1] == [9]*2
        assert mt[1:, ['a','b'], -2:] == [{'a':[8,9], 'b':[8,9]}]*2
        assert mt[2, 'a', 5::2] == [5,7,9]
        assert mt[1:, 'a', -5::2] == [[5,7,9]]*2
        assert mt[:, 'a', -5::2] == [[5,7,9]]*3
        assert mt[2, :, :] == {k:list(range(10)) for k in ['a','b','c','d','e']}
        assert mt[:,:,:] == mt.chains
        assert mt[:,:,:] is not mt.chains
