# -*- coding: utf-8 -*-

import pytest
from pyhy.base import hydrograph

__author__ = "lboutin"
__copyright__ = "lboutin"
__license__ = "mit"


def test_read_run_info():
    #Test string
    instance = epaswmm("test.xml")
    assert instance.read_run_info() == 'test.xml'


    #with pytest.raises(AssertionError):
    #    fib(-10)
