from testing import TestSuite

def test_hello():
    print('hi?')

def main():
    TestSuite.discover_tests[__functions_in_module()]().run()