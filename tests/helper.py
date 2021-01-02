import logging
import pytest
# datasets
from sklearn import datasets

def parseParams():
    import argparse
    parser = argparse.ArgumentParser(description='Run tests',
                                     epilog="- rut tests")
    parser.add_argument('--test',
                        type=int,
                        default=0,
                        action='store',
                        help='run test id',
                        required=False)
    args = parser.parse_args()

    return args


def checkTestID(newClass, id):
    functions = [getattr(newClass, m) for m in dir(newClass) if not m.startswith('__')]
    findTest = "runTest" + str(id)
    testFound = False
    for testFunc in functions:
        if findTest in str(testFunc):
            print("[*] Test to run {}() :".format(findTest))
            testFound = True
            testFunc()
            return testFound

    return testFound