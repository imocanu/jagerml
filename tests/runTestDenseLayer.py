#!/usr/bin/env python3

import jagerml
from helper import parseParams, checkTestID

args = parseParams()
ml = jagerml.ML()

if(args.test > 0):
    if checkTestID(ml.dense, args.test) is False:
        print("[!] Test number {} was NOT found !!!".format(args.test))

else:
    print("[*] Run all ..")