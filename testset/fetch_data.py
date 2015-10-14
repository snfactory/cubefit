#!/usr/bin/env python
"""
Fetch data for a given SN/band
"""

import os
import sys
import json
import subprocess

if "SNF_CC_USER_MACHINE" not in os.environ:
    raise RuntimeError("set SNF_CC_USER_MACHINE environment variable to "
                       "USER@MACHINE")

if len(sys.argv) != 2:
    print("usage: fetch_data.py PTF09fox_B")
    exit()
name = sys.argv[1]

with open("data/{}/{}.json".format(name, name)) as f:
    conf = json.load(f)

cmd = ["rsync", "-vzmt",
       os.environ["SNF_CC_USER_MACHINE"] +
       ":{" + ",".join(conf["filepaths"]) + "}",
       "data/" + name]
print(" ".join(cmd))
subprocess.check_call(cmd)
