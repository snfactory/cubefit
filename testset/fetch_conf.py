#!/usr/bin/env python
"""
Fetch config files from production jobs and update filenames for local use.
"""

import json
import os
import subprocess
import glob

names = ["LSQ12dbr",
         "PTF09fox",
         "PTF10ndc",
         "PTF10nlg",
         "PTF11dzm",
         "PTF12ena",
         "SN2004dt",
         "SN2005ki",
         "SN2006ac",
         "SN2006ob",
         "SN2011bl",
         "SN2011by",
         "SNF20050919-000",
         "SNF20051003-003",
         "SNF20060624-019",
         "SNF20060512-002",
         "SNF20060609-002",
         "SNF20061009-008",
         "SNF20070427-001",
         "SNF20070429-000",
         "SNF20070504-017",
         "SNF20070712-003",
         "SNF20070831-015",
         "SNF20070902-021",
         "SNF20080707-012",
         "SNF20080717-000",
         "SNF20080720-001",
         "SNF20080725-004",
         "SNF20080821-000",
         "SNF20080918-000",
         "SNNGC4424",
         "SNNGC6801"]

REMOTE_PARENT_DIR = ("/sps/snovae/user/snprod/snprod/jobs/SNF-02-03/"
                     "MoreFlux/CUBEFIT/PCF")
TAG = "0203-CABALLO"

if "SNF_CC_USER_MACHINE" not in os.environ:
    raise RuntimeError("set SNF_CC_USER_MACHINE environment variable to "
                       "USER@MACHINE")

# Fetch all config files to data/config_orig using rsync.
#
# Note: On the command line, arguments containing * need quotes. e.g.,
# --include='*/'. With subprocess, we're not going through the shell.
# so here we must NOT use quotes: --include=*/
#
# Note: The order of the --include and --exclude args is significant.
cmd = ["rsync", "-vzrmt",
       "--include=*/",
       "--include=*/SNF-{}_?b-*_config.json".format(TAG),
       "--exclude=*",
       os.environ["SNF_CC_USER_MACHINE"] + ":" + REMOTE_PARENT_DIR +
       "/{" + ",".join(names) + "}",
       "data/config_orig"]
print(" ".join(cmd))
subprocess.check_call(cmd)

# translate for local use, but putting 'filepaths' into 'filenames'.
for name in names:
    for band in ['B', 'R']:

        # Get filename (this is overly complicated because the SN name in the
        # file name has a different format than in `names`.
        fnames = glob.glob("data/config_orig/{}/"
                           "SNF-{}_{}b-*_config.json"
                           .format(name, TAG, band))
        if len(fnames) != 1:
            raise RuntimeError("Found {:d} config files for {}. Expected one."
                               .format(len(fnames), name))

        # read it
        with open(fnames[0]) as f:
            contents = json.load(f)
            
        # Correct filenames attribute
        contents['filenames'] = [os.path.basename(p)
                                 for p in contents['filepaths']]
        
        # ensure that target location exists
        dirname = "data/{}_{}".format(name, band)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # write config file to new location.
        outfname = "{}/{}_{}.json".format(dirname, name, band)
        print("writing", outfname)
        with open(outfname, 'w') as f:
            json.dump(contents, f, indent=True)
