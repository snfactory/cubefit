"""
Script support for running DDT
"""
from __future__ import print_function
import ddtpy

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="DDT: Datacube Deconvolution Thingy")
    parser.add_argument("filename", help="path to configuration file")

    args = parser.parse_args()

    ddtdata = ddtpy.DDTData(args.filename)
    print(ddtdata)
