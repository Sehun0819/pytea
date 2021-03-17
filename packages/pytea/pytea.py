#!/usr/bin/env python

"""
json2z3.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Ho Young Jhoo

Main starting point of Pytea analyzer
"""
import os
import subprocess
import argparse
import tempfile
from pathlib import Path
from z3wrapper.json2z3 import run_default


DEFAULT_FRONT_PATH = Path(__file__).absolute().parent / "index.js"


def parse_arg():
    """
    pytorch_pgm_path : <path_to_pytorch>/<pytorch_pgm_name>.py
    json_file        : <pytorch_pgm_name>_z3.json
    """
    parser = argparse.ArgumentParser("PyTea: PyTorch Tensor shape Analyzer")
    parser.add_argument("path", help="PyTorch entry file path")
    parser.add_argument(
        "--out",
        default=None,
        help="output file path of constraint json. if not set, send constraints to temporary file",
    )
    parser.add_argument(
        "--z3_only", action="store_true", help="run z3py on z3 json file <path>"
    )
    parser.add_argument(
        "--no_z3",
        action="store_true",
        help="do not run z3py and just produce z3 json file",
    )
    parser.add_argument(
        "--front_path",
        default=str(DEFAULT_FRONT_PATH),
        help="path to constraint generator (index.js)",
    )
    parser.add_argument(
        "--node-arguments", default="", help="arguments for constraint generator"
    )
    parser.add_argument(
        "--silent", action="store_true", help="do not print result (for server)"
    )

    return parser.parse_args()


def main_with_temp(args, entry_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "constraint.json"

        if json_path.exists():
            os.remove(json_path)

        frontend_command = f"node {args.front_path} {entry_path} {args.node_arguments} --resultPath={json_path}"
        print(frontend_command)
        subprocess.call(frontend_command, shell=True)

        # run backend with json formatted constraints.
        if not args.no_z3:
            run_default(json_path, args)


def main():
    args = parse_arg()

    entry_path = Path(args.path)
    if not entry_path.exists():
        raise Exception(f"entry path {entry_path} does not exist")

    if args.out is None:
        return main_with_temp(args, entry_path)

    base_dir = entry_path.parent

    json_path = Path(args.out)

    if args.z3_only:
        json_path = entry_path

    # run frontend with given path, creates json formatted constraints.
    if not args.z3_only:
        if json_path.exists():
            os.remove(json_path)

        frontend_command = f"node {args.front_path} {entry_path} {args.node_arguments} --resultPath={json_path}"
        print(frontend_command)
        subprocess.call(frontend_command, shell=True)

    # run backend with json formatted constraints.
    if not args.no_z3:
        run_default(json_path, args)


if __name__ == "__main__":
    main()
