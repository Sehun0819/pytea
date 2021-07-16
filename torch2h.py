#!/usr/bin/env python

import os
import subprocess
import argparse
import tempfile
from pathlib import Path


DEFAULT_FRONT_PATH = Path(__file__).absolute().parent / "packages"/ "pytea" / "index.js"


def parse_arg():
    """
    pytorch_pgm_path : <path_to_pytorch>/<pytorch_pgm_name>.py
    """
    parser = argparse.ArgumentParser("Torch2H: PyTorch Graph to H Program")
    parser.add_argument("path", help="PyTorch entry file path")
    parser.add_argument(
        "--config", default=None, help="set path to pyteaconfig.json",
    )
    parser.add_argument(
        "--front_path",
        default=str(DEFAULT_FRONT_PATH),
        help="path to constraint generator (index.js)",
    )
    parser.add_argument(
        "--silent", action="store_true", help="do not print result (for server)"
    )
    parser.add_argument(
        "-l",
        "--log",
        default=-1,
        type=int,
        help="severity of analysis result (0 to 3)",
    )

    return parser.parse_args()


def parse_log_level(args):
    if 0 <= args.log <= 3:
        if args.log == 0:
            log_level = "--logLevel=none"
        elif args.log == 1:
            log_level = "--logLevel=result-only"
        elif args.log == 2:
            log_level = "--logLevel=reduced"
        else:
            log_level = "--logLevel=full"
    else:
        log_level = ""

    return log_level

def main():
    args = parse_arg()

    entry_path = Path(args.path)
    if not entry_path.exists():
        raise Exception(f"entry path {entry_path} does not exist")

    log_level = parse_log_level(args)
    config = args.config
    config = f"--configPath={config} " if config else ""

    frontend_command = f"node {args.front_path} {entry_path} {config}{log_level}"
    print(frontend_command)
    subprocess.call(frontend_command, shell=True)

if __name__ == "__main__":
    main()
