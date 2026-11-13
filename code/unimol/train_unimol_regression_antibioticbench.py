#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from unimol_benchmark_common import add_common_cli, run_benchmark

def main():
    parser = argparse.ArgumentParser(
        description="Uni-Mol v1 AntibioticBench regression runner"
    )
    add_common_cli(parser)
    args = parser.parse_args()
    run_benchmark(args, task="regression")

if __name__ == "__main__":
    main()
