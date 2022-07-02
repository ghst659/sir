#!/usr/bin/env python3

import argparse
import collections
import logging
import sys

import sir

def main(argv: collections.abc.Sequence[str]) -> int:
    """Driver for models."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--alpha", type=float, default=0.25, dest="alpha",
                        help="Model parameter alpha.")
    parser.add_argument("--beta", type=float, default=0.1, dest="beta",
                        help="Model parameter beta.")
    parser.add_argument("--rho", type=float, default=1.0, dest="rho",
                        help="Model parameter rho.")
    parser.add_argument("--delta", type=float, default=0.003, dest="delta",
                        help="Model parameter delta.")
    parser.add_argument("--i0", type=float, default=0.001, dest="i0",
                        help="Model parameter i0.")
    parser.add_argument("--cycles", type=int, default=100, dest="cycles",
                        help="Simulation cycle count.")
    parser.add_argument("--tau", type=int, default=5, dest="tau",
                        help="Resusceptibility latency.")
    parser.add_argument("-v","--verbose",
                        dest='verbose', action="store_true",
                        help="run verbosely")
    args = parser.parse_args(args=argv[1:])
    model = sir.SIRXModel(args.i0, args.alpha, args.beta,
                          args.rho, args.delta, args.tau)
    cycles = model.run(args.cycles)
    data = model.dump()
    keys = data.keys()
    print(",".join(f"{key}" for key in keys))
    for i in range(cycles):
        line = ",".join(f"{data[key][i]:0<8.5}" for key in keys)
        print(line)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
