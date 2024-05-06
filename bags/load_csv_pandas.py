#!/usr/bin/env python3
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("csvfile", type=str)
args = parser.parse_args()


df = pd.read_csv(args.csvfile)
df = df.set_index("epoch_s", verify_integrity=True)
