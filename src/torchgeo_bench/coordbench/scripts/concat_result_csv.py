import glob
import re
import sys
import pandas as pd

prefix = sys.argv[1]  # e.g. coordbench_chelsa_agg_embeddings_

paths = sorted(glob.glob(f"{prefix}.taskid_*.csv"),
               key=lambda p: int(re.search(r"taskid_(\d+)\.csv$", p).group(1)))

pd.concat((pd.read_csv(p) for p in paths), ignore_index=True).to_csv(f"{prefix}.csv", index=False)
