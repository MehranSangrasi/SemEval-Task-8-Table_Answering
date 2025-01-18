import pandas as pd
import ast
from datasets import load_dataset


dev_set = load_dataset("cardiffnlp/databench", name="semeval", split="dev")

dev_set = dev_set.to_pandas()

dev_set.to_csv("data/dev_set.csv", index=False)


