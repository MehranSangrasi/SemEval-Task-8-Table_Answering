import pandas as pd
from datasets import load_dataset
import os

semeval_eval = load_dataset("cardiffnlp/databench", name="semeval", split="train")


semeval_eval_df = pd.DataFrame(semeval_eval)

semeval_eval_df.to_csv("data/train_set.csv", index=False)