import argparse
from pathlib import Path
import pandas as pd
# university
table_list = [
    "P31_Q3918",
    "P31_Q902104",
    "P31_Q875538",
    "P31_Q62078547",

]

input_dir = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata/tables/entity_rels")
output_dir = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata/entity_lists")
output_name = "university"


qid_list = []
for table_name in table_list:
    df = pd.read_csv(input_dir / f"{table_name}.tsv", sep='\t')
    qid_list.extend(list(df["qid"]))
qid_list = list(set(qid_list))
print("# entity:", len(qid_list))
with open(output_dir / f"{output_name}.txt", 'w') as f:
    for qid in qid_list:
        f.write(qid + '\n')