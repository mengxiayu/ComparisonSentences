import json
import pandas as pd
from pathlib import Path 
import os


def transform_quantity(value):
    value = value.lstrip('+')
    return value

def transform_time(value):
    # only extract year now
    # format : +%Y-%m-%dT%H:%M:%SZ
    try:
        date, time = value.rstrip('Z').lstrip('+').split('T')
        year, month, day = date.split('-')
        return year
    except:
        print(value)
        return value

# Deprecated. We don't want to save this to disk.
def transform_alias_batch(pid2type, input_table_path, output_table_path): 
    table = pd.read_csv(input_table_path, sep='\t', header=0, on_bad_lines='warn', quoting=3)
    with open(output_table_path, 'w') as fw:
        fw.write("property_id\tqid\tvalue\n")
        for index, row in table.iterrows():
            pid = row["property_id"]
            if pid not in pid2type:
                # print(f"Property {pid} not in index")
                continue
            if pid2type[pid] == "quantity":
                value = transform_quantity(row["value"])
            elif pid2type[pid] == "time":
                value = transform_time(row["value"])
            else:
                value = row["value"]
            fw.write(f"{pid}\t{row['qid']}\t{value}\n")

def transform_alias(pid2type, row):

    pid = row["property_id"]
    if pid not in pid2type:
        # print(f"Property {pid} not in index")
        return value
    if pid2type[pid] == "quantity":
        value = transform_quantity(row["value"])
    elif pid2type[pid] == "time":
        value = transform_time(row["value"])
    else:
        value = row["value"]
    return value

if __name__ == "__main__":

# build pid2datatype
    df = pd.read_csv("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_labels/0_tmp.tsv", sep='\t', header=0)
    input_dir = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/entity_values")
    output_dir = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/entity_values_processed")

    pid2type = {}
    for index, row in df.iterrows():
        pid2type[row["pid"]] = row["datatype"]

    for path in input_dir.glob("*.tsv"):
        batch_file = os.path.basename(path)
        transform_alias(pid2type, input_table_path=(input_dir / batch_file), output_table_path=(output_dir / batch_file))



