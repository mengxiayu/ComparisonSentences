from pathlib import Path
import os
import pandas as pd
import shutil
import argparse

# step 1: minimize table size; remove claim_id
def remove_useless_column(WIKIDATA_DIR, OUTPUT_DIR, table):
    # table_names = ["entity_values", "entity_rels", "external_ids"]
    data_path = WIKIDATA_DIR / table
    output_path = OUTPUT_DIR / table
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True,exist_ok=False)

    for batch_file in os.listdir(data_path):
        print(batch_file)
        assert batch_file.endswith(".tsv")
        df = pd.read_csv(data_path / batch_file, header=0, sep='\t', on_bad_lines='warn', quoting=3) # ignore quoting, or you will fail
        if "claim_id" in df:
            df = df.drop(columns=['claim_id'])
        df.to_csv(output_path / batch_file, sep='\t', index=False)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikidata', help='path to input directory')
    parser.add_argument('--table_name', type=str, default='entity_values', help='table name')
    parser.add_argument('--output', type=str, default="/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed", help='path to output directory')
    return parser

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    WIKIDATA_DIR = Path(args.data)
    OUTPUT_DIR = Path(args.output)

    remove_useless_column(WIKIDATA_DIR, OUTPUT_DIR, args.table_name)


