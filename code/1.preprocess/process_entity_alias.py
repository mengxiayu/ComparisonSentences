
# step 2: combine alias

from pathlib import Path
import os
import pandas as pd
import shutil
import argparse

def clean_alias(text):
    text = str(text)
    assert '|sep|' not in text
    # text = text.replace(' ', '_')
    return text


# step 2: replace entity's alias
def process_alias(WIKIDATA_DIR, OUTPUT_DIR, table):
    assert table == "aliases"
    # table_names = ["entity_values", "entity_rels", "external_ids"]
    data_path = WIKIDATA_DIR / table
    output_path = OUTPUT_DIR / table
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True,exist_ok=False)

    for batch_file in os.listdir(data_path):
        print(batch_file)
        assert batch_file.endswith(".tsv")
        df = pd.read_csv(data_path / batch_file, header=0, sep='\t', on_bad_lines='warn', quoting=3)
        
        # merge aliases
        qid2alias = {}
        for index, row in df.iterrows():
            alias = clean_alias(row["alias"])
            qid = row["qid"]
            if qid not in qid2alias:
                qid2alias[qid] = []
            qid2alias[qid].append(alias)
        
        with open(output_path / batch_file, 'w') as f:
            f.write(f"qid\talias\n")
            for qid, aliases in qid2alias.items():
                alias_text = '|sep|'.join(aliases)
                f.write(f"{qid}\t{alias_text}\n")
        


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikidata', help='path to input directory')
    parser.add_argument('--table_name', type=str, default='aliases', help='table name')
    parser.add_argument('--output', type=str, default="/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed", help='path to output directory')
    return parser

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    WIKIDATA_DIR = Path(args.data)
    OUTPUT_DIR = Path(args.output)

    process_alias(WIKIDATA_DIR, OUTPUT_DIR, args.table_name)


