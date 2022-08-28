import os
import json
import argparse
# count the number of properties. Directly open the file. Count: 9516

# entity_values, entity_rels, external_ids: claim_id        property_id     qid     value
# 

# index: file to entity
def build_index(table_name):
    data_path = "/afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikidata"
    index_path = "/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata/indices"
    file2entity = {}
    for batch_file in os.listdir(os.path.join(data_path, table_name)):
        entity_set = set()
        with open (os.path.join(data_path, table_name, batch_file), 'r') as f:
            next(f)
            for line in f:
                entity_set.add(line.strip().split('\t')[2])
        print(batch_file, len(entity_set))
        file2entity[batch_file] = list(entity_set)
    with open(os.path.join(index_path, table_name)+".json", 'w') as f:
        f.write(json.dumps(file2entity))

def reconstruct_index(table_name):
    index_path = "/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata/indices"
    with open(os.path.join(index_path, table_name)+".json", 'r') as f:
        file2entity = json.load(f)
    entity2file = {}
    for batch_file, entity_list in file2entity.items():
        for e in entity_list:
            if e not in entity2file:
                entity2file[e] = []
            entity2file[e].append(batch_file)
    with open(os.path.join(index_path, table_name) + ".e2f.json", 'w') as f:
        f.write(json.dumps(entity2file))    


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table_name', type=str, default="entity_values", help="folder name")
    return parser

if __name__ == "__main__":
    args = get_arg_parser().parse_args()

    # build_index(args.table_name)
    reconstruct_index(args.table_name)
