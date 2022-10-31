import os
import json
from pathlib import Path
from collections import Counter
# count the number of entities.
def count_entities():
    path = "/afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikidata/labels"
    batch_files = os.listdir(path)
    print(f"{len(batch_files)} files in total")
    entity_set = set()
    count = 0
    for bf in batch_files:
        print(bf)
        with open(os.path.join(path, bf), 'r') as f:
            for line in f:
                pair = line.strip().split('\t') # label, qid
                assert len(pair) == 2
                entity_set.add(pair[1])
                count += 1
    print(len(entity_set)) 
    print(count) 
# count_entities()
# output:
# 81830321
# 81830402



def _load_property2aliases(path):
    # from the property_aliases table
    pid2aliases = {}
    with open(path, 'r') as f:
        header = f.readline()
        assert header.strip().split('\t') == ["alias", "pid"]
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            alias, pid = line.strip().split('\t')
            if pid not in pid2aliases:
                pid2aliases[pid] = []
            pid2aliases[pid].append(alias)
    return pid2aliases

path_prop_alias = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_aliases/0.tsv")
pid2alias = _load_property2aliases(path_prop_alias)




def count_property_data(data_dir):
    prop2freq_val = Counter()
    prop2freq_rel = Counter()
    split_list = ["AA", "AB"]
    for split in split_list:
        for batch_file in os.listdir(dir_data / split):
            with open(dir_data / split / batch_file) as f:
                for line in f:
                    obj = json.loads(line)

                    num_entity_rels = len(set(obj["entity_rels"]))
                    num_entity_values = len(set(obj["entity_values"]))
                    for _s in set(obj["entity_rels"]):
                        s = _s.split('\t')
                        if len(s) != 3:
                            continue
                        p, e, v = s
                        prop2freq_rel[p] += 1
                    for _s in set(obj["entity_values"]):
                        s = _s.split('\t')
                        if len(s) != 3:
                            continue
                        p, e, v = s
                        prop2freq_val[p] += 1

    with open (dir_data / 'property_distribution_"relation"_statements', 'w') as f:
        for k,v in prop2freq_rel.most_common():
            f.write(f"{v}\t{k}\t{pid2alias[k][0]}\n")
        
    with open (dir_data / 'property_distribution_"value"_statements', 'w') as f:
        for k,v in prop2freq_val.most_common():
            f.write(f"{v}\t{k}\t{pid2alias[k][0]}\n")
dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data/")

count_property_data(dir_data)