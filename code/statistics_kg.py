import os
import json
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





