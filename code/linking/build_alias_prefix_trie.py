from pathlib import Path
import os

# input: alias table
# output: prefix tree that can match text to qid


dir_entity_alias = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/aliases") # for all


keys = []
values = []
for batch_file in os.listdir(dir_entity_alias)[:1]:
    if not batch_file.endswith('tsv'):
        continue
    with open(dir_entity_alias / batch_file, 'r') as f:
        header = f.readline()
        assert header.strip().split('\t') == ["qid", "alias"]
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            arr = line.strip().split('\t')
            keys.append(f"{arr[1]}")
            values.append(f"{arr[0]}")
print(len(keys))

import marisa_trie
fmt = "<s"
trie = marisa_trie.RecordTrie(fmt, zip(keys, values))
print(trie.get["China"])
            