from pathlib import Path
import json
import pickle


dir_textdata = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data")
path_output = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/entity2type_textdata.pkl")

entity2types = {}

for split in ["AA", "AB"]:
    for batch_file in (dir_textdata / split).glob("wiki*"):
        with open(batch_file) as f:
            for line in f:
                obj = json.loads(line)

                for rel in obj["entity_rels"]:
                    p, q, v = rel.split('\t')
                    if p == "P31":
                        if q not in entity2types:
                            entity2types[q] = set()
                        entity2types[q].add(v)

        print(batch_file)
        print(len(entity2types))

with open (path_output, 'wb') as f:
    pickle.dump(entity2types, f)
                        

# p, q, v
