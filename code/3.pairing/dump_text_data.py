import json
from pathlib import Path
import pickle

entity2type_data = pickle.load(open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/entity2type_textdata.pkl", 'rb'))
print("entity2type_data", len(entity2type_data))

def load_linked_entities():
    entity2type_linked = {}
    dir_linked = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1/combined")
    for split in ["AA", "AB"]:
        for batch_file in (dir_linked / split).glob("wiki*"):
            with open(batch_file) as f:
                for line in f:
                    obj = json.loads(line)
                    qid = obj["qid"]
                    if qid in entity2type_data:
                        entity2type_linked[qid] = entity2type_data[qid]
    return entity2type_linked
entity2type_linked = load_linked_entities()
print("entity2type_linked", len(entity2type_linked))

def dump_textdata_etype(target_etype):
    # for large entity type
    # entity2type_data = pickle.load(open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/entity2type_textdata.pkl", 'rb'))
    dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data")
    dir_output = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data_by_type/text_data_{target_etype}")
    for split in ["AA", "AB"]:
        (dir_output / split).mkdir(exist_ok=True, parents=True)
        for batch_file in (dir_data / split).glob("wiki*"):
            with open(batch_file ) as f, open( dir_output / split / batch_file.name, 'w') as fw:
                for line in f:
                    obj = json.loads(line)
                    qid = obj["qid"]
                    if qid not in entity2type_linked:
                        continue
                    if entity2type_linked[qid] != target_etype:
                        continue
                    fw.write(line)


def dump_textdata_etype_small(target_etype):
    # size: 5879050
    dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data")
    dir_output = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data_by_type/text_data_{target_etype}")
    (dir_output / "AA").mkdir(exist_ok=True, parents=True)
    fw = open( dir_output / "AA/wiki_all", 'w')
    for split in ["AA", "AB"]:
        for batch_file in (dir_data / split).glob("wiki*"):
            with open(batch_file) as f:
                for line in f:
                    obj = json.loads(line)
                    qid = obj["qid"]
                    if qid not in entity2type_linked:
                        continue
                    if entity2type_linked[qid] != target_etype:
                        continue
                    fw.write(line)
    fw.close()




def dump_textdata_all():
    with open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1/statistics/combined/linked_cnt_P31_alias") as f:
        for idx, line in enumerate(f):
            
            qid, alias, freq = line.strip().split('\t')
            print(idx, qid, alias, freq)
            freq = int(freq)
            if freq > 10000:
                dump_textdata_etype(qid)
            elif freq > 2:
                dump_textdata_etype_small(qid)


dump_textdata_all()