import json
from pathlib import Path
from collections import Counter
import numpy as np
import pickle


"""

What should be the feature to identify comparable statements?
(e1, e2, a, v1, v2)
"""


def dump_textdata_etype(target_etype):
    entity2type_data = pickle.load(open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/entity2type_textdata.pkl", 'rb'))
    dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data")
    dir_output = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data_by_type/text_data_{target_etype}")
    cnt_property = {}
    for split in ["AA", "AB"]:
        (dir_output / split).mkdir(exist_ok=True, parents=True)
        for batch_file in (dir_data / split).glob("wiki*"):
            with open(batch_file ) as f, open( dir_output / split / batch_file.name, 'w') as fw:
                for line in f:
                    obj = json.loads(line)
                    qid = obj["qid"]
                    if qid not in entity2type_data:
                        continue
                    if entity2type_data[qid] != target_etype:
                        continue
                    fw.write(line)

dump_textdata_etype("Q134556")


def build_property_feature(target_etype):

    pid2freq_rel = Counter() # entity_rels type properties
    pid2freq_val = Counter() # entity_vals type properties
    
    pid2value2freq = {}
    dir_data = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data_by_type/text_data_{target_etype}")
    dir_output = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/{target_etype}")
    dir_output.mkdir(exist_ok=True, parents=True)
    
    for split in ["AA", "AB"]:
        for batch_file in (dir_data / split).glob("wiki*"):
            # print(batch_file.name)
            f = open(batch_file)
            for line in f:
                obj = json.loads(line)
                qid = obj["qid"]
                for s_type in ["entity_rels", "entity_values"]:
                    for s in obj[s_type]:
                        pid, qid, value = s.split('\t')
                        if s_type == "entity_rels":
                            pid2freq_rel[pid] += 1
                        else:
                            pid2freq_val[pid] += 1
                        if pid not in pid2value2freq:
                            pid2value2freq[pid] = Counter()
                        pid2value2freq[pid][value] += 1
            f.close()


    with open (dir_output / "pid2freq_rel.json", 'w') as f:
        # for k,v in sorted(pid2freq.items(), key=lambda x:x[1], reverse=True):
        #     f.write(f"{v}\t{k}\n")
        json.dump(pid2freq_rel, f)
    with open (dir_output / "pid2freq_val.json", 'w') as f:
        json.dump(pid2freq_val, f)

    with open (dir_output / "pid2value2freq.json", 'w') as f:
        json.dump(pid2value2freq, f)



def extract_entity_feature(target_etype):
    dir_data = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data_by_type/text_data_{target_etype}")
    dir_output = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/{target_etype}")
    dir_output.mkdir(exist_ok=True, parents=True)
    
    # load pid2idx
    qid2profile = {}
    
    # extract entity feature
    # initialize feature vector
    for split in ["AA", "AB"]:
        for batch_file in (dir_data / split).glob("wiki*"):
            # print(batch_file.name)
            f = open(batch_file)
            for line in f:
                obj = json.loads(line)
                qid = obj["qid"]
                profile = {}
                # entity_rels = obj["entity_rels"]
                # entity_values = obj["eneity_values"]
                for s_type in ["entity_rels"]: # NOTE we ignore the entity_values
                    for s in obj[s_type]:
                        pid, qid, vid = s.split('\t')
                        if pid not in profile:
                            profile[pid] = []
                        profile[pid].append(vid)
                qid2profile[qid] = profile
            f.close()
    print(len(qid2profile))

    with open (dir_output / "entity2profile.json", 'w') as f:
        json.dump(qid2profile, f)



if __name__ == "__main__":
    dump_textdata_etype("Q105543609")
    build_property_feature(target_etype="Q105543609")
    extract_entity_feature("Q105543609")
