import json
from pathlib import Path
from collections import Counter
import numpy as np
import pickle


"""

What should be the feature to identify comparable statements?
(e1, e2, a, v1, v2)
"""

def build_property_feature():
    pid2idx = {}
    pid2freq = Counter()
    pid2vid2freq = {}
    target_etype = "Q5"
    dir_data = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data_{target_etype}")
    dir_output = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/{target_etype}")
    dir_output.mkdir(exist_ok=True, parents=True)
    
    for split in ["AA", "AB"]:
        for batch_file in (dir_data / split).glob("wiki*"):
            # print(batch_file.name)
            f = open(batch_file)
            for line in f:
                obj = json.loads(line)
                qid = obj["qid"]
                # entity_rels = obj["entity_rels"]
                # entity_values = obj["eneity_values"]
                for s_type in ["entity_rels", "entity_values"]:
                    for s in obj[s_type]:
                        pid, qid, vid = s.split('\t')
                        if pid not in pid2idx:
                            pid2idx[pid] = len(pid2idx)
                        pid2freq[pid] += 1
                        if pid not in pid2vid2freq:
                            pid2vid2freq[pid] = Counter()
                        pid2vid2freq[pid][vid] += 1
            f.close()

    print(len(pid2idx))
    with open (dir_output / "pid2idx.tsv", 'w') as f:
        for k,v in sorted(pid2idx.items(), key=lambda x:x[1]):
            f.write(f"{v}\t{k}\n")
    with open (dir_output / "pid2freq.tsv", 'w') as f:
        for k,v in sorted(pid2freq.items(), key=lambda x:x[1], reverse=True):
            f.write(f"{v}\t{k}\n")
    with open (dir_output / "pid2vid2freq.json", 'w') as f:
        json.dump(pid2vid2freq, f)




def extract_entity_feature():
    pid2idx = {}
    target_etype = "Q5"
    dir_data = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data_{target_etype}")
    dir_output = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/{target_etype}")
    dir_output.mkdir(exist_ok=True, parents=True)
    
    # load pid2idx
    pid2idx = {}
    idx2pid = {}
    qid2feature = {}
    qid2profile = {}
    with open (dir_output / "pid2idx.tsv") as f:
        for line in f:
            idx, pid = line.strip().split('\t')
            pid2idx[pid] = int(idx)
            idx2pid[int(idx)] = pid
    print(len(pid2idx), len(idx2pid))
    assert len(pid2idx) == len(idx2pid)
    
    # extract entity feature
    # initialize feature vector
    num_feature = len(pid2idx)
    for split in ["AA", "AB"]:
        for batch_file in (dir_data / split).glob("wiki*"):
            # print(batch_file.name)
            f = open(batch_file)
            for line in f:
                obj = json.loads(line)
                qid = obj["qid"]
                feature = np.zeros(num_feature)
                profile = {}
                # entity_rels = obj["entity_rels"]
                # entity_values = obj["eneity_values"]
                for s_type in ["entity_rels"]: # NOTE we ignore the entity_values
                    for s in obj[s_type]:
                        pid, qid, vid = s.split('\t')
                        feature[pid2idx[pid]] = 1
                        if pid not in profile:
                            profile[pid] = []
                        profile[pid].append(vid)
                qid2feature[qid] = feature
                qid2profile[qid] = profile
            f.close()
    with open (dir_output / "entity2feature.pkl", 'wb') as f:
        pickle.dump(qid2feature, f)
    with open (dir_output / "entity2profile.pkl", 'wb') as f:
        pickle.dump(qid2profile, f)
extract_entity_feature()
