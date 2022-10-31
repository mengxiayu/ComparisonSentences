from pathlib import Path
import json
import time
import os
import pickle
import argparse
from collections import Counter
from itertools import product, combinations
import json
from tqdm import tqdm


"""
This code is used for extracting statement pairs with common properties for paired entities. 
Note that "with common property" is not a sufficient condition for "comparable", so this code is deprecated. 
2022.10.31
"""

def load_linked_data():
    dir_linked = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1/combined")
    data_linked = {}
    for split in ["AA", "AB"]: 
        for batch_file in (dir_linked / split).glob("wiki*"):
            # print(batch_file.name)
            f = open(batch_file)
            for line in f:
                obj = json.loads(line)
                qid = obj["qid"]
                data_linked[qid] = obj
    return data_linked

def entity_pairing():
    print("=== Running entity pairing! ===")
    target_etype = "Q4830453"
    

    dir_output = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/paired_v1/{target_etype}")
    dir_output.mkdir(parents=True, exist_ok=True)
    dir_matched = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/matched/{target_etype}")
    assert dir_matched.exists()
    data_linked = load_linked_data()
    print("linked data loaded")
    for v_file in dir_matched.glob("*.json"): # 
        print(v_file.name)
        matched_pairs = set()
        
        data_matched = {}
        with open(v_file) as f:
            for line in f:
                obj = json.loads(line)
                _matched = {v[0]:v for v in obj["matched"]}
                obj["matched"] = _matched
                data_matched[obj["qid"]] = obj
                
        for a, obj in data_matched.items():
            for b in obj["matched"]:
                if a == b:
                    continue
                if (b, a) in matched_pairs:
                    continue
                if b not in data_matched:
                    continue
                b_obj = data_matched[b]
                if a in b_obj["matched"]:
                    # yeah! They matched each other
                    matched_pairs.add((a,b))
        print(len(matched_pairs))
        if len(matched_pairs) == 0:
            continue
        cnt_write = 0
        with open(dir_output / v_file.name, 'w') as fw:

            for a,b in matched_pairs:
                if a not in data_linked or b not in data_linked:
                    continue
                a_linked = data_linked[a]
                b_linked = data_linked[b]
                new_obj = {
                    "entity_pair": [a, b],
                    "title_pair": [a_linked["title"], b_linked["title"]],
                    "matched_data": []
                }
                for stype in ["entity_rels", "entity_values"]:
                    for a_sent in a_linked[f"linked_{stype}"]:
                        for b_sent in b_linked[f"linked_{stype}"]:
                            
                            triple_pairs = [] # for a sentence, count its triple pairs
                            for a_statement in a_sent[2]:
                                pid = a_statement[0][1]
                                for b_statement in b_sent[2]:
                                    if b_statement[0][1] == pid: # comparable statement
                                        triple_pairs.append([a_statement[0], b_statement[0], a_statement[1], b_statement[1]])
                            if len(triple_pairs) == 0:
                                continue
                            _obj = {
                                "sentence_pair": [[a_sent[0], a_sent[1]], [b_sent[0], b_sent[1]]],
                                "triple_pairs": triple_pairs
                            }                    
                            new_obj["matched_data"].append(_obj)
                fw.write(json.dumps(new_obj) + '\n')
                cnt_write += 1
        print(cnt_write)
                       
entity_pairing()                          
                    