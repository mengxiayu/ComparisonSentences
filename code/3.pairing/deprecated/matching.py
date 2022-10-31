from pathlib import Path
import json
import time
import os
import pickle
import argparse
from collections import Counter
from itertools import product, combinations
import json
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from ahocorapy.keywordtree import KeywordTree

"""
This code is used for finding comparable entities by human-defined heuristitcs, e.g., two entities are comparable when they are of the same occupation.
However, we think it's not realistic to define rules for every entity type. And it's even more complicated for defining comparable statements. 
The code is deprecated now.
2022.10.31
"""

def get_entity_by_etype(target_etype):
    entity_list = []
    entity2type_data = pickle.load(open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/entity2type_textdata.pkl", 'rb'))
    for e,t in entity2type_data.items():
        if t == target_etype:
            entity_list.append(e)
    return set(entity_list)

def match_entities():
    print("=== Running Entity Matching!")
    map_ptype = {
        "Q5": "P106", 
        "Q482994": "P136", # album: genre
        "Q16521": "P171", # taxon: parent taxon
        "Q11424": "P136", # film: genre
        "Q134556": "P136", # music single: genre
        "Q7725634": "P136", # literatary work: genre
        "Q4830453": "P452", # business: industry
    }

    target_etype = "Q11424"
    target_ptype = map_ptype[target_etype]
    print("Condition:", target_etype, target_ptype)
    entity_list_linked = pickle.load(open(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/entity_set_linked_v1_combined.pkl", 'rb'))
    entity_list = get_entity_by_etype(target_etype)
    entity_list = entity_list & entity_list_linked
    print("entity list size", len(entity_list))
    dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data")
    dir_output = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/matched/{target_etype}")
    dir_output.mkdir(exist_ok=True, parents=True)

    feature2entity = {}
    label2entity = {}
    entity2label = {}
    entity2text = {}
    def in_text(entity_label, sent_list):
        for sent in sent_list:
            if entity_label in sent:
                return sent
        return None
    for split in ["AA", "AB"]:
        print(split)
        for batch_file in (dir_data / split).glob("wiki*"):
            # print(batch_file.name)
            f = open(batch_file)
            for line in f:
                obj = json.loads(line)
                qid = obj["qid"]
                if qid not in entity_list:
                    continue
                if obj["title"] in label2entity:
                    print(obj["title"])
                    continue
                label2entity[obj["title"]] = qid
                entity2label[qid] = obj["title"]
                entity2text[qid] = sent_tokenize(obj["text"])
                for s in obj["entity_rels"]:
                    pid, qid, vid = s.split('\t')
                    if pid == target_ptype:
                        if vid not in feature2entity:
                            feature2entity[vid] = set()
                        feature2entity[vid].add(qid)
    pickle.dump(entity2text, open(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity2text/{target_etype}.pkl", 'wb'))
    # entity2text = pickle.load(open(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity2text/{target_etype}.pkl", 'rb'))
    print("text data loaded!")

    
    print("total number of feature:", len(feature2entity))
    for feature, entity_set in sorted(feature2entity.items(), key=lambda x:len(x[1]), reverse=True): # from large to small
        print(feature, len(entity_set))
        if len(entity_set) < 2:
            break
        # build keyword tree
        kwtree = KeywordTree(case_insensitive=True)
        for e in entity_set:
            kwtree.add(entity2label[e])
        kwtree.finalize()

        
        f = open (dir_output / f"{feature}.json", 'w')
        result_objects = []
        for i, e in enumerate(entity_set):
            if i % 2000 == 0:
                print(i)
            all_occur_entity = {} # record all matched entities for this entity, and sentence-level evidence
            for sent in entity2text[e]:
                results = kwtree.search_all(sent)
                for label, _ in results:
                    e_matched = label2entity[label]
                    if e_matched == e:
                        continue
                    if e_matched in all_occur_entity:
                        continue
                    all_occur_entity[e_matched] = [e_matched, label, sent]
            matched_data = [v for v in all_occur_entity.values()]
            if len(matched_data) == 0:
                continue
            new_obj = {
                "qid": e,
                "label": entity2label[e],
                "matched": matched_data
            }
            result_objects.append(new_obj)
        if len(result_objects) == 0:
            continue
        for new_obj in result_objects:
            f.write(json.dumps(new_obj) + '\n')
        f.close()


match_entities()
