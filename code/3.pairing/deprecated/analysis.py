
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


'''
In this script we group entities by important (frequent) property value.
instance of : top 20 values
filter by (instance of, value) -> qid lists
for each, calculate top 20 properties
filter by (property, vluaes) -> qid lists
'''
dir_entity_rels = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/entity_rels")
dir_entity_aliases = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/aliases_sorted")


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

def _lookup_qid2alias(qid):
    if len(qid) < 3:
        table_name = f"Q{int(qid[1]):02d}.tsv"
    else:
        table_name = f"Q{qid[1:3]}.tsv"
    record = None
    with open (dir_entity_aliases / table_name, 'r') as f:
        for line in f:
            if not line.startswith(qid):
                continue
            record = line.strip().split('\t')
            if record[0] == qid:
                return record[1].split('|sep|')
    return None

def filter_P31():
    path_output = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/")
    path_output.mkdir(parents=True, exist_ok=True)
    target_property = "P31"
    fw = open(path_output / f"{target_property}.tsv", 'w')
    for batch_file in os.listdir(dir_entity_rels):
        with open(dir_entity_rels / batch_file) as f:
            header = f.readline()
            cnt = 0
            assert header.strip().split('\t') == ["property_id", "qid", "value"]
            for line in f:
                pid, qid, vid = line.strip('\n').split('\t')
                if pid == target_property:
                    fw.write(line)
    fw.close()

def analysis_property_distribution():

    # cnt_property = {}
    # '''
    # Record the property distribution of each entity type.
    # "article": {"x1": y1, "x2": y2, ...}
    # '''
    # with open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/cnt_P31.tsv") as f:    
    #     for line in f:
    #         freq, qid, alias = line.strip().split('\t')
    #         cnt_property[qid] = Counter()

    # entity2type = pickle.load(open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/entity2type.pkl", 'rb'))
    # # enumerate all statements and count their statement values
    # for batch_file in os.listdir(dir_entity_rels):
    #     with open(dir_entity_rels / batch_file) as f:
    #         header = f.readline()
    #         cnt = 0
    #         assert header.strip().split('\t') == ["property_id", "qid", "value"]
    #         for line in f:
    #             pid, qid, vid = line.strip('\n').split('\t')
    #             if qid in entity2type: # if this entity belongs to a type
    #                 entity_type = entity2type[qid]
    #                 cnt_property[entity_type][pid] += 1
    # pickle.dump(cnt_property, open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/cnt_property.pkl", 'wb'))


    path_prop_alias = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_aliases/0.tsv")
    pid2alias = _load_property2aliases(path_prop_alias)
    print("pid2alias loaded. size:", len(pid2alias))

    cnt_property = pickle.load(open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/cnt_property.pkl", 'rb'))
    with open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/frequent_properties.txt", 'w') as f:
        for entity_type, cnt in cnt_property.items():
            entity_type_label = _lookup_qid2alias(entity_type)
            f.write(f"\n==={entity_type}\t{entity_type_label}===\n")
            for pid, freq in cnt.most_common(30):
                alias = pid2alias[pid] if pid in pid2alias else ""
                f.write(f"{freq}\t{pid}\t{alias}\n")
        
    

'''0. preprocessing: dump a subset of data for later use'''



def scoring(dist: Counter):
    # score property by value distribution
    return (sum(dist.values())**2 / len(dist))


'''1. property ranking'''
def rank_properties(target_etype, scoring_func, topk) -> set:
    # needs text_data
    entity2type_data = pickle.load(open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/entity2type_textdata.pkl", 'rb'))
    dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data")
    path_prop_alias = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_aliases/0.tsv")
    pid2alias = _load_property2aliases(path_prop_alias)
    
    cnt_value = {}
    num_entity = 0
    for split in ["AA", "AB"]:
        for batch_file in (dir_data / split).glob("wiki*"):
            with open(batch_file ) as f:
                for line in f:
                    obj = json.loads(line)
                    qid = obj["qid"]
                    if qid not in entity2type_data:
                        continue
                    if entity2type_data[qid] != target_etype:
                        continue
                    num_entity += 1
                    for s in obj["entity_rels"]:
                        pid, qid, vid = s.split('\t')    
                        if pid not in cnt_value:
                            cnt_value[pid] = Counter()
                        cnt_value[pid][vid] += 1

    
    print("target etype:", target_etype)
    print("total number of entity:", num_entity)

    topk_properties = []
    for pid, v_dist in sorted(cnt_value.items(), key=lambda x:scoring_func(x[1]), reverse=True)[:topk]:
        # num_vtype = len(v_dist)
        # num_entry = sum(list(v_dist.values()))
        # score = scoring_func(v_dist)
        # print(pid, pid2alias[pid][0], num_vtype, num_entry, score)
        topk_properties.append(pid)
    pickle.dump(topk_properties, open(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/top_properties_{target_etype}_{topk}.pkl", 'wb'))
    return topk_properties


'''3. entity filtering'''
def dump_linked_entity_ids(linking_version="linked_v1", linking_criteria="combined"):
    entity_list = []
    dir_linked = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/{linking_version}/{linking_criteria}")
    for split in ["AA", "AB"]:
        for batch_file in (dir_linked / split).glob("wiki*"):
            with open(batch_file) as f:
                for line in f:
                    obj = json.loads(line)
                    entity_list.append(obj["qid"])
    print(len(entity_list))
    entity_list = set(entity_list)
    print(len(entity_list))
    pickle.dump(entity_list, open(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/entity_set_{linking_version}_{linking_criteria}.pkl", 'wb'))

'''2. construct entity_feature'''
def dump_entity_features(entity_set: set, topk_properties: set):
    entity_feature = {}

    dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data_Q5")
    for split in ["AA", "AB"]:
        for batch_file in (dir_data / split).glob("wiki*"):

            f = open(batch_file)
            for line in f:
                obj = json.loads(line)
                for s in obj["entity_rels"]:
                    pid, qid, vid = s.split('\t')
                    if pid not in topk_properties:
                        continue
                    if qid not in entity_set:
                        continue
                    if qid not in entity_feature:
                        entity_feature[qid] = {}
                    if pid not in entity_feature[qid]:
                        entity_feature[qid][pid] = set()
                    entity_feature[qid][pid].add(vid)
            f.close()
    pickle.dump(entity_feature, open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/entity_feature_Q5.pkl", 'wb'))






                    

                    
                    
        
