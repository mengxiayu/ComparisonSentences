import numpy as np
from pathlib import Path
import random
import json
import math
import csv
import pickle
import pandas as pd
from collections import Counter
from statistics import mean
import argparse
from itertools import combinations, product

"""
construct dataset (item, features, labels) for statment pair scoring.

Prerequisite tables for each entity type:
entity2profile.pkl
pid2freq.tsv
pid2value2freq.json

"""

class DataLoader:
    def __init__(self, target_etype):
        self.entity2profile = json.load(open(f"/Users/mengxiayu/Documents/Research/ComparisonSentences/data/wikidata_analysis/global_feature/entity2profile.json"))
        self.entity2indegree = json.load(open(f"/Users/mengxiayu/Documents/Research/ComparisonSentences/data/wikidata_analysis/global_feature/entity2indegree.json"))
        self.pid2freq_rel = json.load(open(Path(f"/Users/mengxiayu/Documents/Research/ComparisonSentences/data/wikidata_analysis/global_feature/pid2freq_rel.json")))
        self.pid2freq_val = json.load(open(Path(f"/Users/mengxiayu/Documents/Research/ComparisonSentences/data/wikidata_analysis/global_feature/pid2freq_val.json")))
        self.pid2value2freq = json.load(open(Path(f"/Users/mengxiayu/Documents/Research/ComparisonSentences/data/wikidata_analysis/global_feature/pid2value2freq.json")))
        self.total_prop_freq_rel = sum([v for k,v in self.pid2freq_rel.items()]) # total property frequency (number of statements)
        self.total_prop_freq_val = sum([v for k,v in self.pid2freq_val.items()])
        self.total_num_prop_rel = len(self.pid2freq_rel) # total property number of this entity type

    def load_positive_pairs(self, path):
        # load positive pairs from news matched data.
        positive_pair_freq = Counter()
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                p = obj["property"]
                e1, e2 = obj["entity_pair"]
                values_e1 = set([x[0][1] for x in obj["evidence_e1"]])
                values_e2 = set([x[0][1] for x in obj["evidence_e2"]])
                positive_pair_freq.update([(e1, e2, p, v1, v2) for v1 in values_e1 for v2 in values_e2])
        return positive_pair_freq

    def sample_negative_pairs(self, positive_pairs_freq, num_sample, sampling_method):
        positive_pairs = set(tuple(p) for p in positive_pairs_freq.keys())
        statement_pool = set() # keep track of statements
        entity_pair_pool = set() # keep track of the positive **entity** pairs
        property_pool = set()
        negative_pair_pool = set()
        for pos_pair in positive_pairs:
            e1, e2, p, v1, v2 = pos_pair
            entity_pair_pool.add((e1, e2))
            entity_pair_pool.add((e2, e1))
            property_pool.add(p)
            s1 = (e1, p, v1)
            s2 = (e2, p, v2)
            statement_pool.add(s1)
            statement_pool.add(s2) # this is sample from positive statement pair pool, only contain the successful properties.
            profile_e1 = self.build_entity_profile(e1)
            profile_e2 = self.build_entity_profile(e2)

            # to introduce more properties, we add other statements of the existing entities into the pool.
            if sampling_method == "random":
                for _p,v in profile_e1.items():
                    for _v1 in v:
                        statement_pool.add((e1, _p, _v1))

                for _p,v in profile_e2.items():
                    for _v2 in v:
                        statement_pool.add((e2, _p, _v2))

            elif sampling_method == "nard_negative":
                for _p in set(profile_e1.keys()) & set(profile_e2.keys()):
                    _v1 = profile_e1[_p][0]
                    _v2 = profile_e2[_p][0] # sample the first value
                    statement_pool.add((e1, _p, _v1))
                    statement_pool.add((e2, _p, _v2))
                    if (e1, e2, _p, _v1, _v2) not in positive_pairs and (e2, e1, _p, _v2, _v1) not in positive_pairs:
                        negative_pair_pool.add((e1, e2, _p, _v1, _v2))
                print("negative pair pool size", len(negative_pair_pool))
            else:
                print("Error: sampling method should be 'random' or 'hard negative'")
                return

        statement_pool = list(statement_pool) 
        print("statement pool size", len(statement_pool))
        

        negative_pairs = set()
        # Negative type one: comparable entities with not comparable properties.
        if sampling_method == "hard_negative":
            negative_pairs.update(random.sample(negative_pair_pool, int(num_sample * (1/2))))                
            print("sampled 50%", len(negative_pairs))

        # Negative type two: with frequently compared properties.
        _cnt = 0
        while len(negative_pairs) < num_sample:
            # sample 
            if _cnt > len(statement_pool)*len(statement_pool) / 2:
                break
            _s1, _s2 = random.sample(statement_pool, 2)
            _cnt += 1
            e1, p1, v1 = _s1
            e2, p2, v2 = _s2

            if p1 != p2: # only sample statement pairs with the same properties.
                continue
            if p1 == "P31":
                continue
            # if p1 not in property_pool:
            #     continue
            _pair = (e1, e2, p1, v1, v2)
            _pair_mirror = (e2, e1, p1, v2, v1)
            if _pair not in positive_pairs and _pair_mirror not in positive_pairs and _pair_mirror not in negative_pairs and _pair not in negative_pairs:
                negative_pairs.add(_pair)

        # Negative type three: random sample
        print("sampling done!")
        return negative_pairs

    def build_entity_profile(self, qid):
        statements = self.entity2profile[qid]
        profile = {}
        for s in statements:
            if len(s.split('\t')) != 3:
                continue
            pid, qid, vid = s.split('\t')
            if pid not in profile:
                profile[pid] = set()
            profile[pid].add(vid)
        return profile
    def value_diversity(self, vid2freq): 
        # property feature
        # measure the diversity of a property's values
        # H = -∑[(pi) * log(pi)],
        total_num = sum(v for k,v in vid2freq.items())
        H = 0
        for k,v in vid2freq.items():
            pi = (v/total_num)
            H += pi * math.log(pi)
        return -H
        
    def value_freq(self, vid2freq, vid):
        # value feature
        # measure the global frequency of a value
        total_num = sum(v for k,v in vid2freq.items())
        freq = vid2freq[vid] / total_num
        return freq
    
    def get_indegree(self, e):
        return self.entity2indegree[e] if e in self.entity2indegree else 0

    def data2feature(self, pair):
        e1, e2, p, v1, v2 = pair
        
        if e1 not in self.entity2profile or e2 not in self.entity2profile:
            return None
        # profile_e1 = self.entity2profile[e1]
        # profile_e2 = self.entity2profile[e2]
        profile_e1 = self.build_entity_profile(e1)
        profile_e2 = self.build_entity_profile(e2)
        indegree_e1 = self.get_indegree(e1)
        indegree_e2 = self.get_indegree(e2)

        def degree_e(profile_e):
            # number of entity_rels of an entity
            degree = 0
            prop_rel = 0
            prop_val = 0
            for _p, _values in profile_e.items():
                if _p in self.pid2freq_rel:
                    degree += len(_values)
                    prop_rel += 1
                else:
                    prop_val += 1
            return degree, prop_rel, prop_val

        # 1. Entity pair features
        degree_e1, num_prop_rel_e1, num_prop_val_e1 = degree_e(profile_e1)
        degree_e2, num_prop_rel_e2, num_prop_val_e2 = degree_e(profile_e2)

        avg_rel_coverage = (num_prop_rel_e1 + num_prop_rel_e2) / self.total_num_prop_rel
        min_rel_coverage = min(num_prop_rel_e1, num_prop_rel_e2) / self.total_num_prop_rel
        max_rel_coverage = max(num_prop_rel_e1, num_prop_rel_e2) / self.total_num_prop_rel
        diff_rel_coverage = abs(num_prop_rel_e1 - num_prop_rel_e2) / self.total_num_prop_rel
        common_prop = set(profile_e1.keys()) & set(profile_e2.keys())
        num_common_prop = len(common_prop)
        common_prop_rate = len(common_prop) / (len(profile_e1) + len(profile_e2)) # #comon_prop / #total_prop_of_two_entities

        # common features (values) of two entities (common neighbors)
        common_feature_prop = []
        prop_freq = [] # for each common property, record their frequency (the number of associated statements)
        value_freq = [] # for each common value
        
        common_value_degrees = [] # if their common value is entity, record this entity's degree and indegree
        common_value_indegrees = []

        for _p in common_prop:
            # if _p == 'P31':
            #     continue # skip "instance of" property
            if _p not in self.pid2value2freq:
                print(_p)
                continue
            _prop_freq = self.pid2freq_rel[_p]/ self.total_prop_freq_rel if _p in self.pid2freq_rel else self.pid2freq_val[_p] / self.total_prop_freq_val # the popularity of a common property
            prop_freq.append(_prop_freq) # 0~1 
            common_values = set(profile_e1[_p]) & set(profile_e2[_p])
            if len(common_values) > 0:
                common_feature_prop.append(_p) # common properties with same value
                _total_value_freq = sum([v for k,v in self.pid2value2freq[_p].items()])

                for _v in common_values:
                    if _v not in self.pid2value2freq[_p]:
                        print(_p, _v)
                    indegree_common_v = self.get_indegree(_v)
                    if _v in self.entity2profile:
                        profile_common_v = self.build_entity_profile(_v)
                        degree_common_v, _, _ = degree_e(profile_common_v)
                    else:
                        degree_common_v = 0
                    common_value_degrees.append(degree_common_v)
                    common_value_indegrees.append(indegree_common_v)
                # how popular are their common values to that property
                # for a paticular common property, the average frequency of common value.
                _v_freq = mean([self.pid2value2freq[_p][_v] for _v in common_values]) / _total_value_freq # 0～1
                value_freq.append(_v_freq)
        if len(value_freq) == 0:
            value_freq.append(0)


        if len(prop_freq) == 0:
            print("fail",pair, prop_freq, common_prop)
        num_common_feature = len(common_feature_prop)
        common_feat_rate = len(common_feature_prop) / len(common_prop) # among common properties, how many have common value
        sorted_prop_freq = sorted(prop_freq, reverse=True)

        avg_common_prop_freq = mean(prop_freq) 
        max_common_prop_freq = sorted_prop_freq[0]
        second_common_prop_freq = sorted_prop_freq[1] if  len(sorted_prop_freq) > 1 else sorted_prop_freq[0]
        min_common_prop_freq = sorted_prop_freq[-1]
        avg_common_value_freq = mean(value_freq)
        max_common_value_freq = max(value_freq)
        min_common_value_freq = min(value_freq)

        # common value degree: the degree of entity when common value is an entity
        nonzero_common_value_degrees = sorted([x for x in common_value_degrees if x > 0], reverse=True)
        min_common_v_degree = nonzero_common_value_degrees[-1] if len(nonzero_common_value_degrees) > 0 else 0
        max_common_v_degree = nonzero_common_value_degrees[0] if len(nonzero_common_value_degrees) > 0 else 0
        second_common_v_degree = nonzero_common_value_degrees[1] if len(nonzero_common_value_degrees) > 1 else max_common_v_degree
        
        # TODO add common value indegree feature
        nonzero_common_value_indegrees = sorted([x for x in common_value_indegrees if x > 0], reverse=True)
        min_common_v_indegree = nonzero_common_value_indegrees[-1] if len(nonzero_common_value_indegrees) > 0 else 0
        max_common_v_indegree = nonzero_common_value_indegrees[0] if len(nonzero_common_value_indegrees) > 0 else 0
        second_common_v_indegree = nonzero_common_value_indegrees[1] if len(nonzero_common_value_indegrees) > 1 else max_common_v_indegree


        # 2. Property feature
        # the relationship of the current property and the common properties.
        # the popularity of the property in some entity group. define entity group: (?,a,v), also only consider entity_rels as value

        property_freq = self.pid2freq_rel[p] if p in self.pid2freq_rel else self.pid2freq_val[p]
        property_freq /= (self.total_prop_freq_rel + self.total_prop_freq_val)
        property_diversity = self.value_diversity(self.pid2value2freq[p])
        
        # 3. Value features
        total_value_freq = sum([v for k,v in self.pid2value2freq[p].items()])
        freq_v1 = self.pid2value2freq[p][v1] / total_value_freq
        freq_v2 = self.pid2value2freq[p][v2] / total_value_freq
        avg_value_freq = mean([freq_v1, freq_v2])
        diff_value_freq = abs(freq_v1 - freq_v2)
        
        # entity feature of value
        # if value is entity, add entity degree, else add 0?

        if v1 in self.entity2profile:
            profile_v1 = self.build_entity_profile(v1)
            degree_v1, num_prop_rel_v1, num_prop_val_v1 = degree_e(profile_v1)
        else:
            profile_v1 = None
            degree_v1 = 0
        if v2 in self.entity2profile:
            profile_v2 = self.build_entity_profile(v2)
            degree_v2, num_prop_rel_v2, num_prop_val_v2 = degree_e(profile_v2)
        else:
            profile_v2 = None
            degree_v2 = 0
        
        # if profile_v1 and profile_v2:
        #     common_prop_rate_v = 
    
        indegree_v1 = self.get_indegree(v1)
        indegree_v2 = self.get_indegree(v2)


        feature = [degree_e1, degree_e2, indegree_e1, indegree_e2, avg_rel_coverage, min_rel_coverage, max_rel_coverage, diff_rel_coverage, num_common_prop, common_prop_rate, num_common_feature, common_feat_rate, avg_common_prop_freq, max_common_prop_freq, min_common_prop_freq, avg_common_value_freq, max_common_value_freq, min_common_value_freq, min_common_v_indegree, max_common_v_indegree, second_common_v_indegree,  property_freq, property_diversity, freq_v1, freq_v2, avg_value_freq, diff_value_freq, degree_v1, degree_v2, indegree_v1, indegree_v2, second_common_prop_freq, min_common_v_degree, max_common_v_degree, second_common_v_degree]
        return feature

    def create_training_data(self, etype2positive_pairs_freq, num_positive, sampling_method="random"):  
        global_positive_pairs = []
        global_negative_pairs = []
        etype2size = {}
        for etype, positive_pairs_freq in etype2positive_pairs_freq.items():

            all_positive_pairs = list(positive_pairs_freq.keys())
            random.shuffle(all_positive_pairs)
            # print("all positive pairs", len(all_positive_pairs))
            positive_pairs = list(all_positive_pairs)[:num_positive] if len(all_positive_pairs) > num_positive else all_positive_pairs

            # sample negatives using global positive pairs
            negative_pairs = self.sample_negative_pairs(positive_pairs_freq, len(positive_pairs) , sampling_method)
            negative_pairs = list(negative_pairs)
            # print("data pairs loaded!", len(positive_pairs), len(negative_pairs))
            # global_positive_pairs.extend(positive_pairs)
            # global_negative_pairs.extend(negative_pairs)
            for pair in positive_pairs:
                freq = positive_pairs_freq[pair]
                global_positive_pairs.append((pair, freq))
            for pair in negative_pairs:
                global_negative_pairs.append((pair, 0))
            etype2size[etype] = len(global_positive_pairs) + len(global_negative_pairs)
        return global_positive_pairs, global_negative_pairs, etype2size


    def write_data(self, positive_pairs, negative_pairs, dir_output, num_positive, fname_output):
        dir_output.mkdir(exist_ok=True, parents=True)
        fw = open(dir_output / fname_output, 'w')
        writer = csv.writer(fw)
        writer.writerow(["pair", "degree_e1", "degree_e2", "indegree_e1", "indegree_e2", "avg_rel_coverage", "min_rel_coverage", "max_rel_coverage", "diff_rel_coverage", "num_common_prop", "common_prop_rate", "num_common_feature", "common_feat_rate", "avg_common_prop_freq", "max_common_prop_freq", "min_common_prop_freq", "avg_common_value_freq", "max_common_value_freq", "min_common_value_freq", "min_common_v_indegree", "max_common_v_indegree", "second_common_v_indegree", "property_freq", "property_diversity", "freq_v1", "freq_v2", "avg_value_freq", "diff_value_freq", "degree_v1", "degree_v2", "indegree_v1", "indegree_v2",  "second_common_prop_freq", "min_common_v_degree", "max_common_v_degree", "second_common_v_degree", "label"])
        for pair, freq in positive_pairs:
            # freq = positive_pairs_freq[pair]
            feature = [pair]
            feature.extend(self.data2feature(pair))
            feature.append(math.log(1+freq)) # positive label
            writer.writerow(feature)

        for pair, freq in negative_pairs:
            feature = [pair]
            _feature = self.data2feature(pair)
            if _feature is None:
                continue
            feature.extend(_feature)
            feature.append(freq) # negative label
            writer.writerow(feature)
        fw.close()
    
    def create_all_data(self, dir_matched, dir_output, fname_output, num_positive, sampling_method):
        etype2positive_pairs_freq = {}
        cnt = 0
        for path_matched in dir_matched.glob("*_matched.json"):
            # if cnt == 3:
            #     break
            cnt += 1
            etype = path_matched.stem.rstrip("_matched")
            etype2positive_pairs_freq[etype] = self.load_positive_pairs(path_matched)
        print("loaded positive pairs for types", len(etype2positive_pairs_freq))
        global_positive_pairs, global_negative_pairs, etype2size = self.create_training_data(etype2positive_pairs_freq, num_positive, sampling_method)
        print("positive_pairs", len(global_negative_pairs), len(global_negative_pairs))
        # write training data
        self.write_data(global_positive_pairs, global_negative_pairs, dir_output, num_positive, fname_output)
        print("positive", len(global_positive_pairs), "negative", len(global_negative_pairs), "data points written!")
        with open (dir_output / "etype2datasize.json", 'w') as f:
            json.dump(etype2size, f)
        # write testing data
        # write_data(test_positive_pairs, test_negative_pairs, dir_outpupt, "test_"+fname_output)
        # print("positive", len(test_positive_pairs), "negative", len(test_negative_pairs), "test data points written!")


    def save_positive_pairs(self, path, path_output):
        with open(path_output, 'w') as f:
            positive_pairs = self.load_positive_pairs(path)
            print(len(positive_pairs))
            f.write(json.dumps(list(positive_pairs)))
        print("positive pairs written!")


    def load_inference_data(self, dir_linked, dir_output):
        # 1. load entities and linked statements of a same etype, and then generate pairs. and then get features.
        entity2type_data = pickle.load(open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/entity2type_textdata.pkl", 'rb'))
        print("# entity2type_data", len(entity2type_data))
        etype2entities = {}
        for e, types in entity2type_data.items():
            for etype in types:
                if etype not in etype2entities:
                    etype2entities[etype] = set()
                etype2entities[etype].add(etype)
        print("# etype", len(etype2entities))

        def load_linked_triples(dir_linked):
            entity2triples = {} # linked triples
            cnt_triples = 0
            for split in ["AA", "AB"]:
                for batch_file in (dir_linked / split).glob("wiki_"):
                    with open(batch_file) as f:
                        for line in f:
                            obj = json.loads(line)
                            qid = obj["qid"]
                            if qid in entity2triples:
                                continue
                            entity2triples[qid] = set()
                            for s_type in ["linked_entity_rels", "linked_entity_values"]:
                                for _sent in obj[s_type]:
                                    for _triple in _sent[2]:
                                        _, pid, vid = _triple[0]
                                        entity2triples[qid].add((pid, vid))
                                        cnt_triples += 1
            print("# linked triples", cnt_triples)
            print("# linked entities", len(entity2triples))
            return entity2triples
        entity2triples = load_linked_triples(dir_linked) # linked triples
        
        # 2. filter by etype, and make pairs
        def _build_entity_profile_linked(triples):
            # property to values set
            profile = {}
            for p,v in triples:
                if p not in profile:
                    profile[p] = set()
                profile[p].add(v)
            return profile
        cnt = 0
        batch = 0
        dir_output.mkdir(exists_ok=True, parents=True)
        fw = open(dir_output / f"pairs_{batch}.json")
        print("batch", batch)
        for etype, entities in etype2entities.items():
            for e1, e2 in combinations(entities, 2): # entity pairs
                triples_e1 = entity2triples[e1]
                triples_e2 = entity2triples[e2]

                profile_e1 = _build_entity_profile_linked(triples_e1)
                profile_e2 = _build_entity_profile_linked(triples_e2)
                
                common_props = set(profile_e1.keys()) & set(profile_e2.keys())
                # if no property overlap, then skip
                if not common_props :
                    continue
                for p in common_props:
                    for v1, v2 in product(profile_e1[p], profile_e2[p]):
                        pair = (e1, e2, p, v1, v2)
                        pair_id = cnt
                        output = f"{pair_id}\t{etype}\t{pair}"
                        fw.write(output + '\n')
                        cnt += 1
                        if cnt > 100000:
                            fw.close()
                            cnt = 0
                            batch += 1
                            print("batch", batch)
                            fw = open(dir_output / f"pairs_{batch}.json")

                            
                        

        


            


        

        

                        
                        

                        
        return

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_etype', type=str, default=None, help='instance of')
    parser.add_argument('--file_output', type=str, default=None, help='output file name')
    parser.add_argument('--num_positive', type=int, default=None, help='number of positive examples')
    parser.add_argument('--sampling_method', type=str, default=None, help='random or hard_negative')
    parser.add_argument('--dir_linked', type=str, default='/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1/combined', help='data of linked Wikipedia and Wikidata')

    return parser



def main():
    args = get_arg_parser().parse_args()
    print(args)
    assert args.sampling_method in ["random", "hard_negative"]
    dir_matched = Path(f"/Users/mengxiayu/Documents/Research/ComparisonSentences/data/statement_scoring/news_v1")
    dir_output = Path(f"/Users/mengxiayu/Documents/Research/ComparisonSentences/experiments/pair_scoring/global")

    loader = DataLoader(args.target_etype)
    print("# entity", len(loader.entity2profile))
    loader.create_all_data(dir_matched, dir_output, args.file_output, args.num_positive, args.sampling_method)

if __name__ == "__main__":
    main()



