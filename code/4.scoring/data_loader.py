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


"""
construct dataset (item, features, labels) for statment pair scoring.

Prerequisite tables for each entity type:
entity2profile.pkl
pid2freq.tsv
pid2value2freq.json


"""

class DataLoader:
    def __init__(self, target_etype):
        self.entity2profile = json.load(open(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/{target_etype}/entity2profile.json"))
        self.pid2freq_rel = json.load(open(Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/{target_etype}/pid2freq_rel.json")))
        self.pid2freq_val = json.load(open(Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/{target_etype}/pid2freq_val.json")))
        self.pid2value2freq = json.load(open(Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/{target_etype}/pid2value2freq.json")))
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
            profile_e1 = self.entity2profile[e1]
            profile_e2 = self.entity2profile[e2]

            # to introduce more properties, we add other statements of the existing entities into the pool.
            if sampling_method == "normal":
                for _p,v in profile_e1.items():
                    _v1 = v[0]
                    statement_pool.add((e1, _p, _v1))

                for _p,v in profile_e2.items():
                    _v2 = v[0]
                    statement_pool.add((e2, _p, _v2))

            elif sampling_method == "nard_negative":
                for _p in set(profile_e1.keys()) & set(profile_e2.keys()):
                    _v1 = profile_e1[_p][0]
                    _v2 = profile_e2[_p][0] # sample the first value
                    statement_pool.add((e1, _p, _v1))
                    statement_pool.add((e2, _p, _v2))
                    if (e1, e2, _p, _v1, _v2) not in positive_pairs and (e2, e1, _p, _v2, _v1) not in positive_pairs:
                        negative_pair_pool.add((e1, e2, _p, _v1, _v2))

            

        statement_pool = list(statement_pool) 
        
        print("statement pool size", len(statement_pool))
        print("negative pair pool size", len(negative_pair_pool))

        negative_pairs = set()

        # Negative type one: comparable entities with not comparable properties.
        if sampling_method == "hard_negative":
            negative_pairs.update(random.sample(negative_pair_pool, int(num_sample * (1/2))))                
            print("sampled 50%", len(negative_pairs))

        # Negative type two: with frequently compared properties.
        while len(negative_pairs) < num_sample:
            _s1, _s2 = random.sample(statement_pool, 2)
            e1, p1, v1 = _s1
            e2, p2, v2 = _s2

            if p1 != p2: # only sample statement pairs with the same properties.
                continue
            # if p1 not in property_pool:
            #     continue
            _pair = (e1, e2, p1, v1, v2)
            _pair_mirror = (e2, e1, p1, v2, v1)
            if _pair not in positive_pairs and _pair_mirror not in positive_pairs:
                negative_pairs.add(_pair)

        # Negative type three: random sample
        print("sampling done!")
        return negative_pairs


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
    
    
    def data2feature(self, pair):
        e1, e2, p, v1, v2 = pair
        
        if e1 not in self.entity2profile or e2 not in self.entity2profile:
            return None
        profile_e1 = self.entity2profile[e1]
        profile_e2 = self.entity2profile[e2]

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

        avg_rel_coverage = (num_prop_rel_e1 + num_prop_rel_e2) / (2* self.total_num_prop_rel)
        diff_rel_coverage = abs(num_prop_rel_e1 - num_prop_rel_e2) / self.total_num_prop_rel
        common_prop = set(profile_e1.keys()) & set(profile_e2.keys())
        common_prop_rate = len(common_prop) / (len(profile_e1) + len(profile_e2)) # #comon_prop / #total_prop_of_two_entities
        

        common_feature_prop = []
        prop_freq = []
        value_freq = []
        for _p in common_prop:
            common_values = set(profile_e1[_p]) & set(profile_e2[_p])
            if len(common_values) > 0:
                common_feature_prop.append(_p)
                if _p in self.pid2freq_rel:
                    _prop_freq = self.pid2freq_rel[_p]
                prop_freq.append(_prop_freq / self.total_prop_freq_rel) # 0~1
                _total_value_freq = sum([v for k,v in self.pid2value2freq[_p].items()])
                if _p not in self.pid2value2freq:
                    print(_p)
                for _v in common_values:
                    if _v not in self.pid2value2freq[_p]:
                        print(_p, _v)
                _v_freq = mean([self.pid2value2freq[_p][_v] for _v in common_values]) / _total_value_freq # 0～1
                value_freq.append(_v_freq)

        
        common_feat_rate = len(common_feature_prop) / len(common_prop) # among common properties, how many have common value
        avg_common_prop_freq = mean(prop_freq)
        max_common_prop_freq = max(prop_freq)
        min_common_prop_freq = mean(prop_freq)
        avg_common_value_freq = mean(value_freq)
        max_common_value_freq = max(value_freq)
        min_common_value_freq = min(value_freq)

        

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


        feature = [degree_e1, degree_e2, avg_rel_coverage, diff_rel_coverage, common_prop_rate, common_feat_rate, avg_common_prop_freq, max_common_prop_freq, min_common_prop_freq, avg_common_value_freq, max_common_value_freq, min_common_value_freq , property_freq, property_diversity, freq_v1, freq_v2, avg_value_freq, diff_value_freq]
        return feature

    def create_training_data(self, path_matched, dir_positive, dir_output, fname_output, num_positive, sampling_method="random"):    
        if dir_positive is None:
            positive_pairs_freq = self.load_positive_pairs(path_matched)
        else:
            with open(dir_positive / "all_positive_pairs.json", 'r') as f:
                positive_pairs_freq = json.load(f)
        negative_pairs = self.sample_negative_pairs(positive_pairs_freq, num_positive, sampling_method)
        positive_pairs = list(positive_pairs_freq.keys())
        positive_pairs = list(positive_pairs)[:num_positive] if len(positive_pairs) > num_positive else positive_pairs
        print("training pairs loaded!", len(positive_pairs), len(negative_pairs))
        # dir_output = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/pair_scoring/Q5")
        dir_output.mkdir(exist_ok=True, parents=True)
        fw = open(dir_output / fname_output, 'w')
        writer = csv.writer(fw)
        writer.writerow(["pair", "degree_e1", "degree_e2", "avg_rel_coverage", "diff_rel_coverage", "common_prop_rate", "common_feat_rate", "avg_common_prop_freq", "max_common_prop_freq", "min_common_prop_freq", "avg_common_value_freq", "max_common_value_freq", "min_common_value_freq", "property_freq", "property_diversity", "freq_v1", "freq_v2", "avg_value_freq", "diff_value_freq",  "label"])
        for pair in positive_pairs:
            freq = positive_pairs_freq[pair]
            feature = [pair]
            feature.extend(self.data2feature(pair))
            feature.append(math.log(1+freq)) # positive label
            writer.writerow(feature)

        for pair in negative_pairs:
            feature = [pair]
            _feature = self.data2feature(pair)
            if _feature is None:
                continue
            feature.extend(_feature)
            feature.append(0)
            writer.writerow(feature)
        fw.close()

        print(len(positive_pairs) + len(negative_pairs), "data points written!")

    def save_positive_pairs(self, path, path_output):

        with open(path_output, 'w') as f:
            positive_pairs = self.load_positive_pairs(path)
            print(len(positive_pairs))
            f.write(json.dumps(list(positive_pairs)))
        print("positive pairs written!")


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_etype', type=str, default=None, help='instance of')
    parser.add_argument('--file_output', type=str, default=None, help='output file name')
    parser.add_argument('--num_positive', type=int, default=None, help='number of positive examples')
    parser.add_argument('--sampling_method', type=str, default=None, help='random or hard_negative')

    return parser



def main():
    args = get_arg_parser().parse_args()
    print(args)
    assert args.sampling_method in ["random", "hard_negative"]
    loader = DataLoader(args.target_etype)
    

    print("# entity", len(loader.entity2profile))
    dir_matched = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/statement_scoring/news/{args.target_etype}_matched.json")
    dir_output = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/pair_scoring/{args.target_etype}")

    loader.create_training_data(path_matched=dir_matched, dir_positive=None, dir_output=dir_output, fname_output=args.file_output, num_positive=args.num_positive, sampling_method=args.sampling_method)

if __name__ == "__main__":
    main()


