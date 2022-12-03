
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
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop -= set(['he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself']) # we want to keep these forms to link statements about gender
from news_matching import _load_property2aliases, _load_property2datatype, _load_qid2aliases, get_entity_by_etype, match_sentence_eav, _reformat_value
# from ahocorapy.keywordtree import KeywordTree
import ahocorasick

"""
Match comparable statement pairs from Wikipedia and generate "Qx_matched.json" file
"""


path_prop_datatype = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_labels/0.tsv")
pid2datatype = _load_property2datatype(path_prop_datatype)
print("pid2datatype loaded. size:", len(pid2datatype))

# load pid2alias
path_prop_alias = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_aliases/0.tsv")
pid2alias = _load_property2aliases(path_prop_alias)
print("pid2alias loaded. size:", len(pid2alias))

# load qid2alias
dir_entity_aliases = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/aliases_sorted")
qid2alias = _load_qid2aliases(dir_entity_aliases)
print("qid2alias loaded. size:", len(qid2alias))

def get_occurrence(target_etype):
    dir_output = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/statement_scoring/wiki")
    dir_output.mkdir(parents=True, exist_ok=True)
    # path_news = Path("/afs/crc.nd.edu/group/dmsquare/vol1/data/EngGigV5/corpus/data_enggigv5.txt")
    dir_wikipedia = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data_by_type/text_data_{target_etype}")
    
    # num_lines_news = 9870506
    # assert path_news.exists()
    dir_linked = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1/combined")
    assert dir_linked.exists()
    entity_set_etype = get_entity_by_etype(target_etype)

    print("entity set etype size:", len(entity_set_etype))


    # 1. get entity2label from linked data, constrained by entity type
    entity2label = {}
    label2entity = {}
    entity2triples = {} # linked properties
    for split in ["AA", "AB"]: 
        for batch_file in (dir_linked / split).glob("wiki*"):
            # print(batch_file.name)
            f = open(batch_file)
            for line in f:
                obj = json.loads(line)
                qid = obj["qid"]
                if qid not in entity_set_etype:
                    continue
                entity2label[qid] = obj["title"]
                if obj["title"] in entity2label:
                    print(obj["title"])
                    continue
                label2entity[obj["title"]] = qid
                entity2triples[qid] = set()
                for s_type in ["linked_entity_rels", "linked_entity_values"]:
                # for s_type in ["linked_entity_values"]:
                    for _sent in obj[s_type]:
                        for _triple in _sent[2]:
                            _, pid, vid = _triple[0]
                            entity2triples[qid].add((pid, vid))           
    print("entity2label size", len(entity2label))

    # 2. build keyword tree
    start = time.time()
    kwtree = ahocorasick.Automaton()
    for idx, label in enumerate(entity2label.values()):
        kwtree.add_word(label, (idx, label))
    kwtree.make_automaton()

    # print("keyword tree built!", time.time() - start)

    cnt_matched = 0
    # 3. match in wikipedia data. Statement linking criteria: (e, a). Statment pairing criteria: 
    fw = open(dir_output / f"{target_etype}_matched.json", 'w')
    for split in ["AA", "AB"]: 
        for batch_file in (dir_wikipedia / split).glob("wiki*"): 
            # print(batch_file)
            with open (batch_file) as f:
                # for line in tqdm(f, total=num_lines_news):
                for linenum,line in enumerate(f):
                    # if linenum % 1000 == 0:
                    #     print(linenum)
                    line = json.loads(line)["text"] # split paragraphs
                    matched_labels = set() # record all matched entities
                    for item in kwtree.iter(line):
                        matched_labels.add(item[1][1])
                    if len(matched_labels) < 2:
                        continue
                    matched_qid2pid2triples = {} # eid: set(pid), [pid, alias, sent]
                    for sent in sent_tokenize(line.strip()):
                        _entities = [label for label in matched_labels if label in sent]
                        if len(_entities) == 0: # no entity matched this sentence
                            continue 
                        tokenized_sent = word_tokenize(sent)
                        for label in _entities:
                            e = label2entity[label]
                            linked_triples = entity2triples[e]
                            for pid, vid in linked_triples:
                                q_aliases = qid2alias[e] if e in qid2alias else []
                                p_aliases = pid2alias[pid] if pid in pid2alias else []
                                v_aliases = qid2alias[vid] if vid in qid2alias else _reformat_value(pid, vid)
                                matched_triple = match_sentence_eav(sent, tokenized_sent, (q_aliases, p_aliases, v_aliases))
                                if matched_triple is None:
                                    continue

                                if e not in matched_qid2pid2triples:
                                    matched_qid2pid2triples[e] = {}
                                if pid not in matched_qid2pid2triples[e]:
                                    matched_qid2pid2triples[e][pid] = []
                                matched_qid2pid2triples[e][pid].append([(pid, vid), matched_triple, sent])

                    
                    for e1, e2 in combinations(list(matched_qid2pid2triples.keys()), 2):
                        # enumerate all entity pairs
                        common_properties = set(matched_qid2pid2triples[e1].keys()) & set(matched_qid2pid2triples[e2].keys())
                        if len(common_properties) == 0:
                            continue
                        aliases_e1 = qid2alias[e1]
                        aliases_e2 = qid2alias[e2]
                        _flag = False
                        for a1, a2 in product(aliases_e1, aliases_e2):
                            if a1 in a2 or a2 in a1:
                                _flag = True
                                break
                        if _flag:
                            continue
                        for p in common_properties:
                            s1 = matched_qid2pid2triples[e1][p]
                            s2 = matched_qid2pid2triples[e2][p]            
                            matched_statement = {
                                "entity_pair": (e1, e2),
                                "property": p,
                                "evidence_e1": s1,
                                "evidence_e2": s2
                            }
                            cnt_matched += 1

                            fw.write(json.dumps(matched_statement)+'\n')
                            # print(matched_statement)
    fw.close()
    return cnt_matched
def get_arg_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--target_etype', type=str, default=None, help='etype')
    parser.add_argument('--dir_etype', type=str, default="/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data_by_type", help='text data by etype dir')
    return parser



if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    print(args)
    dir_etype = Path(args.dir_etype)
    all_etypes = set()
    for p in dir_etype.glob("*text_data_Q*"):
        etype = p.name.split('_')[-1]
        all_etypes.add(etype)
    print(f"Start matching for {len(all_etypes)} entity types")
    for i,etype in enumerate(all_etypes):
        assert etype.startswith("Q")
        print(i, etype)
        cnt_matched = get_occurrence(etype)
        print("cnt matched:", cnt_matched)
    print("Finish matching!")

                    
                    



            

    

