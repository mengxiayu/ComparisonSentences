
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


"""
Match comparable statement pairs from news and generate "Qx_matched.json" file
requirements:
- linked data
- news data
- alias table

criteria:
- q cannot be substring of v
- no stop words
- q1 cannot be substring of q2
"""


# from ahocorapy.keywordtree import KeywordTree
import ahocorasick

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


def _load_property2datatype(path):
    # from the property_labels table
    pid2type = {}
    with open(path, 'r') as f:
        header = f.readline()
        assert header.strip().split('\t') == ["datatype", "label", "pid"]
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            datatype, label, pid = line.strip().split('\t')
            pid2type[pid] = datatype
    return pid2type


def _load_qid2aliases(dir_entity_aliases):
    qid2alias = {}
    for batch_file in os.listdir(dir_entity_aliases):
        with open(dir_entity_aliases / batch_file) as f:
            for line in f:
                qid, aliases = line.strip().split('\t')
                aliases = [a for a in aliases.split('|sep|') if a not in stop]
                qid2alias[qid] = aliases
    return qid2alias

# entity list : same entity type
def get_entity_by_etype(target_etype):
    entity_list = []
    entity2type_data = pickle.load(open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_analysis/entity_rels/entity2type_textdata.pkl", 'rb'))
    for e, types in entity2type_data.items():
        if target_etype in types:
            entity_list.append(e)
    return set(entity_list)


def match_sentence_eav(sentence: str, tokenized_sentence: list, alias: tuple):
    """
    Match a single sentence with a single statement. Condition: (e, a, v)
    sentence: str
    alias: a tuple of aliases of q,p,v
    """

    def is_sublist(a, b): # if a is a sublist of b
        if a is None or b is None:
            return False
        if len(a) > len(b):
            return False
        for i in range(0, len(b) - len(a) + 1):
            if b[i:i+len(a)] == a:
                return True
        return False

    q_alias_list, p_alias_list, v_alias_list = alias
    q_candidates = []
    p_candidates = []
    v_candidates = []

    p_candidates = [x for x in p_alias_list if x in sentence]
    if len(p_candidates) == 0:
        return None
    q_candidates = [x for x in q_alias_list if x in sentence]
    if len(q_candidates) == 0:
        return None
    v_candidates = [x for x in v_alias_list if x in sentence]
    if len(v_candidates) == 0:
        return None

    # check sublist
    tokenized_sentence = word_tokenize(sentence)
    q_text = []
    p_text = []
    v_text = []

    for _p in p_candidates:
        p = word_tokenize(_p)
        if is_sublist(p, tokenized_sentence):
            p_text.append(_p)
    if len(p_text) is None:
        return None

    for _q in q_candidates:
        q = word_tokenize(_q)
        if is_sublist(q, tokenized_sentence):
            q_text.append(_q)
    if len(q_text) is None:
        return None 

    for _v in v_candidates:
        v = word_tokenize(_v)
        if is_sublist(v, tokenized_sentence):
            v_text.append(_v)
    if len(v_text) is None:
        return None
    # avoid duplicate:
    for q in q_text:
        for p in p_text:
            for v in v_text:
                if (q != p) and (q != v) and (p != v) and (v not in q) and (q not in stop) and (v not in stop):
                    return q, p, v  # triple text
    return None


def _reformat_value(pid, value) -> list:

    def _transform_quantity(value):
        aliases = [value.lstrip('+')]
        return aliases

    def _transform_time(value):
        # only extract year now
        # format : +%Y-%m-%dT%H:%M:%SZ
        aliases = []
        try:
            date, time = value.rstrip('Z').lstrip('+').split('T')
            year, month, day = date.split('-')
            aliases.append(year)
        except:
            # print(value)
            aliases.append(value)
        return aliases

    datatype = pid2datatype[pid]
    if datatype == "quantity":
        vtext = _transform_quantity(value)
    elif datatype == "time":
        vtext = _transform_time(value)
    else:
        vtext = [value]
    return vtext


path_prop_datatype = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_labels/0.tsv")
pid2datatype = _load_property2datatype(path_prop_datatype)
print("pid2datatype loaded. size:", len(pid2datatype))





def get_occurrence(target_etype, dir_output):
    assert target_etype is not None
    print(target_etype)
    dir_output = Path(dir_output)
    print(dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)
    path_news = Path("/afs/crc.nd.edu/group/dmsquare/vol1/data/EngGigV5/corpus/data_enggigv5.txt")
    num_lines_news = 9870506
    assert path_news.exists()
    dir_linked = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1/combined")
    assert dir_linked.exists()
    entity_set_etype = get_entity_by_etype(target_etype)

    print("entity set etype size:", len(entity_set_etype))

    # load pid2alias
    path_prop_alias = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_aliases/0.tsv")
    pid2alias = _load_property2aliases(path_prop_alias)
    print("pid2alias loaded. size:", len(pid2alias))
    
    # load qid2alias
    print("loading qid2alias ...")
    dir_entity_aliases = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/aliases_sorted")
    qid2alias = _load_qid2aliases(dir_entity_aliases)
    print("qid2alias loaded. size:", len(qid2alias))
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

    print("keyword tree built!", time.time() - start)

    cnt_matched = 0
    # 3. match in news data. Statement linking criteria: (e, a). Statment pairing criteria: 
    with open (path_news) as f, open(dir_output / f"{target_etype}_matched.json", 'w') as fw:
        # for line in tqdm(f, total=num_lines_news):
        for linenum,line in enumerate(f):
            if linenum % 100000 == 0:
                print(linenum)
            line = line.split('\t')[-1]
            matched_labels = set() # record all matched entities
            for item in kwtree.iter(line):
                matched_labels.add(item[1][1])
            # print(matched_entities)
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
                    # print(len(linked_triples))
                    for pid, vid in linked_triples:
                        q_aliases = qid2alias[e] if vid in qid2alias else []
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
                for p in common_properties:
                    s1 = matched_qid2pid2triples[e1][p]
                    s2 = matched_qid2pid2triples[e2][p]
                    
                    surface_e1 = set([evidence[1][0] for evidence in s1])
                    surface_e2 = set([evidence[1][0] for evidence in s2])
                    
                    flag_keep = False # 
                    for _sf1 in surface_e1:
                        if flag_keep:
                            break
                        for _sf2 in surface_e2:
                            if flag_keep:
                                break
                            if _sf1 not in _sf2 and _sf2 not in _sf1:
                                flag_keep = True # v3: To avoid false positive, if both surfaces are not substrings of the other, then we can safely keep it.
                    # print (flag_keep, surface_e1, surface_e2)
                    if not flag_keep:
                        continue
                    

                    matched_statement = {
                        "entity_pair": (e1, e2),
                        "property": p,
                        "evidence_e1": s1,
                        "evidence_e2": s2
                    }
                    cnt_matched += 1

                    fw.write(json.dumps(matched_statement)+'\n')
                    # print(matched_statement)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_etype', type=str, default=None, help='etype')
    parser.add_argument('--dir_output', type=str, default="/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/statement_scoring/news_v1")

    return parser

if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    print(args)
    # get_occurrence(args.target_etype, args.dir_output)

    etypes = []
    with open("etype_list_1230.txt") as f:
        for line in f:
            etype, freq = line.strip().split()
            etypes.append(etype)
    print("types", len(etypes))
    for etype in etypes:
        get_occurrence(etype, args.dir_output)


                    
                    



            

    

