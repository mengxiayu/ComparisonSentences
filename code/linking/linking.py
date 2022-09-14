'''
input: wikipedia page + its wikidata statements

process: we need to transfer statements(qid, pid) into text, and then match them to sentences

output: wikipedia page + matched pairs
'''
from pathlib import Path
import json
from nltk.tokenize import sent_tokenize, word_tokenize
import time
import os
import argparse

path_prop_datatype = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_labels/0.tsv")
path_prop_alias = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_aliases/0.tsv")
dir_entity_rels = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/entity_rels_sorted")
dir_entity_values = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/entity_values_sorted")
dir_entity_aliases = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/aliases_sorted")

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str, default=None, help='path to input directory')
    parser.add_argument('--dir_output', type=str, default=None, help='path to output directory')
    parser.add_argument('--input_filename', type=str, default=None, help='file to be process. e.g., "wiki00"')
    parser.add_argument('--mode', type=str, default='eav', help="'eav' or 'av'")
    return parser

args = get_arg_parser().parse_args()
print(args)

_cache_qid2lias = {}

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

pid2datatype = _load_property2datatype(path_prop_datatype)
print("pid2datatype loaded. size:", len(pid2datatype))

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

pid2alias = _load_property2aliases(path_prop_alias)
print("pid2alias loaded. size:", len(pid2alias))

def _load_qid2aliases(dir_entity_aliases):
    qid2alias = {}
    for batch_file in os.listdir(dir_entity_aliases):
        with open(dir_entity_aliases / batch_file) as f:
            for line in f:
                qid, aliases = line.strip().split('\t')
                qid2alias[qid] = aliases
    return qid2alias

qid2alias = _load_qid2aliases(dir_entity_aliases)
print("qid2alias loaded. size:", len(qid2alias))

def _lookup_pid2alias(pid): 
    # return a list of aliases
    return pid2alias[pid] if pid in pid2alias else None

# def _lookup_qid2alias(qid):
#     if len(qid) < 3:
#         table_name = f"Q{int(qid[1]):02d}.tsv"
#     else:
#         table_name = f"Q{qid[1:3]}.tsv"
#     record = None
#     with open (dir_entity_aliases / table_name, 'r') as f:
        # for line in f:
        #     if not line.startswith(qid):
        #         continue
        #     record = line.strip().split('\t')
        #     if record[0] == qid:
        #         return record[1].split('|sep|')
#     return None


def _lookup_qid2alias(qid):
    if qid in qid2alias:
        return qid2alias[qid].split('|sep|')
    return None

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

def _reformat_value(pid, value) -> list:

    datatype = pid2datatype[pid]
    if datatype == "quantity":
        vtext = _transform_quantity(value)
    elif datatype == "time":
        vtext = _transform_time(value)
    else:
        vtext = [value]
    return vtext
    

def get_alias_statement(statement, v_type):
    # statement: tuple (e, a, v)
    # deal with both entity_rels and entity_values
    global _cache_qid2lias
    qid, pid, value = statement
    if qid in _cache_qid2lias:
        q_alias = _cache_qid2lias[qid]
    else:
        q_alias = _lookup_qid2alias(qid)
        _cache_qid2lias[qid] = q_alias
    p_alias = _lookup_pid2alias(pid)
    if v_type == "entity":
        v_alias = _lookup_qid2alias(value)
    elif v_type == "value":
        v_alias = _reformat_value(pid, value)
    else:
        print("error: undefined v_type")
        v_alias = None
    if q_alias is not None and p_alias is not None and v_alias is not None:
        return q_alias, p_alias, v_alias
    else:
        return None


def match_sentence_av(sentence: str, alias: list):
    """
    Match a single sentence with a single statement. Condition: (?, a, v)
    sentence: str
    alias: a tuple of aliases of e,a,v
    """
    q_alias_list, p_alias_list, v_alias_list = alias
    q_candidates = []
    p_candidates = []
    v_candidates = []

    p_candidates = [x for x in p_alias_list if x in sentence]
    if len(p_candidates) == 0:
        return None

    v_candidates = [x for x in v_alias_list if x in sentence]
    if len(v_candidates) == 0:
        return None

    # check sublist to make sure the word boundries are correct
    tokenized_sentence = word_tokenize(sentence)
    q_text = q_alias_list[0]
    p_text = None
    v_text = None

    for _p in p_candidates:
        p = word_tokenize(_p)
        if is_sublist(p, tokenized_sentence):
            p_text = _p
            break
    if p_text is None:
        return None
    for _v in v_candidates:
        v = word_tokenize(_v)
        if is_sublist(v, tokenized_sentence):
            v_text = _v
            break
    if v_text is None:
        return None
    return q_text, p_text, v_text

# match (e, a, v)
def match_sentence_eav(sentence: str, alias: tuple):
    """
    Match a single sentence with a single statement. Condition: (e, a, v)
    sentence: str
    alias: a tuple of aliases of q,p,v
    """
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
    q_text = None
    p_text = None
    v_text = None

    for _p in p_candidates:
        p = word_tokenize(_p)
        if is_sublist(p, tokenized_sentence):
            p_text = _p
            break
    if p_text is None:
        return None

    for _q in q_candidates:
        q = word_tokenize(_q)
        if is_sublist(q, tokenized_sentence):
            q_text = _q
            break
    if q_text is None:
        return None 

    for _v in v_candidates:
        v = word_tokenize(_v)
        if is_sublist(v, tokenized_sentence):
            v_text = _v
            break
    if v_text is None:
        return None
    return q_text, p_text, v_text


def match_page(sentence_list: list, statement_list, v_type):

    results = {}
    cnt = 0
    for i,statement in enumerate(statement_list):
        alias = get_alias_statement(statement, v_type)
        if alias is None:
            continue
        for j,sentence in enumerate(sentence_list):
            if len(sentence) < 15:
                continue
            if args.mode == 'eav':
                matched = match_sentence_eav(sentence, alias)
            elif args.mode == 'av':
                matched = match_sentence_av(sentence, alias)
            else:
                print(f"error: mode {args.mode}")
            if matched is None:
                continue
            statement_surface = matched
            # print(statement_surface)
            # print(alias)

            if j not in results:
                results[j] = [] # the 'j' here should matched the index of input sentences_list
            results[j].append([statement, str(statement_surface)])
    return results


def _reorganize_statement(statement_list):
    statements = []
    for s in set(statement_list): # Sep 13 update: remove duplicates
        info = s.split('\t')
        if len(info) < 3:
            continue
        p, q, v = info
        statements.append((q, p, v))
    return statements

def is_sublist(a, b): # if a is a sublist of b
    if a is None or b is None:
        return False
    if len(a) > len(b):
        return False
    for i in range(0, len(b) - len(a) + 1):
        if b[i:i+len(a)] == a:
            return True
    return False

def main():
    start = time.time()
    dir_data = Path(args.dir_data)
    dir_output = Path(args.dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)

    for num in args.input_filename.split(','):
        input_filename = f"wiki_{num}"
        with open(dir_data / input_filename) as f, open(dir_output / input_filename, 'w') as fw:
            cnt_all = 0 # all pages
            cnt_linked = 0 # linked pages
            for line in f:
                cnt_all += 1
                if cnt_all % 2000 == 0:
                    print(f"{cnt_all} read, {cnt_linked} linked, {(time.time() - start)/60:.2f} min")
                obj = json.loads(line)
                # print(obj["title"])
                text = obj["text"]
                sentences = sent_tokenize(text)
                
                # entity_rels
                entity_rels = _reorganize_statement(obj["entity_rels"]) 
                _t = time.time()
                matched_entity_rels = match_page(sentences, entity_rels, "entity")
                # print("rels:", time.time() - _t)
                # entity_values
                entity_values = _reorganize_statement(obj["entity_values"])
                _t = time.time()
                matched_entity_values = match_page(sentences, entity_values, "value")
                # print("values:", time.time() - _t)
                if len(matched_entity_rels)==0 and len(matched_entity_values) == 0:
                    continue
                cnt_linked += 1
                new_obj = {
                    "id": obj["id"],
                    "title": obj["title"],
                    "qid": obj["qid"],
                    "linked_entity_rels": [],
                    "linked_entity_values": []
                }
                
                if len(matched_entity_rels) > 0:
                    for j, info in sorted(matched_entity_rels.items()):
                        new_obj["linked_entity_rels"].append([j, sentences[j], info])
                if len(matched_entity_values) > 0:
                    for j, info in sorted(matched_entity_values.items()):
                        new_obj["linked_entity_values"].append([j, sentences[j], info])   
                fw.write(json.dumps(new_obj)+'\n')
                  
        print(f"Finished!, time:{time.time() - start}, cnt_total: {cnt_all}, cnt_linked: {cnt_linked}")


if __name__ == "__main__":
    main()