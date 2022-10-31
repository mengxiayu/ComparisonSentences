from pathlib import Path
import json
import os
from collections import Counter
import pickle
import numpy as np
import scipy.stats as stats
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

path_prop_alias = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_aliases/0.tsv")

pid2alias = _load_property2aliases(path_prop_alias)

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

def write_data_info(dir_data, output_dir):
    split_list = ["AA", "AB"]
    fw = open(output_dir / "data_info", 'w')
    fw.write("qid\ttitle\tnum_sentences\tnum_entity_rels\tnum_entity_values\n")
    for split in split_list:
        for batch_file in os.listdir(dir_data / split):
            with open(dir_data / split / batch_file) as f:
                for line in f:
                    obj = json.loads(line)
                    # start analysis
                    num_sentences = 0
                    num_entity_rels = 0
                    num_entity_values = 0
                    for x in sent_tokenize(obj["text"]):
                        if len(x) >= 15:
                            num_sentences += 1
                    num_entity_rels = len(set(obj["entity_rels"]))
                    num_entity_values = len(set(obj["entity_values"]))
                    # write analysis
                    fw.write(f"{obj['qid']}\t{obj['title']}\t{num_sentences}\t{num_entity_rels}\t{num_entity_values}\n")
    fw.close()


    cnt_P17 = Counter() # country
    cnt_P21 = Counter() # gender
    cnt_P31 = Counter() # instance of
    cnt_P106 = Counter() # occupation
    cnt_P102 = Counter() # member of political party
    for split in split_list:
        for batch_file in os.listdir(dir_data / split):
            with open(dir_data / split / batch_file) as f:
                for line in f:
                    obj = json.loads(line)
                    # start analysis
                    for s in obj["entity_rels"]:
                        _s = s.split('\t')
                        if len(_s) != 3:
                            continue
                        p, q, v = _s
                        if p == "P17":
                            cnt_P17[v] += 1
                        elif p == "P21":
                            cnt_P21[v] += 1
                        elif p == "P31":
                            cnt_P31[v] += 1
                        elif p == "P106":
                            cnt_P106[v] += 1
                        elif p == "P102":
                            cnt_P102[v] += 1   
    with open(output_dir / "cnt_P17", 'w') as f:
        for k,v in cnt_P17.most_common():
            f.write(f"{k} {v}\n")

    with open(output_dir / "cnt_P21", 'w') as f:
        for k,v in cnt_P21.most_common():
            f.write(f"{k} {v}\n")

    with open(output_dir / "cnt_P31", 'w') as f:
        for k,v in cnt_P31.most_common():
            f.write(f"{k} {v}\n")

    with open(output_dir / "cnt_P106", 'w') as f:
        for k,v in cnt_P106.most_common():
            f.write(f"{k} {v}\n")

    with open(output_dir / "cnt_P102", 'w') as f:
        for k,v in cnt_P102.most_common():
            f.write(f"{k} {v}\n")                   

def analyze_linked_special(dir_data, dir_linked, output_dir, target_property):
    # to see whether the linked statement distribution (value distribution of a target property) matches the original data distribution
    # needs avoid duplicate statements from different linking criteria
    origin_dist = {} # value2count
    cnt = 0
    with open (dir_data / "statistics" / f"cnt_{target_property}") as f:
        for line in f:
            if cnt == 15:
                break
            arr = line.strip().split(' ')
            origin_dist[arr[0]] = int(arr[1])
            cnt += 1

    value2statement = {} # value to statement set
    for v in origin_dist:
        value2statement[v] = set() # only keep track of those are top ones in original data
    criteria = ["eav", "av", "ev"]
    split_list = ["AA", "AB"]
    value2count_list = []
    for criterion in criteria:
        for split in split_list:
            path = dir_linked / criterion / split
            for batch_file in path.glob("wiki*"):
                with open(batch_file) as f:
                    for line in f:
                        obj = json.loads(line)
                        for x in obj["linked_entity_rels"]:
                            for s in x[2]:
                                q, p, v = s[0]
                                if p != target_property:
                                    continue
                                if v in value2statement:
                                    value2statement[v].add(' '.join(s[0]))
        pickle.dump(value2statement, open(output_dir / f"value2statement_{target_property}_{criterion}.pkl", 'wb'))
        _value2count = {k:len(v) for k,v in value2statement.items()}
        value2count_list.append(_value2count)
    with open(output_dir / f"value2count_accumulate_{target_property}.tsv", 'w') as f:
        f.write('value\tlabel\tdata\t' + '\t'.join(criteria) + '\n')
        for value, freq in sorted(origin_dist.items(), key=lambda x:x[1], reverse=True):
            alias = _lookup_qid2alias(value)
            value_label = alias[0] if alias else ""
            f.write(f"{value}\t{value_label}\t{freq}\t")
            add_av = value2count_list[1][value] - value2count_list[0][value]
            add_ev = value2count_list[2][value] - value2count_list[1][value]
            f.write(f"{value2count_list[0][value]}\t{add_av}\t{add_ev}\n")
            # f.write('\t'.join([str(v2c[value]) for v2c in value2count_list]) + '\n')           


def write_linked_info(dir_linked, output_dir):
    # write linked info for later analysis
    split_list = ["AA", "AB"]
    fw = open(output_dir / "linked_info", 'w')
    for split in split_list:
        for batch_file in os.listdir(dir_linked / split):
            with open(dir_linked / split / batch_file) as f:
                for line in f:
                    obj = json.loads(line)
                    num_sentences = 0
                    num_entity_rels = 0
                    num_entity_values = 0
                    idx_sentence = set() # record sentence id to avoid counting dupliacation

                    for x in obj["linked_entity_rels"]: # idx, sent, triples
                        if x[0] not in idx_sentence:
                            num_sentences += 1
                            idx_sentence.add(x[0])
                        num_entity_rels += len(x[2])
                                  
                    for x in obj["linked_entity_values"]:
                        if x[0] not in idx_sentence:
                            num_sentences += 1
                            idx_sentence.add(x[0])
                        num_entity_values += len(x[2])
        
                    fw.write(f"{obj['qid']}\t{obj['title']}\t{num_sentences}\t{num_entity_rels}\t{num_entity_values}\n")
    fw.close()

    cnt_P17 = Counter() # country
    cnt_P21 = Counter() # gender
    cnt_P31 = Counter() # instance of
    cnt_P106 = Counter() # occupation
    cnt_P102 = Counter() # member of political party
    
    for split in split_list:
        for batch_file in os.listdir(dir_linked / split):
            with open(dir_linked / split / batch_file) as f:
                for line in f:
                    obj = json.loads(line)
                    for x in obj["linked_entity_rels"]:
                        for s in x[2]:
                            q, p, v = s[0]
                            if p == "P17":
                                cnt_P17[v] += 1
                            elif p == "P21":
                                cnt_P21[v] += 1
                            elif p == "P31":
                                cnt_P31[v] += 1
                            elif p == "P106":
                                cnt_P106[v] += 1
                            elif p == "P102":
                                cnt_P102[v] += 1                        
    with open(output_dir / "linked_cnt_P17", 'w') as f:
        for k,v in cnt_P17.most_common():
            f.write(f"{k} {v}\n")

    with open(output_dir / "linked_cnt_P21", 'w') as f:
        for k,v in cnt_P21.most_common():
            f.write(f"{k} {v}\n")

    with open(output_dir / "linked_cnt_P31", 'w') as f:
        for k,v in cnt_P31.most_common():
            f.write(f"{k} {v}\n")

    with open(output_dir / "linked_cnt_P106", 'w') as f:
        for k,v in cnt_P106.most_common():
            f.write(f"{k} {v}\n")       
    with open(output_dir / "linked_cnt_P102", 'w') as f:
        for k,v in cnt_P102.most_common():
            f.write(f"{k} {v}\n")  



def _load_cnt(fn):
    cnt = Counter()
    with open(fn) as f:
        for line in f:
            a = line.strip().split(' ')
            if len(a) != 2:
                continue
            cnt[a[0]] = int(a[1])
    return cnt

def _property_stats(f_name, fw, property_name):

    cnt = _load_cnt(f_name)
    pid = str(f_name).split('_')[-1]
    fw.write(f"-------------Popular values of property\t{pid}\t{property_name}-------------\n\n")
    for k,v in cnt.most_common(15):
        fw.write(f"{k}\t'{_lookup_qid2alias(k)[0]}'\t{v}\n")
    fw.write('\n')
        # print(f"{k}\t'{_lookup_qid2alias(k)[0]}'\t{v}")

def _page_stats(f_name, fw):
    # rank by # sentences
    data = []
    with open(f_name) as f:
        header = f.readline()
        num_sent_list = []
        num_rels_list = []
        num_vals_list = []
        cnt_sentences = Counter()
        cnt_statements = Counter()

        for line in f:
            qid, title, num_sent, num_rels, num_vals = line.strip().split('\t')
            num_sent = int(num_sent)
            num_rels = int(num_rels)
            num_vals = int(num_vals)
            arr = [qid, title, num_sent, num_rels, num_vals]
            data.append(arr)
            num_sent_list.append(num_sent)
            num_rels_list.append(num_rels)
            num_vals_list.append(num_vals)
            
            cnt_sentences[num_sent] += 1
            cnt_statements[num_rels + num_vals] += 1

    # overall distribution
    num_entity = sum(list(cnt_sentences.values()))
    fw.write(f"Number of linked entity: {num_entity}\n")
    fw.write('\n')
    # total sentence number
    num_sentence = sum([k * v for k,v in cnt_sentences.items()])
    fw.write(f"Number of linked sentence: {num_sentence}\n")
    fw.write("#Sentence Frequency (Top 5)\n")
    for k,v in cnt_sentences.most_common(5):
        fw.write(f"{k} {v}\n")
    fw.write('\n')
    # total statement number
    num_statement = sum([k * v for k,v in cnt_statements.items()])
    fw.write(f"Number of linked statement: {num_statement}\n")
    fw.write("#Statement Frequency (Top 5)\n")
    for k,v in cnt_statements.most_common(5):
        fw.write(f"{k} {v}\n")

    data = np.array(data)
    topk_sent = np.argpartition(-np.array(num_sent_list), range(15))[:15]
    topk_rels = np.argpartition(-np.array(num_rels_list), range(15))[:15]
    topk_vals = np.argpartition(-np.array(num_vals_list), range(15))[:15]

    # distriution
    fw.write(f"\n Distribution of number of sentences\n, {stats.describe(num_sent_list)}")
    fw.write(f"\n Distribution of number of relational statements\n, {stats.describe(num_rels_list)}")
    fw.write(f"\n Distribution of number of value statements\n, {stats.describe(num_vals_list)}")

    # extreme cases
    fw.write("----------\nRanked by num_sentences:\n")
    fw.write("qid\ttitle\tnum_sentences\tnum_entity_rels\tnum_entity_values\n")
    for item in data[topk_sent]:
        fw.write("\t".join(item)+'\n')
    fw.write("----------\n\nRanked by num_entity_rels:\n")
    fw.write("qid\ttitle\tnum_sentences\tnum_entity_rels\tnum_entity_values\n")
    for item in data[topk_rels]:
        fw.write("\t".join(item)+'\n')
    fw.write("----------\n\nRanked by num_entity_values:\n")
    fw.write("qid\ttitle\tnum_sentences\tnum_entity_rels\tnum_entity_values\n")
    for item in data[topk_vals]:
        fw.write("\t".join(item)+'\n')                    

def process_data_info(output_dir):

    fw = open(output_dir / "data_analysis", 'w')
    _page_stats(output_dir / "data_info", fw)
    _property_stats(output_dir / "cnt_P17", fw, "'country'")
    _property_stats(output_dir / "cnt_P21", fw, "'gender'")
    _property_stats(output_dir / "cnt_P31", fw, "'instance of'")
    _property_stats(output_dir / "cnt_P106", fw, "'occupation'")
    _property_stats(output_dir / "cnt_P102", fw, "'political party'")
    fw.close()  

def process_linked_info(output_dir):

    fw = open(output_dir / "linked_analysis", 'w')
    _page_stats(output_dir / "linked_info", fw)
    _property_stats(output_dir / "linked_cnt_P17", fw, "'country'")
    _property_stats(output_dir / "linked_cnt_P21", fw, "'gender'")
    _property_stats(output_dir / "linked_cnt_P31", fw, "'instance of'")
    _property_stats(output_dir / "linked_cnt_P106", fw, "'occupation'")
    _property_stats(output_dir / "linked_cnt_P102", fw, "'political party'")
    fw.close()

def print_most_popular_entity_rels(dir_linked, f_output):
    split_list = ["AA", "AB"]
    statement2freq = Counter()
    statement2surface = {}
    for split in split_list:
        for batch_file in (dir_linked / split).glob("wiki*"):
            with open(batch_file) as f:
                for line in f:
                    obj = json.loads(line)
                    for x in obj["linked_entity_rels"]:
                        for s in x[2]:
                            q, p, v = s[0]
                            statement = "\t".join(s[0])
                            statement2freq[statement] += 1
                            statement2surface[statement] = s[1]
    with open(f_output, 'w') as f:
        
        f.write(f"freq\tstatement\tsurface_form\n")
        cnt = 0
        for statement, freq in statement2freq.most_common():
            if cnt == 1000:
                break
            surface = statement2surface[statement]
            q_name, p_name, v_name = eval(surface)
            if set(q_name.split()) & set(v_name.split()):
                continue
            if q_name.lower() in v_name.lower() or v_name.lower() in q_name.lower():
                continue
            f.write(f"{freq}\t({q}, {p}, {v})\t{surface}\n")
            cnt += 1

def print_most_popular_entity_values(dir_linked, f_output):
    split_list = ["AA", "AB"]
    statement2freq = Counter()
    statement2surface = {}
    for split in split_list:
        for batch_file in (dir_linked / split).glob("wiki*"):
            with open(batch_file) as f:
                for line in f:
                    obj = json.loads(line)
                    for x in obj["linked_entity_values"]:
                        for s in x[2]:
                            q, p, v = s[0]
                            statement = "\t".join(s[0])
                            statement2freq[statement] += 1
                            statement2surface[statement] = s[1]
    with open(f_output, 'w') as f:
        f.write(f"freq\tstatement\tsurface_form\n")
        cnt = 0
        for statement, freq in statement2freq.most_common():
            if cnt == 1000:
                break
            surface = statement2surface[statement]
            q_name, p_name, v_name = eval(surface)
            if set(q_name.split()) & set(v_name.split()):
                continue
            if q_name.lower() in v_name.lower() or v_name.lower() in q_name.lower():
                continue
            f.write(f"{freq}\t({q}, {p}, {v})\t{surface}\n")
            cnt += 1


def main():
    dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data/")
    dir_linked = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1/")


    '''value distribution of some properties'''
    # for prop in ["P17", "P21", "P31", "P102", "P106"]:
        # output_dir = dir_linked / "statistics" / "_output"
    #     analyze_linked_special(dir_data, dir_linked, output_dir, prop)

    '''write distribution infomation'''
    # for criterion in ["eav", "av", "ev"]:
    #     output_dir = dir_linked / "statistics" / criterion
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     write_linked_info(dir_linked / criterion, output_dir)
    
    '''statistics of distribution'''
    # for criterion in ["eav", "av", "ev"]:
    #     output_dir = dir_linked / "statistics" / criterion
    #     process_info(output_dir)

    criterion = "combined"
    output_dir = dir_linked / "statistics" / criterion
    output_dir.mkdir(parents=True, exist_ok=True)
    write_linked_info(dir_linked / criterion, output_dir)
    process_linked_info(output_dir)
    
    # output_dir = dir_data / "statistics" 
    # process_data_info(output_dir)

    # criterion = "av"
#     output_f = dir_linked / "statistics" / criterion / "freq_entity_rels_nonoverlap.txt"
#     print_most_popular_entity_rels(dir_linked / criterion, output_f)

#     output_f = dir_linked / "statistics" / criterion / "freq_entity_values_nonoeverlap.txt"
#     print_most_popular_entity_values(dir_linked / criterion, output_f)
if __name__ == "__main__":
    main()