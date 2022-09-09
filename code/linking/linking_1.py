'''
input: wikipedia page + its wikidata statements

process: we need to transfer statements(qid, pid) into text, and then match them to sentences

output: wikipedia page + matched pairs
'''
import pathlib from Path

def _load_property2datatype(path):
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

def _load_property2aliases(path):
    pid2aliases = {}
    with open(path, 'r') as f:
        header = f.readline()
        assert header.strip().split('\t') == ["alias", "pid"]
        for line in 
        

def _lookup_pid2alias(pid2alias, pid):
    
    return # wait for processing dump


def _lookup_qid2alias(dir_entity_label, qid):
    if len(qid) < 3:
        table_name = f"Q{int(qid[1]):02d}.tsv"
    else:
        table_name = f"Q{qid[1:3]}.tsv"
    record = None
    with open (dir_entity_label / table_name, 'r') as f:
        for line in f:
            if line.startswith(qid):
                record = line.strip().split('\t')
                break
    if record is None:
        # print("alias not found:", qid)
        return None
    aliases = record[1].split('|sep|')
    return aliases

def _transform_quantity(value):
    value = value.lstrip('+')
    return value

def _transform_time(value):
    # only extract year now
    # format : +%Y-%m-%dT%H:%M:%SZ
    try:
        date, time = value.rstrip('Z').lstrip('+').split('T')
        year, month, day = date.split('-')
        return year
    except:
        # print(value)
        return value


def _reformat_value(pid, value)

    ptext, datatype = pid2info[pid]
    if datatype == "quantity":
        vtext = _transform_quantity(value)
    elif datatype == "time":
        vtext = _transform_time(value)
    else:
        vtext = row["value"]
    return vtext
    
def get_statement_aliases(statement):
    # statement: tuple (e, a, v)
    





def match_page(sentence_list, statement_list):
    for sentence in sentence_list:
        for statement in statement_list:
            
    return