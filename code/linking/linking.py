


# qid to label/alias
# pid to label
# value to alias

def qid2title(dir_wikipedia_link, qid):

    table_name = f"Q{qid[1]}.tsv"
    record = None
    with open(dir_wikipedia_link / table_name, 'r') as f:
        for line in f:
            if line.startswith(qid):
                record = line.strip().split('\t')
                break
    if record is None:
        return None
    return record[1]

def qid2text(dir_entity_label, qid):

    if len(qid) < 3:
        table_name = f"Q{qid[1]:02d}.tsv"
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

def pid2text(df, pid):
    # can directly load property_label table
    record = df.loc[df["pid"] == pid]
    assert len(record) == 1
    text = record.iloc[0]["label"]
    return text 

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

_cache_qtext_alias = {}
_cache_wikititle = {}

def transform_entity_value(dir_entity_label, property_df, row):
    # qid can have multiple labels(aliases)
    # property has one label
    # value needs to be transformed to one label
    qid = row["qid"]
    # search cache first to avoid unneccesary I/O
    if qid in _cache_qtext_alias:
        qtext_list = _cache_qtext_alias[qid]
    else:
        qtext_list = qid2text(dir_entity_label, qid)
        _cache_qtext_alias[qid] = qtext_list
    pid = row["property_id"]
    info = property_df.loc[property_df["pid"] == pid]
    if len(info) == 0:
        # print(f"Property {pid} not in index")
        return None
    info = info.iloc[0]
    ptext = info["label"]
    if info["datatype"] == "quantity":
        vtext = _transform_quantity(row["value"])
    elif info["datatype"] == "time":
        vtext = _transform_time(row["value"])
    else:
        vtext = row["value"]
    # enumertae qid's aliases
    if qtext_list is None or vtext_list is None or ptext is None:
        return None
    triples = []
    for qtext in qtext_list:
        triples.append((qtext, ptext, vtext))
    return triples


def transform_entity_rels(dir_entity_label, property_df, row):
    qid = row["qid"]
    # search cache first to avoid unneccesary I/O
    if qid in _cache_qtext_alias:
        qtext_list = _cache_qtext_alias[qid]
    else:
        qtext_list = qid2text(dir_entity_label, qid)
    pid = row["property_id"]
    info = property_df.loc[property_df["pid"] == pid]
    if len(info) == 0:
        # print(f"Property {pid} not in index")
        return None
    info = info.iloc[0]  
    ptext = info["label"]
    vtext_list = qid2text(dir_entity_label, row["value"])
    if qtext_list is None or vtext_list is None or ptext is None:
        return None
    triples = []
    for qtext in qtext_list:
        for vtext in vtext_list:
            triples.append((qtext, ptext, vtext))
    return triples


# wikipedia 
import csv
import bz2
from wikiextractor.extract import Extractor

def search_index(index_filename, title):
    byte_flag = False
    data_length = start_byte = 0
    index_file = open(index_filename, 'r')
    csv_reader = csv.reader(index_file, delimiter=':')
    for line in csv_reader:
        if not byte_flag and search_term == line[2]:
            start_byte = int(line[0])
            byte_flag = True
        elif byte_flag and int(line[0]) != start_byte:
            data_length = int(line[0]) - start_byte
            break
    index_file.close()
    if data_length > 0:
        return (title, start_byte, data_length)
    return None

def extract_text(dump_file, start_byte, data_length):
    decomp = bz2.BZ2Decompressor()
    with open(dump_file, 'rb') as f:
        f.seek(start_byte)
        readback = f.read(data_length)
        page_xml = decomp.decompress(readback).decode()
    soup = BeautifulSoup(page_xml, "lxml")
    pages = soup.find_all("page")
    page_titles= [p.find("title").text for p in pages]
    page_index = page_titles.index(title)
    raw_text = pages[page_index].find("text").text

    return raw_text


def title2text(index_filename, dump_filename, extractor):
    index_info = search_index(index_filename, title)
    if index_info is not None:
        title, start_byte, data_length = index_info
    raw_text = extract_text(dump_filename, start_byte, data_length)
    clean_text = "\n".join(extractor.clean_text(raw_text))
    return clean_text

import pandas as pd
from pathlib import Path
def _load_table(path):
    df = pd.read_csv(path, sep='\t', header=0)
    return df

# df_entity_label = _load_table("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata/tables/aliases/university.tsv")
dir_entity_label = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/aliases_sorted")
df_property_label = _load_table("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_labels/0.tsv")
dir_wikipedia_link = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/wikipedia_links_sorted")

fn_entity_vals = "/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata/tables/entity_values/university.tsv"
fn_entity_rels = "/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata/tables/entity_rels/university.tsv"

with open(fn_entity_rels, 'r', encoding='utf-8') as f:
    header = f.readline()
    assert header.strip().split('\t') == ["claim_id", "property_id", "qid", "value"]
    cnt = 0
    while cnt < 100:
        x = f.readline().strip().split('\t')
        row = {"property_id": x[2], "qid": x[3], "value": x[4]}
        # print(row)
        # transform id to text
        
        text_triples = transform_entity_rels(dir_entity_label, df_property_label, row)
        # link to passage
        if text_triples is not None:
            qid = row["qid"]
            if qid in _cache_wikititle:
                title = _cache_wikititle[qid]
            else:
                title = qid2title(dir_wikipedia_link, qid)
            if title is not None:
                print(row)
                print(title)
                print(text_triples)
                
        cnt += 1