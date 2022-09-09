# qid to label/alias
# pid to label
# value to alias
import time


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

# statement to text for entity_values table
def transform_entity_value(dir_entity_label, pid2info, row):
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
    if pid not in pid2info:
        # print(f"Property {pid} not in index")
        return None
    ptext, datatype = pid2info[pid]
    if datatype == "quantity":
        vtext = _transform_quantity(row["value"])
    elif datatype == "time":
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

# statement to text for entity_rels table
def transform_entity_rels(dir_entity_label, pid2info, row):
    qid = row["qid"]
    # search cache first to avoid unneccesary I/O
    if qid in _cache_qtext_alias:
        qtext_list = _cache_qtext_alias[qid]
    else:
        qtext_list = qid2text(dir_entity_label, qid)
        _cache_qtext_alias[qid] = qtext_list
    pid = row["property_id"]
    if pid not in pid2info:
        # print(f"Property {pid} not in index")
        return None
    ptext, _ = pid2info[pid]
    vtext_list = qid2text(dir_entity_label, row["value"])
    if qtext_list is None or vtext_list is None or ptext is None:
        return None
    triples = []
    for qtext in qtext_list:
        for vtext in vtext_list:
            triples.append((qtext, ptext, vtext))
    return triples


# search wikipedia page 
import csv
import bz2
from bs4 import BeautifulSoup

from wikiextractor.extract import Extractor

def search_index(index_filename, title):
    byte_flag = False
    data_length = start_byte = 0
    index_file = open(index_filename, 'r')
    csv_reader = csv.reader(index_file, delimiter=':')
    for line in csv_reader:
        if not byte_flag and title == line[2]:
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


_cache_title2text = {}
def title2text(index_filename, dump_filename, extractor, title):
    if title in _cache_title2text:
        return _cache_title2text[title]
    index_info = search_index(index_filename, title)
    if index_info is None:
        return None
    title, start_byte, data_length = index_info
    raw_text = extract_text(dump_filename, start_byte, data_length)
    clean_text = "\n".join(extractor.clean_text(raw_text))
    _cache_title2text[title] = clean_text
    return clean_text


from nltk.tokenize import sent_tokenize
# linking
# multiple triple aliases stand for one distinct triple
def linking(page_text, triple_aliases): 
    sentences = sent_tokenize(page_text)
    match_sentences = []
    for sent in sentences:
        if len(sent) < 30:
            continue
        # print(sent)
        for t in triple_aliases: # t: (e, a, v)
            # print(t)
            # if t[0] in sent:
            if t[1] in sent and t[2] in sent:
                match_sentences.append(sent)
                break # go to next sent
    return match_sentences
    

import pandas as pd
from pathlib import Path

def _load_property_label(path):
    pid2info = {}
    with open(path, 'r') as f:
        header = f.readline()
        assert header.strip().split('\t') == ["datatype", "label", "pid"]
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            datatype, label, pid = line.strip().split('\t')
            pid2info[pid] = (label, datatype)
    return pid2info

import os

            
def qid2text(dir_entity_label, qid):
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


# df_entity_label = _load_table("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata/tables/aliases/university.tsv")
dir_entity_label = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/aliases_sorted") # for all
pid2info = _load_property_label("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_labels/0.tsv") # for all
dir_wikipedia_link = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/wikipedia_links_sorted") # for all

fn_entity_rels = "/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/entity_rels/0.tsv"
fn_entity_values = "/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/entity_values/0.tsv"

# below are for wikipedia
extractor = Extractor("", "", "", "", "")
index_filename = '/afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikipedia_dump/enwiki-20220101-pages-articles-multistream-index.txt'
dump_filename = '/afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikipedia_dump/enwiki-20220101-pages-articles-multistream.xml.bz2'



start = time.time()
with open(fn_entity_rels, 'r', encoding='utf-8') as f:
    header = f.readline()
    assert header.strip().split('\t') == ["property_id", "qid", "value"]
    cnt = 0
    while cnt < 100:
        # print(cnt)
        x = f.readline().strip().split('\t')
        cnt += 1
        row = {"property_id": x[0], "qid": x[1], "value": x[2]}
        # print(row)
        # transform id to text
        t = time.time()
        text_triples = transform_entity_rels(dir_entity_label, pid2info, row)
        print("find triple:", time.time() - t)
        # link to passage
        
        if text_triples is None:
            print("text triple not found")
            continue
        t = time.time()
        qid = row["qid"]
        if qid in _cache_wikititle:
            title = _cache_wikititle[qid]
        else:
            title = qid2title(dir_wikipedia_link, qid)
            _cache_wikititle[qid] = title
        if title is None:
            print(qid, "title not found")
            continue
            # print(row)
            # print(title)
            # print(text_triples)
        wikipedia_text = title2text(index_filename, dump_filename, extractor, title)
        print("find page", time.time()-t)

        if wikipedia_text is None:
            continue
        # print(f"title: {title}; # triples: {len(text_triples)}; text len: {len(wikipedia_text)}")
        # matched_sentences = linking(wikipedia_text, text_triples)
        # if len(matched_sentences) > 0:
        #     print("matched:", text_triples[0],  matched_sentences)
         
        
        if cnt % 100 == 0:
            print(cnt)
    lapsed = time.time() - start
    print("Process finished! Time:", lapsed)
        