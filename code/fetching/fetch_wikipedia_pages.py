
# input: a table of wikipedia_links
# output: wikipedia texts

import csv
import bz2
from bs4 import BeautifulSoup
from wikiextractor.extract import Extractor
import pandas as pd
from pathlib import Path
import os
import multiprocessing
from multiprocessing import Queue, Process
import json
import time
import argparse

def search_index(index_filename, title_queue, index_queue):
    while True:
        search_term = title_queue.get(title_queue)
        if search_term is None:
            break
        # grab index for a single wiki title
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
            index_queue.put((search_term, start_byte, data_length))
    


def decompress_pages(dump_file, start_byte, data_length):
    decomp = bz2.BZ2Decompressor()
    with open(dump_file, 'rb') as f:
        f.seek(start_byte)
        readback = f.read(data_length)
        page_xml = decomp.decompress(readback).decode()
    soup = BeautifulSoup(page_xml, "lxml")
    pages = soup.find_all("page")
    return pages


def extract_page(index_filename, dump_filename, index_queue, write_queue):
    while True:
        index_info = index_queue.get()
        if index_info is None:
            break
        
        title, start_byte, data_length = index_info
        # print(title, start_byte, data_length)
        pages = decompress_pages(dump_filename, start_byte, data_length)
        page_titles= [p.find("title").text for p in pages]
        # print(page_titles)
        page_index = page_titles.index(title)
        raw_text = pages[page_index].find("text").text
        write_queue.put([title, raw_text])


def write_text(output_dir, write_queue):
    with open(output_dir / "wikipedia_pages.jsonl", 'w') as f:
        while True:
            obj = write_queue.get()
            if obj is None:
                break
            title, raw_text = obj
            extractor = Extractor("", "", "", "", "")
            clean_text = "\n".join(extractor.clean_text(raw_text))
            json_obj = {
                "title": title,
                "text": clean_text
            }
            
            f.write(json.dumps(json_obj) + '\n')
        
def read_data(input_file, title_queue):
    target_titles = list(pd.read_csv(input_file, sep='\t')["wiki_title"])
    print(len(target_titles))
    for title in target_titles:
        title_queue.put(title)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiki_filename', type=str, default='/afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikipedia_dump/enwiki-20220101-pages-articles-multistream.xml.bz2', help="folder name")
    parser.add_argument('--index_filename', type=str, default='/afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikipedia_dump/enwiki-20220101-pages-articles-multistream-index.txt', help="folder name")
    parser.add_argument('--output_dir', type=str, default=None, help="folder name")
    parser.add_argument('--title_filename', type=str, default=None, help="path")

    return parser


def main():
    args = get_arg_parser().parse_args()


    index_filename = args.index_filename
    wiki_filename = args.wiki_filename
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    # read target titles
    data_path = args.title_filename
    start = time.time()

    n_procs = 30
    maxsize = 10 * n_procs
    title_queue = Queue(maxsize)
    index_queue = Queue(maxsize)
    write_queue = Queue(maxsize)

    read_process = Process(
        target=read_data,
        args=(data_path, title_queue)
    )
    read_process.start()
    # reading index process
    index_processes = []
    for _ in range(max(1, n_procs-2)):
        index_process = Process(
            target=search_index,
            args=(index_filename, title_queue, index_queue)
        )
        index_process.daemon = True
        index_process.start()
        index_processes.append(index_process)
    print("Index processes initiated")
    # write process
    write_process = Process(
        target=write_text,
        args=(output_dir, write_queue)
    )
    write_process.start()
    print("Write process initiated")
    # work process
    # work_process = Process(
    #     target=extract_page,
    #     args=(index_filename, wiki_filename, index_queue, write_queue)
    # )  
    # work_process.daemon = True
    # work_process.start()
    work_processes = []
    for _ in range(max(1, n_procs-2)):
        work_process = Process(
            target=extract_page,
            args=(index_filename, wiki_filename, index_queue, write_queue)
        )
        work_process.daemon = True
        work_process.start()
        work_processes.append(work_process)
    print("Worker process initiated")
    for index_process in index_processes:
        index_process.join()
    for index_process in index_processes:
        title_queue.put(None)
    print("Finish indexing.")
    # cause all worker process to quit
    for work_process in work_processes:
        index_queue.put(None)
    # Now join the work process
    for work_process in work_processes:
        work_process.join()
    # index_queue.put(None)
    # work_process.join()
    write_queue.put(None)
    write_process.join()

    

if __name__ == "__main__":
    main()