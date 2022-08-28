import multiprocessing
from multiprocessing import Queue, Process
import time
import os
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
import json

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikidata', help='path to input directory')
    parser.add_argument('--table_name', type=str, default='entity_values', help='table name')
    parser.add_argument('--rel', type=str, default=None, help='relationship')
    parser.add_argument('--value', type=str, default=None, help='target value')
    parser.add_argument('--num_procs', type=int, default=10, help='Number of processes')
    parser.add_argument('--output', type=str, default="/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata", help='path to output directory')
    parser.add_argument('--index', type=str, default=None, help='entity2file index')
    parser.add_argument('--entities_file', type=str, default=None, help="entity list file")
    parser.add_argument('--output_name', type=str, default="extracted", help="output file name")
    return parser

# 1. define functions for multiprocessor
def _read_dir(
        work_queue: Queue, 
        args
    ):
    path = Path(args.data) / args.table_name
    dir_list = os.listdir(path)
    print(f"searching in {len(dir_list)} files...")
    for batch_file in dir_list[:5]: # FIXME
        # work_queue.put(path+batch_file)
        work_queue.put(path / batch_file)

def _read_filtered_dir(work_queue: Queue, args):
    assert args.index_file is not None
    assert args.entities_file is not None
    
    path = Path(args.data) / table_name
    with open (args.entities_file, 'r') as f:
        entity_list = [line.strip() for line in f]
    entity2file = json.load(open(args.index_file, 'r'))
    file_list = []
    for e in entity_list:
        file_list.extend(entity2file[e])
    for batch_file in file_list:
        work_queue.put(path / batch_file)

def _write_data(output_queue: Queue, args):
    df_list = []
    while True:
        df_obj = output_queue.get()
        if df_obj is None:
            break
        df_list.append(df_obj)
    result = pd.concat(df_list)
    print("result length:", len(result))
    table_dir = Path(args.output) / args.table_name
    table_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(table_dir / f"{args.output_name}.tsv", sep='\t')

def _fetch_by_qid(work_queue: Queue, args):
    '''
    work_quque: queue of input file paths

    '''
    with open (args.entities_file, 'r') as f:
        entity_list = [line.strip() for line in f]
    while True:
        file_path = work_queue.get()
        if file_path is None:
            break
        print(file_path[-10:])
        df = pd.read_csv(file_path, sep='\t', header=0, on_bad_lines='warn')
        record = df.loc[df["qid"].isin(entity_list)] # query in the table
        if len(record) > 0:
            print(record)

def _fetch_by_property(work_queue: Queue, output_queue: Queue, args):
    assert args.rel is not None
    assert args.value is not None
    while True:
        file_path = work_queue.get()
        if file_path is None:
            break
        df = pd.read_csv(file_path, sep='\t', header=0, on_bad_lines='warn')
        record = df.loc[(df["property_id"] == args.rel) & (df["value"] == args.value)]
        if len(record) == 0:
            continue
        output_queue.put(record)


def work_flow(args, read_func, work_func, write_func):
    '''
    read from data
    process with work_queue
    write with output_queue
    '''

    start_time = time.time()
    n_process = args.num_procs
    max_size = 100 * n_process
    output_queue = Queue(maxsize=max_size) 
    work_queue = Queue(maxsize=max_size)
    data_path = Path(args.data)
    output_path = Path(args.output)

    # put file names to queue
    read_process = Process(
        target=read_func,
        args=(work_queue, args)
    )
    read_process.start()

    write_process = Process(
        target=write_func,
        args=(output_queue, args) # TODO condition
    )
    write_process.start()
    
    work_processes = []
    for _ in range(max(1, n_process-2)):
        work_process = Process(
            target=work_func,
            args=(work_queue, output_queue, args)
        )
        work_process.daemon = True
        work_process.start()
        work_processes.append(work_process)
    read_process.join()
    for work_process in work_processes:
        work_queue.put(None)
    # Now join the work processes
    for work_process in work_processes:
        work_process.join()
    output_queue.put(None)
    write_process.join()
    print(f"Finished processing in {time.time() - start_time}s")



def fetch_by_entity(args):
    work_flow(args, _read_filtered_dir, _fetch_by_qid, _write_data)

def fetch_by_property_value(args):
    work_flow(args, _read_dir, _fetch_by_property, _write_data)

if __name__ == "__main__":


    args = get_arg_parser().parse_args()
    if args.entities_file is not None:
        fetch_by_entity(args)
    if args.rel is not None and args.value is not None:
        fetch_by_property_value(args)
