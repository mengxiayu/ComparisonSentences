# Comparison Sentences

wikipedia dump location: /afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikipedia_dump/enwiki-20220101-pages-articles-multistream.xml.bz2
wikipedai dump index location: /afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikipedia_dump/enwiki-20220101-pages-articles-multistream-index.txt.bz2
wikidata dump: /afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikidata_dump/wikidata-20220103-all.json.gz

## Parse Data Dump
wikipedia parsed:
wikidata parsed: /afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikidata (wikipedia_links, property_labels, labels, external_ids, entity_values, entity_rels, descriptions, aliases)


## Data Instructions

### Wikidata
There are 7 types of data tables in the parsed Wikidata. Each of them is a folder. Under the folder there are files storing the records in batch. The files are in TSV format. You can open them with `pandas.read_csv(f, sep='\t')`. Below are the column names of each type of the tables
- wikipedia_links ()




**Wikipedia Extractor**
/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/code/run_wikiextractor.sh


## Fetch statements

### some property and value
Q902104: private university
Q875538: public university
Q62078547: public research university
Q3918: university

Q5: human
P31: instance of




Fetching statements for a list of entity and save to a table
```sh
python code/fetching/fetch_kg.py \
--table_name entity_rels \
--num_procs 20 \
--output /afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata/tables \
--entities_file /afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata/entity_lists/trump.txt \
--output_name trump
```

Fetching statements for a specific property and value, and save to a table
```sh
python code/fetching/fetch_kg.py --table_name entity_rels --num_procs 10 --output /afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata/tables --rel P31  --value Q902104 --output_name P31_Q902104
```

