
def _search_statement(path, qid):

    if len(qid) < 3:
        batch_file = f"Q{int(qid[1]):02d}.tsv"
    else:
        batch_file = f"Q{qid[1:3]}.tsv"
    
    flag = False
    results = []
    with open(path / batch_file) as f:
        for line in f:
            if f"\t{qid}\t" in line:
                results.append(line.strip())
                flag = True # found the entity
            elif flag == True: # 
                break
            else:
                continue
    return results

from pathlib import Path
path = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/entity_rels_sorted")
qid = "Q4729312"
print(_search_statement(path, qid))