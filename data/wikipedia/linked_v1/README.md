# Linking (Wikipedia <-> Wikidata) v1.0
Criteria: eav, av, ev
Conditions for v1.0: 
- avoid duplicated surfaces
- avoid stopwords as entity/value surfaces
- surface matching with word boundry
- other: (1) value form transformation: date -> year

The linking code can be found here: `vol2/myu2/ComparisonSentences/code/linking/linking.py`

## Linked data:

- eav: `./eav`
- av: `./eav`
- ev: `./ev`
- three criteria combined: `./combined`

Each `wiki_xx` file contains lines of json objects. See an example at the bottom of this page.


## Linking statistics

*Detailed analysis can be found in `./statistics`*

- original data (before linking):
    - Number of entity: 6,123,051
    - Number of sentence: 112,811,173
    - Number of statement: 58,470,459

- eav:
    - Number of linked entity: 2,020,247
    - Number of linked sentence: 2,562,439
    - Number of linked statement: 3,497,204

- av:
    - Number of linked entity: 3,468,418
    - Number of linked sentence: 7,095,175
    - Number of linked statement: 8,707,800

- ev:
    - Number of linked entity: 4,495,131
    - Number of linked sentence: 8,824,827
    - Number of linked statement: 22,119,924

- **combined**:
    - Number of linked entity: 5,096,290 (coverage: 83%)
    - Number of linked sentence: 13,302,148 (coverage: 12%)
    - Number of linked statement: 27,330,519 (coverage: 47%)

I pick some properties to compare their value distribution with the original data, here's a Google sheet to visualize it. [Google Sheet](https://docs.google.com/spreadsheets/d/1K9uvUsQVKv42WfmUGgQVf1HaK_-nv9O310CGdW7fgPA/edit?usp=sharing)

The analysis code can be found here: `vol2/myu2/ComparisonSentences/code/linking/analysis.py`




## Example 
Criterion: eav

```
{
    "id": "659",
    "title": "American National Standards Institute",
    "qid": "Q180003",
    "linked_entity_rels": [
        [
            5,
            "The organization's headquarters are in Washington, D.C. ANSI's operations office is located in New York City.",
            [
                [
                    [
                        "Q180003",
                        "P159",
                        "Q61"
                    ],
                    "('ANSI', 'headquarters', 'Washington')"
                ]
            ]
        ]
    ],
    "linked_entity_values": [
        [
            8,
            "ANSI was most likely originally formed in 1918, when five engineering societies and three government agencies founded the American Engineering Standards Committee (AESC).",
            [
                [
                    [
                        "Q180003",
                        "P571",
                        "+1918-10-19T00:00:00Z"
                    ],
                    "('ANSI', 'formed in', '1918')"
                ]
            ]
        ]
    ]
}
```