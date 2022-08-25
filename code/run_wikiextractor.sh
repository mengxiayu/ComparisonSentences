python -m wikiextractor.WikiExtractor \
    -b 100M \ 
    -o /Users/mengxiayu/Downloads/wikipedia/text \
    --json --processes 4 \
    --no-templates \
    /Users/mengxiayu/Downloads/wikipedia/enwiki-20220101-pages-articles-multistream/enwiki-20220101-pages-articles-multistream.xml.bz2