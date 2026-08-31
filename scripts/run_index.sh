#!/bin/bash
python -m src.build_index --chunks chunks.pkl --save-dir index --use-local --embedding-model snowflake-arctic-embed2
# python -m src.build_index --chunks chunks.pkl --save-dir index "$@"