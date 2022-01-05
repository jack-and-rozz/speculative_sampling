#!/bin/bash

source_dir=prof.gathered
if [ ! -e $source_dir ]; then
    echo "Error: the target directory 'prof.gathered' was not found. Run process_description.py"
    exit 1
fi
if [ ! -e $source_dir/user.prof.description.processed.mecab ]; then
    processed_prof=$source_dir/user.prof.description.processed
    mecab -Owakati \
	  < $processed_prof \
	  > $processed_prof.mecab
fi

# cat $source_dir/user.prof.description.joined | grep -i " bot " | cut -f1 > $source_dir
cat $source_dir/user.prof.description.processed.mecab.joined | grep -i " bot " | cut -f1 > $source_dir/bot.uid
