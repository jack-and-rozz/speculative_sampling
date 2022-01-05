#!/bin/bash

usage() {
    echo "Usage:$0"
    exit 1
}

# if [ $# -lt 1 ];then
#     usage;
# fi

years=(2017 2018 2019)
source_dir=original.tweets
target_dir=original.dialogs
langs=(ja en)
for year in ${years[@]}; do
    for lang in ${langs[@]}; do
	for day in $(seq 1 31); do
	    day=`printf %02d $day`
	    for month in $(seq 1 12); do
		month=`printf %02d $month`
		if [ -e $source_dir/$year-$month-$day.$lang.sort ]; then
		    source_path=$source_dir/$year-$month-$day.$lang.sort
		    python scripts/dataset/twitterv3/extract_dialogs.py \
		    	   $source_path $lang \
		    	   --target_dir $target_dir  &
		fi
	    done;
	    wait
	done;
    done;
done;
