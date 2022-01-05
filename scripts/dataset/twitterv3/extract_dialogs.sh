#!/bin/bash

usage() {
    echo "Usage:$0"
    exit 1
}

# if [ $# -lt 1 ];then
#     usage;
# fi

# years=(2019)
years=(2016 2017 2018 2019)
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
# exit 1
# for year in ${years[@]}; do
#     for lang in ${langs[@]}; do
# 	if [ ! -e $target_dir/$year-all.$lang.dialogs ]; then
# 	    for file in $(ls $target_dir/$year-*.$lang.dialogs); do
# 		cat $file >> $target_dir/$year-all.$lang.dialogs
# 	    done;
# 	fi
# 	if [ ! -e $target_dir/$year-all.$lang.tids ]; then
# 	    for file in $(ls $target_dir/$year-*.$lang.tids); do
# 		cat $file >> $target_dir/$year-all.$lang.tids
# 	    done;
# 	fi
# 	if [ ! -e $target_dir/$year-all.$lang.uids ]; then
# 	    for file in $(ls $target_dir/$year-*.$lang.uids); do
# 		cat $file >> $target_dir/$year-all.$lang.uids
# 	    done;
# 	fi
# 	if [ ! -e $target_dir/$year-all.$lang.utime ]; then
# 	    for file in $(ls $target_dir/$year-*.$lang.utime); do
# 		cat $file >> $target_dir/$year-all.$lang.utime
# 	    done;
# 	fi
# 	if [ ! -e $target_dir/$year-all.$lang.hashtags ]; then
# 	    for file in $(ls $target_dir/$year-*.$lang.hashtags); do
# 		cat $file >> $target_dir/$year-all.$lang.hashtags
# 	    done;
# 	fi
# 	if [ ! -e $target_dir/$year-all.$lang.distractors ]; then
# 	    for file in $(ls $target_dir/$year-*.$lang.distractors); do
# 		cat $file >> $target_dir/$year-all.$lang.distractors
# 	    done;
# 	fi
#     done;
# done;
