#!/bin/bash

. ./const.sh
root=$(pwd)
current=$(cd $(dirname $0);pwd)

# python $current/create_dataset.py ja \
#        --train-dev-years 2018 \
#        --test-years 2019 \
#        --target_dir dataset.ja 


cd $twitterv3ja_data_dir
suffixes=(src tgt)
train_dev_year=2018
test_year=2019
for suffix in ${suffixes[@]}; do
    ln -sf train.$train_dev_year.$suffix train.$suffix
    ln -sf dev.$train_dev_year.$suffix dev.$suffix
    ln -sf test.$test_year.$suffix test.$suffix
done






