#!/bin/bash


# target_dir=dataset.ja.1turn.filtered
# target_dir=dataset.en.1turn.filtered
target_dir=dataset.en.1turn.train10M.filtered

merged_ext=2017-2018
dtypes=(train dev)
train_dev_years=(2017 2018)
test_year=2019
suffixes=(src tgt tids uids utc emojis hashtags)

rm $target_dir/*${merged_ext}*

for dtype in ${dtypes[@]}; do
    for suffix in ${suffixes[@]}; do
	for year in ${train_dev_years[@]}; do
	    cat $target_dir/$dtype.$year.$suffix >> $target_dir/$dtype.$merged_ext.$suffix
	done
	ln -sfn $dtype.$merged_ext.$suffix $target_dir/$dtype.$suffix
    done
done

dtype=test
for suffix in ${suffixes[@]}; do
    ln -sfn $dtype.$test_year.$suffix $target_dir/$dtype.$suffix
done
ln -sfn $dtype.$test_year.mulres $target_dir/$dtype.mulres
ln -sfn $dtype.$test_year.mulres.tids $target_dir/$dtype.mulres.tids
