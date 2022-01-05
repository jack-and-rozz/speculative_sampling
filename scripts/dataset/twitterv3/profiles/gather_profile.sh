#!/bin/bash

# date=20201101
date=$(date "+%Y%m%d")
earthquake=crawls/earthquake
earthquake_update=crawls/earthquake.update

if [ ! -e $date/user.prof.uniq ]; then

    if [ ! -e list.update ] || [ ! -e list ]; then
	ls $earthquake_update/tweets*[0-9].gz |  LC_ALL=C sort > list.update 
	for year in $(seq 2011 2019); do
	    for month in $(seq 01 12); do
		month=$(printf "%02d" $month)
		ls $earthquake_update/$year-$month/tweets*[0-9].gz | LC_ALL=C sort >> list.update
	    done
	done
	ls $earthquake/tweets*[0-9].gz | LC_ALL=C sort > list
	wait
    fi
    cat list list.update | ruby -ne 'f = $_.split("/")[-1][0..-5]; puts "zcat #{$_.chomp} | ./tweets_to_user > prof/#{f}.prof 2> prof/#{f}.prof.err"' | xargs -P 12 -n 1 -I % sh -c '%' &
    wait

    find prof -type f | LC_ALL=C sort | xargs cat | LC_ALL=C sort -S 40G -T . -t $'\7' -k1,1n -k2,2n -k3,3n | ./merge_user > $date/user.prof 2> $date/user.prof.log
    cat $date/user.prof | ./merge_user_uniq > $date/user.prof.uniq
fi

cut -f2 -d $'\7' $date/user.prof.uniq > $date/user.prof.uid
cut -f10 -d $'\7' $date/user.prof.uniq > $date/user.prof.location
cut -f13 -d $'\7' $date/user.prof.uniq > $date/user.prof.description
