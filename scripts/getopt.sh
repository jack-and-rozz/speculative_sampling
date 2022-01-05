#!/bin/bash
 #自分でオプションを解析。順番自由、ロングオプションも可能
# http://qiita.com/b4b4r07/items/dcd6be0bb9c9185475bb
# . ./scripts/manually_getopt.sh $@

declare -i argc=0
declare -a argv=()

opt_names=()
opt_values=()
long_flags=()
short_flags=()

while (( $# > 0 ))
do
    case "$1" in
        --*=*)
            if [[ "$1" =~ ^--(.+?)=(.+)$ ]]; then
		opt_names=("${opt_names[@]}" ${BASH_REMATCH[1]})
		opt_values=("${opt_values[@]}" ${BASH_REMATCH[2]})
            fi
            shift
            ;;
        --*)
            if [[ "$1" =~ ^--(.+?)$ ]]; then
		long_flags=("${long_flags[@]}" ${BASH_REMATCH[1]})
            fi
            shift
            ;;
        -*)
            if [[ "$1" =~ ^-(.+?)$ ]]; then
		short_flags=("${short_flags[@]}" ${BASH_REMATCH[1]})
            fi
            shift
            ;;
            # if [[ "$1" =~ 'n' ]]; then
            #     nflag='-n'
            # fi
            # shift
            # ;;
        *)
            ((++argc))
            argv=("${argv[@]}" "$1")
            shift
            ;;
    esac
done

for i in $(seq 0 $(expr ${#opt_names[@]} - 1)); do
    name=${opt_names[$i]}
    value=${opt_values[$i]}
    eval $name=$value
done;


function has_flag(){
    varname=$1
    for flag in ${long_flags[@]}; do
	if [ $flag == $varname ]; then
	    res='included'
	fi
    done
    for flag in ${short_flags[@]}; do
	if [ $flag == $varname ]; then
	    res='included'
	fi
    done
    if [ ! -z $res ]; then 
	echo 0
    else
	echo 1
    fi
}

# echo opt_names=${opt_names[@]}
# echo opt_vales=${opt_values[@]}
# echo long_flags=${long_flags[@]}
# echo short_flags=${short_flags[@]}
# echo 'argc='$argc
# echo 'argv='${argv[@]}
