#!/bin/bash
echo "Running '$0 $1 $2'..."

usage() {
    echo "Usage:$0 mode task"
    exit 1
}
if [ $# -lt 2 ];then
    usage;
fi


. ./const.sh $mode $task

mode=$1
task=$2
is_valid=$(validate_mode $mode $task)
if [ -n "$is_valid" ]; then
    exit 1
fi

src_vocab_size=$(parse_src_vocab_size $mode)
tgt_vocab_size=$(parse_tgt_vocab_size $mode)
size=$(parse_size $mode)
src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
src_domain_wd=$(remove_tok_suffix $src_domain)
tgt_domain_wd=$(remove_tok_suffix $tgt_domain)

src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 

suffix=".$size"
case $mode in
    *)
	data_dir=$(get_data_dir $mode $tgt_domain)
	data_dir_wd=$(get_data_dir $mode $tgt_domain_wd)

	# Training data is not needed.
	train_data=$data_dir/dummy
	dev_data=$data_dir/dev
	test_data=$data_dir/test_sampled
	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
                 --nwordssrc $src_vocab_size \
                 --nwordstgt $tgt_vocab_size \
		 --trainpref $train_data     \
		 --validpref $dev_data   \
		 --testpref $test_data 
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt

	if [ ! -e $data_dir/dummy.$src_lang ]; then
	    touch $data_dir/dummy.$src_lang
	    echo "dummy." > $data_dir/dummy.$src_lang
	fi
	if [ ! -e $data_dir/dummy.$tgt_lang ]; then
	    touch $data_dir/dummy.$tgt_lang
	    echo "dummy." > $data_dir/dummy.$tgt_lang
	fi
	destdir=$data_dir/fairseq.analysis
	;;
esac


if [ -e $data_dir_wd/test.mulres.$tgt_lang ] && [ ! -e $data_dir_wd/test.mulres.$tgt_lang ]; then
    python scripts/analysis/encode_mulres.py \
	   --input-path $data_dir_wd/test.mulres.$tgt_lang \
	   --output-path $data_dir/test.mulres.$tgt_lang \
	   --spm-path $data_dir/spm.$tgt_lang.model
fi

test_data_suffixes=($src_lang $tgt_lang)
if [ -e $data_dir/test.mulres.$tgt_lang ]; then
    test_data_suffixes=("${test_data_suffixes[@]}" mulres.$tgt_lang)
fi

for suffix in ${test_data_suffixes[@]}; do
    if [ ! -e $data_dir/test_sampled.$suffix ]; then
	python scripts/analysis/pickup_data_from_indice.py \
	       --indice-path $data_dir/test_sampled.idx \
	       < $data_dir/test.$suffix \
	       > $data_dir/test_sampled.$suffix
    fi
done



if [ $task != translation ]; then
    options="$options --srcdict $src_dict  --joined-dictionary"
else
    options="$options --srcdict $src_dict --tgtdict $tgt_dict"
fi

if [ ! -e $destdir/test.$src_lang-$tgt_lang.$src_lang.bin ] || [ ! -n "$(ls $destdir)" ]; then

    echo "Creating binary files with fairseq format to '$destdir'..."
    mkdir -p $destdir
    python fairseq/preprocess.py \
	   --destdir $destdir \
	   $options \
	   --workers 16
fi
