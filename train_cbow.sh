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
src_vocab_size=$(parse_src_vocab_size $mode)
tgt_vocab_size=$(parse_tgt_vocab_size $mode)

is_valid=$(validate_mode $mode $task)
if [ ! -z $is_valid ]; then
    exit 1
fi

src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 

case $mode in
    *.${baseline_suffix}*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	;;
    *.finetune.*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	;;
    *@*.backtranslation_aug*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	;;
    *@*.multidomain.domainweighting*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainweighting)
	;;
    *@*.multidomain.domainmixing*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainmixing)
	;;

    * ) echo "invalid mode"
        exit 1 ;;
esac

if [ $task != translation ]; then 
    # src_data=$data_dir/train.joined
    if [ ! -e $data_dir/monolingual.joined ]; then
	cat $data_dir/monolingual.$src_lang > $data_dir/monolingual.joined
	cat $data_dir/monolingual.$tgt_lang >> $data_dir/monolingual.joined
    fi
    src_data=$data_dir/monolingual.joined
    src_output=$data_dir/word2vec.${src_lang}.${emb_size}d
else
    src_data=$data_dir/monolingual.${src_lang}
    tgt_data=$data_dir/monolingual.${tgt_lang}
    src_output=$data_dir/word2vec.${src_lang}.${emb_size}d
    tgt_output=$data_dir/word2vec.${tgt_lang}.${emb_size}d
fi

# Sample examples from monolingual data to reduce computational costs.
if [ ! -e $src_data.$w2v_samplesize ]; then
    python -c "import random; random.seed($random_seed); data=[l for l in open('$src_data')]; idxs=set(random.sample(range(len(data)), min($w2v_samplesize, len(data)))); f=open('$src_data.$w2v_samplesize', 'w'); data=[f.write(l) for i, l in enumerate(data) if i in idxs]"
fi
if [ ! -z $tgt_data ] && [ ! -e $tgt_data.$w2v_samplesize ]; then
    python -c "import random; random.seed($random_seed); data=[l for l in open('$tgt_data')]; idxs=set(random.sample(range(len(data)), min($w2v_samplesize, len(data)))); f=open('$tgt_data.$w2v_samplesize', 'w'); data=[f.write(l) for i, l in enumerate(data) if i in idxs]"
fi


if [ ! -e $src_output ]; then
    echo "Training word2vec $src_output from $src_data..."
    ./tools/word2vec/word2vec -size ${emb_size} \
			      -train $src_data.$w2v_samplesize \
			      -output $src_output \
			      -save-vocab $src_output.vocab \
			      -min-count $w2v_mincount & 
fi

if [ ! -e $tgt_output ]; then
    if [ ! -z $tgt_data ]; then
	echo "Training word2vec $tgt_output from $tgt_data..."
	./tools/word2vec/word2vec -size ${emb_size} \
				  -train $tgt_data.$w2v_samplesize \
				  -output $tgt_output \
				  -save-vocab $tgt_output.vocab \
				  -min-count $w2v_mincount &
    # else
    # 	echo "$tgt_output already exists."
    else
	ln -sf word2vec.${src_lang}.${emb_size}d $data_dir/word2vec.${tgt_lang}.${emb_size}d
	ln -sf word2vec.${src_lang}.${emb_size}d.vocab $data_dir/word2vec.${tgt_lang}.${emb_size}d.vocab

    fi
fi
wait

# # Removing lines which can't be decoded in utf-8.
# if [ $mode == reddit ];then
#     mv $src_output $src_output.origin
#     python scripts/reddit/filter_cbows.py $src_output.origin > $src_output
# fi

# 1. <pad>, </s>, <unk>, <s> (symbols defined in fairseq/data/dictionary.py)
# 2. ▁<eot>, ▁<eou> (must be included to handle multi-turn dialogs)
n_special_words=4
src_dict=$data_dir/dict.${src_lang}.txt
tgt_dict=$data_dir/dict.${tgt_lang}.txt


if [ ! -e $src_dict ] && [ -e $src_output.vocab ]; then
   n_words=$((tgt_vocab_size-n_special_words)) # tgt_vocab_size indicates the vocabulary size in the **target domain** (TODO: two meanings of 'src' are misleading...).
   if [ $(cat $src_output.vocab | grep "${eou_token}" | wc -l) == 0 ]; then
       echo "${eou_token} 0" >> $src_dict
       n_words=$((n_words-1))
   fi

   if [ $(cat $src_output.vocab | grep "${eot_token}" | wc -l) == 0 ]; then
       echo "${eot_token} 0" >> $src_dict
       n_words=$((n_words-1))
   fi
   sed -n "2,$((n_words+1))p" $src_output.vocab >> $src_dict
   echo "Creating vocabulary file (#tokens=$n_words) from '${src_output}.vocab' to '$src_dict'."

fi

if [ ! -e $tgt_dict ] && [ ! -z ${tgt_lang} ] && [ -e $tgt_output.vocab ]; then
   n_words=$((tgt_vocab_size-n_special_words))
   if [ $(cat $tgt_output.vocab | grep "${eou_token}" | wc -l) == 0 ]; then
       echo "${eou_token} 0" >> $tgt_dict
       n_words=$((n_words-1))
   fi

   if [ $(cat $tgt_output.vocab | grep "${eot_token}" | wc -l) == 0 ]; then
       echo "${eot_token} 0" >> $tgt_dict
       n_words=$((n_words-1))
   fi
   sed -n "2,$((n_words+1))p" $tgt_output.vocab >> $tgt_dict
   echo "Creating vocabulary file (#tokens=$n_words) from '${tgt_output}.vocab' to '$tgt_dict'."

fi

