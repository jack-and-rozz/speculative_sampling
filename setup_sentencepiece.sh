#!/bin/bash
# This script trains and applies subword tokenization for the source (and the target) domain(s), respectively. As for the datasets which are constructed from the combination of multiple datasets, such as ones for multi-domain learning or back-translation, subword tokenization should be done in ./setup_*** by using subword-level datasets created by this script. 

echo "Running '$0 $@'..."

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
if [ ! -z $is_valid ]; then
    exit 1
fi

src_vocab_size=$(parse_src_vocab_size $mode)
tgt_vocab_size=$(parse_tgt_vocab_size $mode)
src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)

src_domain=$(remove_tok_suffix $src_domain)
tgt_domain=$(remove_tok_suffix $tgt_domain)

src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 


if [ -z $(which spm_encode) ]; then
    echo "spm_encode was not found. Install sentencepiece following 'https://github.com/google/sentencepiece'."
    exit 1
fi

if [[ $mode =~ multidomain ]]; then
    # echo "Subword tokenization for multidomain data should be done in ./setup_multidomain_data.sh."
    exit 1
fi
data_dir=$(get_data_dir $mode $tgt_domain)
data_sp_dir=$(get_data_dir $mode ${tgt_domain}_${sp_suffix})
sp_training_file=$(get_hyperparameter_by_domain $tgt_domain sp_training_file)

if [ ! -e $data_sp_dir ]; then
    mkdir -p $data_sp_dir
fi


files=(train dev test)
langs=($src_lang $tgt_lang)

if [ $task == translation ]; then
    for lang in ${langs[@]}; do
	if [ ! -e $data_sp_dir/spm.$lang.model ]; then
	    if [ ! -e $sp_training_file.$lang ]; then
		echo "$sp_training_file.$lang was not found".
		exit 1
	    fi

	    echo "training '$data_sp_dir/spm.$lang'..."
	    spm_train --vocab_size ${tgt_vocab_size} \
		      --model_prefix $data_sp_dir/spm.$lang \
		      --unk_surface $unk_surface \
		      --input_sentence_size $n_sentences_for_training_sp \
		      --shuffle_input_sentence \
		      --hard_vocab_limit=false \
		      --model_type=$spm_model_type \
		      --input $sp_training_file.$lang &
	fi
    done
else
    lang=$src_lang
    if [ ! -e $data_sp_dir/spm.$lang.model ]; then
	if [ ! -e $sp_training_file.$lang ]; then
	    echo "$sp_training_file.$lang was not found".
	    exit 1
	fi
	echo "training '$data_sp_dir/spm.$lang'..."
	spm_train --vocab_size ${tgt_vocab_size} \
		  --model_prefix $data_sp_dir/spm.$src_lang \
		  --unk_surface $unk_surface \
		  --input_sentence_size=$n_sentences_for_training_sp \
		  --shuffle_input_sentence \
		  --hard_vocab_limit=false \
		  --model_type=$spm_model_type \
		  --input $sp_training_file.$lang \
		  --user_defined_symbols $user_defined_symbols &
    fi
    ln -sf spm.$src_lang.model $data_sp_dir/spm.$tgt_lang.model 
    ln -sf spm.$src_lang.vocab $data_sp_dir/spm.$tgt_lang.vocab
fi
wait

for file in ${files[@]}; do
    for lang in ${langs[@]}; do
	if [ ! -e $data_sp_dir/$file.$lang ] && [ -e $data_dir/$file.$lang ]; then
	    echo "encoding '$data_dir/$file.$lang' to '$data_sp_dir/$file.$lang'..."
	    spm_encode --model $data_sp_dir/spm.$lang.model \
		       --output $data_sp_dir/$file.$lang \
		       < $data_dir/$file.$lang &
	fi
    done
done

file=monolingual
for lang in ${langs[@]}; do
    # Encode the monolingual corpus if it is provided.
    if [ ! -e $data_sp_dir/$file.$lang ] && [ -e $data_dir/$file.$lang ]; then
	echo "encoding '$data_dir/$file.$lang' to '$data_sp_dir/$file.$lang'..."
	spm_encode --model $data_sp_dir/spm.$lang.model \
    		   --output $data_sp_dir/$file.$lang \
    		   < $data_dir/$file.$lang &
    fi
done
wait 

