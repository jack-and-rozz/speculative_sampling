#!/bin/bash
echo "Running '$0 $@'..."
usage() {
    echo "Usage:$0 src_domain tgt_domain task (n_vocab)"
    exit 1
}
if [ $# -lt 3 ];then
    usage;
fi

. ./const.sh $mode $task
# mode=$1
# task=$2

src_domain=$1
tgt_domain=$2
task=$3
max_rows=$4
# n_vocab=$4

if [ -z $n_vocab ]; then
    n_vocab=0
fi
if [ -z $max_rows ]; then
    max_rows=0
fi

src_domain=$(remove_tok_suffix $src_domain)
tgt_domain=$(remove_tok_suffix $tgt_domain)

src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 

# (TODO)
src_data_dir=$(get_data_dir $src_domain.$baseline_suffix.all $src_domain)
tgt_data_dir=$(get_data_dir $tgt_domain.$baseline_suffix.all $tgt_domain)

input_file=$tgt_data_dir/test.$src_lang
reference_filename=$tgt_data_dir/test.$tgt_lang
src_enc_vocab=$src_data_dir/dict.$src_lang.txt
src_dec_vocab=$src_data_dir/dict.$tgt_lang.txt
tgt_enc_vocab=$tgt_data_dir/dict.$src_lang.txt
tgt_dec_vocab=$tgt_data_dir/dict.$tgt_lang.txt
output_filename=$tgt_domain.outputs

if [ $task == translation ]; then
    ja_tokenizer_command="kytea -notags"
else
    ja_tokenizer_command="mecab -Owakati"
fi

# Apply normalization to the outputs, remove all spaces, and tokenize them when the output language is Japanese.
if [ $tgt_lang == ja.tgt ]; then
    if [ ! -e $reference_filename.normed ]; then
	python -c "import unicodedata; [print(''.join(unicodedata.normalize('NFKC', s).strip().split())) for s in open('$reference_filename')]" > $reference_filename.normed
    fi
    if [ ! -e $reference_filename.normed.tok ]; then
	$ja_tokenizer_command \
	    < $reference_filename.normed \
	    > $reference_filename.normed.tok &

    fi

    for out in $(ls $ckpt_root/*/tests/$output_filename); do
	test $out.normed -ot $out
	if [ $? == 0 ]; then
	    echo "Applying tokenization to "$out
	    python -c "import unicodedata; [print(unicodedata.normalize('NFKC', ''.join(s.strip().split()))) for s in open('$out')]" > $out.normed
	    $ja_tokenizer_command < $out.normed > $out.normed.tok &
	fi
    done
    wait
    output_filename=$output_filename.normed.tok
    reference_filename=$reference_filename.normed.tok

# TODO: fix hardcoding
elif [ $tgt_lang == en.tgt ] || [ $tgt_domain == twitterv3en ]; then 
    if [ ! -e $reference_filename.tok ]; then
	echo "The tokenized reference file '$reference_filename.tok' was not found. Applying the tokenizer..."
	perl $tokenizer_path < $reference_filename > $reference_filename.tok & 

    fi
    for out in $(ls $ckpt_root/*/tests/$output_filename); do
	test $out.tok -ot $out
	if [ $? == 0 ]; then
	    perl $tokenizer_path \
		  < $out \
		  > $out.tok &
	fi
    done
    wait
    output_filename=$output_filename.tok
    reference_filename=$reference_filename.tok
fi


options=""
# if [[ $tgt_domain =~ twitterv3en ]]; then
#     options="$options --case-insensitive"
# fi


python scripts/summarize_result.py $ckpt_root \
       $output_filename $input_file $reference_filename \
       --task $task \
       --src_enc_vocab $src_enc_vocab \
       --src_dec_vocab $src_dec_vocab \
       --tgt_enc_vocab $tgt_enc_vocab \
       --tgt_dec_vocab $tgt_dec_vocab \
       # --case-insensitive \
       --n_vocab $n_vocab \
       --max_rows $max_rows \
       $options \
       --target_sizes all
       # --target_sizes all \
       # --disable_output_all

python scripts/calc_dist.py $ckpt_root \
       $output_filename \
       $input_file \
       $reference_filename
       # --case-insensitive \

exit 1

if [ $task == descgen ]; then
    word_list=$tgt_data_dir/test.word
fi
if [ $task == descgen ] || [ $task == dialogue ]; then
    if [ -z $word_list ]; then
	python scripts/calc_bleu_descgen.py $ckpt_root $output_filename $reference_filename --sentence_bleu_path=$sentence_bleu_path
    else
	python scripts/calc_bleu_descgen.py $ckpt_root $output_filename $reference_filename --word_list=$word_list --sentence_bleu_path=$sentence_bleu_path
    fi
fi

