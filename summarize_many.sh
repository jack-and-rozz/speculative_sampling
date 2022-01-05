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
reference_path=$tgt_data_dir/test.$tgt_lang
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


if [ $tgt_lang == ja ]; then
    if [ ! -e $reference_path.retok ]; then
	echo $reference_path
	python -c "[print(''.join(l.strip().split())) for l in open('${reference_path}')]" > $reference_path.tmp
	$ja_tokenizer_command \
	    < $reference_path.tmp \
	    > $reference_path.retok 
	rm $reference_path.tmp
    fi
    for out in $(ls $ckpt_root/$tgt_domain*/tests/*/$output_filename); do
	test $out.retok -ot $out
	if [ $? == 0 ]; then
	    python -c "[print(''.join(l.strip().split())) for l in open('${out}')]" > $out.tmp
	    $ja_tokenizer_command \
		< $out.tmp \
		> $out.retok &
	fi
    done
else
    echo "TODO"
    exit 1
fi

python scripts/summarize_many.py $ckpt_root \
       $output_filename.retok $input_file $reference_path.retok \
       --task $task \
       --src_enc_vocab $src_enc_vocab \
       --src_dec_vocab $src_dec_vocab \
       --tgt_enc_vocab $tgt_enc_vocab \
       --tgt_dec_vocab $tgt_dec_vocab \
       --n_vocab $n_vocab \
       --max_rows $max_rows \
       --target_sizes all 
       # --target_sizes all \
       # --disable_output_all 

