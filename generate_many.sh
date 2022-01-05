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
if [ ! -z $is_valid ]; then
    exit 1
fi


shard_id=0 # Test file ID for generation. The correspondense is defined by args.testpref in preprocess.sh.
num_shards=1 # Number of testing files.

size=$(parse_size $mode)
src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 
input_lang=$src_lang
output_lang=$tgt_lang
test_file=$tgt_domain

model_dir=$(get_model_dir $ckpt_root $mode)
if [ ! -e $model_dir/tests ]; then
    mkdir -p $model_dir/tests
fi
data_dir=$(get_data_dir $mode $tgt_domain)

case $mode in
    #####################################
    ##        Translation
    #####################################

    ##### Test in ASPEC #####
    *.${baseline_suffix}*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size
	;;

    ## w/o fine-tuning
    *${direction_tok}*.noadapt*) # Test ASPEC data with JESC vocabulary, by JESC model
	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.v_$src_domain.all
	./preprocess.sh $mode $task
	;;

    ## w/ fine-tuning
    *${direction_tok}*.finetune.v_$src_domain.*)
	model_dir=$ckpt_root/$mode
	data_dir=$(get_data_dir $mode $tgt_domain)
	src_data_dir=$(get_data_dir $mode $src_domain)
	data=$data_dir/fairseq.v_$src_domain.$size
	;;

    *${direction_tok}*.finetune.v_$tgt_domain.*)
 	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size
	;;

    *${direction_tok}*.finetune.*.*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size
	;;

    ## Multi-domain
    *${direction_tok}*.multidomain.domainmixing.*) 
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainmixing)
	data=$data_dir/fairseq.$size
	;;
    *${direction_tok}*.multidomain.domainweighting.*) 
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainweighting)
	data=$data_dir/fairseq.$size
	;;

    ## Back-translation
    *${direction_tok}*.backtranslation_aug*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain backtranslation_aug)
	data=$data_dir/fairseq.$size
	output_lang=$src_lang
	input_lang=$tgt_lang
	data_options="--skip-invalid-size-inputs-valid-test"
	;;

    *${direction_tok}*.backtranslation_ft.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain backtranslation_ft)
	data=$data_dir/fairseq.$size
	;;
    *${direction_tok}*.backtranslation_va.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain backtranslation_va)
	data=$data_dir/fairseq.$size
	;;
    * ) echo "invalid mode: $mode"
        exit 1 
	;;
esac


if [ ! -e $model_dir/checkpoints/checkpoint_best.pt ]; then
    echo "$model_dir/checkpoints/checkpoint_best.pt was not found."
    exit 1
fi


decode(){
    test_dir=$1
    decode_options=$2
    data_options=$3
    ckpt_options=$4
    suffix=$5
    if [ ! -e $test_dir ]; then
	mkdir -p $test_dir
    fi

    # If the output file is older than the best checkpoint
    log_file=$test_dir/$test_file.log${suffix}
    output_file=$test_dir/$test_file.outputs${suffix}
    input_file=$test_dir/$test_file.inputs
    ref_file=$test_dir/$test_file.refs


    # Run testing again if the outputs were older than the best checkpoint.
    test $log_file -nt $model_dir/checkpoints/checkpoint_best.pt
    subword_output_is_latest=$?

    if [ -s $log_file ] && [ -e $log_file ] && [ $subword_output_is_latest == 0 ]; then
	echo "$log_file is up-to-date."
	return
    else
	python fairseq/generate_many.py \
	       --user-dir ${fairseq_user_dir} \
	       --task ${fairseq_task} \
	       --shard-id $shard_id \
	       --num-shards $num_shards \
	       --skip-invalid-size-inputs-valid-test \
	       --source-lang $src_lang \
	       --target-lang $tgt_lang \
	       --max-tokens 16000 \
	       $decode_options \
	       $ckpt_options \
	       $data_options \
	       >> $log_file
    fi


    if [[ ${mode} =~ domainmixing ]]; then
	# Remove the first generated token when domain mixing is employed.
	# cat $log_file | grep "^H-" | cut -f1,3 | cut -c 3- |sort -k 1 -n | python -c "import sys; out=[l.strip().split('\t') for l in sys.stdin]; out={int(l[0]):l[1] for l in out};[print(out[i]) if i in out else print("") for i in range(max(out.keys()) + 1)]" | cut -d ' ' -f2-  > $output_file
	echo "TODO"
	return 
    else
	# cat $log_file | grep "^H-" | cut -c 3- |sort -k 1 -n | cut -f1,3 | python -c "import sys; out=[l.strip().split('\t') for l in sys.stdin]; out={int(l[0]):l[1] for l in out};[print(out[i]) if i in out else print("") for i in range(max(out.keys()) + 1)]" > $output_file
	cat $log_file | grep "^H-" | cut -c 3- |sort -k 1 -n | cut -f1,3 | python -c "import sys; out=[print(l.strip().split('\t')[1]) for l in sys.stdin];" > $output_file

    fi


    if [ ! -e $input_file ]; then
	cat $log_file | grep "^S-" | cut -c 3- |sort -k 1 -n | python -c "import sys; out=[l.strip().split('\t') for l in sys.stdin]; out={int(l[0]):l[1] for l in out};[print(out[i]) if i in out else print("") for i in range(max(out.keys()) + 1)]" > $input_file
    fi

    if [ ! -e $ref_file ]; then
	cat $log_file | grep "^T-" | cut -c 3- |sort -k 1 -n | python -c "import sys; out=[l.strip().split('\t') for l in sys.stdin]; out={int(l[0]):l[1] for l in out};[print(out[i]) if i in out else print("") for i in range(max(out.keys()) + 1)]" > $ref_file
    fi

    if [[ ${mode} =~ _${sp_suffix} ]]; then
	tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
	word_level_outputs=$test_dir/${tgt_domain_wd}.outputs${suffix}
	word_level_refs=$test_dir/${tgt_domain_wd}.refs
	word_level_inputs=$test_dir/${tgt_domain_wd}.inputs

	# Run decoding again if the word-level outputs were older than the subword-level outputs.
	test $word_level_outputs -nt $output_file
	word_output_is_latest=$?
	if [ $word_output_is_latest != 0 ]; then
	    echo "Applying subword detokenization to $output_file..."
	    if [ -e $model_dir/subword/spm.$output_lang.model ]; then
		spm_decode --model $model_dir/subword/spm.$output_lang.model \
			   --output $word_level_outputs \
			   < $output_file
		spm_decode --model $model_dir/subword/spm.$input_lang.model \
			   --output $word_level_refs \
			   < $ref_file
		spm_decode --model $model_dir/subword/spm.$input_lang.model \
			   --output $word_level_inputs \
			   < $input_file
	    else
		spm_decode --model $data_dir/spm.$output_lang.model \
			   --output $word_level_outputs \
			   < $output_file
		spm_decode --model $data_dir/spm.$output_lang.model \
			   --output $word_level_refs \
			   < $ref_file
		spm_decode --model $data_dir/spm.$input_lang.model \
			   --output $word_level_inputs \
			   < $input_file
	    fi
	fi
    fi
}


data_options="$data $data_options"
ckpt_options="--path $model_dir/checkpoints/checkpoint_best.pt \
	      --results-path $test_dir/$test_file "

# Change the output directory depending on the decoding method.
if [[ $mode =~ \.tcvae\. ]]; then
    K=40
    beam_size=5
    nbest=5
    test_dir="$model_dir/tests/sample${beam_size}-topK${K}"
    decode_options="--lenpen 1.2 --seed $random_seed --sampling \
                    --sampling-topk $K \
		    --nbest $beam_size --beam $beam_size"
    decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options"


    beam_size=1
    num_sample=5
    test_dir="$model_dir/tests/latent${num_sample}-beam${beam_size}"
    for i in $(seq 0 $(($num_sample-1))); do
	i=`printf %03d ${i}`
	decode_options="--lenpen 1.2 --seed $i \
			--nbest $beam_size --beam $beam_size"
	suffix=".$i"
	decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options" "$suffix"
    done
    tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
    word_level_outputs=$test_dir/${tgt_domain_wd}.outputs
    python -c "from itertools import chain; outputs=[[l.rstrip() for l in open('${word_level_outputs}.%03d' % i)] for i in range(${num_sample})]; outputs=list(zip(*outputs)); outputs=list(chain.from_iterable(outputs)); [print(l) for l in outputs]" > $word_level_outputs

    beam_size=5
    num_sample=5
    test_dir="$model_dir/tests/latent${num_sample}-beam${beam_size}"
    for i in $(seq 0 $(($num_sample-1))); do
	i=`printf %03d ${i}`
	decode_options="--lenpen 1.2 --seed $i \
			--nbest 1 --beam $beam_size"
	suffix=".$i"
	decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options" "$suffix"
    done
    tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
    word_level_outputs=$test_dir/${tgt_domain_wd}.outputs
    python -c "from itertools import chain; outputs=[[l.rstrip() for l in open('${word_level_outputs}.%03d' % i)] for i in range(${num_sample})]; outputs=list(zip(*outputs)); outputs=list(chain.from_iterable(outputs)); [print(l) for l in outputs]" > $word_level_outputs




    K=20
    beam_size=1
    num_sample=5
    test_dir="$model_dir/tests/latent${num_sample}-topK${K}"
    for i in $(seq 0 $(($num_sample-1))); do
	i=`printf %03d ${i}`
	decode_options="--lenpen 1.2 --seed $i --sampling \
	                --sampling-topk ${K} \
			--nbest $beam_size --beam $beam_size"
	suffix=".$i"
	decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options" "$suffix"
    done
    tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
    word_level_outputs=$test_dir/${tgt_domain_wd}.outputs
    python -c "from itertools import chain; outputs=[[l.rstrip() for l in open('${word_level_outputs}.%03d' % i)] for i in range(${num_sample})]; outputs=list(zip(*outputs)); outputs=list(chain.from_iterable(outputs)); [print(l) for l in outputs]" > $word_level_outputs

    K=40
    beam_size=1
    num_sample=5
    test_dir="$model_dir/tests/latent${num_sample}-topK${K}"
    for i in $(seq 0 $(($num_sample-1))); do
	i=`printf %03d ${i}`
	decode_options="--lenpen 1.2 --seed $i --sampling \
	                --sampling-topk ${K} \
			--nbest $beam_size --beam $beam_size"
	suffix=".$i"
	decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options" "$suffix"
    done
    tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
    word_level_outputs=$test_dir/${tgt_domain_wd}.outputs
    python -c "from itertools import chain; outputs=[[l.rstrip() for l in open('${word_level_outputs}.%03d' % i)] for i in range(${num_sample})]; outputs=list(zip(*outputs)); outputs=list(chain.from_iterable(outputs)); [print(l) for l in outputs]" > $word_level_outputs

    P=0.50
    beam_size=1
    num_sample=5
    test_dir="$model_dir/tests/latent${num_sample}-topP${P}"
    for i in $(seq 0 $(($num_sample-1))); do
	i=`printf %03d ${i}`
	decode_options="--lenpen 1.2 --seed $i --sampling \
	                --sampling-topp ${P} \
			--nbest $beam_size --beam $beam_size"
	suffix=".$i"
	decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options" "$suffix"
    done
    tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
    word_level_outputs=$test_dir/${tgt_domain_wd}.outputs
    python -c "from itertools import chain; outputs=[[l.rstrip() for l in open('${word_level_outputs}.%03d' % i)] for i in range(${num_sample})]; outputs=list(zip(*outputs)); outputs=list(chain.from_iterable(outputs)); [print(l) for l in outputs]" > $word_level_outputs

    P=0.95
    beam_size=1
    num_sample=5
    test_dir="$model_dir/tests/latent${num_sample}-topP${P}"
    for i in $(seq 0 $(($num_sample-1))); do
	i=`printf %03d ${i}`
	decode_options="--lenpen 1.2 --seed $i --sampling \
	                --sampling-topp ${P} \
			--nbest $beam_size --beam $beam_size"
	suffix=".$i"
	decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options" "$suffix"
    done
    tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
    word_level_outputs=$test_dir/${tgt_domain_wd}.outputs
    python -c "from itertools import chain; outputs=[[l.rstrip() for l in open('${word_level_outputs}.%03d' % i)] for i in range(${num_sample})]; outputs=list(zip(*outputs)); outputs=list(chain.from_iterable(outputs)); [print(l) for l in outputs]" > $word_level_outputs


else # non-variational models
    beam_size=1
    nbest=1
    test_dir="$model_dir/tests/beam${beam_size}-nbest${nbest}"
    decode_options="--lenpen 1.2 --seed $random_seed \
		    --nbest $nbest --beam $beam_size"
    decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options"

    beam_size=5
    nbest=1
    test_dir="$model_dir/tests/beam${beam_size}-nbest${nbest}"
    decode_options="--lenpen 1.2 --seed $random_seed \
		    --nbest $nbest --beam $beam_size"
    decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options"


    beam_size=5
    nbest=5
    test_dir="$model_dir/tests/beam${beam_size}-nbest${nbest}"
    decode_options="--lenpen 1.2 --seed $random_seed \
		    --nbest $nbest --beam $beam_size"
    decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options"

    beam_size=5
    num_sample=5
    test_dir="$model_dir/tests/sample${num_sample}-pure"
    decode_options="--lenpen 1.2 --seed $random_seed --sampling \
		    --nbest $beam_size --beam $beam_size"
    decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options"

    K=20
    beam_size=5
    num_sample=5
    test_dir="$model_dir/tests/sample${num_sample}-topK${K}"
    decode_options="--lenpen 1.2 --seed $random_seed --sampling \
                    --sampling-topk ${K} \
		    --nbest $beam_size --beam $beam_size"
    decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options"

    K=40
    beam_size=5
    test_dir="$model_dir/tests/sample${num_sample}-topK${K}"
    decode_options="--lenpen 1.2 --seed $random_seed --sampling \
                    --sampling-topk ${K} \
		    --nbest $beam_size --beam $beam_size"
    decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options"

    K=640
    beam_size=5
    test_dir="$model_dir/tests/sample${num_sample}-topK${K}"
    decode_options="--lenpen 1.2 --seed $random_seed --sampling \
                    --sampling-topk ${K} \
		    --nbest $beam_size --beam $beam_size"
    decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options"

    beam_size=5
    test_dir="$model_dir/tests/sample${num_sample}-topP0.95"
    decode_options="--lenpen 1.2 --seed $random_seed --sampling \
                    --sampling-topp 0.95 \
		    --nbest $beam_size --beam $beam_size"
    decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options"

    beam_size=5
    test_dir="$model_dir/tests/sample${num_sample}-topP0.50"
    decode_options="--lenpen 1.2 --seed $random_seed --sampling \
                    --sampling-topp 0.50 \
		    --nbest $beam_size --beam $beam_size"
    decode "$test_dir" "$decode_options" "$data_options" "$ckpt_options"
fi
