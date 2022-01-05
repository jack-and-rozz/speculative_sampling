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
num_latent=500
model_dir=$(get_model_dir $ckpt_root $mode)
analysis_output_dir=$model_dir/analyses

if [ ! -e $analysis_output_dir ]; then
    mkdir -p $analysis_output_dir
fi

data_dir=$(get_data_dir $mode $tgt_domain)
analysis_target_data=fairseq.analysis

./scripts/analysis/preprocess_for_analysis.sh $mode $task

case $mode in
    *)
	data_dir=$(get_data_dir $mode $tgt_domain)
	# data=$data_dir/fairseq.$size
	data=$data_dir/$analysis_target_data
	;;
esac

analysis_output_dir=$model_dir/analyses/$analysis_target_data.$num_latent
if [ ! -e $analysis_output_dir ]; then
    mkdir -p $analysis_output_dir
fi

data_options="$data $data_options"
ckpt_options="--path $model_dir/checkpoints/checkpoint_best.pt \
	      --results-path $model_dir/tests/$test_file "

if [ ! -e $model_dir/checkpoints/checkpoint_best.pt ]; then
    echo "$model_dir/checkpoints/checkpoint_best.pt was not found."
    exit 1
fi

if [ -e "$analysis_output_dir/0.prior.latent" ]; then
    echo "Outputs already exist in $analysis_output_dir"
    exit 1
else
    python fairseq/analyze_latent.py \
	   --seed $random_seed \
    	   --user-dir ${fairseq_user_dir} \
    	   --beam ${beam_size} \
    	   --lenpen 1.2 \
    	   --task ${fairseq_task} \
    	   --shard-id $shard_id \
    	   --num-shards $num_shards \
	   --skip-invalid-size-inputs-valid-test \
	   --source-lang $src_lang \
	   --target-lang $tgt_lang \
    	   $ckpt_options \
    	   $data_options \
	   --score-reference \
           --analysis-output-dir $analysis_output_dir \
           --num-latent-sampling-per-response $num_latent \
	   --num-batches-for-analysis 0 \
	   --max-tokens $max_tokens_per_batch
	   # --max-tokens 16000 
fi
