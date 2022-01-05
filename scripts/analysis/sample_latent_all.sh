#!/bin/bash
echo "Running '$0 $1 $2'..."

usage() {
    echo "Usage:$0 tgt_domain task"
    exit 1
}
if [ $# -lt 2 ];then
    usage;
fi

. ./const.sh 

tgt_domain=$1
task=$2

models=$(ls $ckpt_root | grep $tgt_domain)
for model_name in ${models[@]}; do
    echo "<Model: $model_name>"
    ./scripts/analysis/sample_latent.sh $model_name $task
    echo "Applying t-SNE..."
    python scripts/analysis/tsne2latent.py $ckpt_root/$model_name/analyses/fairseq.analysis.500 --overwrite 2>/dev/null &
done
