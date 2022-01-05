#!/bin/bash
echo "Running '$0 $1 $2'..."

usage() {
    echo "Usage:$0 mode task [train_steps] [update_freq]"
    exit 1
}
if [ $# -lt 2 ];then
    usage;
fi

mode=$1
task=$2
train_steps_specified=$3
_update_freq=$4
_max_tokens_per_batch=$5

if [ $task != dialog ]; then
    echo "DEBUG: set task=dialog."
    exit 1
fi

. ./const.sh $mode $task


#######################DEBUG#######################
num_encoder_layers=1
num_decoder_layers=1
encoder_ffn_dim=256
decoder_ffn_dim=256
log_interval=1
#######################DEBUG#######################



is_valid=$(validate_mode $mode $task)
if [ ! -z $is_valid ]; then
    echo $is_valid
    exit 1
fi

# default parameters.
num_workers=8
fairseq_task=dialogue # custom task.

max_tokens_per_batch=2000

train_steps=$train_steps_specified
if [ -z $train_steps ]; then
    train_steps=$train_steps_default
fi

# if [ ! -z $_update_freq ]; then
#     update_freq=$_update_freq
# fi
# if [ ! -z $_max_tokens_per_batch ]; then
#     max_tokens_per_batch=$_max_tokens_per_batch
# fi

if [ ! -z $_update_freq ] && [ ! -z $_max_tokens_per_batch ]; then
    update_freq=$_update_freq
    max_tokens_per_batch=$_max_tokens_per_batch
elif [ -z $_update_freq ] && [ -z $_max_tokens_per_batch ]; then
    devices=(${CUDA_VISIBLE_DEVICES//,/ })
    num_gpus=${#devices[@]}
    # Adjust update_freq and max_tokens in batch depending on the number of GPUs to make the size of data per update equal.
    case $num_gpus in 
	1)
	    update_freq=$((update_freq*4))
	    ;;
	2)
	    update_freq=$((update_freq*2))
	    ;;
	3)
	    max_tokens_per_batch=$((max_tokens_per_batch*4/3))
	    ;;
	4)
	    # default (defined in configs/common.sh).
	    ;;
	5)
	    max_tokens_per_batch=$((max_tokens_per_batch*4/5))
	    ;;
	6)
	    update_freq=$((update_freq/2))
	    max_tokens_per_batch=$((max_tokens_per_batch*4/3))
	    ;;
	7)
	    update_freq=$((update_freq/2))
	    max_tokens_per_batch=$((max_tokens_per_batch*8/7))
	    ;;
	8)
	    update_freq=$((update_freq/2))
	    ;;
	*)
	    echo "Error: invalid #GPUs: $num_gpus"
	    exit 1
	    ;;
    esac
else
    echo "Specify both update_freq and max_tokens_per_batch or keep them blank."
    exit 1
fi
max_tokens_per_batch=2000

model_dir=$(get_model_dir $ckpt_root $mode)
architecture=transformer_finetuning
criterion=label_smoothed_cross_entropy

size=$(parse_size $mode)
src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
src_lang=$(get_src_lang $tgt_domain $task)
tgt_lang=$(get_tgt_lang $tgt_domain $task)
emb_type=$(parse_emb_type $mode)
multidomain_type=$(parse_multidomain_type $mode)
fixed=$(parse_fixed $mode)
src_vocab_size=$(parse_src_vocab_size $mode)
tgt_vocab_size=$(parse_tgt_vocab_size $mode)

echo "task="$task
echo "size="$size
echo "src_domain="$src_domain
echo "tgt_domain="$tgt_domain
echo "src_lang="$src_lang
echo "tgt_lang="$tgt_lang
echo "emb_type="$emb_type
echo "multidomain_type="$multidomain_type
echo "fixed="$fixed

case $mode in
    *.${baseline_suffix}.*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size

	train_options="$train_options --max-update $train_steps"
	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}
		     "

	enc_emb_path=$data_dir/word2vec.${src_lang}.${emb_size}d
	dec_emb_path=$data_dir/word2vec.${tgt_lang}.${emb_size}d

	emb_options="$emb_options --encoder-embed-path $enc_emb_path \
		      --share-decoder-input-output-embed \
		      --share-all-embeddings
 		      "
	;;
    * ) echo "invalid mode: $mode"
        exit 1 ;;
esac

# Model type
if [[ $mode =~ .tcvae ]]; then
    architecture=t_cvae
    criterion=cvae_loss
    train_options="$train_options"
elif [[ $mode =~ .t-spacefusion ]]; then
    architecture=tcvae_spacefusion
    criterion=tcvae_spacefusion_loss
    train_options="$train_options"
else
    architecture=transformer_finetuning
    criterion=label_smoothed_cross_entropy_sent
fi

# KL annealing
if [[ ${mode} =~ .kla([0-9]+) ]]; then
    kl_annealing_steps=${BASH_REMATCH[1]}
    train_options="$train_options --kl-annealing-steps $kl_annealing_steps"
else
    kl_annealing_steps=0
fi

if [[ ${mode} =~ .cycle([0-9]+) ]]; then
    kl_annealing_steps_per_cycle=${BASH_REMATCH[1]}
    train_options="$train_options --kl-annealing-steps-per-cycle $kl_annealing_steps_per_cycle"
else
    kl_annealing_steps_per_cycle=0
fi

# Number of sampled latent variables
if [[ $mode =~ .ls([0-9]+) ]]; then
    num_latent_sampling=${BASH_REMATCH[1]}
    train_options="$train_options --num-latent-sampling $num_latent_sampling"

    if [[ $mode =~ .mean ]]; then
	train_options="$train_options --enable-mean-sampled-latent"
    fi
fi

# Bag-of-Words loss
if [[ $mode =~ \.bow([0-9\.]+)\. ]]; then
    bow_loss_weight=${BASH_REMATCH[1]}
    train_options="$train_options --bow-loss-weight $bow_loss_weight"
fi

# if [[ $mode =~ .leaky_relu ]]; then
#     activation=leaky_relu
# else
#     activation=relu
# fi

# if set, the encoder and decoder have independent parameters each other.
if [[ $mode =~ .independent_dec ]] && [ $architecture == t_cvae ]; then
    train_options="$train_options --disable-sharing-decoder"
fi


# if set, the model has independent parameters for inducing prior and posterior distributions by attention. Otherwise they are shared.
if [[ $mode =~ .independent_attn ]]; then
    train_options="$train_options --disable-sharing-prior-post-attn"
fi

if [[ $mode =~ .normalize ]]; then
    train_options="$train_options --encoder-normalize-before --decoder-normalize-before"
fi

if [[ $mode =~ .stop_postKL ]]; then
    train_options="$train_options --enable-stopping-klgrad-to-posterior"
fi


if [[ $tgt_domain =~ _${sp_suffix} ]]; then
    ./setup_sentencepiece.sh $mode $task
fi

if [ ! -e $enc_emb_path ]; then
    ./train_cbow.sh $mode $task
fi

./preprocess.sh $mode $task

if [ ! -e $model_dir ];then
    mkdir -p $model_dir
    mkdir -p $model_dir/tests
    mkdir -p $model_dir/checkpoints
    mkdir -p $model_dir/tensorboard
    mkdir -p $model_dir/embeddings

    mkdir -p $model_dir/subword
fi
if [ ! -e $model_dir/subword ];then
    mkdir -p $model_dir/subword
fi


# Link to the subword tokenization model used in the NMT model.
suffixes=(model vocab)
if [ -z $src_lang_spm_dir ]; then
    src_lang_spm_dir=$data_dir
fi
if [ -z $tgt_lang_spm_dir ]; then
    tgt_lang_spm_dir=$data_dir
fi
for suffix in ${suffixes[@]}; do
    if [[ $mode =~ ${sp_suffix} ]]; then
	if [ ! -e $model_dir/subword/spm.$src_lang.$suffix ]; then
	    ln -sf $(pwd)/$src_lang_spm_dir/spm.$src_lang.$suffix \
	       $model_dir/subword/spm.$src_lang.$suffix
	fi
	if [ ! -e $model_dir/subword/spm.$tgt_lang.$suffix ]; then
	    ln -sf $(pwd)/$tgt_lang_spm_dir/spm.$tgt_lang.$suffix \
	       $model_dir/subword/spm.$tgt_lang.$suffix
	fi
    fi
done



if [ $criterion == cvae_loss ]; then
    train_options="$train_options --kl-annealing-steps=$kl_annealing_steps"
fi

echo "Start training $mode..."
# Start training.
python fairseq/train.py \
       --user-dir ${fairseq_user_dir} \
       --ddp-backend=no_c10d \
       --seed $random_seed \
       --log-interval $log_interval --log-format simple \
       --save-dir $model_dir/checkpoints \
       --save-interval-updates $save_interval_updates \
       --save-interval $save_interval_epochs \
       --tensorboard-logdir $model_dir/tensorboard \
       --skip-invalid-size-inputs-valid-test \
       --keep-last-epochs 100 \
       --arch $architecture \
       --criterion $criterion  \
       $data \
       $task_options \
       $emb_options \
       $train_options \
       --max-epoch $max_epoch \
       --max-tokens $max_tokens_per_batch \
       --update-freq $update_freq \
       --num-workers $num_workers \
       --sentence-avg \
       --optimizer adam --adam-betas '(0.9, 0.98)' \
       --lr 1e-03 --min-lr 1e-09   \
       --lr-scheduler inverse_sqrt \
       --weight-decay $weight_decay \
       --warmup-init-lr 1e-07  \
       --warmup-updates 4000   \
       --label-smoothing $label_smoothing_factor   \
       --activation-fn $activation \
       --dropout $dropout_rate \
       --attention-dropout $attention_dropout \
       --activation-dropout $activation_dropout \
       --clip-norm $clip_norm \
       --encoder-layers $num_encoder_layers \
       --decoder-layers $num_decoder_layers \
       --encoder-attention-heads $num_encoder_attention_heads \
       --decoder-attention-heads $num_decoder_attention_heads \
       --encoder-ffn-embed-dim $encoder_ffn_dim \
       --decoder-ffn-embed-dim $decoder_ffn_dim \
       --encoder-embed-dim $emb_size \
       --decoder-embed-dim $emb_size 

       # --decoder-embed-dim $emb_size \
       # >> $model_dir/train.log



