#!/bin/bash

# ckpt_root=checkpoints/latest
ckpt_root=checkpoints/latest
random_seed=0

# Tokenization and Truecasing by Moses Toolkit.
moses_data_dir=dataset/moses
moses_script_path=tools/mosesdecoder/scripts
tokenizer_path=$moses_script_path/tokenizer/tokenizer.perl
truecaser_script_path=$moses_script_path/recaser/truecase.perl
train_truecaser_script_path=$moses_script_path/recaser/train-truecaser.perl
truecaser_model_path=$moses_data_dir/truecase-model
corpus_bleu_path=scripts/multi-bleu.perl
sentence_bleu_path=scripts/sentence-bleu


# Options refered from *.sh
direction_tok=@
baseline_suffix=baseline
finetune_suffix=finetune
multidomain_suffix=multidomain
fixed_emb_suffix=fixed


# fairseq options.
fairseq_user_dir=fairseq/extensions

# Hyperparameters of preprocessing, including training sentencepiece models, CBoW vectors.
n_vocab_default=16000
llm_nn_default=10
bpe_suffix=bpe
unigram_suffix=uni
unk_surface='<unk>'
w2v_mincount=5
w2v_samplesize=1000000
emb_size=512
n_sentences_for_training_sp=1000000


# Hyperparameters of model's architecture.
encoder_ffn_dim=2048
decoder_ffn_dim=2048
num_encoder_layers=6
num_decoder_layers=6
num_encoder_attention_heads=8
num_decoder_attention_heads=8
max_epoch=1000
# max_tokens_per_batch=4096 # The actual batch size is max_tokens_per_batch * update_freq * num_gpus
max_tokens_per_batch=3360 # The actual batch size is max_tokens_per_batch * update_freq * num_gpus



# Training
save_interval_updates=0   # If zero, save trained model only at the end of each epoch.
save_interval_epochs=1

update_freq=2
beam_size=5
label_smoothing_factor=0.1
clip_norm=25
activation=leaky_relu
dropout_rate=0.1
activation_dropout=0.1
attention_dropout=0.1
weight_decay=0.0001 
log_interval=50
train_steps_default=120000

# Data options
finetune_sizes=(1k 10k 100k 1000k all)
emb_types=(idt llm)
multidomain_types=(domainmixing domainweighting)
