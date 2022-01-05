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

src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 

suffix=".$size"
case $mode in
    ########################################
    ###          In-domain
    ########################################
    *.${baseline_suffix}.v_${tgt_domain}.*)
	src_data_dir=$(get_data_dir $mode $src_domain)
	tgt_data_dir=$(get_data_dir $mode $tgt_domain)
	data_dir=$src_data_dir
	data=$src_data_dir/fairseq.v_$src_domain.$size
	train_options="--max-update $train_steps"
	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}
		     "
	train_data=$data_dir/train
	if [ $size != all ]; then
	    train_data=$train_data.$size
	fi
	dev_data=$src_data_dir/dev
	test_data=$src_data_dir/test

	train_data=$train_data.$tgt_domain
	dev_data=$dev_data.$tgt_domain
	test_data=$test_data.$tgt_domain
	suffix=".v_${tgt_domain}.$size"

	if [[ $src_domain =~ _${sp_suffix} ]]; then
	    if [ ! -e $train_data ]; then
		src_domain_wd=$(remove_tok_suffix $src_domain)
		src_data_dir_wd=$(eval echo '$'$src_domain_wd'_data_dir')
		langs=($src_lang $tgt_lang)
		if [ ! -z $size ] && [ $size != all ]; then
		    set_types=(train.$size dev test)
		else
		    set_types=(train dev test)
		fi


		# Encode the word-level source domain's corpora by the target domain's spm.
		for lang in ${langs[@]}; do
		    for set_type in ${set_types[@]}; do
			if [ ! -e $src_data_dir/$set_type.$tgt_domain.$lang ]; then
			    echo "Encoding $src_data_dir_wd/$set_type.$lang to $src_data_dir/$set_type.$tgt_domain.$lang."
			    spm_encode \
				--model $tgt_data_dir/spm.$lang.model \
				--output $src_data_dir/$set_type.$tgt_domain.$lang \
				< $src_data_dir_wd/$set_type.$lang & 
			fi
		    done
		done
		wait 
	    fi
	fi


	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
                 --nwordssrc $src_vocab_size \
                 --nwordstgt $tgt_vocab_size \
		 --trainpref $train_data \
		 --validpref $dev_data \
		 --testpref $test_data
		 "
	src_dict=$tgt_data_dir/dict.${src_lang}.txt
	tgt_dict=$tgt_data_dir/dict.${tgt_lang}.txt
	;;

    *.${baseline_suffix}*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	if [ ! -z $size ] && [ $size != all ] && [ ! -e $data_dir/train.$size.${src_lang} ]; then
	    python scripts/random_pickup.py \
		   $data_dir/train.${src_lang} $data_dir/train.${tgt_lang} \
		   $src_lang $tgt_lang \
		   $size \
		   --seed $random_seed

	fi

	if [ ! -z $size ] && [ $size != all ]; then
	    train_data=$data_dir/train.$size
	else
	    train_data=$data_dir/train
	fi
	dev_data=$data_dir/dev

	if [ -e $data_dir/test2.$src_lang ]; then
	    test_data=$data_dir/test,$data_dir/test2
	else
	    test_data=$data_dir/test
	fi
	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
                 --nwordssrc $src_vocab_size \
                 --nwordstgt $tgt_vocab_size \
		 --trainpref $train_data     \
		 --validpref $dev_data   \
		 --testpref $test_data 
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;

    ########################################
    ###         Out-domain
    ########################################

    # process tgt dataset with src vocabulary.
    *${direction_tok}*.noadapt*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	src_data_dir=$(get_data_dir $mode $src_domain)

	train_data=$data_dir/train
	if [ $size != all ]; then
	    train_data=$train_data.$size
	fi
	dev_data=$data_dir/dev
	test_data=$data_dir/test
	test_data2=$data_dir/test2
	suffix=".v_${src_domain}.$size"

	# Re-encode the dataset in the target domain by the subword tokenization trained in the source domain.
	if [[ $src_domain =~ _${sp_suffix} ]]; then
	    if [ ! -e $train_data ]; then
		tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
		tgt_data_dir_wd=$(eval echo '$'$tgt_domain_wd'_data_dir')
		langs=($src_lang $tgt_lang)
		if [ ! -z $size ] && [ $size != all ]; then
		    set_types=(train.$size dev test)
		else
		    set_types=(train dev test)
		fi
		if [ -e $tgt_data_dir_wd/test2.$src_lang ]; then
		    set_types+=(test2)
		fi
		if [ $size != all ] && [ ! -e $tgt_data_dir_wd/${set_types[0]}.$size.${langs[0]} ]; then
		    python scripts/random_pickup.py \
			   $data_dir/train.${src_lang} \
			   $data_dir/train.${tgt_lang} \
			   ${src_lang} ${tgt_lang} \
			   $size \
			   --seed $random_seed
		fi

		for lang in ${langs[@]}; do
		    for set_type in ${set_types[@]}; do
			if [ ! -e $data_dir/$set_type.$src_domain.$lang ]; then
			    echo "Encoding $tgt_data_dir_wd/$set_type.$lang to $data_dir/$set_type.$src_domain.$lang."
			    spm_encode \
				--model $src_data_dir/spm.$lang.model \
				--output $data_dir/$set_type.$src_domain.$lang \
				< $tgt_data_dir_wd/$set_type.$lang & 
			fi
		    done
		done
		wait 
	    fi
	    train_data=$train_data.$src_domain
	    dev_data=$dev_data.$src_domain
	    test_data=$test_data.$src_domain
	    test_data2=$test_data2.$src_domain
	fi
	if [ -e $test_data2 ]; then
	    test_data=$test_data,$test_data2
	else
	    test_data=$test_data
	fi

	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
                 --nwordssrc $src_vocab_size \
                 --nwordstgt $tgt_vocab_size \
		 --trainpref $train_data \
		 --validpref $dev_data \
		 --testpref $test_data
		 "
	src_dict=$src_data_dir/dict.${src_lang}.txt
	tgt_dict=$src_data_dir/dict.${tgt_lang}.txt
	;;

    # ########################################
    # ###       Back-translation
    # ########################################

    # Preprocess the dataset to train a model for data augmentation.
    *${direction_tok}*.backtranslation_aug)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain \
					    backtranslation_aug)
	train_data=$data_dir/train
	if [ $size != all ]; then
	    train_data=$train_data.$size
	fi
	dev_data=$data_dir/dev
	test_data=$data_dir/test

    	# Swap src-lang and tgt-lang for bt.
    	options="--source-lang ${tgt_lang} \
	         --target-lang ${src_lang} \
                 --nwordssrc $tgt_vocab_size \
                 --nwordstgt $src_vocab_size \
    		 --trainpref $train_data  \
    		 --validpref $dev_data    \
    		 --testpref $test_data"
    	src_dict=$data_dir/dict.${tgt_lang}.txt
    	tgt_dict=$data_dir/dict.${src_lang}.txt
	;;

    *${direction_tok}*.backtranslation_ft.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain \
		   backtranslation_ft)
	train_data=$data_dir/train
	if [ $size != all ]; then
	    train_data=$train_data.$size
	fi
	dev_data=$data_dir/dev
	test_data=$data_dir/test

    	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
                 --nwordssrc $src_vocab_size \
                 --nwordstgt $tgt_vocab_size \
    		 --trainpref $train_data \
    		 --validpref $dev_data \
    		 --testpref $test_data"
    	src_dict=$data_dir/dict.${src_lang}.txt
    	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;
    *${direction_tok}*.backtranslation_va.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain \
		   backtranslation_va)

	train_data=$data_dir/train
	if [ $size != all ]; then
	    train_data=$train_data.$size
	fi
	dev_data=$data_dir/dev
	test_data=$data_dir/test

    	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
                 --nwordssrc $src_vocab_size \
                 --nwordstgt $tgt_vocab_size \
    		 --trainpref $train_data \
    		 --validpref $dev_data \
    		 --testpref $test_data"
    	src_dict=$data_dir/dict.${src_lang}.txt
    	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;

    *${direction_tok}*.backtranslation_va_enc.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain \
		   backtranslation_va)
	src_data_dir=$(get_data_dir $mode $src_domain)

	exit 1
	# train_data=$data_dir/train.$src_domain
	# if [ $size != all ]; then
	#     train_data=$train_data.$size
	# fi
	# dev_data=$data_dir/dev.$src_domain
	# test_data=$data_dir/test.$src_domain

	
    	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
                 --nwordssrc $src_vocab_size \
                 --nwordstgt $tgt_vocab_size \
    		 --trainpref $train_data \
    		 --validpref $dev_data \
    		 --testpref $test_data"
	# Use the source domain's vocabulary in the decoder.
    	src_dict=$data_dir/dict.${src_lang}.txt
    	tgt_dict=$src_data_dir/dict.${tgt_lang}.txt
	;;


    ########################################
    ###     Multidomain-learning
    ########################################

    # Train with all of JESC dataset + part of ASPEC dataset.
    *${direction_tok}*.multidomain.domainweighting.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainweighting)
	if [ $size == all ]; then
	    train_data=$data_dir/train
	else
	    train_data=$data_dir/train.$size
	fi

	tgt_domain=$(remove_tok_suffix $tgt_domain)

	dev_data=$data_dir/dev.${tgt_domain}

	if [ -e $data_dir/test2.${tgt_domain}.$src_lang ]; then
	    test_data=$data_dir/test.${tgt_domain},$data_dir/test2.${tgt_domain}
	else
	    test_data=$data_dir/test.${tgt_domain}
	fi
	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
                 --nwordssrc $src_vocab_size \
                 --nwordstgt $tgt_vocab_size \
		 --trainpref $train_data \
		 --validpref $dev_data \
		 --testpref $test_data \
		 --extra-features domain
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;

    *${direction_tok}*.multidomain.domainmixing.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainmixing)
	if [ $size == all ]; then
	    train_data=$data_dir/train
	else
	    train_data=$data_dir/train.$size
	fi

	tgt_domain=$(remove_tok_suffix $tgt_domain)

	dev_data=$data_dir/dev.${tgt_domain}

	if [ -e $data_dir/test2.${tgt_domain}.$src_lang ]; then
	    test_data=$data_dir/test.${tgt_domain},$data_dir/test2.${tgt_domain}
	else
	    test_data=$data_dir/test.${tgt_domain}
	fi

	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
                 --nwordssrc $src_vocab_size \
                 --nwordstgt $tgt_vocab_size \
		 --trainpref $train_data \
		 --validpref $dev_data \
		 --testpref $test_data
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;

    * ) echo "invalid mode: $mode"
        exit 1
	;;
esac

destdir=$data_dir/fairseq$suffix

if [ $task != translation ]; then
    options="$options --srcdict $src_dict  --joined-dictionary"
else
    options="$options --srcdict $src_dict --tgtdict $tgt_dict"
fi

if [ ! -e $destdir/test.$src_lang-$tgt_lang.$src_lang.bin ] || [ ! -n "$(ls $destdir)" ]; then
    if [ $size == all ] && [ -e $data_dir/train.$src_lang ] &&  [ ! -e $data_dir/train.$src_lang.all ]; then
	ln -sf train.$src_lang $data_dir/train.all.$src_lang
	ln -sf train.$tgt_lang $data_dir/train.all.$tgt_lang
    fi

    if [ $size == all ] && [ -e $data_dir/fairseq ] && [ -z $suffix ]; then
	ln -sf fairseq $data_dir/fairseq.all
    else
	echo "Creating binary files with fairseq format to '$destdir'..."

	mkdir -p $destdir
	python fairseq/preprocess.py \
	       --destdir $destdir \
	       $options \
	       --workers 16
    fi

fi



