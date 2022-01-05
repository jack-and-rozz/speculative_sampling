######################################
###       Util bash functions
######################################

validate_mode(){
    mode=$1
    task=$2

    if [[ ${mode} =~ ([0-9a-zA-Z_\-]+)${direction_tok}([0-9a-zA-Z_\-]+)\. ]]; then
	src_domain=${BASH_REMATCH[1]}
	tgt_domain=${BASH_REMATCH[2]}
    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+)${direction_tok}([0-9a-zA-Z_\-]+)\. ]]; then
	src_domain=${BASH_REMATCH[1]}
	tgt_domain=${BASH_REMATCH[2]}
    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+)\.$baseline_suffix ]]; then
	src_domain=${BASH_REMATCH[1]}
    else
    	echo "Invalid mode: $mode"
    fi

}

parse_src_domain(){
    mode=$1
    if [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)${direction_tok}([0-9a-zA-Z_\-]+)\. ]]; then
	src_domain=${BASH_REMATCH[1]}
	tgt_domain=${BASH_REMATCH[2]}
    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)${direction_tok}([0-9a-zA-Z_\-]+)\. ]]; then
	src_domain=${BASH_REMATCH[1]}
	tgt_domain=${BASH_REMATCH[2]}
    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)\.$baseline_suffix ]]; then
	src_domain=${BASH_REMATCH[1]}
    fi

    if [[ $src_domain =~ (.+_${sp_suffix}) ]]; then
	src_domain=${BASH_REMATCH[1]}
    fi
    echo $src_domain
}

parse_tgt_domain(){
    mode=$1
    if [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)${direction_tok}([0-9a-zA-Z_\-]+)\. ]]; then
	src_domain=${BASH_REMATCH[1]}
	tgt_domain=${BASH_REMATCH[2]}
    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)${direction_tok}([0-9a-zA-Z_\-]+)\. ]]; then
	src_domain=${BASH_REMATCH[1]}
	tgt_domain=${BASH_REMATCH[2]}
    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)\.$baseline_suffix.v_(.+?)\. ]]; then
	tgt_domain=${BASH_REMATCH[2]}

    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)\.$baseline_suffix ]]; then
	tgt_domain=${BASH_REMATCH[1]}
    fi

    if [[ $tgt_domain =~ (.+_${sp_suffix}) ]]; then
	tgt_domain=${BASH_REMATCH[1]}
    fi
    echo $tgt_domain

}

remove_tok_suffix(){
    domain=$1
    if [[ $domain =~ _$sp_suffix ]]; then
	l=$(expr $(expr length $sp_suffix) + 1) # remove '_sp', '_uni' or '_bpe'.
	domain=${domain:0:-$l}
    fi
    echo $domain
}

parse_emb_type(){
    mode=$1
    if [[ ${mode} =~ \.([llm|linear].+?)\.(nn[0-9]+)\. ]]; then
	emb_type=${BASH_REMATCH[1]}
    elif [[ ${mode} =~ \.([llm|linear].+?)\.(.+) ]]; then
	emb_type=${BASH_REMATCH[1]}
    elif [[ ${mode} =~ \.finetune\.(.+?)\.(.+) ]]; then
	emb_type=${BASH_REMATCH[1]}
    fi

    # if [[ ${mode} =~ \.([llm|linear].+?)\..+ ]]; then
    # 	emb_type=${BASH_REMATCH[1]}
    # 	echo ddd
    # elif [[ ${mode} =~ \.([llm|linear].+?)\.all ]]; then
    # 	emb_type=${BASH_REMATCH[1]}
    # 	echo eee
    # elif [[ ${mode} =~ \.finetune\.(.+?)\.[0-9]+k ]]; then
    # 	emb_type=${BASH_REMATCH[1]}
    # 	echo aaa
    # elif [[ ${mode} =~ \.finetune\.(.+?)\.all ]]; then
    # 	emb_type=${BASH_REMATCH[1]}
    # 	echo bbb
    # elif [[ ${mode} =~ \.finetune\.(.+?)\. ]]; then
    # 	emb_type=${BASH_REMATCH[1]}
    # 	echo ccc
    # fi
    echo $emb_type
}
parse_multidomain_type(){
    mode=$1
    if [[ ${mode} =~ \.multidomain\.(.+?)\. ]]; then
	multidomain_type=${BASH_REMATCH[1]}
    fi
    echo $multidomain_type
}

parse_size(){
    mode=$1
    if [[ ${mode} =~ \.([0-9]+k).* ]]; then
	size=${BASH_REMATCH[1]}
    elif [[ ${mode} =~ .all ]]; then
	size=all
    fi
    if [ -z $size ]; then
	echo all
    else
	echo $size
    fi
}

parse_llm_nn(){
    mode=$1
    if [[ ${mode} =~ \.llm.+\.nn([0-9]+)\.* ]]; then
	num_nn=${BASH_REMATCH[1]}
    fi
    echo $num_nn
}


parse_src_vocab_size(){
    mode=$1
    if [[ ${mode} =~ [${word_suffix}|${sp_suffix}|${bpe_suffix}]([0-9]+)${direction_tok} ]]; then
	vocab_size=${BASH_REMATCH[1]}
    elif [[ ${mode} =~ [${word_suffix}|${sp_suffix}|${bpe_suffix}]([0-9]+)\. ]]; then
	vocab_size=${BASH_REMATCH[1]}
    fi
    echo $vocab_size
}

parse_tgt_vocab_size(){
    mode=$1
    if [[ ${mode} =~ ${direction_tok}(.+)[${word_suffix}|${sp_suffix}|${bpe_suffix}]([0-9]+)\. ]]; then
	vocab_size=${BASH_REMATCH[2]}
    elif [[ ${mode} =~ [${word_suffix}|${sp_suffix}|${bpe_suffix}]([0-9]+)\. ]]; then
	vocab_size=${BASH_REMATCH[1]}
    fi
    echo $vocab_size
}

parse_fixed(){
    mode=$1
    if [[ ${mode} =~ \.($fixed_emb_suffix) ]]; then
	fixed=${BASH_REMATCH[1]}
    fi
    echo $fixed
}

get_hyperparameter_by_domain(){
    domain=$1
    parameter_name=$2
    if [[  $domain =~ _${sp_suffix} ]]; then
	domain_wd=$(remove_tok_suffix $domain)
	echo $(eval echo '$'$domain_wd'_'$parameter_name)
    else
	echo $(eval echo '$'$domain'_'$parameter_name)
    fi

}

get_src_lang(){
    domain=$1
    task=$2
    if [ 1 = 1 ]; then # deprecated if-condition
	# lang=echo $(eval echo '$'$domain'_src_lang')
	# if [ ! -z $lang ]; then
	#     domain_wd=$(remove_tok_suffix $domain)
	#     lang=$(eval echo '$'$domain_wd'_src_lang')
	# fi
	if [[  $domain =~ _${sp_suffix} ]]; then
	    domain_wd=$(remove_tok_suffix $domain)
	    echo $(eval echo '$'$domain_wd'_src_lang')
	else
	    echo $(eval echo '$'$domain'_src_lang')
	fi
	echo $lang
    fi
}
get_tgt_lang(){
    domain=$1
    task=$2
    if [ 1 = 1 ]; then # deprecated if-condition
	if [[  $domain =~ _${sp_suffix} ]]; then
	    domain_wd=$(remove_tok_suffix $domain)
	    echo $(eval echo '$'$domain_wd'_tgt_lang')
	else
	    echo $(eval echo '$'$domain'_tgt_lang')
	fi
    fi
}

get_data_dir(){
    mode=$1
    domain=$2

    if [[ $domain =~ _${sp_suffix} ]]; then
	if [[ $mode =~ $domain([0-9]+) ]];then
	    vocab_size=${BASH_REMATCH[1]}
	fi
	domain_wd=$(remove_tok_suffix $domain)
	data_dir=$(eval echo '$'${domain_wd}'_data_dir')
	if [ -z $data_dir ]; then
	    exit 1
	fi
	data_dir=$data_dir.${sp_suffix}${vocab_size}
    else
	data_dir=$(eval echo '$'${domain}'_data_dir')
	if [ -z $data_dir ]; then
	    exit 1
	fi
    fi
    echo $data_dir
}

get_model_dir(){
    ckpt_root=$1
    mode=$2
    src_domain=$(parse_src_domain $mode)
    src_vocab_size=$(parse_src_vocab_size $mode)
    fixed=$(parse_fixed $mode)

    if [[ $mode =~ $sp_suffix ]]; then
	src_domain=$src_domain$src_vocab_size
    fi

    case $mode in
	*${direction_tok}*.noadapt*)
	    # Evaluate the source domain model in the target domain.
	    model_dir=$ckpt_root/$src_domain.${baseline_suffix}.all
	    if [ ! -z $fixed ]; then
		model_dir=$model_dir.fixed
	    fi
	    ;;

	*${direction_tok}*.backtranslation_aug*)
	    # Share the model trained in the same source domain.
	    model_dir=$ckpt_root/$src_domain.backtranslation_aug
	    if [ ! -z $fixed ]; then
		model_dir=$model_dir.fixed
	    fi
	    ;;
	*)
	    model_dir=$ckpt_root/$mode
	;;
    esac
    echo $model_dir
}


# Get a path to the dataset constructed from a pair of domains.
get_multidomain_data_dir(){
    mode=$1
    src_domain=$2
    tgt_domain=$3
    mdtype=$4
    if [[ $src_domain =~ _${sp_suffix} ]] && [[ $tgt_domain =~ _${sp_suffix} ]]; then
	src_vocab_size=$(parse_src_vocab_size $mode)
	tgt_vocab_size=$(parse_tgt_vocab_size $mode)

	# The vocabulary size of a backward model is defined in source domain.
	if [[ $mdtype =~ backtranslation_aug ]]; then
	    vocab_size=$src_vocab_size
	elif [[ $mdtype =~ backtranslation_ft ]] || [[ $mdtype =~ backtranslation_va ]]; then
	    vocab_size=$tgt_vocab_size
	else
	    vocab_size=$tgt_vocab_size
	fi
	src_domain_wd=$(remove_tok_suffix $src_domain)
	tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
	data_dir=$(eval echo '$'${src_domain_wd}'2'${tgt_domain_wd}'_data_dir')/$mdtype
	data_dir=$data_dir.${sp_suffix}${vocab_size}
    elif [[ ! $src_domain =~ _${sp_suffix} ]] && [[ ! $tgt_domain =~ _${sp_suffix} ]]; then
	data_dir=$(eval echo '$'$src_domain'2'$tgt_domain'_data_dir')/$mdtype
    fi
    echo $data_dir
}


get_backtranslation_type(){
    mode=$1
    if [[ $mode =~ .(backtranslation.+)\.(.+) ]]; then
	bt_type=${BASH_REMATCH[1]}
    elif [[ $mode =~ .(backtranslation.+)\.? ]]; then
	bt_type=${BASH_REMATCH[1]}
    fi
    echo $bt_type
}


get_domain_token(){
    domain=$1
    if [[ $domain =~ _${sp_suffix} ]]; then
	domain_wd=$(remove_tok_suffix $domain)
	domain_token=$(eval echo '$'$domain_wd'_domain_token')
	if [ -z $domain_token ]; then
	    echo '$'$(remove_tok_suffix $domain)'_domain_token' is not defined!
	    exit 1
	fi

    else
	domain_token=$(eval echo '$'$domain'_domain_token')
	if [ -z $domain_token ]; then
	    echo '$'${domain}'_domain_token' is not defined!
	    exit 1
	fi
    fi
    echo $domain_token
}

