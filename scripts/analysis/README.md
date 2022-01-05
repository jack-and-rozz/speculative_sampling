## Overview

# How to use
### Sample N latent variables for analysis
```
model_name=twitterv3ja_sp16000.baseline.tcvae.all
task=dialog
./scripts/analysis/sample_latent.sh $model_name $task 
```

### Reduce the dimension of latent variables
```
latent_dir=checkpoints/latest/twitterv3ja_sp16000.baseline.tcvae.all/analyses/fairseq.analysis
python scripts/analysis/tsne2latent.py $latent_dir
```



# Overview of the scripts 
- sample_latent.sh 
Sample N latent variables for each example in testing (or analysis) dataset

- sample_latent_all.sh $domain_name $task 
Run `./sample_latent.sh` for all models whose name includes the first argument, ${domain_name}.

- preprocess_for_analysis.sh $domain_name $task 
This script is automatically run from `sample_latent.sh`. Convert data for analysis into fairseq binarized format.



