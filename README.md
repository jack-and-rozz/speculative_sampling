# Speculative Sampling in Variational Autoencoders for Dialogue Response Generation

## Requirements
- Python 3.7.3 (other versions can work)
- Sentencepiece 0.1.83 (https://github.com/google/sentencepiece)


## Setup
```
pip install -r requirements.txt

if [ ! -e tools ]; then
   mkdir -p tools
fi
git clone https://github.com/jack-and-rozz/fairseq

# install libraries for fairseq
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ # Using this setting makes training faster but can cause errors depending on your environment.
cd ..

cd tools
git clone https://github.com/moses-smt/mosesdecoder.git 
cd ..
```


## Data preparation

refer to [scripts/dataset/twitterv3](https://github.com/jack-and-rozz/speculative_sampling/blob/master/scripts/dataset/twitterv3/README.md).

## Training
```
model_name=twitterv3ja_sp16000.baseline.tcvae.all # T-CVAE
task=dialog
./train.sh $model_name $task

#### Model candidates
- T-CVAE: twitterv3ja_sp16000.baseline.tcvae.all
- T-CVAE + Cycical Annealing: twitterv3ja_sp16000.baseline.tcvae.kla9387.cycle18775.all
- T-CVAE + BoW loss: twitterv3ja_sp16000.baseline.tcvae.bow1.all
- Transformer-based SPACEFUSION: twitterv3ja_sp16000.baseline.t-spacefusion.all
- T-CVAE + Monte Carlo sampling (k=5): twitterv3ja_sp16000.baseline.tcvae.ls5.mean.all
- T-CVAE + Speculative sampling (k=5): twitterv3ja_sp16000.baseline.tcvae.ls5.all
```


## Evaluation
```
mkdir exp_logs
task=dialog
domain=twitterv3ja
./generate_many.sh $domain $domain $task # Outputs of each model will be saved to "checkpoints/latest/$model_name/tests".
./summarize.sh $domain $domain $task > exp_logs/$domain.summary

```


## Citation
If you use this code for research, please cite the following paper.
```
@inproceedings{sato-etal-2021-speculative-sampling,
    title = "Speculative Sampling in Variational Autoencoders for Dialogue Response Generation",
    author = "Sato, Shoetsu  and
      Yoshinaga, Naoki  and
      Toyoda, Masashi  and
      Kitsuregawa, Masaru",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.407",
    doi = "10.18653/v1/2021.findings-emnlp.407",
    pages = "4739--4745",
}
```
