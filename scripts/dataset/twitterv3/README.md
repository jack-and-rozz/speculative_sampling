# Reproduction
- [This storage](https://drive.google.com/drive/folders/1cosefd7ZjEqWFh-vOqyYuc13hryaBdDv?usp=sharing) contains lists of message IDs used for creating dialogues in the train/dev/test sets used for our experiments ([Twitter terms of service](https://developer.twitter.com/en/developer-terms/more-on-restricted-use-cases) allow us to share tweet IDs only).
- You need to crawl messages in the lists and make dialogues separated into `*.src (contexts)` and `*.tgt (responses)` files.
- `const.sh` specifies the directory where the dialogues should be stored in (by default, `dataset/twitter-v3/en/processed.1turn` for En dialogue data).
- Subword tokenization for each set is automatically done by executing `train.sh`.
- If you need to use your own dataset, define the directory, training data for sentencepiece, and suffixes for context/response files in `const.sh`.

```bash
# An example of dialogues in a fairseq format...
$ head dataset/twitter-v3/en/processed.1turn.sp16000/train.* -n2
==> dataset/twitter-v3/en/processed.1turn.sp16000/train.src <==
▁It ' s ▁gonna ▁be ▁a ▁JE P IC ▁Year ▁2017
▁na ▁na ▁na aaaa . ▁After ▁tonight ▁ima ▁get ▁on ▁my ▁salad ▁shit t t .

==> dataset/twitter-v3/en/processed.1turn.sp16000/train.tgt <==
▁happy ▁new ▁year ▁my ▁love s !!!!
▁i ' m ▁on ▁a ▁trip ▁but ▁when ▁i ▁get ▁back ▁...
```




# (FYI) our procedures to make the twitter dataset
### 0. crawl tweets and make lists of daily tweets in 'original.tweets'. 
The format of the message lists is as follows. Each column is separated by '\t'.
```
tweet-type(T or M), unixtime, tweet-id, mention-target-id, user-id, user-name, user-screen, text,
```

```
# 2017-01-01.en.sort
T       1546300800      000000000000     -       100000000      username1      screenname1      a tweet....
M       1546300800      000000000001     000000000000     100000001       username2      screenname2        @username1 a mention...
```

### 1. create 'original.dialogs' from 'original.tweets'
```
./scripts/dataset/twitterv3/extract_dialogs.sh
```

### 2. create 'dataset.ja.1turn' from 'original.dialogs'
```
python scripts/dataset/twitterv3/create_dataset.py \
       ja \
       --train-dev-years 2017 2018 \
       --test-years 2019 \
       --source-dir original.dialogs \
       --target-dir-basename dataset \
       --num-turns 1
```


### 3. (if necessary) tokenize messages in 'dataset.ja.1turn' for the following filtering step.
```
#### Ja data
python scripts/dataset/twitterv3/tokenize.py \
       dataset.ja.1turn \
       ja \
       mecab \
       --target-suffixes dialogs src tgt mulres
```


### 4. (if necessary) filter conversations by their length, whether they were made by bots, etc. `bot.uids` is also stored in the storage above.
```
#### Ja data
python scripts/dataset/twitterv3/filter_dialogs.py \
       dataset.ja.1turn.mecab \
       --lang ja \
       --bot-id-path scripts/dataset/twitterv3/ids/bot.uids \
       --min-words 2 \
       --max-words 256 

#### En data
python scripts/dataset/twitterv3/filter_dialogs.py \
       dataset.en.1turn \
       --lang en \
       --bot-id-path scripts/dataset/twitterv3/ids/bot.uids \
       --min-words 2 \
       --max-words 256 
```