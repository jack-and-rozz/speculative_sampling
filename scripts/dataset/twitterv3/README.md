
### 0. crawl tweets and make lists of daily tweets. The format of the lists is as follows. Each column is separated by '\t'.
```
tweet-type(tweet or mention), epochtime, tweet-id, mention-target-id, user-id, user-name, user-screen, text,
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


### 3. (if necessary) tokenize text files in 'dataset.ja.1turn'.
```
#### Ja data
python scripts/dataset/twitterv3/tokenize.py \
       dataset.ja.1turn \
       ja \
       mecab \
       --target-suffixes dialogs src tgt mulres
```


### 4. (if necessary) filter conversations by their length, whether they were made by bots, etc. 
```
#### Ja data
python scripts/dataset/twitterv3/filter_dialogs.py \
       dataset.ja.1turn.mecab \
       --lang ja \
       --bot-id-path scripts/dataset/twitterv3/profiles/bot.uid \
       --min-words 2 \
       --max-words 256 

#### En data
python scripts/dataset/twitterv3/filter_dialogs.py \
       dataset.en.1turn \
       --lang en \
       --bot-id-path scripts/dataset/twitterv3/profiles/bot.uid \
       --min-words 2 \
       --max-words 256 
```