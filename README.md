# Char CNN LSTM Part-of-Speech Tagger
Char CNN LSTM Part-of-Speech Tagger based on the architecture described in the paper
[Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615).
The main difference is usage of bidirectional LSTM.
Some pieces of code are borrowed from
[TensorFlow implementation of lstm-char-cnn](https://github.com/mkroutikov/tf-lstm-char-cnn).

## Running
We train this network on [OpenCorpora corpus](http://opencorpora.org/) for Russian language.
To download the latest version of the corpus execute:
```sh
python download_data.py
```
Then you can train the network by executing the following command:
```sh
python train.py
```
