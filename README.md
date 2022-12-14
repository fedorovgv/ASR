# ASR

--- 

## Installation guide

### Requirements

1) Python 3.8 or above
2) GPU for training

### Installation

#### pyenv

It is recommended, but not required, to use a 
virtual environment when using the library.

<details><summary>Linux</summary>

```shell
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

</details>

<details><summary>Mac OS</summary>

```shell
brew install pyenv
```

</details>

```shell
pyenv install PYTHON_VERSION
```

Install virtual environment in repository.

```shell
git clone https://github.com/fedorovgv/ASR

cd <local_path_to_asr>/
~/.pyenv/versions/PYTHON_VERSION/bin/python -m venv VENV_NAME
source VENV_NAME/bin/activate

pip install -r ./requirements.txt
```

#### without pyenv

```shell
git clone https://github.com/fedorovgv/ASR 
cd ASR && pip install -r requirements.txt
```

In some cases, it may need to add the path to the python path.

```shell
export PYTHONPATH="${PYTHONPATH}:/<local_path_to_asr>/hw_asr/"
```

--- 

## Pre-trained models

```shell
cd <local_path_to_asr>/
mkdir pre_trained && cd pre_trained

# libri speech pre-trained language model
mkdir -p lm_models/ && cd lm_models/
wget "https://www.dropbox.com/s/9sla5bg7q0mmv25/3-gram.arpa?dl=1"
mv '3-gram.arpa?dl=1' 3-gram.arpa

# deep speech checkpoint
mkdir -p ../checkpoint && cd ../checkpoint/
wget "https://www.dropbox.com/s/pfxgw812ye1q8be/checkpoint.pth?dl=1"
mv 'checkpoint.pth?dl=1' checkpoint.pth
wget "https://www.dropbox.com/s/r6ru5jq4ypjsanc/config.json?dl=1"
mv config.json\?dl\=1 config.json
```

To reproduce model performance on the test-other in repo do: 

```shell
mkdir -p tmp && echo "tmp/" >> .gitignore
python test.py \
      -r pre_trained/checkpoint/checkpoint.pth \
      -o tmp/test_result.json \
      --libri test-other
```

| 	               | wer argmax	 | cer argmax	 | wer beam cearch + lm 	 |
|-----------------|------------|-------------|------------------------|
| 	**test-other** | 	 0.28211391         | 	 0.1049535411          | 	0.1929986279          |

---

## Tests

General tests:
```shell
python -m unittest discover hw_asr/tests
```

---

## Some features

### Language Model

You can load libri speech lm models manually from [libri](http://www.openslr.org/11/) site or do
it automatically by selecting libri model name from available model list:

```shell
3-gram.arpa
3-gram.pruned.1e-7.arpa
3-gram.pruned.3e-7.arpa
```

and passing it into config:

```json lines
"text_encoder": {
    "type": "CTCCharTextEncoder",
    "args": {
        "alphabet" : null,
        "lm_model" : MODEL_NAME,
        "alpha" : ALPHA,
        "beta" : BETA
    }
}
```

If you use lm model it is possible to evaluate wer metric by beam search with 
lm:

```shell
"metrics": [
        {
            "type": "LMWERMetric",
            "args": {
                "name": "WER LM"
            }
        },
        ...
    ],
```

### Augmentation

#### Available

Now it is available 

```shell
wave augment
  Noise
  Gain
 
spec augment
  TimeMasking
  FrequencyMasking
```

#### Random apply 

You can transfer the probability of using augmentation via config with `random_apply_proba`
parameter like:

```shell
"augmentations": {
        "wave": [
            {
                "type": "Noise",
                "random_apply_proba": 0.4,
                "args": {
                    "sample_rate": 16000
                }
            },
            {
                "type": "Gain",
                "args": {
                    "min_gain_in_db": -10,
                    "max_gain_in_db": 10,
                    "sample_rate": 16000
                }
            }
        ],
        ...
```

In the example above, the Noise will be used with a probability of 0.4 and the
Gain will be used all times.

--- 

## Credits


This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
