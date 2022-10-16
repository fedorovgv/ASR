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

cd <local_path_to_asr>/ASR
~/.pyenv/versions/PYTHON_VERSION/bin/python -m venv venv_asr
source venv_asr/bin/activate

pip install -r ./requirements.txt
```

#### without pyenv

```shell
git clone https://github.com/fedorovgv/ASR
pip install -r ./requirements.txt
```

In some cases, it may need to add the path to the python path.

```shell
export PYTHONPATH="${PYTHONPATH}:/<local_path_to_asr>/ASR/hw_asr/"
```

--- 

## Pre-trained models

```shell
mkdir <local_path_to_asr>/ASR/pre_trained

# libri speech pre-trained language model
mkdir lm_models/ && cd lm_models/
wget https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz
gzip -d 3-gram.pruned.1e-7.arpa.gz

# deep speech checkpoint
cd ../ && mkdir checkpoints && cd checkpoints/
wget "https://www.dropbox.com/s/u768yu1t6d3ucwe/checkpoint.pth?dl=1"
wget "https://www.dropbox.com/s/nmy0xp2erk7qf8n/ckpt_config.json?dl=1"
```

To reproduce model performance on test-other: 

```shell
python test.py \
      -r pre_trained/checkpoints/checkpoint.pth \
      -o test_result.json \
      --libri test-other
```

---

### Tests

General tests:
```shell
python -m unittest discover hw_asr/tests
```

Model tests: 

```shell
# tmp is also in .gitignore
mkdir tmp

# model test
python test.py \
      -c hw_asr/tests/default_test_config.json \
      -r pre_trained/checkpoints/checkpoint.pth \
      -o tmp/test_result.json
```

---

## Credits


This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
