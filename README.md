<div align="center">

# Global Landslide Segmentation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2310.05959-B31B1B.svg)](https://arxiv.org/abs/2310.05959)
[![Data](http://img.shields.io/badge/Data-Zenodo-4b44ce.svg)](https://zenodo.org/records/13471162)

</div>

## Description

Code release for the paper <b>"Automating global landslide detection with heterogeneous ensemble deep-learning classification"</b>

Ganerød, A. J., Franch, G., Lindsay, E., & Calovi, M. (2023). Automating global landslide detection with heterogeneous ensemble deep-learning classification. arXiv preprint arXiv:2310.05959.

<b>preprint</b>: https://arxiv.org/abs/2310.05959

<b>data, code, and models</b>: https://zenodo.org/records/13471162

## How to run

Install dependencies

```bash
# clone project repository
git clone https://github.com/franchg/global-landslide-mapping
cd global-landslide-mapping

# create and activate environment
# adjust python version if needed
PYTHON_VERSION=3.11
python$PYTHON_VERSION -m venv .venv
source .venv/bin/activate

# install requirements
pip install -r requirements.txt
```

Download data from Zenodo by executing the script in the `data/` folder

```bash
cd data
bash download_data.sh
```

Download models from Zenodo by executing the script in the `models/` folder

```bash
cd models
bash download_pretrained_models.sh
```

See the notebook [notebooks/test_enseble_classification.ipynb](notebooks/test_enseble_classification.ipynb) for running the models on the test data and generating the results.

## Train from scratch

Train an ensemble of models with one of the following configurations contained in the folder [configs/experiment/](configs/experiment/):
- [all_channels](configs/experiment/all_channels.yaml) - all channels
- [sentinel1_2](configs/experiment/sentinel1_2.yaml) - Sentinel-1 and Sentinel-2 bands
- [sentinel1](configs/experiment/sentinel1.yaml) - Sentinel-1 bands
- [sentinel2](configs/experiment/sentinel2.yaml) - Sentinel-2 bands

```bash
# train on GPU using all channels as input
# this will train an ensemble of 90 models (2 lr x 5 losses x 9 models)
# the result will be saved in the folder `logs/train/multirun/`
python src/train.py --multirun hparams_search=grid_search trainer=gpu experiment=all_channels.yaml 
```