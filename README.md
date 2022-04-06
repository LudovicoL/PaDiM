# PaDiM - Anomaly Detection Localization
Extended implementation of [PaDiM](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master).

Original paper: [**PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization**](https://arxiv.org/pdf/2011.08785v1.pdf)

## Datasets
* **MVTec AD**: Download from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
* **AITEX**: Download from [AITEX website](https://www.aitex.es/afid/)
* **BTAD**: Download from [AViReS Laboratory website](http://avires.dimi.uniud.it/papers/btad/btad.zip)

## How to use
To train the model, use:
```
python train.py
```
To evaluate the model:
```
python evaluate.py
```
The option ```-d``` permits to choose the dataset (```aitex```, ```mvtec```, ```btad```).
In the train file, the datasets will be automatically downloaded.

## Minimum requirements
Python 3.9 with PyTorch 1.9.0. Use the file ```environment.yml``` for the conda environment.


## Reference
[1] Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier. *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. https://arxiv.org/pdf/2011.08785

[2] https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

