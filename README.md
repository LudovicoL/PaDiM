# PaDiM - Anomaly Detection Localization
Extended implementation of [PaDiM](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master).

Original paper: [**PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization**](https://arxiv.org/pdf/2011.08785v1.pdf)

## Datasets
* **MVTec AD**: Download from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
* **AITEX**: Download from [AITEX website](https://www.aitex.es/afid/)
* **BTAD**: Download from [Kaggle](https://www.kaggle.com/thtuan/btad-beantech-anomaly-detection)

## How to use
```
python main.py
```
The option ```-d``` permits to choose the dataset.
MVTec AD and AITEX datasets will be automatically downloaded. For BTAD, it is necessary to download the archive and copy into ```datasets``` directory.

## Reference
[1] Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier. *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. https://arxiv.org/pdf/2011.08785

[2] https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

