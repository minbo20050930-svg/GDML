<p align="center">
  <h1 align="center">GDML: Geometric Decoupling Mutual Learning for Robust Skin Lesion Segmentation Using Distance Transforms</h1>
  <p align="center">
    <a href=""><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8%2B-green.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/pytorch-1.10%2B-orange.svg"></a>
  </p>
</p>

## 📝 Introduction

Accurate segmentation of skin lesions is crucial for the diagnosis and management of skin diseases. However, this task remains highly challenging due to the structural complexity of lesion interiors and the ambiguity of boundary regions, characterized by low contrast, textural heterogeneity, and blurred interfaces between lesions and healthy skin. To address these critical limitations, we propose a geometric decoupling mutual learning (GDML) model based on distance transforms, which aims to extract deeper geometric representations by explicitly modeling the distinct characteristics of internal and boundary pixels through separate learning branches. The GDML framework employs an information interaction module as its fundamental building block, upon which two parallel and mutually interactive pathways are constructed to facilitate rich cross-modal information exchange. Furthermore, we design a feature enhancement module that bridges the encoder and decoder, dynamically guiding the network’s attention toward lesion regions to mine fine-grained geometric features. Additionally, a distance transform-based supervision strategy is introduced to decouple original lesion images into internal-specific and boundary-specific labels. These dual labels respectively guide the learning of internal and boundary pixels: the internal-specific label emphasizes the consistency of intra-lesion features, while the boundary-specific label enforces the continuity and precision of boundary features. Experimental results on three benchmark datasets (ISIC2017, ISIC2018, and PH2) demonstrate that our method outperforms current state-of-the-art approaches in terms of Dice and IoU metrics.

## 📋 Highlights

- 🔹 A novel geometric decoupling mutual learning framework for robust skin lesion segmentation (GDML). The proposed GDML model constructs decoupled internal- and boundary-focused branches and enforces mutual learning between them, explicitly enabling differential learning of complex intra-lesion structures and fine-grained boundary details, thereby overcoming the limitations of conventional coupled feature learning and yielding more accurate and reliable lesion masks.
- 🔹 A lesion-focused feature enhancement module for stronger geometric representation learning (FEM). The proposed FEM adaptively reweights multi-scale encoder features with lesion-aware attention, highlighting structurally critical regions while suppressing background noise, thereby strengthening geometric feature learning and enabling more accurate extraction of complex internal lesion patterns and fine-grained boundary details.
- 🔹 A distance transform-based supervision strategy for decoupled internal and boundary learning (DTSS). We construct internal-specific and boundary-specific soft label from the original lesion image via distance transforms and use them to supervise both the dual output branches and the fused prediction, thereby guiding the network to learn internal and boundary features separately while simultaneously preserving, enhancing, and enforcing their mutual geometric and semantic consistency.

## 🧠 Network Architecture

<p align="center">
  <img src="figure/架构图.png" alt="Network Architecture" width="80%">
</p>

## 🧰 Installation

```bash
# 创建conda环境
conda create -n [env_name] python=3.10 -y
conda activate [env_name]

# 安装依赖
pip install -r requirements.txt
```

## 📁 Dataset Preparation

请将 [数据集名称] 下载至 `data/` 目录下，目录结构如下：

```
data/
    /ddw_data
      /skin-iamge
        ISIC2017/
        ├── images/
        └── masks/
        ISIC2018/
        PH2/
```

## 🚀 Train and Test

### Training

```bash
python train_and_test_isic.py --dataset ISIC2018
```

### Inference

```bash
python train_and_test_isic.py --dataset PH2 --eval_only
```

## 📊 Quantitative Results

| Method             | Pub. Year | ISIC2017<br>mDice | ISIC2017<br>mIoU | ISIC2018<br>mDice | ISIC2018<br>mIoU | ISIC2018→PH2<br>mDice | ISIC2018→PH2<br>mIoU |
|-------------------|-----------|-------------------|------------------|-------------------|------------------|------------------------|-----------------------|
| UNet              | MICCAI’15 | 0.919             | 0.799            | 0.901             | 0.769            | 0.911                  | 0.827                 |
| ResUNet           | ITME’18   | 0.918             | 0.797            | 0.895             | 0.764            | 0.882                  | 0.737                 |
| UNet++            | DLMIA’18  | 0.821             | 0.743            | 0.794             | 0.729            | 0.789                  | 0.713                 |
| CENet             | TMI’19    | 0.911             | 0.790            | 0.898             | 0.764            | 0.888                  | 0.782                 |
| TransUNet         | arXiv’21  | 0.934             | 0.834            | 0.921             | 0.820            | 0.915                  | 0.829                 |
| AttUNet           | MICCAI’21 | 0.713             | 0.419            | 0.704             | 0.442            | 0.707                  | 0.448                 |
| I2UNet            | MedIA’24  | 0.919             | 0.832            | 0.916             | 0.831            | 0.907                  | 0.825                 |
| UKAN              | AAAI’25   | 0.767             | 0.517            | 0.693             | 0.401            | 0.760                  | 0.529                 |
| XBoundFormer      | TMI’23    | 0.936             | 0.830            | 0.922             | 0.808            | 0.923                  | 0.846                 |
| LBUNet            | MICCAI’24 | 0.928             | 0.815            | 0.913             | 0.794            | 0.910                  | 0.821                 |
| LiteMambaBound    | Methods’25| 0.938             | 0.832            | 0.923             | 0.819            | 0.921                  | 0.840                 |
| **Ours**          | –         | **0.947**         | **0.844**        | **0.938**         | **0.835**        | **0.925**              | **0.873**             |



## 📦 Project Structure

```
├── data/
├── Datasets/
├── Models/
├── utils/
├── saved_models/
├── train_and_test_isic.py
└── README.md
```


## 📬 Contact

If you have any questions, feel free to contact us at minbo20050930@gmail.com.
