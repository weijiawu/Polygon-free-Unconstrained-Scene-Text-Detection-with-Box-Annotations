# SelfText Beyond Polygon: Unconstrained Text Detection with Box Supervisionand Dynamic Self-Training
![Alt text](https://user-images.githubusercontent.com/40711216/100269392-69fec600-2f91-11eb-8225-735d435caf58.png)
### Introduction
&#160;&#160;&#160;&#160; &#160;&#160;This is a project concerning SelfText Beyond Polygon: Unconstrained Text Detection with Box Supervisionand Dynamic Self-Training

&#160;&#160;&#160;&#160; &#160;&#160;we propose a novel text detection system termed SelfText Beyond Polygon~(SBP) with Bounding Box Supervision~(BBS) and Dynamic Self Training~(DST), where training a polygon-based text detector with only a limited set of upright bounding box annotations. 

&#160;&#160;&#160;&#160; &#160;&#160;From more details,please refer to our arXiv paper.(TDB)
## Environments

- python 3
- torch = 1.1.0 
- torchvision
- Pillow
- numpy

## Dataset
Supported:
- [x] ICDAR15
- [x] ICDAR17MLI
- [x] sythtext800K
- [x] TotalText
- [x] MSRA-TD500
- [ ] CTW1500


## model zoo

Supported text detection:
- [x] EAST [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)
- [x] Psenet [Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/abs/1903.12473)
- [ ] DB

## Bounding Box Supervision(BBS)

### Train

The training strategy includes three steps: 
(1) training SASN with synthetic data 
(2) generating pseudo label on real data based on bounding box annotation with SASN
(3) training the detectors(EAST and PSENet) with the pseudo label

#### training SASN with [synthtext](https://github.com/ankush-me/SynthText) or [curved synthtext](https://github.com/PkuDavidGuan/CurvedSynthText) 
```
(TDB)
```

#### generating pseudo label on real data with SASN
```
(TDB)
```

#### training EAST or PSENet with the pseudo label
```
(TDB)
```

### Eval
for example (batchsize=2)
```
(TDB)
```
### Visualization


## Dynamic Self Training
### Train
```
(TDB)
```

### Eval
for example (batchsize=2)
```
(TDB)
```
### Visualization


## Experiments

### Bounding Box Supervision

#### The performance of EAST on ICDAR15
|Method|Dataset|Pretrain|precision|recall|f-score|
|:---:|:---:|:---:|:---:|:---:|:---:|
|EAST_box|ICDAR15|-|65.8|63.8|64.8|
|EAST|ICDAR15|-|76.9|77.1|77.0|
|EAST_pseudo(SynthText)|ICDAR15|-|77.8|78.2|78.0|
|EAST_box|ICDAR15|SynthText|70.8|72.0|71.4|
|EAST|ICDAR15|SynthText|82.0|82.4|82.2|
|EAST_pseudo(SynthText)|ICDAR15|SynthText|81.3|82.2|81.8|
***

#### The performance of EAST on MSRA-TD500
|Method|Dataset|Pretrain|precision|recall|f-score|
|:---:|:---:|:---:|:---:|:---:|:---:|
|EAST_box|MSRA-TD500|-|40.49|31.05|35.15|
|EAST|MSRA-TD500|-|71.76|69.05|70.38|
|EAST_pseudo(SynthText)|MSRA-TD500|-|71.27|67.54|69.36|
|EAST_box|MSRA-TD500|SynthText|48.34|42.37|45.16|
|EAST|MSRA-TD500|SynthText|77.91|76.45|77.17|
|EAST_pseudo(SynthText)|MSRA-TD500|SynthText|77.42|73.85|75.59|
***

#### The performance of PSENet on ICDAR15
|Method|Dataset|Pretrain|precision|recall|f-score|
|:---:|:---:|:---:|:---:|:---:|:---:|
|PSENet_box|ICDAR15|-|70.17|69.09|69.63|
|PSENet|ICDAR15|-|81.6|79.5|80.5|
|PSENet_pseudo(SynthText)|ICDAR15|-|82.9|77.6|80.2|
|PSENet_box|ICDAR15|SynthText|72.65|74.29|73.46|
|PSENet|ICDAR15|SynthText|86.42|83.54|84.96|
|PSENet_pseudo(SynthText)|ICDAR15|SynthText|86.77|83.34|85.02|

***

#### The performance of PSENet on MSRA-TD500
|Method|Dataset|Pretrain|precision|recall|f-score|
|:---:|:---:|:---:|:---:|:---:|:---:|
|PSENet_box|MSRA-TD500|-|47.17|36.90|41.41|
|PSENet|MSRA-TD500|-|80.86|77.72|79.13|
|PSENet_pseudo(SynthText)|MSRA-TD500|-|80.32|77.26|78.86|
|PSENet_box|MSRA-TD500|SynthText|47.45|39.49|43.11|
|PSENet|MSRA-TD500|SynthText|84.11|84.97|84.54|
|PSENet_pseudo(SynthText)|MSRA-TD500|SynthText|84.03|84.03|84.03|
***

#### The performance of PSENet on Total Text
|Method|Dataset|Pretrain|precision|recall|f-score|
|:---:|:---:|:---:|:---:|:---:|:---:|
|PSENet_box|Total Text|-|46.5|43.6|45.0|
|PSENet|Total Text|-|80.4|76.5|78.4|
|PSENet_pseudo(SynthText)|Total Text|-|80.33|73.54|76.78|
|PSENet_pseudo(Curved SynthText)|Total Text|-|81.68|74.61|78.0|
|PSENet_box|Total Text|SynthText|51.94|47.45|49.59|
|PSENet|Total Text|SynthText|83.4|78.1|80.7|
|PSENet_pseudo(SynthText)|Total Text|SynthText|81.57|75.54|78.44|
|PSENet_pseudo(Curved SynthText)|Total Text|SynthText|82.51|77.57|80.0|

***


## links
https://github.com/SakuraRiven/EAST

https://github.com/WenmuZhou/PSENet.pytorch

## License

For academic use, this project is licensed under the Apache License - see the LICENSE file for details. For commercial use, please contact the authors. 

## Citations
Please consider citing our paper in your publications if the project helps your research.



Eamil: wwj123@zju.edu.cn