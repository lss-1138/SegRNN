# SegRNN
Welcome to the official repository of the SegRNN paper: "[Segment Recurrent Neural Network for Long-Term Time Series Forecasting.](https://arxiv.org/abs/2308.11200)"

## Updates
🚩 **News** (2024.04) SegRNN has been included in [[Time-Series-Library]](https://github.com/thuml/Time-Series-Library) and serves as the only RNN-based baseline.

## Introduction
SegRNN is an innovative RNN-based model designed for Long-term Time Series Forecasting (LTSF). It incorporates two fundamental
strategies:
1. The replacement of point-wise iterations with segment-wise iterations
2. The substitution of Recurrent Multi-step Forecasting (RMF) with Parallel Multi-step Forecasting (PMF)

![image](Figures/Figure4.png)

By combining these two strategies, SegRNN achieves state-of-the-art results with just **a single layer of GRU**, making it extremely lightweight and efficient.

![image](Figures/Table2.png)

Lots of readers have inquired about why there is a significant difference between the MSE and MAE metrics for Traffic data in the paper. 
This is because the presence of outlier extreme values in the Traffic data amplifies the MSE error. 
After adopting the mainstream [ReVIN](https://openreview.net/pdf?id=cGDAkQo1C0p) strategy, this issue was resolved, and the forecast accuracy was further improved.
## Getting Started

### Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:


```
conda create -n SegRNN python=3.8
conda activate SegRNN
pip install -r requirements.txt
```

### Data Preparation

All the datasets needed for SegRNN can be obtained from the [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. 
Create a separate folder named ```./dataset``` and place all the CSV files in this directory.
**Note**: Place the CSV files directly into this directory, such as "./dataset/ETTh1.csv"
### Training Example

You can easily reproduce the results from the paper by running the provided script command. For instance, to reproduce the main results, execute the following command:

```
sh run_main.sh
```

Similarly, you can specify separate scripts to run independent tasks, such as obtaining results on etth1:

```
sh scripts/SegRNN/etth1.sh
```

You can reproduce the results of the ablation learning by using other instructions:

```
sh scripts/SegRNN/ablation/rnn_variants.sh
```

## Citation
If you find this repo useful, please cite our paper.
```
@misc{lin2023segrnn,
      title={SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting}, 
      author={Shengsheng Lin and Weiwei Lin and Wentai Wu and Feiyu Zhao and Ruichao Mo and Haotong Zhang},
      year={2023},
      eprint={2308.11200},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgement

We extend our heartfelt appreciation to the following GitHub repositories for providing valuable code bases and datasets:

https://github.com/yuqinie98/patchtst

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/ts-kim/RevIN

https://github.com/timeseriesAI/tsai
