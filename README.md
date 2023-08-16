# SegRNN 

Welcome to the anonymous repository of SegRNN: "Segment Recurrent Neural Network for Long-Term Time Series Forecasting."

SegRNN is an innovative RNN-based model designed for Long-term Time Series Forecasting (LTSF). It incorporates two fundamental
strategies:
1. The replacement of point-wise iterations with segment-wise iterations
2. The substitution of Recurrent Multi-step Forecasting (RMF) with Parallel Multi-step Forecasting (PMF)

## Getting Started

### Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:


```
conda create -n SegRNN python=3.8
conda activate SegRNN
pip install -r requirements.txt
```

### Data Preparation

All the datasets needed for SegRNN can be obtained from the [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. Create a separate folder named ```./dataset``` and place all the CSV files in this directory.

### Training Example

You can easily reproduce the results from the paper by running the provided script command. For instance, to reproduce the main results, execute the following command:

```
sh run_main.sh
```

Similarly, you can specify separate scripts to run independent tasks, such as obtaining results on etth1:

```
sh script/SegRNN/etth1.sh
```

You can reproduce the results of the ablation learning by using other instructions:

```
sh script/SegRNN/ablation/rnn_variants.sh
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
