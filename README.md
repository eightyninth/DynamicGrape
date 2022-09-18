# DynamicGrape

## 环境

```
torch                   1.8.0+cu111
torch-cluster           1.5.9
torch-geometric         2.0.3
torch-scatter           2.0.6
torch-sparse            0.6.9
torch-spline-conv       1.2.1
torchaudio              0.8.0
torchvision             0.9.0+cu111
```

## 参数设置

> 参数设置均在<font color=blue>config</font>中的json文件中。
```
{ 
  "training": 0代表评估,1代表训练,
  "device": "cuda"代表使用GPU训练,"cpu"代表使用CPU训练,
  "dataset":{
    "path": 数据集位置,
    "batch_size": batch数
  },
  "hyperparams": {
    "epochs": 训练轮数,
    "visualization": 1代表可视化训练过程损失变化,0代表不进行可视化,
    "resume": 参数加载位置,
    }
}
```

## 预处理

下载数据集[Google Drive](https://drive.google.com/file/d/1OM6CCrUvuBuuykXjBigKyvmYUfjzudO5/view?usp=sharing)到本地

## 训练

设置好参数后
```bash
python solver.py
```

## 评估

设置好参数后
```bash
python solver.py
```