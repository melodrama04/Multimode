# Multimodel 

*基于*BERT+ResNet的NaiveCombine融合方法

## 文件结构：

```
|---DAtaProcess.py
|---Utils.py
|---main.py
|---test_without_label
|---train.txt
|---data
| |---test.json
| |---trian.json
| |---img
| |--text
```

## Requirements

```
torch~=2.0.0
argparse~=1.4.0
transformers~=4.29.1
utils~=1.0.1
tqdm~=4.64.1
torchvision~=0.15.1
chardet~=5.1.0
Pillow~=8.4.0
numpy~=1.22.4
sklearn~=0.0
scikit-learn~=1.0.1
```

## Train

```
--train --epoch 10 
```

## Test

```
--test --load_model_path ./output/NaiveCombine/pytorch_model.bin
```

## 结果

NaiveCombine  0.72

#### 消融实验

| Feature    | Acc  |
| ---------- | ---- |
| Text Only  | 0.69 |
| Image Only | 0.63 |