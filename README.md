### 环境配置

```
pip install -r requirements.txt
```



### 文件结构及说明

dataset

- train：训练过程中用于存放相关数据的文件夹
- predict：预测过程中用于存放相关数据的文件夹


- `dataset.py`：用于辅助DataLoader的类

model

- `model.py`：mlp模型：输入维度100 ( audio encoder的输出维度 ) + 384 ( sentence encoder的输出维度 )

records：存放权重记录，每组权重放在一个命名为 "`vx`-`epochs`-`train loss`-`train acc`-`validation loss`-`validation acc`"的文件夹中，如 "v2-100-1.4343-0.5854-1.2001-0.6667"

- audio+text+mlp：三个模型一起训练的权重参数
- audio+mlp：仅训练audio encoder和mlp
- text+mlp：仅训练sentence encoder和mlp
- mlp：仅训练mlp

utils：

- `oss_utils.py`：oss操作的类

`benchmark.py`：构建benchmark的代码

`benchmark_v1.csv`：初版benchmark，每个场景选取了100组测试数据

`predict.py`：含单组数据的预测函数，也可根据benchmark的测试数据来构建arena任务，

`train.py`：训练函数

`README.md`：说明文档

`requirements.txt`：环境依赖

`synthesis_oss_log_v2.csv`：合成音频的oss存放地址信息