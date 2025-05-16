# CV 期中任务一

本项目通过微调在ImageNet上预训练的卷积神经网络ResNet-18实现 [Caltech-101 ](https://data.caltech.edu/records/mzrjq-6wc02)分类，支持模型训练、测试和超参数搜索。

## 项目结构

```bash
CV_Mid/
├── data/
│   └── 101_ObjectCategories/         # 数据集目录
├── saved_models/                     # 保存模型参数的目录
│   └── ...                           # *.pth 文件
├── runs/                             # TensorBoard 日志目录
│   └── ...                           # 日志文件
├── data_process.py                   # 数据处理与划分模块
├── train.py                          # 训练脚本
├── test.py                           # 测试脚本
├── hyperparam_search.py              # 超参数搜索脚本
└── README.md                         # 项目说明
```

### 训练模型

1. 打开 `train.py` 文件，自定义以下超参数：

   - `lr`：学习率
   - `batch_size`：每批样本数
   - `weight_decay`：正则化系数
   - `step_size`：每隔多少个 epoch 学习率进行一次调整
   - `gamma`：学习率衰减系数
   - `epochs`：训练轮数

2. 运行 `train.py` 文件训练模型：

   ```bash
   python train.py
   ```
   
   训练好的模型权重文件将自动保存并放置在 `saved_models/` 目录下。同时，脚本会自动调用测试函数完成模型测试。最终将生成结果文件 `experiment_results.csv` 自动保存在当前目录下，**无需手动测试**。

3. 训练结束后，可视化文件将自动保存到 `runs/` 目录下，可以通过TensorBoard查看训练过程中训练集和验证集的 loss 曲线，以及验证集 accuracy 曲线。可通过以下命令启动 TensorBoard：

   ```bash
   tensorboard --logdir=runs
   ```

   在浏览器中打开 http://localhost:6006 即可查看。

### 测试模型

如需单独测试已训练好的模型，请打开 `test.py` 文件，自定义以下内容：

- `data_path` ：数据集路径
- `model_path`：待测试的模型路径

填写完毕后，运行 `test.py` 文件：

```bash
python test.py
```


### 超参数搜索

打开 `hyperparam_search.py` 文件，在”超参数搜索空间“部分填写需要搜索的超参数范围，并运行文件：

```bash
python hyperparam_search.py
```

训练好的模型权重文件将自动保存并放置在 `saved_models/` 目录下。同样的，脚本会自动调用测试函数完成模型测试。最终将生成结果文件 `hyperparam_search_results.csv` 自动保存在当前目录下，无需手动测试。

搜索结束后，所有可视化文件将自动保存到 `runs/` 目录下，可通过TensorBoard查看（方法见上）。
