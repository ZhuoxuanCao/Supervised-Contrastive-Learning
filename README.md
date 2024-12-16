对比学习相关的代码都在contrastive Learning文件夹里面

# 代码结构
Contrastive_Learning/：包含与对比学习相关的代码。
- init.py：该文件是简单的初始化文件，将 Contrastive_Learning 文件夹标记为一个 Python 包。通过该文件，项目中其他模块可以导入 Contrastive_Learning 文件夹中的函数、类或配置。
- config_con.py：该文件是配置文件，包含对比学习训练过程中的参数设置。
- train_con.py：该文件实现了监督式对比学习的训练过程。

data_augmentation/：包含数据增强的相关代码，用于在训练过程中对输入数据进行预处理。

losses/：定义了损失函数，supin和supout。

models/：存放用于测试的模型代码，包括不同架构的神经网络模型定义，以及MLP的实现。

my_logs/：用于存储训练过程中的日志信息，便于跟踪和分析模型的训练情况。

saved_models/：用于保存训练后的模型，便于后续加载和评估。
- classification/：该目录保存了用于分类任务的模型权重
- pretrain/：经过对比学习预训练的分类任务模型
- scratch/：从头开始，未经过预训练的分类任务模型
- pretraining/：该目录保存了经过对比监督学习的预训练权重
- ResNet34/：使用 ResNet34 进行监督式对比学习训练后的权重
- ResNet101/：使用 ResNet101 进行监督式对比学习训练后的权重

## 模型保存目录结构

`saved_models/`：用于保存训练后的模型，便于后续加载和评估。

- **classification/**：  
  该目录保存了用于分类任务的模型权重。  
  - **pretrain/**：  
    经过对比学习预训练的分类任务模型。  
  - **scratch/**：  
    从头开始，未经预训练的分类任务模型。  

- **pretraining/**：  
  该目录保存了经过对比监督学习的预训练权重。  
  - **ResNet34/**：  
    使用 **ResNet34** 进行监督式对比学习训练的权重。  
  - **ResNet101/**：  
    使用 **ResNet101** 进行监督式对比学习训练的权重。  


environment.yml：列出了项目所需的Python库及其版本，便于环境的搭建和依赖管理。

main_con.py：主程序入口，负责解析命令行参数，并调用相应的训练和测试函数。

test_classifier.py：用于测试分类器性能的代码，评估模型在测试集上的表现。

train_pretrained_classifier.py：用于训练预训练分类器的代码，加载预训练模型并进行微调。

train_scratch_classifier.py：用于从头开始训练分类器的代码，初始化模型并进行训练。

utils.py：包含辅助函数，如数据加载、模型保存和日志记录等功能。


## 参数解释

以下是项目中各个命令行参数的解释及其用途，您可以根据项目需求灵活配置这些参数：

* `--batch_size`：设置批量大小（默认值：32）。示例： `--batch_size 8` ，表示每次迭代使用8个样本进行训练。
* `--learning_rate`：设置学习率（默认值：0.001），控制优化器的步长。较小的学习率会使模型收敛更慢但可能获得更好的结果。示例：`--learning_rate 0.01`。
* `--epochs`：设置训练的 epoch 数（默认值：15），即数据集的完整迭代次数。示例：`--epochs 25` 表示训练25个完整的 epoch。
* `--temp`：设置对比损失中的温度参数（默认值：0.07），用于缩放对比损失。示例：`--temp 0.1`。
* `--save_freq`：设置模型检查点的保存频率（默认值：5），即每隔多少个 epoch 保存一次模型。示例：`--save_freq 3` 表示每3个 epoch 保存一次。
* `--log_dir`：设置 TensorBoard 日志保存目录（默认值：`./logs`）。所有的训练日志会保存在该目录下。示例：`--log_dir ./my_logs`。
* `--model_save_dir`：设置模型检查点保存目录（默认值：`./checkpoints`）。训练过程中每个保存的模型文件会保存在该目录下。示例：`--model_save_dir ./my_checkpoints`。
* `--gpu`：指定使用的 GPU 设备 ID（默认值：0）。设置为 `None` 时则使用 CPU。示例：`--gpu 0` 表示使用第一个 GPU。
* `--dataset`：指定数据集的存储路径（默认值：`./data`）。示例：`--dataset ./data`。
* `--dataset_name`：设置数据集名称，支持 `cifar10`、`cifar100`、`imagenet`（默认值：`cifar10`）。示例：`--dataset_name cifar10`。
* `--model_type`：选择用于训练的模型类型（默认值：`resnet34`），支持 `resnet34`、`ResNeXt101`、`WideResNet` 等。示例：`--model_type ResNeXt101`。
* `--loss_type`：选择损失函数类型（默认值：`supcon`），可选项包括 `cross_entropy`、`supcon`、`supin`。示例：`--loss_type supcon` 表示使用 `SupConLoss_out` 损失函数。
* `--augmentation`：设置数据增强方式（默认值：`basic`），支持 `basic` 和 `advanced`。示例：`--augmentation advanced`。

### 示例运行指令

```bash
python main.py --batch_size 8 --learning_rate 0.0001 --epochs 5 --temp 0.1 --save_freq 3 --log_dir ./my_logs --model_save_dir ./my_checkpoints --gpu 0 --dataset ./data --dataset_name cifar10 --model_type ResNet34 --loss_type SupOut --augmentation basic
```


