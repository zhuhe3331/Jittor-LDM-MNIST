# 基于Latent Diffusion Model的Jittor手写数字生成

## 简介

本项目基于[Jittor](vscode-file://vscode-app/c:/Users/ASUS/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)框架实现了基于潜在扩散模型（Latent Diffusion Model, LDM）的手写数字生成任务。通过对MNIST数据集的训练，模型能够生成高质量的手写数字图片。项目同时与PyTorch版本进行了对齐实验，验证了Jittor版本的正确性和生成效果。

## 环境配置

本项目可在单张RTX4090D上运行

#### 运行环境

+ ubuntu 22.04 LTS
+ python >= 3.7
+ jittor >= 1.3.0

#### 安装依赖

````
pip install -r requirements.txt
````

### 预训练模型

预训练模型放在`checkpoint`下。

## 数据准备

项目会自动下载MNIST数据集，无需额外操作。

## 训练

以下是各模型的训练脚本：

- **DDPM**: `jittor_version/script/train_DDPM_MNIST.py`
- **VAE**: `jittor_version/script/train_VAE_jittor.py`
- **LDM**: `jittor_version/script/train_LDM_MNIST.py`

运行示例：

```
cd jittor_version
python ./script/train_DDPM_MNIST.py
```

## 测试

以下是各模型的测试脚本：

- **DDPM**: `jittor_version/script/eval_DDPM_MNIST.py`
- **VAE**: `jittor_version/script/eval_VAE_jittor.py`
- **LDM**: `jittor_version/script/eval_LDM_MNIST.py`

运行实例：

```
cd jittor_version
python ./script/eval_DDPM_MNIST.py
```

## 与PyTorch版本对齐

为了验证Jittor版本的正确性，我们与PyTorch版本进行了对齐实验。以下是对齐的内容：

1. **训练过程日志**：
   - 记录了训练过程中的loss曲线。
   - 对比了Jittor和PyTorch版本的loss变化趋势。
2. **生成结果**：
   - 对比了生成图片的质量。

## 实验结果

#### Loss曲线对比

**DDPM**：

![DDPM_loss](.\results\loss_ddpm.png)

**DDPM-Conditonal**:

![DDPM_cond_loss](.\results\loss_ddpm_cond.png)

**VAE**:

![VAE_loss](.\results\loss_VAE.png)

**LDM**:

![LDM_loss](.\results\loss_ldm.png)

**LDM-Conditional**:

![LDM_cond_loss](.\results\loss_ldm_cond.png)

#### 生成结果对比

**DDPM**：

+ Jittor

![ddpm_jittor](.\jittor_version\results\mnist_ddpm_1746265160.169301.png)

+ PyTorch

![ddpm_jittor](.\pytorch_version\results\mnist_ddpm_1746264647.3923814.png)

**DDPM-Conditional**：

+ Jittor

![ddpm_cond_jittor](.\jittor_version\results\mnist_ddpm_cond_1746265200.9708383.png)

+ PyTorch

![ddpm_cond_jittor](.\pytorch_version\results\mnist_ddpm_cond_1746264662.2544203.png)

**VAE**：

+ Jittor

![vae_jittor](.\jittor_version\results\mnist_VAE_1746265317.1330144.png)

+ PyTorch

![vae_jittor](.\pytorch_version\results\mnist_VAE_1746264726.71298.png)

**LDM**：

+ Jittor

![ldm_jittor](.\jittor_version\results\mnist_ldm_1746265365.8868375.png)

+ PyTorch

![ldm_jittor](.\pytorch_version\results\mnist_ldm_1746264769.0900123.png)

**LDM-Conditional**：

+ Jittor

![ldm_cond_jittor](.\jittor_version\results\mnist_cond_ldm_1746265422.801435.png)

+ PyTorch

![ldm_cond_jittor](.\pytorch_version\results\mnist_ldm_cond1746265690.3888752.png)

## 性能对比

对相同结构的UNet进行测试

| 框架    | 效率               | 对比 |
| ------- | ------------------ | ---- |
| Jittor  | 365.15027328023933 | 158% |
| PyTorch | 229.58482533495663 | 100% |

## 致谢

此项目基于论文*Denoising Diffusion Probabilistic Models*, *High-Resolution Image Synthesis with Latent Diffusion Models* 实现，代码部分参考 [latent-diffusion](https://github.com/CompVis/latent-diffusion)。

