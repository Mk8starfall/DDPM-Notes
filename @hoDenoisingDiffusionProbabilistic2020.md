---
Type: paper-conference
Authors: Jonathan Ho, Ajay Jain, Pieter Abbeel
Year: 2020
Title: Denoising Diffusion Probabilistic Models
Journal: Advances in Neural Information Processing Systems
Pages: 6840–6851
DOI: 
Publisher: Curran Associates, Inc.
Place: 
---
[[@hoDenoisingDiffusionProbabilistic2020]][[@hoDenoisingDiffusionProbabilistic2020]]
# Jonathan Ho, Ajay Jain, Pieter Abbeel (2020) - Denoising Diffusion Probabilistic Models

[==_Read it now! See in Zotero_==](zotero://select/items/@hoDenoisingDiffusionProbabilistic2020a)
**Web:** [Open online](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
**Citekey:** hoDenoisingDiffusionProbabilistic2020a

## Abstract

We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN.
我们使用扩散概率模型（diffusion probabilistic models）展示了高质量图像合成的结果，这是一类受非平衡热力学理论影响的潜变量模型。我们通过根据扩散概率模型与 Langevin 动力学下的去噪得分匹配之间的新颖联系设计的加权变分下界进行训练，获得了最佳结果。我们的模型自然地采用了一种渐进有损解压方案，可以被解释为自回归解码的一种推广。在无条件的 CIFAR10 数据集上，我们获得了 9.46 的 Inception score 和 3.17 的 FID score，达到了最新的技术水平。在 256x256 的 LSUN 数据集上，我们获得了与 ProgressiveGAN 相当的样本质量。

## Methods

![[Notes Images/Pasted image 20231128000952.png]]
扩散模型：
$$
p_{\theta}(x_0):=​\int p\left(x_{T}\right) \prod_{t=1}^{T} p_{\theta}\left(x_{t-1} \mid x_{t}\right)\mathrm{d}x_{1:T}
$$
其中
$$\begin{array}{c}
p_{\theta}\left(x_{t-1} \mid x_{t}\right):=\mathcal{N}\left(x_{t-1} ; {\mu}_{\theta}\left(x_{t}, t\right), {\Sigma}_{\theta}\left(x_{t}, t\right)\right)\\
p(x_{T})=\mathcal{N}(x_T;0,I)
\end{array}$$
含义：生成的样本 $x_0$ 是由服从正态分布的噪声 $x_T$ 经过一系列**可学习**的高斯变换（每一步的均值 ${\mu}_{\theta}\left(x_{t}, t\right)$ 与方差 $\Sigma_{\theta}(x_{t}, t)$ 只与参数 $\theta$ 及上一步的结果 $x_{t}$ 有关，即具有 Markov 性）去噪而成。
去噪过程也被称为**反向过程**，而**正向过程**定义如下：
$$
q(x_t|x_{t-1}):=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)
$$
含义：逐渐对样本 $x_0$ 加噪，直至变为高斯噪声
（参数 $\beta_t$ 可学习，也可设为超参数，本文取后者）
通过定义 $\alpha_t:=1-\beta_t$ 和 $\bar{\alpha_t}:=\prod_{s=1}^t \alpha_s$ 可得：
$$
q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha_t}}x_0,(1-\bar{\alpha_t})I)
$$
优化目标：$\theta$ 的极大似然估计 $p_{\theta}(x_0)$
损失函数：
$$\begin{align}
-\log p_{\theta}(x_0) \le L & = \mathbb{E}_q[D_{KL}(q(x_T|x_0)||p(x_T))+\sum_{t>1}D_{KL}(q(x_{t-1}|x_t,x_0)||p_{\theta}(x_{t-1}|x_t))-\log p_{\theta}(x_0|x_1)]
\\
&:=\mathbb{E}_q[L_T+\sum_{t>1}L_{t-1}+L_0]
\end{align}$$
*注意* $\mathbb{E}_q$ *是* $\mathbb{E}_{x_{1:T}\sim q(x_{1:T}|x_0)}$  *的简写*
观察损失函数：
（1）$L_T$ 称为**先验匹配损失**，表征正向加噪过程的结果与反向去噪过程的起点（高斯噪声）的相似度。在 $\beta_t$ 不作为学习对象的情况下，$L_T$ 也是常数，可忽略。
（2）$L_{t-1}$ 称为**去噪匹配损失**，表征去噪过程和加噪过程的相似度。注意 $p_{\theta}\left(x_{t-1} \mid x_{t}\right):=\mathcal{N}\left(x_{t-1} ; {\mu}_{\theta}\left(x_{t}, t\right), {\Sigma}_{\theta}\left(x_{t}, t\right)\right)$ ，首先取方差为常数 ${\Sigma}_{\theta}\left(x_{t}, t\right)=\sigma_t^2 I$ ，然后经过数学处理可得优化目标
$$
\mathbb{E}_{\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\frac{1}{2\sigma_t^2}\left\|\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t(\mathbf{x}_0,\boldsymbol{\epsilon})-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}\right)-\boldsymbol{\mu}_\theta(\mathbf{x}_t(\mathbf{x}_0,\boldsymbol{\epsilon}),t)\right\|^2\right]
$$
可以训练网络预测 $\mu_\theta$ ；另外，进一步地，可令（重参数化）
$$
\mu_{\theta}(x_t,t)=\frac{1}{\sqrt{{\alpha_t}}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha_t}}}\epsilon_{\theta}(x_t,t))
$$
则优化目标变为
$$
\mathbb{E}_{\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},t)\right\|^2\right]
$$
相当于需要训练一个**噪声拟合网络**。
（3）$L_0$ 称为**重建损失**，表征生成过程最后一步生成真实图像（各分量取值为 $\{0,1,\dots,255\}$ 到 $[-1,1]$ 上的线性映射）
$$
\begin{aligned}
p_\theta(\mathbf{x}_0|\mathbf{x}_1)& =\prod_{i=1}^D\int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)}\mathcal{N}(x;\mu_\theta^i(\mathbf{x}_1,1),\sigma_1^2)dx  \\
\delta_{+}(x)& =\begin{cases}\infty&\text{if }x=1\\x+\frac{1}{255}&\text{if }x<1\end{cases}\quad\delta_-(x)=\begin{cases}-\infty&\text{if }x=-1\\x-\frac{1}{255}&\text{if }x>-1\end{cases} 
\end{aligned}
$$
$$
\begin{aligned}
L_{0}& =\mathbb{E}_q\bigg[-\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)\bigg]  \\
&\approx\mathbb{E}_q\Big[\sum_i\frac1{2\sigma_1^2}\left(\mu_\theta^i(\mathbf{x}_1,1)-x_0^i\right)^2\Big] \\
&=\mathbb{E}_{q}\Big[\frac{1-\alpha_{1}}{2\alpha_{1}\sigma_{1}^{2}}\|\epsilon_{1}-\epsilon_{\theta}\big(\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}}\boldsymbol{\epsilon}_{1},1)\|^{2}\Big]
\end{aligned}
$$
于是最小化 $L_0$ 的训练过程与 $L_{t-1}$ 相同。
最终的**简化损失函数**：
$$
L_{\mathrm{simple}}(\theta):=\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\left\|\epsilon-\boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},t)\right\|^2\right]
$$
训练与采样算法：
![[Notes Images/Pasted image 20231128233047.png]]
## Summary

正向过程：对真实图像加噪至高斯噪声；
反向过程：对高斯噪声去噪生成图像；
反向过程的参数通过神经网络学习（预测正向过程的噪声）。
贡献：提出了崭新的生成式模型，现在已成为图像生成领域的重要方法。
优点：生成效果好；不容易出现GAN容易出现的模式崩溃现象；
缺点：**一个重要的缺点：采样慢**；对数似然相比其他模型没有竞争力；

