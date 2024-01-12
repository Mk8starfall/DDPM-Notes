---
Type: article
Authors: Yang Song, Liyue Shen, Lei Xing, Stefano Ermon
Year: 2022
Title: Solving Inverse Problems in Medical Imaging with Score-Based Generative Models
Journal: 
Pages: 
DOI: 
Publisher: arXiv
Place: 
---

# Yang Song, Liyue Shen, Lei Xing, Stefano Ermon (2022) - Solving Inverse Problems in Medical Imaging with Score-Based Generative Models
[==*Read it now! See in Zotero*==](zotero://select/items/@songSolvingInverseProblems2022)
**Web:** [Open online](http://arxiv.org/abs/2111.08005)
**Citekey:** songSolvingInverseProblems2022
**Tags:** #source, (type), (status), (decade)

## Abstract

Reconstructing medical images from partial measurements is an important inverse problem in Computed Tomography (CT) and Magnetic Resonance Imaging (MRI). Existing solutions based on machine learning typically train a model to directly map measurements to medical images, leveraging a training dataset of paired images and measurements. These measurements are typically synthesized from images using a fixed physical model of the measurement process, which hinders the generalization capability of models to unknown measurement processes. To address this issue, we propose a fully unsupervised technique for inverse problem solving, leveraging the recently introduced score-based generative models. Specifically, we first train a score-based generative model on medical images to capture their prior distribution. Given measurements and a physical model of the measurement process at test time, we introduce a sampling method to reconstruct an image consistent with both the prior and the observed measurements. Our method does not assume a fixed measurement process during training, and can thus be flexibly adapted to different measurement processes at test time. Empirically, we observe comparable or better performance to supervised learning techniques in several medical imaging tasks in CT and MRI, while demonstrating significantly better generalization to unknown measurement processes.
在计算机断层扫描（CT）和磁共振成像（MRI）中，从部分测量中重建医学图像是一个重要的反问题。现有基于机器学习的解决方案通常训练模型，直接将测量映射到医学图像，利用成对图像和测量的训练数据集。这些测量通常是使用测量过程的固定物理模型从图像合成的，这限制了模型对未知测量过程的泛化能力。为了解决这个问题，我们提出了一种完全无监督的反问题解决技术，利用最近引入的基于得分的生成模型。具体而言，我们首先在医学图像上训练一个基于得分的生成模型，以捕捉它们的先验分布。在测试时，给定测量和测量过程的物理模型，我们引入一种采样方法来重建一个既与先验一致又与观察到的测量一致的图像。我们的方法在训练期间不假设固定的测量过程，因此可以在测试时灵活适应不同的测量过程。实证上，我们观察到在CT和MRI的多个医学成像任务中，与监督学习技术相比，我们的方法在性能上具有可比较或更好的表现，同时对未知的测量过程具有显着更好的泛化能力。

## Backgrounds

线性**反问题**：假设 $x\in \mathbb{R}^n$ 是未知信号，$y \in \mathbb{R}^m = Ax +\epsilon$ 是带噪声的对 $x$ 的观测。求解反问题即是从观测 $y$ 中恢复信号 $x$ . $m<n$ 时该问题是 ill-defined 的，因此还需要以下假设：
（1）先验分布 $x \sim p(x)$ 
（2）测量分布 $p(y\mid x)=q_{\epsilon}(y-Ax)$ 
是已知的。则由 Bayes 公式可以从 $p(x \mid y)$ 中采样来得到反问题的解。
不失一般性地设 $\mathrm{rank}(A)=m$ （否则可以只考虑更低维的 $A$ ），可以证明：

**定理**：存在一个可逆矩阵 $T\in \mathbb{R}^n$ 和一个对角矩阵 $\Lambda\in \{0,1\}^{n \times n},\quad \mathrm{tr}(A)=m$ ，使得
$$
A = \mathcal{P}(\Lambda)T
$$
其中 $\mathcal{P}$ 表示删去矩阵中全为 $0$ 的行。

这里可以理解为 $T$ 表示完整的观测，而 $\mathrm{diag}(\Lambda)\in \{0,1\}^{n \times n}$ 表示对完整观测结果的**降采样**掩码，
$\mathcal{P}(\Lambda)\in \{0,1\}^{m\times n}$ 表示按照这一掩码对完整结果进行降采样。

本文解决的实际问题：稀疏视图CT重建、CT金属伪影去除、降采样MRI重建。CT 的测量值是由物体从不同方向的 X 射线投影给出的，而 MRI 的测量值则是通过检查物体在磁场作用下的傅里叶频谱获得的。重建问题即**从测量结果还原医学图像**。然而，由于获取 CT 的全正弦曲线图会对患者造成过多电离辐射，而测量 MRI 的全 k 空间又非常耗时，因此减少 CT 和 MRI 的测量次数变得非常重要。在许多情况下，只能进行**部分测量**，如稀疏视图正弦曲线和降采样 k 空间。

| 问题       | $x$  | $T$       | $A$         | $y$         |
|:---------|:-----|:----------|:------------|:------------|
| 稀疏视图CT重建 | 医学图像 | Radon变换   | 部分角度CT测量    | 稀疏视图正弦曲线    |
| CT金属伪影去除 | 医学图像 | Radon变换   | 去掉伪影区域的CT测量 | 去掉伪影区域的正弦曲线 |
| 降采样MRI重建 | 医学图像 | Fourier变换 | 低采样率的MRI测量  | 降采样k空间      |  

这里CT金属伪影去除也属于反问题的逻辑：测量结果中包括带有伪影的区域的部分，为了还原真实医学图像，需要在测量结果中去掉这一部分，只使用其余部分正弦曲线图完成还原工作。
![[Notes Images/Pasted image 20240111185920.png]]

已有工作：大多为监督学习技术，它们通过在由 CT/MRI 图像和测量数据组成的大型数据集上进行训练，学习将部分测量数据直接映射到医学图像上。这些测量结果需要通过测量过程的固定物理模型从医学图像中合成。缺点：测量过程发生变化时（比如使用不同数量的 CT 投影或不同的 MRI 采样率）就需要重新收集配对数据并重新训练模型。

## Methods

思路：用生成式模型估计医学图像的先验分布 $p(x)$ ，再结合测量分布 $p(y \mid x)$ ，由 Bayes 原理从 $p(x \mid y)$ 中采样。具体来说需要修改生成模型的生成路径使生成结果符合条件分布。

![[Notes Images/Pasted image 20240111191009.png]]

论文使用 Song Yang 的另一篇文章 SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS 中提出的分数生成模型。这是著名的扩散模型 DDPM 的推广，简单来说就是使用前向 SDE ：
$$
\mathrm{d}\mathbf{x}_t=f(t)\mathbf{x}_t\:\mathrm{d}t+g(t)\:\mathrm{d}\mathbf{w}_t,\quad t\in[0,1],
$$
将数据分布转变为噪声。其中 $f$ 和 $g$ 经过特别选择使得
$$
p_{0\boldsymbol{t}}(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_t\mid\alpha(t)\mathbf{x}_0,\beta^2(t)\boldsymbol{I})
$$
即任意时刻添加噪声后的数据服从 Gauss 分布，其中 $\alpha(t),\beta(t)$ 可以由 $f(t),g(t)$ 表示。
用反向 SDE：
$$
\mathrm{d}\mathbf{x}_t=\left[f(t)\mathbf{x}_t-g(t)^2\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)\right]\mathrm{d}t+g(t)\mathrm{d}\bar{\mathbf{w}}_t,\quad t\in[0,1],
$$
从噪声样本中逐步去除噪声得到数据样本 $x_0$ . 式中 $\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)$ 称为 Score function，需要用神经网络估计 ${s_{\boldsymbol{\theta}*}}(\mathbf{x},t)\approx\nabla_{\mathbf{x}}\log p_{t}(\mathbf{x})$ ：
$$
\boldsymbol{\theta}^{*}=\arg\min_{\boldsymbol{\theta}}\frac1N\sum_{i=1}^{N}\mathbb{E}_{t\sim\mathcal{U}[0,1]}\mathbb{E}_{\mathbf{x}_{t}^{(i)}\sim p_{0t}(\mathbf{x}_{t}^{(i)}|\mathbf{x}^{(i)})}\bigg[\left\|s_{\boldsymbol{\theta}}(\mathbf{x}_{t}^{(i)},t)-\nabla_{\mathbf{x}_{t}^{(i)}}\log p_{0t}(\mathbf{x}_{t}^{(i)}\mid\mathbf{x}^{(i)})\right\|_{2}^{2}\bigg],
$$
于是求解以下 SDE：
$$
\mathrm{d}\mathbf{x}_t=\begin{bmatrix}f(t)\mathbf{x}_t-g(t)^2s_{\boldsymbol{\theta}*}(\mathbf{x}_t,t)\end{bmatrix}\mathrm{d}t+g(t)\:\mathrm{d}\bar{\mathbf{w}}_t,\quad t\in[0,1],
$$
就能从噪声样本中生成服从原始数据分布的数据样本。例如用 Euler-Maruyama 方法求解 SDE ： 
![[Notes Images/Pasted image 20240111193935.png]]

为了求解反问题，可以将上述方法中的随机过程 $\{x_t\}_{t \in [0,1]}$ 条件化得到**条件随机过程** $\{x_t \mid y\}_{t \in [0,1]}$ .于是可以像无条件生成一样求解以下条件反向 SDE 来从 $p_0(x_0 \mid y)$ 中采样：
$$
\mathrm{d}\mathbf{x}_t=\left[f(t)\mathbf{x}_t-g(t)^2\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid\mathbf{y})\right]\mathrm{d}t+g(t)\:\mathrm{d}\bar{\mathbf{w}}_t,\quad t\in[0,1].
$$
从而得到反问题的解。其中关键在于计算 $\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid\mathbf{y})$ , 有两种思路：
（1）训练一个明确依赖于 $y$ 的 Score 模型 $s_{\theta *}(x_t,y,t)\approx \nabla_{x_t} \log p_t(x_t \mid y)$ . 这需要配对数据进行训练，有与监督学习技术相同的缺点。
（2）用无条件 Score 模型 $s_{\theta *}(x_t,t)$ 和测量分布 $p(y \mid x)$ 估计条件 Score。
例如 Song Yang 在提出 Score 生成模型时就在附录中简单介绍了这一思路（在本文中被实现用来与本文的主要方法进行比较）：
$$
\nabla_\mathbf{x}\log p_t(\mathbf{x}(t)\mid\mathbf{y})=\nabla_\mathbf{x}\log\int p_t(\mathbf{x}(t)\mid\mathbf{y}(t),\mathbf{y})p(\mathbf{y}(t)\mid\mathbf{y})\mathrm{d}\mathbf{y}(t),
$$
假设：1. $p(\mathbf{y}(t)\mid\mathbf{y})$ 是可处理的；2. $p_t(\mathbf{x}(t)\mid\mathbf{y}(t),\mathbf{y})\approx p_t(\mathbf{x}(t)\mid\mathbf{y}(t)).$ 那么
$$
\begin{aligned}
\nabla_{\mathbf{x}}\log p_{t}(\mathbf{x}(t)\mid\mathbf{y})& \approx\nabla_{\mathbf{x}}\log\int p_t(\mathbf{x}(t)\mid\mathbf{y}(t))p(\mathbf{y}(t)\mid\mathbf{y})\mathrm{d}\mathbf{y}  \\
&\approx\nabla_{\mathbf{x}}\log p_t(\mathbf{x}(t)\mid\hat{\mathbf{y}}(t)) \\
&=\nabla_\mathbf{x}\log p_t(\mathbf{x}(t))+\nabla_\mathbf{x}\log p_t(\hat{\mathbf{y}}(t)\mid\mathbf{x}(t)) \\
&\approx\mathrm{s}_{\boldsymbol{\theta}*}(\mathbf{x}(t),t)+\nabla_{\mathbf{x}}\log p_{t}(\hat{\mathbf{y}}(t)\mid\mathbf{x}(t)),
\end{aligned}
$$
其中 $\hat y(t)$ 是从 $p(y(t)\mid y)$ 中采样的。
另外还有其它一部分使用该思路的方法需要计算 $A$ 的奇异值分解，而这对于医学成像的许多测量过程来说比较困难。
（3）本文的主要方法。在无条件采样过程中**添加近端优化步骤**使得生成的样本符合条件。
在 Score 模型中有
$$
p_{0\boldsymbol{t}}(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_t\mid\alpha(t)\mathbf{x}_0,\beta^2(t)\boldsymbol{I})
$$
定义
$$
\mathbf{y}_t=A\mathbf{x}_t+\alpha(t)\epsilon
$$
则
$$
{\mathbf{y}}_{t}=\alpha(t)\mathbf{y}+\beta(t)\boldsymbol{A}\mathbf{z},\quad \mathbf{z}\sim\mathcal{N}(\mathbf{0},\boldsymbol{I})
$$
设一个现有的求解无条件反向 SDE 的迭代算法为
$$
\hat{\mathbf{x}}_{t_{i-1}}=\boldsymbol{h}(\hat{\mathbf{x}}_{t_i},\mathbf{z}_i,s_{\boldsymbol{\theta}*}(\hat{\mathbf{x}}_{t_i},t_i)),\quad i=N,N-1,\cdots,1,
$$
其中 $\mathbf{z}_i \sim \mathcal{N}(\mathbf{0},\boldsymbol{I})$ , 则本文迭代算法
$$
\begin{aligned}\hat{\mathbf{x}}_{t_i}^{\prime}&=\boldsymbol{k}(\hat{\mathbf{x}}_{t_i},\hat{\mathbf{y}}_{t_i},\lambda)\\\hat{\mathbf{x}}_{t_{i-1}}&=\boldsymbol{h}(\hat{\mathbf{x}}_{t_i}^{\prime},\mathbf{z}_i,s_{\boldsymbol{\theta}*}(\hat{\mathbf{x}}_{t_i},t_i)),\quad i=N,N-1,\cdots,1,\end{aligned}
$$
其中 $\boldsymbol{k}$ 表示求解以下**近端优化**步骤（proximal optimization step） 
$$\hat{\mathbf{x}}_{t_{i}}^{\prime}=\underset{\boldsymbol{z} \in \mathbb{R}^{n}}{\arg \min }\left\{(1-\lambda)\left\|\boldsymbol{z}-\hat{\mathbf{x}}_{t_{i}}\right\|_{\boldsymbol{T}}^{2}+\min _{\boldsymbol{u} \in \mathbb{R}^{n}} \lambda\|\boldsymbol{z}-\boldsymbol{u}\|_{\boldsymbol{T}}^{2}\right\} \quad \text { s.t. } \quad \boldsymbol{A} \boldsymbol{u}=\hat{\mathbf{y}}_{t_{i}} .$$
其中 $0\le \lambda \le 1$ 是超参数，$\|\boldsymbol{a}\|_{\boldsymbol{T}}^{2}:=\|\boldsymbol{T} \boldsymbol{a}\|_{2}^{2}$ . 这表示同时最小化 $\hat{\mathbf{x}}_{t_{i}}^{\prime}$ 与 $\hat{\mathbf{x}}_{t_i}$ 以及超平面 $\{\boldsymbol{x}\in\mathbb{R}^{n}\mid Ax=\hat{\mathbf{y}}_{t_i}\}$ 的距离，$\lambda$ 平衡两者。由 $A = \mathcal{P}(\Lambda)T$ , 该优化问题有显式解：
$$
\hat{\mathbf{x}}_{t_i}^{\prime}=T^{-1}[\lambda\boldsymbol{\Lambda}\mathcal{P}^{-1}(\boldsymbol{\Lambda})\hat{\mathbf{y}}_{t_i}+(1-\lambda)\boldsymbol{\Lambda}\boldsymbol{T}\hat{\mathbf{x}}_{t_i}+(\boldsymbol{I}-\boldsymbol{\Lambda})\boldsymbol{T}\hat{\mathbf{x}}_{t_i}],
$$
其中 $\mathcal{P}^{-1}(\Lambda)$ 表示 $\mathcal{P}(\Lambda)$ 的任何右逆。
实验中 $\lambda$ 通过在验证集上使用 Bayes 优化自动调整。
例：用该方法修改无条件的 Euler-Maruyama 迭代：
![[Notes Images/Pasted image 20240112023319.png]]
问题：这个方法好像又跟
$$
\mathrm{d}\mathbf{x}_t=\left[f(t)\mathbf{x}_t-g(t)^2\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid\mathbf{y})\right]\mathrm{d}t+g(t)\:\mathrm{d}\bar{\mathbf{w}}_t,\quad t\in[0,1].
$$
这个式子又没什么关系了？方法的理论依据不够扎实（至少文章没给出），但是实验效果确实不错。

## Experiments
![[Notes Images/Pasted image 20240112133355.png]]
![[Notes Images/Pasted image 20240112133407.png]]
![[Notes Images/Pasted image 20240112133417.png]]

## Summary

这篇文章提出了用基于分数的生成模型解决线性反问题的新方法。
优点：无监督，不需要配对数据训练；有着与监督学习方法相当甚至更优的效果；对于不同的线性测量过程修改简单，无需重新训练。