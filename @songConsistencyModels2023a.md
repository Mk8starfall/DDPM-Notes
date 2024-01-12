---
Type: article
Authors: Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever
Year: 2023
Title: Consistency Models
Journal: 
Pages: 
DOI: 
Publisher: arXiv
Place: 
---

# Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever (2023) - Consistency Models
[==*Read it now! See in Zotero*==](zotero://select/items/@songConsistencyModels2023a)
**Web:** [Open online](http://arxiv.org/abs/2303.01469)
**Citekey:** songConsistencyModels2023a
**Tags:** #source, (type), (status), (decade)

## Abstract

Diffusion models have significantly advanced the fields of image, audio, and video generation, but they depend on an iterative sampling process that causes slow generation. To overcome this limitation, we propose consistency models, a new family of models that generate high quality samples by directly mapping noise to data. They support fast one-step generation by design, while still allowing multistep sampling to trade compute for sample quality. They also support zero-shot data editing, such as image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks. Consistency models can be trained either by distilling pre-trained diffusion models, or as standalone generative models altogether. Through extensive experiments, we demonstrate that they outperform existing distillation techniques for diffusion models in one- and few-step sampling, achieving the new state-of-the-art FID of 3.55 on CIFAR-10 and 6.20 on ImageNet 64x64 for one-step generation. When trained in isolation, consistency models become a new family of generative models that can outperform existing one-step, non-adversarial generative models on standard benchmarks such as CIFAR-10, ImageNet 64x64 and LSUN 256x256.
扩散模型极大地推动了图像、音频和视频生成领域的发展，但它们依赖于迭代采样过程，导致生成速度缓慢。为了克服这一局限，我们提出了一致性模型，这是一种通过直接将噪声映射到数据来生成高质量样本的新型模型系列。这些模型在设计上支持一步快速生成，同时仍允许多步采样，以计算量换取样本质量。它们还支持零镜头数据编辑，如图像着色、着色和超分辨率，而不需要对这些任务进行明确的训练。一致性模型既可以通过提炼预训练的扩散模型来训练，也可以作为独立的生成模型来训练。通过大量实验，我们证明一致性模型在单步和少步采样中的表现优于现有的扩散模型蒸馏技术，在 CIFAR-10 和 ImageNet 64 x 64 上，一致性模型的单步生成 FID 分别达到了 3.55 和 6.20 的最新水平。在单独训练时，一致性模型成为一个新的生成模型系列，可以在 CIFAR10、ImageNet 64 x 64 和 LSUN 256 x 256 等标准基准上超越现有的一步非对抗生成模型。
## Backgrounds

