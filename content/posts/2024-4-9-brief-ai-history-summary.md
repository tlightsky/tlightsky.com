+++
title = "浅谈AI的历史"
date = 2024-04-09
description = "本文会简单介绍下个人对于AI历史的理解"

[taxonomies]
tags = ["AI"]
+++

本文会简单介绍下个人对于AI历史的理解，
主要着重于目前已经产生较为广泛的影响并且有较好的创新性（Novel）的工作的方面，
尝试对于目前AI发展的脉络进行梳理

# 时间线

* [AlextNet 2012.9.30 ImageNet Chanllenge](https://en.wikipedia.org/wiki/AlexNet)
    * [papper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
* [VAE 2012.12.20 Auto-Encoding Variational Bayes
](https://en.wikipedia.org/wiki/Variational_autoencoder)
    * [papper](https://arxiv.org/abs/1312.6114)
* [U-Net 2014.11.14 Fully Convolutional Networks for Semantic Segmentation
](https://en.wikipedia.org/wiki/U-Net)
    * [papper](https://arxiv.org/abs/1411.4038)
* [ResNet 2015.12.10](https://en.wikipedia.org/wiki/Residual_neural_network) 
    * [papper](https://arxiv.org/abs/1512.03385)
* [Transformer 2017.6.12 Attention Is All You Need](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need) 
    * [papper](https://arxiv.org/abs/1706.03762)
    * [read](https://zhuanlan.zhihu.com/p/338817680)
* [BERT 2018.10.11 Pre-training of Deep Bidirectional Transformers for Language Understanding](https://en.wikipedia.org/wiki/BERT_(language_model))
    * [papper](https://arxiv.org/abs/1810.04805v2)
* [GPT-2 partially:2019.2, fully:2019.11.5 Language Models are Unsupervised Multitask Learners
](https://en.wikipedia.org/wiki/GPT-2)
    * [repo](https://github.com/openai/gpt-2)
    * [papper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [Vit 2020.10.22 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://en.wikipedia.org/wiki/Vision_transformer)
    * [papper](https://arxiv.org/abs/2010.11929)
    * [read](https://zhuanlan.zhihu.com/p/412910412)
* [DALL-E/CLIP 2021.2.26](https://en.wikipedia.org/wiki/DALL-E)
    * [repo](https://github.com/OpenAI/CLIP)
    * [papper](https://arxiv.org/abs/2103.00020)
* [MAE 2021.11.11 Masked Autoencoders Are Scalable Vision Learners
](https://www.zhihu.com/question/498364604/answer/3337675217)
    * [papper](https://arxiv.org/abs/2111.06377)
    * [read](https://zhuanlan.zhihu.com/p/439554945)
* [Stable Diffusion 2022.8.22](https://en.wikipedia.org/wiki/Stable_Diffusion)
    * [pappers](https://en.wikipedia.org/wiki/Stable_Diffusion#Releases)
* [LLaMA2](https://en.wikipedia.org/wiki/LLaMA)
    * [Llama-Chinese](https://github.com/LlamaFamily/Llama-Chinese)
    * [read](https://zhuanlan.zhihu.com/p/636784644)
    * [mini LLaMA](https://zhuanlan.zhihu.com/p/652664029)

# 深度神经网络的再崛起，起源于视觉
一切的起源还需要追述到李飞飞创建ImageNet，并且举办挑战赛说起。

### [AlexNet](https://zhuanlan.zhihu.com/p/618545757f)(ImageNet2012)
AlexNet当时参加了2012年的挑战赛，并且一举大幅减少了图像识别的错误率。（26.2% -> 15.3%）（Alex，Ilya，Hinton三个作者都是大神）

具体的可以通过沐神的[这个视频](https://www.zhihu.com/zvideo/1432155856322920448)来了解下。

主要的点：
* 观察到使用更深的网络效果会更好
* 使用卷积神经网络
* 使用GPU
* 使用较多的参数（60M）
* DropOut防止过拟合

### [ResNet](https://zhuanlan.zhihu.com/p/101332297) (ImageNet2015)

* 问题：深的神经网络很难训练，比如会出现梯度爆炸或消失的问题无法收敛
* 结果：使用残差神经网络使得网络更容易训练收敛，让更深的网络效果更好得以实现

#### 思路：
* 即使新层都是identity mapping，效果不应该更差
* 新的层不再直接学习X，而是学习 H(X)-X，即是学习残差

# 从有监督到无监督

### [RNN](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html#id4) Seq2Seq 的Encoder/Decoder使用RNN来作为序列的预测器

### [Transformer](https://www.zhihu.com/zvideo/1437034536677404672) (Attention Is All You Need)
(CNN / RNN 终结者？)

* 把Encoder/Decoder 中 RNN/CNN 转为只使用 Attention
* 对比RNN
    * RNN的H(t)需要依赖H(t-1)的结果，难以并行
    * 深度较深之后，前后关联性会消失
* 对比CNN
    * 较难跨较大区域进行关联
    * CNN有多输出通道，可以同时关注多个区域（MultiHead来源）
* Attention
    * 可以并行
    * 更好的结果
    * 前身[Seq2Seq](https://zh.d2l.ai/chapter_recurrent-modern/seq2seq.html#sec-seq2seq)，引入了Encoder/Decoder+RNN
        * Encoder使用RNN依次将词的影响嵌入到隐藏层中
        * Decoder使用Encoder的state作为上下文输入，再结合每个步骤已产生的内容，来预测下一个字符
        * 编码器与通过```<eos>```结束，解码器开始为```<bos>```
    * [Attention](https://zh.d2l.ai/chapter_attention-mechanisms/attention-cues.html)
        * Q（查询，意志线索）x K（键，非意志线索）x V（值，感觉输入） 
        * Q x K (注意力评分函数，如高斯核)
        * [加性注意力](https://zh.d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html#subsec-additive-attention)，注意力评分函数中具备Wq，Wk的可学参数
    * [SelfAttention](https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html)，看起来是QKV都等于X，只学一个Wq，Wk？
    * [MultiHead](https://zh.d2l.ai/chapter_attention-mechanisms/multihead-attention.html): 因为没有使用Additive Attention，将QKV以可学习参数W投影到低纬度进行Attention计算，再加和，使得反向传播的时候可以学习到不同的主题
    * [Softmax](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html)，转换成概率分布（人话，和为1）
    * 位置编码

* 根据Karpathy的[分享](https://www.youtube.com/watch?v=zjkBMFhNj_g&t=4s)，只是预测下个单词的Transformer其实没有办法完成问答，需要一些User/Assitant的预料进行第一步的FineTune，再经过HFML进行第二步的强化学习的过程
* 较为适应翻译类型的任务

### GPT 2018/06
* 先训练通用的预测下个语句的模型
* 再进行FineTune，即训练一个线性层对于下游任务进行适配，分类，蕴含，相似，多选

### BERT 2018/10
* encoder only
* 受GPT启发
* 数据集是GPT 4倍数据，参数大三倍

### [GPT-2 2019/02](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) （Language Models are Unsupervised Multitask Learners）

### GPT-3 2020/05

# 弱监督的视觉模型

### CLIP

# 3D

### NeRF
  * 体渲染隐式表达，空间每个点的不同方向，透明度，颜色

### 3DGS
  * 高斯表达


[MLi ReadPapper](https://github.com/mli/paper-reading)
