# 论文精读

<span style="font-size: 30pt; color: purple">ShadowFormer: Global Context Helps Shadow Removal</span>

## 前言科普

### 什么是阴影去除

阴影往往会给图像带来不必要的复杂性和混淆，影响图像的视觉质量和内容理解。

阴影去除（Shadow Removal）是一种`图像处理技术`，旨在从图像中消除或减弱由光照条件引起的阴影。阴影去除的目标是通过分析图像中的`颜色、亮度和纹理`等特征，将阴影部分与其他图像区域进行区分，并尽可能恢复原始图像中的信息。

### 专业术语科普

#### 模型相关术语优化整理

**1.Transformer**

概述: Transformer是一种深度学习模型，于2017年提出，主要用于处理序列数据，特别在自然语言处理（NLP）领域表现卓越。
自注意力机制: 这是Transformer的核心特性，它使得模型能够在序列的不同位置之间建立直接的依赖关系，有效处理长距离依赖问题。
查询、键、值: 对于序列中的每个元素，模型都会生成三个向量：查询（Q），键（K）和值（V）。
注意力权重: 通过计算查询和键之间的相似度来分配给值的注意力权重。
加权求和: 最后，将注意力权重应用于值，对其进行加权求和，以得到每个位置的输出。
多头注意力: 通过并行运用多个注意力机制，多头注意力能够让模型在不同的表示子空间中捕获序列的不同特征。

**2.注意力机制**

概述: 一种让模型动态聚焦于输入数据中最重要部分的技术，适用于多个领域，包括NLP和计算机视觉。
自注意力: 一种特别的注意力机制，允许模型在同一序列内部的不同位置之间建立联系。
优势: 注意力机制具有灵活性、高效的并行处理能力和增强的模型解释性。

**3.编码器与解码器**

编码器: 在Transformer架构中处理输入序列，将其转化为中间表示。
解码器: 使用编码器的输出以及先前的输出来生成最终的输出序列。
图像处理相关术语优化整理

**4.MLP（多层感知机）**

概述: 多层感知机（MLP）是一种基础的神经网络结构，由多个层次的感知器（简单的神经元模型）构成。它通过一个或多个隐藏层来学习数据的非线性特征，广泛应用于各种机器学习任务中。

**5.卷积神经网络（CNN）**

概述: 卷积神经网络（CNN）是一种专门用于处理具有网格结构数据（如图像）的深度学习模型。它通过卷积层来提取数据的空间特征，非常适合于图像识别、图像分类等视觉任务。
接受域（Receptive Field）: 接受域是指在CNN中，输出特征图上的一个元素对应的原始输入数据的区域大小。它是理解和分析卷积神经网络的重要概念，反映了网络对局部特征的敏感性。

#### 图像去噪相关术语优化整理

**1.图像超分辨率:**

通过算法从低分辨率图像中生成高分辨率图像的过程。

**2.图像去模糊:**

从模糊图像中恢复清晰图像的技术，涉及去除运动模糊或焦点模糊。

**3.峰值信噪比 (PSNR):**

衡量图像质量的指标，用于比较原始图像和压缩或重构后图像的相似度。

**4.伪影:**

图像处理中由算法引起的非原始图像的任何异常或不期望的特征。

**5.照度与照度一致性:**

照度: 光照强度在特定表面上的分布，以勒克斯（lux）度量。
照度一致性: 在图像处理中指去除图像中的阴影时保持光照强度在空间上的连续性和一致性。

**6.上采样与下采样**

上采样: 增加图像的像素数，使图像变得更大，通常通过插值算法实现。

下采样: 减少图像的像素数，使图像变得更小，通常通过丢弃像素或合并邻近像素来实现。

### 传统方法去除阴影

#### 传统方法概述

**1.物理模型方法：**

这种方法基于对光照和阴影生成机制的物理理解。它通常涉及到识别图像中的光照条件和表面特性，然后相应地调整像素值以减少阴影效果。

**2.图像分割和分类：**

这种方法通过图像分割技术将阴影区域与非阴影区域分开。然后，可以对这些区域分别进行处理。

**3.频域方法：**

这种方法涉及到将图像从空间域转换到频域（例如使用傅里叶变换），在频域中处理阴影，然后再转换回空间域。这种方法可以帮助识别和去除图像中的周期性阴影模式。

**4.修复和填充技术：**

在某些情况下，去除阴影可能会在图像中留下“空白”区域。这种情况下可以使用图像修复技术，如Inpainting，来填补这些区域，使图像看起来更加自然和连贯。

**5.机器学习和深度学习：**

近年来，基于机器学习的方法，特别是深度学习，已经在图像去阴影方面取得了显著的进展。使用如卷积神经网络（CNN）的模型可以学习从大量数据中识别和去除阴影的模式。

#### 其他方法的弱点

1. 现有的阴影去除方法大多集中在局部处理阴影和非阴影区域，导致阴影边界周围存在严重的伪影，并且阴影区域与非阴影区域之间的照明不一致。
2. 部分方法在一定程度上减轻了边界痕迹，但却需要多个具有巨大计算开销的后处理模块，对算力要求高。
3. 许多深阴影去除算法都无法保持恢复图像的照度和颜色一致性。

## 论文解读

### 该方法的创新点及贡献

1. 扩展了经典的 `Retinex` 阴影模型来模拟阴影退化，利用`非阴影区信息`帮助阴影区域的恢复
2. 提出了一种基于`多尺度通道注意`框架的新的`单阴影去除变压器 (ShadowFormer)`
3. 引入具有`阴影相互作用注意力 (SIA)`的阴影相互作用模块 (SIM)，有效利用阴影区域和非阴影区域在空间和通道维度上的`全局上下文相关性`

### 基于retinex的阴影模型

- 经典原模型：

$$
\mathbf{I}_{s}=\mathbf{I}_{m}\odot\mathbf{I}_{s}+(1-\mathbf{I}_{m})\odot\mathbf{I}_{ns}
$$
>其中$\mathbf{I}_{ns}$和$\mathbf{I}_{m}$分别表示非阴影区域，掩码表示阴影区域。$\odot$表示逐元素乘法

- 引入的模型：

$$
\mathbf{I}_{sf}=\mathbf{L}_{sf}\odot \mathbf{R}_{o}
$$

- 基于人类的感知，图像$\mathbf{I}$可以分解为照度$\mathbf{L}$和反射率$\mathbf{R}$
引入上述公式后，改进后的模型为：

$$
\mathbf{I}_{s}=\mathbf{I}_{m}\odot\mathbf{L}_{s}\odot\mathbf{R}+(1-\mathbf{I}_{m})\odot\mathbf{L}_{ns}\odot\mathbf{R}
$$

>$\mathbf{I}_{s}、\mathbf{I}_{ns}和\mathbf{I}_{sf}$分别表示阴影区域、非阴影区域、无阴影图像的光照

- 基于此模型的改进，我们可以得出：
  1. 虽然光照退化在阴影和非阴影区域不同，但底层图像却拥有相同的照度。阴影和非阴影区域的照度一致性不可忽略，否则会导致恢复图像的照度在空间上是变化的
  2. 阴影区域和非阴影区域都捕获相同的底层R，两个区域之间有很强的全局上下文关联性。

- 基于此启发：
  1. 模型的 receptive feld 应该尽可能大，以捕获全局信息。
  2. 来自非阴影区域的光照信息是阴影区域恢复的重要先验信息。

### 基于变压器的网络——ShadowFormer

#### 总体架构

```{figure} ../images/paper/1.png
:width: 1000px
:align: center

模型结构
```

- **输入**：网络接收两个输入，原始影子图像 **I<sub>s</sub>** 和对应的掩码图像 **I<sub>m</sub>** ，掩码图像用于区分影子区域和非影子区域。
- **线性投影**（Linear Projection）：进行线性投影，将输入数据映射到一个低维度的特征空间或进行特征提取。
- **编码器**：利用通道注意力变换器(encoder)基于原始影子图像提取层次化信息。编码器通过通道注意力（Channel Attention, CA）模块，可以专注于影像的特定通道，提取重要特征。
- **阴影交互模块(SIM)**：SIM利用非影子区域和影子区域之间的上下文信息，通过空间和通道维度帮助恢复影子区域。
- **解码器**：解码器使用一系列通道注意力模块，根据编码器提供的层次化信息重构无影图像。
- **输出**：经过编码器、SIM和解码器的处理后，得到的无影图像 **I** 。

#### 输入部分详解

```{figure} ../images/paper/2.png
:width: 1000px
:align: center

输入
```

1. $I_s$（阴影图像）：

  这是网络的输入，即一张包含阴影的图片。

  ```{admonition} code
  :class: note
  对应模型函数（代码）：的`forward` 的输入参数 `x`，即网络的输入图像。
  ```

2. $I_m$（掩模图像）：

  这是一个二值图像，用于标识原始图像中哪些区域是阴影区域，哪些是非阴影区域。阴影区域通常用较暗的像素表示。

  ```{admonition} code
  :class: note
  对应于模型函数（代码）： `forward` 的输入参数 `xm`。这个掩模用于指示输入图像中哪些部分属于阴影。
  ```

#### 线性投影部分详解

```{figure} ../images/paper/3.png
:width: 1000px
:align: center

线性投影
```

这是网络的第一步，它将输入图像的每个像素点转换成一系列数字（这个过程称为特征嵌入），使得网络能够更好地处理这些数据。

```{admonition} code
:class: note
`LinearProjection` 在代码中通过 `InputProj` 类实现，它使用 `nn.Conv2d` 来转换输入图像的像素到高维特征空间。
```

#### 编码器和解码器详解

```{figure} ../images/paper/4.png
:width: 1000px
:align: center

ENCODER DECODER
```

**编码器和解码器中的信道注意模块（CA）：**

```{figure} ../images/paper/5.png
:width: 150px
:align: center

CA模块
```

每个编码器和解码器由`L个CA模块`（如图所示）组成，以叠加多尺度全局特征。每个CA模块包括两个CA块，以及编码器中的下采样层或解码器中的上采样层。 CA块依次通过CA压缩空间信息，并通过前馈MLP捕获远程相关性。

$$\begin{gathered}
\mathbf{\tilde{X}}=\mathrm{CA}(\mathrm{LN}(\mathbf{X}))+\mathbf{X} \\
\hat{\mathbf{X}}=\mathrm{GELU}(\mathrm{MLP}(\mathrm{LN}(\mathbf{X})))+\mathbf{\tilde{X}} 
\end{gathered}$$

> $\mathrm{LN}$表示层归一化，$\mathrm{GELU}$表示$\mathrm{GELU}$激活层，$\mathrm{MLP}$表示多层感知器
经过编码器后的${L}$个模块后，我们可以得到层次特征：

$$
\{\mathbf{X}_{1},\mathbf{X}_{2},\ldots,\mathbf{X}_{L}\},\mathrm{where~}\mathbf{X}_{L}\in\mathbb{R}^{2^{L}C\times\frac{H}{2^{L}}\times\frac{W}{2^{L}}}
$$

```{admonition} 编码器对应代码：
:class: note


在编码器部分，代码通过多个层级（`encoderlayer_0`、`encoderlayer_1`、`encoderlayer_2`）进行特征提取和下采样。每一层都使用 `BasicShadowFormer` 类，其中包含了自注意力机制和MLP（多层感知机）。

为了降低空间分辨率，每个编码器层后跟随一个 `Downsample` 模块，用于减少特征图的尺寸。

```

```{admonition} 解码器对应代码：
:class: note

解码器部分逆转了编码器的操作。它通过 `decoderlayer_0`、`decoderlayer_1` 和 `decoderlayer_2` 类进行特征上采样和重建。这些层同样使用 `BasicShadowFormer` 类。

上采样是通过 `Upsample` 类实现的，该类使用转置卷积（`nn.ConvTranspose2d`）来增加特征图的尺寸。

```

#### 阴影交互模块SIM详解

```{figure} ../images/paper/6.png
:width: 1000px
:align: center

SIM模块
```

**阴影交互模块SIM以及阴影交互注意SIA：**

```{figure} ../images/paper/7.png
:width: 1000px
:align: center

SIA模块
```

这是网络的核心部分，它通过分析非阴影区域与阴影区域的联系来提升对阴影区域的理解。这样做可以帮助网络更好地预测阴影区域应该是什么样子，以便在最后的输出中去除阴影。

左图(b)展示了SIM的详细架构，由两个模块组成，每个模块对应公式如下：

$$
\begin{aligned}\tilde{\mathbf{X}}&=\mathrm{SIA}(\mathrm{CA}(\mathrm{LN}(\mathbf{X})),\mathbf{\Sigma})+\mathbf{X}\\\hat{\mathbf{X}}&=\mathrm{GELU}(\mathrm{MLP}(\mathrm{LN}(\mathbf{X})))+\tilde{\mathbf{X}}\end{aligned}
$$

SIA部分的具体公式如下：

$$
\mathrm{SIA}(\mathbf{X},\mathbf{\Sigma})=\mathrm{softmax}(\frac{\mathbf{Q}\mathbf{K}^T}d)\mathbf{V}[\sigma\mathbf{\Sigma}+(1-\sigma)\mathbf{1}]
$$

可见SIA接受两个参数X和sigma：

- 参数X来源：

```{figure} ../images/paper/8.png
:width: 650px
:align: center


```

```{figure} ../images/paper/9.png
:width: 300px
:align: center

```

先进行LN层的归一化特征映射，再通过CA来重新加权通道。将重新加权后的特征映射拆分为一系列不重叠的窗口，Q、K和V表示输入特征映射X的投影查询、键和值 。

- 参数sigma来源：

```{figure} ../images/paper/10.png
:width: 650px
:align: center


```

```{figure} ../images/paper/11.png
:width: 300px
:align: center


```

池化特征图XL的一个位置向量可以对应输入阴影图像中的一个 patch。同时，对阴影掩码$I_m$应用max-pooling到与池化特征图$X_L$相同的空间维度中，记为M。根据阴影掩码，阴影图像中的patch可以分为阴影patch和非阴影patch，分别记为1和 0。直观地，我们可以利用非阴影和阴影区域之间的逐块相关图Σ，如下所示：

$$
\Sigma^{ij}=\mathbf{M}^{i}\oplus\mathbf{M}^{j}\quad\forall i,j
$$

将参数X和参数sigma代入SIM部分的公式中，即可获得SIM部分的输出。

最后，我们应用线性投影来获得残差图像I_r，最后的输出由:

$$
\mathrm{I=I_s+I_r}
$$

得到损失函数如下：

$$
\mathcal{L}(\mathbf{I}_{gt},\hat{\mathbf{I}})=\begin{Vmatrix}\mathbf{I}_{gt}-\hat{\mathbf{I}}\end{Vmatrix}
$$

```{admonition} SIM对应代码：
:class: note

SIM 通过 `SIMTransformerBlock` 类实现。这个类中使用了 `WindowAttention` 来实现基于局部窗口的多头自注意力。特别地，它通过循环移位操作（cyclic shift）和掩码处理来实现对不同窗口区域的注意力聚焦。
```
