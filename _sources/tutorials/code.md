
# 代码介绍

## 代码运行要求

[详见](https://github.com/GuoLanqing/ShadowFormer)

## 代码介绍

```bash
ShadowFormer-main/
├── LICENSE
├── README.md
├── dataset.py        # 数据集加载和预处理
├── losses.py         # 定义损失函数
├── model.py          # 定义ShadowFormer模型架构
├── options.py        # 命令行参数解析
├── requirements.txt  # 项目依赖列表
├── test.py           # 测试模型的脚本
├── train.py          # 训练模型的脚本
├── doc/              # 包含项目文档的图像文件
│   ├── details.jpg
│   ├── pipeline.jpg
│   └── res.jpg
├── evaluation/       # 包含评估模型性能的MATLAB脚本
│   └── measure_shadow.m
└── utils/            # 包含各种实用工具函数
    ├── __init__.py
    ├── antialias.py          # 抗锯齿工具
    ├── bundle_submissions.py # 打包提交工具
    ├── dataset_utils.py      # 数据集工具
    ├── dir_utils.py          # 目录操作工具
    ├── image_utils.py        # 图像处理工具
    ├── loader.py             # 加载工具
    └── model_utils.py        # 模型操作工具
└── warmup_scheduler/  # 包含热身调度器相关代码
    ├── __init__.py
    ├── run.py
    └── scheduler.py
```

## 主要文件主要代码分析

### model.py主要函数

**定义了ShadowFormer模型的架构，以下是其主要组成部分：**

```{tip}
1. ShadowFormer 类: 这是模型的主体类，其中定义了ShadowFormer网络的结构。
   - `__init__` 方法初始化模型，设置各层的参数，如图像大小(`img_size`), 输入通道(`in_chans`), 嵌入维度(`embed_dim`), 层数(`depths`), 多头注意力的头数(`num_heads`), 窗口大小(`win_size`), MLP比率(`mlp_ratio`), 是否使用通道注意力(`qkv_bias`), dropout 率等。
   - 定义了编码器(encoder)和解码器(decoder)的多个层，以及输入和输出的投影层。
   - `forward` 方法定义了模型的前向传播逻辑，即如何处理输入数据并生成输出。
2. 其他辅助层: 如`InputProj`和`OutputProj`类，用于处理模型输入和输出的投影。
3. 通道注意力(Channel Attention) 和 多层感知机(MLP) 模块: 在模型的多个位置使用，以增强特征表示。
```

### train.py 主要函数

**用于训练ShadowFormer模型的脚本，以下是其主要功能：**

```{tip}
1. 训练循环: 包含模型训练的完整流程，从数据加载到反向传播和参数更新。
   - 初始化模型、优化器、损失函数等。
   - 加载训练数据集，并在每个epoch遍历数据。
   - 对于每个batch数据，执行模型的前向传播、计算损失、执行反向传播、更新模型参数。
2. 模型评估: 在验证集上评估模型性能，通常使用PSNR等指标。
3. 日志记录: 将训练过程中的损失和性能指标记录到日志文件中。
4. 模型保存: 在训练过程中，按一定条件保存模型的最佳状态和最新状态。
```

### ShadowFormer核心代码

```python
class ShadowFormer(nn.Module):
    def __init__(self, img_size=256, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff', se_layer=True,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

    def forward(self, x, xm, mask=None):
        # Input  Projection
        xi = torch.cat((x, xm), dim=1)
        self.img_size = (x.shape[2], x.shape[3])
        y = self.input_proj(xi)
        y = self.pos_drop(y)

        #Encoder
        conv0 = self.encoderlayer_0(y, xm, mask=mask, img_size = self.img_size)
        pool0 = self.dowsample_0(conv0, img_size = self.img_size)
        m = nn.MaxPool2d(2)
        xm1 = m(xm)
        self.img_size = (int(self.img_size[0]/2), int(self.img_size[1]/2))
        conv1 = self.encoderlayer_1(pool0, xm1, mask=mask, img_size = self.img_size)
        pool1 = self.dowsample_1(conv1, img_size = self.img_size)
        m = nn.MaxPool2d(2)
        xm2 = m(xm1)
        self.img_size = (int(self.img_size[0] / 2), int(self.img_size[1] / 2))
        conv2 = self.encoderlayer_2(pool1, xm2, mask=mask, img_size = self.img_size)
        pool2 = self.dowsample_2(conv2, img_size = self.img_size)
        self.img_size = (int(self.img_size[0] / 2), int(self.img_size[1] / 2))
        m = nn.MaxPool2d(2)
        xm3 = m(xm2)

        # Bottleneck
        conv3 = self.conv(pool2, xm3, mask=mask, img_size = self.img_size)

        #Decoder
        up0 = self.upsample_0(conv3, img_size = self.img_size)
        self.img_size = (int(self.img_size[0] * 2), int(self.img_size[1] * 2))
        deconv0 = torch.cat([up0,conv2],-1)
        deconv0 = self.decoderlayer_0(deconv0, xm2, mask=mask, img_size = self.img_size)

        up1 = self.upsample_1(deconv0, img_size = self.img_size)
        self.img_size = (int(self.img_size[0] * 2), int(self.img_size[1] * 2))
        deconv1 = torch.cat([up1,conv1],-1)
        deconv1 = self.decoderlayer_1(deconv1, xm1, mask=mask, img_size = self.img_size)

        up2 = self.upsample_2(deconv1, img_size = self.img_size)
        self.img_size = (int(self.img_size[0] * 2), int(self.img_size[1] * 2))
        deconv2 = torch.cat([up2,conv0],-1)
        deconv2 = self.decoderlayer_2(deconv2, xm, mask=mask, img_size = self.img_size)

        # Output Projection
        y = self.output_proj(deconv2, img_size = self.img_size) + x
        return y

```
