# Swin Transformer
# 输入图像最好为224x
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


# W-MSA/SW-MSA中,将feature map按照window_size划分成一个个没有重叠的window
def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    """
    B, H, W, C = x.shape
    # [B, H, W, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 调换维度permute
    # [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # 强制数据连续
    # [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = windows.view(-1, window_size, window_size, C)
    return windows  # [num_windows*B, window_size, window_size, C]


# SW-MSA中,将一个个window还原成一个feature map
def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.view(B, H, W, -1)
    return x  # [B, H, W, C]


# 将图片划分为不重叠的patches
# Patch Partition + Linear Embedding
class Patch_Embedding(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_channel=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_channel
        self.embed_dim = embed_dim
        # 卷积层kernel_size=stride, 代替划分
        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 如果norm_layer==None则此层为线性映射nn.Identity()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        # 判断是否填充：如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        # Padding
        pad_flag = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_flag:
            '''
            在H, W后三个维度填充
            参数含义：(W_left, W_right, H_top,H_bottom, C_front, C_back)
            见https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            '''
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1], 
                            0, self.patch_size[0] - H % self.patch_size[0], 
                            0, 0))
        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # 展平处理 flatten: [B, C, H, W] -> [B, C, HW]
        x = x.flatten(2)
        # 调换维度 transpose: [B, C, HW] -> [B, HW, C]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# 下采样
# Patch Merging
class Patch_Merging(nn.Module):
    """ Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        # 最后Linear
        self.FC_reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        # 传入x只知道H*W. x: [B, H*W, C]
        B, L, C = x.shape
        # 判断是否合法
        assert L == H * W, "input feature has wrong size"
        # 展开：只能使得(H,W)在第(1,2)维度, C只能在第
        x = x.view(B, H, W, C)
        # 判断是否填充：如果输入feature map的H，W不是2的整数倍，需要进行padding
        # Padding
        pad_flag = (H % 2 == 1) or (W % 2 == 1)
        if pad_flag:
            '''
            在H, W后三个维度填充. 从最后一个维度向前移动填充
            参数含义：(C_front, C_back, W_left, W_right, H_top, H_bottom)
            注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            见https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            '''
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        # 划分为4个(h/2, w/2)大小的填充窗口
        # [B, H, W, C] -> [B, H/2, W/2, C]
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        # 在第3(-1)维进行拼接
        # [B, H/2, W/2, C] -> [B, H/2, W/2, 4*C]
        x = torch.cat([x0, x1, x2, x3], -1)
        # 展平H,W
        # [B, H/2, W/2, 4*C] -> [B, H/2*W/2, 4*C]
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        # 全连接
        # [B, H/2*W/2, 4*C] -> [B, H/2*W/2, 2*C]
        x = self.FC_reduction(x)
        return x


# SwinT Block中最后一层的MLP Layer
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.FC1 = nn.Linear(in_features, hidden_features)
        # GELU激活函数
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.FC2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.FC1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.FC2(x)
        x = self.drop(x)
        return x


# 实现W-MSA(Window based multi-head self attention) / SW-MSA(不包括调整mask行列): 结构类似ViT的Ecoder Block
class WindowAttention(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个head的维度
        self.scale = head_dim ** -0.5  # Attention公式的系数中 \dfr{1}{\sqrt{d}}

        # relative position bias table: 初始化Attention公式中的relative position bias的可训练参数
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, num_heads]
        
        ##################################################
        # 构建相对位置引索(relative position index)的矩阵  #
        ##################################################
        # 初始化：为window中的每个token获取成对的相对位置引索
        coords_h = torch.arange(self.window_size[0])  # [0, 1, 2, ..., Mh-1]
        coords_w = torch.arange(self.window_size[1])  # [0, 1, 2, ..., Mw-1]
        # 这里 indexing="ij" 运行时出错，可以去掉
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        # 展平：[2, Mh, Mw] -> [2, Mh*Mw]
        ''' 绝对位置索引
        coords_flatten[0]对应feature map中每个像素对应的行标
        coords_flatten[1]对应feature map中每个像素对应的列标
        '''
        coords_flatten = torch.flatten(coords, 1)
        ''' 得到二维相对位置索引
        广播运算(统一两个数组的维度): [2, Mh*Mw, 1] - [2, 1, Mh*Mw] -> [2, Mh*Mw, Mh*Mw]
        '''
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # 维度调换: [2, Mh*Mw, Mh*Mw] -> [Mh*Mw, Mh*Mw, 2]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # 二元索引变为一元索引
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        ''' 得到一维相对位置索引
        [Mh*Mw, Mh*Mw, 2] -> [Mh*Mw, Mh*Mw]
        '''
        relative_position_index = relative_coords.sum(-1)
        # 放入模型缓存
        self.register_buffer("relative_position_index", relative_position_index)


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 使用全连接层得到qkv
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # Wo：最后得到结果进行拼接，使用Wo进行映射
        self.proj_drop = nn.Dropout(proj_drop)
        # 初始化
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        # dim=-1: 针对每行进行softmax操作
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        qkv = self.qkv(x)
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # 通过切片拆分成qkv
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # torchscrpt不能将tensor用作tuple

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        q = q * self.scale
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        attn = (q @ k.transpose(-2, -1))

        ''' 展平relative_position_index后, index从bias_table中取出对应bias
        relative_position_bias_table -> B_relative_position_bias: [Mh*Mw*Mh*Mw,num_heads] -> [Mh*Mw,Mh*Mw,num_heads]
        '''
        B_relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        # permute: [Mh*Mw,Mh*Mw,num_heads] -> [num_heads, Mh*Mw, Mh*Mw]
        B_relative_position_bias = B_relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, Mh*Mw, Mh*Mw]
        # 加入偏置B
        # [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw] + [1, num_heads, Mh*Mw, Mh*Mw]
        attn = attn + B_relative_position_bias.unsqueeze(0)

        if mask is not None:  # SW-MSA
            # mask: [num_windows, Mh*Mw, Mh*Mw]
            num_windows = mask.shape[0]  # num_windows
            # attn.view: [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw] -> [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            # mask.unsqueeze: [num_windows, Mh*Mw, Mh*Mw] -> [1, num_windows, 1, Mh*Mw, Mh*Mw]
            mask = mask.unsqueeze(1).unsqueeze(0)
            # 不同区域mask = -100, 相同区域mask无影响
            attn = attn + mask
            # [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw] -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
            attn = attn.view(-1, self.num_heads, N, N)
            # 不同区域softmax = 0
            attn = self.softmax(attn)
        else:  # W-MSA
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        x = (attn @ v)
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        x = x.transpose(1, 2)
        # 拼接多头中最后两个维度的信息
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = x.reshape(B_, N, C)
        # 全连接层进行映射
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Swin Transformer Block
class SwinTransformerBlock(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self,
                dim,
                num_heads,
                window_size=7,
                shift_size=0,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0 ~ window_size"
        self.norm_layer1 = norm_layer(dim)
        # 构建W-MSA/SW-MSA
        self.attn = WindowAttention(dim,
                                    window_size=(self.window_size, self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_layer2 = norm_layer(dim)
        # 构建MLP
        self.mlp = MLP(in_features=dim,
                        hidden_features=int(dim * mlp_ratio),
                        act_layer=act_layer,
                        drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W  # BasicLayer中传入
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # 捷径分支
        shortcut = x  # [B, H*W, C]
        x = self.norm_layer1(x)
        # [B, H*W, C] -> [B, H, W, C]
        x = x.view(B, H, W, C)

        # 对feature map给pad到window_size的整数倍
        pad_left = pad_top = 0
        pad_right = (self.window_size - W % self.window_size) % self.window_size
        pad_bottom = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_left, pad_right, pad_top, pad_bottom))
        _, Hp, Wp, _ = x.shape

        # 若SW-MSA,需要移动(shift)feature map对不同窗口来交互信息
        if self.shift_size > 0:  # 针对SW-MSA
            # 移动上shift_size行和左shift_size列后
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:  # W-MSA
            shifted_x = x
            attn_mask = None

        # 划分窗口
        # [B, Hp, Wp, C] -> [num_windows*B, Mh, Mw, C]
        x_windows = window_partition(shifted_x, self.window_size)
        # 将宽高展平: [num_windows*B, Mh, Mw, C] -> [num_windows*B, Mh*Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # 得到W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [num_windows*B, Mh*Mw, C]

        # 融合窗口
        # [num_windows*B, Mh*Mw, C] -> [num_windows*B, Mh, Mw, C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # [num_windows*B, Mh, Mw, C] -> [B, Hp, Wp, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # 若SW-MSA,需要还原(reverse)移动(shift)feature map
        if self.shift_size > 0:  # 针对SW-MSA
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 移除pad数据
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        # [B, H, W, C] -> [B, H*W, C]
        x = x.view(B, H * W, C)

        # Feed-Forward Networks 前馈神经网络
        x = shortcut + self.drop_path(x)
        shortcut = x
        x = self.norm_layer2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)
        return x


# 编写每个Stage的每个Block
class BasicLayer(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self, 
                dim, 
                depth, 
                num_heads, 
                window_size,
                mlp_ratio=4., 
                qkv_bias=True, 
                drop=0., 
                attn_drop=0.,
                drop_path=0., 
                norm_layer=nn.LayerNorm, 
                downsample=None, 
                use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        # 向右/下偏移的像素数
        self.shift_size = window_size // 2
        # 构建blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        # 是否实例化Patch Merging Layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    # Masked MSA: SW-MSA计算需要设置mask来隔绝不同区域的信息
    def create_mask(self, x, H, W):
        # 保证 Hp(H_padding) 和 Wp(W_padding) 是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 构造mask模板,要求size同feature map相同
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [B=1, Hp, Wp, C=1]
        # 对mask设计分割切片的window
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        # 对mask的切片windows编号
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        # 对mask划分窗口windows
        # [B=1, H, W, C=1] -> [1*num_windows, Mh, Mw, C=1]
        mask_windows = window_partition(img_mask, self.window_size)
        # 将宽高展平: [num_windows, Mh, Mw, 1] -> [num_windows, Mh*Mw]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # 同值相减为零，不同相减非零: 为判断原先是否在同一窗口window
        # 广播运算:统一两个数组的维度
        # [num_windows, 1, Mh*Mw] - [num_windows, Mh*Mw, 1] -> [num_windows, Mh*Mw, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # 为区分不同区域,对不同值区域(attn_mask != 0)-100使得softmax后为0,
        # [num_windows, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        # 同一Stage中的每个Block的size相同，因此使用同一个mask即可
        attn_mask = self.create_mask(x, H, W)  # [num_windows, Mh*Mw, Mh*Mw]
        for block in self.blocks:
            block.H, block.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, attn_mask)
            else:
                # 通过Swin Transformer
                x = block(x, attn_mask)
        # 下采样Patch Merging
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2
        return x, H, W


# SwinT汇总
class SwinTransformer(nn.Module):
    """ SwinTransformer
    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4 (Patch Partition下采样4倍)
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96 (Linear Embedding输出的通道数C)
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4 (Encoder Block中MLP Block的第一个全连接层将输入节点个数翻倍的倍数)
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """
    def __init__(self,
                patch_size=4,
                in_chans=3, 
                num_classes=1000,
                embed_dim=96, 
                depths=(2, 2, 6, 2), 
                num_heads=(3, 6, 12, 24),
                window_size=7, 
                mlp_ratio=4., 
                qkv_bias=True,
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, 
                patch_norm=True,
                use_checkpoint=False, 
                **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # Stage4输出特征矩阵的Channels
        self.num_features = int(embed_dim * (2 ** (self.num_layers - 1)))  # C * 8
        self.mlp_ratio = mlp_ratio
        # Patch Partition层 + Linear Embedding层
        self.patch_embedding = Patch_Embedding(
            patch_size=patch_size, in_channel=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth随机深度
        # 构建每个Block的drop_rate形成的序列. linspace成等差数列
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # 构建Stage
        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的Stage和论文Structure图中有些差异
            # 这里的Stage不包含该Stage的Patch_Merging层，包含的是下个Stage的. 
            Stage = BasicLayer(dim=int(embed_dim * (2 ** i_layer)),  # C * 1/2/4/8
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # 对应depths[i_later]个Blocks的drop_rate
                                norm_layer=norm_layer,
                                downsample=Patch_Merging if (i_layer < self.num_layers - 1) else None,  # 只有最后一层无需PatchMerging
                                use_checkpoint=use_checkpoint)
            self.stages.append(Stage)
        # 分类模型后期
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 自适应的全局平均池化
        # Classifier FC_last(s): 最后分类的全连接层
        self.FC_last = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # 调用_init_weights, 对模型进行权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, H, W = self.patch_embedding(x)  # x: [B, H*W, C]
        x = self.pos_drop(x)
        # 遍历4个Stage
        for stage in self.stages:
            x, H, W = stage(x, H, W)
        # x: [B, H*W, C]
        x = self.norm(x)
        # [B, H*W, C] -> [B, C, H*W]
        x = x.transpose(1, 2)
        # [B, C, H*W] -> [B, C, 1]
        x = self.avgpool(x)
        # [B, C, 1] -> [B, C]
        x = torch.flatten(x, 1)
        # [B, C] -> [B, num_classes]
        x = self.FC_last(x)
        return x


# SwinT-T/4
def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model


# SwinT-S/4
def swin_small_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 18, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model


# SwinT-B/4
def swin_base_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


# SwinT-B/4
def swin_base_patch4_window12_384(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


# SwinT-B/4
def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


# SwinT-B/4
def swin_base_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


# SwinT-L/4
def swin_large_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            **kwargs)
    return model


# SwinT-L/4
def swin_large_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            **kwargs)
    return model