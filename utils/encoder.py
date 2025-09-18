import torch, math
from torch import nn
from typing import Optional, Union, Tuple, List, Callable, Dict
# 位置编码类
# 生成Timestep Embeddings
def fourier_embedding(timesteps, outdim=256, max_period=10000):
    """
    Classical sinusoidal timestep embedding
    as commonly used in diffusion models
    : param inputs : batch of integer scalars shape [b ,]
    : param outdim : embedding dimension
    : param max_period : max freq added
    : return : batch of embeddings of shape [b, outdim ]
    """
    half = outdim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class PositionalEncoder(nn.Module):
  """
  对输入点, 做sine或者consine位置编码。
  """
  def __init__(
    self,
    d_input: int, 
    n_freqs: int,
    log_space: bool = False
  ):
    super().__init__()
    self.d_input = d_input
    self.n_freqs = n_freqs
    self.log_space = log_space
    self.d_output = d_input * (1 + 2 * self.n_freqs)
    self.embed_fns = [lambda x: x]

    # 定义线性或者log尺度的频率
    if self.log_space:
      freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
    else:
      freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

    # 替换sin和cos
    for freq in freq_bands:
      self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
      self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
  def forward(
    self,
    x
  ) -> torch.Tensor:
    """
    实际使用位置编码的函数。
    输入值是归一化到区间 [−1,1] 的绝对值
    """
    return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

# 定义一个类似NeRF的MLP结构的映射函数
class LatentEncoder(nn.Module):

  def __init__(
    self,
    d_input: int = 4,
    n_layers: int = 8,
    d_filter: int = 256,
    skip: Tuple[int] = (4,),
    dim_clip_emb: Optional[int] = None
  ):
    super().__init__()
    self.d_input = d_input # 输入
    self.skip = skip # 残差连接
    self.act = nn.functional.relu # 激活函数
    self.dim_clip_emb = dim_clip_emb # 类别信息

    # 创建模型的层结构
    self.layers = nn.ModuleList(
      [nn.Linear(self.d_input, d_filter)] +
      [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
       else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
    )

    # Bottleneck 层
    self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            # torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            # nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            # torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            # nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            # torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            # nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            # torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            # nn.SiLU(),
            zero_module(torch.nn.Conv2d(128, 4, kernel_size=3, padding=1, stride=1)),
        )
    

  
  def forward(
    self,
    x: torch.Tensor,
    clip_emb: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    r"""
    带有类别信息的前向传播
    """

    # 判断是否设置类别信息
    if self.dim_clip_emb is None and clip_emb is not None:
      raise ValueError('Cannot input x_direction if dim_clip_emb was not given.')

    # 运行bottleneck层之前的网络层
    x_input = x
    for i, layer in enumerate(self.layers):
      x = self.act(layer(x))
      if i in self.skip:
        x = torch.cat([x, x_input], dim=-1)

    # 运行 bottleneck
    x = x.reshape(-1,1,16,16)
    x = x.repeat(1,4,1,1)
    x = self.blocks(x)
    return x
  


# 定义一个类似Unet mask-to-img的warpper函数
class Warpper(nn.Module):

  def __init__(
    self,
  ):
    super().__init__()

    # Bottleneck 层
    self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(1+4, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            zero_module(torch.nn.Conv2d(256, 4, kernel_size=3, padding=1, stride=1, padding_mode = "reflect")),
        )
    

  
  def forward(
    self,
    x: torch.Tensor,
  ) -> torch.Tensor:

    x = self.blocks(x)
    return x
  

# 定义一个类似Unet mask-to-img的warpper函数
class DPWarpper(nn.Module):

  def __init__(
    self,
    num_classes = 5
  ):
    super().__init__()

    # Bottleneck 层
    self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes+4, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            zero_module(torch.nn.Conv2d(256, 4, kernel_size=3, padding=1, stride=1, padding_mode = "reflect")),
        )
    

  
  def forward(
    self,
    x: torch.Tensor,
  ) -> torch.Tensor:

    x = self.blocks(x)
    return x
  

# 定义一个类似Unet mask-to-img的warpper函数
class Warpperv2(nn.Module):

  def __init__(
    self,
  ):
    super().__init__()

    # Bottleneck 层
    self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(256, 4, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
        )
    

  
  def forward(
    self,
    x: torch.Tensor,
  ) -> torch.Tensor:

    x = self.blocks(x)
    return x

# 定义一个类似Unet mask-to-img的warpper函数
class DPWarpperv2(nn.Module):

  def __init__(
    self,
    num_classes = 5
  ):
    super().__init__()

    # Bottleneck 层
    self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
            nn.SiLU(),
            torch.nn.Conv2d(256, 4, kernel_size=3, padding=1, stride=1, padding_mode = "reflect"),
        )
    

  
  def forward(
    self,
    x: torch.Tensor,
  ) -> torch.Tensor:

    x = self.blocks(x)
    return x
  

# 定义NeRF模型
class NeRF(nn.Module):
  """
  神经辐射场模块。
  """
  def __init__(
    self,
    d_input: int = 3,
    n_layers: int = 8,
    d_filter: int = 256,
    skip: Tuple[int] = (4,),
    d_viewdirs: Optional[int] = None
  ):
    super().__init__()
    self.d_input = d_input # 输入
    self.skip = skip # 残差连接
    self.act = nn.functional.relu # 激活函数
    self.d_viewdirs = d_viewdirs # 视图方向

    # 创建模型的层结构
    self.layers = nn.ModuleList(
      [nn.Linear(self.d_input, d_filter)] +
      [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
       else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
    )

    # Bottleneck 层
    if self.d_viewdirs is not None:
      # 如果使用视图方向，分离alpha和RGB
      self.alpha_out = nn.Linear(d_filter, 1)
      self.rgb_filters = nn.Linear(d_filter, d_filter)
      self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
      self.output = nn.Linear(d_filter // 2, 3)
    else:
      # 如果不使用试图方向，则简单输出
      self.output = nn.Linear(d_filter, 4)
  
  def forward(
    self,
    x: torch.Tensor,
    viewdirs: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    r"""
    带有视图方向的前向传播
    """

    # 判断是否设置视图方向
    if self.d_viewdirs is None and viewdirs is not None:
      raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

    # 运行bottleneck层之前的网络层
    x_input = x
    for i, layer in enumerate(self.layers):
      x = self.act(layer(x))
      if i in self.skip:
        x = torch.cat([x, x_input], dim=-1)

    # 运行 bottleneck
    if self.d_viewdirs is not None:
      # Split alpha from network output
      alpha = self.alpha_out(x)

      # 结果传入到rgb过滤器
      x = self.rgb_filters(x)
      x = torch.concat([x, viewdirs], dim=-1)
      x = self.act(self.branch(x))
      x = self.output(x)

      # 拼接alpha一起作为输出
      x = torch.concat([x, alpha], dim=-1)
    else:
      # 不拼接，简单输出
      x = self.output(x)
    return x