import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        #这个参数定义了用于B样条插值的网格的大小。
        # 网格点用于在输入特征空间上定义样条函数的控制点。网格越大，样条插值的能力越强，但计算成本也越高
        spline_order=3,
        #这个参数指定了B样条的阶数。
        # 阶数决定了样条曲线的平滑度和灵活性。较高的阶数允许更复杂的曲线形状，但也可能导致过拟合或计算不稳定
        scale_noise=0.1,
        #在初始化样条权重时，这个参数用于向网格点上的样条系数添加噪声。
        # 噪声的大小由scale_noise控制，它有助于打破初始化时的对称性，可能有助于模型的泛化能力
        scale_base=1.0,
        scale_spline=1.0,
        #这两个参数分别用于缩放基础线性层的权重和样条插值层的权重。
        # 它们允许在初始化时调整两层之间的相对重要性，可能对模型的训练速度和最终性能有所影响
        enable_standalone_scale_spline=True,
        #这是一个布尔参数，用于控制是否启用独立的样条权重缩放器（spline_scaler）。
        # 如果启用，则每个输出特征对应的样条权重可以独立缩放，这可能为模型提供了更多的灵活性
        base_activation=torch.nn.SiLU,
        # 这个参数指定了在将输入传递给基础线性层之前应用的激活函数。默认是SiLU（Sigmoid Linear Unit），但可以是任何PyTorch支持的激活函数。
        # 激活函数为模型引入了非线性，有助于捕捉复杂的数据模式
        grid_eps=0.02,
        #在更新网格点时，这个参数用于控制均匀网格和自适应网格之间的插值权重。
        # 较小的grid_eps值使网格更接近均匀分布，而较大的值则使网格更多地基于输入数据的分布进行调整
        grid_range=[-1, 1],
        #这个参数定义了网格点的范围，即输入特征空间的最小值和最大值。网格点将在这个范围内均匀或自适应地分布
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        # 根据网格范围和大小生成网格点
        h = (grid_range[1] - grid_range[0]) / grid_size
        #生成网格点张量
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        # 注册网格点为缓冲区注册缓冲区的主要目的是让该缓冲区成为模块状态的一部分，这样当模块被保存到磁盘或从磁盘加载时，缓冲区也会一起被保存和加载。
        # 但是，与模块的参数（通过self.register_parameter注册）不同，缓冲区不会被视为模型的参数，因此在模型训练过程中不会对其进行优化（即不会计算梯度）
        self.register_buffer("grid", grid)
        # 初始化权重参数
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise# 初始化样条权重时的噪声比例
        self.scale_base = scale_base# 基础线性层权重的缩放因子
        self.scale_spline = scale_spline# 样条插值层权重的缩放因子
        self.enable_standalone_scale_spline = enable_standalone_scale_spline # 是否启用独立的样条权重缩放器
        self.base_activation = base_activation() # 基础线性层前的激活函数实例
        self.grid_eps = grid_eps # 控制均匀网格和自适应网格插值的权重
        #初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        # 使用Kaiming均匀初始化方法初始化基础线性层的权重
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        # 在无梯度模式下初始化样条插值层的权重
        with torch.no_grad():
            #生产噪声并调整样条权重生成噪声，形状为(grid_size + 1, in_features, out_features)
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:#启用独立的样条权重缩放器（spline_scaler）
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):#B样条基函数计算
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        #获取网格点张量，形状为(in_features, grid_size + 2 * spline_order + 1)
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        # 对输入张量x进行unsqueeze操作，在最后一个维度上增加一个大小为1的维度，使其形状变为(batch_size, in_features, 1)
        x = x.unsqueeze(-1)
        # 初始化基函数张量。通过比较输入张量x与网格点张量grid来生成一个布尔张量
        # 如果x中的某个元素位于grid中相邻网格点之间（包括左边界但不包括右边界），则对应位置的值为True（在后续操作中会转换为x的dtype），否则为False
        # 这里使用了广播机制，使得比较操作能够同时应用于所有批次中的样本和所有输入特征
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # 使用B样条的递归公式计算更高阶的基函数值
        # 对于每个阶数k（从1到spline_order），都根据前一个阶数的基函数值来计算当前阶数的基函数值
        for k in range(1, self.spline_order + 1):
            # 计算当前阶数的基函数值。这里涉及到了网格点之间的差分和基函数值的加权和
            # 注意：由于PyTorch张量的索引是从0开始的，因此这里使用了:-(k+1)和k+1:来避免索引越界
            # 同时，由于基函数值在初始时是布尔类型的，这里隐式地将True视为1.0，False视为0.0进行计算
            # 计算完成后，新的基函数值会覆盖旧的基函数值
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        # 返回计算好的基函数张量。使用.contiguous()确保张量在内存中连续存储，便于后续操作
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        # 使用最小二乘法求解曲线系数
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        # 根据是否启用独立的样条权重缩放器来计算缩放后的样条权重
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        # 通过基础线性层和激活函数计算基础输出
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # 计算B样条基函数，并通过缩放后的样条权重计算样条输出
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        # 返回基础输出和样条输出的和
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        # 验证输入张量x的维度和特征数是否符合要求
        assert x.dim() == 2 and x.size(1) == self.in_features
        # 获取批次大小
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        # 交换张量的维度顺序，以便后续进行批量矩阵乘法
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        # 获取缩放后的样条权重，并交换维度顺序以匹配splines的形状
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        # 计算未简化的样条输出。这里使用了批量矩阵乘法（torch.bmm）
        # 结果是一个形状为(in_features, batch, out_features)的张量，表示每个输入特征对每个样本的样条输出贡献
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        # 对输入张量x的每个特征通道进行排序，以收集数据分布
        x_sorted = torch.sort(x, dim=0)[0] # 沿着批次维度排序，得到每个特征通道的排序后数据
        # 根据排序后的数据计算自适应网格点
        # 使用linspace在0到batch-1之间均匀取样，得到self.grid_size+1个点，用于在排序后的数据中插值
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]
        # 计算均匀网格点的步长，并考虑一个小的边距margin来避免边界效应
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
        # 通过插值结合自适应网格点和均匀网格点来更新网格
        # self.grid_eps控制插值的权重
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        # 由于B样条插值需要额外的控制点，我们在网格的两端添加这些点
        # 这些点是通过在均匀步长的基础上向外扩展得到的
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        # 更新模块的网格点缓冲区
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        # 计算样条权重的绝对值的均值（沿着最后一个维度，即系数维度）
        # 这可以看作是对样条权重的一种L1正则化的简化模拟
        l1_fake = self.spline_weight.abs().mean(-1)
        # 对l1_fake求和，得到与激活相关的正则化损失项
        regularization_loss_activation = l1_fake.sum()
        # 将l1_fake归一化，使其可以解释为概率分布（尽管这通常不是一个有效的概率分布）
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        # 返回加权后的正则化损失，其中包含了与激活相关的损失项和熵损失项
        # regularize_activation和regularize_entropy是两个超参数，用于控制这两项在总损失中的权重
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        #一个整数列表，指定了网络隐藏层的特征数。
        # 注意，这个列表实际上定义了每对相邻层之间的特征数，因为除了第一层输入和最后一层输出外，每层的输出特征数也是下一层的输入特征数
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        #创建了一个空的nn.ModuleList实例，并将其赋值给实例变量self.layers。
        # nn.ModuleList是一个特殊的列表，用于存储nn.Module对象（如网络层）。
        # 与普通的Python列表不同，nn.ModuleList能够自动注册其包含的模块，以便它们能够参与模型的参数更新、保存和加载等操作
        self.layers = torch.nn.ModuleList()
        #这行代码创建了一个空的nn.ModuleList实例，并将其赋值给实例变量self.layers。
        # nn.ModuleList是一个特殊的列表，用于存储nn.Module对象（如网络层）。
        # 与普通的Python列表不同，nn.ModuleList能够自动注册其包含的模块，以便它们能够参与模型的参数更新、保存和加载等操作
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    #这是KAN类的前向传播方法。它定义了数据通过网络的前向传播路径。方法接收一个输入张量x和一个可选的布尔参数update_grid
    def forward(self, x: torch.Tensor, update_grid=False):
        #在前向传播过程中，代码遍历self.layers列表中的每个层。
        # 如果update_grid参数为True，则调用当前层的update_grid方法来根据输入数据动态更新网格点。
        # 然后，将输入x传递给当前层，并将输出作为下一层的输入。这个过程一直持续到所有层都被遍历完毕
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        #最后，返回最终层的输出作为整个前向传播的结果
        return x

    #这是计算整个网络正则化损失的方法。它接收两个可选参数来控制正则化损失的权重
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        #方法通过遍历self.layers列表中的每个层，并调用每个层的regularization_loss方法来计算正则化损失。
        # 然后，将所有层的正则化损失求和，并返回总和作为整个网络的正则化损失。
        # 这个过程有助于防止模型过拟合，通过惩罚样条权重的某些特性（如L1范数和熵）
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
