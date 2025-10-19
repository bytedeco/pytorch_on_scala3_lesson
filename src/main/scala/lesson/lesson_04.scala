package lesson


import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{modules, functional as F}
import torch.numpy.TorchNumpy as np
import torch.{Tensor, *}

//01
class MySimpleNetwork[ParamType <: FloatNN: Default] extends TensorModule[ParamType]{

  //  示例 ：一个线性层
  val layer1 = nn.Linear(in_features = 10, out_features = 5)
  //  示例 ：一个激活函数实例
  val activation = nn.ReLU()

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    //   定义数据流经组件的方式
    var x = layer1(input)
    x = activation(x)
    x
  }

}

//02
class CustomModuleWithParameter[ParamType <: FloatNN: Default] extends TensorModule[ParamType]  with HasParams[ParamType] {

  //一个可学习的参数张量
  val my_weight = nn.Parameter("my_weight",torch.randn(5, 2))
  // 一个普通的张量属性（不会自动跟踪用于优化）
  val my_info = torch.tensor(Seq(1.0, 2.0))

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    // 示例用法
    val tensor = torch.matmul(x, my_weight)
    tensor.to(this.paramType)
  }
}

//03
class SimpleLinearModel[ParamType <: FloatNN: Default](input_features: Int, output_features: Int) extends TensorModule[ParamType]  with HasParams[ParamType] {

  // 定义单个线性层
  val linear_layer = nn.Linear(input_features, output_features)
  // 打印初始化信息
  println(f"已初始化 SimpleLinearModel，输入特征数= ${input_features}，输出特征数= ${output_features}")
  println(f"已定义层: ${linear_layer}")

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)

  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    // 定义前向传播：将输入通过线性层
    println(f"前向传播输入形状: ${x.shape}")
    val output = linear_layer(x)
    println(f"前向传播输出形状: ${output.shape}")
    return output
  }
}


class SimpleMLP[ParamType <: FloatNN: Default](input_size: Int, hidden_size: Int, output_size: Int) extends TensorModule[ParamType]  with HasParams[ParamType] {

  // 定义层
  val layer1 = nn.Linear(input_size, hidden_size)
  val activation = nn.ReLU() // 将激活函数定义为层
  val layer2 = nn.Linear(hidden_size, output_size)
  println(f"已初始化 SimpleMLP: 输入=${input_size}, 隐藏层=${hidden_size}, 输出=${output_size}")
  println(f"层 1: ${layer1}")
  println(f"激活函数: ${activation}")
  println(f"层 2: ${layer2}")

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    // 定义前向传播序列
    println(f"前向传播输入形状: ${input.shape}")
    var x = layer1(input)
    println(f"经过层 1 后的形状: ${x.shape}")
    x = activation(x) // 应用 ReLU 激活
    println(f"经过激活函数后的形状: ${x.shape}")
    x = layer2(x)
    println(f"经过层 2（输出）后的形状: ${x.shape}")
    x
  }
}

//08
// 在简单模型中的示例
class SimpleNet[ParamType <: FloatNN: Default] extends TensorModule[ParamType] {
  val layer1 = nn.Linear(10, 20)
  val activation = nn.ReLU()
  val layer2 = nn.Linear(20, 5)

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = layer1(input)

    x = activation(x) // 应用 ReLU
    x = layer2(x)
    return x
  }
}
object lesson_04 {

//  @main
  def main(): Unit = {

    //10
    // 示例用法
    val tanh_activation = nn.Tanh()
    val input_tensor10 = torch.randn(4).to(torch.float32) // 示例输入张量
    val output_tensor10 = tanh_activation(input_tensor10)

    println(f"输入: ${input_tensor10}")
    println(f"Tanh 输出: ${output_tensor10}")


    //09
    // 示例用法
    val sigmoid_activation = nn.Sigmoid()
    val input_tensor9 = torch.randn(4).to(torch.float32) // 示例输入张量
    val output_tensor9 = sigmoid_activation(input_tensor9)

    println(f"输入: ${input_tensor9}")
    println(f"Sigmoid 输出: ${output_tensor9}")

    //08
    // 示例用法
    val relu_activation = nn.ReLU()
    val input_tensor = torch.randn(4).to(torch.float32) // 示例输入张量
    val output_tensor = relu_activation(input_tensor)

    println(f"输入: ${input_tensor}")
    println(f"ReLU 输出: ${output_tensor}")

    val model8 = SimpleNet()

    //07
    // 示例：处理一批 10 个序列，每个序列长 20 步，每步有 5 个特征。
    // 使用大小为 30 的隐藏状态。
    // 设置 batch_first=True 以便更方便地处理数据。
    val rnn_layer = nn.RNN(input_size = 5, hidden_size = 30, batch_first = true)

    // 创建一个示例输入张量（批，序列长度，输入特征）
    val input_sequence_batch = torch.randn(10, 20, 5).to(torch.float32)

    // 初始化隐藏状态（层数，批，隐藏状态大小）
    // 如果未提供，默认为零。
    val initial_hidden_state = torch.randn(1, 10, 30).to(torch.float32) // 层数=1

    // 将输入序列和初始隐藏状态通过 RNN
    // 输出包含所有时间步的输出
    // final_hidden_state 包含最后一个时间步的隐藏状态
    val (output_sequence, final_hidden_state) = rnn_layer(input_sequence_batch, initial_hidden_state)

    println(f"Input shape: ${input_sequence_batch.shape}")
    println(f"Initial hidden state shape: ${initial_hidden_state.shape}")
    println(f"Output sequence shape: ${output_sequence.shape}") // (批，序列长度，隐藏状态大小)
    println(f"Final hidden state shape: ${final_hidden_state.shape}") // (层数，批，隐藏状态大小)
    // 预期输出：
    // Input shape: torch.Size([10, 20, 5])
    // Initial hidden state shape: torch.Size([1, 10, 30])
    println(f"Output sequence shape: ${output_sequence.shape}") // (批，序列长度，隐藏状态大小)
    println(f"Final hidden state shape: ${final_hidden_state.shape}") // (层数，批，隐藏状态大小)
    // 预期输出：
    // Output sequence shape: torch.Size([10, 20, 30])
    // Final hidden state shape: torch.Size([1, 10, 30])


    //06
    // 示例：处理一批 16 张图像，3 通道（RGB），32x32 像素
    // 应用 6 个滤波器（输出通道），每个大小为 5x5
    val conv_layer = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5)

    // 创建一个示例输入张量（批大小，通道，高，宽）
    // PyTorch 通常期望通道优先的格式 (N, C, H, W)
    val input_image_batch = torch.randn(16, 3, 32, 32).to(torch.float32)

    // 将输入通过卷积层
    val output_feature_maps = conv_layer(input_image_batch)

    println(f"Input shape: ${input_image_batch.shape}")
    println(f"Output shape: ${output_feature_maps.shape}")
    // 没有填充/步幅时，输出大小会减小：32 - 5 + 1 = 28
    // 预期输出：
    // Input shape: torch.Size([16, 3, 32, 32])
    // Output shape: torch.Size([16, 6, 28, 28])

    // 检查参数
    println(f"\nWeight (filter) shape: ${conv_layer.weight.shape}") // (输出通道，输入通道，卷积核高，卷积核宽)
    println(f"Bias shape: ${conv_layer.bias().shape}") // (输出通道) //todo add bias for conv2d

    // 预期输出：
    // Weight (filter) shape: torch.Size([6, 3, 5, 5])
    // Bias shape: torch.Size([6])


    //05
    // 示例：创建一个线性层，输入特征大小为 20，输出特征大小为 30
    val linear_layer = nn.Linear(in_features = 20, out_features = 30)

    // 创建一个示例输入张量（批大小 64，20 个特征）
    val input_tensor2 = torch.randn(64, 20).to(torch.float32)

    // 将输入通过该层
    val output_tensor2 = linear_layer(input_tensor2)

    println(f"Input shape: ${input_tensor2.shape}")
    println(f"Output shape: ${output_tensor2.shape}")
    // 预期输出：
    // Input shape: torch.Size([64, 20])
    // Output shape: torch.Size([64, 30])

    // 检查层的参数（自动创建）
    println(f"\nWeight shape: ${linear_layer.weight.shape}")
    println(f"Bias shape: ${linear_layer.bias.shape}")
    // 预期输出：
    // Weight shape: torch.Size([30, 20])
    // Bias shape: torch.Size([30])

    //04
    // --- 使用示例 ---
    // 定义维度
    val in_size = 784 // 示例：展平的 28x28 图像
    val hidden_units = 128
    val out_size = 10 // 示例：用于分类的 10 个类别

    // 实例化 MLP
    val mlp_model = SimpleMLP(input_size = in_size, hidden_size = hidden_units, output_size = out_size)

    // 创建模拟输入（批大小=32）
    val dummy_mlp_input = torch.randn(32, in_size).to(torch.float32)
    println(f"\n模拟 MLP 输入形状: ${dummy_mlp_input.shape}")

    // 前向传播
    val mlp_output = mlp_model(dummy_mlp_input)
    println(f"MLP 输出形状: ${mlp_output.shape}")

    // 检查参数
    println("\nMLP 模型参数:")
    for ((name, param) <- mlp_model.named_parameters()) {
      if param.requires_grad then
        println(f"  名称: ${name}, 形状: ${param.shape}")
    }

    //03
    // --- 使用示例 ---
    // 定义输入和输出维度
    val in_dim = 10
    val out_dim = 1

    // 实例化自定义模型
    val model3 = SimpleLinearModel(in_dim, out_dim)

    // 创建一些模拟输入数据（batch_size=5，特征数=10）
    val dummy_input = torch.randn(5, in_dim).to(torch.float32)
    println(f"\n模拟输入张量形状: ${dummy_input.shape}")

    // 将数据通过模型
    val output = model3(dummy_input)
    println(f"模型输出张量形状: ${output.shape}")

    // 检查参数（自动注册）
    println("\n模型参数:")
    for ((name, param) <- model3.named_parameters()) {

      if param.requires_grad then
        println(s"  名称: ${name}, 形状: ${param.shape}")
    }

    //02
    val module = CustomModuleWithParameter()

    // 访问模块跟踪的参数
    for( (name, param) <- module.named_parameters(true)){
      println(f"Parameter name: ${name}, Shape: ${param.shape}, Requires grad: ${param.requires_grad}")

    }

    for (param <- module.parameters(true)) {
      println(f"Parameter , Shape: ${param.shape}, Requires grad: ${param.requires_grad}")

    }

    //01
    // 实例化网络
    val model2 = MySimpleNetwork()
    println(model2)

  }

}
