package lesson


import torch.Device.{CPU, CUDA}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{modules, functional as F}
import torch.{Tensor, *}


class SimpleCNNs[ParamType <: FloatNN: Default] extends TensorModule[ParamType]  with HasParams[ParamType] {

  // 层定义
  // 卷积层1
  val conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
  // 最大池化层1
  val pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
  // 卷积层2
  val conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
  // 最大池化层2
  val pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
  // 全连接层
  // fc1的输入特征取决于池化后的输出形状
  // 输入：28x28 -> Conv1 (padding=2) -> 28x28 -> Pool1 (stride=2) -> 14x14
  // -> Conv2 (padding=2) -> 14x14 -> Pool2 (stride=2) -> 7x7
  // 因此，展平后的尺寸是 32 个通道 * 7 高度 * 7 宽度 = 1568
  val fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
  val fc2 = nn.Linear(in_features=128, out_features=10) // 用于10个类别的输出

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    // 定义数据流经各层的方式
    // 输入x形状：[批大小, 1, 28, 28]
    // 应用Conv1、ReLU、Pool1
    var x = pool1(F.relu(conv1(input)))
    // pool1后的形状：[批大小, 16, 14, 14]

    // 应用Conv2、ReLU、Pool2
    x = pool2(F.relu(conv2(x)))
    // pool2后的形状：[批大小, 32, 7, 7]

    // 展平张量以用于全连接层
    // -1 保持批大小维度不变
    x = x.view(-1, 32 * 7 * 7)
    // view后的形状：[批大小, 1568]

    // 应用FC1和ReLU
    x = F.relu(fc1(x))
    // fc1后的形状：[批大小, 128]

    // 应用FC2（输出层，此处无激活函数，通常与损失函数一起应用）
    x = fc2(x)
    // fc2后的形状：[批大小, 10]
    return x
  }
}
object lesson_07 {


//  @main
  def main(): Unit = {



    //03
    val conv_layer = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
    // 输入: N=16, Cin=3, Hin=32, Win=32
    val input_tensor = torch.randn(16, 3, 32, 32).to(torch.float32)

    // 参数: K=3, S=1, P=1, D=1 (默认)
    // H_out = floor((32 + 2*1 - 1*(3-1) - 1)/1 + 1) = floor((32 + 2 - 2 - 1)/1 + 1) = floor(31/1 + 1) = 32
    // W_out = floor((32 + 2*1 - 1*(3-1) - 1)/1 + 1) = floor((32 + 2 - 2 - 1)/1 + 1) = floor(31/1 + 1) = 32
    // 简化公式 (D=1):
    // H_out = floor((32 + 2*1 - 3)/1 + 1) = floor(31/1 + 1) = 32
    // W_out = floor((32 + 2*1 - 3)/1 + 1) = floor(31/1 + 1) = 32

    // 前向传播
    val output_tensor = conv_layer(input_tensor)
    println(s"Output shape: ${output_tensor.shape}") // 预期：[16, 64, 32, 32]

    //04
    val conv_layer_s2 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
    // H_out = floor((32 + 2*1 - 3)/2 + 1) = floor(31/2 + 1) = floor(15.5 + 1) = floor(16.5) = 16
    // W_out = floor((32 + 2*1 - 3)/2 + 1) = floor(31/2 + 1) = floor(15.5 + 1) = floor(16.5) = 16
    // 前向传播
    val output_tensor_s2 = conv_layer_s2(input_tensor)
    println(s"Output shape: ${output_tensor_s2.shape}") // 预期：[16, 64, 16, 16]

    //05
    // 来自前一个卷积层的输入: N=16, Cin=64, Hin=32, Win=32
    val pool_layer = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0) // 常见设置

    // 参数: K=2, S=2, P=0, D=1 (默认)
    // H_out = floor((32 + 2*0 - 1*(2-1) - 1)/2 + 1) = floor((32 - 1 - 1)/2 + 1) = floor(30/2 + 1) = floor(15 + 1) = 16
    // W_out = floor((32 + 2*0 - 1*(2-1) - 1)/2 + 1) = floor((32 - 1 - 1)/2 + 1) = floor(30/2 + 1) = floor(15 + 1) = 16
    // 前向传播
    val pooled_output = pool_layer(output_tensor)
    println(s"Output shape: ${pooled_output.shape}") // 预期：[16, 64, 16, 16]

    //02
    // 实例化模型
    val model = SimpleCNNs()
    println(model)

    // 创建一个虚拟输入张量（4张图像的批次，1个通道，28x28）
    // 如果您打算训练，需要梯度跟踪
    val dummy_input = torch.randn(4, 1, 28, 28).to(torch.float32)

    // 将输入传入模型（前向传播）
    val output = model(dummy_input)

    // 检查输出形状
    println(s"\nInput shape: ${dummy_input.shape}")
    println(s"Output shape: ${output.shape}") // 预期：[4, 10]

    //01
    // 示例：一个Conv2d层，接收3个输入通道（例如RGB图像），
    // 使用5x5滤波器生成16个输出通道。
    val conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 5, stride = 1, padding = 2)

    // 示例：一个MaxPool2d层，使用2x2窗口和步长为2。
    // 这通常会将输入的高度和宽度减半。
    val pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    // ReLU激活函数
    val relu1 = nn.ReLU()
    // 示例：一个线性层，接收一个展平的512个特征向量，
    // 并输出10个值（例如，用于10个类别）。
    val fc1 = nn.Linear(in_features = 512, out_features = 10)
  }

}