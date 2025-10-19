
package lesson


import torch.{&&, ---, ::, BFloat16, DType, Default, Float32, FloatNN, Int64, Slice, Tensor, nn}
import torch.nn.{modules, functional as F}
import torch.nn.modules.{HasParams, TensorModule}



/*
input_dim (int): 每个时间步输入特征的维度。
hidden_dim (int): RNN隐藏状态的维度。
output_dim (int): 最终输出的维度。
num_rnn_layers (int): 堆叠RNN层的数量。默认值为1。
      */
class SimpleRNNModel[ParamType <: FloatNN: Default](input_dim: Int, hidden_dim: Int, output_dim: Int, num_rnn_layers: Int = 1) extends TensorModule[ParamType]  with HasParams[ParamType] {

  // 定义RNN层
  // batch_first=True 表示输入/输出张量形状为: (batch, seq, feature)
  val rnn = nn.RNN(
    input_size=input_dim,
    hidden_size=hidden_dim,
    num_layers=num_rnn_layers,
    batch_first=true, // 确保输入形状是 (batch, seq_len, input_size)
    nonlinearity="tanh" // 默认激活函数
  )

  // 定义输出层（全连接层）
  // 它以RNN的最终隐藏状态作为输入
  val fc = nn.Linear(hidden_dim, output_dim)

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
  /*
        定义模型的正向传播。
        param:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_dim)。
        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_dim)。
        """
   */
  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {

    // 初始化隐藏状态为零
    // 形状: (num_layers, batch_size, hidden_size)
    val batch_size = x.size(0)
    val h0 = torch.zeros(num_rnn_layers, batch_size, hidden_dim).to(x.device)

    // 通过RNN层传递数据
    // rnn_out 形状: (batch_size, seq_len, hidden_size)
    // hn 形状: (num_layers, batch_size, hidden_size)
    val (rnn_out, hn) = rnn(x, h0.to(this.paramType))

    // 我们只需要最后一层、最后一个时间步的隐藏状态
    // hn 包含所有层的最终隐藏状态。
    // hn[-1] 获取最后一层的最终隐藏状态。
    // hn[-1] 的形状: (batch_size, hidden_size)
    val last_layer_hidden_state = hn(-1)

    // 将最后一个隐藏状态通过全连接层
    // out 形状: (batch_size, output_dim)
    val out = fc(last_layer_hidden_state)

    out
  }
}

class SimpleCNN[ParamType <: FloatNN: Default](num_classes: Int = 10) extends TensorModule[ParamType]  with HasParams[ParamType] {

  // 输入形状: (批次, 1, 28, 28) - 假设是像MNIST那样的灰度图像
  val conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
  // conv1后的形状: (批次, 16, 28, 28) -> (28 - 3 + 2*1)/1 + 1 = 28
  val relu1 = nn.ReLU()
  val pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
  // pool1后的形状: (批次, 16, 14, 14) -> 28 / 2 = 14

  val conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
  // conv2后的形状: (批次, 32, 14, 14) -> (14 - 3 + 2*1)/1 + 1 = 14
  val relu2 = nn.ReLU()
  val pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
  // pool2后的形状: (批次, 32, 7, 7) -> 14 / 2 = 7

  // 展平输出以便连接到线性层
  // 展平后的尺寸 = 32 * 7 * 7 = 1568
  val fc = nn.Linear(32 * 7 * 7, num_classes)

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    // 应用第一个卷积块
    var out = conv1(input)
    out = relu1(out)
    out = pool1(out)

    // 应用第二个卷积块
    out = conv2(out)
    out = relu2(out)
    out = pool2(out)

    // 展平卷积层的输出
    // -1 表示推断批次大小
    out = out.view(out.size(0), -1)

    // 应用全连接层
    out = fc(out)
    return out
  }
}


class SimpleRNN[ParamType <: FloatNN: Default](input_size: Int, hidden_size: Int, output_size: Int, num_layers: Int = 1) extends TensorModule[ParamType]  with HasParams[ParamType] {

  // RNN层
  // 输入尺寸: input_size
  // 隐藏状态尺寸: hidden_size
  // 层数: num_layers
  // batch_first=False是默认值，期望输入格式为: (序列长度, 批次, 输入特征)
  val rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=false)

  // 全连接层，将RNN输出映射到最终输出尺寸
  val fc = nn.Linear(hidden_size, output_size)

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = ???

  def apply(x: Tensor[ParamType], h0: Option[Tensor[ParamType]] = None): (Tensor[ParamType], Tensor[ParamType]) = forward(x, h0)

  def forward(x: Tensor[ParamType], h0: Option[Tensor[ParamType]] = None): (Tensor[ParamType], Tensor[ParamType]) = {
    // x 形状: (序列长度, 批次, 输入特征)

    // 如果未提供，则初始化隐藏状态
    // 形状: (层数 * 方向数, 批次, 隐藏尺寸)
    val hidden = if !h0.isDefined then torch.zeros(num_layers, x.size(1), hidden_size).to(x.device) else h0.get
    // 数据通过RNN层
    // out 形状: (序列长度, 批次, 隐藏尺寸) -> 包含每个时间步的输出特征
    // hn 形状: (层数 * 方向数, 批次, 隐藏尺寸) -> 包含最终隐藏状态
    val (out, hn) = rnn(x, hidden.to(this.paramType))

    // 我们可以选择使用最后一个时间步的输出
    // out[-1] 形状: (批次, 隐藏尺寸)
    // 或者，如果需要，处理整个序列 'out'
    val out_last_step = out(-1,::,:: )

    // 将最后一个时间步的输出通过线性层
    val final_output = fc(out_last_step)
    // final_output 形状: (批次, 输出尺寸)

    return (final_output, hn) // 返回最终输出和最后一个隐藏状态
  }

}

object lesson_07_2 {

//  @main
  def main(): Unit = {

    //06
    // 定义参数
    val input_features = 10 // 例如，字符/单词的嵌入维度
    val hidden_nodes = 20
    val output_classes = 5 // 例如，根据序列预测5个类别之一
    val sequence_length = 15
    val batch_size = 4

    // 实例化模型
    val rnn_model = SimpleRNN(input_size = input_features, hidden_size = hidden_nodes, output_size = output_classes)

    // 创建虚拟输入批次（序列长度，批次大小，输入特征）
    // requires_grad=False 用于演示
    val dummy_input_rnn = torch.randn(sequence_length, batch_size, input_features, requires_grad = false).to(torch.float32)

    // 执行前向传播（不提供h0，它将被初始化）
    val (output_rnn, final_hidden_state) = rnn_model(dummy_input_rnn, None)

    // 打印输入和输出形状
    println(s"Input sequence shape: ${dummy_input_rnn.shape}")
    println(s"Output prediction shape: ${output_rnn.shape}")
    println(s"Final hidden state shape: ${final_hidden_state.shape}")

    //05
    // 实例化模型
    val cnn_model = SimpleCNN(num_classes = 10)

    // 创建虚拟输入批次（例如，4张图像，1通道，28x28像素）
    // requires_grad=False，因为我们只是进行前向传播演示
    val dummy_input_cnn = torch.randn(4, 1, 28, 28, requires_grad = false).to(torch.float32)

    // 执行前向传播
    val output_cnn = cnn_model(dummy_input_cnn)

    // 打印输入和输出形状
    println(s"Input shape: ${dummy_input_cnn.shape}")
    println(s"Output shape: ${output_cnn.shape}")


    //04
    // 示例：定义一个 LSTM 层
    val input_size = 10
    val hidden_size = 20
    val num_layers = 2

    val lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first = true)

    // 示例输入 (batch_size, seq_length, input_size)
    val batch_size_02 = 5
    val seq_length = 15
    val dummy_input = torch.randn(batch_size_02, seq_length, input_size).to(torch.float32)

    // 前向传播需要初始隐藏状态和单元状态 (h_0, c_0)
    // 如果不提供，它们默认为零。
    // 形状： (num_layers * num_directions, batch_size, hidden_size)
    val h0 = torch.randn(num_layers, batch_size_02, hidden_size).to(torch.float32)
    val c0 = torch.randn(num_layers, batch_size_02, hidden_size).to(torch.float32)

    // 前向传播
    val (output, (hn, cn)) = lstm_layer(dummy_input, (h0, c0))

    // output 形状： (batch_size, seq_length, hidden_size)
    // hn 形状： (num_layers, batch_size, hidden_size) - 每个层的最终隐藏状态
    // cn 形状： (num_layers, batch_size, hidden_size) - 每个层的最终单元状态
    println(s"LSTM Output shape:${output.shape}")
    println(s"LSTM Final Hidden State shape:${hn.shape}")
    println(s"LSTM Final Cell State shape:${cn.shape}")


    //05
    // 示例：定义一个 GRU 层
    val gru_layer = nn.GRU(input_size, hidden_size, num_layers, batch_first = true)

    // 前向传播需要初始隐藏状态 (h_0)
    // 如果不提供，它默认为零。
    // 形状： (num_layers * num_directions, batch_size, hidden_size)
    val h0_gru = torch.randn(num_layers, batch_size_02, hidden_size).to(torch.float32)

    // 前向传播
    val (output_gru, hn_gru) = gru_layer(dummy_input, h0_gru)

    // output 形状： (batch_size, seq_length, hidden_size)
    // hn 形状： (num_layers, batch_size, hidden_size) - 每个层的最终隐藏状态
    println(s"\nGRU Output shape: ${output_gru.shape}")
    println(s"GRU Final Hidden State shape: ${hn_gru.shape}")

    mains()
  }
//  @main
  def mains(): Unit = {

    //03
    // 示例参数（同前）
    val seq_len = 20 // 最长序列长度
    val batch_size = 32 // 批次中的序列数量
    val input_features = 100 // 每个词嵌入的维度
    val hidden_size = 50 // RNN的示例隐藏大小

    // 创建一个批次维度在前的虚拟输入张量
    // 形状: (批次大小, 序列长度, 输入特征)
    val rnn_input_batch_first = torch.randn(batch_size, seq_len, input_features).to(torch.float32)

    // 使用 batch_first=True 初始化RNN层
    val rnn_layer = nn.RNN(input_size = input_features, hidden_size = hidden_size, batch_first = true)

    // 将输入通过层（输出形状也将是批次维度在前）
    val (output, hidden_state) = rnn_layer(rnn_input_batch_first, None) //todo rnn remove hidden state None

    println(s"批次维度优先的RNN输入形状: ${rnn_input_batch_first.shape}")
    // 输出: 批次维度优先的RNN输入形状: torch.Size([32, 20, 100])
    println(s"批次维度优先的RNN输出形状: ${output.shape}")
    // 输出: 批次维度优先的RNN输出形状: torch.Size([32, 20, 50])


    //02
    // 示例参数
    val seq_len2 = 20 // 最长序列长度
    val batch_size2 = 32 // 批次中的序列数量
    val input_features2 = 100 // 每个词嵌入的维度

    // 创建一个虚拟输入张量（例如，用随机数填充）
    // 形状: (序列长度, 批次大小, 输入特征)
    val rnn_input = torch.randn(seq_len2, batch_size2, input_features2)

    println(s"标准RNN输入形状: ${rnn_input.shape}")
    // 输出: 标准RNN输入形状: torch.Size([20, 32, 100])


    //01
    //--- 示例用法 ---

    // 定义模型参数
    val INPUT_DIM = 10 // 输入特征维度（例如，嵌入大小）
    val HIDDEN_DIM = 20 // 隐藏状态维度
    val OUTPUT_DIM = 5 // 输出维度（例如，类别数量）
    val NUM_LAYERS = 1 // RNN层数

    // 创建模型
    val model = SimpleRNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS)
    print("模型结构:")
    print(model)

    // 创建一些虚拟输入数据
    val BATCH_SIZE = 4
    val SEQ_LEN = 15
    val dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM).to(torch.float32) // 形状: (batch, seq, feature)

    // 执行前向传播
    val output_02 = model(dummy_input)
    // 打印输入和输出形状
    print(f"\n输入形状: ${dummy_input.shape}")
    print(f"输出形状: ${output_02.shape}")

    // 验证输出形状是否与 (BATCH_SIZE, OUTPUT_DIM) 匹配
//    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)

  }
}
