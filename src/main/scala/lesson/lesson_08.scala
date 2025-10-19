
package lesson

import org.bytedeco.pytorch
import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch.{Example, InputArchive, OutputArchive, TensorExampleVectorIterator}
import org.bytedeco.pytorch.{ChunkDatasetOptions, Example, ExampleIterator, ExampleStack, ExampleVector}
import org.bytedeco.pytorch.global.torch as torchNative

import java.net.URL
import java.util.zip.GZIPInputStream
import java.nio.file.{Files, Path, Paths}
import scala.collection.{mutable, Set as KeySet}
import scala.util.{Failure, Random, Success, Try, Using}
import torch.Device.{CPU, CUDA}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.{&&, ---, ::, BFloat16, DType, Default, Float32, FloatNN, Int64, Slice, Tensor, nn}
import torch.nn.{modules, functional as F}
import torch.nn.modules.{HasParams, TensorModule}
import torch.optim.Adam
import torch.utils.data.{DataLoader, DataLoaderOptions, Dataset, NormalTensorDataset}
import torch.utils.data.dataset.custom.{FashionMNIST, MNIST}
import torch.utils.data.*
import torch.utils.data.dataloader.*
import torch.utils.data.datareader.ChunkDataReader
import torch.utils.data.dataset.*
import torch.utils.data.sampler.RandomSampler

import scala.collection.mutable.SortedMap as OrderedDict
import torch.numpy.TorchNumpy as np
import torch.optim as optim

class SimpleNet2[ParamType <: FloatNN: Default] extends TensorModule[ParamType]  with HasParams[ParamType] {

  val linear = nn.Linear(10, 5)

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    linear(input)
  }

}
class SimpleNetz[ParamType <: FloatNN: Default](input_size: Int, hidden_size: Int, output_size: Int) extends TensorModule[ParamType]  with HasParams[ParamType] {

  val conv1 = nn.Conv2d(1, 10, kernel_size=5) // 输出: (1, 10, 24, 24)
  val relu = nn.ReLU()
  val pool = nn.MaxPool2d(2) // 输出: (1, 10, 12, 12)
  // 错误：in_features 计算不正确或硬编码不当
  val fc1 = nn.Linear(10 * 12 * 12, 50) // 预期1440个特征

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    println(s"Input shape: ${input.shape}")
    var x = pool(relu(conv1(input)))
    println(s"Shape after conv/pool: ${x.shape}")
    // 不正确的展平尝试
    // x = x.view(-1, 10 * 10 * 10) // 如果运行，这将导致运行时错误
    // 正确的展平
    x = x.view(x.size(0), -1) // 展平除批次之外的所有维度
    println(s"Shape after flattening: ${x.shape}")
    // 现在 x 的形状是 [1, 1440]，因为 10 * 12 * 12 = 1440
    // 下面的 fc1 层预期有1440个特征，与展平后的匹配
    x = fc1(x)
    println(s"Input shape to fc1: ${x.shape}")
    println(s"fc1 expects input features: ${fc1.in_features}")
    x
//    try
//      x = fc1(x)
//      x
//    catch
//      case e: RuntimeException =>
//      println(s"\nError occurred: ${e}")
//      println(s"Input shape to fc1: ${x.shape}")
//      println(s"fc1 expects input features: ${fc1.in_features}")
//    finally
//      return x

  }




object lesson_08 {


//  @main
  def main(): Unit = {

    //01
    // 示例层
    val conv_layer = nn.Conv2d[Float32](in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
//    val linear_layer = nn.Linear(in_features =???, out_features = 10) // 问题：in_features 是什么？

    // 模拟输入数据
    val input_data = torch.randn(64, 3, 32, 32).to(torch.float32) // (批量大小, 输入通道数, 高, 宽)

    // 卷积层前向传播
    val conv_output = conv_layer(input_data)
    println(s"Conv output shape: ${conv_output.shape}")
    // 输出：卷积输出形状：torch.Size([64, 16, 32, 32])

    // 尝试直接传递给线性层（会失败）
    // output = linear_layer(conv_output) // 这会引发一个 RuntimeError

    // 正确方法需要展平
    val flattened_output = conv_output.view(conv_output.size(0), -1) // 展平除批量维度外的所有维度
    println(s"Flattened output shape: ${flattened_output.shape}")
    // 输出：展平后输出形状：torch.Size([64, 16384]) # 16 * 32 * 32 = 16384

    // 现在我们知道了线性层所需的 in_features
    val correct_linear_layer = nn.Linear[Float32](in_features = 16384, out_features = 10)
    val output = correct_linear_layer(flattened_output)
    println(s"Final output shape: ${output.shape}")
    // 输出：最终输出形状：torch.Size([64, 10])


    //02
    val device = if torch.cuda.is_available() then torch.Device("cuda")
                else torch.Device("cpu")
    println(s"Using device: ${device}")
    val model = nn.Linear[Float32](10, 5)
    val input_cpu = torch.randn(1, 10).to(torch.float32) // 默认在CPU上的张量
    // 将模型移至GPU（如果可用）
    model.to(device)
//    println(s"Model device: ${model.parameters().device}")
    // 尝试用CPU张量和GPU模型进行前向传播（如果设备是cuda则会失败）
    try {
      var output = model(input_cpu)
    }
    catch
      case e: RuntimeException =>
        println(s"Error: ${e}")
    // 输出可能是：错误：要求所有张量在同一设备上，
    // 但找到了至少两个设备，cuda:0 和 cpu！

    // 正确方法：将输入张量移至与模型相同的设备
    val input_gpu = input_cpu.to(device)//.to(torch.float32)
    println(s"Input tensor device: ${input_gpu.device}")
    val output_gpu = model(input_gpu) // 这可以正常运行
    println(s"Output tensor device: ${output_gpu.device}")
    println("Forward pass successful!")


    //03
    // 示例：CNN输出 -> 展平 -> 线性
    // 假设 conv_output 的形状为 [batch_size, channels, height, width]
    val num_classes = 10
    val flatten = nn.Flatten[Float32]() // 默认从维度1开始展平
    val flat_output = flatten(conv_output)
    // flat_output 形状：[batch_size, channels * height * width]

    // 计算线性层预期的特征数
    val num_features = flat_output.shape(1)
    val linear_layer2 = nn.Linear[Float32](num_features, num_classes)
    val output2 = linear_layer2(flat_output)

    //04
    val dummy_input = torch.randn(1, 1, 28, 28).to(torch.float32) // 示例输入：MNIST 图像
    val simplenet = SimpleNetz[Float32](28*28, 128, num_classes)
    simplenet(dummy_input)


    //05
    // 在 CPU 上创建的张量（默认）
    val cpu_tensor = torch.randn(2, 2)
    println(s"cpu_tensor 位于: ${cpu_tensor.device}")

    // 检查 GPU 是否可用并移动张量
    if torch.cuda.is_available() then
      val gpu_tensor = cpu_tensor.to(CUDA)
      println(s"gpu_tensor 位于: ${gpu_tensor.device}")
    else
      println("GPU 不可用，无法创建 gpu_tensor。")

    // 输出（如果 GPU 可用）：
    // cpu_tensor 位于: cpu
    // gpu_tensor 位于: cuda:0


    //实例化模型（最初在 CPU 上）
    val model2 = SimpleNet2()
    // 参数最初在 CPU 上
    
//    println(s"模型最初位于: ${model2.parameters().device}")

    // 如果 GPU 可用，将模型移动到 GPU
    if torch.cuda.is_available() then
      val device = torch.Device("cuda")
      model2.to(device)
//      println(s"模型已移至: ${model2.parameters().device}")
    else
      val device = torch.Device("cpu")
      println("GPU 不可用，模型仍留在 CPU 上。")

    // 输出（如果 GPU 可用）：
    // 模型最初位于: cpu
    // 模型已移至: cuda:0

    // 在训练循环中，在 loss.backward() 之后

    var total_norm = 0.0
    for(p <- model.parameters()) {
      if p.grad.isDefined then
        val param_norm = p.grad.get.detach().data.norm[Int](2) // 计算此参数梯度的 L2 范数
//        total_norm += math.pow(param_norm.item()) // ** 2 // 平方和
        total_norm =  math.sqrt(total_norm) // ** 0.5 // 平方和的平方根

        println(s"总梯度范数: $total_norm")
    }
    // 通常，你会使用 TensorBoard 或其他日志框架来记录此值








    //设置 SummaryWriter
    import torch.utils.tensorboard.SummaryWriter
    import torch.* // 假设 torch 已被导入

    // 创建一个 SummaryWriter 实例
    // 这将创建一个类似 'runs/experiment_name' 的目录
    // 如果未提供参数，则默认为 'runs/CURRENT_DATETIME_HOSTNAME'
    val log_dir = "runs/my_first_experiment"
    val writer = new SummaryWriter(log_dir)

    println(s"TensorBoard 日志目录: $log_dir")
    // 之后可以通过以下命令查看: tensorboard --logdir runs

//    writer.add_scalar(tag, scalar_value, global_step=None)


    model.to(device)

    
    val num_epochs = 100
    val optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    val dataset = DummyDataset(num_samples = 105)
    val train_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle = true, num_workers = 4)

    val valid_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle = true, num_workers = 4)
    val criterion = nn.CrossEntropyLoss()
    for (epoch <- 0 until num_epochs) {
      model.train() // 设置模型为训练模式
      var running_loss = 0.0
      var total_train_samples = 0

      for ((data,i )<-  train_loader.zipWithIndex) {
//        val data = train_loader(i)
        val inputs = data._1.to(device)
        val labels = data._2.to(device)
        optimizer.zero_grad()
        val outputs = model(inputs.to(torch.float32))
        val loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() // inputs.size(0) // 累加按批次大小加权的损失
        total_train_samples += inputs.size(0)
        // 每 N 次迭代（例如 100 次）记录批次损失
        val log_interval = 100
        if i % log_interval == log_interval - 1 then
          val current_step = epoch * train_loader.size + i
          val avg_batch_loss = running_loss / (log_interval * train_loader.batch_size) // 近似区间平均值
          writer.add_scalar("Loss / train_batch", avg_batch_loss, current_step)
        // 注意: 这只是一个示例记录方案
      }
      val epoch_loss = running_loss / total_train_samples // 该 epoch 的平均损失
      writer.add_scalar("Loss / train_epoch", epoch_loss, epoch)
      println(s"Epoch ${epoch + 1} / ${num_epochs}, 训练损失: ${epoch_loss}")

    }
    // --- 验证阶段 ---
    model.eval() // 设置模型为评估模式
    var validation_loss = 0.0
    var correct = 0f
    var total_val_samples = 0
    torch.no_grad { // 禁用梯度计算
      for(epoch <- (0 to num_epochs)){
        for ((inputs, labels) <- valid_loader) {
          //        val data = valid_loader(i)
          inputs.to(device)
          labels.to(device)
          val outputs = model(inputs.to(torch.float32))
          val loss = criterion(outputs, labels)
          validation_loss += loss.item() // inputs.size(0) // 累加按批次大小加权的损失

          val predicted = torch.max(outputs, 1)._2
          total_val_samples += labels.size(0)
          val l = predicted == labels
          
          correct = correct + (predicted == labels).sum().item().asInstanceOf[Float]

          val avg_val_loss = validation_loss / total_val_samples // 该 epoch 的平均损失
          val accuracy = 100.0 * correct / total_val_samples // 该 epoch 的准确率
          println(s"Epoch ${epoch + 1} / ${num_epochs}, 验证损失: ${avg_val_loss} , 准确率: ${accuracy}")
          writer.add_scalar("Loss / validation", avg_val_loss, epoch)
          writer.add_scalar("Accuracy / validation", accuracy, epoch)
        }
      }

    }
    // 训练完成后关闭写入器
    writer.close()
    println("训练完成。TensorBoard 日志已保存。")




  }

}
}