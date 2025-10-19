
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
import torch.nn.loss.LossFunc
import torch.{&&, ---, ::, BFloat16, DType, Default, Device, Float32, FloatNN, Int64, Slice, Tensor, nn}
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
import torch.utils.tensorboard.SummaryWriter

import scala.collection.mutable.SortedMap as OrderedDict
import torch.numpy.TorchNumpy as np
import torch.optim

class SimpleMLP2[ParamType <: FloatNN: Default] extends TensorModule[ParamType]  with HasParams[ParamType] {

  val layer1 = nn.Linear(784, 128) // 输入 784，输出 128
  val activation = nn.ReLU()
  val layer2 = nn.Linear(128, 64) // 输入 128，输出 64
  // layer3 的输入大小不正确 - 应该是 64
  val layer3 = nn.Linear(64, 10) // 输入 64，输出 10

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = layer1(input)
    x = activation(x)
    x = layer2(x)
    x = activation(x)
    // 这行代码将导致错误
    x = layer3(x)
    x
  }
}


object lesson_08_2 {

  // 假设模型、val_dataloader、loss_fn 已定义

  def evaluate_model[ParamType <: FloatNN: Default](model: TensorModule[ParamType], val_dataloader: DataLoader[ParamType], loss_fn: LossFunc, device: Device): (Double, Float) = {
    model.eval() // 设置模型为评估模式
    var running_loss = 0.0f
    var correct_predictions = 0f
    var total_samples = 0
    torch.no_grad{
      for ((inputs, labels)<- val_dataloader){
        inputs.to(device)
        labels.to(device)
        // 前向传播
        var outputs = model(inputs)
        // 计算损失
        var loss = loss_fn(outputs)(labels)
        // --- 记录步骤 ---
        running_loss = running_loss + loss.item().asInstanceOf[Float] * inputs.size(0)
        val predicted = torch.max(outputs, 1)._2
        total_samples += labels.size(0)
        correct_predictions = correct_predictions + (predicted == labels).sum().item().asInstanceOf[Float]
        // --- 记录步骤结束 ---
      }


    }
    // 计算当前轮次的平均损失和准确度
    val epoch_loss = running_loss / total_samples
    val epoch_acc = correct_predictions / total_samples

    println(s"Validation: Loss: ${epoch_loss}, Accuracy: ${epoch_acc}")
    (epoch_loss, epoch_acc)
  }



  // 假设模型、train_dataloader、loss_fn、optimizer 已定义

  def train_one_epoch[D <: FloatNN : Default](model: TensorModule[D], train_dataloader: DataLoader[D], loss_fn: LossFunc, optimizer: optim.Optimizer, device: Device): (Double, Float) = {
    model.train()// 设置模型为训练模式
    var running_loss = 0.0f
    var correct_predictions = 0f
    var total_samples = 0
    for (((inputs, labels),batch_idx) <- train_dataloader.zipWithIndex){
      inputs.to(device)
      labels.to(device)
      // 1. 清零梯度
      optimizer.zero_grad()
      // 2. 前向传播
      var outputs = model(inputs)
      // 3. 计算损失
      var loss:Tensor[D] = loss_fn(outputs)(labels)
      // 4. 反向传播
      loss.backward()
      // 5. 优化器步骤
      optimizer.step()
      // --- 记录步骤 ---
      // 累加损失（使用 .item() 获取 Python 数字）
      running_loss = running_loss + loss.item().asInstanceOf[Float] * inputs.size(0) // 按批次大小加权
      // 累加准确度（分类示例）
      val predicted = torch.max(outputs, 1)._2
      total_samples += labels.size(0)
      correct_predictions = correct_predictions + (predicted == labels).sum().item().asInstanceOf[Float]
      // --- 记录步骤结束 ---

    }
    // 计算当前轮次的平均损失和准确度
    val epoch_loss = running_loss / total_samples
    val epoch_acc = correct_predictions / total_samples

    println(s"Training Epoch: Loss: ${epoch_loss}, Accuracy: ${epoch_acc}")

    // 返回指标以供后续记录或分析
    (epoch_loss, epoch_acc)

  }


//  @main
  def main(): Unit = {

    //03
    // --- 在你的主训练脚本中 ---
    val num_epochs = 10
    val device = if torch.cuda.is_available() then torch.Device("cuda") else torch.Device("cpu")

    // ... 初始化模型、数据、损失函数、优化器 ...

    // 存储指标
    var train_losses = List.empty[Double]
    var train_accuracies = List.empty[Double]
    var val_losses = List.empty[Double]
    var val_accuracies = List.empty[Double]
    val model = nn.Linear[Float32](10, 2)
    val optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    val dataset = DummyDataset(num_samples = 105)
    val train_dataloader = DataLoader(dataset = dataset, batch_size = 32, shuffle = true, num_workers = 4)

    val val_dataloader = DataLoader(dataset = dataset, batch_size = 32, shuffle = true, num_workers = 4)
    val loss_fn = nn.CrossEntropyLoss()
    for(epoch <- (0 until num_epochs)){
      println(f"--- Epoch {epoch+1}/{num_epochs} ---")
      val (train_loss, train_acc) = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
      val (val_loss, val_acc) = evaluate_model(model, val_dataloader, loss_fn, device)

      // 存储指标
      train_losses = train_losses :+ train_loss
      train_accuracies = train_accuracies :+ train_acc
      val_losses = val_losses :+ val_loss
      val_accuracies = val_accuracies :+ val_acc

      // 可选：根据验证性能在此处保存模型检查点
      // 例如：
      // if val_acc > best_val_acc:
      //     best_val_acc = val_acc
      //     torch.save(model.state_dict(), 'best_model.pth')

      // 可选：将指标记录到 TensorBoard（使用上述存储的值）
      // writer.add_scalar('Loss/train', train_loss, epoch)
      // writer.add_scalar('Loss/validation', val_loss, epoch)
      // ... 等等。

      println("Training finished.")
      // 现在你可以分析这些列表：train_losses、val_losses 等。
      // 例如，将它们保存到文件或绘制图表。
    }



    //04
    // 1. 设置 TensorBoard 写入器
    // 日志文件将保存在 'runs/simple_experiment' 目录中
    val writer = new SummaryWriter("runs/simple_experiment")

    // 2. 定义一个简单模型、损失函数和优化器
    val model2 = nn.Linear[Float32](10, 2) // 简单的线性模型
    val criterion = nn.MSELoss()
    val optimizer2 = optim.SGD(model2.parameters(), lr = 0.01)

    // 模拟一个简单数据集
    val inputs = torch.randn(100, 10) // 100 个样本，10 个特征
    val targets = torch.randn(100, 2) // 100 个样本，2 个输出值

    // 3. 简单训练循环
    println("开始模拟训练...")
    val num_epochs_2 = 50
    for (epoch <- (0 until num_epochs_2)){
      optimizer2.zero_grad() // 梯度清零
      val outputs = model2(inputs.to(torch.float32)) // 前向传播
      val loss = criterion(outputs, targets) // 计算损失
      // 模拟损失变化（在实际训练中替换为真实损失）
      // 为演示目的，让损失随周期递减
      val simulated_loss = loss + torch.randn(1) * 0.1 + (num_epochs - epoch) / num_epochs
      simulated_loss.backward() // 反向传播（使用模拟损失进行演示）
      optimizer2.step() // 更新权重
      // 4. 将指标记录到 TensorBoard
      if (epoch + 1) %5 == 0 then //// 每 5 个周期记录一次
        // 记录标量“损失”值
        writer.add_scalar("Training / Loss", simulated_loss.item().asInstanceOf[Float], epoch)
        // 记录模型权重分布（以线性层为例）
        writer.add_histogram("Model / Weights", model.weight.tolist(), epoch)
        writer.add_histogram("Model / Bias", model.bias.tolist(), epoch)
        println(s"周期[ ${epoch + 1} / ${num_epochs}]，模拟损失${simulated_loss.item()}")
//        time.sleep(0.1)
      //模拟训练时间

    }
    // 5. 添加模型图（可选）
    // 确保输入形状与模型预期一致
    // writer.add_graph(model, inputs[0].unsqueeze(0)) // 提供一个样本输入批次

    // 6. 关闭写入器
    writer.close()
    println("模拟训练完成。TensorBoard 日志已保存到 'runs/simple_experiment'。")
    println("在你的终端中运行 'tensorboard --logdir=runs' 来查看。")

  }

}
