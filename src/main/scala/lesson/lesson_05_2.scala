
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


// 假设 'YourCustomDataset' 已如前所示定义
// 或者使用内置数据集，例如 datasets.MNIST
// 为了演示，我们创建一个简单的虚拟数据集：
class DummyDataset[Input <: BFloat16 | FloatNN: Default, Target <: BFloat16 | FloatNN: Default](num_samples: Int = 100)
  extends Dataset[Input, Target] {

  override def get_batch(request: Long*): ExampleVector = ???

  val features = torch.randn(num_samples, 10) // 示例：100 个样本，10 个特征
  val labels = torch.randint(0, 2, Seq(num_samples)).to(dtype =implicitly[Default[Target]].dtype) // 示例：100 个二元标签

  override def targets: Tensor[Target] = labels

  override def length: Long = features.shape(0)

  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = (features(idx), labels(idx))

  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ??? //(0 until this.length).iterator.map(getItem)
}

// 返回可变长度张量的示例Dataset
class VariableSequenceDataset[Input <: BFloat16 | FloatNN: Default](data: Tensor[Input]*)
  extends Dataset[Input, Int64] {

  // data是一个张量列表，例如 [torch.randn(5), torch.randn(8), ...]

  // 为简单起见，假设每个项目也有一个标签（例如其长度）

  override def get_batch(request: Long*): ExampleVector = ???
  
  override def features: Tensor[Input] = ???

  override def targets: Tensor[Int64] = ???

  override def length: Long = data.length

  override def getItem(idx: Int): (Tensor[Input], Tensor[Int64]) = {
    val sequence = data(idx)
    val label = sequence.shape(0)
    (sequence, torch.tensor(label).to(torch.int64))
  }

  override def iterator: Iterator[(Tensor[Input], Tensor[Int64])] = ???
}


class SyntheticDataset[Input <: BFloat16 | FloatNN: Default, Target <: BFloat16 | FloatNN: Default](
  featurez: Tensor[Input],
  labels: Tensor[Target],
  transform: Option[TensorModule[Input]] = None
) extends Dataset[Input, Target] {
  """一个用于合成特征和标签的自定义数据集。"""


  """
        参数:
            features (Tensor): 包含特征数据的张量。
            labels (Tensor): 包含标签的张量。
            transform (callable, optional): 可选的样本转换。
        """
  // 基本检查，确保特征和标签具有相同的样本数量
  //  assert features.shape[0] == labels.shape[0], \
  "特征和标签的样本数量必须一致"
  // 存储特征和标签
  //  val features = features
  //  val labels = labels
  // 存储转换
  //  val transform = transform

  """
        根据给定索引获取特征向量和标签。
        参数:
            idx (int): 要获取的样本索引。
        返回:
            tuple: (特征, 标签)，其中 feature 是特征向量，label 是对应的标签。
        """
  override def get_batch(request: Long*): ExampleVector = ???
  
  override def features: Tensor[Input] = featurez

  override def targets: Tensor[Target] = labels

//  """返回样本总数。"""
  override def length: Long = features.shape(0)

  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = {
    // 获取原始特征和标签
    val feature_sample = features(idx)
    val label_sample = labels(idx)
    // 创建一个样本字典（或元组）
    var sample = Map( "feature" -> feature_sample, "label" -> label_sample)
//    if transform.isDefined then
//      sample = transform.get(sample)
    // 返回可能已转换的样本
    // 常见做法是分别返回特征和标签
//    (sample("feature"), sample("label"))
    (feature_sample,label_sample)
  }

  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???
}

object lesson_05_2 {

  // 自定义collate函数
  def pad_collate(batch: Seq[(Tensor[Float32], Tensor[Int64])]): (Tensor[Float32], Tensor[Int64]) = {
    // batch是一个元组列表：[(序列1, 标签1), (序列2, 标签2), ...]
    // 按序列长度对批次元素进行排序（可选，但通常为了RNN效率而进行）
    // batch.sort(key=lambda x: len(x[0]), reverse=True) // 对于填充不是严格必需的

    // 分离序列和标签
    val sequences = batch.map(_._1)
    val labels = batch.map(_._2)

    // 将序列填充到批次中最长序列的长度
    // `batch_first=True` 使输出形状变为 (batch_size, 最大序列长度, 特征)
    val padded_sequences = torch.pad_sequence(sequences, batch_first = true, padding_value = 0.0)

    // 堆叠标签（假设它们是简单的标量）
    val labels_tensor = torch.cat(labels)
    (padded_sequences, labels_tensor)
  }
//    @main
  def main(): Unit = {

      //01
      // 实例化数据集
      val dataset = DummyDataset(num_samples = 105)
      // 实例化 DataLoader
      // batch_size: 每个批次的样本数量
      // shuffle: 设置为 True 以在每个周期打乱数据（对训练很重要）
      val train_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle = true)
      // 迭代 DataLoader
      println(s"Dataset size: ${dataset.length}")
      println(s"DataLoader batch size: ${train_loader.batch_size}")
//      for( epoch <- Seq(1)) {
//        // 一个周期的示例
//        println(s"\n--- Epoch ${epoch + 1} ---")
//        // DataLoader 产生批次。每个 'batch' 通常是元组或列表
//        // 包含特征和标签的张量。 val features, labels = batch
//        for (((features, label), i) <- train_loader.zipWithIndex) {
//          println(s"Batch ${i + 1}: Features shape=${features.shape}, Labels shape=${label.shape}")
//          // 在这里你通常会执行训练步骤：
//          model.train()
//          optimizer.zero_grad()
//          outputs = model(features)
//          loss = criterion(outputs, labels)
//          loss.backward()
//          optimizer.step()
//        }
//      }

      //02
      // 如果数据集大小不能被批次大小整除，则丢弃最后一个不完整的批次
      val train_loader_drop_last = DataLoader(dataset = dataset, batch_size = 32, shuffle = true, drop_last = true)

      println("\n--- DataLoader with drop_last=True ---")
      for( (batch, i) <- train_loader_drop_last.zipWithIndex){
//        val batch = train_loader_drop_last(i)
        val (features, labels ) = batch
        println(s"Batch  ${i + 1}: Features shape=${features.shape}, Labels shape=${labels.shape}")
      }
      //预期输出 ：只有 3 个大小为 32 的批次 。最后的 9 个样本被丢弃 。

      //03
      // 使用 4 个工作进程加载数据
      // num_workers > 0 启用多进程数据加载
      // 一个常见的起始点是 num_workers = 4 * num_gpus，但最优值取决于
      // 系统（CPU 核心数、磁盘速度）和批次大小。通常需要通过实验确定。
      val fast_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle = true, num_workers = 4)

      // 迭代看起来相同，但数据加载发生在后台进程中
      for( (batch,i) <-  fast_loader.zipWithIndex) {
        val (features, labels) = batch
        println(s"Batch  ${i + 1}: Features shape=${features.shape}, Labels shape=${labels.shape}")
      }


      //04
      // 启用 pin_memory=True 以将数据加载到 GPU 内存中
      // 这对于小批量训练（如图像分类）非常重要
      // 因为它可以避免 CPU 到 GPU 的数据传输延迟
      val gpu_optimized_loader = DataLoader(dataset = dataset,
        batch_size = 32,
        shuffle = true,
        num_workers = 4,
        pin_memory = true)

      // 在训练循环内部（假设你有 GPU）
      for ((features, labels) <- gpu_optimized_loader) {
//        val batch = gpu_optimized_loader(i)
//        var (features, labels) = batch
        features.to(device = CUDA) // 传输变得更快
        labels.to( device = CUDA)
        // ...其余训练步骤...
      }


      //05
      // 假设“dataset”是你的torch.utils.data.Dataset实例
      // 假设“targets”是一个列表或张量，包含每个样本的类别标签
      // e.g., targets = [0, 0, 1, 0, ..., 1, 0]

      // 为每个样本计算权重
      val targets = Seq(900,100) //torch.randint(0, 2, Seq(900,100)).to(torch.int64) // 示例：1000 个二元标签
      val class_counts = torch.bincount(torch.tensor(targets)) // 各类别的计数：例如 [900, 100]
      val num_samples = targets.size // 总样本数：1000

      // 每个样本的权重是 1 / (其所属类别的样本数)
      var sample_weight = targets.map(t =>  {
        val t1 = torch.tensor(1.0)/ class_counts(t)
        t1
      }) //.toArray
      val sample_weights = torch.cat(sample_weight )

      // 创建采样器
      val sampler = WeightedRandomSampler(weights = sample_weights, num_samples = num_samples, replacement = true)

      // 使用自定义采样器创建DataLoader
      // 注意：使用采样器时，shuffle必须为False
      val dataloader = DataLoader(dataset, batch_size = 32, sampler = sampler)

      // 现在，从这个dataloader中抽取的批次将随着时间推移，在类别表示上更加平衡。
      // for batch_features, batch_labels in dataloader:
      //     // 训练步骤...
      //     pass
  }

}