package lesson

import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.*
import torch.Device.{CPU, CUDA}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{modules, functional as F}
import torch.numpy.TorchNumpy as np
import torch.optim.Adam
import torch.optim.lr_scheduler.{CosineAnnealingLR, CosineAnnealingWarmRestarts, LambdaLR}
import torch.utils.data.dataloader.*
import torch.utils.data.datareader.ChunkDataReader
import torch.utils.data.dataset.*
import torch.utils.data.dataset.custom.{FashionMNIST, MNIST}
import torch.utils.data.sampler.RandomSampler
import torch.utils.data.*
import torch.{Tensor, *}

import java.net.URL
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPInputStream
import scala.annotation.StaticAnnotation
import scala.collection.mutable.{ListBuffer, SortedMap as OrderedDict}
import scala.collection.{mutable, Set as KeySet}
import scala.util.*
//class LargeTextFileDataset(file_path: String, tokenizer: (String) => Tensor[ParamType]) extends IterableDataset{
//
//
//
//
//  override def getItem(idx: Int): (Tensor[Float32], Tensor[Float32]) = {
//
//    // 迭代器在此处为每个 epoch/worker 创建
//    val file_iterator = io.Source.fromFile(file_path).getLines()
//    // map 函数将处理函数应用于迭代器中的每一行
//    return file_iterator.map(tokenizer)
//  }
//}
//
//class ShardedLargeFileDataset(file_path: String, processor_fn: (String) => Tensor[ParamType]) extends IterableDataset {
//
//  // 如果需要分片，确定文件大小或行/记录数量
//  // self.num_records = self._get_num_records(file_path) # 辅助函数示例
//
//
//  def _get_records_iterator(self):
//  // 将此替换为遍历您的特定数据记录/文件的逻辑
//  with io.Source.fromFile(file_path) as f:
//    for line in f.getLines():
//      yield line
//  #生成原始记录
//
//  def __iter__(self):
//
//  val worker_info = get_worker_info()
//  val record_iterator = _get_records_iterator()
//
//  if worker_info is None then // 单进程加载
//    val worker_id = 0
//    val num_workers = 1
//  else// 多进程加载
//    val worker_id = worker_info.id
//    val num_workers = worker_info.num_workers
//
//  // 基础工作进程分片：每个工作进程处理每第 N 条记录
//  // 更复杂的分片可能涉及字节偏移量或文件拆分
//  val sharded_iterator = (record
//  for i
//  , record in enumerate(record_iterator)
//  if i % num_workers == worker_id
//  )
//
//  // 在工作进程的迭代器链中应用处理
//  val processed_iterator = sharded_iterator.map(processor_fn)
//  processed_iterator
//
//}
// 1. 定义一个简单模型
class SimpleCNN21[ParamType <: FloatNN: Default](num_classes: Int=10) extends TensorModule[ParamType] {

  val conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
  val relu = nn.ReLU()
  val pool = nn.MaxPool2d(kernel_size=2, stride=2)
  val conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
  // 为全连接层展平特征
  val fc = nn.Linear(64 * 16 * 16, num_classes) // 假设输入图像为 32x32

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] ={
    var x = pool(relu(conv1(input)))
    x = pool(relu(conv2(x)))
    x = torch.flatten(x, 1) // 展平除批次维度外的所有维度
    x = fc(x)
    x
  }
}
object lesson_11_2 {

}
