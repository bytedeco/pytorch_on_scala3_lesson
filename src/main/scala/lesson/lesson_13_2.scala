//package lesson
//
//import org.bytedeco.javacpp.{FloatPointer, PointerScope}
//import org.bytedeco.pytorch
//import org.bytedeco.pytorch.global.torch as torchNative
//import org.bytedeco.pytorch.*
//import torch.Device.{CPU, CUDA}
//import torch.internal.NativeConverters.{fromNative, toNative}
//import torch.nn.modules.{HasParams, TensorModule}
//import torch.nn.{modules, functional as F}
//import torch.numpy.TorchNumpy as np
//import torch.optim.Adam
//import torch.utils.data.dataloader.*
//import torch.utils.data.datareader.ChunkDataReader
//import torch.utils.data.dataset.*
//import torch.utils.data.dataset.custom.{FashionMNIST, MNIST}
//import torch.utils.data.sampler.RandomSampler
//import torch.utils.data.*
//import torch.{BFloat16, Tensor, *}
//
//import java.net.URL
//import java.nio.file.{Files, Path, Paths}
//import java.util.zip.GZIPInputStream
//import scala.collection.mutable.SortedMap as OrderedDict
//import scala.collection.{mutable, Set as KeySet}
//import scala.util.*
//
//// 假设分布式环境已初始化（rank, world_size 等）
//// dist.init_process_group(backend="nccl")
//// torch.cuda.set_device(local_rank) // local_rank 通常获取
//
//class LargeTransformerBlock[ParamType <: FloatNN: Default](dim: Int, ff_dim: Int) extends TensorModule[ParamType] {
//  // 子模块定义示例
//  val layer_norm = nn.LayerNorm(dim)
//  val attention = nn.MultiheadAttention(dim, num_heads = 8) // Simplified
//  val ffn = nn.Sequential(
//    nn.Linear(dim, ff_dim),
//    nn.ReLU(),
//    nn.Linear(ff_dim, dim)
//  )
//
//  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
//
//  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
//    var x = layer_norm(input + attention(input, input, input)(0))
//    x = x + ffn(x)
//    x
//  }
//
//}
//class BigModel[ParamType <: FloatNN: Default](num_layers: Int, dim: Int, ff_dim: Int, vocab_size: Int) extends TensorModule[ParamType] {
//  // 子模块定义示例
//  val embedding = nn.Embedding(vocab_size, dim)
//  val layers = nn.ModuleList(
//    (0 until num_layers).map(_ => LargeTransformerBlock(dim, ff_dim))*
//  )
//  val output_head = nn.Linear(dim, vocab_size)
//  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
//
//  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
//    var x = embedding(input)
//    for(layer <- layers){
//      x = layer(x)
//    }
//    x = output_head(x)
//    x
//  }
//}
//
//// --- 模拟模型和数据集 ---
//class ToyModel[ParamType <: FloatNN: Default] extends TensorModule[ParamType] {
//
//  val linear = nn.Linear(10, 1)
//
//  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
//
//  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
//    linear(input)
//  }
//
//}
//
//class ToyDataset[Input <: BFloat16 | FloatNN: Default, Target <: BFloat16 | FloatNN: Default](num_samples: Int = 100) extends Dataset[Input, Target] {
//
//  val features = torch.randn(num_samples, 10)
//  val labels = torch.randn(num_samples, 1).to[Tensor[Target]] //.to(dtype =implicitly[Default[Target]].dtype)
//
////  override def features: Tensor[Input] = features
//  override def targets: Tensor[Target] = labels
//
//  override def length: Long = num_samples
//
//  override def getItem(idx:  Int): (Tensor[Input], Tensor[Target]) = (features(idx), labels(idx))
//
//  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???
//}
////--- 模拟结束 ---
//object lesson_13_2 {
//
//
//  def main():Unit={
//
//  }
//}
