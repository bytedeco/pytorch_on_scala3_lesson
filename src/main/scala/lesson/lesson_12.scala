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
import scala.collection.mutable.SortedMap as OrderedDict
import scala.collection.{mutable, Set as KeySet}
import scala.util.*

class ControlFlowModel[ParamType <: FloatNN: Default](num_classes: Int=10)  extends TensorModule[ParamType] with HasParams[ParamType]  {

  val linear1 = nn.Linear(10, 5)
  val linear2 = nn.Linear(5, 1)

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = torch.relu(linear1(input))
    // 数据依赖的控制流
    val mean = x.mean().data_ptr_double
    if x.mean() >> 0.5 then
      linear2(x)
    else
      torch.zeros_like(linear2(x))

  }
}
class SimpleModel[ParamType <: FloatNN: Default](num_classes: Int=10) extends TensorModule[ParamType]  {

  val linear = nn.Linear(10, 5)

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
  // 简单、直接的计算
    torch.relu(linear(x))
  }
}
object lesson_12 {


  //  @main
  def main(): Unit = {


  }
}