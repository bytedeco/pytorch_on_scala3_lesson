package lesson

import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{modules, functional as F}
import torch.{Tensor, *}

import scala.util.control.Breaks.{break, breakable}

class ControlFlowModel[ParamType <: FloatNN: Default](num_classes: Int = 10)
    extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val linear1 = nn.Linear(10, 5)
  val linear2 = nn.Linear(5, 1)

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = torch.relu(linear1(input))
    // 数据依赖的控制流
    val mean = x.mean().data_ptr_double
    if x.mean() >> 0.5 then linear2(x)
    else torch.zeros_like(linear2(x))

  }
}
class SimpleModel[ParamType <: FloatNN: Default](num_classes: Int = 10)
    extends TensorModule[ParamType] {

  val linear = nn.Linear(10, 5)

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    // 简单、直接的计算
    torch.relu(linear(x))
  }
}
object lesson_12 {

  //  @main
  def main(): Unit = {}
}
