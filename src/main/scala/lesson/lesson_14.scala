package lesson

import org.bytedeco.pytorch.OptimizerParamState
import torch.nn.modules.{HasParams, TensorModule}
import torch.*
//  param_state('momentum_buffer').mul_(momentum).add_(grad) // v = mu*v + grad
//
//  // 获取更新后的动量缓冲区
//  val //"""实现带有动量的随机梯度下降。"""
//class CustomSGD(params: Iterable[Tensor[ParamType]], lr:Float=0.01, momentum=0.0, weight_decay=0.0) extends Optimizer{
//
//
//  override val optimizerParamState: OptimizerParamState = ???
//
//  override def step(): Unit = super.step()
//} = param_state('momentum_buffer')
//
//  // 执行参数更新步骤
//  // p = p - lr * momentum_buffer
//  p.add_(momentum_buffer, alpha=-lr)
//
//  return loss
//def __init__(params: Iterable[Tensor], lr:Float=0.01, momentum=0.0, weight_decay=0.0):
//
//momentum_buffer

class MyCustomModule[ParamType <: FloatNN: Default](input_features : Int, output_features : Int, hidden_units : Int) extends TensorModule[ParamType] with HasParams[ParamType] {

  // 定义子模块（层）
  val layer1 = nn.Linear(input_features, hidden_units)
  val activation = nn.ReLU()
  val layer2 = nn.Linear(hidden_units, output_features)

  var forward_count = 0
  // 直接定义可学习参数（如果需要）
  // 示例：一个可学习的缩放因子
  val scale = nn.Parameter("scale",torch.randn(1))

  // 定义不可学习的状态（缓冲区）
  // 示例：一个用于正向传播的计数器（仅作演示）
  register_buffer("forward_count", torch.zeros(1, dtype = torch.int64))

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {

    // 使用已初始化的组件定义计算流程
    var x = layer1(input)
    x = activation(x)
    x = layer2(x)
    // 使用自定义参数
    val nex = x * scale
    // 更新缓冲区（如果需要，请确保设备兼容性）
    // 注意：这样的直接修改在标准训练循环中可能不常见，
    // 但它展示了缓冲区的用法。
    forward_count += 1
    nex.to(this.paramType)
  }
}