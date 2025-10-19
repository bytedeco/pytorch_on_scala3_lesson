package lesson

import torch.Device.{CPU, CUDA}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{modules, functional as F}
import torch.{Tensor, *}

// 定义一个简单的 CNN 模型
class SimpleCNN[ParamType <: FloatNN: Default]
    extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1)
  val relu = nn.ReLU()
  val pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
  val fc1 = nn.Linear(16 * 16 * 16, 10) // 假设输入图像为 32x32
  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = pool(relu(conv1(input)))
    x = torch.flatten(x, 1) // 展平除批次维度外的所有维度
    x = fc1(x)
    x
  }
}
object lesson_13 {

  //  @main
  def main(): Unit = {
    // 实例化模型并设置为评估模式
    val model = SimpleCNN()
    model.eval()

    // 创建与预期维度匹配的虚拟输入（批大小、通道、高度、宽度）
    // 注意：这里批大小设置为 1，但我们会使其变为动态
    val dummy_input = torch.randn(1, 3, 32, 32, requires_grad = false)

    // 定义输入和输出名称
    val input_names = List("input_image")
    val output_names = List("output_logits")

    // 定义动态轴（使批大小动态化）
    val dynamic_axes_config = Map(
      "input_image" -> Map(0 -> "batch_size"), // 输入的可变批大小
      "output_logits" -> Map(0 -> "batch_size") // 输出的可变批大小
    )

    // 指定输出文件路径
    val onnx_model_path = "simple_cnn.onnx"

    // 导出模型
//    torch.onnx.export (
//      model,
//      dummy_input,
//      onnx_model_path,
//      export_params = true,
//      opset_version = 12, // 选择合适的 opset 版本
//      do_constant_folding = true,
//      input_names = input_names,
//      output_names = output_names,
//      dynamic_axes = dynamic_axes_config
//    )

    println(s"模型已导出到 $onnx_model_path")

  }

}

//class ColumnParallelLinear[ParamType <: FloatNN: Default](input_size: Int, output_size: Int, bias: Boolean = true) extends nn.Module {
//
//  val world_size = get_tensor_model_parallel_world_size()
//  //确保 output_size 可以被 world_size 整除
//  //  assert output_size % world_size == 0
//  val output_size_per_partition = output_size // world_size
//  val input_size = input_size
//
//  // 权重矩阵沿输出维度（列）拆分
//  val weight = nn.Parameter(torch.empty(output_size_per_partition, input_size))
//  // 初始化权重...（例如，使用 init.kaiming_uniform_）
//
//  if bias then
//    // 偏置也沿输出维度拆分
//    val bias = nn.Parameter(torch.empty(output_size_per_partition))
//  // 初始化偏置...（例如，使用 init.zeros_）
//  else
//    register_parameter("bias", None)
//
//  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
//
//  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
//    // 如果前一层输出已复制（例如 LayerNorm），
//    // 输入可能需要广播或已可用。
//    // 此函数处理必要的通信。
//    val parallel_input = copy_to_tensor_model_parallel_region(input)
//
//    // 执行局部矩阵乘法
//    val output_parallel = nn.functional.linear(parallel_input, weight, bias)
//
//    // 从张量并行组中的所有GPU收集结果
//    // 沿列维度拼接。
//    val output_ = gather_from_tensor_model_parallel_region(output_parallel)
//
//    output_
//  }
//}
