package lesson

import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.*
import torch.Device.{CPU, CUDA}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.numpy.TorchNumpy as np
import torch.optim.Adam
import torch.utils.data.dataloader.*
import torch.utils.data.datareader.ChunkDataReader
import torch.utils.data.dataset.*
import torch.utils.data.dataset.custom.{FashionMNIST, MNIST}
import torch.utils.data.sampler.RandomSampler
import torch.utils.data.*
import torch.*

import java.net.URL
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPInputStream
import scala.collection.mutable.SortedMap
import scala.collection.{mutable, Set as KeySet}
import scala.util.*

object lesson_03 {

//  @main
  def main(): Unit = {
    


   //01
   // Tensors that require gradients
    val x = torch.tensor(2.0, requires_grad = true)
    val w = torch.tensor(3.0, requires_grad = true)
    val b = torch.tensor(1.0, requires_grad = true)
    // Operations
    val y = w * x // Intermediate result 'y'
    val z = y + b // Final result 'z'
    println(f"Result z: ${z}")

    //02
    // 默认行为：requires_grad 为 False
    val x2 = torch.tensor(Seq(1.0, 2.0, 3.0))
    println(f"Tensor x: ${x2}")
    println(f"x.requires_grad: ${x2.requires_grad}")

    // 显式创建另一个张量并将 requires_grad 设置为 False
    val y2 = torch.tensor(Seq(4.0, 5.0, 6.0), requires_grad=false)

    println(f"\nTensor y: ${y2}")
    println(f"y.requires_grad: ${y2.requires_grad}")

    //03
    // 在创建时启用梯度追踪
    val w2 = torch.tensor(Seq(0.5, -1.0), requires_grad = true)
    println(f"Tensor w: ${w2}")
    println(f"w2.requires_grad: ${w2.requires_grad}")


    //04
    // 在创建后启用梯度追踪
    val b2 = torch.tensor(Seq(0.1))
    println(f"Tensor b (before): ${b}")
    println(f"b2.requires_grad (before): ${b2.requires_grad}")

    // 在创建后启用梯度追踪
//    b2.requires_grad_(true)
    println(f"\nTensor b (after): ${b2}")
    println(f"b2.requires_grad (after): ${b2.requires_grad}")

    //05
    // 尝试对整数张量设置 requires_grad
    try
      val int_tensor = torch.tensor(Seq(1, 2), dtype = torch.int64, requires_grad = true)
    // 这一行可能不会立即出错，但后续涉及它的 backward() 调用会出错。
      println(f"Integer tensor created with requires_grad=True: ${int_tensor.requires_grad}")
    // 让我们尝试一个简单的操作，这可能会在以后导致问题
      val result = int_tensor * 2.0 // 乘以浮点数看看是否会引起问题
      println(f"Result requires_grad: ${result.requires_grad}")
    // 如果我们尝试反向传播，这很可能会失败
    // result.backward()
    catch
      case e: RuntimeException =>
      println(f"\n对整数张量设置 requires_grad 时出错: {e}")

    //最佳实践
    //对需要梯度的参数 / 计算使用浮点张量
    val float_tensor = torch.tensor(Seq(1.0, 2.0), requires_grad = true)
    println(f"\n已创建 requires_grad=True 的浮点张量: ${float_tensor.requires_grad}")


    //06
    // 定义张量：x（输入）、w（权重）、b（偏置）
    val x3 = torch.tensor(Seq(1.0, 2.0)) // 输入数据，不需要梯度
    val w3 = torch.tensor(Seq(0.5, -1.0), requires_grad = true) // 权重参数，追踪梯度
    val b3 = torch.tensor(Seq(0.1), requires_grad = true) // 偏置参数，追踪梯度

    println(f"x3 requires_grad: ${x3.requires_grad}")
    println(f"w3 requires_grad: ${w3.requires_grad}")
    println(f"b3 requires_grad: ${b3.requires_grad}")

    // 执行操作：y = w * x + b
    // 注意：PyTorch 处理 b 的广播
    val intermediate = w * x
    println(f"\nintermediate (w * x) requires_grad: ${intermediate.requires_grad}")

    val y3 = intermediate + b3
    println(f"y3 requires_grad: ${y3.requires_grad}")

    //07
    println(f"\nx3.grad_fn: ${x3.grad_fn}")
    println(f"w3.grad_fn: ${w3.grad_fn}")
    println(f"b3.grad_fn: ${b3.grad_fn}")
    println(f"intermediate.grad_fn: ${intermediate.grad_fn}") // 乘法的结果
    println(f"y3.grad_fn: ${y3.grad_fn}") // 加法的结果


    //08
    // 示例设置（想象这些是模型的结果）
    val x4 = torch.tensor(2.0, requires_grad = true)
    val w4 = torch.tensor(3.0, requires_grad = true)
    val b4 = torch.tensor(1.0, requires_grad = true)

    // 执行一些操作（构建图）
    val y4 = w4 * x4 + b4 // y = 3*2 + 1 = 7
    val loss = y4 * y4 // loss = 7*7 = 49 (a scalar)

    // 反向传播之前，梯度为 None
    println(f"Gradient for x before backward: ${x4.grad}")
    println(f"Gradient for w before backward: ${w4.grad}")
    println(f"Gradient for b before backward: ${b4.grad}")

    // 计算梯度
    loss.backward()

    // 反向传播之后，梯度被填充
    println(f"Gradient for x after backward: ${x4.grad}") // d(loss)/dx = d(y^2)/dx = 2*y*(dy/dx) = 2*y*w = 2*7*3 = 42
    println(f"Gradient for w after backward: ${w4.grad}") // d(loss)/dw = d(y^2)/dw = 2*y*(dy/dw) = 2*y*x = 2*7*2 = 28
    println(f"Gradient for b after backward: ${b4.grad}") // d(loss)/db = d(y^2)/db = 2*y*(dy/db) = 2*y*1 = 2*7*1 = 14


    //09
    // 继续前面的例子，但使用非标量 y
    val x_vector = torch.tensor(Seq(2.0, 4.0), requires_grad = true)
    x_vector.requires_grad_(true)
    val w5 = torch.tensor(3.0, requires_grad = true)
    val b5 = torch.tensor(1.0, requires_grad = true)

    // y_non_scalar 现在是非标量张量，包含两个元素：[7.0, 13.0]
    val y_non_scalar = w * x_vector + b

    try
      y_non_scalar.backward()
    //这将导致错误
    catch
      case e: RuntimeException =>
        println(f"Error calling backward() on non-scalar: {e}")

      // 要使其工作，需要提供一个与 y_non_scalar 形状匹配的梯度张量
      // 这代表了某个最终损失相对于 y_non_scalar 的梯度。
      // 为演示目的，我们使用 torch.ones_like(y_non_scalar)
      val grad_tensor = torch.ones_like(y_non_scalar)
      y_non_scalar.backward(gradient = grad_tensor)
      println(f"Gradient for x_vector after y_non_scalar.backward(gradient=...): ${x_vector.grad}")
      println(f"Gradient for w after y_non_scalar.backward(gradient=...): ${w.grad}")









  }
}
