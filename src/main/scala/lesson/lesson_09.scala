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
import torch.*

import java.net.URL
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPInputStream
import scala.collection.mutable.SortedMap as OrderedDict
import scala.collection.{mutable, Set as KeySet}
import scala.util.*

object lesson_09 {


//  @main
  def main(): Unit = {


    //02
    // 创建一个张量
    val t = torch.arange(end =12, dtype = torch.float32).view(3, 4)
    println(f"张量 t:\n ${t}")
    println(f"形状: ${t.shape}")
    println(f"步幅: ${t.strides()}")

    //03
    val t_transposed = t.t()
    println(f"\n转置后的张量 t_transposed:\n ${t_transposed}")
    println(f"形状: ${t_transposed.shape}")
    println(f"步幅: ${t_transposed.strides()}")
    println(f"t_transposed 是否连续？ ${t_transposed.is_contiguous()}") // 输出: False
    println(f"t_transposed 是否与 t 共享存储？ ${t_transposed.storage().data_ptr() == t.storage().data_ptr()}") // 输出: True

    //04
    // 这会引发 RuntimeError，因为 t_transposed 不连续
    // flat_view = t_transposed.view(-1)

    // .contiguous() 在需要时会创建一个具有连续内存布局的新张量
    val t_contiguous_copy = t_transposed.contiguous()
    println(f"\n连续副本是否连续？ ${t_contiguous_copy.is_contiguous()}") // 输出: True
    println(f"连续副本的步幅: ${t_contiguous_copy.strides()}") // 输出: (3, 1)
    println(f"存储共享？ ${t_contiguous_copy.storage().data_ptr() == t_transposed.storage().data_ptr()}") // 输出: False (这是一个副本)

    // 现在视图可以工作了
    val flat_view = t_contiguous_copy.view(-1)
    println(f"展平视图: {flat_view}")

    //05
    // 需要梯度的输入张量
    val a = torch.tensor(Seq(2.0, 3.0), requires_grad = true)
    // 操作 1: 乘以 3
    val b = a * 3
    // 操作 2: 计算均值
    val c = b.mean()
    // 检查 grad_fn 属性
    println(f"Tensor a: requires_grad=${a.requires_grad}, grad_fn=${a.grad_fn}")
    // 预期输出: 张量 a: requires_grad=True, grad_fn=None
    println(f"Tensor b: requires_grad=${b.requires_grad}, grad_fn=${b.grad_fn}")
    // 预期输出: 张量 b: requires_grad=True, grad_fn=<MulBackward0 object at 0x...>
    println(f"Tensor c: requires_grad=${c.requires_grad}, grad_fn=${c.grad_fn}")
    // 预期输出: 张量 c: requires_grad=True, grad_fn=<MeanBackward0 object at 0x...>

    //06
    // 示例设置
    val w = torch.randn(Seq(5, 3), requires_grad = true)
    val x2 = torch.randn(1, 5)
    // 如果 x 仅是输入数据，确保它不需要梯度
    // x.requires_grad_(False) // 或在创建时不设置 requires_grad
    val y2 = x2 @@ w // y 通过矩阵乘法依赖于 w
    val z2 = y2.mean() // z 是从 y 派生的标量
    // 从 z 开始计算梯度
    z2.backward()

    // 梯度现在填充在 w.grad 中
    // d(z)/dw 梯度被计算并存储
    println(w.grad.get.shape)
    // 输出: torch.Size([5, 3])

    //07
    // 我们需要中间张量来检查它们的 grad_fn
    val w2 = torch.randn(Seq(5, 3), requires_grad = true)
    val x3 = torch.randn(1, 5)
    val y3 = x3 @@ w2 // y 通过矩阵乘法依赖于 w2
    val z3 = y3.mean()

    println(f"y3 源自: ${y3.grad_fn}")
    // 输出: y3 originated from: <MmBackward0 object at 0x...>
    println(f"z3 源自: ${z3.grad_fn}")
    // 输出: z3 originated from: <MeanBackward0 object at 0x...>
    // 用户创建的叶张量没有 grad_fn
    println(f"w2.grad_fn: ${w2.grad_fn}")
    // 输出: w2.grad_fn: None
    println(f"x3.grad_fn: ${x3.grad_fn}")
    // 输出: x3.grad_fn: None


    //08
    // 假设已定义 model、optimizer、criterion、dataloader

//    for((inputs, targets) <- dataloader){
//      // 1. 重置上一迭代的梯度
//      optimizer.zero_grad()
//
//      // 2. 执行正向传播
//      val outputs = model(inputs)
//      val loss = criterion(outputs, targets)
//
//      // 3. 执行反向传播以计算梯度
//      loss.backward()
//
//      // 4. 使用计算出的梯度更新模型参数
//      optimizer.step()



    //01
    // 创建一个张量
    val x = torch.arange(end =12, dtype = torch.float32)
    println(f"原始张量 x: ${x}")

    // 存储是一个包含12个浮点数的一维数组
    println(f"存储元素: ${x.storage().data()}")
    println(f"存储类型: ${x.storage().data_ptr}")
    println(f"存储大小: ${x.storage().nbytes()}")

    // 通过重塑创建视图
    val y = x.view(3, 4)
    println(f"\n重塑后的张量 y:\n${y}")

    // y 具有不同的形状/步幅，但共享相同的存储
    println(f"y 是否与 x 共享存储？ ${y.storage().data_ptr() == x.storage().data_ptr()}")

    // 修改视图会影响原始张量（反之亦然）
//    y(0, 0) = 99.0
//    println(f"\n修改后的 y:\n${y}")
//    println(f"修改 y 后的原始 x: ${x}")

  }

}
