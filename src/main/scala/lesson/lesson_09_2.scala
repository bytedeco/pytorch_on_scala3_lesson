package lesson

import org.bytedeco.pytorch.{
  TensorBase,
  TensorExampleVectorIterator,
  TensorTensorHook,
  VoidTensorHook,
  Tensor as NativeTensor
}
import torch.Device.{CPU, CUDA}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{modules, functional as F}
import torch.*
import scala.util.*

class SimpleNetzs[ParamType <: FloatNN: Default]
    extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val linear1 = nn.Linear(10, 5)
  val relu = nn.ReLU()
  val linear2 = nn.Linear(5, 1)

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = linear1(input)
    x = relu(x)
    x = linear2(x)
    x

  }
}
object lesson_09_2 {

//  @main
  def main(): Unit = {
    // 01
    // 输入张量需要梯度
    val x = torch.tensor(Seq(2.0), requires_grad = true)
    // 第一次计算: y = x^3
    val y = x ** 3
    println(f"y = ${y.item()}")
    // 计算一阶导数: dy/dx
    // 使用 create_graph=True 以允许计算高阶梯度
    val grad_y_x = torch.autograd.grad(outputs = y, inputs = x, create_graph = true)(0)
    println(f"x= ${x.item()} 处的 dy/dx: ${grad_y_x.item()}") // 应该是 3 * (2^2) = 12
    // grad_y_x 现在是一个带有自身计算图的张量
    println(f"梯度张量 requires_grad: ${grad_y_x.requires_grad}")
    // 计算二阶导数: d^2y/dx^2 = d/dx (dy/dx)
    // 我们对*一阶梯度* (grad_y_x) 相对于 x 进行微分
    // 除非我们想要三阶梯度，否则这里不需要 create_graph=True
    val grad2_y_x2 = torch.autograd.grad(outputs = grad_y_x, inputs = x)(0)
    println(f"x= ${x.item()} 处的 d^2y/dx^2: ${grad2_y_x2.item()}") // 应该是 6 * 2 = 12
    // 检查二阶导数的 requires_grad 状态
    println(f"二阶导数张量 requires_grad: ${grad2_y_x2.requires_grad}")

    // 02
    val w = torch.tensor(Seq(1.0, math.Pi / 2.0), requires_grad = true) // w1=1，w2=pi/2
    val v = torch.tensor(Seq(0.5, 1.0), dtype = torch.float64) // 一个任意向量

    // 定义函数
    val f = w(0) ** 2 * torch.sin(w(1))

    // 计算一阶梯度: grad_f = nabla f
    val grad_f = torch.autograd.grad(f, w, create_graph = true)(0)
    // 预期 grad_f: [2*1*sin(pi/2), 1^2*cos(pi/2)] = [2, 0]
    println(f"梯度 (nabla f): ${grad_f}")

    // 计算点积: grad_f_dot_v = (nabla f) . v
    // 这个操作需要成为图的一部分，以便进行第二次微分
    val grad_f_dot_v = torch.dot(grad_f, v)
    println(f"点积 (nabla f . v): ${grad_f_dot_v}") // 预期: 2*0.5 + 0*1 = 1.0

    // 计算点积相对于 w 的梯度: nabla (nabla f . v)
    // 这得到 Hessian-向量积 (nabla^2 f) v
    val hvp = torch.autograd.grad(grad_f_dot_v, w)(0)
    // 预期 Hessian: [[2*sin(pi/2), 2*1*cos(pi/2)], [2*1*cos(pi/2), -1^2*sin(pi/2)]]
    // = [[2, 0], [0, -1]]
    // 预期 HVP: [[2, 0], [0, -1]] @ [0.5, 1.0] = [2*0.5 + 0*1, 0*0.5 + (-1)*1] = [1.0, -1.0]
    println(f"Hessian-向量积 (nabla^2 f) v: ${hvp}")

    // 03
    // 示例设置
    val w2 = torch.randn(Seq(5, 3), requires_grad = true)
    val x2 = torch.randn(3, 2)
    val y_true = torch.randn(5, 2)
    // 前向传播
    val y_pred = w2 @@ x2
    val loss = F.mse_loss(y_pred.to(torch.float16), y_true.to(torch.float16))
    // 反向传播
    loss.backward()
    // 查看w中累积的梯度
    println(f"Gradient for w:\n ${w2.grad}")
    // 非叶子张量或requires_grad=False的张量的梯度通常为None
    println(f"Gradient for x:\n ${x2.grad}") // 输出: None (默认requires_grad=False)
    println(f"Gradient for y_pred:\n ${y_pred.grad}") // 输出: None (非叶子张量，默认不保留梯度)

    // 04
//    // 检查None梯度（假设'model'是你的torch.nn.Module实例）
//    for((name, param) <- model.named_parameters() ){
//      if param.grad == None {
//        println(f"Parameter ${name} has no gradient.")
//      }
//    }
//
//    // 检查梯度消失/爆炸
//    var max_grad_norm = 0.0
//    var min_grad_norm = Float.PositiveInfinity
//    var nan_detected = false
//    for( param <-  model.parameters()){
//        if param.grad != None then
//          val grad_norm = param.grad.norm().item()
//          if torch.isnan(param.grad).any():
//            nan_detected = true
//            println(f"NaN gradient detected in parameter: ${name} with shape: ${param.size()}") // 可能需要更具体的识别
//            max_grad_norm = math.max(max_grad_norm, grad_norm)
//            min_grad_norm = math.min(min_grad_norm, grad_norm)
//
//            println(f"Max gradient norm: ${max_grad_norm}")
//            println(f"Min gradient norm: ${min_grad_norm}")
//            if nan_detected then
//              println("Warning: NaN gradients detected!") // 警告：检测到NaN梯度！
//    }

    // 05

    def print_grad_hook[D <: DType](grad: Tensor[D]): Unit = {
      println(f"Gradient received: shape=${grad.shape}, norm=${grad.norm}")
    }

    val voidHook = new VoidTensorHook {
      override def call(grad: TensorBase): Unit = {
        print_grad_hook(fromNative(NativeTensor(grad)))
      }
    }
    val x4 = torch.randn(Seq(3, 3), requires_grad = true)
    val y4 = x4.pow(2).sum()
    // 在张量x上注册hook
//    val hook_handle = x4.register_hook(voidHook)
    val hook_handle = x4.register_hook(print_grad_hook)
    // 计算梯度
    y4.backward()
    // hook函数（print_grad_hook）会自动调用
    // 输出将包含类似以下内容：
    // Gradient received: shape=torch.Size([3, 3]), norm=9.5930
    // 不再需要时应移除hook以避免内存泄漏
//    hook_handle.remove()
    x4.remove_hook(hook_handle)
    // 你也可以在hook中修改梯度，但请谨慎使用：
    def scale_grad_hook[D <: DType](grad: Tensor[D]): Tensor[D] = {
      // 示例：将梯度减半
      println(f"原始梯度: ${grad} 将梯度减半 ")
      val tensor = grad * 0.5
      tensor.to(grad.dtype)
    }
    val scaleGradHook = new TensorTensorHook {
      override def call(grad: TensorBase): TensorBase = {
        scale_grad_hook(fromNative(NativeTensor(grad))).native
      }
    }

    x4.register_hook(scale_grad_hook)
//    x4.register_hook(scaleGradHook)
//    y4.backward() // 现在存储在x.grad中的梯度将减半

    // 06
    val model = SimpleNetzs()
    val input_tensor = torch.randn(Seq(4, 10), requires_grad = true)

    def backward_hook[D <: DType](
        module: nn.Module,
        grad_input: Seq[Tensor[D]],
        grad_output: Seq[Tensor[D]]
    ): Unit = {
      println(f"\nModule: {module.__class__.__name__}")
      grad_input.foreach { g => println(f"  grad_input shape: ${g.shape}") }
      grad_output.foreach { g => println(f"  grad_output shape: ${g.shape}") }
    }
    // 在linear2层上注册hook
//    val hook_handle_bwd = model.linear2.register_full_backward_hook(backward_hook) //todo register_full_backward_hook
    // 前向和反向传播
    val output = model(input_tensor)
    val target = torch.randn(4, 1).to(torch.float32)
    val loss_02 = nn.functional.mse_loss(output, target)
    loss_02.backward()

    println(f"loss_02: ${loss_02}")
    // 输出将显示流经linear2的反向梯度形状
    // Module: Linear
    //   grad_input shapes:  [torch.Size([4, 5]), torch.Size([5]), None] （输入、权重、偏置）如果bias=False，偏置梯度可能为None
    //   grad_output shapes: [torch.Size([4, 1])]

    // 清理hook
//    hook_handle_bwd.remove()

    // 07
    import torch.utils.tensorboard.SummaryWriter

    class SimpleNetk[ParamType <: FloatNN: Default]
        extends TensorModule[ParamType]
        with HasParams[ParamType] {

      val layer1 = nn.Linear(5, 3)
      val relu = nn.ReLU()
      val layer2 = nn.Linear(3, 1)

      override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

      def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
        layer2(relu(layer1(input)))
      }
    }

    val model_02 = SimpleNetk()
    val dummy_input = torch.randn(1, 5) // 提供一个示例输入

    // 创建一个SummaryWriter实例（默认日志保存到./runs/）
//    val writer = new SummaryWriter("runs/graph_demo")

    // 将图添加到TensorBoard
    // writer需要模型和一个示例输入张量
//    writer.add_graph(model_02, dummy_input) //todo SummaryWriter add_graph
//    writer.close()

    // 08
    // 创建一个张量；PyTorch分配存储空间
    val x_08 = torch.randn(2, 3)
    println(f"x_08 storage: ${x_08.storage().data_ptr()}")

    // 切片操作会创建一个共享存储的新张量视图
    val y_08 = x_08(0, ::)
    println(f"y_08 storage: ${y_08.storage().data_ptr()}") // 相同的指针
    println(
      f"Do x_08 and y_08 share storage? ${x_08.storage().data_ptr() == y_08.storage().data_ptr()}"
    )

    // 修改y会影响x，因为它们共享存储
    y_08.fill_(1.0)
    println(s"修改y_08后x_08的值：\n ${x_08}")

    // 09
    // 连续张量
    val a = torch.arange(6).reshape(2, 3)
    println(f"a is contiguous: ${a.is_contiguous()}, Stride: ${a.strides()}") // 步长: (3, 1)
    // 转置会创建非连续视图
    val b = a.t()
    println(f"b is contiguous: ${b.is_contiguous()}, Stride: ${b.strides()}") // 步长: (1, 3)
    // 访问元素仍然正确，但内存访问模式不同
    println("b:\n $b")
    // 某些PyTorch函数需要连续张量
    // 尝试对非连续张量进行view等操作可能会失败
    try b.view(-1)
    catch
      case e: RuntimeException =>
        println(f"\nError viewing non-contiguous tensor: ${e}")
    // 使用 .contiguous() 获取连续副本
    val c = b.contiguous()
    println(f"c is contiguous: ${c.is_contiguous()}, Stride: ${c.strides()}") // 步长: (2, 1)
    println(s"c (contiguous version of b):\n $c")
    println(
      f"Does b and c share storage? ${b.storage().data_ptr() == c.storage().data_ptr()}"
    ) // 否，新的存储空间

    // 10 //todo cuda 内存管理
//    if torch.cuda.is_available() then
//      val device = torch.Device("cuda")
//      println(f"Initial allocated: ${torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB")
//      println(f"Initial reserved:  ${torch.cuda.memory_reserved(device) / 1024**2:.2f} MiB")
//
//      // 分配一些张量
//      val t1 = torch.randn(Seq(1024, 1024), device = device)
//      val t2 = torch.randn(Seq(512, 512), device = device)
//      println(f"\nAfter allocation:")
//      println(f"Allocated: ${torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB")
//      println(f"Reserved:  ${torch.cuda.memory_reserved(device) / 1024**2:.2f} MiB")
//
//      // 删除张量
//      t1.delete()
//      t2.delete()
//      println(f"\nAfter deleting tensors (before empty_cache):")
//      // 已分配内存减少，但由于缓存，保留内存仍然很高
//      println(f"Allocated: ${torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB")
//      println(f"Reserved:  ${torch.cuda.memory_reserved(device) / 1024**2:.2f} MiB")
//
//      // 清除缓存
//      torch.cuda.empty_cache()
//      println(f"\nAfter empty_cache:")
//      // 保留内存也减少（尽管可能由于内部分配而不降为零）
//      println(f"Allocated: ${torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB")
//      println(f"Reserved:  ${torch.cuda.memory_reserved(device) / 1024**2:.2f} MiB")
//    else
//      println("CUDA不可用，跳过GPU内存示例。")

    // 11
    // 设置
    val ax = torch.randn(Seq(100, 100), requires_grad = true)
    val bx = torch.randn(Seq(100, 100), requires_grad = true)

    // 被自动求导跟踪的操作
    val cx = ax * bx
    val dx = cx.sin() // fromNative(cx.native.sin())
    val loss_x = dx.mean()

    // 中间张量'cx'和'dx'被保留在内存中
    // 因为反向传播需要它们。
    // 调用backward会释放缓冲区（除非retain_graph=True）
    loss_x.backward() // 计算a和b的梯度

    // 现在，让我们尝试不跟踪梯度
    torch.no_grad {
      val c_no_grad = ax * bx // 操作已执行，但未被跟踪
      val d_no_grad = c_no_grad.sin() // fromNative(c_no_grad.native.sin()) //
//      c_no_grad.sin
      val loss_no_grad = d_no_grad.mean()
      // PyTorch不需要为未来的反向传播存储'c_no_grad'
      // 中间结果的内存可能更早被释放。
    }

    println(f"ax的梯度：${if ax.grad.isDefined then "存在" else "无"}")
//    loss_no_grad.backward() // 这将引发错误，因为历史记录未被跟踪。
//

  }
}
