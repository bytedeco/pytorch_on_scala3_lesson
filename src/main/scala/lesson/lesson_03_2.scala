package lesson


import torch.*


object lesson_03_2 {

//  @main
  def main(): Unit = {

    //10
    //     创建需要梯度的输入张量
    val x = torch.tensor(2.0, requires_grad = true)
    val w = torch.tensor(3.0, requires_grad = true)
    val b = torch.tensor(1.0, requires_grad = true)

    // 定义一个简单的计算
    val y = w * x + b // y = 3.0 * 2.0 + 1.0 = 7.0

    // 计算梯度
    y.backward()

    // 访问存储在 .grad 属性中的梯度
    println(f"y 对 x 的梯度 (dy/dx): ${x.grad}")
    println(f"y 对 w 的梯度 (dy/dw): ${w.grad}")
    println(f"y 对 b 的梯度 (dy/db): ${b.grad}")

    // 创建一个不需要梯度的张量
    val z = torch.tensor(4.0, requires_grad = false)
    println(f"张量 z 的梯度 (requires_grad=False): ${z.grad}")

    //11
    // 示例张量
    val x2 = torch.randn(Seq(2, 2), requires_grad = true)
    val w2 = torch.randn(Seq(2, 2), requires_grad = true)
    val b2 = torch.randn(Seq(2, 2), requires_grad = true)
    // 在no_grad上下文之外的操作
    val y2 = x2 * w2 + b2
    println(f"y2.requires_grad: ${y2.requires_grad}") // 输出：y2.requires_grad: True
    println(f"y2.grad_fn: ${y2.grad_fn}") // 输出：y2.grad_fn: <AddBackward0 object at ...>
    // 在no_grad上下文内执行操作
    println("\n进入torch.no_grad()上下文：")
    torch.no_grad{
      val z2 = x2 * w2 + b2
      println(f"  z2.requires_grad: ${z2.requires_grad}") // 输出：z2.requires_grad: False
      println(f"  z2.grad_fn: ${z2.grad_fn}") // 输出：z2.grad_fn: None

      // 即使输入需要梯度，输出也不会
      val k2 = x2 * 5
      println(f"  k2.requires_grad: ${k2.requires_grad}") // 输出：k2.requires_grad: False

    }
    // 在上下文之外，如果输入需要梯度，追踪会恢复
    println("\n退出torch.no_grad()上下文：")
    val p = x2 * w2
    println(f"p.requires_grad: ${p.requires_grad}") // 输出：p.requires_grad: True
    println(f"p.grad_fn: ${p.grad_fn}") // 输出：p.grad_fn: <MulBackward0 object at ...>

    //12
    // 评估循环片段
//    val model = nn.Linear(784, 10)
//    model.eval() // 将模型设置为评估模式（对dropout、batchnorm等层很重要）
//    var total_loss = 0.0
//    var correct_predictions = 0
//
//    torch.no_grad{ // 禁用评估期间的梯度计算
//      for inputs, labels in validation_dataloader:
//      val inputs = inputs.to(device) // 将数据移动到相应的设备
//      val labels = labels.to(device) // 将数据移动到相应的设备
//
//      val outputs = model(inputs) // 前向传播
//      val loss = criterion(outputs, labels) // 计算损失
//
//      total_loss += loss.item()
//      val predicted = torch.max(outputs.data, 1).values
//      correct_predictions += (predicted == labels).sum().item()

    // 计算平均损失和准确率...

    // 13

    // 需要梯度的原始张量
    val a2 = torch.randn(Seq(3, 3), requires_grad = true)
    val b3 = a2 * 2
    println(f"b3.requires_grad: ${b3.requires_grad}") // 输出：b3.requires_grad: True
    println(f"b3.grad_fn: ${b3.grad_fn}") // 输出：b3.grad_fn: <MulBackward0 object at ...>

    // 分离张量 'b2'
    val c2 = b2.detach()
    println(f"\n分离b2以创建c2后：")
    println(f"c2.requires_grad: ${c2.requires_grad}") // 输出：c2.requires_grad: False
    println(f"c2.grad_fn: ${c2.grad_fn}") // 输出：c2.grad_fn: None

    // 重要的是，原始张量 'b2' 未改变
    println(f"\n原始张量 b2 仍保持连接：")
    println(f"b2.requires_grad: ${b2.requires_grad}") // 输出：b2.requires_grad: True
    println(f"b2.grad_fn: ${b2.grad_fn}") // 输出：b2.grad_fn: <MulBackward0 object at ...>

    // 使用分离张量 'c2' 的操作将不会被追踪
    val d2 = c2 + 1
    println(f"\n在分离张量 c2 上的操作：")
    println(f"d2.requires_grad: ${d2.requires_grad}") // 输出：d2.requires_grad: False

    //14
    // 初始张量
    val my_tensor = torch.randn(Seq(5), requires_grad = true)
    println(f"初始requires_grad: ${my_tensor.requires_grad}") // 输出：初始requires_grad: True

    // 原地禁用梯度追踪
    my_tensor.requires_grad_(false) // 注意下划线表示原地操作
    println(f"requires_grad_(false)后: ${my_tensor.requires_grad}") // 输出：requires_grad_(false)后: False


    //15
    // 创建一个需要梯度的张量
    val x3 = torch.tensor(Seq(2.0), requires_grad = true)

    // 执行一些操作
    val y3 = x3 * x3
    val z3 = y3 * 3 // z = 3 * x^2

    // 第一次反向传播
    // dz/dx = 6*x = 6*2 = 12

    z3.backward(retain_graph = true) // retain_graph=true 允许后续的反向传播调用
    println(f"After first backward pass, x.grad: ${x3.grad}")

    //执行另一个操作 （可以相同也可以不同 ）
    // 为简单起见，我们再次使用相同的 z 进行演示
    // 注意：在实际应用中，您可能会基于新的输入或模型的不同部分计算新的损失。
    z3.backward() // 第二次反向传播
    // 我们预期新梯度 (12) 会被加到已有的梯度 (12) 上
    println(f"After second backward pass, x.grad: ${x3.grad}")

    // 手动清零梯度
//    x3.grad.zero_()
    println(f"After zeroing, x.grad: ${x3.grad}")


    //16

    //17
    val x4 = torch.tensor(Seq(2.0, 4.0, 6.0))

    // 权重张量 - 需要计算梯度
    val w4 = torch.tensor(Seq(0.5), requires_grad = true)

    println(f"x: ${x4}")
    println(f"w4: ${w4}")
    println(f"x.requires_grad: ${x4.requires_grad}")
    println(f"w4.requires_grad: ${w4.requires_grad}")


    //18
    // 前向传播：y = w * x
    val y4 = w4 * x4

    // 定义一个简单的标量损失 L（例如，y 的均值）
    val L = y4.mean()

    println(f"y: ${y4}")
    println(f"L: ${L}")
    println(f"y.requires_grad: ${y4.requires_grad}")
    println(f"L.requires_grad: ${L.requires_grad}")

    //19
    // 输入数据
    val a5 = torch.tensor(2.0, requires_grad = true)
    val b5 = torch.tensor(3.0, requires_grad = true)
    val c5 = torch.tensor(4.0, requires_grad = false) // 不需要梯度

    println(f"a: ${a5}, requires_grad= ${a5.requires_grad}")
    println(f"b: ${b5}, requires_grad= ${b5.requires_grad}")
    println(f"c: ${c5}, requires_grad= ${c5.requires_grad}")


    //20
    // 前向传播
    val d5 = a5 * b5
    val e5 = d5 + c5
    val f5 = e5 * 2

    println(f"d: ${d5}, requires_grad= ${d5.requires_grad}") // True（依赖于 a, b）
    println(f"e: ${e5}, requires_grad= ${e5.requires_grad}") // True（依赖于 d）
    println(f"f: ${f5}, requires_grad= ${f5.requires_grad}") // True（依赖于 e）


    //21
    // 从最终的标量输出 f 进行反向传播
    f5.backward()

    // 检查梯度
    println(f"Gradient df/da: ${a5.grad}")
    println(f"Gradient df/db: ${b5.grad}")
    println(f"Gradient df/dc: ${c5.grad}") // 预期结果：None
    if b5.grad.isDefined then {
      println(s"Before zeroing, b5.grad: ${b5.grad}, 梯度即将归零")
      b5.grad.get.zero_()
      println(s"After zeroing, b5.grad: ${b5.grad}, 梯度已归零")
    }

    //25
    // 上下文管理器 torch.no_grad()
    val a7 = torch.tensor(2.0, requires_grad = true)
    println(f"Outside context: a.requires_grad = ${a7.requires_grad}")

    torch.no_grad{
      print(f"Inside context: a.requires_grad = ${a7.requires_grad}")
      //仍然是 True
      val b7 = a7 * 2
      print(f"Inside context: b7 = ${b7}, b7.requires_grad = ${b7.requires_grad}")
      //False ！
    }
    // 在上下文之外，如果输入需要梯度，计算会恢复跟踪
    val c7 = a7 * 3
    println(f"Outside context: c7 = ${c7}, c7.requires_grad = ${c7.requires_grad}") // True


//    val a_7 = torch.tensor(2.0, requires_grad = true)
//    println(s"Outside context: a_7.requires_grad = ${a_7.requires_grad}")
//
//    if a_7.grad.isDefined then {
//      println(s"Before zeroing, a_7.grad: ${a_7.grad}, 梯度即将归零")
//      a_7.grad.get.zero_()
//      println(s"After zeroing, a_7.grad: ${a_7.grad}, 梯度已归零")
//    }

    //26
    // 分离 a，创建一个不需要梯度的新张量 c
    val c8 = a7.detach()
    println(f"a.requires_grad: ${a7.requires_grad}") // True
    println(f"c.requires_grad: ${c8.requires_grad}") // False

    // 分离 a，创建一个不需要梯度的新张量 c
    val d8 = a7.detach()
    println(f"a.requires_grad: ${a7.requires_grad}") // True
    println(f"d.requires_grad: ${d8.requires_grad}") // False

    // 涉及 c 的操作不会跟踪回 a
    val e8 = c8 * 3 // d 不需要梯度
    println(f"e.requires_grad: ${e8.requires_grad}") // False

    // 如果您对涉及 'b' 的计算执行反向传播，
    // 它会流回 'a'。如果您使用 'd'，则不会。
    val L1 = b5.mean() // 依赖于 'a'
    L1.backward()
    println(f"Gradient dL1/da: ${a7.grad}") // 预期结果：2*a = 10.0

    // 在下一次反向调用前清零梯度
    if a7.grad.isDefined  then {
      println(s"Before zeroing, a.grad: ${a7.grad}, 梯度即将归零")
      a7.grad.get.zero_()
      println(s"After zeroing, a.grad: ${a7.grad}, 梯度已归零")
    }

    //尝试通过 'd8' 进行反向传播 -它不会影响 'a' 的梯度
    try

//    #L2 = d.mean()
//    #最终需要一个需要梯度的计算
    // 示例：再次使用 'a' 与分离后的结果
      val L2 = (a7 + d8).mean() // L2 = (a + a.detach()*3).mean()
      L2.backward()
      println(f"Gradient dL2/da: ${a7.grad}") // 只计算来自 'a' 路径的梯度 (1.0)
    catch
      case e: RuntimeException =>
        println(f"Error demonstrating backward with detached: ${e}")
    // 如果最终的标量不依赖于
    // 分离后任何需要梯度的输入，您可能会得到一个错误。
    // 这里，L2 依赖于 'a'，所以梯度是 1.0。
    // 经由 'd' 的路径对 a.grad 没有贡献。

    // 修改 c（分离的张量） - 它会影响 a，因为它们共享数据！
    torch.no_grad{
//      c8(0) = 100.0 // 原位修改 c（对标量使用索引）
    }

    println(f"After modifying c, a = ${a7}") // 'a' 也改变了！
    println(f"After modifying c, c = ${c8}")

  }
}