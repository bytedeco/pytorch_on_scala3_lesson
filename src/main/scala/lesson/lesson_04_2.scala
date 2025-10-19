package lesson

import torch.{Tensor, *}
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{modules, functional as F}
import scala.collection.mutable.SortedMap as OrderedDict
import scala.collection.{mutable, Set as KeySet}

// 定义网络结构
class SimpleNets[ParamType <: FloatNN: Default](input_size: Int, hidden_size: Int, output_size: Int)
    extends TensorModule[ParamType]
    with HasParams[ParamType] {

  // 初始化父类
  val layer_1 = nn.Linear(input_size, hidden_size)
  val relu = nn.ReLU()
  val layer_2 = nn.Linear(hidden_size, output_size)

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    // 定义前向传播
    var x = layer_1(input)
    x = relu(x)
    x = layer_2(x)
    // 注意：如果后续使用BCEWithLogitsLoss，这里不应用Sigmoid
    x
  }
}

object lesson_04_2 {

//  @main
  def main(): Unit = {

    // 08
    // 定义网络参数
    val input_features = 2
    val hidden_units = 10
    val output_classes = 1 // 二分类logit的单个输出

    // 实例化网络
    val model_08 = SimpleNets(input_features, hidden_units, output_classes)

    // 打印模型结构
    println(model_08)

    // 09
    // --- 数据准备（示例占位符）---
    // 假设我们有一些输入数据(X)和目标标签(y)
    // 对于此示例，我们创建一些虚拟张量
    // 包含5个样本的迷你批次，每个样本有2个特征
    val dummy_input = torch.randn(5, input_features).to(torch.float32)
    // 对应的虚拟标签（0或1）- BCEWithLogitsLoss需要浮点类型
    val dummy_labels = torch.randint(0, 2, Seq(5, 1)).float()

    // --- 实例化模型、损失和优化器 ---
    // 模型已在上面实例化：model = SimpleNet(...)

    // 损失函数：带Logits的二元交叉熵
    // 此损失函数适用于二分类，并期望接收原始logits作为输入
    val criterion = nn.BCEWithLogitsLoss()

    // 优化器：Adam是一个常用的选择
    // 我们将模型的参数传递给优化器
    val learning_rate = 0.01
    val optimizer_09 = optim.Adam(model_08.parameters(true), lr = learning_rate)

    println(s"\n使用的损失函数: $criterion")
    println(s"使用的优化器: $optimizer_09")

    // 10
    // --- 模拟单个训练步骤 ---

    // 1. 前向传播：获取模型预测（logits）
    val outputs = model_08(dummy_input)
    println(s"\n模型输出（logits）形状：${outputs.shape}")
    // println(f"Sample outputs: {outputs.detach().numpy().flatten()}") // 可选：查看输出

    // 2. 计算损失
    val loss = criterion(outputs, dummy_labels)
    println(s"计算的损失：${loss.item()}") // .item()获取标量值 ${loss.item():.4f}"

    // 3. 反向传播：计算梯度
    // 首先，确保梯度已从上一步归零（在实际循环中很重要）
    optimizer_09.zero_grad()
    loss.backward() // 计算损失相对于模型参数的梯度

    // 4. 优化器步骤：更新模型权重
    optimizer_09.step() // 根据计算出的梯度更新参数

    // --- 检查参数（可选）---
    // 您可以在反向传播后（在optimizer.step()之前）检查梯度
    println("\nlayer_1权重的梯度（示例）：")
//    println(model_08.layer_1.weight.grad(0,::)) // 访问特定参数的梯度 //todo 检查梯度是否为0

    // 或者在步骤后检查参数值
    println("\n更新后的layer_1权重（示例）：")
    println(model_08.layer_1.weight(0, ::))

    // 07
    // --- 训练循环外部 ---
    // 示例：多类别分类
    val num_classes = 10
    val model = nn.Linear(784, num_classes) // 简单的线性模型示例
    val optimizer = torch.optim.SGD(model.parameters(true), lr = 0.01)
    val loss_fn = nn.CrossEntropyLoss()

    // 模拟数据加载器（替换为实际的 DataLoader）
    val dummy_dataloader = Seq.fill(5)(torch.randn(64, 784), torch.randint(0, num_classes, Seq(64)))

    // --- 训练循环内部 ---
    model.train() // 将模型设置为训练模式
    dummy_dataloader.zipWithIndex.foreach { (data_target, batch_idx) =>
      {
        val (data, target) = data_target
        println(f"Batch ${batch_idx}, data: ${data}")
        println(f"Batch ${batch_idx}, target: ${target}")
        // 1. 清零梯度
        optimizer.zero_grad()
        // 2. 前向传播：获取预测值（logits）
        val predictions = model(data.to(torch.float32))

        // 3. 计算损失
        val loss = loss_fn(predictions, target)

        // 4. 反向传播：计算梯度
        loss.backward()

        // 5. 优化器步进：更新权重
        optimizer.step()

        if batch_idx % 2 == 0 then // 定期打印损失
          println(f"Batch ${batch_idx}, Loss: ${loss.item()}")
      }
    }

    // 08
    // 假设 'model' 是你的 nn.Module 子类的一个实例
    // 示例：使用随机梯度下降 (SGD)
    val optimizer_01 = optim.SGD(model.parameters(true), lr = 0.01)

    // 示例：使用 Adam 优化器
    val optimizer_02 = optim.Adam(model.parameters(true), lr = 0.001)

    // 带有动量和权重衰减的 SGD
    val optimizer_03 =
      optim.SGD(model.parameters(true), lr = 0.01, momentum = 0.9, weight_decay = 1e-4)

    // 带有默认 betas 和指定学习率的 Adam
    val optimizer_04 = optim.Adam(model.parameters(true), lr = 0.001)

    // 带有自定义 betas 和权重衰减的 Adam
    val optimizer_05 =
      optim.Adam(model.parameters(true), lr = 0.001, betas = (0.9, 0.999), weight_decay = 1e-5)

    // 06
    // 实例化二元交叉熵损失函数
    val loss_fn_bce_logits = nn.BCEWithLogitsLoss()

    // 示例：包含 4 个样本的批次，1 个输出节点（二元分类）
    val predictions_logits_bin = torch.randn(Seq(4, 1), requires_grad = true) // 原始 logits
    // 目标应为浮点数（0.0 或 1.0）
    val targets_bin = torch.tensor(Seq(1.0, 0.0, 0.0, 1.0)).view(4, -1) // 4 个样本的目标

    // 计算损失
    val loss_bce = loss_fn_bce_logits(predictions_logits_bin, targets_bin)
    println(f"BCE With Logits Loss: ${loss_bce.item()}")

    // 05
    // 实例化交叉熵损失函数
    val loss_fn_ce = nn.CrossEntropyLoss()
    // 示例：包含 3 个样本的批次，5 个类别
    // 来自模型的原始分数（logits）
    val predictions_logits = torch.randn(Seq(3, 5), requires_grad = true)
    // 真实类别索引（必须是 LongTensor）
    val targets_classes = torch.tensors(Seq(1, 0, 4), dtype = torch.int64) // 3 个样本的类别索引
    // 计算损失
    val loss_ce = loss_fn_ce(predictions_logits, targets_classes)
    println(f"Cross-Entropy Loss: ${loss_ce.item()}")
    // 现在可以通过 loss_ce.backward() 计算梯度
    loss_ce.backward()
    println(predictions_logits.grad)

    // 04
    val loss_fn_l1 = nn.L1Loss()
    // 示例预测值和目标值（批大小为 3，1 个输出特征）
    val predictions_04 = torch.tensor(Seq(Seq(1.0), Seq(2.5), Seq(0.0)), requires_grad = true)
    val targets_04 = torch.tensor(Seq(Seq(1.2), Seq(2.2), Seq(0.5)))
    // 计算损失
    val loss_l1 = loss_fn_l1(predictions_04, targets_04)
    println(f"L1 Loss: ${loss_l1.item()}") // |1-1.2|, |2.5-2.2|, |0-0.5| 的平均值
    // (0.2 + 0.3 + 0.5) / 3 = 1.0 / 3 = 0.333...

    // 03
    // 实例化均方误差损失函数
    val loss_fn_03 = nn.MSELoss()

    // 示例预测值和目标值（批大小为 3，1 个输出特征）
    val predictions = torch.randn(Seq(3, 1), requires_grad = true)
    val targets = torch.randn(3, 1)

    // 计算损失
    val loss_03 = loss_fn_03(predictions, targets)
    println(f"MSE Loss: ${loss_03.item()}")

    // 现在可以通过 loss.backward() 计算梯度
    loss_03.backward()
    println(predictions.grad)

    // 01
    // 定义输入、隐藏层和输出维度
    val input_size = 784
    val hidden_size = 128
    val output_size = 10

    // 方法1：直接将模块作为参数传递
    val model_v1 = nn.Sequential(
      nn.Linear(input_size, hidden_size), // 第1层：线性变换
      nn.ReLU(), // 激活函数1：非线性
      nn.Linear(hidden_size, output_size) // 第2层：线性变换
    )

    // 打印模型结构
    println("Model V1 (Unnamed Layers):")
    println(model_v1.summarize)

    // 示例用法：创建一个虚拟输入张量
    // 假设批量大小为64
    val dummy_input_02 = torch.randn(64, input_size).to(torch.float32)
    val output_02 = model_v1(dummy_input_02)
    println(s"\nOutput shape:  ${output_02.shape}") // 预期：torch.Size([64, 10])

    // 02
    // 方法2：使用OrderedDict进行命名层
    val layerDict: OrderedDict[String, TensorModule[Float32]] = OrderedDict(
      "fc1" -> nn.Linear(input_size, hidden_size), // 全连接层1
      "relu1" -> nn.ReLU(), // ReLU激活
      "fc2" -> nn.Linear(hidden_size, output_size) // 全连接层2
    )
    val model_v2 = nn.Sequential(
      layerDict
    )

    // 打印模型结构
    println("\nModel V2 (Named Layers):")
    println(model_v2.summarize)

    //     现在可以通过名称访问特定层
//    println(s"\nAccessing fc1 weights shape:${model_v2.fc1.weight().shape}")
    //    // 如果需要，也可以使用整数索引访问
//    println(s"Accessing layer at index 0:${model_v2(0)}")
    //    // 或者如果使用OrderedDict，直接通过字符串名称访问
    //    println(s"Accessing layer by name 'relu1':${model_v2.relu1}")

  }
}
