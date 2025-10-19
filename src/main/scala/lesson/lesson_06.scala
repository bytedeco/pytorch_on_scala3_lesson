package lesson

import torch.Device.{CPU, CUDA}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{modules, functional as F}
import torch.utils.data.*
import torch.*


object lesson_06 {


//  @main
  def main(): Unit = {
    // 使用随机梯度下降 (SGD)
    val learning_rate = 0.01
    val num_epochs = 100
    val batch_size = 16

    // 设备配置（如果可用则使用GPU）
    val device = torch.Device(if torch.cuda.is_available() then "cuda" else "cpu")
    println(s"正在使用设备: $device")
    //    val optimizer = optim.SGD(model.parameters(true), lr = learning_rate)
    //
    //    // 或者，使用Adam优化器
    //    optimizer = optim.Adam(model.parameters(true), lr = 0.001)


    //02
    // 生成合成数据: y = 2x + 1 + 噪声
    val true_weight = torch.tensor(Seq(2.0))
    val true_bias = torch.tensor(Seq(1.0))

    // 生成训练数据
    val X_train_tensor = torch.randn(100, 1) * 5 // 100 个样本, 1 个特征
    val y_train_tensor = true_weight * X_train_tensor + true_bias + torch.randn(100, 1) * 0.5 // 添加一些噪声

    // 生成验证数据（独立数据集）
    val X_val_tensor = torch.randn(20, 1) * 5 // 20 个样本, 1 个特征
    val y_val_tensor = true_weight * X_val_tensor + true_bias + torch.randn(20, 1) * 0.5 // 添加一些噪声

    // 创建数据集
    val train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    // 创建数据加载器
    val train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = true)
    val val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = false) // 验证数据无需打乱


    //03
    // 定义模型（一个简单的线性层）
    // 输入特征尺寸 = 1, 输出特征尺寸 = 1
    val model = nn.Linear(1, 1).to(device) // 将模型移动到选定设备

    // 定义损失函数（用于回归的均方误差）
    val loss_fn = nn.MSELoss() // 将损失函数移动到选定设备

    // 定义优化器（随机梯度下降）
    val optimizer = optim.SGD(model.parameters(), lr = learning_rate)

    println("模型定义:")
    println(model.summarize)
    println("\n初始参数:")
    model.named_parameters().foreach((name, param) =>
      if param.requires_grad then
        println(s"$name: ${param.data.squeeze()}")
    )
    println("\n开始训练...")
    for (epoch <- (0 until num_epochs)) {
      model.train() // 将模型设置为训练模式
      var running_loss = 0.0
      var num_batches = 0
      // 遍历DataLoader中的批次数据
      for (((features, labels), index) <- train_loader.zipWithIndex) {
        // 将批次数据移动到与模型相同的设备上
        features.to(device)
        labels.to(device)
        // 1. 前向传播：计算模型的预测
        val outputs = model(features.to(torch.float32))
        // 2. 计算损失
        val loss = loss_fn(outputs, labels)
        // 3. 反向传播：计算梯度
        // 首先，清除上一步的梯度
        optimizer.zero_grad()
        // 然后，执行反向传播
        loss.backward()
        // 4. 优化器步骤：更新模型权重
        optimizer.step()
        // 累加损失以便报告
        running_loss += loss.item()
        num_batches += 1
      }
      // 打印本轮的平均损失
      val avg_epoch_loss = running_loss / num_batches
      if (epoch + 1) % 10 == 0 then // 每10轮打印一次
        println(s"Epoch [${epoch + 1}/${num_epochs}], Training Loss: ${avg_epoch_loss}") //:.4f
    }
    println("训练完成！")


    //04
    println("\n开始评估...")
    model.eval() // 将模型设置为评估模式
    var total_val_loss = 0.0
    var num_val_batches = 0
    // 评估时禁用梯度计算
    torch.no_grad {
      for (((features, labels), index) <- val_loader.zipWithIndex) {
        // 将批次数据移动到设备上
        features.to(device)
        labels.to(device)
        // 前向传播
        val outputs = model(features.to(torch.float32))
        // 计算损失
        val loss = loss_fn(outputs, labels)
        total_val_loss += loss.item()
        num_val_batches += 1
        val avg_val_loss = total_val_loss / num_val_batches
        println(String.format("验证损失: %.4f", avg_val_loss))
        // 检查学习到的参数
        println("\n学习到的参数:")
        for ((name, param) <- model.named_parameters()) {
          if param.requires_grad then
            println(f"$name: ${param.data.squeeze()}")
          println(f"(真实权重: ${true_weight.item()}, 真实偏置: ${true_bias.item()})")
        }
      }
    }


    //04
    // 保存模型学习到的参数
    val model_save_path = "linear_regression_model.pth"
    torch.save(model.state_dict(), model_save_path)
    println(f"\n模型 state_dict 已保存到 ${model_save_path}")

    // 加载模型状态的例子
    // 首先，再次实例化模型架构
    val loaded_model = nn.Linear(1, 1).to(device)

    // 然后，加载保存的状态字典
    //    loaded_model.load_state_dict(torch.load(model_save_path))
    println("模型 state_dict 加载成功。")

    // 请记住，如果用于推断，请将加载的模型设置为评估模式
    loaded_model.eval()

    // 现在您可以使用 loaded_model 进行预测了
    // 使用已加载模型进行预测的例子：
    torch.no_grad {
      val sample_input = torch.tensor(10.0).to(device) // 示例输入
      val prediction = loaded_model(sample_input)
      println(f"输入 10.0 的预测值: ${prediction.item()}")
    }
    // 预期输出应接近 2*10 + 1 = 21


  }


}