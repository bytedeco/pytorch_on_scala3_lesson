package lesson

import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{modules, functional as F}
import torch.numpy.TorchNumpy as np
import torch.optim.{Adam, AdamW, SGD}
import torch.optim.lr_scheduler.{CosineAnnealingLR, CosineAnnealingWarmRestarts, LambdaLR}
import torch.{Tensor, *}
import torch.nn
import scala.collection.mutable.{ListBuffer, SortedMap as OrderedDict}
import scala.collection.{mutable, Set as KeySet}
import scala.util.*

class MyResidualBlock[ParamType <: FloatNN: Default](dim: Int, drop_prob: Double = 0)
    extends TensorModule[ParamType]
    with HasParams[ParamType] {

  val norm1 = nn.LayerNorm(dim)
  val linear1 = nn.Linear(dim, dim * 4)
  val activation = nn.GELU()
  val linear2 = nn.Linear(dim * 4, dim)
  // 随机深度层
//  val drop_path =  if drop_prob > 0 then DropPath(drop_prob) else nn.Identity()

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    val shortcut = input
    var x = norm1(input)
    x = linear1(x)
    x = activation(x)
    x = linear2(x)
    // 将 DropPath 应用于残差函数的输出
    x = shortcut + x // drop_path(x)
    x
  }
}

// 在大型模型中的使用示例
// drop_probabilities = torch.linspace(0, 0.1, num_layers) // 线性衰减
// block = MyResidualBlock(dim=embed_dim, drop_prob=drop_probabilities(i).item())
object lesson_11 {

  def mainz(): Unit = {

    // 示例设置
    val model_params = Seq(torch.randn(Seq(10, 5), requires_grad = true))
    val initial_lr = 0.01
    val optimizer = AdamW(model_params, lr = initial_lr)
    // 参数
    val warmup_epochs = 10
    val total_epochs = 100
    val cosine_epochs = total_epochs - warmup_epochs

    // 调度器1：线性预热
    val lr_lambda_warmup: (Int) => Float = (current_epoch: Int) => {
      if current_epoch < warmup_epochs then return (current_epoch + 1) / math.max(1, warmup_epochs)
      else
        // 预热结束后，让余弦调度器间接接管
        // 我们计算相对于预热阶段结束的衰减因子
        val progress = (current_epoch - warmup_epochs) / math.max(1, cosine_epochs)
        val cosine_decay = 0.5 * (1.0 + math.cos(math.Pi * progress))
        return cosine_decay // 这个因子将乘以initial_lr
    }
    val scheduler = LambdaLR(optimizer, lr_lambda = Left(lr_lambda_warmup))
    // 模拟训练循环
    val lrs_warmup_cosine = ListBuffer[Float]()
    for (epoch <- (0 until total_epochs + 20)) { // 模拟稍长一点的时间
      // optimizer.step()
      lrs_warmup_cosine.append(optimizer.param_groups(0).paramGroupDict("lr").asInstanceOf[Float])
      scheduler.step()
    }

  }

  def mains(): Unit = {

    // 假设模型、优化器、数据加载器已定义
    val model = nn.Linear(10, 1) // 示例模型
    val optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    // 假设 data_loader 提供大小为 MICRO_BATCH_SIZE 的微批次
    val MICRO_BATCH_SIZE = 16
    val data_loader = (1 to 10).map { _ =>
      (torch.randn(MICRO_BATCH_SIZE, 10), torch.randn(MICRO_BATCH_SIZE, 1))
    } // 示例数据

    val ACCUMULATION_STEPS = 4 // 梯度累积的步数
    val EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * ACCUMULATION_STEPS

    println(s"微批次大小: $MICRO_BATCH_SIZE")
    println(s"累积步数: $ACCUMULATION_STEPS")
    println(s"有效批次大小: $EFFECTIVE_BATCH_SIZE")

    model.train()
    optimizer.zero_grad()
    // 在循环前将梯度初始化为零

    for (((inputs, targets), index) <- data_loader.zipWithIndex) {
      val outputs = model(inputs.to(torch.float32))
      val loss = nn.functional.mse_loss(outputs, targets.to(torch.float32))

      // --- 累积前归一化损失 ---
      // 将损失按累积步数进行缩放
      val normalized_loss = loss / ACCUMULATION_STEPS
      // --------------------------------------
      normalized_loss.backward() // 累积梯度
      // --- 累积后执行优化器步骤 ---
      if (index + 1) % ACCUMULATION_STEPS == 0 then
        // 可选：在累积*之后*应用梯度裁剪
        // torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

        optimizer.step() // 根据累积的梯度更新权重
        optimizer.zero_grad() // 为下一个累积周期重置梯度
        // 处理数据集大小不能被完美整除时可能存在的剩余梯度
        if (data_loader.length % ACCUMULATION_STEPS != 0) then
          optimizer.step()
          optimizer.zero_grad()
        println(s"步骤 ${index + 1}: 优化器步骤已执行 (有效批次 ${(index + 1) / ACCUMULATION_STEPS})")
        print("对剩余批次执行最终优化器步骤。")
    }

  }

  //  @main
  def main(): Unit = {

    // 假设模型、优化器、数据加载器已定义
    val model = nn.Linear(10, 1) // 示例模型
    val optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    val data_loader = Seq((torch.randn(16, 10), torch.randn(16, 1))) // 示例数据

    val MAX_GRAD_NORM = 1.0 // 定义裁剪阈值

    var total_norm = 0.0
    model.train()
    for ((inputs, targets) <- data_loader) {
      optimizer.zero_grad()

      val outputs = model(inputs.to(torch.float32))
      val loss = nn.functional.mse_loss(outputs, targets.to(torch.float32))

      loss.backward() // 计算梯度

      // --- 梯度裁剪 ---
      // 应该在 .backward() 之后、optimizer.step() 之前调用
      total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = MAX_GRAD_NORM)
      // 可选：记录 total_norm 以监控梯度量值
      // -------------------------
      optimizer.step() // 更新权重
    }
    println(s"训练步骤完成。潜在裁剪前的梯度范数: ${total_norm}")

    // 假设 'model' 是你的 nn.Module 实例
    // AdamW 使用示例
    val optimizer2 = optim.AdamW(
      model.parameters(),
      lr = 1e-4, // 学习率
      betas = (0.9, 0.999), // 移动平均的系数
      eps = 1e-8, // 添加到分母以提高数值稳定性的项
      weight_decay = 1e-2, // 权重衰减系数（正确应用）
      amsgrad = false // 是否使用 AMSGrad 变体
    )
    // 典型的训练循环步骤
    // optimizer.zero_grad()
    // loss.backward()
    // optimizer.step()

    val base_optimizer = optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 1e-2)

    // 使用 Lookahead 封装
//    val optimizer_head = Lookahead(base_optimizer, la_steps = 5, la_alpha = 0.5)

    // 02
    // 示例设置
    val model_params = torch.randn(Seq(10, 5), requires_grad = true)

    val optimizer3 = SGD(Seq(model_params), lr = 0.1)

    // 余弦退火：将学习率从0.1退火到0，持续100个epoch
    val scheduler = CosineAnnealingLR(optimizer, T_max = 100, eta_min = 0)

    // 模拟训练循环以可视化学习率变化
    val lrs = ListBuffer[Float]()
    for (epoch <- (1 to 150)) { // 模拟超过T_max的epoch数
      // optimizer.step() // 通常在loss.backward()之后调用
      lrs.append(optimizer3.param_groups(0).paramGroupDict("lr").asInstanceOf[Float])
      scheduler.step()
    }

    // 示例设置
    val model_params3 = torch.randn(Seq(10, 5), requires_grad = true)

    val optimizer4 = AdamW(Seq(model_params), lr = 0.01) // 初始学习率

    // 带热启动的余弦退火：
    // 每50个回合重启一次 (T_0=50)。
    // 每次重启后周期长度加倍 (T_mult=2)。
    // 最小学习率为1e-5。
    val scheduler2 = CosineAnnealingWarmRestarts(optimizer, T_0 = 50, T_mult = 2, eta_min = 1e-5)

    // 模拟训练循环
    val lrs_restarts = ListBuffer[Float]()
    val num_epochs = 300 // T_0 + T_0*T_mult + T_0*T_mult*T_mult = 50 + 100 + 200 = 350
    for (epoch <- (0 until num_epochs)) { // 模拟超过T_max的epoch数
      // optimizer.step()
      val firstParamGroup = optimizer4.param_groups(0)
      val lr = firstParamGroup.paramGroupDict("lr")
      lrs_restarts.append(optimizer4.param_groups(0).paramGroupDict("lr").asInstanceOf[Float])
      scheduler.step()
    }

  }
}
