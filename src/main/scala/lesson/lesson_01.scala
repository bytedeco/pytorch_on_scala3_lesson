package lesson

import torch.Device.{CPU, CUDA}
import torch.nn.functional as F
import torch.numpy.TorchNumpy as np
import scala.util.*
object lesson_01 {

//  @main
  def mains(): Unit = {
    println(s"PyTorch Version: {torch.__version__}")

    // 检查 CUDA（GPU 支持）是否可用
    val cuda_available = torch.cuda.is_available() // val cuda_available = torch.cuda.is_available()
    println(s"CUDA Available: ${cuda_available}")

    if cuda_available then
      // 获取可用 GPU 的数量
      print(f"Number of GPUs: ${torch.cuda.device_count()}")
      println(s"Current GPU: ${torch.cuda.current_device}")
      // 获取当前 GPU 的名称
//      print(f"Current GPU Name: ${torch.cuda.get_device_name(torch.cuda.current_device())}")
    else print("PyTorch is using CPU.")

    // 创建一个简单的张量
    val x = torch.rand(2, 3)
    println("成功创建了一个张量:")
    println(x)
    val data_list = Seq(Seq(1, 2), Seq(3, 4))
    val tensor_from_list = torch.tensor(data_list)

    println("从列表生成的张量:")
    println(tensor_from_list)

    // 02
    val data_numpy = np.array[Double, Double](
      Array(5.0, 6.0, 7.0, 8.0)
    ) // np.array(Seq(Seq(5.0, 6.0), Seq(7.0, 8.0)))
    data_numpy.printArray()

    val rand_numpy = np.rand(Array(4, 3))
    rand_numpy.printArray()
    val tensor_from_numpy = torch.tensor(rand_numpy)
    println("\n从 NumPy 数组生成的张量:")
    println(tensor_from_numpy)
    println(s"数据类型: ${tensor_from_numpy.dtype}")
    println(s"形状: ${tensor_from_numpy.shape}")

    val tensor_from_numpy_2 = torch.tensor(data_numpy)
    println("\n从 NumPy 数组 numpy_array2 创建的张量：")
    println(tensor_from_numpy_2)
    println(s"数据类型: ${tensor_from_numpy_2.dtype}")
    println(s"形状: ${tensor_from_numpy_2.shape}")

    // 03

    val shape = Seq(2, 3) // 2 行, 3 列

    // 创建具有特定值的张量
    val zeros_tensor = torch.zeros(shape)
    val ones_tensor = torch.ones(shape)
    val empty_tensor = torch.empty(shape*) // 值是任意的

    println(s"\n零张量 (形状 ${shape}):")
    println(zeros_tensor)
    println(s"\n一张量 (形状 ${shape}):")
    println(ones_tensor)

    println(s"\n空张量 (形状 ${shape}):")
    println(empty_tensor)

    // 04
    val ones_int_tensor = torch.ones(shape, dtype = torch.int32)
    println(s"\n一张量 (dtype=torch.int32):")
    println(ones_int_tensor)

    // 05
    // 创建带随机值的张量
    val rand_tensor = torch.rand(shape) // 均匀分布 [0, 1)
    val randn_tensor = torch.randn(shape) // 标准正态分布

    println(s"\n随机张量 (均匀分布 [0, 1), 形状 ${shape}):")
    println(rand_tensor)

    println(s"\n随机张量 (标准正态分布, 形状 ${shape}):")
    println(randn_tensor)

    // 06

    // 使用现有张量作为模板
    val base_tensor =
      torch.tensor(Seq(Seq(1, 2), Seq(3, 4))).to(dtype = torch.float32) // , dtype = torch.float32)
    println(s"\n基础张量 (形状 ${base_tensor.shape}, dtype ${base_tensor.dtype}):")
    println(base_tensor)

    // 创建与基础张量属性匹配的张量
    val zeros_like_base = torch.zeros_like(base_tensor)
    val rand_like_base = torch.rand_like(base_tensor)

    println("\n类似基础张量的零张量:")
    println(zeros_like_base)
    println(s"形状: ${zeros_like_base.shape}, dtype: ${zeros_like_base.dtype}")

    println("\n类似基础张量的随机张量:")
    println(rand_like_base)
    println(s"形状: ${rand_like_base.shape}, dtype: ${rand_like_base.dtype}")

    // 07
    import torch.*

    // 创建两个张量
    val a = torch.tensor(Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)))
    val b = torch.tensor(Seq(Seq(5.0, 6.0), Seq(7.0, 8.0)))

    // 加法
    val sum_tensor = a + b
    println(s"加法 (a + b):\n ${sum_tensor}")
    println(s"加法 (torch.add(a, b)):\n ${torch.add(a, b)}")

    // 减法
    val diff_tensor = a - b
    println(s"\n减法 (a - b):\n ${diff_tensor}")

    // 按元素乘法
    val mul_tensor = a * b
    println(s"\n按元素乘法 (a * b):\n ${mul_tensor}")
    println(s"按元素乘法 (torch.mul(a, b)):\n ${torch.mul(a, b)}")

    // 除法
    val div_tensor = a / b
    println(s"\n除法 (a / b):\n ${div_tensor}")

    // 幂运算
    val pow_tensor = a ** 2
    println(s"\n幂运算 (a ** 2):\n ${pow_tensor}")
    println(s"幂运算 (torch.pow(a, 2)):\n ${torch.pow(a, 2)}")

    // 08
    val a2 = torch.tensor(Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)))
    val b2 = torch.tensor(Seq(Seq(5.0, 6.0), Seq(7.0, 8.0)))

    println(s"原始张量 'a2':\n ${a2}")

    // 执行就地加法
    a2.add_(b2) // a 被直接修改
    println(s"\na2.add_(b2) 后张量 'a2':\n ${a2}")

    // 如果取消注释，这将引发错误，
    // 因为 a + b 的结果是一个新张量，
    // 不适合直接重新赋值给 'a' 的内存
    // a = a + b // 标准加法会创建一个新张量

    // 另一个就地操作
    a2.mul_(2) // 将 'a2' 就地乘以 2
    println(s"\na2.mul_(2) 后张量 'a2':\n ${a2}")

    // 09
    val t = torch.tensor(Seq(Seq(1.0, 2.0, 3.0), Seq(4.0, 5.0, 6.0)))
    val scalar = 10.0

    // 加标量
    println(s"t + 标量:\n ${t + scalar}")

    // 乘以标量
    println(s"\nt * 标量:\n ${t * scalar}")

    // 减去标量
    println(s"\nt - 标量:\n ${t - scalar}")

    // 10
    val t2 = torch.tensor(Seq(Seq(1.0, 4.0), Seq(9.0, 16.0)))

    // 平方根
    println(s"平方根 (torch.sqrt(t2)):\n ${torch.sqrt(t2)}")

    // 指数
    println(s"\n指数 (torch.exp(t2)):\n ${torch.exp(t2)}") // e^x

    // 自然对数
    // 注意：确保对数的值为正
    val t_pos = torch.abs(t) + 1e-6 // 添加小的 epsilon 以提高稳定性（如果存在零）
    println(s"\n自然对数 (torch.log(t_pos)):\n ${torch.log(t_pos)}")

    // 绝对值
    val t_neg = torch.tensor(Seq(Seq(-1.0, 2.0), Seq(-3.0, 4.0)))
    println(s"\n绝对值 (torch.abs(t_neg)):\n ${torch.abs(t_neg)}")

    // 11
    val t3 = torch.tensor(Seq(Seq(1.0, 2.0, 3.0), Seq(4.0, 5.0, 6.0)))
    println(s"原始张量:\n ${t3}")

    // 所有元素的和
    val total_sum = torch.sum(t3)
    println(s"\n所有元素的和 (torch.sum(t3)):\n ${total_sum}")

    // 所有元素的平均值
    // 注意：平均值计算需要浮点张量
    val mean_val = torch.mean(t3.float())
    println(s"所有元素的平均值 (torch.mean(t3.float())):\n ${mean_val}")

    // 最大值
    val max_val = torch.max(t3)
    println(s"张量中的最大值 (torch.max(t3)):\n ${max_val}")

    // 最小值
    val min_val = torch.min(t3)
    println(s"张量中的最小值 (torch.min(t3)):\n ${min_val}")

    // 12
    // 创建张量
    val a3 = torch.tensor(Seq(Seq(1, 2), Seq(3, 4)))
    val b3 = torch.tensor(Seq(Seq(1, 5), Seq(0, 4)))
    println(s"张量 'a':\n ${a3}")
    println(s"张量 'b':\n ${b3}")

    // 相等检查
    println(s"\na3 == b3:\n ${a3 == b3}")

    // 大于检查
    println(s"\na3 > b3:\n ${a3 > b3}")

    // 小于或等于检查
    println(s"\na3 <= b3:\n ${a3 <= b3}")

    // 13
    // 创建布尔张量
    val bool_a = torch.tensor(Seq(Seq(true, false), Seq(true, true)))
    val bool_b = torch.tensor(Seq(Seq(false, true), Seq(true, false)))

    println(s"布尔张量 'bool_a':\n ${bool_a}")
    println(s"布尔张量 'bool_b':\n ${bool_b}")

    // 逻辑与
    println(s"\ntorch.logical_and(bool_a, bool_b):\n ${torch.logical_and(bool_a, bool_b)}")

    // 逻辑或
    println(s"\ntorch.logical_or(bool_a, bool_b):\n ${torch.logical_or(bool_a, bool_b)}")

    // 逻辑非
    println(s"\ntorch.logical_not(bool_a):\n ${torch.logical_not(bool_a)}")

    // 14

    import torch.numpy.enums.DType as npDType
//    val numpy_array = np.array(Seq(Seq(1, 2), Seq(3, 4)))//, dType = npDType.Float32)
    val numpy_array = np.rand(Array(2, 2)) // , dType = npDType.Float32)

    numpy_array.printArray()
    println(s"NumPy 数组:\n ${numpy_array}")
    println(s"NumPy 数组类型: ${numpy_array.getDType}")

    // 将 NumPy 数组转换为 PyTorch 张量
    val pytorch_tensor = torch.from_numpy(numpy_array)
    println(s"\nPyTorch 张量:\n ${pytorch_tensor}")
    println(f"PyTorch 张量类型: ${pytorch_tensor.dtype}")

    // 15

    // 16  //Unsupported dtype for numpy conversion: float64
    val cpu_tensor = torch.tensor(Seq(Seq(10.0, 20.0), Seq(30.0, 40.0)))
    println(f"原始 PyTorch 张量 (CPU):\n ${cpu_tensor}")

    // 将张量转换为 NumPy 数组
    val numpy_array_converted = cpu_tensor.numpy()
    println(f"\n转换后的 NumPy 数组:\n{ $numpy_array_converted}")
    numpy_array_converted.printArray()
    println(s"NumPy 数组类型: ${numpy_array_converted.getDType}")

    // 17
    // 修改张量
//    cpu_tensor.update(0, 1) = 25.0
    cpu_tensor.update(indices = Seq(0, 1), values = 25.0)
    println(f"\n修改后的 PyTorch 张量:\n ${cpu_tensor}")
    println(f"修改张量后的 NumPy 数组:\n ${numpy_array_converted}")

    // 修改 NumPy 数组
//    numpy_array_converted.update(1, 0) = 35.0
//    numpy_array_converted.update(indices = Seq(1, 0), values = 35.0)
    println(f"\n修改后的 NumPy 数组:\n ${numpy_array_converted}")
    println(f"修改 NumPy 数组后的张量:\n ${cpu_tensor}")

    // 18   Unsupported dtype for numpy conversion: float64
    if torch.cuda.is_available() then
      val gpu_tensor = torch.tensor(Seq(Seq(1.0, 2.0), Seq(3.0, 4.0))) // ,device = "cuda")
      println(f"\nGPU 上的张量:\n ${gpu_tensor}")
      // 这将导致错误: numpy_from_gpu = gpu_tensor.numpy()

      // 正确方法: 先移到 CPU
      val cpu_tensor_from_gpu = gpu_tensor.cpu()
      val numpy_from_gpu = cpu_tensor_from_gpu.numpy()
      println(f"\n转换后的 NumPy 数组 (来自 GPU 张量):\n ${numpy_from_gpu}")

    // 注意: numpy_from_gpu 与 cpu_tensor_from_gpu 共享内存，
    // 但不与原始的 gpu_tensor 共享。
    else println("\nCUDA 不可用，跳过 GPU 到 NumPy 的示例。")

    // 19
    // 打印PyTorch版本
    println(f"PyTorch Version: {torch.__version__}")

    // 检查CUDA（GPU支持）是否可用
    if torch.cuda.is_available() then
      println(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
      // 获取PyTorch将使用的默认设备
      val device = torch.Device("cuda")
    else
      println("CUDA not available. Using CPU.")
      val device = torch.Device("cpu")

    println(f"Default device: {device}")

    // 20

    // 从Python列表创建张量
    val data = Seq(Seq(1, 2, 3), Seq(4, 5, 6))
    val tensor_from_list2 = torch.tensor(data)

    println("从列表创建的张量：")
    println(tensor_from_list2)
    println(f"形状: ${tensor_from_list2.shape}")
    println(f"数据类型: ${tensor_from_list2.dtype}") // 通常默认为int64

    // 21
    val tensor_float32 = torch.tensor(data) // , dtype = torch.float32)

    println("\nfloat32数据类型的张量：")
    println(tensor_float32)
    println(f"形状: ${tensor_float32.shape}")
    println(f"数据类型: ${tensor_float32.dtype}") // 通常默认为float32

    // 22
    val zeros_tensor2 = torch.zeros(3, 4)
    println("\n全零张量 (3x4)：")
    println(zeros_tensor2)

    // 创建一个2x2的全一张量，类型为整数
    val ones_tensor_int2 = torch.ones(2, 2) // , dtype = torch.int32)
    println("\n全一张量 (2x2, int32)：")
    println(ones_tensor_int2)

    // 创建一个表示数字范围的一维张量
    val range_tensor = torch.arange(start = 0, end = 5, step = 1) // 类似于Python的range函数
    println("\n范围张量 (0到4)：")
    println(range_tensor)

    // 创建一个包含随机值（0到1均匀分布）的2x3张量
    val rand_tensor2 = torch.rand(2, 3)
    println("\n随机张量 (2x3)：")
    println(rand_tensor2)

    // 23
    val a4 = torch.tensor(Seq(Seq(1, 2), Seq(3, 4))).to(torch.float32) // , dtype = torch.float32)
    val b4 = torch.ones(2, 2) // 默认为float32

    println("张量 'a4'：")
    println(a4)
    println("张量 'b4'：")
    println(b4)

    // 元素级加法
    val sum_tensor2 = a4 + b4
    // 另一种写法：sum_tensor = torch.add(a, b)
    println("\n元素级和 (a4 + b4)：")
    println(sum_tensor2)

    // 元素级乘法
    val prod_tensor = a4 * b4
    // 另一种写法：prod_tensor = torch.mul(a, b)
    println("\n元素级积 (a4 * b4)：")
    println(prod_tensor)

    // 标量乘法
    val scalar_mult = a4 * 3
    println("\n标量乘法 (a4 * 3)：")
    println(scalar_mult)

    // 就地加法（修改张量'a'）
    println(f"\n就地加法前 'a4'：ID {id(a4)}")
    a4.add_(b4) // 注意就地操作的下划线后缀
    println("就地加法后 'a4' (a4.add_(b4))：")
    println(a4)
    println(f"就地加法后 'a4'：ID {id(a4)}") // ID保持不变

    // 矩阵乘法
    // 确保维度兼容矩阵乘法
    // 让我们创建兼容的张量：x (2x3), y (3x2)
    val x2 = torch.rand(2, 3)
    val y = torch.rand(3, 2)
    val matmul_result = torch.matmul(x2, y)
    // 另一种写法：matmul_result = x @ y
    println("\n矩阵乘法 (x @ y)：")
    println(f"张量 x2 的形状: ${x2.shape}, 张量 y 的形状: ${y.shape}")
    println(f"结果形状: ${matmul_result.shape}")
    println(matmul_result)

    // 24
    // 1. NumPy数组到PyTorch张量
    val numpy_array2 = np.array[Double, Double](Array(1.0, 2.0, 3.0, 4.0)) // .reshape(2,2)
    println(s"\nnumpy_array2 NumPy数组：${numpy_array2.getArray.mkString(",")}")
    numpy_array2.printArray()
    println(s"numpy_array2 形状: ${numpy_array2.getShape.mkString(",")}")
    println(f"类型: ${numpy_array2.getDType}")
//
//    // 转换为PyTorch张量
    val tensor_from_numpy2 = torch.from_numpy(numpy_array2)
    println("\n从NumPy数组 numpy_array2 创建的张量：")
    println(tensor_from_numpy2)
    println(f"类型: ${tensor_from_numpy2.dtype}")
//
//    // 重要提示：在CPU上，torch.from_numpy与NumPy数组共享内存
//    // 修改其中一个会影响另一个
//    numpy_array2(0, 0) = 99.0
    println("\n修改后的NumPy数组：")
    println(numpy_array)
    println("修改NumPy数组后的张量（共享内存）：")
    println(tensor_from_numpy2)
//
//    // 2. PyTorch张量到NumPy数组
//    // 让我们使用不同的张量，以避免之前的修改
    val another_tensor =
      torch.tensor(Seq(Seq(5, 6), Seq(7, 8))).to(torch.float64) // , dtype=torch.float64)
    println("\n另一个PyTorch张量：")
    println(another_tensor)

    // 转换为NumPy数组
    val numpy_from_tensor = another_tensor.numpy()
    println("\n从张量创建的NumPy数组：")
    numpy_from_tensor.printArray()
    println(f"类型: ${numpy_from_tensor.getDType}")
//
//    // 同样，在CPU上内存是共享的
//    another_tensor(1, 1) = 100.0
    another_tensor.update(Seq(1, 1), 100.0)
    println("\n修改后的张量：")
    println(another_tensor)
    println("修改张量后的NumPy数组（共享内存）：")
    numpy_from_tensor.printArray()

  }

}
