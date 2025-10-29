package lesson

import torch.Device.{CPU, CUDA}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.*

object lesson_02_2 extends App {

//  @main
  def main(): Unit = {

    // 48
    val device6 = torch.Device(if torch.cuda.is_available() then "cuda" else "cpu")
    // 创建张量（除非另有说明，否则默认为CPU）
    val cpu_tensor_a = torch.randn(2, 2)
    println(f"\n张量在CPU上: ${cpu_tensor_a.device}\n ${cpu_tensor_a}")

    // 移动到配置的设备（如果可用则为GPU，否则为CPU）
    val device_tensor = cpu_tensor_a.to(device6)
    println(f"\n张量已移动到 ${device_tensor.device}:\n ${device_tensor}")

    // 明确移回CPU
    val cpu_tensor_again = device_tensor.to(CPU)
    println(f"\n张量已移回CPU: ${cpu_tensor_again.device}\n ${cpu_tensor_again}")

    // 执行操作 - 需要张量在同一设备上
    if device_tensor.device != cpu_tensor_a.device then
      println("\n在不同设备上的张量相加会导致错误。")
      // 这会失败: cpu_tensor + device_tensor
      // 正确方法:
      val result_on_device = device_tensor + device_tensor
      println(f"在 ${result_on_device.device} 上的操作结果:\n ${result_on_device}")
    else
      println("\n两个张量都在CPU上，相加没问题。")
      val result_on_cpu = cpu_tensor_a + cpu_tensor_again
      println(f"在 ${result_on_cpu.device} 上的操作结果:\n ${result_on_cpu}")

    // 47
    // 创建一个浮点张量
    val float_tensor_orig = torch.tensor(Seq(1.1, 2.7, 3.5, 4.9))
    println(s"\n原始浮点张量: ${float_tensor_orig}")
    println(s"数据类型: ${float_tensor_orig.dtype}")

    // 转换为 int32
    val int_tensor_cast = float_tensor_orig.to(torch.int32)
    // 替代方法: int_tensor_cast = float_tensor_orig.int()
    println(s"\n转换为整数张量: ${int_tensor_cast}")
    println(s"数据类型: ${int_tensor_cast.dtype}")

    // 46
    // 创建一个整数张量
    val int_tensor2 = torch.tensor(Seq(1, 2, 3, 4))
    println(s"\n整数张量:${int_tensor2}")
    println(s"数据类型:${int_tensor2.dtype}")

    // 转换为 float32
    val float_tensor2 = int_tensor2.to(torch.float32)
    // 替代方法: float_tensor = int_tensor.float()
    println(s"\n转换为浮点张量: ${float_tensor2}")
    println(s"数据类型:${float_tensor2.dtype}")

    // 45
    // 创建一个3x3张量和一个3x1张量（列向量）
    val matrix2 = torch.tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6), Seq(7, 8, 9)))
    val col_vector = torch.tensor(Seq(Seq(100), Seq(200), Seq(300))) // 形状 3x1
    println(s"\n矩阵（3x3）:\n ${matrix2}")
    println(s"列向量（3x1）:\n ${col_vector}")

    // 广播加法: col_vector 被扩展以匹配 matrix 的形状
    val result_col = matrix2 + col_vector
    println(s"\n矩阵 + 列向量（广播）:\n ${result_col}")

    // 44
    // 创建一个3x3张量和一个1x3张量（行向量）
    val matrix = torch.tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6), Seq(7, 8, 9)))
    val row_vector = torch.tensor(Seq(Seq(10, 20, 30))) // 形状 1x3
    println(s"\n矩阵（3x3）:\n ${matrix}")
    println(s"行向量（1x3）:\n ${row_vector}")

    // 广播加法: row_vector 被扩展以匹配 matrix 的形状
    val result = matrix + row_vector
    println(s"\n矩阵 + 行向量（广播）:\n ${result}")

    // 43
    // 创建一个值从0到23的张量
    val tensor_to_split = torch.arange(24).reshape(6, 4)
    println(s"\n待拆分张量（6x4）:\n ${tensor_to_split}")

    // 沿维度0拆分为3个部分
    val chunks = torch.chunk(tensor_to_split, chunks = 3, dim = 0)
    println("\n拆分为3个部分:")
    chunks.zipWithIndex.foreach((chunk, i) => {
      println(f"部分 {i}（形状 ${chunk.shape}）:\n ${chunk}")
    })

    // 40
    // 创建两个2x3张量
    val tensor_a1 = torch.tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6)))
    val tensor_b1 = torch.tensor(Seq(Seq(7, 8, 9), Seq(10, 11, 12)))
    println(s"\n张量A:\n ${tensor_a1}")
    println(s"张量B:\n ${tensor_b1}")

    // 沿维度0连接（堆叠行）
    val concatenated_rows = torch.cat(Seq(tensor_a1, tensor_b1), dim = 0)
    println(s"\n沿行连接（dim=0）:\n ${concatenated_rows}")
    println(s"形状: ${concatenated_rows.shape}") // 应该是 4x3

    // 41
    // 沿维度1连接（连接列）
    val concatenated_cols = torch.cat(Seq(tensor_a1, tensor_b1), dim = 1)
    println(s"\n沿列连接（dim=1）:\n ${concatenated_cols}")
    println(s"形状: ${concatenated_cols.shape}") // 应该是 2x6

    // 42
    // 堆叠张量 - 创建一个新维度（默认 dim=0）
    val stacked_tensor = torch.stack(Seq(tensor_a1, tensor_b1), dim = 0)
    println(s"\n堆叠的张量（dim=0）:\n ${stacked_tensor}")
    println(s"形状: ${stacked_tensor.shape}") // 应该是 2x2x3

    // 沿维度1堆叠
    val stacked_tensor_dim1 = torch.stack(Seq(tensor_a1, tensor_b1), dim = 1)
    println(s"\n堆叠的张量（dim=1）:\n ${stacked_tensor_dim1}")
    println(s"形状: ${stacked_tensor_dim1.shape}") // 应该是 2x2x3

    // 38
    // 创建一个值从0到11的张量
    val tensor_1d = torch.arange(12)
    println(s"\n原始一维张量:\n ${tensor_1d}")

    // 使用 reshape() 改变形状
    val reshaped_tensor = tensor_1d.reshape(3, 4)
    println(s"\n改变形状为3x4:\n ${reshaped_tensor}")

    // 使用 view() 改变形状 - 注意 view 适用于内存连续的张量
    // arange 创建的是连续张量，因此 view 在这里适用。
    val view_tensor = tensor_1d.view(3, 4)
    println(s"\n视为3x4:\n ${view_tensor}")

    // 39
    // 原始3x4张量
    println(s"\n原始3x4张量:\n ${reshaped_tensor}")

    // 交换维度0和1
    val permuted_tensor = reshaped_tensor.permute(1, 0)
    println(s"\n置换为4x3:\n ${permuted_tensor}")
    println(s"原始形状: ${reshaped_tensor.shape}")
    println(s"置换后形状: ${permuted_tensor.shape}")

    // 34
    // 创建一个示例二维张量（3行，4列）
    val data = Seq(Seq(1, 2, 3, 4), Seq(5, 6, 7, 8), Seq(9, 10, 11, 12))
    val tensor_2d = torch.tensor(data)
    println(f"原始张量:\n  ${tensor_2d}")

    // 选择行索引为1、列索引为2的元素
    val element = tensor_2d(1, 2)
    println(f"\n在 [1, 2] 的元素: ${element}")
    println(f"值: ${element.item()}") // 使用 .item() 获取Scala数值

    // 35
    // 选择索引为1的行
    val row_1 = tensor_2d(1)
    println(s"\n第二行（索引1）:\n ${row_1}")

    // 使用切片的替代方法（选择第1行，所有列）
    val row_1_alt = tensor_2d(1, ::)
    println(s"\n第二行（替代方法）:\n ${row_1_alt}")

    // 36
    // 选择所有行，列索引为2
    val col_2 = tensor_2d(::, 2)
    println(s"\n第三列（索引2）:\n ${col_2}")

    // 37
    // 创建布尔掩码
    val mask = tensor_2d > 7
    println(s"\n布尔掩码（张量 > 7）:\n ${mask}")

    // 应用掩码
    val selected_elements = tensor_2d(mask)
    println(s"\n大于7的元素:\n ${selected_elements}")

    // 33
    // 检查CUDA是否可用并设置设备
    val device5 = torch.Device(if torch.cuda.is_available() then "cuda" else "cpu")
    println(f"正在使用设备: ${device5}")

    // 32
    val device4 = torch.Device("cpu") // ("CUDA")
    // 错误示例（假设 device='cuda'）
    val cpu_a = torch.randn(2, 2)
    val gpu_b = torch.randn(2, 2).to(device4) // , device = device)
    try

      // 如果 device 是 'cuda'，这很可能会失败
      val c = cpu_a + gpu_b
    catch
      case e: RuntimeException =>
        println(f"在不同设备上执行操作时出错：${e}")

    // 31
    val device3 = torch.Device("CUDA")
    val cpu_tensor3 = torch.ones(2, 2)

    // 使用便利方法（假设 GPU 可用且 'device' 为 'cuda'）
    if device3.device == CUDA then
      // 将 cpu_tensor 移动到 GPU
      val gpu_tensor_alt = cpu_tensor3.cuda()
      println(f"使用 .cuda()：${gpu_tensor_alt.device}")

      // 将 gpu_tensor_alt 移回 CPU
      val cpu_tensor_alt = gpu_tensor_alt.cpu()
      println(f"使用 .cpu()：${cpu_tensor_alt.device}")

    // 30
    val device2 = torch.Device("cpu")

    // 从 CPU 张量开始
    val cpu_tensor2 = torch.ones(2, 2)
    println(f"原始张量：${cpu_tensor2.device}")

    // 将张量移动到选定设备（如果 GPU 可用，则为 GPU；否则为 CPU）
    // 请记住，'device' 是根据可用性预先设置的
    val moved_tensor = cpu_tensor2.to(device2)
    println(f"移动后的张量：${moved_tensor.device}")

    // 如果张量在 GPU 上，则显式移回 CPU
    if moved_tensor.is_cuda then // 检查张量是否在 CUDA 设备上
      val back_to_cpu = moved_tensor.to(device = CPU)
      println(f"张量移回至：${back_to_cpu.device}")

    // 29
    // 直接在选定设备上创建张量
    val device = torch.Device("cpu")
    try

      // 如果 device='cpu'，此张量将在 CPU 上；如果 device='cuda'，则在 GPU 上
      val device_tensor = torch.randn(3, 4) // , device = "cpu")
      println(f"张量创建于：${device_tensor.device}")
    catch
      case e: RuntimeException =>
        println(f"无法直接在 ${device} 上创建张量：{e}") // 处理未找到 GPU 等情况

    // 28
    // 检查 CUDA 可用性并相应设置设备
    if torch.cuda.is_available() then
      val device = torch.Device("cuda") // 使用第一个可用的 CUDA 设备
      println(f"CUDA (GPU) 可用。使用设备：${device}")
    // 您也可以指定特定 GPU，例如 torch.device("cuda:0")
    else
      val device = torch.Device("cpu")
      println(f"CUDA (GPU) 不可用。使用设备：${device}")

    // device 现在包含 torch.device('cuda') 或 torch.device('cpu')

    // 27
    // 默认在 CPU 上创建张量
    val cpu_tensor = torch.tensor(Seq(1.0, 2.0, 3.0))
    println(f"默认张量设备：${cpu_tensor.device}")

    // 26
    val int_t = torch.tensor(Seq(1, 2), dtype = torch.int32)
    val float_t = torch.tensor(Seq(0.5, 0.5), dtype = torch.float32)
    val double_t = torch.tensor(Seq(0.1, 0.1), dtype = torch.float64)

    // int32 + float32 -> float32
    val result1 = int_t + float_t
    println(f"\nint32 + float32 = ${result1}, dtype: ${result1.dtype}")

    // float32 + float64 -> float64
    val result2 = float_t + double_t
    println(f"float32 + float64 = ${result2}, dtype: ${result2.dtype}")

    // 25
    // 原始整数张量
    val tensor_a = torch.tensor(Seq(0, 1, 0, 1))
    println(f"Original tensor: ${tensor_a}, dtype: ${tensor_a.dtype}")

    // 使用 .float() 转换为浮点数
    val tensor_b = tensor_a.float() // 等同于 .to(torch.float32)
    println(f".float(): ${tensor_b}, dtype: ${tensor_b.dtype}")

    // 使用 .long() 转换为长整型
    val tensor_c = tensor_b.long() // 等同于 .to(torch.int64)
    println(f".long(): ${tensor_c}, dtype: ${tensor_c.dtype}")

    // 使用 .bool() 转换为布尔型
    val tensor_d = tensor_a.bools() // 等同于 .to(torch.bool)
    println(f".bool(): ${tensor_d}, dtype: ${tensor_d.dtype}")

    // 24
    // 原始浮点张量
    val float_tensor = torch.tensor(Seq(1.1, 2.2, 3.3), dtype = torch.float32)
    println(f"Original tensor: ${float_tensor}, dtype: ${float_tensor.dtype}")

    // 使用 .to() 转换为 int64
    val int_tensor = float_tensor.to(torch.int64)
    println(f"Casted to int64: ${int_tensor}, dtype: ${int_tensor.dtype}") // 注意截断
    // 使用 .to() 转换回 float16
    val half_tensor = int_tensor.to(dtype = torch.float16) // 可以只指定 dtype
    println(f"Casted to float16: ${half_tensor}, dtype: ${half_tensor.dtype}")
    // 使用 .to() 转换为 bfloat16
    val bf_tensor = int_tensor.to(dtype = torch.bfloat16) // 可以只指定 dtype
    println(f"Casted to bfloat16: ${bf_tensor}, dtype: ${bf_tensor.dtype}")

    // 23
    // 创建一个64位浮点数张量
    val c6 = torch.tensor(Seq(1.0, 2.0), dtype = torch.float64)
    println(f"\nTensor c: ${c6}")
    println(f"dtype of c: ${c6.dtype}")

    // 创建一个32位整数张量
    val d = torch.ones(2, 2).to(torch.int32) // , dtype = torch.int32)
    println(f"\nTensor d:\n ${d}")
    println(f"dtype of d: ${d.dtype}")

    // 22
    // 默认浮点张量
    val a5 = torch.tensor(Seq(1.0, 2.0, 3.0))
    println(f"Tensor a: ${a5}")
    println(f"dtype of a: ${a5.dtype}")

    // 默认整数张量
    val b5 = torch.tensor(Seq(1, 2, 3))
    println(f"\nTensor b: ${b5}")
    println(f"dtype of b:  ${b5.dtype}")

    // 21
    // 张量 A: 形状 [2, 3]
    val a4 = torch.tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6)))
    // 张量 B: 形状 [2]
    val b4 = torch.tensor(Seq(10, 20))

    try {
      val c4 = a4 + b4
    } catch
      case e: Exception =>
        println(f"Error: {e}")
    // 错误: 张量 a (3) 的大小必须与张量 b (2) 在非单例维度 1 处匹配

    // 20
    // 张量 A: 形状 [2, 3]
    val a3 = torch.tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6)))
    // 张量 B: 形状 [2, 1]
    val b3 = torch.tensor(Seq(Seq(10), Seq(20)))

    // 将列向量添加到矩阵
    val c3 = a3 + b3

    println(f"Shape of a: ${a3.shape}") // torch.Size([2, 3])
    println(f"Shape of b: ${b3.shape}") // torch.Size([2, 1])
    println(f"Shape of c: ${c3.shape}") // torch.Size([2, 3])
    println(f"Result c:\n${c3}")
    // 结果 c:
    // tensor([[11, 12, 13],
    //         [24, 25, 26]])

    // 19
    // 张量 A: 形状 [2, 3]
    val a2 = torch.tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6)))
    // 张量 B: 形状 [3] (为了广播，可以视为 [1, 3])
    val b2 = torch.tensor(Seq(10, 20, 30))

    // 将行向量添加到矩阵
    val c2 = a2 + b2

    println(f"Shape of a: ${a2.shape}") // torch.Size([2, 3])
    println(f"Shape of b: ${b2.shape}") // torch.Size([3])
    println(f"Shape of c: ${c2.shape}") // torch.Size([2, 3])
    println(f"Result c:\n ${c2}")
    // 结果 c:
    // tensor([[11, 22, 33],
    //         [14, 25, 36]])

    // 18
    // 张量 A: 形状 [2, 3]
    val a = torch.tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6)))
    // 标量 B: 形状 [] (0 维度)
    val b = torch.tensor(10)

    // 将标量添加到张量
    val c = a + b

    println(f"Shape of a: ${a.shape}")
    // 张量 a 的形状: torch.Size([2, 3])
    println(f"Shape of b: ${b.shape}")
    // 标量 b 的形状: torch.Size([])
    println(f"Shape of c: ${c.shape}")
    // 张量 c 的形状: torch.Size([2, 3])
    println(f"Result c:\n ${c}")
    // 结果 c:
    // tensor([[11, 12, 13],
    //         [14, 15, 16]])


    val x1x = torch.randn(3, 4)
    val mask2 = x1x.ge(0.5)
    val yy = torch.masked_select(x1x, mask2)

    //x1x: Tensor[torch.Tensor[torch.DType$package.FloatNN | torch.DType$package.ComplexNN]] = tensor dtype=float32, shape=[3, 4], device=CPU
    //[[-1.2909, -1.4690, -1.7415, 2.0152],
    // [2.0195, 0.9681, 0.6905, 1.0762],
    // [1.6381, 1.8311, 0.5637, -0.4784]]
    //mask: Tensor[Bool] = tensor dtype=bool, shape=[3, 4], device=CPU
    //[[false, false, false, true],
    // [true, true, true, true],
    // [true, true, true, false]]
    //yy: Tensor[torch.Tensor[torch.Float64 | torch.BFloat16 | torch.Complex32 | torch.Float16 | torch.Float32 | torch.Complex64 | torch.Complex128]] = tensor dtype=float32, shape=[8], device=CPU
    //[2.0152, 2.0195, 0.9681, ..., 1.6381, 1.8311, 0.5637]

    val aa = torch.tensor(Seq(1, 2, 3, 4))
    val ab = aa.masked_fill(mask = torch.tensor(Seq(1, 1, 0, 0)).to(torch.bool), value = -1)

    val ac = torch.masked_fill(aa, torch.tensor(Seq(1,1,0,0)).to(torch.bool), value = -12)
    //aa: Tensor[Float32] = tensor dtype=float32, shape=[4], device=CPU
    //[1.0000, 2.0000, 3.0000, 4.0000]
    //ab: Tensor[Float32] = tensor dtype=float32, shape=[4], device=CPU
    //[-1.0000, -1.0000, 3.0000, 4.0000]

    val dd = torch.tensor(Seq(Seq(0, 0, 0, 0, 0), Seq(0, 0, 0, 0, 0))).to(torch.int32)
    val maskd = torch.tensor(Seq(Seq(0, 0, 0, 1, 1), Seq(1, 1, 0, 1, 1)))
    val source = torch.tensor(Seq(Seq(0, 1, 2, 3, 4), Seq(5, 6, 7, 8, 9))).to(torch.int32)
    dd.masked_scatter_(maskd.to(torch.bool), source)
    torch.masked_scatter(dd, maskd.to(torch.bool),source)
    //tensor([[0, 0, 0, 0, 1],
    //        [2, 3, 0, 4, 5]])

    val ax = torch.tensor(Seq(1, 2, 3))
    val bx = torch.tensor(Seq(4, 5, 6))
    val resultx = torch.einsum("i,i->", ax, bx)
    //tensor(32)


    val ad = torch.tensor(Seq(Seq(1, 2), Seq(3, 4)))
    val bd = torch.tensor(Seq(Seq(5, 6), Seq(7, 8)))
    val res = torch.einsum("ij,jk->ik", ad, bd)
    //res: Tensor[Float32] = tensor dtype=float32, shape=[2, 2], device=CPU
    //[[19.0000, 22.0000],
    // [43.0000, 50.0000]]

    val as = torch.randn(2, 3, 4)
    val bs = torch.randn(2, 4, 5)
    val results = torch.einsum("bij,bjk->bik", as, bs)
    //dtype=float32, shape=[2, 3, 5],

    val ar = torch.tensor(Seq(Seq(1, 2, 3), Seq(4, 5, 6)))
    val resr = torch.einsum("ij -> ji", ar)
    //ar: Tensor[Float32] = tensor dtype=float32, shape=[2, 3], device=CPU
    //[[1.0000, 2.0000, 3.0000],
    // [4.0000, 5.0000, 6.0000]]
    //resr: Tensor[Float32] = tensor dtype=float32, shape=[3, 2], device=CPU
    //[[1.0000, 4.0000],
    // [2.0000, 5.0000],
    // [3.0000, 6.0000]]


    //n: batch_size，批次大小。
    //q: seq_len_q，查询序列的长度。
    //k: seq_len_k，键序列的长度。
    //h: num_heads，多头注意力中的头数。
    //d: head_dim，每个头的维度。
//    queries: (batch_size, seq_len_q, num_heads, head_dim)
//    keys: (batch_size, seq_len_k, num_heads, head_dim)

   val queries = torch.randn(32,1024,8,64)
   val keys = torch.randn(32, 512, 8, 64)
   val trans = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
//trans: Tensor[torch.Tensor[torch.Float64 | torch.BFloat16 | torch.Complex32 | torch.Float16 | torch.Float32 | torch.Complex64 | torch.Complex128]] = tensor dtype=float32,
    // shape=[32, 8, 1024, 512], device=CPU


    //Q（Query）：形状为 (batch_size, seq_len, d_k)
    //K（Key）：形状为 (batch_size, seq_len, d_k)
   val Q = torch.randn(2, 10, 64) // (batch_size, seq_len, d_k)
   val K = torch.randn(2, 10, 64) // (batch_size, seq_len, d_k)

    // # (batch_size, seq_len, seq_len)  tensor dtype=float32, shape=[2, 10, 10], device=CPU 
   val attention_scores = torch.einsum("bqd,bkd->bqk", Q, K) / torch.sqrt(torch.tensor(64.0))
    //(batch_size, seq_len, seq_len)
   val attention_weights = F.softmax(attention_scores, dim = -1)
    // tensor dtype=float32, shape=[2, 10, 10], device=CPU 

  }
}
