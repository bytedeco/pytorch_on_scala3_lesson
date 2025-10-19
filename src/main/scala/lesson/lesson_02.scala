package lesson

import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.numpy.TorchNumpy as np
import torch.*

object lesson_02 {

//  @main
  def mains(): Unit = {

    //17
    // 创建一个张量
    val tensor_h = torch.arange(10).reshape(5, 2) // 沿着维度0的大小为5
    println(f"原始张量 (形状: ${tensor_h.shape}):\n ${tensor_h}\n")

    // 沿着维度0分割成3个块
    // 5行 / 3块 -> 大小将是 [2, 2, 1] （前几个块取 ceil(5/3)=2）
    val chunked_tensor = torch.chunk(tensor_h, 3, dim = 0)
    println("分割成3个部分（dim=0）:")
    chunked_tensor.zipWithIndex.foreach{ case (chunk, i) =>
      println(f" 块 ${i} (形状: ${chunk.shape}):\n ${chunk}")
    }

    println("-" * 20)

    // 创建另一个张量
    val tensor_i = torch.arange(12).reshape(3, 4) // 沿着维度1的大小为4
    println(f"\n原始张量 (形状: ${tensor_i.shape}):\n ${tensor_i}\n")

    // 沿着维度1分割成2个块
    // 4列 / 2块 -> 大小将是 [2, 2] （ceil(4/2)=2）
    val chunked_tensor_dim1 = torch.chunk(tensor_i, 2, dim = 1)
    println("分割成2个部分（dim=1）:")
    chunked_tensor_dim1.zipWithIndex.foreach{ case (chunk, i) =>
      println(f" 块 ${i} (形状: ${chunk.shape}):\n ${chunk}")
    }

    //16
    // 创建一个要分割的张量
    val tensor_g = torch.arange(12).reshape(6, 2)
    println(f"原始张量 (形状: ${tensor_g.shape}):\n ${tensor_g}\n")

    // 沿着维度0（行）按大小2分割成块
    // 6行 / 2行/块 = 3块
    val split_equal = torch.split(tensor_g, 2, dim = 0)
    println("分割成大小为2的等份（dim=0）:")
    split_equal.zipWithIndex.foreach{ case (chunk, i) =>
      println(f" 块 ${i} (形状: ${chunk.shape}):\n ${chunk}")
    }

    println("-" * 20)

    // 沿着维度0按大小 [1, 2, 3] 分割成块
    // 总大小必须等于该维度的大小 (1 + 2 + 3 = 6)
    val split_unequal = torch.split(tensor_g, List(1, 2, 3), dim = 0)
    println("\n分割成大小不等的块 [1, 2, 3]（dim=0）:")
    split_unequal.zipWithIndex.foreach{ case (chunk, i) =>
      println(f" 块 ${i} (形状: ${chunk.shape}):\n ${chunk}")
    }

    println("-" * 20)

    // 沿着维度1（列）进行分割
    // 形状: (6, 2)。沿着维度1按大小1分割成块
    val split_dim1 = torch.split(tensor_g, 1, dim = 1)
    println("\n分割成大小为1的等份（dim=1）:")
    split_dim1.zipWithIndex.foreach{ case (chunk, i) =>
      // 使用 squeeze 移除大小为1的维度，以便更清晰地显示
        println(f" 块 ${i} (形状: ${chunk.shape}):\n ${chunk.squeeze()}")
    }


    //15
    // 创建两个形状相同的张量
    val tensor_e = torch.arange(start = 0, end =6).reshape(2, 3)
    val tensor_f = torch.arange(start = 6, end = 12).reshape(2, 3)
    println(f"Tensor E (Shape: ${tensor_e.shape}):\n ${tensor_e}")
    println(f"Tensor F (Shape: ${tensor_f.shape}):\n ${tensor_f}\n")

    // 沿着新维度0进行堆叠
    // 结果形状: (2, 2, 3)
    val stack_dim0 = torch.stack(Seq(tensor_e.to(torch.int32), tensor_f), dim = 0)
    println(f"沿着新维度0堆叠 (形状: ${stack_dim0.shape}):\n ${stack_dim0}\n")

    // 沿着新维度1进行堆叠
    // 结果形状: (2, 2, 3)
    val stack_dim1 = torch.stack(Seq(tensor_e.to(torch.int32), tensor_f), dim = 1)
    println(f"沿着新维度1堆叠 (形状: ${stack_dim1.shape}):\n ${stack_dim1}\n")

    // 沿着新维度2（最后一个维度）进行堆叠
    // 结果形状: (2, 3, 2)
    val stack_dim2 = torch.stack(Seq(tensor_e, tensor_f), dim = 2)
    println(f"沿着新维度2堆叠 (形状: ${stack_dim2.shape}):\n ${stack_dim2}")

    //14
    // 创建两个张量
    val tensor_a = torch.randn(2, 3)
    val tensor_b = torch.randn(2, 3)
    println(f"Tensor A (Shape: ${tensor_a.shape}):\n ${tensor_a}")
    println(f"Tensor B (Shape: ${tensor_b.shape}):\n ${tensor_b}\n")

    // 沿着维度0（行）进行拼接
    // 结果形状: (2+2, 3) = (4, 3)
    val cat_dim0 = torch.cat(Seq(tensor_a, tensor_b), dim = 0)
    println(f"沿着维度0拼接 (形状: ${cat_dim0.shape}):\n ${cat_dim0}\n")

    // 沿着维度1（列）进行拼接
    // 张量必须在其他维度（维度0）上匹配
    // 结果形状: (2, 3+3) = (2, 6)
    val cat_dim1 = torch.cat(Seq(tensor_a, tensor_b), dim = 1)
    println(f"沿着维度1拼接 (形状: ${cat_dim1.shape}):\n ${cat_dim1}")

    // 3D张量示例
    val tensor_c = torch.randn(1, 2, 3)
    val tensor_d = torch.randn(1, 2, 3)
    // 沿着维度0（批次维度）进行拼接
    // 结果形状: (1+1, 2, 3) = (2, 2, 3)
    val cat_3d_dim0 = torch.cat(Seq(tensor_c, tensor_d), dim = 0)
    println(f"\n3D张量沿着维度0拼接 (形状: ${cat_3d_dim0.shape})")



    //12
    // 创建一个三维张量（例如，表示通道、高、宽）
    val image_tensor = torch.randn(3, 32, 32) // 通道，高，宽
    println(f"原始形状: ${image_tensor.shape}") // torch.Size([3, 32, 32])

    // 调整为（高，宽，通道）
    val permuted_tensor = image_tensor.permute(1, 2, 0) // 指定新顺序：维度 1，维度 2，维度 0
    println(f"调整后的形状: ${permuted_tensor.shape}") // torch.Size([32, 32, 3])

    // permute 通常返回一个非连续的视图
    println(f"permuted_tensor 是否连续? ${permuted_tensor.is_contiguous}")

    // 调回原状
    val original_again = permuted_tensor.permute(2, 0, 1) // 回到通道，高，宽
    println(f"调回后的形状: ${original_again.shape}") // torch.Size([3, 32, 32])
    println(f"original_again 是否连续? ${original_again.is_contiguous}") // （可能仍然是非连续的）

    // 检查存储共享
    println(f"与原始张量共享存储吗? ${original_again.storage().data_ptr() == image_tensor.storage().data_ptr()}")

    //13
    // 使调整维度的张量连续
    val contiguous_permuted = permuted_tensor.contiguous()
    println(f"\ncontiguous_permuted 是否连续? ${contiguous_permuted.is_contiguous}")

    // 现在可以安全地使用 view()
    val flattened_permuted = contiguous_permuted.view(-1)
    println(f"展平后的形状: ${flattened_permuted.shape}")

    //10
    // view() 在非连续张量上失败的例子
    val a = torch.arange(12).view(3, 4)
    val b = a.t() // 转置操作会创建一个非连续张量
    println(f"\nb 是否连续? ${b.is_contiguous}")

    try
      val c = b.view(12)
    catch
      case e: RuntimeException =>
        println(f"\n尝试 b.view(12) 时出错: {e}")

    //11
    // 在非连续张量 'b' 上使用 reshape()
    println(f"\n原始非连续张量 b:\n ${b}")
    println(f"b 的形状: ${b.shape}")
    println(f"b 是否连续? ${b.is_contiguous}")

    // 即使 'b' 不连续，reshape 也能工作
    val c = b.reshape(12)
    println(f"\nb.reshape(12) 后的张量 c:\n ${c}")
    println(f"c 的形状: ${c.shape}")
    println(f"c 是否连续? ${c.is_contiguous}")

    // 检查 'c' 是否与 'b' 共享存储。由于 reshape 可能进行了复制，所以它们很可能不共享。
    println(f"与 b 共享存储吗? ${c.storage().data_ptr() == b.storage().data_ptr()}")

    // reshape 也可以用 -1 推断维度
    val d = b.reshape(2, -1) // 推断出最后一个维度为 6
    println(f"\nb.reshape(2, -1) 后的张量 d:\n ${d}")
    println(f"d 的形状: ${d.shape}")
    println(f"d 是否连续? ${d.is_contiguous}")





    //09
    // 创建一个连续张量
    val x2 = torch.arange(12) // tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    println(f"原始张量: ${x2}")
    println(f"原始形状: ${x2.shape}")
    println(f"是否连续? ${x2.is_contiguous}")

    // 使用 view() 重塑
    val y2 = x2.view(3, 4)
    println("\nview(3, 4) 后的张量:")
    println(y2)
    println(f"新形状: ${y2.shape}")
    println(f"与 x2 共享存储吗? ${y2.storage().data_ptr() == x2.storage().data_ptr()}") // 检查它们是否共享内存
    println(f"y2 是否连续? ${y2.is_contiguous}")

    // 尝试另一个视图
    val z = y2.view(2, 6)
    println("\nview(2, 6) 后的张量:")
    println(z)
    println(f"新形状: ${z.shape}")
    println(f"与 x2 共享存储吗? ${z.storage().data_ptr() == x2.storage().data_ptr()}")
    println(f"z 是否连续? ${z.is_contiguous}")


    //08
    val x = torch.arange(start = 10, end = 20) // Tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    println(f"原始一维张量: \n ${x}")

    // 注意索引2的重复
    val indices = torch.tensor(Seq(0, 4, 2, 2),dtype = torch.int32)
    println(f"\n使用索引 ${indices} 选择的元素: \n ${x(indices)}")

    // 对于二维张量
    val y = torch.arange(12).reshape(3, 4)
    // [[ 0,  1,  2,  3],
    //  [ 4,  5,  6,  7],
    //  [ 8,  9, 10, 11]]
    println(f"\n原始二维张量:\n ${y}")

    // 选择特定行
    val row_indices = torch.tensor(Seq(0, 2),dtype = torch.int32)
    val selected_rows = y(row_indices, ---)
    println(f"\n使用索引 ${row_indices} 选择的行:\n ${selected_rows}")

    // 选择特定列
    val col_indices = torch.tensor(Seq(1, 3),dtype = torch.int32)
    val selected_cols = y(---, col_indices) // 从所有行中选择第1列和第3列
    println(f"\n使用索引 ${col_indices} 选择的列:\n ${selected_cols}")

    // 使用索引对选择特定元素
    val row_idx = torch.tensor(Seq(0, 1, 2),dtype = torch.int32)
    val col_idx = torch.tensor(Seq(1, 3, 0),dtype = torch.int32)
    val selected_elements2 = y(row_idx, col_idx) // 选择 (0,1), (1,3), (2,0) -> [1, 7, 8]
    println(f"\n使用 (row_idx, col_idx) 选择的特定元素:\n ${selected_elements2}")


    //01
    val x_1d = torch.tensor(Seq(10, 11, 12, 13, 14))
    println(f"原始一维张量:\n ${x_1d}")

    // 访问第一个元素
    val first_element = x_1d(0)
    println(f"\n第一个元素 (x_1d(0)): ${first_element}, 类型: ${first_element.dtype}")

    // 访问最后一个元素
    val last_element = x_1d(-1)
    println(f"最后一个元素 (x_1d(-1)): ${last_element}, 类型: ${last_element.dtype}")

    // 修改一个元素
//    x_1d(1) = 110
    x_1d.update(Seq(1),110)
    println(f"\n修改后的张量:\n ${x_1d}")


    //02
    val x_2d = torch.tensor(Seq(Seq(1, 2, 3),
      Seq(4, 5, 6),
      Seq(7, 8, 9)))
    println(f"原始二维张量:\n ${x_2d}")

    // 访问第0行第1列的元素
    val element_0_1 = x_2d(0, 1)
    println(f"\n在 [0, 1] 的元素: ${element_0_1}, 类型: ${element_0_1.dtype}")

    // 访问整个第一行 (索引0)
    val first_row = x_2d(0) // or x_2d(0, *)
    println(f"\n第一行 (x_2d(0)): ${first_row}, 类型: ${first_row.dtype}")

    // 访问整个第二列 (索引1)
    val second_col = x_2d(---, 1) // or x_2d(*, 1)
    println(f"第二列 (x_2d(*, 1)): ${second_col}, 类型: ${second_col.dtype}")

    // 修改一个元素
//    x_2d(1, 1) = 55
    x_2d.update(Seq(1, 1),55)
    println(f"\n修改后的二维张量:\n ${x_2d}")


    //03
    val y_1d = torch.arange(10) // Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    println(f"原始一维张量: ${y_1d}")

    // 选择从索引2开始到（不包含）索引5的元素
    val slice1 = y_1d(2.::(5)) //y_1d(2 until 5)
    println(f"\n切片 y_1d(2 until 5): ${slice1}")

    // 选择从开头到索引4的元素
    val slice2 = y_1d(0.::(4)) //(0 until 4)
    println(f"切片 y_1d(0 until 4): ${slice2}")

    // 选择从索引6到末尾的元素
    val slice3 = y_1d(6.::(y_1d.size(0))) //(6 until y_1d.size(0))
    println(f"切片 y_1d(6 until y_1d.size(0)): ${slice3}")

    // 选择每隔一个的元素
    val slice41 = y_1d(0.by(2))
//    val slice4 = y_1d(0.::(y_1d.size(0)).by(2)) //(0 until y_1d.size(0) by 2)
    println(f"切片 y_1d(0 until y_1d.size(0) by 2): ${slice41}")

//    // 选择从索引1到7的元素，步长为2
//    val slice5 = y_1d(1.::(8).by(2)) //(1 until 8 by 2)
//    val slice51 = y_1d(1.::(8).by(2)) //(1 until 8 by 2)
//    println(f"切片 y_1d(1 until 8 by 2): ${slice5}")

    val slice52 = y_1d(1.untils(8)) //(1 until 8 by 2)
    println(f"切片 y_1d(1 until 8 by 2): ${slice52}")
//
//    // 反转张量
//    val slice6 = y_1d((y_1d.size(0) - 1).::(0).by(-1)) //(y_1d.size(0) - 1 until 0 by -1)
//    val slice61 = y_1d((y_1d.size(0) - 1).by(-1)) //todo here have a problem
//    println(f"切片 y_1d(y_1d.size(0) - 1 until 0 by -1): ${slice61}")


  //04
    val x_2d2 = torch.tensor(Seq(Seq(0, 1, 2, 3),
    Seq(4, 5, 6, 7),
    Seq(8, 9, 10, 11)))
    println(f"原始二维张量:\n ${x_2d2}")

    // 选择前两行以及第1和第2列
    val sub_tensor1 = x_2d2(0.::(2), 1.::(3))  //(0 until 2, 1 until 3)
    println(f"\n切片 x_2d2(0 until 2, 1 until 3):\n ${sub_tensor1}")

    // 选择所有行，但只选择最后两列
    val sub_tensor2 = x_2d2 (---, -2.::(x_2d2.size(1)))//(*, -2 until x_2d2.size(1))
    println(f"\n切片 x_2d2(*, -2 until x_2d2.size(1)):\n ${sub_tensor2}")

    // 选择第一行，从第1列到末尾
    val sub_tensor3 = x_2d2 (0, 1.::(x_2d2.size(1)))//(0, 1 until x_2d2.size(1))
    println(f"\n切片 x_2d2(0, 1 until x_2d2.size(1)):\n ${sub_tensor3}")

    // 选择第0行和第2行（使用步长），所有列
//    val sub_tensor4 = x_2d2(0.::(x_2d2.size(0)).by(2), *) //(0 until x_2d2.size(0) by 2, *)
    val sub_tensor41 = x_2d2(0.by(2), ---) //(0 until x_2d2.size(0) by 2, *)
    println(f"\n切片 x_2d2(0 until x_2d2.size(0) by 2, *):\n ${sub_tensor41}")

  //05
    println(f"修改切片前的原始 x_2d:\n ${x_2d2}")

    // 获取一个切片
    val sub_tensor = x_2d2(0.::(2), 1.::(3)) //(0 until 2, 1 until 3)

    //修改切片
//    sub_tensor(0, 0) = 101
    sub_tensor.update(Seq(0, 0),101)
//
    println(f"\n修改后的切片:\n ${sub_tensor}")
    println(f"\n修改切片后的原始 x_2d2:\n ${x_2d2}") // 注意变化！


    //06
    // 创建一个张量
    val data = torch.tensor(Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6)))
    println(f"原始数据张量:\n ${data}")

    // 创建一个布尔遮罩 (例如，选择大于3的元素)
    val mask = data > 3
    println(f"\n布尔遮罩 (data > 3):\n ${mask}")

    // 应用遮罩
    val selected_elements = data(mask)
    println(f"\n通过遮罩选择的元素:\n ${selected_elements}")
    println(f"所选元素的形状: ${selected_elements.shape}")

    // 根据条件修改元素
    val index  = data <= 3
    println(f"\n将小于等于3的元素为零后的数据索引 mask :\n ${data}")
//    data(index) = 0
//    data.update(Seq(index),0)
    data.update(index,0)
    println(f"\n将小于等于3的元素设置为零后的数据 更新后 :\n ${data}")


    //07
    val row_mask = data(---, 0) >2
    println(f"\n行遮罩 (data[:, 0] > 2): ${row_mask}")

    // 使用 ':' 选择所选行中的所有列
    // Or simply: data[row_mask] - PyTorch 通常会推断出完整的行选择
    val selected_rows2 = data(row_mask, ---)
    println(f"\n第一列大于2的行:\n ${selected_rows2}")



  }









}
