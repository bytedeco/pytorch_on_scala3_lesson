package lesson

import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.{ChunkDatasetOptions, Example, ExampleIterator, ExampleStack, ExampleVector, InputArchive, OutputArchive, TensorBase, TensorExampleVectorIterator, TensorTensorHook, VoidTensorHook, Tensor as NativeTensor}
import torch.Device.{CPU, CUDA}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{modules, functional as F}
import torch.*
import java.net.URL
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPInputStream
import scala.collection.mutable.SortedMap as OrderedDict
import scala.collection.{mutable, Set as KeySet}
import scala.math.Pi
import scala.util.*

class ODEFunc[ParamType <: FloatNN: Default](hidden_dim: Int) extends TensorModule[ParamType]  with HasParams[ParamType] {

  val net = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, hidden_dim),
  )

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = ??? //forward(x)

  def forward(t: Float, h: Tensor[ParamType]): Tensor[ParamType] =
    // t：当前时间（标量）
    // h：当前隐藏状态（张量）
    // 返回 dh/dt
    net(h)
}
// 定义网络结构
class SimpleGNNLayer[ParamType <: FloatNN: Default](in_features: Int, out_features: Int) extends TensorModule[ParamType]  with HasParams[ParamType] {

  // 定义可学习的权重矩阵
  val linear = nn.Linear(in_features, out_features, bias = false)
  // 初始化权重（可选但通常是好的做法）
  nn.init.xavier_uniform_(linear.weight)

  override def apply(x: Tensor[ParamType]): Tensor[ParamType] = ??? //forward(x)

  def apply(input: Tensor[ParamType], edge_index: Tensor[Int64]): Tensor[ParamType] = forward(input, edge_index)

  def forward(input: Tensor[ParamType], edge_index: Tensor[Int64]): Tensor[ParamType] = {
    """
     定义每次调用时执行的计算。
     Args:
         x (torch.Tensor): 节点特征张量，形状为 [num_nodes, in_features]。
         edge_index (torch.Tensor): COO格式的图连接信息，形状为 [2, num_edges]。
                                    edge_index[0] = 源节点，edge_index[1] = 目标节点。
     Returns:
         torch.Tensor: 更新后的节点特征张量，形状为 [num_nodes, out_features]。
     """
    val num_nodes = input.size(0)

    // 1. 为edge_index表示的邻接矩阵添加自环
    // 创建节点索引张量 [0, 1, ..., num_nodes-1]
    var self_loops = torch.arange(0, num_nodes, device = input.device).unsqueeze(0)
    val self_loopss = self_loops.repeat(2, 1) // 形状 [2, num_nodes]
    // 将原始边与自环拼接
    val edge_index_with_self_loops = torch.cat(Seq(edge_index, self_loopss.to(torch.int64)), dim = 1)

    // 提取源节点和目标节点索引
    val row = edge_index_with_self_loops(0)
    val col = edge_index_with_self_loops(1)
    // 2. 线性变换节点特征
    val x_transformed = linear(input) // 形状: [num_nodes, out_features]

    // 3. 聚合来自邻居（包括自身）的特征
    // 我们希望对每个目标节点（col）求和源节点（row）的特征
    // 使用零初始化输出张量
    val aggregated_features = torch.zeros(Seq(num_nodes, out_features), device = input.device)

    // 使用 index_add_ 进行高效聚合（散列求和）
    // 将 x_transformed[row] 的元素添加到 aggregated_features 中由 col 指定的索引处
    // index_add_(维度, 索引张量, 要添加的张量)
    aggregated_features.index_add_(0, col, x_transformed(row))

    // 4. 应用最终激活函数（可选）
    // 在此示例中，我们使用ReLU
    val output_features = F.relu(aggregated_features)

    output_features.to(this.paramType)

  }
}
object lesson_10_2 {

//    @main
  def main(): Unit = {

      // 示例用法

      // 定义图数据
      val num_nodes = 4
      val num_features = 8
      val out_layer_features = 16

      // 节点特征（随机）
      val x = torch.randn(num_nodes, num_features)

      // 边索引表示连接（例如，0->1, 0->2, 1->3, 2->3；对于无向图则反之）
      val edge_index = torch.tensor(Seq(
        Seq(0, 0, 1, 2, 1, 2, 3, 3), // 源节点
        Seq(1, 2, 0, 0, 3, 3, 1, 2) // 目标节点
      ), dtype = torch.int64)

      // 实例化层
      val gnn_layer = SimpleGNNLayer(in_features = num_features, out_features = out_layer_features)
      println(s"已实例化层: $gnn_layer")

      // 将数据通过该层
      val output_node_features = gnn_layer(x.to(torch.float32), edge_index)

      // 检查输出形状
      println(s"\n输入节点特征形状: ${x.shape}")
      println(s"边索引形状: ${edge_index.shape}")
      println(s"输出节点特征形状: ${output_node_features.shape}")

      // 验证输出形状是否符合预期: [num_nodes, out_features]
//      assert output_node_features.shape == (num_nodes, out_layer_features)

      print("\n数据已成功通过自定义GNN层。")
      // 显示节点0的前几个输出特征
      println(s"节点0的输出特征（前5维）: ${output_node_features(0, 0 untils 5)}")

  }
}
