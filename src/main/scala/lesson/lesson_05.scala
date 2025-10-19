package lesson


import org.bytedeco.pytorch.ExampleVector
import torch.Device.{CPU, CUDA}
import torch.internal.NativeConverters.{fromNative, toNative}
import torch.{&&, ---, ::, BFloat16, DType, Default, Float32, FloatNN, Int64, Slice, Tensor, nn}
import torch.nn.{modules, functional as F}
import torch.nn.modules.{HasParams, TensorModule}
import torch.optim.Adam
import torch.utils.data.{DataLoader, DataLoaderOptions, Dataset, NormalTensorDataset}
import torch.utils.data.dataset.custom.{FashionMNIST, MNIST}
import torch.utils.data.*
import torch.utils.data.dataloader.*
import torch.utils.data.datareader.ChunkDataReader
import torch.utils.data.dataset.*
import torch.utils.data.sampler.RandomSampler

import scala.collection.mutable.SortedMap as OrderedDict
import torch.numpy.TorchNumpy as np
import torch.numpy.matrix.NDArray
import torch.pandas.DataFrame as pd
import torch.optim


class SimpleCustomDataset[Input <: BFloat16 | FloatNN: Default, Target <: BFloat16 | FloatNN: Default](feature: NDArray[Double], label: NDArray[Int])
  extends Dataset[Input, Target] {
  require(feature.getSize == label.getSize, "特征和标签的长度必须相同。")

  override def get_batch(request: Long*): ExampleVector = ???

  override def features: Tensor[Input] = torch.tensors(feature).to(dtype =implicitly[Default[Input]].dtype)

  override def targets: Tensor[Target] = torch.tensors(label).to(dtype =implicitly[Default[Target]].dtype)

  //  """返回样本总数。"""
  override def length: Long = feature.getSize
  // 生成一个数据样本。
  // 参数:
  //      idx (int): 元素的索引。
  // 返回:
  //      tuple: 给定索引对应的 (特征, 标签)。
  //  //获取给定索引的特征和标签
  override def getItem(idx:  Int): (Tensor[Input], Tensor[Target]) = {
    (features(idx),targets(idx))
  }

  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = new Iterator[(Tensor[Input], Tensor[Target])] {
    private var index = 0

    override def hasNext: Boolean = index < this.length

    override def next(): (Tensor[Input], Tensor[Target]) = {
      if (!hasNext) throw new NoSuchElementException("No more elements")
      val item = getItem(index)
      index += 1
      item
    }
  }

}


object lesson_05 {


  @main
  def main(): Unit = {

    //01
    // --- 示例用法 ---
    // 样本数据（请替换为你的实际数据）
    val num_samples = 100
    val num_features = 10
    val features_data = np.rand(Array(num_samples, num_features)) //todo numpy randn随机生成特征数据
    val labels_data = np.randint(0, 5, size = Array(num_samples)) // 示例：5 个类别

    // 创建自定义数据集实例
    val my_dataset = SimpleCustomDataset(features_data, labels_data)

    // 访问数据集属性和元素
    println(s"数据集大小: ${my_dataset.length}")

    // 获取第一个样本
    val first_sample = my_dataset(0)
    val feature_sample = first_sample._1
    val label_sample = first_sample._2
    println(s"\n第一个样本特征:\n$feature_sample")
    println(s"第一个样本形状: ${feature_sample.shape}")
    println(s"第一个样本标签: $label_sample")

    // 获取第十个样本
    val tenth_sample = my_dataset(9)
    val tenth_feature_sample = tenth_sample._1
    val tenth_label_sample = tenth_sample._2
    println(s"\n第十个样本特征:\n$tenth_feature_sample")
    println(s"第十个样本形状: ${tenth_feature_sample.shape}")
    println(s"第十个样本标签: $tenth_label_sample")

  }

}


//class ImageFilelistDataset[Input <: BFloat16 | FloatNN: Default, Target <: BFloat16 | FloatNN: Default](csv_file: String, root_dir: String, transform: Option[TensorModule[Input]] = None)
//  extends Dataset[Input, Target] {
////  require(feature.getSize == label.getSize, "特征和标签的长度必须相同。")
//  """用于从 CSV 文件加载图像路径和标签的数据集。"""
//
//  """
//        参数:
//            csv_file (字符串): 包含标注的 CSV 文件路径。
//                               假设列有：'image_path', 'label'
//            root_dir (字符串): 包含所有图像的目录。
//            transform (可调用, 可选): 可选的数据变换，用于对样本进行处理。
//                                           应用于样本。
//        """
//
//  val annotations = pd.readCsv(csv_file)
////  val root_dir = root_dir
////  val transform = transform // 我们稍后会讨论数据变换
//
//  // 从 CSV 获取相对于 root_dir 的图像路径
//  val img_rel_path = annotations.iloc[idx, 0] // 假设第一列是路径
//  val img_full_path = os.path.join(root_dir, img_rel_path)
//
//    except FileNotFoundError :
//    println(f"错误：未在 {img_full_path} 找到图像")
//    // 适当处理错误，例如返回 None 或抛出异常
//    // 为简单起见，这里我们将返回 None，并依赖 DataLoader 的 collate_fn
//    // 来处理它（或稍后过滤）。一个更好的方法
//    // 可能是事先清理 CSV 文件。
//  // 从 CSV 获取标签
//  val label = annotations.iloc[idx, 1] // 假设第二列是标签
//  val label_tensor = torch.tensor(int(label), dtype = torch.long)
//
//  // 如果有，应用数据变换
//  if transform then
//    image = transform(image) // 数据变换通常会将 PIL 图像转换为张量
//
//  override def features: Tensor[Input] = ???
//
//  override def targets: Tensor[Target] = ???
//
//  override def length: Long = annotations.length
//
//  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = {
//    val img_rel_path = annotations.iloc(idx, 0)
//    val img_full_path = os.path.join(root_dir, img_rel_path)
//
//    var image = Image.open(img_full_path).convert("RGB") //确保有3 个通道
//    val label = annotations.iloc(idx, 1) // 假设第二列是标签
//    val label_tensor = torch.tensor(int(label), dtype = torch.long)
//    image = torch.tensor(np.array(image), dtype = torch.float32).permute(2, 0, 1) / 255.0
//    image = transform(image) // 数据变换通常会将 PIL 图像转换为张量
//    image
//
//  }
//
//  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???
//}