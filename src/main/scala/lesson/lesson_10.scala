package lesson

//package lesson
//
//import org.bytedeco.pytorch
//import org.bytedeco.javacpp.{FloatPointer, PointerScope}
//import org.bytedeco.pytorch.{Example, InputArchive, OutputArchive, TensorExampleVectorIterator}
//import org.bytedeco.pytorch.{ChunkDatasetOptions, Example, ExampleIterator, ExampleStack, ExampleVector}
//import org.bytedeco.pytorch.global.torch as torchNative
//
//import java.net.URL
//import java.util.zip.GZIPInputStream
//import java.nio.file.{Files, Path, Paths}
//import scala.collection.{mutable, Set as KeySet}
//import scala.util.{Failure, Random, Success, Try, Using}
//import torch.Device.{CPU, CUDA}
//import torch.internal.NativeConverters.{fromNative, toNative}
//import torch.{&&, ---, ::, BFloat16, DType, Default, Float32, FloatNN, Int64, Slice, Tensor, nn}
//import torch.nn.{modules, functional as F}
//import torch.nn.modules.{HasParams, TensorModule}
//import torch.optim.Adam
//import torch.utils.data.{DataLoader, DataLoaderOptions, Dataset, NormalTensorDataset}
//import torch.utils.data.dataset.custom.{FashionMNIST, MNIST}
//import torch.utils.data.*
//import torch.utils.data.dataloader.*
//import torch.utils.data.datareader.ChunkDataReader
//import torch.utils.data.dataset.*
//import torch.utils.data.sampler.RandomSampler
//
//import scala.collection.mutable.SortedMap as OrderedDict
//import torch.numpy.TorchNumpy as np
//import torch.optim as optim
//class PositionWiseFeedForward[ParamType <: FloatNN: Default](d_model: Int, d_ff: Int, dropout: Float = 0.1) extends TensorModule[ParamType]  with HasParams[ParamType] {
//
//  val linear1 = nn.Linear(d_model, d_ff)
//  val activation =  nn.GELU() //nn.ReLU() # 或
//  val dropout = nn.Dropout(dropout)
//  val linear2 = nn.Linear(d_ff, d_model)
//
//  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)
//
//  def forward(input: Tensor[ParamType]): Tensor[ParamType] ={
//    // x: (批次大小, 序列长度, d_model)
//    var x = linear1(input) // (批次大小, 序列长度, d_ff)
//    x = activation(x)
//    x = dropout(x)
//    x = linear2(x) // (批次大小, 序列长度, d_model)
//    x
//
//  }
//}
//
//
//
//class EncoderLayer[ParamType <: FloatNN: Default](d_model: Int, num_heads: Int, d_ff: Int, dropout: Float) extends TensorModule[ParamType]  with HasParams[ParamType] {
//
//  val self_attn = MultiHeadAttention(d_model, num_heads)
//  val add_norm1 = AddNorm(d_model, dropout)
//  val ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
//  val add_norm2 = AddNorm(d_model, dropout)
//
//  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = ???
//
//  def apply(x: Tensor[ParamType], mask: Tensor[ParamType]): Tensor[ParamType] = forward(x, mask)
//
//  def forward(input: Tensor[ParamType], mask: Tensor[ParamType]): Tensor[ParamType] =
//    // 自注意力子层
//    val attn_output = self_attn(q = input, k = input, v = input, mask = mask)
//    var x = add_norm1(input, attn_output) // 残差连接 + 归一化
//    // 前馈子层
//    val ffn_output = ffn(x)
//    x = add_norm2(x, ffn_output) // 残差连接 + 归一化
//    x
//
//}
//
//
//
//class DecoderLayer[ParamType <: FloatNN: Default](d_model: Int, num_heads: Int, d_ff: Int, dropout: Float) extends TensorModule[ParamType]  with HasParams[ParamType] {
//
//  val masked_self_attn = MultiHeadAttention(d_model, num_heads)
//  val add_norm1 = AddNorm(d_model, dropout)
//  val encoder_decoder_attn = MultiHeadAttention(d_model, num_heads)
//  val add_norm2 = AddNorm(d_model, dropout)
//  val ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
//  val add_norm3 = AddNorm(d_model, dropout)
//
//  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = ???
//
//  def forward(input: Tensor[ParamType], encoder_output: Tensor[ParamType],
//              look_ahead_mask: Tensor[ParamType], padding_mask: Tensor[ParamType]): Tensor[ParamType] =
//    // 1. 带掩码的自注意力子层
//    // Q=x, K=x, V=x; 使用前瞻掩码
//    val self_attn_output = masked_self_attn(q = input, k = input, v = input, mask = look_ahead_mask)
//    var x = add_norm1(input, self_attn_output)
//    // 2. 编码器-解码器注意力子层
//    // Q=x (来自上一层), K=编码器输出, V=编码器输出
//    // 使用与编码器输出相关的填充掩码
//    val enc_dec_attn_output = encoder_decoder_attn(q = x, k = encoder_output, v = encoder_output, mask = padding_mask)
//    x = add_norm2(x, enc_dec_attn_output)
//
//    // 3. 前馈子层
//    val ffn_output = ffn(x)
//    x = add_norm3(x, ffn_output)
//    x
//}
//
//
//class AddNorm[ParamType <: FloatNN: Default](normalized_shape: Int, dropout: Float = 0.1) extends TensorModule[ParamType]  with HasParams[ParamType] {
//  val layer_norm = nn.LayerNorm(normalized_shape)
//  val dropoutLayer = nn.Dropout(dropout)
//
//  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = ???
//
//  def apply(input: Tensor[ParamType], sublayer_output: Tensor[ParamType]): Tensor[ParamType] = forward(input, sublayer_output)
//
//  def forward(input: Tensor[ParamType], sublayer_output: Tensor[ParamType]): Tensor[ParamType] = {
//
//    // 应用残差连接和Dropout，然后是层归一化
//    layer_norm(input + dropoutLayer(sublayer_output))
//  }
//}
//
//def scaled_dot_product_attention[ParamType <: FloatNN: Default](q: Tensor[ParamType], k: Tensor[ParamType], v: Tensor[ParamType], mask: Option[Tensor[ParamType]] = None):(Tensor[ParamType], Tensor[ParamType]) = {
//  """计算缩放点积注意力"""
//  val d_k = q.size(-1) // 获取最后一个维度（K的嵌入维度）
//  // Q与K转置的矩阵乘法: (..., 查询序列长度, d_k) x (..., 键序列长度, d_k) -> (..., 查询序列长度, 键序列长度)
//  var scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) // (..., 查询序列长度, 键序列长度)
//
//  // 应用掩码（如果提供），将掩码位置设置为一个非常小的数字 (-1e9)
//  if mask.isDefined then
//    val mask_scores = scores.masked_fill(mask.get == 0, -1e9) //.to(torch.float64)
//    scores = mask_scores
//
//  // 应用softmax以获取注意力权重
//  val attn_weights = torch.softmax(scores, dim = - 1) // (..., 查询序列长度, 键序列长度)
//
//  // 权重与V的矩阵乘法: (..., 查询序列长度, 键序列长度) x (..., 值序列长度, d_v) -> (..., 查询序列长度, d_v)
//  // 注意: 键序列长度 == 值序列长度
//  val output = torch.matmul(attn_weights, v) // (..., 查询序列长度, d_v)
//  (output, attn_weights)
//}
//
//class MultiHeadAttention[ParamType <: FloatNN: Default](d_model: Int,num_heads: Int, dropout: Float = 0.1, max_len: Int = 5000) extends TensorModule[ParamType]  with HasParams[ParamType] {
//
//  //  assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
//  //  val d_model = d_model
//  //  val num_heads = num_heads
//  val d_k = d_model // num_heads // 每个头的键/查询维度
//  val d_v = d_model // num_heads // 每个头的值维度
//
//  // Q、K、V投影的线性层（应用于所有头）
//  val W_q = nn.Linear(d_model, d_model)
//  val W_k = nn.Linear(d_model, d_model)
//  val W_v = nn.Linear(d_model, d_model)
//
//  // 连接后的最终线性层
//  val W_o = nn.Linear(d_model, d_model)
//
//  def split_heads(input: Tensor[ParamType]): Tensor[ParamType] = {
//    // 输入 x: (批次大小, 序列长度, d_model)
//    val batch_size = input.size(0)
//    val seq_len = input.size(1)
//    // 重塑为 (批次大小, 序列长度, 头数, d_k)
//    var x = input.view(batch_size, seq_len, num_heads, d_k)
//    // 转置为 (批次大小, 头数, 序列长度, d_k) 以进行注意力计算
//    return x.transpose(1, 2)
//  }
//
//  def combine_heads(input: Tensor[ParamType]): Tensor[ParamType] = {
//    // 输入 x: (批次大小, 头数, 序列长度, d_k)
//    val batch_size = input.size(0)
//    val seq_len = input.size(2)
//    // 转置回 (批次大小, 序列长度, 头数, d_k)
//    var x = input.transpose(1, 2).contiguous() // 确保转置后内存连续
//    // 重塑为 (批次大小, 序列长度, d_model)
//    return x.view(batch_size, seq_len, d_model)
//  }
//
//  def forward(q: Tensor[ParamType], k: Tensor[ParamType], v: Tensor[ParamType], mask: Option[Tensor[ParamType]] = None): Tensor[ParamType] ={
//    // q, k, v: (批次大小, 序列长度, d_model)
//    // 掩码: (批次大小, 1, 查询序列长度, 键序列长度) 或类似的可广播形状
//
//    // 1. 应用线性投影
//    var new_q = W_q(q) // (批次大小, 查询序列长度, d_model)
//    var new_k = W_k(k) // (批次大小, 键序列长度, d_model)
//    var new_v = W_v(v) // (批次大小, 值序列长度, d_model) // 注意: 键序列长度 == 值序列长度
//
//    // 2. 分割成多个头
//    new_q = split_heads(new_q) // (批次大小, 头数, 查询序列长度, d_k)
//    new_k = split_heads(new_k) // (批次大小, 头数, 键序列长度, d_k)
//    new_v = split_heads(new_v) // (批次大小, 头数, 值序列长度, d_k)
//
//    // 3. 应用缩放点积注意力
//    // 输出: (批次大小, 头数, 查询序列长度, d_k)
//    // 注意力权重: (批次大小, 头数, 查询序列长度, 键序列长度)
//    val (attention_output, attn_weights) = scaled_dot_product_attention(new_q, new_k, new_v, mask)
//    // 4. 合并头
//    var output = combine_heads(attention_output) // (批次大小, 查询序列长度, d_model)
//    // 5. 最终线性层
//    output = W_o(output) // (批次大小, 查询序列长度, d_model)
//
//    output //通常我们只需要输出，不需要权重，用于下一层
//  }
//}
//class PositionalEncoding[ParamType <: FloatNN: Default](d_model: Int, dropout: Float = 0.1, max_len: Int = 5000) extends TensorModule[ParamType]  with HasParams[ParamType] {
//
//  val dropoutLayer = nn.Dropout(p = dropout)
//
//  // 创建位置索引 (0 到 max_len - 1)
//  val position = torch.arange(max_len).unsqueeze(1) // 形状: (max_len, 1)
//
//  // 计算正弦和余弦参数的除数项
//  val div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
//  // 形状: (d_model / 2)
//
//  // 初始化位置编码矩阵
//  var pe = torch.zeros(max_len, d_model) // 形状: (max_len, d_model)
//
//  val sinPos = torch.sin(position * div_term)
//  val cosPos = torch.cos(position * div_term)
//  // 对偶数索引应用sin，对奇数索引应用cos
////  pe(::, 0::2) = sinPos
//  pe(---, 1::2) = cosPos
//
//  // 添加批次维度并注册为缓冲区（非模型参数）
//  pe = pe.unsqueeze(0) // 形状: (1, max_len, d_model)
//  register_buffer("pe", pe)
//
//  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)
//
//  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
//    """
//        参数:
//            x: 张量, 形状 [批次大小, 序列长度, d_model]
//        返回:
//            张量, 形状 [批次大小, 序列长度, d_model]
//        """
//    // 将位置编码添加到输入嵌入
//    // self.pe 形状为 (1, max_len, d_model)。我们取 x 的序列长度范围内的切片。
//    // x 的形状为 (批次大小, 序列长度, d_model)
//    var x = input + pe(---, 0.::(input.size(1)), ::)
//    dropoutLayer(x)
//  }
//}
//
//class Transformer[ParamType <: FloatNN: Default](num_encoder_layers: Int, num_decoder_layers: Int,
//                                                 d_model: Int, num_heads: Int, d_ff: Int,
//                                                 input_vocab_size: Int, target_vocab_size: Int,
//                                                 max_seq_len: Int, dropout: Float = 0.1) extends TensorModule[ParamType]  with HasParams[ParamType] {
//
//
//  val encoder_embedding = nn.Embedding(input_vocab_size, d_model)
//  val decoder_embedding = nn.Embedding(target_vocab_size, d_model)
//  val positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
//
//  val encoder_layers = nn.ModuleList(
//    for _ <- 0 until num_encoder_layers
//      yield EncoderLayer(d_model, num_heads, d_ff, dropout)
//  )
//  val decoder_layers = nn.ModuleList(
//    for _ <- 0 until num_decoder_layers
//      yield DecoderLayer(d_model, num_heads, d_ff, dropout)
//  )
//
//  val final_linear = nn.Linear(d_model, target_vocab_size)
//  val d_model = d_model
//  val dropout = nn.Dropout(dropout)
//
//  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = ???
//
//  def create_padding_mask(seq: Tensor[ParamType], pad_token_idx: Int = 0): Tensor[ParamType] =
//    // 序列形状: (批次大小, 序列长度)
//    // 输出掩码形状: (批次大小, 1, 1, 序列长度)
//    val mask = (seq != pad_token_idx).unsqueeze(1).unsqueeze(2)
//    mask
//
//  def create_look_ahead_mask(size: int): Tensor[ParamType] =
//    // 创建一个上三角矩阵用于掩盖未来词元
//    // 输出掩码形状: (1, 1, 大小, 大小)
//    val mask = torch.triu(torch.ones(size, size), diagonal = 1).bool()
//    // 我们希望在掩盖处为0，所以我们进行反转（如果注意力中使用0进行掩盖）
//    // 或者如果注意力函数期望在掩盖处为True，则按原样返回
//    // 假设 scaled_dot_product_attention 使用 masked_fill(mask == 0, -1e9) 或 masked_fill(mask == True, -1e9)，请相应调整。
//    // 让我们假设后者 (True表示掩码)
//    mask.unsqueeze(0).unsqueeze(0) //在掩盖处设置为False
//
//  def encode(src: Tensor[ParamType], src_mask: Tensor[ParamType]): Tensor[ParamType] =
//    //源: (批次大小, 源序列长度)
//    // 源掩码: (批次大小, 1, 1, 源序列长度)
//    val src_emb = encoder_embedding(src) * math.sqrt(d_model)
//    val src_pos_emb = positional_encoding(src_emb)
//    val enc_output = dropout(src_pos_emb)
//
//    for(layer <- encoder_layers)
//          enc_output = layer(enc_output, src_mask)
//    enc_output //(批次大小, 源序列长度, d_model)
//
//  def decode(tgt: Tensor[ParamType], encoder_output: Tensor[ParamType],
//             look_ahead_mask: Tensor[ParamType], padding_mask: Tensor[ParamType]): Tensor[ParamType] =
//    // 目标: (批次大小, 目标序列长度)
//    // 编码器输出: (批次大小, 源序列长度, d_model)
//    // 前瞻掩码: (批次大小, 1, 目标序列长度, 目标序列长度)
//    // 填充掩码: (批次大小, 1, 1, 源序列长度) # 在编码器-解码器注意力中使用
//    val tgt_emb = decoder_embedding(tgt) * math.sqrt(d_model)
//    val tgt_pos_emb = positional_encoding(tgt_emb)
//    val dec_output = dropout(tgt_pos_emb)
//
//    tgt_emb = decoder_embedding(tgt) * math.sqrt(d_model)
//    tgt_pos_emb = positional_encoding(tgt_emb)
//    dec_output = dropout(dec_output)
//
//    for(layer <- decoder_layers)
//      dec_output = layer(dec_output, encoder_output, look_ahead_mask, padding_mask)
//    dec_output // (批次大小, 目标序列长度, d_model)
//
//  def forward(src: Tensor[ParamType], tgt: Tensor[ParamType]): Tensor[ParamType] =
//    // 源: (批次大小, 源序列长度)
//    // 目标: (批次大小, 目标序列长度) 通常为训练目的而右移
//    // 输出: (批次大小, 目标序列长度, 目标词汇表大小)
//
//    val src_padding_mask = create_padding_mask(src)
//    val tgt_padding_mask = create_padding_mask(tgt) // 如果目标也有填充，也需要
//    val look_ahead_mask = create_look_ahead_mask(tgt.size(1)).to(tgt.device)
//
//    // 将前瞻掩码和目标填充掩码结合用于解码器自注意力
//    // 确保两个掩码都可广播: (批次大小, 1, 目标序列长度, 目标序列长度)
//    val combined_look_ahead_mask = torch.logical_and(tgt_padding_mask.transpose(-2, -1), look_ahead_mask)
//    val encoder_output = encode(src, src_padding_mask)
//    val decoder_output = decode(tgt, encoder_output, combined_look_ahead_mask, src_padding_mask)
//    // 最终线性投影
//    val output = final_linear(decoder_output) // (批次大小, 目标序列长度, 目标词汇表大小)
//    output // 通常在推理/损失计算期间，模型外部接着Softmax
//}
//
//
//object lesson_10 {
//
//  @main
//  def main(): Unit = {
//
//
//    //01
//    // 示例参数
//    val vocab_size = 10000 // 词汇表大小
//    val d_model = 512 // 嵌入维度
//
//    val embedding = nn.Embedding(vocab_size, d_model)
//
//    // 示例用法：2个序列的批次，长度为10
//    val input_tokens = torch.randint(0, vocab_size, Seq(2, 10)) // (批次大小, 序列长度)
//    val input_embeddings = embedding(input_tokens) // (批次大小, 序列长度, d_model)
//
//    println(s"输入形状: ${input_tokens.shape}")
//    println(s"嵌入形状: ${input_embeddings.shape}")
//
//    // 示例用法
//    val pos_encoder = PositionalEncoding(d_model, dropout = 0.1)
//    val final_input = pos_encoder(input_embeddings * math.sqrt(d_model)) // 在添加位置编码前对嵌入进行缩放
//
//    println("位置编码后的形状:", final_input.shape)
//    // 注意：原始论文在添加位置编码前，会按 sqrt(d_model) 缩放嵌入。
//
//    //03
//    // 示例用法
//    // 创建一个多头注意力实例
//    val mha = MultiHeadAttention(d_model = 512, num_heads = 8)
//    // 在自注意力中，Q、K和V最初通常是同一个张量
//    val query = key = value = final_input // 形状: (批次大小, 序列长度, d_model)
//    val attention_result = mha(query, key, value, mask = None) // 掩码对填充/解码很重要
//
//    println(s"多头注意力输出形状: ${attention_result.shape}")
//
//
//    //04
//    // 示例：在多头注意力后应用加和归一化
//    val dropout_rate = 0.1
//    val add_norm1 = AddNorm(d_model, dropout_rate)
//    // 'final_input' 是MHA层的输入
//    val normed_attention_output = add_norm1(final_input, attention_result)
//
//    println(s"加和归一化输出形状: ${normed_attention_output.shape}")
//
//    //05
//    // 示例用法
//    val d_ff = d_model * 4 // 常见做法
//    val ffn = PositionWiseFeedForward(d_model, d_ff, dropout_rate)
//    val ffn_output = ffn(normed_attention_output)
//
//    // 应用第二个加和归一化层
//    val add_norm2 = AddNorm(d_model, dropout_rate)
//    // 'normed_attention_output' 是FFN的输入
//    val encoder_layer_output = add_norm2(normed_attention_output, ffn_output)
//
//    println(s"FFN输出形状: ${ffn_output.shape}")
//    println(s"编码器层输出形状: ${encoder_layer_output.shape}"）
//
//
//    //06
//    // 示例实例化（参数仅作说明）
//    val transformer_model = Transformer(
//      num_encoder_layers = 6, num_decoder_layers = 6,
//      d_model = 512, num_heads = 8, d_ff = 2048,
//      input_vocab_size = 10000, target_vocab_size = 12000,
//      max_seq_len = 500, dropout = 0.1
//    )
//
//    // 用于形状检查的虚拟输入（假设批次大小为2）
//    val src_dummy = torch.randint(1, 10000, Seq(2, 100)) // (批次, 源长度)
//    val tgt_dummy = torch.randint(1, 12000, Seq(2, 120)) // (批次, 目标长度) - 例如右移的目标
//
//    // 如果GPU可用，将模型和数据移至GPU
//    // device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
//    // transformer_model.to(device)
//    // src_dummy = src_dummy.to(device)
//    // tgt_dummy = tgt_dummy.to(device)
//
//    // 前向传播
//    val output_logits = transformer_model(src_dummy, tgt_dummy)
//    println(s"最终输出形状 (logits):${output_logits.shape}") // 应为 (2, 120, 12000)
//
//
//    //07
//    // 在标准 PyTorch 函数式 API 中使用注意力遮罩的示例
//    // 假设 embed_dim = 64, num_heads = 8, seq_len = 5, batch_size = 2
//    val embed_dim = 64
//    val num_heads = 8
//    val seq_len = 5
//    val batch_size = 2
//
//    // 虚拟输入张量 (Batch, SeqLen, EmbedDim)
//    val query = torch.randn(batch_size, seq_len, embed_dim)
//    val key = torch.randn(batch_size, seq_len, embed_dim)
//    val value = torch.randn(batch_size, seq_len, embed_dim)
//
//    // 如果函数需要，为多头注意力重塑形状
//    // 或在 nn.Module 包装器内处理
//
//    // 创建因果遮罩（例如，用于解码器）
//    // 遮罩需要根据注意力函数设置适当的维度
//    // 对于 scaled_dot_product_attention，(SeqLen, SeqLen) 遮罩通常是可广播的
//    val causal_mask_bool = torch.triu(torch.ones(seq_len, seq_len, dtype = torch.bool), diagonal = 1)
//
//    // 使用 torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+)
//    // 注意：此函数在内部处理重塑和缩放
//    // 它期望布尔遮罩，其中 True 表示“遮盖掉”
//    try
//      output_sdpa = nn.functional.scaled_dot_product_attention(
//      query, value, attn_mask = causal_mask_bool, is_causal = false)
//    //显式遮罩示例
//    // 或者使用 is_causal=True 进行自动因果遮罩：
//    // output_sdpa = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
//
//    // println("已使用 nn.functional.scaled_dot_product_attention")
//    catch
//      case e: AttributeError =>
//      // println("scaled_dot_product_attention 不可用（需要 PyTorch 2.0+）。")
//      // 回退到 nn.MultiheadAttention 或手动实现
//      println("回退到 nn.MultiheadAttention 或手动实现")
//
//    // 使用 nn.MultiheadAttention 的示例（需要特定格式的遮罩）
//    // MHA 期望布尔遮罩为 (Batch * NumHeads, TargetSeqLen, SourceSeqLen) 或 (TargetSeqLen, SourceSeqLen)
//    val multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first = true)
//    // MHA 遮罩：True 表示该位置*将被阻止*关注。
//    // 创建一个更简单的遮罩用于说明（适用于所有头/批次）
//    val mha_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).bools()
//    // attn_output, attn_weights = multihead_attn(query, key, value, attn_mask=mha_mask)
//    // println("已使用带遮罩的 nn.MultiheadAttention")
//
//
//    //07
//    val x = torch.tensor(Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6)), dtype = torch.float32)
//
//    // 边：(0 -> 1), (1 -> 0), (1 -> 2), (2 -> 1)
//    // 表示为源节点和目标节点
//    val edge_index = torch.tensor(Seq(Seq(0, 1, 1, 2), // 源节点
//      Seq(1, 0, 2, 1)), // 目标节点
//      dtype = torch.Int64)
//
//    // 可选的边特征：4 条边，每条边 1 个特征
//    val edge_attr = torch.tensor(Seq(Seq(0.5), Seq(0.5), Seq(0.8), Seq(0.8)), dtype = torch.float32)
//
//    // 可选的节点标签（例如，用于节点分类）
//    val y = torch.tensor(Seq(0, 1, 0), dtype = torch.Int64)
//
//    // 创建 Data 对象
//    val graph_data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = y)
//
//    println(graph_data)
//    // 输出：Data(x=[3, 2], edge_index=[2, 4], edge_attr=[4, 1], y=[3])
//
//  }
//
//}
//
//class SimpleGCN[ParamType <: FloatNN: Default](num_node_features: Int, num_classes: Int, hidden_channels: Int) extends TensorModule[ParamType]  with HasParams[ParamType] {
//
//  val conv1 = GCNConv(num_node_features, hidden_channels)
//  val conv2 = GCNConv(hidden_channels, num_classes)
//
//  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = ???
//
//  def apply(data: GraphData): Tensor[ParamType] = forward(data)
//
//  def forward(data: GraphData): Tensor[ParamType] = {
//    var x = data.x
//    val edge_index = data.edge_index
//    x = conv1(x, edge_index)
//    x = F.relu(x)
//    x = F.dropout(x, p = 0.5, training = self.training) // 经常使用 Dropout
//    x = conv2(x, edge_index)
//    // 对节点进行分类，通常使用 LogSoftmax
//    F.log_softmax(x, dim = 1)
//  }
//}
//
//class SimpleGAT[ParamType <: FloatNN: Default](num_node_features: Int, num_classes: Int, hidden_channels: Int, heads: Int = 8) extends TensorModule[ParamType]  with HasParams[ParamType] {
//
//
//  // 在第一层中使用多头注意力
//  val conv1 = GATConv(num_node_features, hidden_channels, heads = heads, dropout = 0.6)
//  // 多头注意力的输出特征为 heads * hidden_channels
//  // 对于最后一层，通常平均各头或使用单头
//  val conv2 = GATConv(hidden_channels * heads, num_classes, heads = 1, concat = False, dropout = 0.6)
//
//  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = ???
//
//  def apply(data: GraphData): Tensor[ParamType] = forward(data)
//
//  def forward(data: GraphData): Tensor[ParamType] = {
//    var x = data.x
//    val edge_index = data.edge_index
//    x = F.dropout(x, p = 0.6, training = self.training) // 对输入特征应用 Dropout
//    x = conv1(x, edge_index)
//    x = F.elu(x) // ELU 激活在 GAT 中很常见
//    x = F.dropout(x, p = 0.6, training = self.training)
//    x = conv2(x, edge_index)
//    F.log_softmax(x, dim = 1)
//  }
//}
//
//
//
//
//class SimpleGraphSAGE[ParamType <: FloatNN: Default](num_node_features: Int, num_classes: Int, hidden_channels: Int) extends TensorModule[ParamType]  with HasParams[ParamType] {
//
//  // 默认聚合器是 'mean'
//  val conv1 = SAGEConv(num_node_features, hidden_channels)
//  val conv2 = SAGEConv(hidden_channels, num_classes)
//
//  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = ???
//
//  def apply(data: GraphData): Tensor[ParamType] = forward(data)
//
//  def forward(data: GraphData): Tensor[ParamType] = {
//    var x = data.x
//    val edge_index = data.edge_index
//    x = conv1(x, edge_index)
//    x = F.relu(x)
//    x = F.dropout(x, p = 0.5, training = self.training) // 经常使用 Dropout
//    x = conv2(x, edge_index)
//    // 对节点进行分类，通常使用 LogSoftmax
//    F.log_softmax(x, dim = 1)
//  }
//
//}
