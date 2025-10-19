package lesson

// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import java.io._
import java.util._
import java.util.concurrent._
import com.google.gson._
import org.bytedeco.javacpp._
import org.bytedeco.tritonserver.tritondevelopertoolsserver._
import org.bytedeco.tritonserver.global.tritondevelopertoolsserver._

object SimpleCPPs {
  // Helper functions
  def FAIL(MSG: String): Unit = {
    System.err.println("Failure: " + MSG)
    System.exit(1)
  }

  def Usage(msg: String): Unit = {
    if (msg != null) System.err.println(msg)
    System.err.println("Usage: java "+ " [options]")
    System.err.println("\t-v Enable verbose logging")
    System.err.println("\t-r [model repository absolute path]")
    System.exit(1)
  }

  def CompareResult(output0_name: String, output1_name: String, input0: IntPointer, input1: IntPointer, output0: IntPointer, output1: IntPointer): Unit = {
    for (i <- 0 until 16) {
      System.out.println(input0.get(i) + " + " + input1.get(i) + " = " + output0.get(i))
      System.out.println(input0.get(i) + " - " + input1.get(i) + " = " + output1.get(i))
      if ((input0.get(i) + input1.get(i)) != output0.get(i)) FAIL("incorrect sum in " + output0_name)
      if ((input0.get(i) - input1.get(i)) != output1.get(i)) FAIL("incorrect difference in " + output1_name)
    }
  }

  def GenerateInputData(input0_data: Array[IntPointer], input1_data: Array[IntPointer]): Unit = {
    input0_data(0) = new IntPointer(16l)
    input1_data(0) = new IntPointer(16l)
    for (i <- 0 until 16) {
      input0_data(0).put(i*1l, 2)
      input1_data(0).put(i*1l, 1 * i)
    }
  }

  def RunInference(verbose_level: Int, model_repository_path: String, model_name: String): Int = {
    val model_repository_paths = new StringVector(model_repository_path)
    val options = new ServerOptions(model_repository_paths)
    val logging_options = options.logging_
    logging_options.SetVerboseLevel(verbose_level)
    options.SetLoggingOptions(logging_options)
    val server = GenericTritonServer.Create(options)
    val loaded_models = server.LoadedModels
    System.out.println("Loaded_models count : " + loaded_models.size)
    val infer_options = new InferOptions(model_name)
    val request = GenericInferRequest.Create(infer_options)
    val p0 = Array[IntPointer]()
    val p1 = Array[IntPointer]()
    GenerateInputData(p0, p1)
    var input0_data: BytePointer  = p0(0).getPointer(classOf[BytePointer])
    var input1_data: BytePointer = p1(0).getPointer(classOf[BytePointer])
    val shape0 = new LongPointer(2)
    val shape1 = new LongPointer(2)
    shape0.put(0, 1)
    shape0.put(1, 16)
    shape1.put(0, 1)
    shape1.put(1, 16)
    val tensor0 = new Tensor(input0_data, 4 * 16, 8, shape0, 0, 1)
    val tensor1 = new Tensor(input1_data, 4 * 16, 8, shape1, 0, 1)
    request.AddInput("INPUT0", tensor0)
    request.AddInput("INPUT1", tensor1)
    val result = server.Infer(request)
    val output = result.Output("OUTPUT0")
    val buffer = output.buffer_
    System.out.println("buffer to string : " + buffer.toString)
    System.out.println("output val at index 0 : " + buffer.getInt(0))
    System.out.println("output val at index 1 : " + buffer.getInt(1 * 4))
    System.out.println("output val at index 2 : " + buffer.getInt(2 * 4))
    System.out.println("output val at index 3 : " + buffer.getInt(3 * 4))
    System.out.println("output val at index 4 : " + buffer.getInt(4 * 4))
    System.out.println("output val at index 5 : " + buffer.getInt(5 * 4))
    System.out.println("output val at index 6 : " + buffer.getInt(6 * 4))
    System.out.println("output val at index 7 : " + buffer.getInt(7 * 4))
    0
  }

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    var model_repository_path = "./models"
    var verbose_level = 0
    // Parse commandline...
    var i = 0
    while (i < args.length) {
      args(i) match {
        case "-r" =>
          model_repository_path = args({
            i += 1; i
          })
        case "-v" =>
          verbose_level = 1
        case "-?" =>
          Usage(null)
      }
      i += 1
    }
    // We use a simple model that takes 2 input tensors of 16 strings
    // each and returns 2 output tensors of 16 strings each. The input
    // strings must represent integers. One output tensor is the
    // element-wise sum of the inputs and one output is the element-wise
    // difference.
    val model_name = "simple"
    if (model_repository_path == null) Usage("-r must be used to specify model repository path")
    RunInference(verbose_level, model_repository_path, model_name)
    System.exit(0)
  }
}