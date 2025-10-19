package lesson

import com.google.gson.*
import org.bytedeco.cuda.cudart.*
import org.bytedeco.cuda.global.cudart.*
import org.bytedeco.javacpp.*
import org.bytedeco.tritonserver.global.tritonserver.*
import org.bytedeco.tritonserver.tritonserver.*

import java.io.*
import java.util.*
import java.util.concurrent.*
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.util.control.Breaks.{break, breakable}

object SimpleGPU {
  val TRITON_MIN_COMPUTE_CAPABILITY = 6.0

  def FAIL(MSG: String): Unit = {
    System.err.println("Cuda failure: " + MSG)
    System.exit(1)
  }

  def FAIL_IF_ERR(err: TRITONSERVER_Error, MSG: String): Unit = {
    if (err != null) {
      System.err.println(
        "error: " + MSG + ":" + TRITONSERVER_ErrorCodeString(
          err
        ) + " - " + TRITONSERVER_ErrorMessage(err)
      )
      TRITONSERVER_ErrorDelete(err)
      System.exit(1)
    }
  }

  def FAIL_IF_CUDA_ERR(err: Int, MSG: String): Unit = {
    if (err != cudaSuccess) {
      System.err.println("error: " + MSG + ": " + cudaGetErrorString(err))
      System.exit(1)
    }
  }

  var enforce_memory_type = false
  var requested_memory_type = 0

  object CudaDataDeleter {
    protected class FreeDeallocator(p: Pointer) extends Pointer(p) with Pointer.Deallocator {
      override def deallocate(): Unit = {
        if (!isNull) {
          val attr = new cudaPointerAttributes(null)
          var cuerr = cudaPointerGetAttributes(attr, this)
          if (cuerr != cudaSuccess)
            System.err.println(
              "error: failed to get CUDA pointer attribute of " + this + ": " + cudaGetErrorString(
                cuerr
              ).getString
            )
          if (attr.`type` == cudaMemoryTypeDevice) cuerr = cudaFree(this)
          else if (attr.`type` == cudaMemoryTypeHost) cuerr = cudaFreeHost(this)
          if (cuerr != cudaSuccess)
            System.err.println(
              "error: failed to release CUDA pointer " + this + ": " + cudaGetErrorString(
                cuerr
              ).getString
            )
        }
      }
    }
  }

  class CudaDataDeleter extends BytePointer {
    def reset(p: Pointer): Unit = {
      this.address = p.address
      this.deallocator(new CudaDataDeleter.FreeDeallocator(this))
    }
  }

  object TRITONSERVER_ServerDeleter {
    protected class DeleteDeallocator(p: Pointer)
        extends TRITONSERVER_Server(p)
        with Pointer.Deallocator {
      override def deallocate(): Unit = {
        TRITONSERVER_ServerDelete(this)
      }
    }
  }

  class TRITONSERVER_ServerDeleter(p: TRITONSERVER_Server) extends TRITONSERVER_Server(p) {
    deallocator(new TRITONSERVER_ServerDeleter.DeleteDeallocator(this))
  }

  def Usage(msg: String): Unit = {
    if (msg != null) System.err.println(msg)
    System.err.println("Usage: java " + "simpleGpu" + " [options]")
    System.err.println(
      "\t-m <\"system\"|\"pinned\"|gpu>" + " Enforce the memory type for input and output tensors." + " If not specified, inputs will be in system memory and outputs" + " will be based on the model's preferred type."
    )
    System.err.println("\t-v Enable verbose logging")
    System.err.println("\t-r [model repository absolute path]")
    System.exit(1)
  }

  class ResponseAlloc extends TRITONSERVER_ResponseAllocatorAllocFn_t {
    override def call(
        allocator: TRITONSERVER_ResponseAllocator,
        tensor_name: String,
        byte_size: Long,
        preferred_memory_type: Int,
        preferred_memory_type_id: Long,
        userp: Pointer,
        buffer: PointerPointer[? <: Pointer],
        buffer_userp: PointerPointer[? <: Pointer],
        actual_memory_type: IntPointer,
        actual_memory_type_id: LongPointer
    ): TRITONSERVER_Error = {
      // Initially attempt to make the actual memory type and id that we
      // allocate be the same as preferred memory type
      actual_memory_type.put(0L, preferred_memory_type)
      actual_memory_type_id.put(0, preferred_memory_type_id)
      // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
      // need to do any other book-keeping.
      if (byte_size == 0) {
        buffer.put(0, null)
        buffer_userp.put(0, null)
        System.out.println("allocated " + byte_size + " bytes for result tensor " + tensor_name)
      } else {
        var allocated_ptr = new Pointer
        if (enforce_memory_type) actual_memory_type.put(0L, requested_memory_type)
        actual_memory_type.get match {
          case TRITONSERVER_MEMORY_CPU_PINNED =>
            var err = cudaSetDevice(actual_memory_type_id.get.toInt)
            if (
              (err != cudaSuccess) && (err != cudaErrorNoDevice) && (err != cudaErrorInsufficientDriver)
            )
              return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                "unable to recover current CUDA device: " + cudaGetErrorString(err).getString
              )
            err = cudaHostAlloc(allocated_ptr, byte_size, cudaHostAllocPortable)
            if (err != cudaSuccess)
              return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                "cudaHostAlloc failed: " + cudaGetErrorString(err).getString
              )
          case TRITONSERVER_MEMORY_GPU =>
            var err = cudaSetDevice(actual_memory_type_id.get.toInt)
            if (
              (err != cudaSuccess) && (err != cudaErrorNoDevice) && (err != cudaErrorInsufficientDriver)
            )
              return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                "unable to recover current CUDA device: " + cudaGetErrorString(err).getString
              )
            err = cudaMalloc(allocated_ptr, byte_size)
            if (err != cudaSuccess)
              return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                "cudaMalloc failed: " + cudaGetErrorString(err).getString
              )

          // Use CPU memory if the requested memory type is unknown
          // (default case).
          case TRITONSERVER_MEMORY_CPU =>
          case _ =>
            actual_memory_type.put(0L, TRITONSERVER_MEMORY_CPU)
            allocated_ptr = Pointer.malloc(byte_size)
        }
        // Pass the tensor name with buffer_userp so we can show it when
        // releasing the buffer.
        if (!allocated_ptr.isNull) {
          buffer.put(0, allocated_ptr)
          buffer_userp.put(0, Loader.newGlobalRef(tensor_name))
          System.out.println(
            "allocated " + byte_size + " bytes in " + TRITONSERVER_MemoryTypeString(
              actual_memory_type.get
            ) + " for result tensor " + tensor_name
          )
        }
      }
      null // Success
    }
  }

  class ResponseRelease extends TRITONSERVER_ResponseAllocatorReleaseFn_t {
    override def call(
        allocator: TRITONSERVER_ResponseAllocator,
        buffer: Pointer,
        buffer_userp: Pointer,
        byte_size: Long,
        memory_type: Int,
        memory_type_id: Long
    ): TRITONSERVER_Error = {
      var name: String = null
      if (buffer_userp != null) name = Loader.accessGlobalRef(buffer_userp).asInstanceOf[String]
      else name = "<unknown>"
      System.out.println(
        "Releasing buffer " + buffer + " of size " + byte_size + " in " + TRITONSERVER_MemoryTypeString(
          memory_type
        ) + " for result '" + name + "'"
      )
      memory_type match {
        case TRITONSERVER_MEMORY_CPU =>
          Pointer.free(buffer)
        case TRITONSERVER_MEMORY_CPU_PINNED =>
          var err = cudaSetDevice(memory_type_id.toInt)
          if (err == cudaSuccess) err = cudaFreeHost(buffer)
          if (err != cudaSuccess)
            System.err.println(
              "error: failed to cudaFree " + buffer + ": " + cudaGetErrorString(err)
            )
        case TRITONSERVER_MEMORY_GPU =>
          var err = cudaSetDevice(memory_type_id.toInt)
          if (err == cudaSuccess) err = cudaFree(buffer)
          if (err != cudaSuccess)
            System.err.println(
              "error: failed to cudaFree " + buffer + ": " + cudaGetErrorString(err)
            )
        case _ =>
          System.err.println("error: unexpected buffer allocated in CUDA managed memory")
      }
      Loader.deleteGlobalRef(buffer_userp)
      null // Success
    }
  }

  class InferRequestComplete extends TRITONSERVER_InferenceRequestReleaseFn_t {
    override def call(request: TRITONSERVER_InferenceRequest, flags: Int, userp: Pointer): Unit = {

      // We reuse the request so we don't delete it here.
    }
  }

  class InferResponseComplete extends TRITONSERVER_InferenceResponseCompleteFn_t {
    override def call(
        response: TRITONSERVER_InferenceResponse,
        flags: Int,
        userp: Pointer
    ): Unit = {
      if (response != null) {
        // Send 'response' to the future.
        futures.get(userp).complete(response)
      }
    }
  }

  val futures = new ConcurrentHashMap[Pointer, CompletableFuture[TRITONSERVER_InferenceResponse]]
  val responseAlloc = new SimpleGPU.ResponseAlloc
  val responseRelease = new SimpleGPU.ResponseRelease
  val inferRequestComplete = new SimpleGPU.InferRequestComplete
  val inferResponseComplete = new SimpleGPU.InferResponseComplete

  def ParseModelMetadata(
      model_metadata: JsonObject,
      is_int: Array[Boolean],
      is_torch_model: Array[Boolean]
  ): TRITONSERVER_Error = {
    var seen_data_type: String = null
//    import scala.collection.JavaConversions._

    for (input_element <- model_metadata.get("inputs").getAsJsonArray.asScala) {
      val input = input_element.getAsJsonObject
      if (
        !(input.get("datatype").getAsString == "INT32") && !(input
          .get("datatype")
          .getAsString == "FP32")
      )
        return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "simple lib example only supports model with data type INT32 or " + "FP32"
        )
      if (seen_data_type == null) seen_data_type = input.get("datatype").getAsString
      else if (!(seen_data_type == input.get("datatype").getAsString))
        return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of 'simple' model must have the data type"
        )
    }
//    import scala.collection.JavaConversions._
    for (output_element <- model_metadata.get("outputs").getAsJsonArray.asScala) {
      val output = output_element.getAsJsonObject
      if (
        !(output.get("datatype").getAsString == "INT32") && !(output
          .get("datatype")
          .getAsString == "FP32")
      )
        return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "simple lib example only supports model with data type INT32 or " + "FP32"
        )
      else if (!(seen_data_type == output.get("datatype").getAsString))
        return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of 'simple' model must have the data type"
        )
    }
    is_int(0) = seen_data_type == "INT32"
    is_torch_model(0) = model_metadata.get("platform").getAsString == "pytorch_libtorch"
    null
  }

  def GenerateInputData(input0_data: Array[IntPointer], input1_data: Array[IntPointer]): Unit = {
    input0_data(0) = new IntPointer(16L)
    input1_data(0) = new IntPointer(16L)
    for (i <- 0 until 16) {
      input0_data(0).put(i * 1L, i)
      input1_data(0).put(i * 1L, 1)
    }
  }

  def GenerateInputData(
      input0_data: Array[FloatPointer],
      input1_data: Array[FloatPointer]
  ): Unit = {
    input0_data(0) = new FloatPointer(16)
    input1_data(0) = new FloatPointer(16)
    for (i <- 0 until 16) {
      input0_data(0).put(i, i)
      input1_data(0).put(i, 1)
    }
  }

  def CompareResult(
      output0_name: String,
      output1_name: String,
      input0: IntPointer,
      input1: IntPointer,
      output0: IntPointer,
      output1: IntPointer
  ): Unit = {
    for (i <- 0 until 16) {
      System.out.println(input0.get(i) + " + " + input1.get(i) + " = " + output0.get(i))
      System.out.println(input0.get(i) + " - " + input1.get(i) + " = " + output1.get(i))
      if ((input0.get(i) + input1.get(i)) != output0.get(i))
        FAIL("incorrect sum in " + output0_name)
      if ((input0.get(i) - input1.get(i)) != output1.get(i))
        FAIL("incorrect difference in " + output1_name)
    }
  }

  def CompareResult(
      output0_name: String,
      output1_name: String,
      input0: FloatPointer,
      input1: FloatPointer,
      output0: FloatPointer,
      output1: FloatPointer
  ): Unit = {
    for (i <- 0 until 16) {
      System.out.println(input0.get(i) + " + " + input1.get(i) + " = " + output0.get(i))
      System.out.println(input0.get(i) + " - " + input1.get(i) + " = " + output1.get(i))
      if ((input0.get(i) + input1.get(i)) != output0.get(i))
        FAIL("incorrect sum in " + output0_name)
      if ((input0.get(i) - input1.get(i)) != output1.get(i))
        FAIL("incorrect difference in " + output1_name)
    }
  }

  def Check(
      response: TRITONSERVER_InferenceResponse,
      input0_data: Pointer,
      input1_data: Pointer,
      output0: String,
      output1: String,
      expected_byte_size: Long,
      expected_datatype: Int,
      is_int: Boolean
  ): Unit = {
    val output_data = new mutable.HashMap[String, Pointer]
    val output_count = Array(0)
    FAIL_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(response, output_count),
      "getting number of response outputs"
    )
    if (output_count(0) != 2) FAIL("expecting 2 response outputs, got " + output_count(0))
    for (idx <- 0 until output_count(0)) {
      val cname = new BytePointer(null.asInstanceOf[Pointer])
      val datatype = new IntPointer(1L)
      val shape = new LongPointer(null.asInstanceOf[Pointer])
      val dim_count = new LongPointer(1)
      val base = new Pointer
      val byte_size = new SizeTPointer(1)
      val memory_type = new IntPointer(1L)
      val memory_type_id = new LongPointer(1)
      val userp = new Pointer
      FAIL_IF_ERR(
        TRITONSERVER_InferenceResponseOutput(
          response,
          idx,
          cname,
          datatype,
          shape,
          dim_count,
          base,
          byte_size,
          memory_type,
          memory_type_id,
          userp
        ),
        "getting output info"
      )
      if (cname.isNull) FAIL("unable to get output name")
      val name = cname.getString
      if ((!(name == output0)) && (!(name == output1))) FAIL("unexpected output '" + name + "'")
      if ((dim_count.get != 2) || (shape.get(0) != 1) || (shape.get(1) != 16))
        FAIL("unexpected shape for '" + name + "'")
      if (datatype.get != expected_datatype)
        FAIL(
          "unexpected datatype '" + TRITONSERVER_DataTypeString(
            datatype.get
          ) + "' for '" + name + "'"
        )
      if (byte_size.get != expected_byte_size)
        FAIL(
          "unexpected byte-size, expected " + expected_byte_size + ", got " + byte_size.get + " for " + name
        )
      if (enforce_memory_type && (memory_type.get != requested_memory_type))
        FAIL(
          "unexpected memory type, expected to be allocated in " + TRITONSERVER_MemoryTypeString(
            requested_memory_type
          ) + ", got " + TRITONSERVER_MemoryTypeString(
            memory_type.get
          ) + ", id " + memory_type_id.get + " for " + name
        )
      // We make a copy of the data here... which we could avoid for
      // performance reasons but ok for this simple example.
      val odata = new BytePointer(byte_size.get)
      output_data.put(name, odata)
      memory_type.get match {
        case TRITONSERVER_MEMORY_CPU =>
          System.out.println(name + " is stored in system memory")
          odata.put(base.limit(byte_size.get))
        case TRITONSERVER_MEMORY_CPU_PINNED =>
          System.out.println(name + " is stored in pinned memory")
          odata.put(base.limit(byte_size.get))
        case TRITONSERVER_MEMORY_GPU =>
          System.out.println(name + " is stored in GPU memory")
          FAIL_IF_CUDA_ERR(
            cudaMemcpy(odata, base, byte_size.get, cudaMemcpyDeviceToHost),
            "getting " + name + " data from GPU memory"
          )
        case _ =>
          FAIL("unexpected memory type")
      }
    }
    if (is_int)
      CompareResult(
        output0,
        output1,
        new IntPointer(input0_data),
        new IntPointer(input1_data),
        new IntPointer(output_data.get(output0).get),
        new IntPointer(output_data.get(output1).get)
      )
    else
      CompareResult(
        output0,
        output1,
        new FloatPointer(input0_data),
        new FloatPointer(input1_data),
        new FloatPointer(output_data.get(output0).get),
        new FloatPointer(output_data.get(output1).get)
      )
  }

  @throws[Exception]
  def RunInference(model_repository_path: String, verbose_level: Int): Unit = {
    // Check API version.
    val api_version_major = Array(0)
    val api_version_minor = Array(0)
    FAIL_IF_ERR(
      TRITONSERVER_ApiVersion(api_version_major, api_version_minor),
      "getting Triton API version"
    )
    if (
      (TRITONSERVER_API_VERSION_MAJOR != api_version_major(
        0
      )) || (TRITONSERVER_API_VERSION_MINOR > api_version_minor(0))
    ) FAIL("triton server API version mismatch")
    // Create the server...
    val server_options = new TRITONSERVER_ServerOptions(null)
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsNew(server_options), "creating server options")
    FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options, model_repository_path),
      "setting model repository path"
    )
    FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level),
      "setting verbose logging level"
    )
    FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetBackendDirectory(server_options, "/opt/tritonserver/backends"),
      "setting backend directory"
    )
    FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
        server_options,
        "/opt/tritonserver/repoagents"
      ),
      "setting repository agent directory"
    )
    FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true),
      "setting strict model configuration"
    )
    val min_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY
    FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
        server_options,
        min_compute_capability
      ),
      "setting minimum supported CUDA compute capability"
    )
    val server_ptr = new TRITONSERVER_Server(null)
    FAIL_IF_ERR(TRITONSERVER_ServerNew(server_ptr, server_options), "creating server")
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsDelete(server_options), "deleting server options")
    val server = new SimpleGPU.TRITONSERVER_ServerDeleter(server_ptr)
    // Wait until the server is both live and ready.
    var health_iters = 0
    while (true) {
      val live = Array(false)
      val ready = Array(false)
      FAIL_IF_ERR(TRITONSERVER_ServerIsLive(server, live), "unable to get server liveness")
      FAIL_IF_ERR(TRITONSERVER_ServerIsReady(server, ready), "unable to get server readiness")
      System.out.println("Server Health: live " + live(0) + ", ready " + ready(0))
      if (live(0) && ready(0)) break // todo: break is not supported
      if ({
        health_iters += 1; health_iters
      } >= 10) FAIL("failed to find healthy inference server")
      Thread.sleep(500)
    }
    // Print status of the server.
    val server_metadata_message = new TRITONSERVER_Message(null)
    FAIL_IF_ERR(
      TRITONSERVER_ServerMetadata(server, server_metadata_message),
      "unable to get server metadata message"
    )
    val buffer = new BytePointer(null.asInstanceOf[Pointer])
    val byte_size = new SizeTPointer(1)
    FAIL_IF_ERR(
      TRITONSERVER_MessageSerializeToJson(server_metadata_message, buffer, byte_size),
      "unable to serialize server metadata message"
    )
    System.out.println("Server Status:")
    System.out.println(buffer.limit(byte_size.get).getString)
    FAIL_IF_ERR(TRITONSERVER_MessageDelete(server_metadata_message), "deleting status metadata")
    val model_name = "simple"
    // Wait for the model to become available.
    val is_torch_model = Array(false)
    val is_int = Array(true)
    val is_ready = Array(false)
    health_iters = 0
    while (!is_ready(0)) {
      FAIL_IF_ERR(
        TRITONSERVER_ServerModelIsReady(server, model_name, 1, is_ready),
        "unable to get model readiness"
      )
      breakable {
        if (!is_ready(0)) {
          if ({
            health_iters += 1; health_iters
          } >= 10) FAIL("model failed to be ready in 10 iterations")
          Thread.sleep(500)
          break // continue //todo: continue is not supported
        }
      }
      val model_metadata_message = new TRITONSERVER_Message(null)
      FAIL_IF_ERR(
        TRITONSERVER_ServerModelMetadata(server, model_name, 1, model_metadata_message),
        "unable to get model metadata message"
      )
      val buffer = new BytePointer(null.asInstanceOf[Pointer])
      val byte_size = new SizeTPointer(1)
      FAIL_IF_ERR(
        TRITONSERVER_MessageSerializeToJson(model_metadata_message, buffer, byte_size),
        "unable to serialize model status protobuf"
      )
      val parser = new JsonParser
      var model_metadata: JsonObject = null
      try model_metadata = parser.parse(buffer.limit(byte_size.get).getString).getAsJsonObject
      catch {
        case e: Exception =>
          FAIL("error: failed to parse model metadata from JSON: " + e)
      }
      FAIL_IF_ERR(TRITONSERVER_MessageDelete(model_metadata_message), "deleting status protobuf")
      if (!(model_metadata.get("name").getAsString == model_name))
        FAIL("unable to find metadata for model")
      var found_version = false
      if (model_metadata.has("versions")) {
        val versions = model_metadata.get("versions").getAsJsonArray.asScala
        found_version = versions.exists(_.getAsString == "1")
//        for (version <- versions) {
//          if (version.getAsString == "1") {
//            found_version = true
//            break //todo: break is not supported
//          }
//        }
      }
      if (!found_version) FAIL("unable to find version 1 status for model")
      FAIL_IF_ERR(
        ParseModelMetadata(model_metadata, is_int, is_torch_model),
        "parsing model metadata"
      )
    }
    // Create the allocator that will be used to allocate buffers for
    // the result tensors.
    val allocator = new TRITONSERVER_ResponseAllocator(null)
    FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorNew(
        allocator,
        responseAlloc,
        responseRelease,
        null /* start_fn */
      ),
      "creating response allocator"
    )
    // Inference
    val irequest = new TRITONSERVER_InferenceRequest(null)
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestNew(irequest, server, model_name, -1 /* model_version */ ),
      "creating inference request"
    )
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestSetId(irequest, "my_request_id"),
      "setting ID for the request"
    )
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestSetReleaseCallback(
        irequest,
        inferRequestComplete,
        null /* request_release_userp */
      ),
      "setting request release callback"
    )
    // Inputs
    val input0 =
      if (is_torch_model(0)) "INPUT__0"
      else "INPUT0"
    val input1 =
      if (is_torch_model(0)) "INPUT__1"
      else "INPUT1"
    val input0_shape = Array(1, 16)
    val input1_shape = Array(1, 16)
    val datatype =
      if (is_int(0)) TRITONSERVER_TYPE_INT32
      else TRITONSERVER_TYPE_FP32
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
        irequest,
        input0,
        datatype,
        input0_shape.map(_.toLong),
        input0_shape.length
      ),
      "setting input 0 meta-data for the request"
    )
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
        irequest,
        input1,
        datatype,
        input1_shape.map(_.toLong),
        input1_shape.length
      ),
      "setting input 1 meta-data for the request"
    )
    val output0 =
      if (is_torch_model(0)) "OUTPUT__0"
      else "OUTPUT0"
    val output1 =
      if (is_torch_model(0)) "OUTPUT__1"
      else "OUTPUT1"
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output0),
      "requesting output 0 for the request"
    )
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output1),
      "requesting output 1 for the request"
    )
    // Create the data for the two input tensors. Initialize the first
    // to unique values and the second to all ones.
    var input0_data: BytePointer = null
    var input1_data: BytePointer = null
    if (is_int(0)) {
      val p0 = Array[IntPointer]()
      val p1 = Array[IntPointer]()
      GenerateInputData(p0, p1)
      input0_data = p0(0).getPointer(classOf[BytePointer])
      input1_data = p1(0).getPointer(classOf[BytePointer])
    } else {
      val p0 = Array[IntPointer]()
      val p1 = Array[IntPointer]()
      GenerateInputData(p0, p1)
      input0_data = p0(0).getPointer(classOf[BytePointer])
      input1_data = p1(0).getPointer(classOf[BytePointer])
    }
    val input0_size = input0_data.limit
    val input1_size = input1_data.limit
    var input0_base = input0_data
    var input1_base = input1_data
    val input0_gpu = new CudaDataDeleter
    val input1_gpu = new CudaDataDeleter
    val use_cuda_memory = enforce_memory_type && (requested_memory_type != TRITONSERVER_MEMORY_CPU)
    if (use_cuda_memory) {
      FAIL_IF_CUDA_ERR(cudaSetDevice(0), "setting CUDA device to device 0")
      if (requested_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
        val dst = new Pointer
        FAIL_IF_CUDA_ERR(cudaMalloc(dst, input0_size), "allocating GPU memory for INPUT0 data")
        input0_gpu.reset(dst)
        FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, input0_data, input0_size, cudaMemcpyHostToDevice),
          "setting INPUT0 data in GPU memory"
        )
        FAIL_IF_CUDA_ERR(cudaMalloc(dst, input1_size), "allocating GPU memory for INPUT1 data")
        input1_gpu.reset(dst)
        FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, input1_data, input1_size, cudaMemcpyHostToDevice),
          "setting INPUT1 data in GPU memory"
        )
      } else {
        val dst = new Pointer
        FAIL_IF_CUDA_ERR(
          cudaHostAlloc(dst, input0_size, cudaHostAllocPortable),
          "allocating pinned memory for INPUT0 data"
        )
        input0_gpu.reset(dst)
        FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, input0_data, input0_size, cudaMemcpyHostToHost),
          "setting INPUT0 data in pinned memory"
        )
        FAIL_IF_CUDA_ERR(
          cudaHostAlloc(dst, input1_size, cudaHostAllocPortable),
          "allocating pinned memory for INPUT1 data"
        )
        input1_gpu.reset(dst)
        FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, input1_data, input1_size, cudaMemcpyHostToHost),
          "setting INPUT1 data in pinned memory"
        )
      }
      input0_base = input0_gpu
      input1_base = input1_gpu
    }
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAppendInputData(
        irequest,
        input0,
        input0_base,
        input0_size,
        requested_memory_type,
        0 /* memory_type_id */
      ),
      "assigning INPUT0 data"
    )
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAppendInputData(
        irequest,
        input1,
        input1_base,
        input1_size,
        requested_memory_type,
        0 /* memory_type_id */
      ),
      "assigning INPUT1 data"
    )
    // Perform inference...
    val completed = new CompletableFuture[TRITONSERVER_InferenceResponse]
    futures.put(irequest, completed)
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestSetResponseCallback(
        irequest,
        allocator,
        null /* response_allocator_userp */,
        inferResponseComplete,
        irequest
      ),
      "setting response callback"
    )
    FAIL_IF_ERR(
      TRITONSERVER_ServerInferAsync(server, irequest, null /* trace */ ),
      "running inference"
    )
    // Wait for the inference to complete.
    var completed_response = completed.get
    futures.remove(irequest)
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(completed_response), "response status")
    Check(
      completed_response,
      input0_data,
      input1_data,
      output0,
      output1,
      input0_size,
      datatype,
      is_int(0)
    )
    FAIL_IF_ERR(
      TRITONSERVER_InferenceResponseDelete(completed_response),
      "deleting inference response"
    )

    // Modify some input data in place and then reuse the request
    // object. For simplicity we only do this when the input tensors are
    // in non-pinned system memory.
//    val completed = new CompletableFuture[TRITONSERVER_InferenceResponse]
    if (!enforce_memory_type || (requested_memory_type == TRITONSERVER_MEMORY_CPU)) {
      if (is_int(0)) new IntPointer(input0_data).put(0L, 27)
      else new FloatPointer(input0_data).put(0, 27.0f)
      val completed = new CompletableFuture[TRITONSERVER_InferenceResponse]
      futures.put(irequest, completed)
      // Using a new promise so have to re-register the callback to set
      // the promise as the userp.
      FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestSetResponseCallback(
          irequest,
          allocator,
          null /* response_allocator_userp */,
          inferResponseComplete,
          irequest
        ),
        "setting response callback"
      )
      FAIL_IF_ERR(
        TRITONSERVER_ServerInferAsync(server, irequest, null /* trace */ ),
        "running inference"
      )
      // Wait for the inference to complete.
      val completed_response = completed.get
      futures.remove(irequest)
      FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(completed_response), "response status")
      Check(
        completed_response,
        input0_data,
        input1_data,
        output0,
        output1,
        input0_size,
        datatype,
        is_int(0)
      )
      FAIL_IF_ERR(
        TRITONSERVER_InferenceResponseDelete(completed_response),
        "deleting inference response"
      )
    }
    // Remove input data and then add back different data.
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestRemoveAllInputData(irequest, input0),
      "removing INPUT0 data"
    )
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAppendInputData(
        irequest,
        input0,
        input1_base,
        input1_size,
        requested_memory_type,
        0 /* memory_type_id */
      ),
      "assigning INPUT1 data to INPUT0"
    )

    futures.put(irequest, completed)
    // Using a new promise so have to re-register the callback to set
    // the promise as the userp.
    FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestSetResponseCallback(
        irequest,
        allocator,
        null /* response_allocator_userp */,
        inferResponseComplete,
        irequest
      ),
      "setting response callback"
    )
    FAIL_IF_ERR(
      TRITONSERVER_ServerInferAsync(server, irequest, null /* trace */ ),
      "running inference"
    )
    // Wait for the inference to complete.
    completed_response = completed.get
    futures.remove(irequest)
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(completed_response), "response status")
    // Both inputs are using input1_data...
    Check(
      completed_response,
      input1_data,
      input1_data,
      output0,
      output1,
      input0_size,
      datatype,
      is_int(0)
    )
    FAIL_IF_ERR(
      TRITONSERVER_InferenceResponseDelete(completed_response),
      "deleting inference response"
    )
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestDelete(irequest), "deleting inference request")
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(allocator), "deleting response allocator")
  }

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    var model_repository_path: String = null
    var verbose_level = 0
    // Parse commandline...
    var i = 0
    while (i < args.length) {
      args(i) match {
        case "-m" =>
          enforce_memory_type = true
          i += 1
          if (args(i) == "system") requested_memory_type = TRITONSERVER_MEMORY_CPU
          else if (args(i) == "pinned") requested_memory_type = TRITONSERVER_MEMORY_CPU_PINNED
          else if (args(i) == "gpu") requested_memory_type = TRITONSERVER_MEMORY_GPU
          else
            Usage(
              "-m must be used to specify one of the following types:" + " <\"system\"|\"pinned\"|gpu>"
            )
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
    if (model_repository_path == null) Usage("-r must be used to specify model repository path")
    try {
      val scope = new PointerScope
      try RunInference(model_repository_path, verbose_level)
      finally if (scope != null) scope.close()
    }
    System.exit(0)
  }
}
