/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#include "delegate_main.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "op_map.h"
#include "utils.h"
#include "vx_delegate_dmabuf.h"
#include "camera_adaptor/camera_adaptor.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tim/transform/layout_inference.h"
#include "tim/transform/mean_stddev_normalize_fusion.h"

#ifdef NODE_TRACE_DB_MODE
#include "json/json.h"
#endif

using namespace tflite;
namespace {

TfLiteRegistration DelegateNodeRegistration() {
  TfLiteRegistration r;

  r.builtin_code = kTfLiteBuiltinDelegate;
  r.custom_name = "Vx Delegate";

  r.init = [](TfLiteContext* context, const char* buffer, size_t) -> void* {
    auto* params = reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    auto* derivedDelegate = reinterpret_cast<vx::delegate::DerivedDelegateData*>(params->delegate);

    std::unique_ptr<vx::delegate::Delegate> delegate(
        new vx::delegate::Delegate);

    // Pass the shared DmaBufManager and parent delegate reference
    if (derivedDelegate->dmabuf_manager) {
      delegate->SetDmaBufManager(derivedDelegate->dmabuf_manager);
    }
    delegate->SetParentDelegate(derivedDelegate);

    std::unique_ptr<vx::delegate::OpData> op_data =
        delegate->Init(context, params);
    op_data->delegate.swap(delegate);
    return op_data.release();
  };

  r.free = [](TfLiteContext* context, void* buffer) -> void {
    std::unique_ptr<vx::delegate::OpData> op_data(
        reinterpret_cast<vx::delegate::OpData*>(buffer));
    op_data->delegate = nullptr;
  };

  r.prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
    auto op_data = reinterpret_cast<vx::delegate::OpData*>(node->user_data);
    return op_data->delegate->Prepare(*op_data, context, node);
  };

  r.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
    auto op_data = reinterpret_cast<vx::delegate::OpData*>(node->user_data);
    return op_data->delegate->Invoke(*op_data, context, node);
  };

  r.profiling_string = nullptr;
  r.builtin_code = kTfLiteBuiltinDelegate;
  r.version = 1;
  r.registration_external = nullptr;

  return r;
}

TfLiteStatus PrepareDelegate(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteIntArray* plan;
  TfLiteNode* node;
  TfLiteRegistration* registration;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));

  // Get a list of supported nodes.
  std::vector<int> supported_nodes = {0};
  for (int node_index : tflite::TfLiteIntArrayView(plan)) {
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    if (vx::delegate::Delegate::SupportedOp(context, node, registration)) {
      supported_nodes.push_back(node_index);
    }
  }

  // Set first element to the number of nodes to replace.
  supported_nodes[0] = supported_nodes.size() - 1;

  // Replace supported subgraphs.
  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context,
      DelegateNodeRegistration(),
      reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()),
      delegate);
}

TfLiteStatus CopyFromBufferHandle(TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  TfLiteBufferHandle buffer_handle,
                                  TfLiteTensor* tensor) {
  // Copies the data from delegate buffer into the tensor raw memory.
  TFLITE_LOG(TFLITE_LOG_INFO, "CopyFromBufferHandle handle: %d , name: %s",  buffer_handle, tensor->name);
  return kTfLiteOk;
}

void FreeBufferHandle(TfLiteContext* context,
                      TfLiteDelegate* delegate,
                      TfLiteBufferHandle* handle) {
  // Do any cleanups.
  TFLITE_LOG(TFLITE_LOG_INFO, "FreeBufferHandle handle: %d", *handle);
}

std::vector<uint32_t> TfLiteTensorDims(const TfLiteTensor* tensor) {
  std::vector<uint32_t> dims(tensor->dims->size);
  for (std::vector<uint32_t>::size_type i = 0; i < dims.size(); i++) {
    dims[i] = tensor->dims->data[i];
  }
  return dims;
}

tim::vx::DataType TfLiteDtypeToVsiDtype(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return tim::vx::DataType::FLOAT32;
    case kTfLiteInt32:
      return tim::vx::DataType::INT32;
    case kTfLiteUInt8:
      return tim::vx::DataType::UINT8;
    case kTfLiteInt16:
      return tim::vx::DataType::INT16;
    case kTfLiteInt8:
      return tim::vx::DataType::INT8;
    case kTfLiteBool:
      return tim::vx::DataType::INT8;
    case kTfLiteFloat16:
      return tim::vx::DataType::FLOAT16;
    default:
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "Unsuppoted datatype: %d, will be ignored.", type);
      break;
  }

  return tim::vx::DataType::FLOAT32;
}

bool IsConstTensor(const TfLiteTensor* tensor) {
  return tensor->allocation_type == kTfLiteMmapRo? true : false;
}

bool IsVariableTensor(const TfLiteTensor* tensor) {
  return tensor->is_variable;
}

tim::vx::TensorSpec CreateTensorSpec(
    const TfLiteTensor* tensor,
    const std::vector<uint32_t>& perm,
    tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::TRANSIENT) {
  tim::vx::DataType datatype = TfLiteDtypeToVsiDtype(tensor->type);
  std::vector<uint32_t> dims(TfLiteTensorDims(tensor));
  tim::vx::ShapeType whcn_shape(dims.size());

  if (dims.size() == 0) {
    // Use rank 1, shape {1} operand for TFLite scalar tensors.
    dims.push_back(1);
  }

  if (perm.size() > 0) {
    assert(perm.size() == dims.size());
    for (size_t i = 0; i < perm.size(); i++) {
      whcn_shape[i] = dims[perm[i]];
    }
    std::reverse(whcn_shape.begin(), whcn_shape.end());
  } else {
    whcn_shape.assign(dims.rbegin(), dims.rend());
  }

  if (tensor->quantization.type == kTfLiteAffineQuantization) {
    const TfLiteAffineQuantization* params =
        reinterpret_cast<const TfLiteAffineQuantization*>(
            tensor->quantization.params);
    tim::vx::Quantization quantization;
    std::vector<float> scales(params->scale->data,
                              params->scale->data + params->scale->size);
    std::vector<int32_t> zero_points(
        params->zero_point->data,
        params->zero_point->data + params->zero_point->size);

    tim::vx::QuantType qtype = tim::vx::QuantType::ASYMMETRIC;
    if (scales.size() > 1) {
      qtype = tim::vx::QuantType::SYMMETRIC_PER_CHANNEL;
      int32_t channel_dim = params->quantized_dimension;
      int32_t vx_channel_dim =
          vx::delegate::utils::ConvertAxis(channel_dim, dims.size());
      quantization =
          tim::vx::Quantization(qtype, vx_channel_dim, scales, zero_points);
    } else {
      quantization = tim::vx::Quantization(qtype, scales[0], zero_points[0]);
    }

    return tim::vx::TensorSpec(datatype, whcn_shape, attr, quantization);
  }

  return tim::vx::TensorSpec(datatype, whcn_shape, attr);
}

bool TransposeTensorData(const TfLiteTensor* tensor,
                         const std::vector<uint32_t>& perm,
                         std::vector<uint8_t>& data_out) {
  const uint8_t* tensor_data =
      reinterpret_cast<const uint8_t*>(tensor->data.raw_const);
  if (!tensor_data) {
    return false;
  }

  tflite::TransposeParams params;
  std::vector<int32_t> output_shape;

  params.perm_count = perm.size();
  output_shape.resize(perm.size());
  for (size_t i = 0; i < perm.size(); i++) {
    params.perm[i] = perm[i];
    output_shape[i] = tensor->dims->data[perm[i]];
  }
  data_out.resize(tensor->bytes);
  switch (tensor->type) {
    case kTfLiteFloat32:
    case kTfLiteInt32:
      tflite::reference_ops::Transpose(
          params,
          tflite::GetTensorShape(tensor),
          tflite::GetTensorData<int32_t>(tensor),
          tflite::RuntimeShape(
              static_cast<int>(output_shape.size()),
              reinterpret_cast<const int32_t*>(output_shape.data())),
          reinterpret_cast<int32_t*>(data_out.data()));
      break;
    case kTfLiteInt16:
    case kTfLiteFloat16:
      tflite::reference_ops::Transpose(
          params,
          tflite::GetTensorShape(tensor),
          tflite::GetTensorData<int16_t>(tensor),
          tflite::RuntimeShape(
              static_cast<int>(output_shape.size()),
              reinterpret_cast<const int32_t*>(output_shape.data())),
          reinterpret_cast<int16_t*>(data_out.data()));
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
      tflite::reference_ops::Transpose(
          params,
          tflite::GetTensorShape(tensor),
          tflite::GetTensorData<int8_t>(tensor),
          tflite::RuntimeShape(
              static_cast<int>(output_shape.size()),
              reinterpret_cast<const int32_t*>(output_shape.data())),
          reinterpret_cast<int8_t*>(data_out.data()));
      break;
    default:
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Unsupported type: %d", tensor->type);
      return false;
  }

  return true;
}

std::shared_ptr<tim::vx::Tensor> CreateTensor(
    std::shared_ptr<tim::vx::Graph>& graph,
    const TfLiteTensor* tensor,
    const tim::vx::TensorAttribute& attr,
    const std::vector<uint32_t>& perm) {
  const uint8_t* tensor_data = nullptr;
  tim::vx::TensorSpec spec = CreateTensorSpec(tensor, perm, attr);
  switch (attr) {
    case tim::vx::TensorAttribute::INPUT:
    case tim::vx::TensorAttribute::OUTPUT:
    case tim::vx::TensorAttribute::VARIABLE:
      break;
    case tim::vx::TensorAttribute::CONSTANT:
      tensor_data = reinterpret_cast<const uint8_t*>(tensor->data.raw_const);
      if (perm.size() > 0) {
        std::vector<uint8_t> data_transposed;
        if (TransposeTensorData(tensor, perm, data_transposed)) {
          return graph->CreateTensor(
              spec, reinterpret_cast<const void*>(data_transposed.data()));
        }
      }
      break;
    case tim::vx::TensorAttribute::TRANSIENT:
      break;
    default:
      break;
  }
  return graph->CreateTensor(spec, reinterpret_cast<const void*>(tensor_data));
}

#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
/**
 * Create a TIM-VX tensor backed by a DMA-BUF for zero-copy I/O.
 */
std::shared_ptr<tim::vx::Tensor> CreateTensorWithDmaBuf(
    std::shared_ptr<tim::vx::Graph>& graph,
    const TfLiteTensor* tensor,
    const tim::vx::TensorAttribute& attr,
    const std::vector<uint32_t>& perm,
    int dmabuf_fd) {
  tim::vx::TensorSpec spec = CreateTensorSpec(tensor, perm, attr);
  tim::vx::DmaBufferDesc dma_desc;
  dma_desc.fd = static_cast<int64_t>(dmabuf_fd);
  return graph->CreateTensor(spec, dma_desc);
}
#endif

std::vector<std::shared_ptr<tim::vx::Tensor>> MapIndexesToTensors(
    const std::map<int32_t, std::shared_ptr<tim::vx::Tensor>>& tensors,
    const std::vector<int>& indexes) {
  std::vector<std::shared_ptr<tim::vx::Tensor>> out_tensors;
  for (const auto& index : indexes) {
    if (index != -1) {
      auto pos = tensors.find(index);
      if (pos != tensors.end())
        out_tensors.push_back(pos->second);
    }
  }
  return out_tensors;
}

}  // namespace

// Live delegate instances — used by VxDelegateGetInstance() so that
// callers who only have the TfLiteExternalDelegate wrapper (not the inner
// DerivedDelegateData*) can still reach the DMA-BUF API. The thread-local
// last-created pointer keeps concurrent per-worker delegate creation
// unambiguous: each worker resolves the delegate it created on its own
// thread. The registry guards against ever returning a deleted instance
// (a sibling thread may have destroyed it). Mirrors the Neutron
// delegate's per-instance registry semantics (EDGEAI-1188 / EDGEAI-1435).
static std::mutex g_vx_instances_mutex;
static std::unordered_set<vx::delegate::DerivedDelegateData*> g_vx_instances;
static vx::delegate::DerivedDelegateData* g_vx_last = nullptr;
static thread_local vx::delegate::DerivedDelegateData* tl_vx_last = nullptr;

extern "C" {
TfLiteDelegate* VxDelegateGetInstance(void) {
  std::lock_guard<std::mutex> lock(g_vx_instances_mutex);
  if (tl_vx_last != nullptr &&
      g_vx_instances.find(tl_vx_last) != g_vx_instances.end()) {
    return reinterpret_cast<TfLiteDelegate*>(tl_vx_last);
  }
  if (g_vx_last != nullptr &&
      g_vx_instances.find(g_vx_last) != g_vx_instances.end()) {
    return reinterpret_cast<TfLiteDelegate*>(g_vx_last);
  }
  if (g_vx_instances.size() == 1) {
    return reinterpret_cast<TfLiteDelegate*>(*g_vx_instances.begin());
  }
  return nullptr;
}
}  // extern "C"

namespace vx {
namespace delegate {
VxDelegateOptions VxDelegateOptionsDefault() {
  VxDelegateOptions options = {0};
  options.enable_dmabuf = true;  // Enable by default
  options.dma_heap_path = nullptr;  // Auto-detect
  return options;
}

TfLiteDelegate* VxDelegate(const VxDelegateOptions* options) {
  return vx::delegate::Delegate::Create(options);
}

TfLiteDelegate* VxDelegateCreate(const VxDelegateOptions* options) {
  return VxDelegate(options);
}

void VxDelegateDelete(TfLiteDelegate* delegate) {
  if (delegate == nullptr) return;
  auto derivedDelegate = reinterpret_cast<DerivedDelegateData*>(delegate);
  {
    std::lock_guard<std::mutex> lock(g_vx_instances_mutex);
    g_vx_instances.erase(derivedDelegate);
    if (g_vx_last == derivedDelegate) g_vx_last = nullptr;
    if (tl_vx_last == derivedDelegate) tl_vx_last = nullptr;
  }
  delete derivedDelegate;
  delegate = nullptr;
}

bool Delegate::SupportedOp(TfLiteContext* context,
                           TfLiteNode* node,
                           const TfLiteRegistration* registration) {
  if (registration->custom_name != nullptr) {
    const auto& supported_custom_ops = vx::op_map::SupportedBuiltinCustomOps();
    const auto& it = supported_custom_ops.find(registration->custom_name);
    if (supported_custom_ops.end() != it) {
      return it->second->IsSupported(context, node, registration);
    }
  }

  const auto& supported_builtins = vx::op_map::SupportedBuiltinOps();
  const auto& it = supported_builtins.find(
      static_cast<TfLiteBuiltinOperator>(registration->builtin_code));
  if (supported_builtins.end() != it) {
    return it->second->IsSupported(context, node, registration);
  }

  TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "Fallback unsupported op %d to TfLite", registration->builtin_code);

  return false;
}

TfLiteDelegate* Delegate::Create(const VxDelegateOptions* options) {
  DerivedDelegateData* delegate = new DerivedDelegateData();
  // Initialize only the POD TfLiteDelegate parent struct
  // DO NOT memset the entire DerivedDelegateData - it contains C++ objects
  // (std::string, std::shared_ptr, std::map) that have non-trivial constructors
  std::memset(&delegate->parent, 0, sizeof(TfLiteDelegate));

  delegate->parent.flags = kTfLiteDelegateFlagsAllowDynamicTensors | kTfLiteDelegateFlagsRequirePropagatedShapes;
  delegate->parent.Prepare = &PrepareDelegate;
  delegate->parent.CopyFromBufferHandle = &CopyFromBufferHandle;
  delegate->parent.FreeBufferHandle = &FreeBufferHandle;

  delegate->device_id = options->device_id;
  delegate->allow_cache_mode = options->allowed_cache_mode;
  delegate->needs_invalidation = false;
  if(delegate->allow_cache_mode){
    delegate->cache_path = options->cache_file_path;
  }
  
  // DMA-BUF support
  delegate->enable_dmabuf = options->enable_dmabuf;
  if (options->dma_heap_path) {
    delegate->dma_heap_path = options->dma_heap_path;
  }
  
  // Initialize the shared DmaBufManager
  if (delegate->enable_dmabuf) {
    delegate->dmabuf_manager = std::make_shared<DmaBufManager>();
    const char* heap_path = delegate->dma_heap_path.empty() ? 
                            nullptr : delegate->dma_heap_path.c_str();
    if (delegate->dmabuf_manager->Initialize(heap_path)) {
      TFLITE_LOG(TFLITE_LOG_INFO, "DMA-BUF support initialized");
    } else {
      TFLITE_LOG(TFLITE_LOG_WARNING, "DMA-BUF initialization failed, falling back to copy mode");
      delegate->enable_dmabuf = false;
    }
  }
  
  {
    std::lock_guard<std::mutex> lock(g_vx_instances_mutex);
    g_vx_instances.insert(delegate);
    g_vx_last = delegate;
    tl_vx_last = delegate;
  }
  return reinterpret_cast<TfLiteDelegate*>(delegate);
}

void Delegate::CreateCacheOp(const OpData& op_data) {
    operations_.resize(1);
    auto& operation = operations_[0];

    operation.custom_name = "vsi-npu";
    std::copy(op_data.subgraph_inputs.begin(),
              op_data.subgraph_inputs.end(),
              std::back_inserter(operation.inputs));
    std::copy(op_data.subgraph_outputs.begin(),
              op_data.subgraph_outputs.end(),
              std::back_inserter(operation.outputs));

    operation.builtin_data.reserve(nbg_size_ + sizeof(TfLiteVsiNpuParams));
    TfLiteVsiNpuParams* nbg_param =
        reinterpret_cast<TfLiteVsiNpuParams*>(operation.builtin_data.data());
    nbg_param->length = nbg_size_;
    nbg_param->input_count = op_data.subgraph_inputs.size();
    nbg_param->output_cout = op_data.subgraph_outputs.size();
    nbg_param->binary = reinterpret_cast<char*>(operation.builtin_data.data()) +
                        sizeof(TfLiteVsiNpuParams);
    fs_.read(nbg_param->binary,nbg_size_);
}

std::unique_ptr<vx::delegate::OpData> Delegate::Init(
    TfLiteContext* context, const TfLiteDelegateParams* params) {
  TFLITE_LOG(TFLITE_LOG_INFO, "vx_delegate Delegate::Init");

  nbg_size_ = 0;
  auto derivedDelegate = reinterpret_cast<DerivedDelegateData*>(params->delegate);
  if (derivedDelegate->allow_cache_mode){
    fs_.open(derivedDelegate->cache_path.c_str(), std::ios::in | std::ios::ate);
    is_cache_present_ = fs_ ? true : false;
    nbg_size_ = fs_.tellg();
    fs_.close();
  }

#ifdef MULTI_DEVICE_FEATURE_MODE
    devices_ = tim::vx::platform::NativeDevice::Enumerate();
    device_id_ = derivedDelegate->device_id;
#endif

  compiled_ = false;

  std::unique_ptr<vx::delegate::OpData> op_data(new OpData());
  // Get the list of input and output tensors. This isn't for a single op, it's
  // for a subgraph.
  tflite::TfLiteIntArrayView input_tensors(params->input_tensors);
  for (int input_tensor_idx : input_tensors) {
    const auto& tensor = context->tensors[input_tensor_idx];
    if (tensor.allocation_type != kTfLiteMmapRo) {
      op_data->subgraph_inputs.push_back(input_tensor_idx);
    }
  }

  tflite::TfLiteIntArrayView output_tensors(params->output_tensors);
  std::copy(output_tensors.begin(),
            output_tensors.end(),
            std::back_inserter(op_data->subgraph_outputs));

  if (is_cache_present_ && is_cache_present_.value()) {
    fs_.open(derivedDelegate->cache_path.c_str(), std::ios::in);
    Delegate::CreateCacheOp(*op_data);
    fs_.close();
  } else {
    if (is_cache_present_ && !is_cache_present_.value()) {
      fs_.open(derivedDelegate->cache_path.c_str(),
               std::ios::in | std::ios::out | std::ios::app | std::ios::binary);
    }
    const auto& supported_customs = vx::op_map::SupportedBuiltinCustomOps();
    const auto& supported_builtins = vx::op_map::SupportedBuiltinOps();
    operations_.resize(params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      TfLiteNode* node;
      TfLiteRegistration* reg;
      int node_idx = params->nodes_to_replace->data[i];
      context->GetNodeAndRegistration(context, node_idx, &node, &reg);
      tflite::TfLiteIntArrayView inputs(node->inputs);
      tflite::TfLiteIntArrayView outputs(node->outputs);

      auto& operation = operations_[i];

      if (reg->custom_name) {
        operation.custom_name = reg->custom_name;
      }
      operation.builtin_code = reg->builtin_code;
      bool isbuiltinOp = operation.custom_name.empty();
      std::copy(
          inputs.begin(), inputs.end(), std::back_inserter(operation.inputs));
      std::copy(outputs.begin(),
                outputs.end(),
                std::back_inserter(operation.outputs));

      std::vector<int> states;
      if ((isbuiltinOp &&
           supported_builtins.at(reg->builtin_code)
               ->GetStateTensorIndexes(context, node, reg, states)) ||
          (!isbuiltinOp &&
           supported_customs.at(operation.custom_name)
               ->GetStateTensorIndexes(context, node, reg, states))) {
        std::copy(
            states.begin(), states.end(), std::back_inserter(operation.states));

        // record state tensor index
        std::copy(states.begin(),
                  states.end(),
                  std::back_inserter(op_data->subgraph_states));
      }

      if (!isbuiltinOp && node->user_data) {
        operation.builtin_data.resize(
            supported_customs.at(operation.custom_name)->GetParamSize());
        memcpy(operation.builtin_data.data(),
               node->user_data,
               operation.builtin_data.size());
      } else if (isbuiltinOp && node->builtin_data) {
        operation.builtin_data.resize(
            supported_builtins.at(reg->builtin_code)->GetParamSize());
        memcpy(operation.builtin_data.data(),
               node->builtin_data,
               operation.builtin_data.size());
      } else {
        continue;
      }
    }
  }
  delegate::Delegate::subgraph_outputs_ = op_data->subgraph_outputs;
  return op_data;
}

TfLiteStatus Delegate::Prepare(const OpData& op_data,
                               TfLiteContext* context,
                               TfLiteNode* node) {
  TFLITE_LOG(TFLITE_LOG_INFO, "Delegate::Prepare node: %p", node->user_data);
  return kTfLiteOk;
}

TfLiteStatus Delegate::Invoke(const OpData& op_data,
                              TfLiteContext* context,
                              TfLiteNode* node) {
  TFLITE_LOG(TFLITE_LOG_INFO, "Delegate::Invoke node: %p, compiled_=%d, dmabuf_enabled_=%d",
             node->user_data, compiled_, dmabuf_enabled_);

  // Check for graph invalidation request (e.g., after caps renegotiation)
  if (CheckAndClearInvalidation()) {
    TFLITE_LOG(TFLITE_LOG_INFO, "Graph invalidated, will recompile");
  }

#ifdef NODE_TRACE_DB_MODE
  std::vector<vx::delegate::TfliteNodeIDPair> tflite_node_id_map;
#endif

  if (!compiled_) {
    TFLITE_LOG(TFLITE_LOG_INFO, "Compiling graph, dmabuf_manager_=%p, dmabuf_enabled_=%d",
               dmabuf_manager_.get(), dmabuf_enabled_);
    // TODO(bo): Handling multi-thread use case
    context_ = tim::vx::Context::Create();
    graph_ = context_->CreateGraph();

    // Create input tensors
    // When CameraAdaptor is configured for a tensor:
    // - Create the model's expected tensor as TRANSIENT (Slice will write to it)
    // - CameraAdaptor creates the actual INPUT tensor with camera format
    for (int tensor_idx : op_data.subgraph_inputs) {
      if (-1 != tensor_idx && tensors_[tensor_idx].get() == nullptr) {
        const auto tensor = &(context->tensors[tensor_idx]);

        // Check if CameraAdaptor is configured for this tensor
        bool has_camera_adaptor = false;
        if (parent_delegate_) {
          auto config_it = parent_delegate_->camera_adaptor_configs.find(tensor_idx);
          if (config_it != parent_delegate_->camera_adaptor_configs.end()) {
            edgefirst::camera_adaptor::CameraAdaptor adaptor(config_it->second);
            has_camera_adaptor = adaptor.RequiresConversion();
          }
        }

        // If CameraAdaptor is configured, create as TRANSIENT (Slice outputs to it)
        if (has_camera_adaptor) {
          tensors_[tensor_idx] =
              CreateTensor(graph_, tensor, tim::vx::TensorAttribute::TRANSIENT, {});
          TFLITE_LOG(TFLITE_LOG_INFO,
                     "Created TRANSIENT tensor %d for CameraAdaptor output", tensor_idx);
          continue;
        }

#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
        // Check if this input tensor has a registered DMA-BUF (no CameraAdaptor case)
        TFLITE_LOG(TFLITE_LOG_INFO, "Checking input tensor %d: dmabuf_manager_=%p, dmabuf_enabled_=%d",
                   tensor_idx, dmabuf_manager_.get(), dmabuf_enabled_);
        if (dmabuf_manager_ && dmabuf_enabled_) {
          TfLiteBufferHandle dmabuf_handle = dmabuf_manager_->FindByTensorIndex(tensor_idx);
          TFLITE_LOG(TFLITE_LOG_INFO, "Input tensor %d: dmabuf_handle=%d", tensor_idx, dmabuf_handle);
          if (dmabuf_handle != kTfLiteNullBufferHandle) {
            int fd = dmabuf_manager_->GetFd(dmabuf_handle);
            TFLITE_LOG(TFLITE_LOG_INFO, "Input tensor %d: fd=%d", tensor_idx, fd);
            if (fd >= 0) {
              tensors_[tensor_idx] = CreateTensorWithDmaBuf(
                  graph_, tensor, tim::vx::TensorAttribute::INPUT, {}, fd);
              TFLITE_LOG_PROD(TFLITE_LOG_INFO,
                         "Created dmabuf-backed input tensor %d (fd=%d)", tensor_idx, fd);
              continue;
            }
          }
        }
#endif
        tensors_[tensor_idx] =
            CreateTensor(graph_, tensor, tim::vx::TensorAttribute::INPUT, {});
      }
    }

    // Inject camera adaptor preprocessing if configured
    // This creates the actual INPUT tensor and connects it to the TRANSIENT tensor via Slice
    if (parent_delegate_) {
      for (int tensor_idx : op_data.subgraph_inputs) {
        auto config_it = parent_delegate_->camera_adaptor_configs.find(tensor_idx);
        if (config_it != parent_delegate_->camera_adaptor_configs.end()) {
          edgefirst::camera_adaptor::CameraAdaptor adaptor(config_it->second);

          if (adaptor.RequiresConversion()) {
            // Get DMA-BUF fd if available
            int dmabuf_fd = -1;
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
            if (dmabuf_manager_ && dmabuf_enabled_) {
              TfLiteBufferHandle dmabuf_handle = dmabuf_manager_->FindByTensorIndex(tensor_idx);
              if (dmabuf_handle != kTfLiteNullBufferHandle) {
                dmabuf_fd = dmabuf_manager_->GetFd(dmabuf_handle);
                TFLITE_LOG(TFLITE_LOG_INFO,
                           "CameraAdaptor: tensor %d has DMA-BUF fd=%d", tensor_idx, dmabuf_fd);
              }
            }
#endif
            // tensors_[tensor_idx] is now TRANSIENT - CameraAdaptor will create INPUT
            // and Slice from INPUT -> TRANSIENT
            auto result = adaptor.InjectPreprocessing(graph_, tensors_[tensor_idx], dmabuf_fd);

            if (result.success) {
              // Check if this is a passthrough (no conversion needed)
              // In passthrough, camera_input == model_input == original_input
              if (result.camera_input == tensors_[tensor_idx] && 
                  result.model_input == tensors_[tensor_idx]) {
                // Passthrough: no actual conversion needed
                // The TRANSIENT tensor has no producer, must recreate as INPUT
                TFLITE_LOG(TFLITE_LOG_INFO,
                           "CameraAdaptor: passthrough for tensor %d (no conversion needed)",
                           tensor_idx);
                
                const auto tensor = &(context->tensors[tensor_idx]);
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
                if (dmabuf_fd >= 0) {
                  tensors_[tensor_idx] = CreateTensorWithDmaBuf(
                      graph_, tensor, tim::vx::TensorAttribute::INPUT, {}, dmabuf_fd);
                  TFLITE_LOG(TFLITE_LOG_INFO,
                             "CameraAdaptor passthrough: recreated tensor %d as INPUT (DMA-BUF fd=%d)",
                             tensor_idx, dmabuf_fd);
                } else {
                  tensors_[tensor_idx] =
                      CreateTensor(graph_, tensor, tim::vx::TensorAttribute::INPUT, {});
                }
#else
                tensors_[tensor_idx] =
                    CreateTensor(graph_, tensor, tim::vx::TensorAttribute::INPUT, {});
#endif
                // Remove from camera adaptor configs since no conversion is used
                parent_delegate_->camera_adaptor_configs.erase(tensor_idx);
              } else {
                // Actual conversion: store the camera_input tensor for DMA-BUF tracking
                // Model operations should continue to use tensors_[tensor_idx] (3-channel TRANSIENT)
                // camera_input is stored separately for data input
                camera_input_tensors_[tensor_idx] = result.camera_input;
                TFLITE_LOG(TFLITE_LOG_INFO,
                           "CameraAdaptor: injected %s->%s preprocessing for tensor %d%s",
                           edgefirst::camera_adaptor::ColorSpaceToString(adaptor.adaptor()),
                           edgefirst::camera_adaptor::ColorSpaceToString(adaptor.model_format()),
                           tensor_idx,
                           dmabuf_fd >= 0 ? " (DMA-BUF backed)" : "");
              }
            } else {
              // CameraAdaptor injection failed - recover by recreating tensor as INPUT
              // The TRANSIENT tensor has no producer, so we must replace it
              TFLITE_LOG(TFLITE_LOG_WARNING,
                         "CameraAdaptor: failed for tensor %d: %s - falling back to direct input",
                         tensor_idx, result.error_message.c_str());

              const auto tensor = &(context->tensors[tensor_idx]);
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
              if (dmabuf_fd >= 0) {
                // Recreate as INPUT with DMA-BUF backing
                tensors_[tensor_idx] = CreateTensorWithDmaBuf(
                    graph_, tensor, tim::vx::TensorAttribute::INPUT, {}, dmabuf_fd);
                TFLITE_LOG(TFLITE_LOG_INFO,
                           "CameraAdaptor fallback: recreated tensor %d as INPUT (DMA-BUF fd=%d)",
                           tensor_idx, dmabuf_fd);
              } else {
                // Recreate as regular INPUT
                tensors_[tensor_idx] =
                    CreateTensor(graph_, tensor, tim::vx::TensorAttribute::INPUT, {});
                TFLITE_LOG(TFLITE_LOG_INFO,
                           "CameraAdaptor fallback: recreated tensor %d as INPUT",
                           tensor_idx);
              }
#else
              // Recreate as regular INPUT
              tensors_[tensor_idx] =
                  CreateTensor(graph_, tensor, tim::vx::TensorAttribute::INPUT, {});
              TFLITE_LOG(TFLITE_LOG_INFO,
                         "CameraAdaptor fallback: recreated tensor %d as INPUT",
                         tensor_idx);
#endif
              // Remove from camera adaptor configs to prevent future attempts
              parent_delegate_->camera_adaptor_configs.erase(tensor_idx);
            }
          }
        }
      }
    }

    // Create output tensors
    for (int tensor_idx : op_data.subgraph_outputs) {
      if (-1 != tensor_idx && tensors_[tensor_idx].get() == nullptr) {
        const auto tensor = &(context->tensors[tensor_idx]);
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
        // Check if this output tensor has a registered DMA-BUF
        TFLITE_LOG(TFLITE_LOG_INFO, "Checking output tensor %d", tensor_idx);
        if (dmabuf_manager_ && dmabuf_enabled_) {
          TfLiteBufferHandle dmabuf_handle = dmabuf_manager_->FindByTensorIndex(tensor_idx);
          TFLITE_LOG(TFLITE_LOG_INFO, "Output tensor %d: dmabuf_handle=%d", tensor_idx, dmabuf_handle);
          if (dmabuf_handle != kTfLiteNullBufferHandle) {
            int fd = dmabuf_manager_->GetFd(dmabuf_handle);
            TFLITE_LOG(TFLITE_LOG_INFO, "Output tensor %d: fd=%d", tensor_idx, fd);
            if (fd >= 0) {
              tensors_[tensor_idx] = CreateTensorWithDmaBuf(
                  graph_, tensor, tim::vx::TensorAttribute::OUTPUT, {}, fd);
              TFLITE_LOG_PROD(TFLITE_LOG_INFO,
                         "Created dmabuf-backed output tensor %d (fd=%d)", tensor_idx, fd);
              continue;
            }
          }
        }
#endif
        tensors_[tensor_idx] =
            CreateTensor(graph_, tensor, tim::vx::TensorAttribute::OUTPUT, {});
      }
    }

    // create op
    for (const auto& op_info : operations_) {
      op_info_ = op_info;
      auto& builtin_code = op_info.builtin_code;
      auto& custom_name = op_info.custom_name;
      auto inputs = op_info.inputs;
      auto outputs = op_info.outputs;
      auto& states = op_info.states;
      auto& builtin_data = op_info.builtin_data;

#ifdef NODE_TRACE_DB_MODE
      vx::delegate::TfliteNodeIDPair tflite_node_id_pair;
      std::vector<std::shared_ptr<tim::vx::Operation>> before_op_vector;
      std::vector<std::shared_ptr<tim::vx::Operation>> after_op_vector;
#endif

      std::vector<int> inputs_outputs;
      std::copy(
          inputs.begin(), inputs.end(), std::back_inserter(inputs_outputs));
      std::copy(
          outputs.begin(), outputs.end(), std::back_inserter(inputs_outputs));

#ifdef NODE_TRACE_DB_MODE
        tflite_node_id_pair.builtin_code = builtin_code;
        std::copy(
            inputs.begin(), inputs.end(), std::back_inserter(tflite_node_id_pair.inputs));
        std::copy(
            outputs.begin(), outputs.end(), std::back_inserter(tflite_node_id_pair.outputs));
        std::copy(
            this->GetGraph()->OpVector().begin(), this->GetGraph()->OpVector().end(), std::back_inserter(before_op_vector));
        tflite_node_id_map.push_back(tflite_node_id_pair);
#endif

      for (size_t port_idx = 0; port_idx < inputs_outputs.size(); port_idx++) {
        int tensor_idx = inputs_outputs[port_idx];
        if (-1 != tensor_idx && tensors_.find(tensor_idx) == tensors_.end()) {
          std::vector<uint32_t> perm;
          auto tensor = &(context->tensors[tensor_idx]);
          tim::vx::TensorAttribute attr = tim::vx::TensorAttribute::TRANSIENT;
          if (IsConstTensor(tensor)) {
            attr = tim::vx::TensorAttribute::CONSTANT;
          } else if (IsVariableTensor(tensor)) {
            attr = tim::vx::TensorAttribute::VARIABLE;
          } else {
            attr = tim::vx::TensorAttribute::TRANSIENT;
          }
          tensors_[tensor_idx] = CreateTensor(graph_, tensor, attr, perm);
        }
        else if (-1 == tensor_idx && (builtin_code == 44 || builtin_code == 52) ){
          // -1 means placeholder for optional inputs: for example LSTM
          if (tensors_.find(placeholder_tensor_idx_) != tensors_.end()) {
            // Assert(false);
          }
          tensors_[placeholder_tensor_idx_] = graph_->CreateTensorPlaceHolder();

          if (port_idx < inputs.size()) {
            inputs[port_idx] = placeholder_tensor_idx_;
          } else {
            outputs[port_idx - inputs.size()] = placeholder_tensor_idx_;
          }

          placeholder_tensor_idx_ --;
        }
      }

      // create state output as graph output
      for (auto tensor_idx : states) {
        // TODO{Sven}: use map.find() instead
        if (-1 != tensor_idx && state_tensors_[tensor_idx].get() == nullptr) {
          const auto tensor = &(context->tensors[tensor_idx]);
          state_tensors_[tensor_idx] = CreateTensor(
              graph_, tensor, tim::vx::TensorAttribute::OUTPUT, {});
        }
      }

      std::vector<std::shared_ptr<tim::vx::Tensor>> inputs_tensors =
          MapIndexesToTensors(tensors_, inputs);
      std::vector<std::shared_ptr<tim::vx::Tensor>> outputs_tensors =
          MapIndexesToTensors(tensors_, outputs);
      std::vector<std::shared_ptr<tim::vx::Tensor>> states_tensors =
          MapIndexesToTensors(state_tensors_, states);

      if (!custom_name.empty()) {
        vx::op_map::SupportedBuiltinCustomOps()
            .at(custom_name)
            ->MapOp(this,
                    inputs_tensors,
                    outputs_tensors,
                    states_tensors,
                    builtin_data.data());
      } else {
        vx::op_map::SupportedBuiltinOps()
            .at(builtin_code)
            ->MapOp(this,
                    inputs_tensors,
                    outputs_tensors,
                    states_tensors,
                    builtin_data.data());
      }
#ifdef NODE_TRACE_DB_MODE
        std::copy(
          this->GetGraph()->OpVector().begin(), this->GetGraph()->OpVector().end(), std::back_inserter(after_op_vector));
        vx::delegate::utils::MapTfliteNodeToTimVxNode(before_op_vector, after_op_vector, tflite_node_id_map);
#endif
    }
#ifdef NODE_TRACE_DB_MODE
      vx::delegate::utils::GenerateVxNodeTraceDb(tflite_node_id_map);
#endif

    TFLITE_LOG(TFLITE_LOG_INFO, "Verifying graph");
    // Do normalization op fusion before layout inference
    tim::transform::MeanStdDevNormalization(graph_);
    // Do layout inference and get a new graph(first) and a tensor map(second).
    layout_infered_ = tim::transform::LayoutInference(graph_, context_);

#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
    // Update dmabuf tensor references after layout inference and SWAP TO DMABUF
    TFLITE_LOG(TFLITE_LOG_INFO, "Post layout inference dmabuf swap: dmabuf_manager_=%p, dmabuf_enabled_=%d",
                    dmabuf_manager_.get(), dmabuf_enabled_);
    if (dmabuf_manager_ && dmabuf_enabled_) {
      // For dmabuf-backed inputs, DO NOT swap - we need to copy from dmabuf to inferred tensor on each invoke
      // SwapHandle would swap the dmabuf handle to the inferred tensor, but then we couldn't access
      // the dmabuf memory via the original tensor anymore.
      // True zero-copy would require TIM-VX layout inference to preserve dmabuf handles.
      for (int tensor_idx : op_data.subgraph_inputs) {
        TfLiteBufferHandle handle = dmabuf_manager_->FindByTensorIndex(tensor_idx);
        TFLITE_LOG(TFLITE_LOG_INFO, "Input tensor %d: handle=%d (no swap, will copy)", tensor_idx, handle);
        // No SwapHandle - we'll copy from dmabuf to inferred tensor on each invoke
      }
      for (int tensor_idx : op_data.subgraph_outputs) {
        TfLiteBufferHandle handle = dmabuf_manager_->FindByTensorIndex(tensor_idx);
        TFLITE_LOG(TFLITE_LOG_INFO, "Output tensor %d: handle=%d (no swap, will copy)", tensor_idx, handle);
        // No SwapHandle - output swap fails because inferred tensor wasn't created from handle
        // We'll copy from inferred tensor to dmabuf after each invoke
      }
    }
#endif

#ifdef MULTI_DEVICE_FEATURE_MODE
      executor_ = std::make_shared<tim::vx::platform::NativeExecutor>(devices_[device_id_]);
      executable_ = tim::vx::platform::Compile(layout_infered_.first, executor_);
#else
    if(is_cache_present_ && !is_cache_present_.value()){
      nbg_size_ = -1;
      compiled_ = layout_infered_.first->CompileToBinary(nullptr, &nbg_size_);
      if (!compiled_) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "compile to binary failed");
        return kTfLiteDelegateError;
        }
        std::vector<uint8_t> nbg_buf(nbg_size_);
        compiled_ = layout_infered_.first->CompileToBinary(nbg_buf.data(), &nbg_size_);
        if (!compiled_) {
          TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "compile to binary failed");
          return kTfLiteDelegateError;
        }
        fs_.write(reinterpret_cast<const char*>(nbg_buf.data()),nbg_size_);
        fs_.close();
    } else {
      compiled_ = layout_infered_.first->Compile();
      if (!compiled_) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to verify graph");
        return kTfLiteDelegateError;
      }
      TFLITE_LOG(TFLITE_LOG_INFO, "Verified graph");
    }
#endif
  }

  // TODO(derekjchow): Return error if compilation failed.
  for (int tensor_idx : op_data.subgraph_inputs) {
    const TfLiteTensor& tf_tensor = context->tensors[tensor_idx];
    auto src_input_tensor = tensors_[tensor_idx];
    if (!src_input_tensor.get()) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to copy input tensor!");
      return kTfLiteDelegateError;
    }

#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
    // Check if this input is dmabuf-backed
    if (dmabuf_manager_ && dmabuf_enabled_) {
      TfLiteBufferHandle dmabuf_handle = dmabuf_manager_->FindByTensorIndex(tensor_idx);
      if (dmabuf_handle != kTfLiteNullBufferHandle) {
        // Check if CameraAdaptor is configured for this tensor
        // When CameraAdaptor is used, the graph has:
        //   camera_input_tensors_[idx] (INPUT w/ dmabuf) -> [Slice] -> [Sub] -> tensors_[idx] (TRANSIENT)
        // Data flows through the pipeline automatically; no manual copy needed
        auto camera_it = camera_input_tensors_.find(tensor_idx);
        if (camera_it != camera_input_tensors_.end()) {
          auto camera_input = camera_it->second;
          auto infered_camera_input = layout_infered_.second[camera_input];
          if (infered_camera_input && infered_camera_input->HasDmaBuf()) {
            if (dmabuf_zerocopy_logged_.insert(tensor_idx).second) {
              TFLITE_LOG_PROD(TFLITE_LOG_INFO,
                         "CameraAdaptor input %d: zero-copy pipeline (fd=%lld)",
                         tensor_idx, (long long)infered_camera_input->GetDmaBufFd());
            }
          } else {
            if (dmabuf_zerocopy_logged_.insert(tensor_idx).second) {
              TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                         "CameraAdaptor input %d: DMABUF LOST after layout inference! (infered=%p, hasDmaBuf=%d)",
                         tensor_idx, infered_camera_input.get(),
                         infered_camera_input ? infered_camera_input->HasDmaBuf() : -1);
            }
          }
          continue;
        }

        // Non-CameraAdaptor dmabuf path
        auto infered_input_tensor = layout_infered_.second[src_input_tensor];
        if (infered_input_tensor) {
          // Check if inferred tensor preserved the dmabuf (true zero-copy path)
          if (infered_input_tensor->HasDmaBuf()) {
            // True zero-copy: NPU reads directly from dmabuf, no copy needed
            if (dmabuf_zerocopy_logged_.insert(tensor_idx).second) {
              TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Dmabuf input %d: zero-copy (fd=%lld)",
                         tensor_idx, (long long)infered_input_tensor->GetDmaBufFd());
            }
          } else {
            // Fallback: inferred tensor doesn't have dmabuf, need to copy
            if (dmabuf_zerocopy_logged_.insert(tensor_idx).second) {
              TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                         "Dmabuf input %d: HIDDEN COPY - layout inference lost dmabuf!",
                         tensor_idx);
            }
            void* dmabuf_data = src_input_tensor->map(true /* invalidate cache */);
            if (dmabuf_data) {
              infered_input_tensor->CopyDataToTensor(dmabuf_data);
              src_input_tensor->unmap();
            } else {
              TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to map dmabuf for input %d", tensor_idx);
            }
          }
        }
        continue;
      }
    }
#endif

    TFLITE_LOG(TFLITE_LOG_INFO, "Copying input %d: %s", tensor_idx, tf_tensor.name);
    const void* tensor_data =
        reinterpret_cast<const void*>(tf_tensor.data.raw_const);
    // TODO(derekjchow): Check result
    auto infered_input_tensor = layout_infered_.second[src_input_tensor];
    if (infered_input_tensor) {
      infered_input_tensor->CopyDataToTensor(const_cast<void*>(tensor_data));
    } else {
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                      "tensor in source graph removed before do layout "
                      "inference - if zero sized tensor involved");
    }
#ifdef MULTI_DEVICE_FEATURE_MODE
    uint32_t tensor_index = 0;
    auto input_spec = infered_input_tensor->GetSpec();
    inputs_.push_back(executable_->AllocateTensor(input_spec));
    executable_->SetInput(inputs_[tensor_index]);
    inputs_[tensor_index]->CopyDataToTensor(tensor_data,
                                            input_spec.GetByteSize());
#endif
  }

#ifdef MULTI_DEVICE_FEATURE_MODE
  uint32_t tensor_index = 0;
  for (int tensor_idx : op_data.subgraph_outputs) {
    TfLiteTensor& tf_tensor = context->tensors[tensor_idx];
    auto src_output_tensor = tensors_[tensor_idx];

    outputs_.push_back(
        executable_->AllocateTensor(src_output_tensor->GetSpec()));
    executable_->SetOutput(outputs_[tensor_index]);
  }
#endif

#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
  // Buffer cycling: swap to active buffers if changed since last invoke
  if (dmabuf_manager_ && dmabuf_enabled_ && compiled_) {
    for (auto& [tensor_idx, swap_tensor] : dmabuf_swap_tensors_) {
      int current_fd = dmabuf_manager_->GetActiveFd(tensor_idx);
      int last_fd = dmabuf_active_fds_[tensor_idx];

      if (current_fd != last_fd && current_fd >= 0 && swap_tensor) {
        void* old_ptr = nullptr;
        // Cast fd to pointer (TIM-VX convention for DMABUF tensors)
        bool success = swap_tensor->SwapHandle(
            reinterpret_cast<void*>(static_cast<intptr_t>(current_fd)),
            false,  // not malloc'd by ovxlib
            &old_ptr);

        if (success) {
          dmabuf_active_fds_[tensor_idx] = current_fd;
          TFLITE_LOG(TFLITE_LOG_INFO,
                     "Buffer cycling: swapped tensor %d from fd=%d to fd=%d",
                     tensor_idx, last_fd, current_fd);
        } else {
          TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                          "Buffer cycling: SwapHandle failed for tensor %d",
                          tensor_idx);
        }
      }
    }
  }
#endif

  TFLITE_LOG(TFLITE_LOG_INFO, "Invoking graph");
#ifdef MULTI_DEVICE_FEATURE_MODE
    auto executable_set = tim::vx::platform::CreateExecutableSet({executable_, executable_});
    executor_->Submit(executable_set,executable_set);
    executor_->Trigger();
    tensor_index = 0;
#else
    if (!layout_infered_.first->Run()) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to run graph");
      return kTfLiteDelegateError;
    }
#endif

  for (int tensor_idx : op_data.subgraph_outputs) {
    TfLiteTensor& tf_tensor = context->tensors[tensor_idx];
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
    // Check if this output is dmabuf-backed
    if (dmabuf_manager_ && dmabuf_enabled_) {
      TfLiteBufferHandle dmabuf_handle = dmabuf_manager_->FindByTensorIndex(tensor_idx);
      if (dmabuf_handle != kTfLiteNullBufferHandle) {
        auto src_output_tensor = tensors_[tensor_idx];
        auto infered_output_tensor = layout_infered_.second[src_output_tensor];
        if (infered_output_tensor) {
          // Check if inferred tensor preserved the dmabuf (true zero-copy path)
          if (infered_output_tensor->HasDmaBuf()) {
            // True zero-copy: NPU wrote directly to dmabuf, no copy needed
            // NOTE: Cache synchronization is the CLIENT's responsibility via
            // DMA_BUF_IOCTL_SYNC when mmap'ing the buffer. The delegate does NOT
            // perform cache operations in the zero-copy path to support true
            // hardware-to-hardware pipelines (camera->NPU->display).
            TFLITE_LOG(TFLITE_LOG_INFO, "Dmabuf output %d: zero-copy (fd=%lld)",
                       tensor_idx, (long long)infered_output_tensor->GetDmaBufFd());
          } else {
            // Fallback: inferred tensor doesn't have dmabuf, need to copy
            void* dmabuf_data = src_output_tensor->map(false /* don't invalidate, we're writing */);
            if (dmabuf_data) {
              infered_output_tensor->CopyDataFromTensor(dmabuf_data);
              src_output_tensor->unmap();
              TFLITE_LOG(TFLITE_LOG_INFO, "Dmabuf output %d: fallback copy from inferred tensor to dmabuf", tensor_idx);
            } else {
              TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to map dmabuf for output %d", tensor_idx);
            }
          }
        }
        continue;
      }
    }
#endif
    TFLITE_LOG(
        TFLITE_LOG_INFO, "Copying output %d, %s", tensor_idx, tf_tensor.name);
#ifdef MULTI_DEVICE_FEATURE_MODE
      void* tensor_data = reinterpret_cast<void*>(tf_tensor.data.raw);
      outputs_[tensor_index]->CopyDataFromTensor(tensor_data);
#else
      auto src_output_tensor = tensors_[tensor_idx];
      if (!src_output_tensor.get()) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to copy output tensor!");
        return kTfLiteDelegateError;
      }

      void* tensor_data = reinterpret_cast<void*>(tf_tensor.data.raw);
      auto infered_output_tesnor = layout_infered_.second[src_output_tensor];
      if (infered_output_tesnor) {
        infered_output_tesnor->CopyDataFromTensor(tensor_data);
      } else {
        TFLITE_LOG(TFLITE_LOG_ERROR,
                   "Output tensor missing: report issue to VSI");
      }
#endif
  }

  // Copy output states to input states
  for (int tensor_idx : op_data.subgraph_states) {
    TfLiteTensor& tf_tensor = context->tensors[tensor_idx];
    TFLITE_LOG(TFLITE_LOG_INFO, "Copying state %d, %s", tensor_idx, tf_tensor.name);
    auto src_state_tensor = state_tensors_[tensor_idx];
    if (!src_state_tensor.get()) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Disaster!");
      return kTfLiteDelegateError;
    }

    void* tensor_data = reinterpret_cast<void*>(tf_tensor.data.raw);
    auto infered_state_tensor = layout_infered_.second[src_state_tensor];
    infered_state_tensor->CopyDataFromTensor(tensor_data);
  }

  return kTfLiteOk;
}

Delegate::Delegate() : dmabuf_enabled_(false), compiled_(false), parent_delegate_(nullptr) {}

void Delegate::SetDmaBufManager(std::shared_ptr<DmaBufManager> mgr) {
  dmabuf_manager_ = mgr;
  dmabuf_enabled_ = (mgr && mgr->IsSupported());
}

TfLiteStatus Delegate::SwapDmaBufTensor(int tensor_index, int new_fd) {
#ifdef VX_CREATE_TENSOR_SUPPORT_PHYSICAL
  if (!dmabuf_enabled_ || !compiled_) {
    return kTfLiteError;
  }

  auto it = dmabuf_swap_tensors_.find(tensor_index);
  if (it == dmabuf_swap_tensors_.end() || !it->second) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                    "SwapDmaBufTensor: tensor %d not found in swap map",
                    tensor_index);
    return kTfLiteError;
  }

  void* old_ptr = nullptr;
  bool success = it->second->SwapHandle(
      reinterpret_cast<void*>(static_cast<intptr_t>(new_fd)),
      false,
      &old_ptr);

  if (success) {
    dmabuf_active_fds_[tensor_index] = new_fd;
    TFLITE_LOG(TFLITE_LOG_INFO,
               "SwapDmaBufTensor: tensor %d swapped to fd=%d",
               tensor_index, new_fd);
    return kTfLiteOk;
  }

  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "SwapDmaBufTensor: SwapHandle failed for tensor %d",
                  tensor_index);
  return kTfLiteError;
#else
  (void)tensor_index;
  (void)new_fd;
  return kTfLiteError;
#endif
}

TfLiteStatus Delegate::InvalidateGraph() {
  TFLITE_LOG(TFLITE_LOG_INFO, "InvalidateGraph: clearing compiled state");

  compiled_ = false;

  // Clear graph and context
  layout_infered_.first.reset();
  layout_infered_.second.clear();
  graph_.reset();
  context_.reset();

  // Clear tensor maps
  tensors_.clear();
  state_tensors_.clear();
  ops_.clear();

  // Clear buffer cycling state
  dmabuf_swap_tensors_.clear();
  dmabuf_active_fds_.clear();

  // Clear log dedup so messages reappear after recompilation
  dmabuf_zerocopy_logged_.clear();

  return kTfLiteOk;
}

bool Delegate::CheckAndClearInvalidation() {
  if (parent_delegate_ && parent_delegate_->needs_invalidation) {
    parent_delegate_->needs_invalidation = false;
    InvalidateGraph();
    return true;
  }
  return false;
}

}  // namespace delegate
}  // namespace vx
