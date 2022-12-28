// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/linear_bias_kernel.h"
#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/dynload/cublasLt.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
#include <cublas_v2.h>
#include <cuda_runtime.h>

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
#include <cublasLt.h>
#endif
*/

namespace phi {

// FP16 Tensor core wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int64_t m,
    int64_t n,
    int64_t k,
    const float* alpha,
    const phi::dtype::float16* A,
    int64_t lda,
    const phi::dtype::float16* B,
    int64_t ldb,
    const float* beta,
    phi::dtype::float16* C,
    int64_t ldc) {
  //return cublasGemmEx(
  return phi::dynload::cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_16F,
      lda,
      B,
      CUDA_R_16F,
      ldb,
      beta,
      C,
      CUDA_R_16F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// BF16 Tensor core wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int64_t m,
    int64_t n,
    int64_t k,
    const float* alpha,
    const phi::dtype::bfloat16* A,
    int64_t lda,
    const phi::dtype::bfloat16* B,
    int64_t ldb,
    const float* beta,
    phi::dtype::bfloat16* C,
    int64_t ldc) {
  //return cublasGemmEx(
  return phi::dynload::cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_16BF,
      lda,
      B,
      CUDA_R_16BF,
      ldb,
      beta,
      C,
      CUDA_R_16BF,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600

int gemm_bias_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int64_t m,
    int64_t n,
    int64_t k,
    const float *alpha, /* host pointer */
    const phi::dtype::float16* A,
    int64_t lda,
    const phi::dtype::float16* B,
    int64_t ldb,
    const float *beta, /* host pointer */
    phi::dtype::float16* C,
    int64_t ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* bias) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_BIAS;
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_16F, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  //status = cublasLtMatmul(ltHandle,
  status = phi::dynload::cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          //&heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

int gemm_bias_lt(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int64_t m,
    int64_t n,
    int64_t k,
    const float *alpha, /* host pointer */
    const phi::dtype::bfloat16* A,
    int64_t lda,
    const phi::dtype::bfloat16* B,
    int64_t ldb,
    const float *beta, /* host pointer */
    phi::dtype::bfloat16* C,
    int64_t ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    const void* bias) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  status = cublasLtMatmulDescInit(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
      epilogue = CUBLASLT_EPILOGUE_BIAS;
  }

  status = cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  status = cublasLtMatrixLayoutInit(
    &Adesc, CUDA_R_16BF, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(
    &Bdesc, CUDA_R_16BF, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatrixLayoutInit(&Cdesc, CUDA_R_16BF, m, n, ldc);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from cudaMalloc)
  status = cublasLtMatmulPreferenceInit(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = cublasLtMatmulPreferenceSetAttribute(
    &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = phi::dynload::cublasLtMatmulAlgoGetHeuristic(
    ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults);
  if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  //status = cublasLtMatmul(ltHandle,
  status = phi::dynload::cublasLtMatmul(ltHandle,
                          &operationDesc,
                          alpha,
                          A,
                          &Adesc,
                          B,
                          &Bdesc,
                          beta,
                          C,
                          &Cdesc,
                          C,
                          &Cdesc,
                          //&heuristicResult.algo,
                          NULL,
                          workspace,
                          workspaceSize,
                          stream);

CLEANUP:
  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

#endif


template <typename T, typename Context>
int linear_bias_forward_cuda(const Context& ctx, const DenseTensor& input, const T *weight, const DenseTensor& bias, int64_t in_features, int64_t batch_size, int64_t out_features, DenseTensor* output, void *lt_workspace) {
    cublasHandle_t handle = ctx.cublas_handle();

    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha          = 1.0;
    const float beta_zero       = 0.0;
    const float beta_one       = 1.0;
    int status = 1;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600
    status = gemm_bias_lt(
    (cublasLtHandle_t)handle,
    CUBLAS_OP_T,
    CUBLAS_OP_N,
    out_features,
    batch_size,
    in_features,
    &alpha, /* host pointer */
    weight,
    in_features,
    input.data<T>(),
    in_features,
    &beta_zero, /* host pointer */
    output->data<T>(),
    out_features,
    lt_workspace,
    1 << 22,
    stream,
    true,
    static_cast<const void*>(bias.data<T>()));
#endif
    if (status != 0){
        phi::Copy<Context>(ctx, bias, ctx.GetPlace(), false, output);

        //phi::funcs::CBlas<T>::GEMM(
        status = gemm_bias(
          handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          out_features,
          batch_size,
          in_features,
          &alpha,
          weight,
          in_features,
          input.data<T>(),
          in_features,
          &beta_one,
          output->data<T>(),
          out_features);
    }
    return status;
}

template <typename T, typename Context>
void LinearBiasKernel(const Context& ctx, const DenseTensor& input, const DenseTensor& weight, const DenseTensor& bias, DenseTensor* out) {

  std::vector<std::int64_t> input_dims = vectorize(input.dims());
  std::vector<std::int64_t> weight_dims = vectorize(weight.dims());

  int64_t batch_size = input_dims[0];
  int64_t in_features = input_dims[1];
  int64_t out_features = weight_dims[0];

  ctx.template Alloc<T>(out);

  // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  DenseTensor lt_workspace = phi::Empty<T, Context>(ctx, {1 << 22});
  T* lt_workspace_ptr = lt_workspace.data<T>();

    const T* w_ptr = weight.data<T>();
    auto result = linear_bias_forward_cuda<T, Context>(
        ctx,
        input,
        w_ptr,
        bias,
        in_features,
        batch_size,
        out_features,
        out,
        (void*) lt_workspace_ptr);
    PADDLE_ENFORCE_EQ(result, 0, phi::errors::InvalidArgument("linear_bias_forward_cuda failed."));
}

}  // namespace phi

PD_REGISTER_KERNEL(linear_bias,
                   GPU,
                   ALL_LAYOUT,
                   phi::LinearBiasKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
