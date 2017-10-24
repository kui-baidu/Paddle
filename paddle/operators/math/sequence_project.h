/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/framework/eigen.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/tensor.h"
#include "paddle/operators/math/im2col.h"

namespace paddle {
namespace operators {
namespace math {

//    template <typename T, int MajorType = Eigen::RowMajor,
//            typename IndexType = Eigen::DenseIndex>
//    using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;
/*
 * \brief SequenceProject projects features of context_length time-steps of each
 * instance.
 *
 * \param in            Input data.
 * \param inShape       The shape of Input data,
 *                      [minibatch, number_of_input_features].
 * \param inShape       A float LoDTensor.
 *
 * \param padding_data  Padding data.
 * \param inShape       The shape of Padding data,
 *                      [up_pad + down_pad, number_of_input_features].
 * \param inShape       A float LoDTensor.
 *
 * \param col           Col data.
 * \param inShape       The shape of Col data,
 *                      [minibatch, 1].
 * \param inShape       A float LoDTensor.
 *
 * For a mini-batch of 2 variable lengths sentences, containing 3, and 1
 * time-steps:
 *
 * Assumed input (X) is a [4, M, N] float LoDTensor, and X->lod()[0] = [0, 3,
 * 4].
 * Besides, for the sake of simplicity, we assume M=1 and N=2.
 *
 * X = [[a1, a2;
 *       b1, b2;
 *       c1, c2]
 *      [d1, d2]]
 *
 * This is to say that input (X) has 4 words and the dimension of each word
 * representation is 2.
 *
 * - Case1:
 * If context_start is -1 and padding_trainable is false, we use zero to pad
 * instead of learned weight to pad,
 * and the context_lenth is 3, the output (Out) is:
 *
 * Out =[[0,  0,  a1, a2, b1, b2;
 *        a1, a2, b1, b2, c1, c2;
 *        b1, b2, c1, c2, 0,  0 ]
 *       [0,  0,  d1, d2, 0,  0 ]]
 *
 * - Case2:
 * If context_start is -1 and padding_trainable is true, we use learned weight
 * to pad,
 * and the context_lenth is 3, the output (Out) is:
 *
 * Out = [[w1, w2, a1, a2, b1, b2;
 *         a1, a2, b1, b2, c1, c2;
 *         b1, b2, c1, c2, w3, w4]
 *        [w1, w2, d1, d2, w3, w4]]
 *
 */

template <typename Place, typename T>
class SequenceProjectFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::LoDTensor* in,
                  const framework::LoDTensor* padding_data,
                  framework::LoDTensor* col, bool padding_trainable,
                  int context_start, int context_length, int context_stride,
                  int up_pad, int down_pad) {
    auto lod_level_0 = in->lod()[0];

    paddle::operators::math::Im2ColFunctor<
        paddle::operators::math::ColFormat::kOCF, Place, float>
        im2col_ocf;

    int input_row_begin, input_row_end;
    int sequence_height, sequence_width;
    sequence_width = in->dims()[1];

    for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
      input_row_begin = (context_start > 0)
                            ? static_cast<int>(lod_level_0[i]) + context_start
                            : static_cast<int>(lod_level_0[i]);
      input_row_end = static_cast<int>(lod_level_0[i + 1]);

      framework::Tensor out_t =
          col->Slice(static_cast<int>(lod_level_0[i]),
                     static_cast<int>(lod_level_0[i + 1]));

      sequence_height = static_cast<int>(out_t.dims()[0]);

      if (input_row_begin < input_row_end) {
        framework::Tensor in_t = in->Slice(input_row_begin, input_row_end);

        std::vector<int64_t> output_shape(
            {sequence_height, 1, 1, context_length,
             sequence_width});  // output_height, output_width,
        // input_channels, filter_height, filter_width

        out_t.Resize(framework::make_ddim(output_shape));

        std::vector<int64_t> input_shape(
            {1, input_row_end - input_row_begin,
             sequence_width});  // input_channels, input_height, input_width
        in_t.Resize(framework::make_ddim(input_shape));

        im2col_ocf(context, in_t, out_t,
                   /*stride_height*/ context_stride, /*stride_width*/ 0, up_pad,
                   down_pad);
      }

      if (padding_trainable) {
        // add up trainable data
        out_t.Resize(framework::make_ddim(
            {sequence_height * context_length, sequence_width}));

        if (up_pad > 0) {  // add up pad
          int padding_rows = std::min(
              up_pad, static_cast<int>(lod_level_0[i + 1] - lod_level_0[i]));

          for (int k = 0; k < padding_rows; ++k) {
            int padding_size =
                k + context_length < up_pad ? context_length : up_pad - k;
            framework::Tensor out_t_sub = out_t.Slice(
                k * context_length, k * context_length + padding_size);
            framework::Tensor w_sub = padding_data->Slice(k, k + padding_size);
            // in this block, using EigenVector<T>::Flatten is ok too.
            auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
            auto w_sub_e = EigenMatrix<T>::From(w_sub);
            out_t_sub_e.device(*context.GetEigenDevice<Place>()) = w_sub_e;
          }
        }
        if (down_pad > 0) {  // add down pad
          int down_pad_begin_row =
              std::max(0,
                       (sequence_height - context_start - context_length) + 1) +
              1;
          int padding_begin = std::max(0, context_start - sequence_height);
          int padding_size =
              sequence_height - context_start >= context_length
                  ? 1
                  : context_length - (sequence_height - context_start);
          if (context_start >= sequence_height) padding_size = context_length;
          int padding_idx = padding_begin;
          for (int t = 0; t + down_pad_begin_row <= sequence_height;
               ++t, ++padding_size) {
            if (context_start >= sequence_height) padding_size = context_length;
            if (padding_size > context_length) {
              padding_size = context_length;
              padding_idx++;
            }
            if (padding_begin > 0 || sequence_height == context_start)
              padding_idx = padding_begin + t;
            framework::Tensor out_t_sub = out_t.Slice(
                (down_pad_begin_row + t) * context_length - padding_size,
                (down_pad_begin_row + t) * context_length);
            framework::Tensor w_sub = padding_data->Slice(
                up_pad + padding_idx, up_pad + padding_idx + padding_size);
            auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
            auto w_sub_e = EigenMatrix<T>::From(w_sub);
            out_t_sub_e.device(*context.GetEigenDevice<Place>()) = w_sub_e;
          }
        }
      }
      out_t.Resize(framework::make_ddim(
          {sequence_height, context_length * sequence_width}));
    }
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
