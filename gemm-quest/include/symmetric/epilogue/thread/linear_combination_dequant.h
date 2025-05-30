/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear combination operations used by dequantize
  epilogues.
*/
#pragma once

#include <torch/extension.h>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {
namespace symmetric {

struct MyScaleType {
  enum Kind {
    Dequantize,
  };
};
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ElementOutput_,
          int Count,
          typename ElementAccumulator_,
          typename ElementCompute_ = cutlass::half_t,
          MyScaleType::Kind Scale = MyScaleType::Dequantize,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
          typename ElementSource_ = cutlass::half_t>
class LinearCombinationDequant {
 public:
  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  //using ElementC = ElementSource_;
  //using ElementD = ElementOutput_;

  static int const kCount = Count;
  static const MyScaleType::Kind kScale = MyScaleType::Dequantize;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentMultSource = Array<cutlass::bfloat16_t, kCount>;
  using FragmentAddSource = Array<int32_t, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  struct Params {
    ElementCompute beta;

    CUTLASS_HOST_DEVICE
    Params() : beta(ElementCompute(0)) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute beta) : beta(beta) {}
  };

 private:
  //
  // Data members
  //

  ElementCompute beta_ = ElementCompute(0);

 public:
  /// Constructs the function object
  CUTLASS_HOST_DEVICE
  LinearCombinationDequant(Params const &params) { beta_ = params.beta; }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentSource const &source,
                            FragmentMultSource const &row_vec_alpha,
                            FragmentMultSource const &col_vec_alpha,
                            FragmentAddSource const &vec_a_add_alpha,
                            FragmentAddSource const &vec_b_add_alpha) const {
    //NumericArrayConverter<cutlass::bfloat16_t, cutlass::bfloat16_t, kCount, Round>
    //    source_converter;
    //NumericArrayConverter<int32_t, int32_t, kCount, Round>
    //    source_add_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    //NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
    //    destination_converter;

    //FragmentCompute converted_source = source_converter(source);
    //FragmentCompute converted_row_vec_alpha = source_converter(row_vec_alpha);
    //FragmentCompute converted_col_vec_alpha = source_converter(col_vec_alpha);
    //FragmentCompute converted_vec_a_add_alpha = source_add_converter(vec_a_add_alpha);
    //FragmentCompute converted_vec_b_add_alpha = source_add_converter(vec_b_add_alpha);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    FragmentOutput result;
    cutlass::bfloat16_t *result_ptr = reinterpret_cast<cutlass::bfloat16_t *>(&result);

    //const cutlass::bfloat16_t *source_ptr =
    //    reinterpret_cast<const cutlass::bfloat16_t *>(&converted_source);
    const cutlass::half_t *acc_ptr =
        reinterpret_cast<const cutlass::half_t *>(&converted_accumulator);

    //const cutlass::bfloat16_t *row_vec_ptr =
    //    reinterpret_cast<const cutlass::bfloat16_t *>(&row_vec_alpha);
    //const cutlass::bfloat16_t *col_vec_ptr =
    //    reinterpret_cast<const cutlass::bfloat16_t *>(&col_vec_alpha);
    const int32_t *vec_a_add_ptr =
        reinterpret_cast<const int32_t *>(&vec_a_add_alpha);
    const int32_t *vec_b_add_ptr =
        reinterpret_cast<const int32_t *>(&vec_b_add_alpha);

    /* if(!blockIdx.x && !blockIdx.y && !blockIdx.z && !threadIdx.x && !threadIdx.y && !threadIdx.z) {
      printf("%.2f\n", static_cast<float>(acc_ptr[0]));
    } */

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      float tmp = (2 * static_cast<float>(acc_ptr[i])
                  + vec_a_add_ptr[i]
                  + vec_b_add_ptr[i]) *
                  .5 * static_cast<float>(col_vec_alpha[i]) *
                  static_cast<float>(row_vec_alpha[i]);

      /* if(!blockIdx.x && !blockIdx.y && !blockIdx.z && threadIdx.x==0 && !threadIdx.y && !threadIdx.z && i==2) {
        printf("%.4f -> %.4f\n", __bfloat162float(acc_ptr[i]), static_cast<float>(tmp));
      } */

      result_ptr[i] = (cutlass::bfloat16_t) __float2bfloat16_rn(tmp);
      //result_ptr[i] = __float2half(tmp);
    }

   //return destination_converter(result);
   return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace symmetric
}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass
