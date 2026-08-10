#include <torch/extension.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "utils.h"

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace norm {

namespace {

constexpr int64_t kHiddenSize = 5376;
constexpr int kBlockSize = 64;
constexpr int kGroupSize = 32;
constexpr int kBlocks = kHiddenSize / kBlockSize;
constexpr int kBlocksPerItem = kBlocks / kGroupSize;
constexpr int kExtraBlocks = kBlocks % kGroupSize;
constexpr int kInputBytes =
    kHiddenSize * static_cast<int>(sizeof(bf16));
constexpr int kPartialBytes =
    ((kGroupSize * static_cast<int>(sizeof(float)) + 15) / 16) * 16;
constexpr int kSlmBytes = kInputBytes + kPartialBytes;

class FusedMiniMaxH3IndexedRMSAdaLNBMGKernel;

void launch_fused_minimax_h3_indexed_rms_adaln(
    const bf16* weight,
    const bf16* input,
    const bf16* scale,
    const bf16* shift,
    const int64_t* indices,
    bf16* output,
    float eps,
    int64_t rows,
    int64_t compact_rows,
    int64_t scale_row_stride,
    int64_t shift_row_stride,
    const c10::Device& device) {
    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for<FusedMiniMaxH3IndexedRMSAdaLNBMGKernel>(
            sycl::nd_range<2>(
                sycl::range<2>(
                    static_cast<size_t>(rows), kGroupSize),
                sycl::range<2>(1, kGroupSize)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<kSlmBytes>();
                const int64_t row = item.get_global_id(0);
                const int local_id = item.get_local_id(1);
                const bf16* input_row = input + row * kHiddenSize;
                bf16* output_row = output + row * kHiddenSize;
                const int first_block =
                    kBlocksPerItem * local_id +
                    std::min(local_id, kExtraBlocks);
                const int last_block =
                    first_block + kBlocksPerItem +
                    (local_id < kExtraBlocks);
                simd<float, kBlockSize> accumulator = 0;

                for (int block = first_block; block < last_block; ++block) {
                    simd<bf16, kBlockSize> values =
                        block_load<bf16, kBlockSize>(
                            input_row + block * kBlockSize);
                    slm_block_store<bf16, kBlockSize>(
                        block * kBlockSize *
                            static_cast<int>(sizeof(bf16)),
                        values);
                    simd<float, kBlockSize> values_f32 = values;
                    accumulator += values_f32 * values_f32;
                }
                const float partial =
                    sycl::ext::intel::esimd::detail::sum<
                        float, float, kBlockSize>(accumulator) /
                    static_cast<float>(kHiddenSize);
                slm_block_store<float, 1>(
                    kInputBytes +
                        local_id * static_cast<int>(sizeof(float)),
                    partial);
                barrier();

                simd<float, kGroupSize> partials =
                    slm_block_load<float, kGroupSize>(kInputBytes);
                const float mean =
                    sycl::ext::intel::esimd::detail::sum<
                        float, float, kGroupSize>(partials);
                const float rms_scale = rsqrt(mean + eps);
                const int64_t modulation_row = indices[row];
                if (modulation_row < 0 || modulation_row >= compact_rows) {
                    for (int block = first_block;
                         block < last_block;
                         ++block) {
                        block_store<bf16, kBlockSize>(
                            output_row + block * kBlockSize,
                            simd<bf16, kBlockSize>(
                                std::numeric_limits<float>::quiet_NaN()));
                    }
                    return;
                }
                const bf16* scale_row =
                    scale + modulation_row * scale_row_stride;
                const bf16* shift_row =
                    shift + modulation_row * shift_row_stride;

                for (int block = first_block; block < last_block; ++block) {
                    const int column = block * kBlockSize;
                    simd<float, kBlockSize> values =
                        slm_block_load<bf16, kBlockSize>(
                            column * static_cast<int>(sizeof(bf16)));
                    simd<float, kBlockSize> weights =
                        block_load<bf16, kBlockSize>(weight + column);
                    simd<bf16, kBlockSize> normalized_bf16 =
                        simd<bf16, kBlockSize>(
                            values * rms_scale * weights);
                    simd<float, kBlockSize> normalized = normalized_bf16;
                    simd<float, kBlockSize> scale_values =
                        block_load<bf16, kBlockSize>(scale_row + column);
                    simd<bf16, kBlockSize> one_plus_scale_bf16 =
                        simd<bf16, kBlockSize>(1.0f + scale_values);
                    simd<float, kBlockSize> one_plus_scale =
                        one_plus_scale_bf16;
                    simd<bf16, kBlockSize> product_bf16 =
                        simd<bf16, kBlockSize>(
                            normalized * one_plus_scale);
                    simd<float, kBlockSize> product = product_bf16;
                    simd<float, kBlockSize> shift_values =
                        block_load<bf16, kBlockSize>(shift_row + column);
                    block_store<bf16, kBlockSize>(
                        output_row + column,
                        simd<bf16, kBlockSize>(
                            product + shift_values));
                }
            });
    };
    utils::submit_kernel(
        cgf, device, "fused_minimax_h3_indexed_rms_adaln_bmg");
}

void check_modulation(
    const torch::Tensor& modulation,
    const torch::Tensor& input,
    const char* name) {
    TORCH_CHECK(
        modulation.device().is_xpu(),
        name, " must be an XPU tensor");
    TORCH_CHECK(
        modulation.device() == input.device(),
        name, " and input must be on the same XPU");
    TORCH_CHECK(
        modulation.scalar_type() == torch::kBFloat16,
        name, " must be BF16");
    TORCH_CHECK(
        modulation.dim() == 2 &&
            modulation.size(0) > 0 &&
            modulation.size(1) == kHiddenSize,
        name, " must have shape [compact_rows, 5376]");
    TORCH_CHECK(
        modulation.stride(1) == 1 &&
            modulation.stride(0) >= kHiddenSize,
        name, " must have a dense hidden dimension");
}

}  // namespace

torch::Tensor fused_minimax_h3_indexed_rms_adaln(
    const torch::Tensor& weight,
    const torch::Tensor& input,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    const torch::Tensor& indices,
    double eps) {
    TORCH_CHECK(input.device().is_xpu(), "input must be an XPU tensor");
    TORCH_CHECK(
        input.scalar_type() == torch::kBFloat16,
        "input must be BF16");
    TORCH_CHECK(
        input.dim() == 2 &&
            input.size(0) > 0 &&
            input.size(1) == kHiddenSize,
        "input must have shape [rows, 5376]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(
        weight.device().is_xpu() &&
            weight.device() == input.device(),
        "weight and input must be on the same XPU");
    TORCH_CHECK(
        weight.scalar_type() == torch::kBFloat16 &&
            weight.dim() == 1 &&
            weight.size(0) == kHiddenSize &&
            weight.is_contiguous(),
        "weight must be contiguous BF16 [5376]");
    TORCH_CHECK(
        scale.sizes() == shift.sizes(),
        "scale and shift must have matching shapes");
    check_modulation(scale, input, "scale");
    check_modulation(shift, input, "shift");
    TORCH_CHECK(
        indices.device().is_xpu() &&
            indices.device() == input.device(),
        "indices and input must be on the same XPU");
    TORCH_CHECK(
        indices.scalar_type() == torch::kInt64 &&
            indices.dim() == 1 &&
            indices.size(0) == input.size(0) &&
            indices.is_contiguous(),
        "indices must be contiguous int64 [rows]");
    TORCH_CHECK(
        std::isfinite(eps) && eps > 0.0,
        "eps must be a positive finite value");
    TORCH_CHECK(
        input.numel() <= std::numeric_limits<uint32_t>::max() &&
            scale.numel() <= std::numeric_limits<uint32_t>::max(),
        "inputs are too large for the MiniMax H3 tuning kernel");

    auto output = torch::empty_like(input);
    launch_fused_minimax_h3_indexed_rms_adaln(
        reinterpret_cast<const bf16*>(weight.data_ptr()),
        reinterpret_cast<const bf16*>(input.data_ptr()),
        reinterpret_cast<const bf16*>(scale.data_ptr()),
        reinterpret_cast<const bf16*>(shift.data_ptr()),
        reinterpret_cast<const int64_t*>(indices.data_ptr()),
        reinterpret_cast<bf16*>(output.data_ptr()),
        static_cast<float>(eps),
        input.size(0),
        scale.size(0),
        scale.stride(0),
        shift.stride(0),
        input.device());
    return output;
}

}  // namespace norm
}  // namespace omni_xpu
