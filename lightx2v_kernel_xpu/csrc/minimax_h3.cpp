#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

#include "utils.h"

using bf16 = sycl::ext::oneapi::bfloat16;

namespace omni_xpu {
namespace norm {

namespace {

constexpr int64_t kMaxHeads = 56;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kRotaryDim = 96;
constexpr int64_t kRotaryHalf = kRotaryDim / 2;

class FusedMiniMaxH3QKRMSNormRopeKernel;

inline bf16 round_to_bf16(float value) {
    return static_cast<bf16>(value);
}

inline float sum_squares_32(const bf16* input) {
    float squares[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const float value = static_cast<float>(input[i]);
        squares[i] = value * value;
    }
#pragma unroll
    for (int stride = 16; stride > 0; stride /= 2) {
#pragma unroll
        for (int i = 0; i < stride; ++i) {
            squares[i] += squares[i + stride];
        }
    }
    return squares[0];
}

inline bf16 normalize_value(
    const bf16* input,
    const bf16* weight,
    int dim,
    float scale) {
    return round_to_bf16(
        static_cast<float>(input[dim]) * scale *
        static_cast<float>(weight[dim]));
}

void launch_fused_minimax_h3_qk_rmsnorm_rope(
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    float eps) {
    // One WG handles either Q or K for one row. Four work-items reduce each
    // head independently while the WG computes the row's 48 cos/sin pairs
    // once and reuses them across all TP-local heads.
    constexpr uint32_t WorkGroupSize = 256;
    constexpr uint32_t LanesPerHead = 4;
    const uint32_t heads = static_cast<uint32_t>(q.size(1));
    const uint32_t active_items = heads * LanesPerHead;
    const auto* q_weight_ptr =
        reinterpret_cast<const bf16*>(q_weight.data_ptr());
    const auto* k_weight_ptr =
        reinterpret_cast<const bf16*>(k_weight.data_ptr());
    const auto* q_ptr = reinterpret_cast<const bf16*>(q.data_ptr());
    const auto* k_ptr = reinterpret_cast<const bf16*>(k.data_ptr());
    const auto* freqs_ptr = freqs.data_ptr<float>();
    auto* q_out_ptr = reinterpret_cast<bf16*>(q_out.data_ptr());
    auto* k_out_ptr = reinterpret_cast<bf16*>(k_out.data_ptr());
    const uint32_t rows = static_cast<uint32_t>(q.size(0));
    const uint32_t q_row_stride = static_cast<uint32_t>(q.stride(0));
    const uint32_t k_row_stride = static_cast<uint32_t>(k.stride(0));
    const uint32_t freqs_row_stride =
        static_cast<uint32_t>(freqs.stride(0));

    auto cgf = [&](sycl::handler& handler) {
        sycl::local_accessor<bf16, 1> cos_cache(
            sycl::range<1>(kRotaryDim), handler);
        sycl::local_accessor<bf16, 1> sin_cache(
            sycl::range<1>(kRotaryDim), handler);
        sycl::local_accessor<float, 1> partial_cache(
            sycl::range<1>(active_items), handler);
        sycl::local_accessor<float, 1> scale_cache(
            sycl::range<1>(heads), handler);

        handler.parallel_for<FusedMiniMaxH3QKRMSNormRopeKernel>(
            sycl::nd_range<1>(
                sycl::range<1>(
                    static_cast<size_t>(rows) * 2 * WorkGroupSize),
                sycl::range<1>(WorkGroupSize)),
            [=](sycl::nd_item<1> item) {
                const uint32_t group =
                    static_cast<uint32_t>(item.get_group(0));
                const bool is_key = group >= rows;
                const uint32_t row = is_key ? group - rows : group;
                const uint32_t local_id =
                    static_cast<uint32_t>(item.get_local_id(0));
                const bool active = local_id < active_items;
                const uint32_t head =
                    active ? local_id / LanesPerHead : 0;
                const uint32_t lane =
                    active ? local_id % LanesPerHead : 0;
                const uint32_t input_row_stride =
                    is_key ? k_row_stride : q_row_stride;
                const bf16* input_base =
                    (is_key ? k_ptr : q_ptr) +
                    row * input_row_stride + head * kHeadDim;
                const bf16* weight =
                    is_key ? k_weight_ptr : q_weight_ptr;

                if (local_id < kRotaryDim) {
                    const float frequency =
                        freqs_ptr[row * freqs_row_stride + local_id];
                    cos_cache[local_id] =
                        static_cast<bf16>(sycl::cos(frequency));
                    sin_cache[local_id] =
                        static_cast<bf16>(sycl::sin(frequency));
                }
                if (active) {
                    partial_cache[local_id] =
                        sum_squares_32(input_base + lane * 32) /
                        static_cast<float>(kHeadDim);
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (active && lane == 0) {
                    const uint32_t offset = head * LanesPerHead;
                    scale_cache[head] = sycl::rsqrt(
                        (partial_cache[offset] +
                         partial_cache[offset + 1]) +
                            (partial_cache[offset + 2] +
                             partial_cache[offset + 3]) +
                        eps);
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (!active) return;

                const float scale = scale_cache[head];
                bf16* output_base =
                    (is_key ? k_out_ptr : q_out_ptr) +
                    (row * heads + head) * kHeadDim;
                const int first_dim = static_cast<int>(lane * 32);
#pragma unroll
                for (int offset = 0; offset < 32; ++offset) {
                    const int dim = first_dim + offset;
                    if (dim < kRotaryHalf) {
                        const bf16 x1 =
                            normalize_value(input_base, weight, dim, scale);
                        const bf16 x2 = normalize_value(
                            input_base, weight, dim + kRotaryHalf, scale);
                        const bf16 x1_cos = round_to_bf16(
                            static_cast<float>(x1) *
                            static_cast<float>(cos_cache[dim]));
                        const bf16 x2_sin = round_to_bf16(
                            static_cast<float>(x2) *
                            static_cast<float>(sin_cache[dim]));
                        output_base[dim] = round_to_bf16(
                            static_cast<float>(x1_cos) -
                            static_cast<float>(x2_sin));
                    } else if (dim < kRotaryDim) {
                        const int pair = dim - kRotaryHalf;
                        const bf16 x1 = normalize_value(
                            input_base, weight, pair, scale);
                        const bf16 x2 =
                            normalize_value(input_base, weight, dim, scale);
                        const bf16 x2_cos = round_to_bf16(
                            static_cast<float>(x2) *
                            static_cast<float>(cos_cache[dim]));
                        const bf16 x1_sin = round_to_bf16(
                            static_cast<float>(x1) *
                            static_cast<float>(sin_cache[dim]));
                        output_base[dim] = round_to_bf16(
                            static_cast<float>(x2_cos) +
                            static_cast<float>(x1_sin));
                    } else {
                        output_base[dim] =
                            normalize_value(input_base, weight, dim, scale);
                    }
                }
            });
    };
    utils::submit_kernel(
        cgf, q.device(), "fused_minimax_h3_qk_rmsnorm_rope");
}

void check_input_tensor(
    const torch::Tensor& tensor,
    const char* name,
    int64_t rows,
    int64_t heads) {
    TORCH_CHECK(tensor.device().is_xpu(), name, " must be an XPU tensor");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kBFloat16,
        name, " must be BF16");
    TORCH_CHECK(
        tensor.dim() == 3 &&
            tensor.size(0) == rows &&
        tensor.size(1) == heads &&
            tensor.size(2) == kHeadDim,
        name, " must have shape [rows, TP-local heads, 128]");
    const int64_t contiguous_row_stride = heads * kHeadDim;
    const int64_t packed_row_stride = 3 * contiguous_row_stride;
    TORCH_CHECK(
        tensor.stride(2) == 1 &&
        tensor.stride(1) == kHeadDim &&
        (tensor.stride(0) == contiguous_row_stride ||
         tensor.stride(0) == packed_row_stride),
        name,
        " must have strides for contiguous or packed-QKV rows");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor>
fused_minimax_h3_qk_rmsnorm_rope(
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs,
    double eps) {
    TORCH_CHECK(
        q.dim() == 3,
        "q must have shape [rows, TP-local heads, 128]");
    const int64_t rows = q.size(0);
    const int64_t heads = q.size(1);
    TORCH_CHECK(rows > 0, "rows must be positive");
    TORCH_CHECK(
        heads > 0 && heads <= kMaxHeads && kMaxHeads % heads == 0,
        "q must have shape [rows, TP-local heads, 128] where heads divide 56");
    check_input_tensor(q, "q", rows, heads);
    check_input_tensor(k, "k", rows, heads);
    TORCH_CHECK(
        q.stride(0) == k.stride(0),
        "q and k must use the same row stride");

    TORCH_CHECK(
        q_weight.device().is_xpu() && k_weight.device().is_xpu() &&
            freqs.device().is_xpu(),
        "q_weight, k_weight, and freqs must be XPU tensors");
    TORCH_CHECK(
        q.device() == k.device() &&
            q.device() == q_weight.device() &&
            q.device() == k_weight.device() &&
            q.device() == freqs.device(),
        "q_weight, k_weight, q, k, and freqs must be on the same XPU");
    TORCH_CHECK(
        q_weight.scalar_type() == torch::kBFloat16 &&
            k_weight.scalar_type() == torch::kBFloat16,
        "q_weight and k_weight must be BF16");
    TORCH_CHECK(
        q_weight.dim() == 1 && q_weight.size(0) == kHeadDim &&
            k_weight.dim() == 1 && k_weight.size(0) == kHeadDim,
        "q_weight and k_weight must have shape [128]");
    TORCH_CHECK(
        q_weight.is_contiguous() && k_weight.is_contiguous(),
        "q_weight and k_weight must be contiguous");
    TORCH_CHECK(
        freqs.scalar_type() == torch::kFloat32,
        "freqs must be FP32");
    TORCH_CHECK(
        freqs.dim() == 2 &&
            freqs.size(0) == rows &&
            freqs.size(1) == kRotaryDim,
        "freqs must have shape [rows, 96]");
    TORCH_CHECK(
        freqs.stride(1) == 1 &&
            freqs.stride(0) >= kRotaryDim,
        "freqs must have a contiguous last dimension");
    const int64_t max_input_offset =
        (rows - 1) * q.stride(0) +
        (heads - 1) * kHeadDim + (kHeadDim - 1);
    const int64_t max_freqs_offset =
        (rows - 1) * freqs.stride(0) + (kRotaryDim - 1);
    TORCH_CHECK(
        max_input_offset <= std::numeric_limits<uint32_t>::max() &&
            max_freqs_offset <= std::numeric_limits<uint32_t>::max() &&
            q.numel() <= std::numeric_limits<uint32_t>::max(),
        "inputs are too large for the MiniMax H3 tuning kernel");
    TORCH_CHECK(
        std::isfinite(eps) && eps > 0.0,
        "eps must be a positive finite value");

    auto q_out = torch::empty(
        {rows, heads, kHeadDim},
        torch::device(q.device()).dtype(q.dtype()));
    auto k_out = torch::empty(
        {rows, heads, kHeadDim},
        torch::device(k.device()).dtype(k.dtype()));
    launch_fused_minimax_h3_qk_rmsnorm_rope(
        q_weight,
        k_weight,
        q,
        k,
        freqs,
        q_out,
        k_out,
        static_cast<float>(eps));
    return {q_out, k_out};
}

}  // namespace norm
}  // namespace omni_xpu
