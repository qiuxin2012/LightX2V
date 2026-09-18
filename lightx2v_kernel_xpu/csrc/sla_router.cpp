#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#include <c10/xpu/XPUStream.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace {

constexpr int kHeadDim = 128;
constexpr int kBlock = 128;
constexpr int kSortSize = 512;

class SlaPoolQKKernel;
class SlaScoreTopKXmxKernel;

void launch_pool_qk(const bf16* q, const bf16* k, bf16* pooled_q,
                    bf16* pooled_k, int64_t batch, int64_t length,
                    int64_t heads, int64_t blocks,
                    const c10::Device& device) {
    const int64_t groups = batch * heads * kSortSize;
    auto& queue = c10::xpu::getCurrentXPUStream(device.index()).queue();
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for<SlaPoolQKKernel>(
            sycl::nd_range<1>(groups, 1),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t linear = item.get_global_linear_id();
                const int64_t block = linear % kSortSize;
                const int64_t head = (linear / kSortSize) % heads;
                const int64_t batch_idx = linear / (kSortSize * heads);
                const int64_t output_offset = linear * kHeadDim;
                if (block >= blocks) {
                    block_store<bf16, kHeadDim>(pooled_q + output_offset,
                                                simd<bf16, kHeadDim>(0));
                    block_store<bf16, kHeadDim>(pooled_k + output_offset,
                                                simd<bf16, kHeadDim>(0));
                    return;
                }
                const int64_t begin = block * kBlock;
                const int64_t end = begin + kBlock < length ? begin + kBlock : length;
                simd<float, kHeadDim> q_sum = 0.0f;
                simd<float, kHeadDim> k_sum = 0.0f;
                for (int64_t token = begin; token < end; ++token) {
                    const int64_t offset =
                        ((batch_idx * length + token) * heads + head) *
                        kHeadDim;
                    q_sum += block_load<bf16, kHeadDim>(q + offset);
                    k_sum += block_load<bf16, kHeadDim>(k + offset);
                }
                const float inverse = 1.0f / static_cast<float>(end - begin);
                block_store<bf16, kHeadDim>(pooled_q + output_offset,
                                            simd<bf16, kHeadDim>(q_sum * inverse));
                block_store<bf16, kHeadDim>(pooled_k + output_offset,
                                            simd<bf16, kHeadDim>(k_sum * inverse));
            });
    });
}

void launch_score_topk(const bf16* pooled_q, const bf16* pooled_k,
                       int32_t* lut, int64_t batch, int64_t heads,
                       int64_t blocks, int64_t topk,
                       const c10::Device& device) {
    constexpr int kQueriesPerGroup = 16;
    constexpr int kKeysPerTile = 8;
    constexpr int kWorkgroup = 16;
    constexpr uint32_t kSlmBytes = kQueriesPerGroup * kSortSize * sizeof(float);
    const int64_t query_groups = (blocks + kQueriesPerGroup - 1) / kQueriesPerGroup;
    const int64_t groups = batch * heads * query_groups;
    auto& queue = c10::xpu::getCurrentXPUStream(device.index()).queue();
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for<SlaScoreTopKXmxKernel>(
            sycl::nd_range<1>(groups * kWorkgroup, kWorkgroup),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                slm_init(kSlmBytes);
                const int64_t group = item.get_group_linear_id();
                const int lid = item.get_local_linear_id();
                const int64_t query_group = group % query_groups;
                const int64_t bh = group / query_groups;
                const int query_start = query_group * kQueriesPerGroup;
                const bf16* q_base = pooled_q + bh * kSortSize * kHeadDim;
                const bf16* k_base = pooled_k + bh * kSortSize * kHeadDim;
                simd<float, 128> accum;

                slm_block_store<float, kSortSize>(
                    lid * kSortSize * sizeof(float),
                    simd<float, kSortSize>(
                        -std::numeric_limits<float>::infinity()));
                barrier();

                for (int key_tile = lid; key_tile < kSortSize / kKeysPerTile;
                     key_tile += kWorkgroup) {
                    accum = 0.0f;
                    const int key_start = key_tile * kKeysPerTile;
#pragma unroll
                    for (int chunk = 0; chunk < kHeadDim / 16; ++chunk) {
                        sycl::ext::intel::experimental::esimd::config_2d_mem_access<
                            uint32_t, 8, 16, 1> q_payload(
                            reinterpret_cast<const uint32_t*>(q_base),
                            kHeadDim * sizeof(bf16) - 1, kSortSize - 1,
                            kHeadDim * sizeof(bf16) - 1, chunk * 8, query_start);
                        simd<uint32_t, 128> q_packed =
                            sycl::ext::intel::experimental::esimd::lsc_load_2d<
                                uint32_t, 8, 16, 1, true, false>(q_payload);
                        simd<bf16, 256> q_tile = q_packed.bit_cast_view<bf16>();
                        sycl::ext::intel::experimental::esimd::config_2d_mem_access<
                            bf16, 16, 8, 1> k_payload(
                            k_base, kHeadDim * sizeof(bf16) - 1, kSortSize - 1,
                            kHeadDim * sizeof(bf16) - 1, chunk * 16, key_start);
                        simd<bf16, 128> k_tile =
                            sycl::ext::intel::experimental::esimd::lsc_load_2d<
                                bf16, 16, 8, 1, false, false>(k_payload);
                        accum = sycl::ext::intel::esimd::xmx::dpas<
                            8, 8, float, float, bf16, bf16>(accum, q_tile, k_tile);
                    }
                    simd<uint32_t, 128> lanes(0, 1);
                    simd<uint32_t, 128> offsets =
                        ((lanes % 16) * kSortSize + key_start + lanes / 16) *
                        sizeof(float);
                    simd_mask<128> valid =
                        (query_start + lanes % 16 < blocks) &
                        (key_start + lanes / 16 < blocks);
                    accum.merge(-std::numeric_limits<float>::infinity(), !valid);
                    slm_scatter<float, 128>(offsets, accum, valid);
                }
                barrier();

                const int query_block = query_start + lid;
                if (query_block < blocks) {
                    simd<float, kSortSize> scores =
                        slm_block_load<float, kSortSize>(lid * kSortSize * sizeof(float));
                    simd<uint32_t, kSortSize> positions(0, 1);
                    for (int64_t slot = 0; slot < topk; ++slot) {
                        const float maximum = hmax<float>(scores);
                        simd<uint32_t, kSortSize> candidates = positions;
                        candidates.merge(0xffffffffu, scores != maximum);
                        const uint32_t selected = hmin<uint32_t>(candidates);
                        block_store<int32_t, 1>(
                            lut + (bh * blocks + query_block) * topk + slot,
                            simd<int32_t, 1>(static_cast<int32_t>(selected)));
                        scores.merge(-std::numeric_limits<float>::infinity(),
                                     positions == selected);
                    }
                }
            });
    });
}

}  // namespace

torch::Tensor sla_route_xpu(const torch::Tensor& q, const torch::Tensor& k,
                            int64_t topk) {
    TORCH_CHECK(q.is_xpu() && k.is_xpu(), "SLA router requires XPU tensors");
    TORCH_CHECK(q.device() == k.device(), "Q and K must be on the same XPU");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
                    k.scalar_type() == torch::kBFloat16,
                "SLA router requires BF16 Q and K");
    TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes(),
                "Q and K must have matching [B,L,H,128] shapes");
    TORCH_CHECK(q.size(3) == kHeadDim, "SLA router head dimension must be 128");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(),
                "SLA router requires contiguous Q and K");
    const int64_t batch = q.size(0);
    const int64_t length = q.size(1);
    const int64_t heads = q.size(2);
    const int64_t blocks = (length + kBlock - 1) / kBlock;
    TORCH_CHECK(batch > 0 && length > 0 && heads > 0,
                "SLA router dimensions must be positive");
    TORCH_CHECK(blocks <= kSortSize,
                "SLA router supports at most 512 sequence blocks");
    TORCH_CHECK(topk > 0 && topk <= blocks,
                "SLA router topk must be in [1, number of blocks]");

    const auto pooled_shape = std::vector<int64_t>{batch, heads, kSortSize, kHeadDim};
    auto pooled_q = torch::empty(pooled_shape, q.options());
    auto pooled_k = torch::empty(pooled_shape, q.options());
    auto lut = torch::empty({batch, heads, blocks, topk},
                            q.options().dtype(torch::kInt32));
    launch_pool_qk(reinterpret_cast<const bf16*>(q.data_ptr()),
                   reinterpret_cast<const bf16*>(k.data_ptr()),
                   reinterpret_cast<bf16*>(pooled_q.data_ptr()),
                   reinterpret_cast<bf16*>(pooled_k.data_ptr()),
                   batch, length, heads, blocks, q.device());
    launch_score_topk(reinterpret_cast<const bf16*>(pooled_q.data_ptr()),
                      reinterpret_cast<const bf16*>(pooled_k.data_ptr()),
                      lut.data_ptr<int32_t>(), batch, heads, blocks, topk,
                      q.device());
    return lut;
}

TORCH_LIBRARY(sycl_kernels_sla_router, m) {
    m.def("route(Tensor q, Tensor k, int topk) -> Tensor");
}

TORCH_LIBRARY_IMPL(sycl_kernels_sla_router, XPU, m) {
    m.impl("route", &sla_route_xpu);
}
