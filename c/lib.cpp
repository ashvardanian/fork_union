/**
 *  @brief  OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
 *  @file   lib.cpp
 *  @author Ash Vardanian
 *  @date   June 27, 2025
 */
#include <fork_union.h>
#include <fork_union.hpp>

#include <variant>

namespace fu = ashvardanian::fork_union;

using pool_variants_t = std::variant< //

#if _FU_DETECT_ARCH_X86_64
    fu::basic_pool<std::allocator<std::thread>, fu::x86_pause_t>,  //
    fu::basic_pool<std::allocator<std::thread>, fu::x86_tpause_t>, //
#endif
#if _FU_DETECT_ARCH_ARM64
    fu::basic_pool<std::allocator<std::thread>, fu::arm64_yield_t>, //
    fu::basic_pool<std::allocator<std::thread>, fu::arm64_wfet_t>,  //
#endif
#if _FU_DETECT_ARCH_RISC5
    fu::basic_pool<std::allocator<std::thread>, fu::risc5_pause_t>, //
#endif

#if FU_ENABLE_NUMA
    fu::linux_distributed_pool<fu::standard_yield_t>, //
#if _FU_DETECT_ARCH_X86_64
    fu::linux_distributed_pool<fu::x86_pause_t>,  //
    fu::linux_distributed_pool<fu::x86_tpause_t>, //
#endif
#if _FU_DETECT_ARCH_ARM64
    fu::linux_distributed_pool<fu::arm64_yield_t>, //
    fu::linux_distributed_pool<fu::arm64_wfet_t>,  //
#endif
#if _FU_DETECT_ARCH_RISC5
    fu::linux_distributed_pool<fu::risc5_pause_t>, //
#endif
#endif
    fu::basic_pool<std::allocator<std::thread>, fu::standard_yield_t> //
    >;

struct opaque_pool_t {
    pool_variants_t variants;

    fu_lambda_context_t current_context = nullptr; // Current context for the unsafe callbacks
    union {
        fu_for_threads_t for_threads; // Callback for `unsafe_for_threads`
        fu_for_prongs_t for_prongs;   // Callback for `unsafe_for_n`
        fu_for_slices_t for_slices;   // Callback for `unsafe_for_slices
    } current_callback;

    void operator()(fu::colocated_thread_t pinned) const noexcept {
        current_callback.for_threads(current_context, pinned.thread, pinned.colocation);
    }

    void operator()(fu::colocated_prong_t prong) const noexcept {
        current_callback.for_prongs(current_context, prong.task, prong.thread, prong.colocation);
    }

    void operator()(fu::colocated_prong_t prong, std::size_t count) const noexcept {
        current_callback.for_slices(current_context, prong.task, count, prong.thread, prong.colocation);
    }
};

#if FU_ENABLE_NUMA
static fu::ram_page_settings_t global_ram_page_settings;
static fu::numa_topology_t global_numa_topology;
#endif

extern "C" {
#pragma region - Metadata

char const *fu_capabilities_string() { return nullptr; }

size_t fu_count_logical_cores() { return global_numa_topology.threads_count(); }

#pragma endregion - Metadata

#pragma region - Lifetime

fu_pool_t *fu_pool_new() { return nullptr; }
void fu_pool_delete(fu_pool_t *pool) {}
fu_bool_t fu_pool_spawn(fu_pool_t *pool, size_t threads, fu_caller_exclusivity_t exclusivity) { return 0; }

void fu_pool_sleep(fu_pool_t *pool, size_t micros) {
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    std::visit([=](auto &variant) { variant.sleep(micros); }, opaque->variants);
}

void fu_pool_terminate(fu_pool_t *pool) {
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    std::visit([](auto &variant) { variant.terminate(); }, opaque->variants);
}

#pragma endregion - Lifetime

#pragma region - Primary API

void fu_pool_for_threads(fu_pool_t *pool, fu_for_threads_t callback, fu_lambda_context_t context) {
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    std::visit(
        [&](auto &variant) {
            variant.for_threads([=](fu::colocated_thread_t pinned) noexcept { //
                callback(context, pinned.thread, pinned.colocation);
            });
        },
        opaque->variants);
}

void fu_pool_for_n(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context) {
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    std::visit(
        [&](auto &variant) {
            variant.for_n(n, [=](fu::colocated_prong_t prong) noexcept { //
                callback(context, prong.task, prong.thread, prong.colocation);
            });
        },
        opaque->variants);
}

void fu_pool_for_n_dynamic(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context) {
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    std::visit(
        [&](auto &variant) {
            variant.for_n_dynamic(n, [=](fu::colocated_prong_t prong) noexcept { //
                callback(context, prong.task, prong.thread, prong.colocation);
            });
        },
        opaque->variants);
}

void fu_pool_for_slices(fu_pool_t *pool, size_t n, fu_for_slices_t callback, fu_lambda_context_t context) {
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    std::visit(
        [&](auto &variant) {
            variant.for_slices(n, [=](fu::colocated_prong_t prong, std::size_t count) noexcept { //
                callback(context, prong.task, count, prong.thread, prong.colocation);
            });
        },
        opaque->variants);
}

#pragma endregion - Primary API

#pragma region - Flexible API

void fu_pool_unsafe_for_threads(fu_pool_t *pool, fu_for_threads_t callback, fu_lambda_context_t context) {
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    opaque->current_context = context;
    opaque->current_callback.for_threads = callback;
    std::visit([&](auto &variant) { variant.unsafe_for_threads(*opaque); }, opaque->variants);
}

void fu_pool_unsafe_for_n(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context) {
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    opaque->current_context = context;
    opaque->current_callback.for_prongs = callback;
    std::visit([&](auto &variant) { variant.unsafe_for_n(n, *opaque); }, opaque->variants);
}

void fu_pool_unsafe_for_n_dynamic(fu_pool_t *pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context) {
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    opaque->current_context = context;
    opaque->current_callback.for_prongs = callback;
    std::visit([&](auto &variant) { variant.unsafe_for_n_dynamic(n, *opaque); }, opaque->variants);
}

void fu_pool_unsafe_for_slices(fu_pool_t *pool, size_t n, fu_for_slices_t callback, fu_lambda_context_t context) {
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    opaque->current_context = context;
    opaque->current_callback.for_slices = callback;
    std::visit([&](auto &variant) { variant.unsafe_for_slices(n, *opaque); }, opaque->variants);
}

void fu_pool_unsafe_join(fu_pool_t *pool) {
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    std::visit([](auto &variant) { variant.unsafe_join(); }, opaque->variants);
    opaque->current_context = nullptr;
    opaque->current_callback.for_threads = nullptr;
}

#pragma endregion - Flexible API
}
