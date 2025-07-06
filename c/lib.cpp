/**
 *  @brief  OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
 *  @file   lib.cpp
 *  @author Ash Vardanian
 *  @date   June 27, 2025
 */
#include <fork_union.h>   // C type aliases
#include <fork_union.hpp> // C++ core implementation

#include <variant> // `std::variant`
#include <utility> // `std::in_place_type_t`

namespace fu = ashvardanian::fork_union;

using thread_allocator_t = std::allocator<std::thread>;

using pool_variants_t = std::variant< //

#if _FU_DETECT_ARCH_X86_64
    fu::basic_pool<thread_allocator_t, fu::x86_pause_t>,  //
    fu::basic_pool<thread_allocator_t, fu::x86_tpause_t>, //
#endif
#if _FU_DETECT_ARCH_ARM64
    fu::basic_pool<thread_allocator_t, fu::arm64_yield_t>, //
    fu::basic_pool<thread_allocator_t, fu::arm64_wfet_t>,  //
#endif
#if _FU_DETECT_ARCH_RISC5
    fu::basic_pool<thread_allocator_t, fu::risc5_pause_t>, //
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
    fu::basic_pool<thread_allocator_t, fu::standard_yield_t> //
    >;

struct opaque_pool_t {
    pool_variants_t variants;
    fu_lambda_context_t current_context; /// Current context for the unsafe callbacks
    fu_for_threads_t current_callback;   /// Current callback for the unsafe callbacks

    template <typename pool_type_, typename... args_types_>
    opaque_pool_t(std::in_place_type_t<pool_type_> inplace, args_types_ &&...args) noexcept
        : variants(inplace, std::forward<args_types_>(args)...), current_context(nullptr), current_callback(nullptr) {}

    /** @brief A shim to redirect unsafe callbacks to the current context. */
    void operator()(fu::colocated_thread_t pinned) const noexcept {
        current_callback(current_context, pinned.thread, pinned.colocation);
    }
};

static bool global_initialized {false};
static fu::numa_topology_t global_numa_topology {};
static fu::capabilities_t global_capabilities {fu::capabilities_unknown_k};
static char global_capabilities_string[128] {};

bool globals_initialize(void) {
    if (global_initialized) return true;

#if FU_ENABLE_NUMA
    if (!global_numa_topology.try_harvest()) return false;
#endif

    fu::capabilities_t cpu_caps = fu::cpu_capabilities();
    fu::capabilities_t ram_caps = fu::ram_capabilities();

    global_capabilities = static_cast<fu::capabilities_t>(cpu_caps | ram_caps);
    global_initialized = true;

    // Now, populate the capabilities string:
    char *pos = global_capabilities_string;
    char *end = global_capabilities_string + sizeof(global_capabilities_string) - 1;
    pos += std::snprintf(pos, end - pos, "serial");

    // Start with base capability level
    if (global_capabilities & fu::capability_numa_aware_k) pos += std::snprintf(pos, end - pos, "+numa");
    if (global_capabilities & fu::capability_huge_pages_k) pos += std::snprintf(pos, end - pos, "+hp");
    if (global_capabilities & fu::capability_huge_pages_transparent_k) pos += std::snprintf(pos, end - pos, "+thp");

    // Add CPU-specific extensions
    if (global_capabilities & fu::capability_x86_pause_k) pos += std::snprintf(pos, end - pos, "+x86_pause");
    if (global_capabilities & fu::capability_x86_tpause_k) pos += std::snprintf(pos, end - pos, "+x86_tpause");
    if (global_capabilities & fu::capability_arm64_yield_k) pos += std::snprintf(pos, end - pos, "+arm64_yield");
    if (global_capabilities & fu::capability_arm64_wfet_k) pos += std::snprintf(pos, end - pos, "+arm64_wfet");
    if (global_capabilities & fu::capability_risc5_pause_k) pos += std::snprintf(pos, end - pos, "+risc5_pause");
    return true;
}

extern "C" {

int fu_version_major(void) { return FORK_UNION_VERSION_MAJOR; }
int fu_version_minor(void) { return FORK_UNION_VERSION_MINOR; }
int fu_version_patch(void) { return FORK_UNION_VERSION_PATCH; }
int fu_enabled_numa(void) { return FU_ENABLE_NUMA; }

#pragma region - Metadata

char const *fu_capabilities_string(void) {
    if (!globals_initialize()) return nullptr;
    return &global_capabilities_string[0];
}

size_t fu_count_logical_cores(void) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    return global_numa_topology.threads_count();
#else
    return std::thread::hardware_concurrency();
#endif
}

size_t fu_count_colocations(void) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    return global_numa_topology.nodes_count();
#else
    return 1;
#endif
}

size_t fu_count_numa_nodes(void) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    return global_numa_topology.nodes_count();
#else
    return 1;
#endif
}

size_t fu_count_quality_levels(void) {
    if (!globals_initialize()) return 0;
    return 1; // TODO: One day I'll get some of those weird CPUs to do this
}

size_t fu_volume_huge_pages(size_t numa_node_index) {
#if FU_ENABLE_NUMA
    size_t total_volume = 0;
    auto const &node = global_numa_topology.node(numa_node_index);
    for (auto const &page_size : node.page_sizes) total_volume += page_size.bytes_per_page * page_size.free_pages;
    return total_volume;
#else
    return 0;
#endif
}

size_t fu_volume_any_pages(size_t numa_node_index) {
#if FU_ENABLE_NUMA
    if (!globals_initialize()) return 0;
    if (numa_node_index >= global_numa_topology.nodes_count()) return 0;

    auto const &node = global_numa_topology.node(numa_node_index);
    return node.total_memory_bytes();
#else
    return fu::get_ram_total_volume();
#endif
}

#pragma endregion - Metadata

#pragma region - Memory

void *fu_allocate_at_least(                       //
    size_t numa_node_index, size_t minimum_bytes, //
    size_t *allocated_bytes, size_t *bytes_per_page) {

#if FU_ENABLE_NUMA
    auto const &node = global_numa_topology.node(numa_node_index);
    fu::linux_numa_allocator_t allocator(node.node_id);
    auto result = allocator.allocate_at_least(minimum_bytes);
    if (!result) return nullptr;
    *allocated_bytes = result.count;
    *bytes_per_page = result.bytes_per_page();
    return result.ptr;
#else
    auto result = std::malloc(minimum_bytes);
    if (!result) return nullptr;
    *allocated_bytes = minimum_bytes;
    *bytes_per_page = fu::get_ram_page_size();
    return result;
#endif
}

void *fu_allocate(size_t numa_node_index, size_t bytes) {

#if FU_ENABLE_NUMA
    auto const &node = global_numa_topology.node(numa_node_index);
    fu::linux_numa_allocator_t allocator(node.node_id);
    return allocator.allocate(bytes);
#else
    return std::malloc(bytes);
#endif
}

void fu_free(size_t numa_node_index, void *pointer, size_t bytes) {
#if FU_ENABLE_NUMA
    auto const &node = global_numa_topology.node(numa_node_index);
    fu::linux_numa_allocator_t allocator(node.node_id);
    allocator.deallocate(reinterpret_cast<char *>(pointer), bytes);
#else
    std::free(pointer);
#endif
}

#pragma endregion - Memory

#pragma region - Lifetime

fu_pool_t *fu_pool_new(char const *name) {
    if (!globals_initialize()) return nullptr;

    opaque_pool_t *opaque = static_cast<opaque_pool_t *>(std::malloc(sizeof(opaque_pool_t)));
    if (!opaque) return nullptr;

    // Best case, use the NUMA-aware distributed pool
#if FU_ENABLE_NUMA
    fu::numa_topology_t copied_topology;
    if (!copied_topology.try_assign(global_numa_topology)) {
        std::free(opaque);
        return nullptr;
    }

#if _FU_DETECT_ARCH_X86_64
    if (global_capabilities & fu::capability_x86_tpause_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::linux_distributed_pool<fu::x86_tpause_t>>, name,
                                   std::move(copied_topology));
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
    if (global_capabilities & fu::capability_x86_pause_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::linux_distributed_pool<fu::x86_pause_t>>, name,
                                   std::move(copied_topology));
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#if _FU_DETECT_ARCH_ARM64
    if (global_capabilities & fu::capability_arm64_wfet_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::linux_distributed_pool<fu::arm64_wfet_t>>, name,
                                   std::move(copied_topology));
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
    if (global_capabilities & fu::capability_arm64_yield_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::linux_distributed_pool<fu::arm64_yield_t>>, name,
                                   std::move(copied_topology));
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#if _FU_DETECT_ARCH_RISC5
    if (global_capabilities & fu::capability_risc5_pause_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::linux_distributed_pool<fu::risc5_pause_t>>, name,
                                   std::move(copied_topology));
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#endif // FU_ENABLE_NUMA

    // Common case of using modern hardware, but not having Linux installed
#if _FU_DETECT_ARCH_X86_64
    if (global_capabilities & fu::capability_x86_tpause_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::basic_pool<thread_allocator_t, fu::x86_tpause_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
    if (global_capabilities & fu::capability_x86_pause_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::basic_pool<thread_allocator_t, fu::x86_pause_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#if _FU_DETECT_ARCH_ARM64
    if (global_capabilities & fu::capability_arm64_wfet_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::basic_pool<thread_allocator_t, fu::arm64_wfet_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
    if (global_capabilities & fu::capability_arm64_yield_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::basic_pool<thread_allocator_t, fu::arm64_yield_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif
#if _FU_DETECT_ARCH_RISC5
    if (global_capabilities & fu::capability_risc5_pause_k) {
        new (opaque) opaque_pool_t(std::in_place_type<fu::basic_pool<thread_allocator_t, fu::risc5_pause_t>>);
        return reinterpret_cast<fu_pool_t *>(opaque);
    }
#endif

    // Worst case, use the standard yield pool
    new (opaque) opaque_pool_t(std::in_place_type<fu::basic_pool<thread_allocator_t, fu::standard_yield_t>>);
    return reinterpret_cast<fu_pool_t *>(opaque);
}

void fu_pool_delete(fu_pool_t *pool) {
    assert(pool != nullptr);

    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    std::visit([](auto &variant) { variant.terminate(); }, opaque->variants);

    // Call the object's destructor and deallocate the memory
    opaque->~opaque_pool_t();
    std::free(opaque);
}

fu_bool_t fu_pool_spawn(fu_pool_t *pool, size_t threads, fu_caller_exclusivity_t c_exclusivity) {
    assert(pool != nullptr);
    assert(c_exclusivity == fu_caller_inclusive_k || c_exclusivity == fu_caller_exclusive_k);
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    auto exclusivity = c_exclusivity == fu_caller_inclusive_k ? fu::caller_inclusive_k : fu::caller_exclusive_k;
    return std::visit([=](auto &variant) { return variant.try_spawn(threads, exclusivity); }, opaque->variants);
}

void fu_pool_sleep(fu_pool_t *pool, size_t micros) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    std::visit([=](auto &variant) { variant.sleep(micros); }, opaque->variants);
}

void fu_pool_terminate(fu_pool_t *pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    std::visit([](auto &variant) { variant.terminate(); }, opaque->variants);
}

size_t fu_pool_count_colocations(fu_pool_t *pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    return std::visit([](auto &variant) { return variant.colocations_count(); }, opaque->variants);
}

size_t fu_pool_count_threads(fu_pool_t *pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    return std::visit([](auto &variant) { return variant.threads_count(); }, opaque->variants);
}

size_t fu_pool_count_threads_in(fu_pool_t *pool, size_t colocation_index) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    return std::visit([=](auto &variant) { return variant.threads_count(colocation_index); }, opaque->variants);
}

size_t fu_pool_locate_thread_in(fu_pool_t *pool, size_t global_thread_index, size_t colocation_index) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    return std::visit([=](auto &variant) { 
        return variant.thread_local_index(global_thread_index, colocation_index);
    }, opaque->variants);
}

#pragma endregion - Lifetime

#pragma region - Primary API

void fu_pool_for_threads(fu_pool_t *pool, fu_for_threads_t callback, fu_lambda_context_t context) {
    assert(pool != nullptr && callback != nullptr);
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
    assert(pool != nullptr && callback != nullptr);
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
    assert(pool != nullptr && callback != nullptr);
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
    assert(pool != nullptr && callback != nullptr);
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
    assert(pool != nullptr && callback != nullptr);
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    opaque->current_context = context;
    opaque->current_callback = callback;
    std::visit([&](auto &variant) { variant.unsafe_for_threads(*opaque); }, opaque->variants);
}

void fu_pool_unsafe_join(fu_pool_t *pool) {
    assert(pool != nullptr);
    opaque_pool_t *opaque = reinterpret_cast<opaque_pool_t *>(pool);
    assert(opaque->current_context != nullptr);
    std::visit([](auto &variant) { variant.unsafe_join(); }, opaque->variants);
    opaque->current_context = nullptr;
    opaque->current_callback = nullptr;
}

#pragma endregion - Flexible API
}
