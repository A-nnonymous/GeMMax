#ifndef UTILS_INL
#define UTILS_INL

template <typename T>
T* copy_to_device(const std::vector<T>& host_data) {
    if (host_data.empty()) {
        return nullptr;
    }

    T* device_ptr;
    const size_t size = host_data.size() * sizeof(T);
    CHECK_CUDA(cuMemAlloc(reinterpret_cast<void**>(&device_ptr), size));
    CHECK_CUDA(cuMemcpyHtoD(
        reinterpret_cast<CUdeviceptr>(device_ptr),
        host_data.data(),
        size
    ));

    return device_ptr;
}

template <typename T>
std::vector<T> copy_from_device(const T* device_data, size_t count) {
    if (count == 0 || device_data == nullptr) {
        return {};
    }

    std::vector<T> host_data(count);
    const size_t size = count * sizeof(T);
    CHECK_CUDA(cuMemcpyDtoH(
        host_data.data(),
        reinterpret_cast<CUdeviceptr>(device_data),
        size
    ));

    return host_data;
}

template <typename T>
void free_device_memory(T* device_ptr) {
    if (device_ptr != nullptr) {
        cuMemFree(reinterpret_cast<CUdeviceptr>(device_ptr));
    }
}

#endif // UTILS_INL
