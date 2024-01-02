#include <dxgi1_6.h>
#include <wrl/client.h>
#include <vector>
#include "AdapterInfo.h"

#pragma comment(lib, "dxguid.lib")

#define RETURN_IF_FAILED(hr) if (FAILED((hr))) { return 0; }

using Microsoft::WRL::ComPtr;

class DxgiModule
{
    using CreateFactoryFunc = decltype(CreateDXGIFactory);
    HMODULE m_module = nullptr;
    CreateFactoryFunc* m_createFactoryFunc = nullptr;

public:
    DxgiModule()
    {
        m_module = LoadLibraryA("dxgi.dll");
        if (m_module)
        {
            auto funcAddr = GetProcAddress(m_module, "CreateDXGIFactory");
            if (funcAddr)
            {
                m_createFactoryFunc = reinterpret_cast<CreateFactoryFunc*>(funcAddr);
            }
        }
    }
    ~DxgiModule() { if (m_module) { FreeModule(m_module); } }

    ComPtr<IDXGIFactory6> CreateFactory()
    {
        ComPtr<IDXGIFactory6> adapterFactory;
        m_createFactoryFunc(IID_PPV_ARGS(&adapterFactory));
        return adapterFactory;
    }
};

extern "C" __declspec(dllexport) int __cdecl GetAdapter(bool preferHighPerformance)
{
    DxgiModule dxgi;

    ComPtr<IDXGIFactory6> factory = dxgi.CreateFactory();;
    if (!factory)
    {
        return 0;
    }

    ComPtr<IDXGIAdapter1> adapter;

    // Store LUIDs for hardware adapters in original unsorted order.
    std::vector<std::pair<int, LUID>> adaptersUnsortedIndexAndLuid;
    for (int i = 0; factory->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf()) != DXGI_ERROR_NOT_FOUND; i++)
    {
        DXGI_ADAPTER_DESC desc = {};
        RETURN_IF_FAILED(adapter->GetDesc(&desc));
        adaptersUnsortedIndexAndLuid.emplace_back(i, desc.AdapterLuid);
    }

    // Find the first adapter meeting GPU preference.
    DXGI_ADAPTER_DESC preferredAdapterDesc = {};
    {
        DXGI_GPU_PREFERENCE gpuPreference = preferHighPerformance ? DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE : DXGI_GPU_PREFERENCE_MINIMUM_POWER;
        RETURN_IF_FAILED(factory->EnumAdapterByGpuPreference(0, gpuPreference, IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf())));
        RETURN_IF_FAILED(adapter->GetDesc(&preferredAdapterDesc));
    }

    // Look up the index of the preferred adapter in the unsorted list order.
    for (auto& hardwareAdapterEntry : adaptersUnsortedIndexAndLuid)
    {
        if (hardwareAdapterEntry.second.HighPart == preferredAdapterDesc.AdapterLuid.HighPart &&
            hardwareAdapterEntry.second.LowPart == preferredAdapterDesc.AdapterLuid.LowPart)
        {
            return hardwareAdapterEntry.first;
        }
    }

    return 0;
}


extern "C" __declspec(dllexport) void __cdecl GetAdapters(AdapterInfo * adapterArray)
{
    DxgiModule dxgi;

    ComPtr<IDXGIFactory6> factory = dxgi.CreateFactory();;
    if (!factory)
        return;

    int adapterCount = 0;
    ComPtr<IDXGIAdapter1> adapter;
    for (int i = 0; factory->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf()) != DXGI_ERROR_NOT_FOUND; i++)
    {
        DXGI_ADAPTER_DESC desc = {};
        HRESULT hr = adapter->GetDesc(&desc);
        if (SUCCEEDED(hr))
        {
            AdapterInfo info{};
            info.Id = adapterCount;
            info.AdapterLuid = desc.AdapterLuid;
            info.DedicatedSystemMemory = desc.DedicatedSystemMemory;
            info.DedicatedVideoMemory = desc.DedicatedVideoMemory;
            info.DeviceId = desc.DeviceId;
            info.Revision = desc.Revision;
            info.SharedSystemMemory = desc.SharedSystemMemory;
            info.SubSysId = desc.SubSysId;
            info.VendorId = desc.VendorId;
            wcsncpy_s(info.Description, desc.Description, _TRUNCATE);
            adapterArray[adapterCount] = info;
            ++adapterCount;
        }
    }
}
