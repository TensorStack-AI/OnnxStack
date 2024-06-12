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


extern "C" __declspec(dllexport) void __cdecl GetAdapters(AdapterInfo * adapterArray)
{
    DxgiModule dxgi;
    ComPtr<IDXGIFactory6> factory = dxgi.CreateFactory();
    if (!factory)
        return;

    int adapterCount = 0;
    ComPtr<IDXGIAdapter1> adapter;
    for (int i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf())) != DXGI_ERROR_NOT_FOUND; i++)
    {
        DXGI_ADAPTER_DESC1 desc = {};
        HRESULT hr = adapter->GetDesc1(&desc);
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
            info.Flags = desc.Flags;
            wcsncpy_s(info.Description, desc.Description, _TRUNCATE);
            adapterArray[adapterCount] = info;
            ++adapterCount;
        }
    }
}
