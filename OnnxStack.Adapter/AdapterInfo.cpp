#include <d3d12.h>
#include <directml.h>
#include <dxgi1_6.h>
#include <dxcore.h>
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


const GUID DXCORE_HARDWARE_TYPE_ATTRIBUTE_GPU = { 0xb69eb219, 0x3ded, 0x4464, {0x97, 0x9f, 0xa0, 0xb, 0xd4, 0x68, 0x70, 0x6 } };
const GUID DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU = { 0xd46140c4, 0xadd7, 0x451b, {0x9e, 0x56, 0x6, 0xfe, 0x8c, 0x3b, 0x58, 0xed } };


static AdapterInfo CreateDXCoreInfo(ComPtr<IDXCoreAdapter> adapter, UINT index, UINT deviceType)
{
    AdapterInfo info{};
    DXCoreHardwareIDParts hardwareID;
    adapter->GetProperty(DXCoreAdapterProperty::InstanceLuid, sizeof(LUID), &info.AdapterLuid);
    adapter->GetProperty(DXCoreAdapterProperty::DedicatedAdapterMemory, sizeof(SIZE_T), &info.DedicatedVideoMemory);
    adapter->GetProperty(DXCoreAdapterProperty::DedicatedSystemMemory, sizeof(SIZE_T), &info.DedicatedSystemMemory);
    adapter->GetProperty(DXCoreAdapterProperty::SharedSystemMemory, sizeof(SIZE_T), &info.SharedSystemMemory);
    adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, sizeof(info.Description), &info.Description);
    adapter->GetProperty(DXCoreAdapterProperty::IsHardware, sizeof(BOOLEAN), &info.IsHardware);
    adapter->GetProperty(DXCoreAdapterProperty::IsIntegrated, sizeof(BOOLEAN), &info.IsIntegrated);
    adapter->GetProperty(DXCoreAdapterProperty::IsDetachable, sizeof(BOOLEAN), &info.IsDetachable);
    adapter->GetProperty(DXCoreAdapterProperty::HardwareIDParts, sizeof(DXCoreHardwareIDParts), &hardwareID);
    info.Id = index;
    info.DeviceType = deviceType;
    info.DeviceId = hardwareID.deviceID;
    info.Revision = hardwareID.revisionID;
    info.SubSysId = hardwareID.subSystemID;
    info.VendorId = hardwareID.vendorID;
    info.IsLegacy = false;
    return info;
}


static void GetDXGIAdapters(AdapterInfo* adapterArray)
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
            info.IsDetachable = 0;
            info.IsIntegrated = 0;
            info.IsHardware = desc.Flags == 0;
            info.DeviceType = 0; // GPU
            info.IsLegacy = true;
            WideCharToMultiByte(CP_ACP, 0, desc.Description, -1, info.Description, sizeof(info.Description), nullptr, nullptr);
            adapterArray[adapterCount] = info;
            ++adapterCount;
        }
    }
}


static void GetDXCoreAdapters(AdapterInfo* adapterArray)
{
    // Adapter Factory
    ComPtr<IDXCoreAdapterFactory1> dxCoreFactory;
    if (FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(&dxCoreFactory)))) {
        // Failed to create DXCoreAdapterFactory
        // Try IDXGIFactory6 method
        return GetDXGIAdapters(adapterArray);
    }

    // GPU Adapters
    ComPtr<IDXCoreAdapterList> gpuAdapterList;
    if (FAILED(dxCoreFactory->CreateAdapterList(1, &DXCORE_HARDWARE_TYPE_ATTRIBUTE_GPU, IID_PPV_ARGS(&gpuAdapterList)))) {
        // Failed to create GPU adapter list
        // Try IDXGIFactory6 method
        return GetDXGIAdapters(adapterArray);
    }

    // NPU Adapters
    ComPtr<IDXCoreAdapterList> npuAdapterList;
    if (FAILED(dxCoreFactory->CreateAdapterList(1, &DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU, IID_PPV_ARGS(&npuAdapterList)))) {
        return; // Failed to create NPU adapter list
    }

    // Sort Adapters
    DXCoreAdapterPreference preference = DXCoreAdapterPreference::HighPerformance;
    gpuAdapterList->Sort(1, &preference);
    npuAdapterList->Sort(1, &preference);

    // Get the number of adapters
    uint32_t gpuCount = gpuAdapterList->GetAdapterCount();
    uint32_t npuCount = npuAdapterList->GetAdapterCount();

    // Create GPU Info
    for (uint32_t i = 0; i < gpuCount; i++) {
        ComPtr<IDXCoreAdapter> adapter;
        if (FAILED(gpuAdapterList->GetAdapter(i, IID_PPV_ARGS(&adapter)))) {
            return; // Failed to create GPU adapter
        }
        adapterArray[i] = CreateDXCoreInfo(adapter, i, 0);
    }

    // Create NPU Info
    for (uint32_t i = 0; i < npuCount; i++) {
        ComPtr<IDXCoreAdapter> adapter;
        if (FAILED(npuAdapterList->GetAdapter(i, IID_PPV_ARGS(&adapter)))) {
            return; // Failed to create NPU adapter
        }
        adapterArray[gpuCount + i] = CreateDXCoreInfo(adapter, i, 1);
    }
}


extern "C" __declspec(dllexport) void __cdecl GetAdapters(AdapterInfo* adapterArray)
{
    // IDXCoreAdapterFactory1
    // Fallback: IDXGIFactory6
    return GetDXCoreAdapters(adapterArray);
}

extern "C" __declspec(dllexport) void __cdecl GetAdaptersLegacy(AdapterInfo* adapterArray)
{
    // IDXGIFactory6
    return GetDXGIAdapters(adapterArray);
}