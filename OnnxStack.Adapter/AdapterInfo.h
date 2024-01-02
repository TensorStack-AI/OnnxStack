#pragma once

struct AdapterInfo
{
    UINT Id;
    UINT VendorId;
    UINT DeviceId;
    UINT SubSysId;
    UINT Revision;
    SIZE_T DedicatedVideoMemory;
    SIZE_T DedicatedSystemMemory;
    SIZE_T SharedSystemMemory;
    LUID AdapterLuid;
    WCHAR Description[128];
};


extern "C" __declspec(dllexport) int __cdecl GetAdapter(bool preferHighPerformance);

extern "C" __declspec(dllexport) void __cdecl GetAdapters(AdapterInfo * adapterArray);