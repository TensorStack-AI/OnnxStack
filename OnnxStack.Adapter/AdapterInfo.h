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
    UINT DeviceType;
    BOOLEAN IsHardware;
    BOOLEAN IsIntegrated;
    BOOLEAN IsDetachable;
    CHAR Description[128];
    BOOLEAN IsLegacy;
};


extern "C" __declspec(dllexport) void __cdecl GetAdapters(AdapterInfo * adapterArray);

extern "C" __declspec(dllexport) void __cdecl GetAdaptersLegacy(AdapterInfo* adapterArray);