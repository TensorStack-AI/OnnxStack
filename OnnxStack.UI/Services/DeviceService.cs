using OnnxStack.UI.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace OnnxStack.UI.Services
{
    public class DeviceService : IDeviceService
    {
        private IReadOnlyList<DeviceInfo> _devices;

        public DeviceService()
        {
            _devices = GetDevices();
        }

        public IReadOnlyList<DeviceInfo> Devices => _devices;

        private static IReadOnlyList<DeviceInfo> GetDevices()
        {
            var devices = new List<DeviceInfo> { new DeviceInfo("CPU", -1, 0) };

            try
            {
                var adapters = new AdapterInfo[10];
                AdapterInterop.GetAdapters(adapters);
                devices.AddRange(adapters
                    .Where(x => x.DeviceId > 0)
                    .Select(GetDeviceInfo)
                    .ToList());
            }
            catch (Exception ex)
            {
                devices.Add(new DeviceInfo("GPU0", 0, 0));
                devices.Add(new DeviceInfo("GPU1", 1, 0));
            }
            return devices;
        }

        private static DeviceInfo GetDeviceInfo(AdapterInfo adapter)
        {
            string description = string.Empty;
            unsafe
            {
                description = new string(adapter.Description);
            }
            var deviceId = (int)adapter.Id;
            var vram = (int)(adapter.DedicatedVideoMemory / 1024 / 1024);
            return new DeviceInfo(description, deviceId, vram);
        }
    }

    public static partial class AdapterInterop
    {
        [LibraryImport("OnnxStack.Adapter.dll")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(CallConvCdecl) })]
        public static partial int GetAdapter([MarshalAs(UnmanagedType.Bool)] bool preferHighPerformance);

        [LibraryImport("OnnxStack.Adapter.dll")]
        [UnmanagedCallConv(CallConvs = new Type[] { typeof(CallConvCdecl) })]
        public static partial void GetAdapters(AdapterInfo[] adapterArray);
    }

    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct AdapterInfo
    {
        public uint Id;
        public uint VendorId;
        public uint DeviceId;
        public uint SubSysId;
        public uint Revision;
        public ulong DedicatedVideoMemory;
        public ulong DedicatedSystemMemory;
        public ulong SharedSystemMemory;
        public Luid AdapterLuid;
        public fixed char Description[128];
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Luid
    {
        public uint LowPart;
        public int HighPart;
    }
}
