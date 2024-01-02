using Microsoft.Extensions.Logging;
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
        private readonly ILogger<DeviceService> _logger;
        private IReadOnlyList<DeviceInfo> _devices;

        /// <summary>
        /// Initializes a new instance of the <see cref="DeviceService"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public DeviceService(ILogger<DeviceService> logger)
        {
            _logger = logger;
            _devices = GetDevices();
        }

        /// <summary>
        /// Gets the devices.
        /// </summary>
        public IReadOnlyList<DeviceInfo> Devices => _devices;


        /// <summary>
        /// Gets the devices.
        /// </summary>
        /// <returns></returns>
        private IReadOnlyList<DeviceInfo> GetDevices()
        {
            _logger.LogInformation("[GetDevices] - Query Devices...");
            var devices = new List<DeviceInfo> { new DeviceInfo("CPU", 0, 0) };

            try
            {
                var adapters = new AdapterInfo[10];
                AdapterInterop.GetAdapters(adapters);
                devices.AddRange(adapters
                    .Where(x => x.DedicatedVideoMemory > 0)
                    .Select(GetDeviceInfo)
                    .ToList());
                devices.ForEach(x => _logger.LogInformation($"[GetDevices] - Found Device: {x.Name}, DeviceId: {x.DeviceId}"));
            }
            catch (Exception ex)
            {
                devices.Add(new DeviceInfo("GPU0", 0, 0));
                devices.Add(new DeviceInfo("GPU1", 1, 0));
                _logger.LogError($"[GetDevices] - Failed to query devices, {ex.Message}");
            }

            _logger.LogInformation($"[GetDevices] - Query devices complete, Devices Found: {devices.Count}");
            return devices;
        }


        /// <summary>
        /// Gets the device information.
        /// </summary>
        /// <param name="adapter">The adapter.</param>
        /// <returns></returns>
        private static DeviceInfo GetDeviceInfo(AdapterInfo adapter)
        {
            string description;
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
