using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace OnnxStack.Device
{
    public static class DeviceInterop
    {
        [DllImport("OnnxStack.Adapter.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GetAdapters([In, Out] AdapterInfo[] adapterArray);

        [DllImport("OnnxStack.Adapter.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GetAdaptersLegacy([In, Out] AdapterInfo[] adapterArray);

        [return: MarshalAs(UnmanagedType.Bool)]
        [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern bool GlobalMemoryStatusEx(ref MemoryStatusEx lpBuffer);


        public static AdapterInfo[] GetAdapters()
        {
            var adaptersDX = GetAdaptersLegacy();
            var adaptersDXCore = GetAdaptersCore();
            var mergedSet = new HashSet<AdapterInfo>(new LuidComparer());
            for (int i = 0; i < adaptersDX.Length; i++)
            {
                var adapterDX = adaptersDX[i];
                var adapterDXCore = adaptersDXCore.FirstOrDefault(x => x.AdapterLuid == adapterDX.AdapterLuid);
                if (adapterDXCore.DeviceId != 0)
                {
                    adapterDX.Type = adapterDXCore.Type;
                    adapterDX.IsDetachable = adapterDXCore.IsDetachable;
                    adapterDX.IsIntegrated = adapterDXCore.IsIntegrated;
                    adapterDX.IsHardware = adapterDXCore.IsHardware;
                    adapterDX.IsLegacy = adapterDXCore.IsLegacy;
                    mergedSet.Add(adapterDX);
                    continue;
                }
                mergedSet.Add(adapterDX);
            }

            mergedSet.UnionWith(adaptersDXCore);
            return mergedSet.ToArray();
        }


        public static AdapterInfo[] GetAdaptersCore()
        {
            var adapters = new AdapterInfo[20];
            GetAdapters(adapters);

            var uniqueSet = new HashSet<AdapterInfo>(new LuidComparer());
            foreach (var adapter in adapters)
            {
                if (adapter.DeviceId == 0)
                    continue;
                if (adapter.DeviceId == 140 && adapter.VendorId == 5140)
                    continue;

                uniqueSet.Add(adapter);
            }
            return uniqueSet.ToArray();
        }


        public static AdapterInfo[] GetAdaptersLegacy()
        {
            var adapters = new AdapterInfo[20];
            GetAdaptersLegacy(adapters);

            var uniqueSet = new HashSet<AdapterInfo>(new LuidComparer());
            foreach (var adapter in adapters)
            {
                if (adapter.DeviceId == 0)
                    continue;
                if (adapter.DeviceId == 140 && adapter.VendorId == 5140)
                    continue;

                uniqueSet.Add(adapter);
            }
            return uniqueSet.ToArray();
        }


        public static MemoryStatusEx GetMemoryStatus()
        {
            var memStatus = new MemoryStatusEx();
            GlobalMemoryStatusEx(ref memStatus);
            return memStatus;
        }

    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.None)]
    public struct AdapterInfo
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
        public AdapterType Type;

        [MarshalAs(UnmanagedType.I1)]
        public bool IsHardware;

        [MarshalAs(UnmanagedType.I1)]
        public bool IsIntegrated;

        [MarshalAs(UnmanagedType.I1)]
        public bool IsDetachable;

        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string Description;

        [MarshalAs(UnmanagedType.I1)]
        public bool IsLegacy;
    }


    [StructLayout(LayoutKind.Sequential)]
    public struct Luid
    {
        public uint LowPart;
        public int HighPart;

        public override bool Equals(object obj)
        {
            return obj is Luid other && this == other;
        }

        public static bool operator ==(Luid left, Luid right)
        {
            return left.LowPart == right.LowPart && left.HighPart == right.HighPart;
        }


        public static bool operator !=(Luid left, Luid right)
        {
            return !(left == right);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(LowPart, HighPart);
        }
    }

    public class LuidComparer : IEqualityComparer<AdapterInfo>
    {
        public bool Equals(AdapterInfo x, AdapterInfo y)
        {
            return x.AdapterLuid.Equals(y.AdapterLuid);
        }

        public int GetHashCode(AdapterInfo obj)
        {
            return obj.AdapterLuid.GetHashCode();
        }
    }



    [Flags]
    public enum AdapterFlags : uint
    {
        None = 0,
        Remote = 1,
        Software = 2
    }

    public enum AdapterType : uint
    {
        GPU = 0,
        NPU = 1,
        Other = 2
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct MemoryStatusEx
    {
        public uint Length;
        public uint MemoryLoad;
        public ulong TotalPhysicalMemory;
        public ulong AvailablePhysicalMemory;
        public ulong TotalPageFile;
        public ulong AvailPageFile;
        public ulong TotalVirtual;
        public ulong AvailVirtual;
        public ulong AvailExtendedVirtual;

        public MemoryStatusEx()
        {
            Length = (uint)Marshal.SizeOf(typeof(MemoryStatusEx));
        }
    }

}
