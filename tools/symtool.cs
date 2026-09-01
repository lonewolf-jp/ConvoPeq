using System;
using System.Runtime.InteropServices;
using System.Text;

class SymTool {
    [DllImport("dbghelp.dll", SetLastError=true, CharSet=CharSet.Ansi)]
    static extern bool SymInitialize(IntPtr hProc, string userPath, bool fInvade);
    [DllImport("dbghelp.dll", SetLastError=true)]
    static extern uint SymSetOptions(uint opts);
    [DllImport("dbghelp.dll", SetLastError=true, CharSet=CharSet.Ansi)]
    static extern ulong SymLoadModuleEx(IntPtr hProc, IntPtr hFile, string ImageName, string ModuleName, ulong DllBase, uint DllSize, IntPtr Data, uint Flags);

    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Ansi)]
    struct SYMBOL_INFO {
        public uint SizeOfStruct;
        public uint TypeIndex;
        public ulong Reserved1, Reserved2;
        public uint Index, Size;
        public ulong ModBase;
        public uint Flags;
        public ulong Value, Address;
        public uint Register, Scope, Tag, NameLen, Reserved3;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst=1024)]
        public string Name;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct IMAGEHLP_LINE64 {
        public uint SizeOfStruct;
        public uint Key;
        public uint LineNumber;
        public IntPtr FileName;
        public uint AddressOffset;
    }

    [DllImport("dbghelp.dll", SetLastError=true)]
    static extern bool SymFromAddr(IntPtr hProc, ulong addr, out ulong disp, ref SYMBOL_INFO sym);
    [DllImport("dbghelp.dll", SetLastError=true)]
    static extern bool SymGetLineFromAddr64(IntPtr hProc, ulong addr, out uint disp, ref IMAGEHLP_LINE64 line);

    static int Main(string[] args) {
        string exe = args[0];
        SymSetOptions(0x00000010 | 0x00000004 | 0x00000020); // UNDNAME | EXECUTABLE | LINE_NUMBERS
        IntPtr h = System.Diagnostics.Process.GetCurrentProcess().Handle;
        if (!SymInitialize(h, System.IO.Path.GetDirectoryName(exe), false)) { Console.WriteLine("SymInit fail " + Marshal.GetLastWin32Error()); return 1; }
        ulong baseAddr = SymLoadModuleEx(h, IntPtr.Zero, exe, null, 0, 0, IntPtr.Zero, 0);
        if (baseAddr == 0) { Console.WriteLine("LoadModule fail " + Marshal.GetLastWin32Error()); return 1; }
        Console.WriteLine("modbase=0x" + baseAddr.ToString("x"));
        for (int i = 1; i < args.Length; i++) {
            ulong rva = Convert.ToUInt64(args[i], 16);
            var si = new SYMBOL_INFO();
            si.SizeOfStruct = (uint)Marshal.SizeOf(typeof(SYMBOL_INFO)) - 1024 + 1;
            ulong disp;
            string name = "(nofound:" + Marshal.GetLastWin32Error() + ")";
            if (SymFromAddr(h, baseAddr + rva, out disp, ref si)) name = si.Name + "+0x" + disp.ToString("x");
            var ln = new IMAGEHLP_LINE64();
            ln.SizeOfStruct = (uint)Marshal.SizeOf(typeof(IMAGEHLP_LINE64));
            uint ldisp;
            string line = "(noline:" + Marshal.GetLastWin32Error() + ")";
            if (SymGetLineFromAddr64(h, baseAddr + rva, out ldisp, ref ln))
                line = Marshal.PtrToStringAnsi(ln.FileName) + ":" + ln.LineNumber + " +0x" + ldisp.ToString("x");
            Console.WriteLine("RVA 0x" + rva.ToString("x") + " -> " + name + "  |  " + line);
        }
        return 0;
    }
}
