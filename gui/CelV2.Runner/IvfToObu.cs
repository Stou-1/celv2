using System;
using System.IO;

namespace CelV2.Runner
{
    public static class IvfToObu
    {
        // Strips the IVF file header (its length is the u16 at offset 6) and each 12-byte frame header
        // (u32 size + u64 pts). celv2enc stores the frame rate at offset 16 and the time scale at 20.
        public static void Convert(string ivf, string obu, out int rate, out int scale)
        {
            using (var inp = new FileStream(ivf, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                var h = new byte[32];
                if (ReadFull(inp, h, 32) != 32 || h[0] != 'D' || h[1] != 'K' || h[2] != 'I' || h[3] != 'F')
                    throw new InvalidDataException("not an IVF file: " + ivf);
                int headerLen = h[6] | (h[7] << 8);
                long r = U32(h, 16), s = U32(h, 20);
                if (headerLen < 32 || headerLen > inp.Length) throw new InvalidDataException("bad IVF header length");
                if (r <= 0 || s <= 0 || r > int.MaxValue || s > int.MaxValue) throw new InvalidDataException("bad IVF frame rate");
                rate = (int)r; scale = (int)s;
                inp.Seek(headerLen, SeekOrigin.Begin);
                try
                {
                    using (var o = new FileStream(obu, FileMode.Create, FileAccess.Write, FileShare.None))
                    {
                        var fh = new byte[12];
                        var buf = new byte[1 << 16];
                        for (;;)
                        {
                            int got = ReadFull(inp, fh, 12);
                            if (got == 0) break;
                            if (got != 12) throw new InvalidDataException("truncated IVF frame header");
                            long size = U32(fh, 0);
                            if (size > inp.Length - inp.Position) throw new InvalidDataException("truncated IVF frame");
                            while (size > 0)
                            {
                                int n = inp.Read(buf, 0, (int)Math.Min(buf.Length, size));
                                if (n <= 0) throw new InvalidDataException("truncated IVF frame");
                                o.Write(buf, 0, n);
                                size -= n;
                            }
                        }
                    }
                }
                catch
                {
                    try { File.Delete(obu); } catch (IOException) { }
                    throw;
                }
            }
        }

        static long U32(byte[] b, int o) { return (uint)(b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)); }

        static int ReadFull(Stream s, byte[] b, int count)
        {
            int total = 0, n;
            while (total < count && (n = s.Read(b, total, count - total)) > 0) total += n;
            return total;
        }
    }
}
