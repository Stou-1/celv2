using System;
using System.IO;
using System.Text;

namespace CelV2.Runner
{
    // The last conversion run's log (the GUI's %APPDATA%\celv2gui\last_encode.log): each new run replaces the
    // file. Lines may come from any thread. Writing never throws: a log that cannot be written must not stop
    // a run.
    public sealed class EncodeLog : IDisposable
    {
        public const long MaxBytes = 8L << 20;

        readonly object gate = new object();
        StreamWriter w;
        long written;
        bool full;

        public EncodeLog(string path)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
                w = new StreamWriter(path, false, new UTF8Encoding(false)) { AutoFlush = true };
            }
            catch (Exception e) when (e is IOException || e is UnauthorizedAccessException || e is ArgumentException || e is NotSupportedException)
            {
                w = null;
            }
        }

        public void Line(string text)
        {
            lock (gate)
            {
                if (w == null || full) return;
                string line = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + "  " + text;
                if (written + line.Length * 3 > MaxBytes)
                {
                    line = "（ログが " + (MaxBytes >> 20) + " MB を超えたため、ここから先は記録しません）";
                    full = true;
                }
                try { w.WriteLine(line); written += Encoding.UTF8.GetByteCount(line) + 2; }
                catch (IOException) { w = null; }
            }
        }

        public void Dispose()
        {
            lock (gate)
            {
                try { if (w != null) w.Dispose(); }
                catch (IOException) { }
                w = null;
            }
        }
    }
}
