using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace CelV2.Runner
{
    // One coded frame from celv2enc's "progress frame=N type=T qindex=Q bytes=B" (T: K keyframe, P lone frame,
    // A anchor, B the others). An older celv2enc's "progress frame=N" gives Type '?', Qindex 0, Bytes -1.
    public sealed class FrameInfo { public long Frame; public char Type = '?'; public int Qindex; public long Bytes = -1; }

    public static class ProgressLine
    {
        const string Prefix = "progress frame=";

        // null for any other line.
        public static FrameInfo Parse(string line)
        {
            if (line == null || !line.StartsWith(Prefix, StringComparison.Ordinal)) return null;
            string[] parts = line.Substring(Prefix.Length).Split(' ');
            long frame;
            if (!long.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out frame)) return null;
            var f = new FrameInfo { Frame = frame };
            foreach (string p in parts.Skip(1))
            {
                int eq = p.IndexOf('=');
                if (eq < 0) continue;
                string k = p.Substring(0, eq), v = p.Substring(eq + 1);
                if (k == "type" && v.Length == 1) f.Type = v[0];
                else if (k == "qindex") int.TryParse(v, NumberStyles.None, CultureInfo.InvariantCulture, out f.Qindex);
                else if (k == "bytes" && !long.TryParse(v, NumberStyles.None, CultureInfo.InvariantCulture, out f.Bytes)) f.Bytes = -1;
            }
            return f;
        }
    }

    // What the details window shows of the running (or last) encode: filled from any thread, read as a Snapshot.
    public sealed class EncodeDetails
    {
        public const int MaxLogLines = 2000;

        public sealed class View
        {
            public string Name = "", Stage = "", Engine = "";
            public int Pass;                      // 1 from the first frame; frame 1 again (target size's second pass) adds 1
            public long Total;                    // the input's frame count, 0 unknown
            public TimeSpan Elapsed;
            public bool Running;
            public IReadOnlyList<FrameInfo> Frames = new FrameInfo[0];
            public long Bytes;                    // the frames' bytes so far (this pass)
            public double Kbps;                   // their bitrate at the input's frame rate
            public long ProjectedBytes = -1;      // the video's bytes at the end at this rate, -1 unknown
            public IReadOnlyList<string> Log = new string[0];
            public long LogLinesSeen;             // every line added, kept or not (the window appends the new ones)
        }

        readonly object gate = new object();
        readonly List<FrameInfo> frames = new List<FrameInfo>();
        readonly Queue<string> log = new Queue<string>();
        readonly System.Diagnostics.Stopwatch clock = new System.Diagnostics.Stopwatch();
        string name = "", stage = "", engine = "";
        int pass;
        long total, bytes, logSeen;
        double fps;

        public void Begin(string name, long totalFrames, int fpsNum, int fpsDen)
        {
            lock (gate)
            {
                this.name = name ?? ""; stage = ""; engine = "";
                pass = 0; total = totalFrames; bytes = 0;
                fps = fpsNum > 0 && fpsDen > 0 ? (double)fpsNum / fpsDen : 0;
                frames.Clear(); log.Clear(); logSeen = 0;
                clock.Restart();
            }
        }

        public void End() { lock (gate) clock.Stop(); }
        public void SetStage(string s) { lock (gate) stage = s ?? ""; }
        public void SetEngine(string line) { lock (gate) engine = line ?? ""; }

        public void AddFrame(FrameInfo f)
        {
            lock (gate)
            {
                if (f.Frame <= 1 || pass == 0) { frames.Clear(); bytes = 0; pass++; stage = ""; }  // what ran before is over
                frames.Add(f);
                if (f.Bytes > 0) bytes += f.Bytes;
            }
        }

        public void AddLog(string line)
        {
            lock (gate)
            {
                log.Enqueue(line ?? "");
                logSeen++;
                while (log.Count > MaxLogLines) log.Dequeue();
            }
        }

        public View Snapshot()
        {
            lock (gate)
            {
                int n = frames.Count;
                double seconds = fps > 0 ? n / fps : 0;
                return new View
                {
                    Name = name, Stage = stage, Engine = engine, Pass = pass, Total = total,
                    Elapsed = clock.Elapsed, Running = clock.IsRunning,
                    Frames = frames.ToArray(), Bytes = bytes,
                    Kbps = seconds > 0 ? bytes * 8 / seconds / 1000 : 0,
                    ProjectedBytes = total > 0 && n > 0 ? bytes * total / n : -1,
                    Log = log.ToArray(), LogLinesSeen = logSeen,
                };
            }
        }
    }
}
