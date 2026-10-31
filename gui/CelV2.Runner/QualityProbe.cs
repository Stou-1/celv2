using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using CelV2.Settings;

namespace CelV2.Runner
{
    // Keeps the output from growing past an already compressed source: short samples of the input are
    // encoded at trial qindex values, and the finest qindex whose video bitrate stays within the
    // source's is used when the setting asks for a finer one. (On a 1.1 Mbps FHD AVC source, qindex 45
    // made 20x the source's bitrate; PSNR against the source showed no knee, as it also rewards
    // copying the source's own artifacts, so the source's size is the limit.)
    public static class QualityProbe
    {
        // Default share of the source's bitrate aimed at (EncodeSettings.SizeCapPercent): the samples
        // leave out keyframes and scene changes.
        public const double Margin = 0.9;

        // The smallest qindex in [start, 255] whose bitrate (kbps(q), decreasing in q) is at most target,
        // within 2; start when it already is. A search on log(bitrate), about 24 qindex a halving.
        public static int Choose(int start, double targetKbps, Func<int, double> kbps, int maxTrials = 7)
        {
            var known = new Dictionary<int, double>();
            Func<int, double> at = q => { double v; if (!known.TryGetValue(q, out v)) known[q] = v = kbps(q); return v; };
            if (at(start) <= targetKbps) return start;
            int lo = start, hi = 255;  // lo fails; hi passes (255 is the coarsest there is)
            while (hi - lo > 2 && known.Count < maxTrials)
            {
                double lk = Math.Log(at(lo)), t = Math.Log(targetKbps), guess;
                double hk;
                if (known.TryGetValue(hi, out hk)) guess = lo + (hi - lo) * (lk - t) / Math.Max(1e-9, lk - Math.Log(hk));
                else guess = lo + (lk - t) / Math.Log(2) * 24;
                int m = Math.Max(lo + 1, Math.Min(hi - 1, (int)Math.Ceiling(guess)));
                if (at(m) <= targetKbps) hi = m; else lo = m;
            }
            return hi;
        }

        // Frames per sample: about 30 megapixels, 3 to 20.
        public static int SampleFrames(int width, int height)
        {
            return Math.Max(3, Math.Min(20, (int)Math.Round(30e6 / Math.Max(1.0, (double)width * height))));
        }

        // Video kbps of an IVF without the first frame of each sample (the keyframes there, rare in the
        // whole video). frames per sample n.
        public static double KbpsWithoutSampleStarts(string ivf, int n, int fpsNum, int fpsDen)
        {
            var sizes = new List<long>();
            using (var f = File.OpenRead(ivf))
            using (var r = new BinaryReader(f))
            {
                f.Position = 32;
                while (f.Position + 12 <= f.Length)
                {
                    long size = r.ReadUInt32();
                    r.ReadUInt64();
                    sizes.Add(size);
                    f.Position += size;
                }
            }
            long bytes = 0;
            int frames = 0;
            for (int i = 0; i < sizes.Count; i++)
                if (i % n != 0) { bytes += sizes[i]; frames++; }
            if (frames == 0) return 0;
            return bytes * 8.0 / 1000 / (frames * (double)fpsDen / fpsNum);
        }

        // The qindex to encode with: settings' own unless that would make more video bits than the source
        // has (info.VideoBitRate; nothing is done when it is unknown). stage reports progress.
        public static int Pick(ToolPaths tools, string input, InputInfo info, EncodeSettings settings, CancellationToken cancel,
                               Action<string> stage)
        {
            if (info.VideoBitRate <= 0 || info.DurationSeconds <= 0 || info.FpsNum <= 0) return settings.QIndex;
            int n = SampleFrames(info.Width, info.Height);
            string dir = Path.Combine(Path.GetTempPath(), "celv2-probe-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                string samples = Path.Combine(dir, "samples.y4m");
                using (var o = File.Create(samples))
                    for (int k = 0; k < 3; k++)
                    {
                        cancel.ThrowIfCancellationRequested();
                        stage("画質を調べています（見本 " + (k + 1) + " / 3 を取り出し中）");
                        string part = Path.Combine(dir, "s" + k + ".y4m");
                        var a = CommandBuilder.DecodeArgs(input, info, settings.Output).ToList();
                        a.InsertRange(a.IndexOf("-i"), new[] { "-ss", (info.DurationSeconds * (0.2 + 0.3 * k)).ToString("0.###", CultureInfo.InvariantCulture) });
                        a.InsertRange(a.IndexOf("-f"), new[] { "-frames:v", n.ToString(CultureInfo.InvariantCulture) });
                        a[a.Count - 1] = part;
                        var r = Proc.Capture(tools.Ffmpeg, a, cancel);
                        cancel.ThrowIfCancellationRequested();
                        if (r.ExitCode != 0 || !File.Exists(part)) throw new InvalidOperationException("見本の取り出しに失敗しました: " + Proc.Tail(r.Err, 3));
                        byte[] d = File.ReadAllBytes(part);
                        int skip = k == 0 ? 0 : Array.IndexOf(d, (byte)'\n') + 1;  // one y4m header
                        o.Write(d, skip, d.Length - skip);
                        File.Delete(part);
                    }
                string ivf = Path.Combine(dir, "trial.ivf");
                Func<int, double> kbps = q =>
                {
                    cancel.ThrowIfCancellationRequested();
                    stage("画質を調べています（qindex " + q + " を試し中）");
                    var s = settings.Clone();
                    s.QIndex = q;
                    var a = CommandBuilder.EncodeArgs(s, info, ivf).ToList();
                    a[a.IndexOf("-i") + 1] = samples;
                    var r = Proc.Capture(tools.CelV2enc, a, cancel);
                    cancel.ThrowIfCancellationRequested();
                    if (r.ExitCode != 0) throw new InvalidOperationException("試しの変換に失敗しました: " + Proc.Tail(r.Err, 3));
                    return KbpsWithoutSampleStarts(ivf, n, info.FpsNum, info.FpsDen);
                };
                double share = settings.SizeCapPercent > 0 ? Math.Min(100, Math.Max(10, settings.SizeCapPercent)) / 100.0 : Margin;
                return Choose(settings.QIndex, info.VideoBitRate / 1000.0 * share, kbps);
            }
            finally
            {
                try { Directory.Delete(dir, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
            }
        }
    }
}
