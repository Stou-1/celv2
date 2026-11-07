using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace CelV2.Settings
{
    public sealed class PassArgs
    {
        public int Pass;           // 1 or 2
        public string Stats;       // celv2enc --stats
        public long TargetBytes;   // --target-size (pass 2)
    }

    // Pure command-line assembly for ffmpeg / ffprobe / celv2enc / av2_mux / MP4Box; nothing here starts a process.
    public static class CommandBuilder
    {
        static string Inv(double d) { return d.ToString("0.######", CultureInfo.InvariantCulture); }
        static string Inv(int n) { return n.ToString(CultureInfo.InvariantCulture); }

        // CommandLineToArgvW rules: backslashes only matter before a quote (or the closing quote).
        public static string Quote(string arg)
        {
            if (arg == null) arg = "";
            if (arg.Length > 0 && arg.IndexOfAny(new[] { ' ', '\t', '\n', '\v', '"' }) < 0) return arg;
            var sb = new StringBuilder("\"");
            int bs = 0;
            foreach (char c in arg)
            {
                if (c == '\\') { bs++; continue; }
                if (c == '"') sb.Append('\\', bs * 2 + 1);
                else sb.Append('\\', bs);
                bs = 0;
                sb.Append(c);
            }
            sb.Append('\\', bs * 2).Append('"');
            return sb.ToString();
        }

        public static string Join(IEnumerable<string> args)
        {
            return string.Join(" ", args.Select(Quote));
        }

        // 8-bit sources go as they are (celv2enc widens them to 10 bits exactly); deeper ones keep 10 bits when coded at 10.
        public static string PixelFormatFor(InputInfo i, OutputFormat f)
        {
            if (f.BitDepth != 8 && f.BitDepth != 10) throw new NotSupportedException("only 8- or 10-bit output is supported (BitDepth=" + f.BitDepth + ")");
            return f.BitDepth == 10 && i.BitDepth > 8 ? "yuv420p10le" : "yuv420p";
        }

        // An HDR source stays HDR at 10 bits (signalled with its metadata); at 8 bits it is tone-mapped to SDR BT.709.
        public static bool KeepsHdr(InputInfo i, OutputFormat f)
        {
            return i.Hdr && f.BitDepth == 10;
        }

        public static string VideoFilterFor(InputInfo i, OutputFormat f)
        {
            string pix = PixelFormatFor(i, f);
            var chain = new List<string>();
            if (i.Interlaced) chain.Add("bwdif=mode=send_frame");
            if (i.VariableFrameRate) chain.Add("fps=" + Inv(i.FpsNum) + "/" + Inv(i.FpsDen));
            if (i.Hdr && !KeepsHdr(i, f))
                chain.Add("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv");
            if (i.Hdr || i.BitDepth > 8) chain.Add("format=" + pix);
            return chain.Count == 0 ? null : string.Join(",", chain);
        }

        public static string[] DecodeArgs(string input, InputInfo i, OutputFormat f)
        {
            string pix = PixelFormatFor(i, f);
            var a = new List<string> { "-hide_banner", "-nostdin", "-loglevel", "error", "-i", input, "-map", "0:" + Inv(i.VideoStreamIndex) };
            string vf = VideoFilterFor(i, f);
            if (vf != null) { a.Add("-vf"); a.Add(vf); }
            a.AddRange(new[] { "-pix_fmt", pix });
            if (pix != "yuv420p") a.AddRange(new[] { "-strict", "-1" }); // ffmpeg writes 10-bit y4m (C420p10) only when told it is not strict
            a.AddRange(new[] { "-f", "yuv4mpegpipe", "-" });
            return a.ToArray();
        }

        // pass: celv2enc --pass 1 / 2 for the target size (null: one pass). ffmpeg: the one --target-vmaf scores with.
        public static string[] EncodeArgs(EncodeSettings s, InputInfo i, string outputIvf, PassArgs pass = null, string ffmpeg = null)
        {
            PixelFormatFor(i, s.Output); // the same depth check as the decode side
            var a = new List<string> { "-i", "-", "-o", outputIvf, "--qindex", Inv(s.QIndex) };
            if (s.Output.BitDepth != 10) { a.Add("--bit-depth"); a.Add(Inv(s.Output.BitDepth)); }
            if (KeepsHdr(i, s.Output))
            {
                a.Add("--colour"); a.Add(i.ColorTransfer == "smpte2084" ? "pq" : "hlg");
                if (i.ColorRange == "pc" || i.ColorRange == "jpeg") { a.Add("--range"); a.Add("full"); }
                if (i.MasteringDisplay != null) { a.Add("--mdcv"); a.Add(i.MasteringDisplay); }
                if (i.ContentLight != null) { a.Add("--cll"); a.Add(i.ContentLight); }
            }
            // Auto leaves the tiles to celv2enc (it picks them from the frame size).
            int cols = TileLog2(s.Tiles), rows = TileLog2(s.TileRows);
            if (cols >= 0) { a.Add("--tile-columns"); a.Add(Inv(cols)); }
            if (rows >= 0) { a.Add("--tile-rows"); a.Add(Inv(rows)); }
            if (s.DeviceValue != "auto") { a.Add("--device"); a.Add(s.DeviceValue); }
            if (s.KeyInterval > 0) { a.Add("--keyint"); a.Add(Inv(s.KeyInterval)); }
            if (s.Threads > 0) { a.Add("--threads"); a.Add(Inv(s.Threads)); }
            if (s.Refs > 0) { a.Add("--refs"); a.Add(Inv(s.Refs)); }
            if (s.BFrames >= 0) { a.Add("--bframes"); a.Add(Inv(Math.Min(15, s.BFrames))); } // 0 differs from celv2enc's default; -1 leaves it to celv2enc
            if (!s.SceneCut) { a.Add("--scenecut"); a.Add("0"); }
            if (s.BQ != 16) { a.Add("--bq"); a.Add(Inv(s.BQ)); }
            if (s.AQ != -16) { a.Add("--aq"); a.Add(Inv(s.AQ)); }
            if (s.KQ != 0) { a.Add("--kq"); a.Add(Inv(s.KQ)); }
            if (s.DeblockDelta != 2) { a.Add("--deblock-delta"); a.Add(Inv(Math.Max(-16, Math.Min(15, s.DeblockDelta)))); }
            if (!s.Cdef) { a.Add("--cdef"); a.Add("0"); }
            if (s.Grain >= 0) { a.Add("--grain"); a.Add(Inv(Math.Min(50, s.Grain))); }
            if (s.ParallelB != ParallelMode.Auto) { a.Add("--parallel-b"); a.Add(s.ParallelB == ParallelMode.On ? "1" : "0"); }
            if (s.Tune != TuneKind.None) { a.Add("--tune"); a.Add(s.Tune.ToString().ToLowerInvariant()); }
            if (s.MaxEffort) { a.Add("--preset"); a.Add("max"); }
            a.Add("--workers"); a.Add(s.Workers > 0 ? Inv(s.Workers) : "auto");
            a.Add("--progress");
            if (pass != null)
            {
                a.AddRange(new[] { "--pass", Inv(pass.Pass), "--stats", pass.Stats });
                if (pass.Pass == 2) { a.Add("--target-size"); a.Add(pass.TargetBytes.ToString(CultureInfo.InvariantCulture)); }
            }
            else if (s.TargetVmaf > 0)
            {
                a.Add("--target-vmaf"); a.Add(s.TargetVmaf.ToString("0.##", CultureInfo.InvariantCulture));
                if (ffmpeg != null) { a.Add("--ffmpeg"); a.Add(ffmpeg); }
            }
            return a.ToArray();
        }

        // The IVF budget (celv2enc --target-size) of a whole output of targetBytes holding audioBytes of audio.
        public static long VideoBudget(long targetBytes, long audioBytes, long frames, bool mp4)
        {
            return targetBytes - audioBytes - ContainerBytes(frames, mp4);
        }

        // サイズ指定: the audio bitrate that keeps it to 1/8 of targetBytes over seconds, at least floor (AAC 32, HE-AACv2
        // 12). 0 when kbps (the setting) is already within it: the audio is made as set, copied where it can be.
        // shortcut: 1/8 is fixed, not weighed against the video; make it a setting if a target ever wants more audio.
        public static int AudioKbpsForSize(long targetBytes, double seconds, int kbps, int floor)
        {
            if (seconds <= 0) return 0;
            int cap = (int)(targetBytes / seconds / 1000);  // bytes / 8 x 8 bits
            return cap >= kbps ? 0 : Math.Max(floor, cap);
        }

        // The IVF's bytes at qindex 255 by celv2enc's own model (src/core/size_control.cpp: pass-1 payload x
        // 2^(-(q - pass-1 qindex) / 24), 32 + 12 a frame of IVF headers) from its --pass 1 stats; -1 unreadable.
        // The model shrinks too fast so far from the pass-1 qindex, so over the budget here is over for real.
        public static long CoarsestBytes(string stats)
        {
            string[] t = stats.Split(new[] { ' ', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            int q, n;
            if (t.Length < 6 || t[0] != "celv2-stats" || !int.TryParse(t[4], out q) || !int.TryParse(t[5], out n) || t.Length < 6 + n)
                return -1;
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                long b;
                if (!long.TryParse(t[6 + i], NumberStyles.None, CultureInfo.InvariantCulture, out b)) return -1;
                sum += b;
            }
            return 32 + 12L * n + (long)Math.Round(sum * Math.Pow(2, -(255 - q) / 24.0));
        }

        // The budget of the next pass 2 after one over by `over` bytes: the excess and 1% less, but at least 3% less,
        // about 1 qindex (2^(1/24)), so the start qindex moves and the try is not the last one again.
        public static long RetryBudget(long budget, long over)
        {
            return budget - Math.Max(over + budget / 100, budget * 3 / 100);
        }

        // MP4's boxes beyond the video and audio data; separate files have none (the IVF's own headers
        // are inside the budget celv2enc keeps). Measured with 192 kbps AAC at 24 fps, mp4 - OBUs - m4a:
        // 605 bytes for 48 frames, 2425 for 600 (about 450 + 3.3 a frame); here 1.5 times that, rounded up.
        public static long ContainerBytes(long frames, bool mp4)
        {
            return mp4 ? 1024 + 5 * frames : 0;
        }

        // log2 of the tile count for --tile-columns / --tile-rows, -1 for Auto (and anything unknown).
        static int TileLog2(TileMode m)
        {
            switch (m)
            {
                case TileMode.One: return 0;
                case TileMode.Two: return 1;
                case TileMode.Four: return 2;
                default: return -1;
            }
        }

        static bool CopyableAudio(string codec)
        {
            switch (codec)
            {
                case "aac": case "mp3": case "opus": case "ac3": case "flac": return true;
                default: return false;
            }
        }

        public static string AudioExtensionFor(string codec)
        {
            switch (codec)
            {
                case "mp3": return ".mp3";
                case "opus": return ".opus";
                case "ac3": return ".ac3";
                case "flac": return ".flac";
                default: return ".m4a";
            }
        }

        static List<string> AudioHead(string input, bool copy, int kbps)
        {
            var a = new List<string> { "-hide_banner", "-nostdin", "-loglevel", "error", "-i", input, "-map", "0:a:0", "-vn" };
            if (copy) a.AddRange(new[] { "-c:a", "copy" });
            else a.AddRange(new[] { "-c:a", "aac", "-b:a", Inv(Math.Max(32, Math.Min(512, kbps))) + "k" });
            return a;
        }

        // kbps: the AAC bitrate when the audio cannot be copied.
        public static string[] AudioArgs(string input, string codec, string outputAudio, int kbps = 192)
        {
            var a = AudioHead(input, CopyableAudio(codec), kbps);
            a.Add(outputAudio);
            return a.ToArray();
        }

        public static string[] Mp4AudioArgs(string input, string codec, double seconds, string outputM4a, int kbps = 192)
        {
            var a = AudioHead(input, codec == "aac", kbps);
            a.AddRange(new[] { "-t", Inv(seconds), outputM4a });
            return a.ToArray();
        }

        // HE-AACv2: ffmpeg writes a stereo 16-bit WAV (fdkaac takes neither mono nor 5.1 as HE-AACv2), cut to seconds when given.
        public static string[] HeAacWavArgs(string input, double? seconds, string outputWav)
        {
            var a = new List<string> { "-hide_banner", "-nostdin", "-loglevel", "error", "-i", input, "-map", "0:a:0", "-vn", "-ac", "2", "-c:a", "pcm_s16le" };
            if (seconds.HasValue) a.AddRange(new[] { "-t", Inv(seconds.Value) });
            a.AddRange(new[] { "-f", "wav", outputWav });
            return a.ToArray();
        }

        // -I: a WAV over 4 GB has a wrong length in its header.
        public static string[] FdkaacArgs(string wav, string outputM4a, int kbps)
        {
            return new[] { "-S", "-I", "-p", "29", "-b", Inv(Math.Max(12, Math.Min(96, kbps))), "-o", outputM4a, wav };
        }

        public static string[] Av2MuxArgs(string obu, string mp4, int rate, int scale, string colrOrNull)
        {
            var a = new List<string> { obu, "-o", mp4, "--fps", ((double)rate / scale).ToString("R", CultureInfo.InvariantCulture) }; // av2_mux --fps is a float, not a fraction
            if (colrOrNull != null) { a.Add("--colr-override"); a.Add(colrOrNull); }
            return a.ToArray();
        }

        public static string[] Mp4BoxAddArgs(string audio, string mp4)
        {
            return new[] { "-noprog", "-add", audio, mp4 };
        }

        public static string ColrFor(InputInfo i, OutputFormat f)
        {
            if (i.Hdr && !KeepsHdr(i, f)) return "1:1:1:0"; // the tonemap in VideoFilterFor outputs bt709/bt709/bt709/tv
            int cp = Lookup(i.ColorPrimaries, "bt709", 1, "bt470bg", 5, "smpte170m", 6, "bt2020", 9);
            int tc = Lookup(i.ColorTransfer, "bt709", 1, "smpte170m", 6, "iec61966-2-1", 13, "bt2020-10", 14, "smpte2084", 16, "arib-std-b67", 18);
            int mc = Lookup(i.ColorSpace, "bt709", 1, "bt470bg", 5, "smpte170m", 6, "bt2020nc", 9);
            int fr = Lookup(i.ColorRange, "tv", 0, "mpeg", 0, "pc", 1, "jpeg", 1);
            if (cp < 0 || tc < 0 || mc < 0 || fr < 0) return null;
            return Inv(cp) + ":" + Inv(tc) + ":" + Inv(mc) + ":" + Inv(fr);
        }

        // -1 when name is null or not in the pairs.
        static int Lookup(string name, params object[] pairs)
        {
            for (int k = 0; name != null && k < pairs.Length; k += 2)
                if ((string)pairs[k] == name) return (int)pairs[k + 1];
            return -1;
        }
    }
}
