using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using CelV2.Settings;

namespace CelV2.Tests
{
    public static class CommandTests
    {
        public static readonly (string name, Action run)[] All =
        {
            ("parses_sdr_h264_with_aac", ParsesSdr),
            ("parses_hdr10", ParsesHdr10),
            ("parses_hdr10_metadata", ParsesHdr10Metadata),
            ("hdr_passes_through_at_10_bits", HdrPassthroughArgs),
            ("probe_estimates_missing_frame_count", EstimatesFrameCount),
            ("detects_vfr_and_interlace", DetectsVfrAndInterlace),
            ("rejects_no_video_stream", RejectsNoVideo),
            ("skips_cover_art", SkipsCoverArt),
            ("tiles_auto_rule", TilesAuto),
            ("encode_args_full", EncodeArgsFull),
            ("device_list_and_args", DeviceListAndArgs),
            ("bit_depth_8_or_10", BitDepth8Or10),
            ("filters_for_inputs", FiltersForInputs),
            ("audio_extensions", AudioExtensions),
            ("he_aac_v2_args", HeAacV2Args),
            ("target_size_args", TargetSizeArgs),
            ("video_budget", VideoBudget),
            ("quotes_paths_with_spaces_and_japanese", QuotesRoundTrip),
            ("namer_avoids_existing_files", NamerAvoidsExisting),
            ("namer_uses_given_name", NamerUsesGivenName),
            ("namer_rejects_bad_names", NamerRejectsBadNames),
            ("mp4_command_lines", Mp4CommandLines),
            ("colr_for_inputs", ColrForInputs),
        };

        static InputInfo Probe(string file)
        {
            // build.ps1 copies tests\data next to the exe
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", file);
            return ProbeParser.Parse(File.ReadAllText(path));
        }

        static readonly OutputFormat Fmt = new OutputFormat();          // 10 bits, the default
        static readonly OutputFormat Eight = new OutputFormat { BitDepth = 8 };

        static InputInfo Sdr(int w = 640) { return new InputInfo { Width = w, Height = 480, FpsNum = 24, FpsDen = 1, BitDepth = 8 }; }

        static void Throws<T>(Action a, string msg) where T : Exception
        {
            try { a(); }
            catch (T) { return; }
            throw new CheckFailure(msg + ": expected " + typeof(T).Name);
        }

        static void ParsesSdr()
        {
            var i = Probe("probe_sdr_h264_aac.json");
            Check.Equal(640, i.Width, "w"); Check.Equal(480, i.Height, "h");
            Check.Equal(24, i.FpsNum, "num"); Check.Equal(1, i.FpsDen, "den");
            Check.Equal(8, i.BitDepth, "depth");
            Check.That(!i.Hdr && !i.VariableFrameRate && !i.Interlaced, "flags");
            Check.Equal("aac", i.AudioCodec, "audio");
            Check.Equal(48L, i.FrameCount, "frames");
            Check.Equal(2.0, i.DurationSeconds, "duration");
        }

        static void ParsesHdr10()
        {
            var i = Probe("probe_hdr10.json");
            Check.Equal(10, i.BitDepth, "depth");
            Check.That(i.Hdr, "hdr");
            Check.Equal("smpte2084", i.ColorTransfer, "transfer");
            Check.Equal("bt2020", i.ColorPrimaries, "primaries");
            Check.Equal("bt2020nc", i.ColorSpace, "matrix");
            Check.That(i.AudioCodec == null, "no audio");
        }

        static void ParsesHdr10Metadata()
        {
            var i = Probe("probe_hdr10.json");
            // a real ffprobe of x265's HDR10 output: master-display G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1), max-cll 1000,400
            ProbeParser.ParseHdrMetadata(File.ReadAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "probe_hdr10_frames.json")), i);
            Check.Equal("G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)", i.MasteringDisplay, "mdcv");
            Check.Equal("1000,400", i.ContentLight, "cll");
            Check.That(ProbeParser.HdrMetadataArgs("in.mkv", 2).SequenceEqual(new[] { "-v", "error", "-print_format", "json", "-select_streams", "2",
                "-read_intervals", "%+#1", "-show_frames", "-show_entries", "frame=side_data_list", "in.mkv" }), "args");

            Func<string, InputInfo> parse = json => { var x = new InputInfo(); ProbeParser.ParseHdrMetadata(json, x); return x; };
            const string md = "{\"side_data_type\":\"Mastering display metadata\",\"red_x\":\"34000/50000\",\"red_y\":\"16000/50000\",\"green_x\":\"13250/50000\",\"green_y\":\"34500/50000\",\"blue_x\":\"7500/50000\",\"blue_y\":\"3000/50000\",\"white_point_x\":\"15635/50000\",\"white_point_y\":\"16450/50000\"";
            Check.That(parse("{\"frames\":[{\"side_data_list\":[" + md + "}]}]}").MasteringDisplay == null, "primaries without luminance: none");
            var scaled = parse("{\"frames\":[{\"side_data_list\":[" + md.Replace("34000/50000", "17/25") + ",\"min_luminance\":\"5/50000\",\"max_luminance\":\"1000/1\"}]}]}");
            Check.Equal("G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)", scaled.MasteringDisplay, "other denominators, rounded");
            Check.That(parse("{\"frames\":[{\"side_data_list\":[" + md + ",\"min_luminance\":\"5/1\",\"max_luminance\":\"1/1\"}]}]}").MasteringDisplay == null, "min above max: none");
            Check.That(parse("{\"frames\":[{\"side_data_list\":[{\"side_data_type\":\"Content light level metadata\",\"max_content\":70000,\"max_average\":1}]}]}").ContentLight == null, "MaxCLL above 65535: none");
            Check.Equal("0,0", parse("{\"frames\":[{\"side_data_list\":[{\"side_data_type\":\"Content light level metadata\",\"max_content\":0,\"max_average\":0}]}]}").ContentLight, "zero is allowed");
            var bad = parse("not json");
            Check.That(bad.MasteringDisplay == null && bad.ContentLight == null, "unreadable output: none");
            Check.That(parse("{\"frames\":[]}").MasteringDisplay == null, "no frame: none");
        }

        static void HdrPassthroughArgs()
        {
            var h = Probe("probe_hdr10.json");
            h.MasteringDisplay = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)";
            h.ContentLight = "1000,400";
            Check.Equal("-i - -o o.ivf --qindex 170 --colour pq --mdcv G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1) " +
                        "--cll 1000,400 --bframes 3 --workers auto --progress", string.Join(" ", CommandBuilder.EncodeArgs(new EncodeSettings(), h, "o.ivf")), "hdr10");
            Check.Equal("-hide_banner -nostdin -loglevel error -i in.mkv -map 0:0 -vf format=yuv420p10le -pix_fmt yuv420p10le -strict -1 -f yuv4mpegpipe -",
                        string.Join(" ", CommandBuilder.DecodeArgs("in.mkv", h, Fmt)), "decode keeps 10 bits, no tonemap");
            Check.Equal("9:16:9:0", CommandBuilder.ColrFor(h, Fmt), "mp4 colr: bt2020 / pq / bt2020nc");
            var hlg = Probe("probe_hdr10.json"); hlg.ColorTransfer = "arib-std-b67"; hlg.ColorRange = "pc";
            var a = CommandBuilder.EncodeArgs(new EncodeSettings(), hlg, "o.ivf");
            Check.Equal("hlg", a[Array.IndexOf(a, "--colour") + 1], "hlg");
            Check.Equal("full", a[Array.IndexOf(a, "--range") + 1], "full range");
            Check.That(!a.Contains("--mdcv") && !a.Contains("--cll"), "no metadata probed: none sent");
            Check.Equal("9:18:9:1", CommandBuilder.ColrFor(hlg, Fmt), "hlg colr");
            var e = CommandBuilder.EncodeArgs(new EncodeSettings { Output = Eight }, h, "o.ivf");
            Check.Equal("-i - -o o.ivf --qindex 170 --bit-depth 8 --bframes 3 --workers auto --progress", string.Join(" ", e), "8 bits: tone-mapped, unsignalled");
            Check.Equal("1:1:1:0", CommandBuilder.ColrFor(h, Eight), "8 bits: bt709 after the tonemap");
            Check.That(!CommandBuilder.EncodeArgs(new EncodeSettings(), Sdr(), "o.ivf").Contains("--colour"), "sdr: no colour");
        }

        static void EstimatesFrameCount()
        {
            var i = Probe("probe_no_frame_count.json");
            Check.Equal(48L, i.FrameCount, "round(2.0 * 24)");
        }

        static void DetectsVfrAndInterlace()
        {
            var v = Probe("probe_vfr.json");
            Check.That(v.VariableFrameRate, "vfr");
            Check.That(!v.Interlaced, "vfr not interlaced");
            Check.Equal(76, v.FpsNum, "avg num"); Check.Equal(3, v.FpsDen, "avg den");
            var l = Probe("probe_interlaced.json");
            Check.That(l.Interlaced, "interlaced");
            Check.That(!l.VariableFrameRate, "interlaced not vfr");
        }

        static void RejectsNoVideo()
        {
            Throws<FormatException>(() => Probe("probe_audio_only.json"), "audio only");
            Throws<FormatException>(() => ProbeParser.Parse("not json"), "garbage");
        }

        static void SkipsCoverArt()
        {
            string msg = null;
            try { Probe("probe_mp3_cover_art.json"); }
            catch (FormatException e) { msg = e.Message; }
            Check.That(msg != null && msg.Contains("映像"), "mp3 with cover art has no video: " + msg);
            var i = Probe("probe_cover_art_first.json");
            Check.Equal(1, i.VideoStreamIndex, "the real video, not the cover");
            Check.Equal(320, i.Width, "w");
            Check.Equal(48L, i.FrameCount, "frames");
            var a = CommandBuilder.DecodeArgs("in.mp4", i, Fmt);
            Check.Equal("0:1", a[Array.IndexOf(a, "-map") + 1], "map");
        }

        static void TilesAuto()
        {
            var s = new EncodeSettings();
            // Auto leaves the tiles to celv2enc (it picks them from the frame size)
            Check.That(CommandBuilder.EncodeArgs(s, Sdr(15360), "o.ivf").SequenceEqual(
                new[] { "-i", "-", "-o", "o.ivf", "--qindex", "170", "--bframes", "3", "--workers", "auto", "--progress" }), "16K");
            TileMode[] m = { TileMode.One, TileMode.Two, TileMode.Four };
            for (int k = 0; k < 3; k++)
            {
                s.Tiles = m[k];
                var a = CommandBuilder.EncodeArgs(s, Sdr(100), "o.ivf");
                Check.Equal(k.ToString(), a[Array.IndexOf(a, "--tile-columns") + 1], "tile " + m[k]);
            }
        }

        static void EncodeArgsFull()
        {
            var s = new EncodeSettings { QIndex = 120, Device = DecisionDevice.Cpu, KeyInterval = 30, Threads = 4, Refs = 7, BFrames = 2 };
            Check.That(CommandBuilder.EncodeArgs(s, Sdr(), "o.ivf").SequenceEqual(new[] {
                "-i", "-", "-o", "o.ivf", "--qindex", "120", "--device", "cpu", "--keyint", "30", "--threads", "4", "--refs", "7",
                "--bframes", "2", "--workers", "auto", "--progress" }), "full");
            s.Device = DecisionDevice.Gpu;
            Check.That(CommandBuilder.EncodeArgs(s, Sdr(), "o.ivf").Contains("gpu"), "gpu");
            var d = CommandBuilder.EncodeArgs(new EncodeSettings(), Sdr(), "o.ivf");
            Check.That(!d.Contains("--device") && !d.Contains("--keyint") && !d.Contains("--threads") && !d.Contains("--refs"),
                "defaults omit");
            var none = CommandBuilder.EncodeArgs(new EncodeSettings { BFrames = 0 }, Sdr(), "o.ivf");
            Check.Equal("0", none[Array.IndexOf(none, "--bframes") + 1], "0 B-frames passed explicitly");
            Check.That(!CommandBuilder.EncodeArgs(new EncodeSettings { BFrames = -1 }, Sdr(), "o.ivf").Contains("--bframes"), "auto B-frames omitted");
            foreach (var k in new[] { "--tile-rows", "--scenecut", "--bq", "--aq", "--kq", "--deblock-delta", "--cdef", "--parallel-b" })
                Check.That(!d.Contains(k), "default omits " + k);
            var all = new EncodeSettings { Workers = 2, TileRows = TileMode.Four, SceneCut = false, BQ = 10, AQ = -20, KQ = -8, DeblockDelta = 40,
                                           Cdef = false, ParallelB = ParallelMode.On };
            Check.Equal("-i - -o o.ivf --qindex 170 --tile-rows 2 --bframes 3 --scenecut 0 --bq 10 --aq -20 --kq -8 --deblock-delta 15 " +
                        "--cdef 0 --parallel-b 1 --workers 2 --progress", string.Join(" ", CommandBuilder.EncodeArgs(all, Sdr(), "o.ivf")), "every option");
            all.ParallelB = ParallelMode.Off;
            var off = CommandBuilder.EncodeArgs(all, Sdr(), "o.ivf");
            Check.Equal("0", off[Array.IndexOf(off, "--parallel-b") + 1], "parallel off");
            Check.That(!d.Contains("--tune"), "default omits --tune");
            foreach (var t in new[] { TuneKind.Animation, TuneKind.Film, TuneKind.Still })
            {
                var ta = CommandBuilder.EncodeArgs(new EncodeSettings { Tune = t }, Sdr(), "o.ivf");
                Check.Equal(t.ToString().ToLowerInvariant(), ta[Array.IndexOf(ta, "--tune") + 1], "tune " + t);
            }
            Check.That(!d.Contains("--preset"), "default omits --preset");
            var mx = CommandBuilder.EncodeArgs(new EncodeSettings { MaxEffort = true }, Sdr(), "o.ivf");
            Check.Equal("max", mx[Array.IndexOf(mx, "--preset") + 1], "max effort");
        }

        static void DeviceListAndArgs()
        {
            var l = DeviceList.Parse("gpu\tNVIDIA GeForce RTX 5090 (CUDA)\r\nvulkan\tNVIDIA GeForce RTX 5090 (Vulkan)\r\n" +
                                     "vulkan2\tIntel(R) Arc(TM) Pro B50 Graphics (Vulkan)\r\ncpu\tCPU\r\nnoise without a tab\r\n");
            Check.Equal(5, l.Count, "auto + 4 devices");
            Check.Equal("auto", l[0].Arg, "auto first");
            Check.Equal("自動", l[0].Label, "auto label");
            Check.Equal("NVIDIA GeForce RTX 5090（CUDA）", l[1].Label, "cuda label");
            Check.Equal("vulkan2", l[3].Arg, "second Vulkan GPU");
            Check.Equal("Intel(R) Arc(TM) Pro B50 Graphics（Vulkan）", l[3].Label, "only the last bracket is the API");
            Check.Equal("CPU", l[4].Label, "cpu label");
            Check.Equal(1, DeviceList.Parse("").Count, "empty output: auto only");
            Check.Equal(3, DeviceList.Fallback().Count, "fallback");

            var s = new EncodeSettings { DeviceArg = "vulkan2" };
            var a = CommandBuilder.EncodeArgs(s, Sdr(), "o.ivf");
            Check.Equal("vulkan2", a[Array.IndexOf(a, "--device") + 1], "chosen GPU");
            Check.That(!CommandBuilder.EncodeArgs(new EncodeSettings { DeviceArg = "auto" }, Sdr(), "o.ivf").Contains("--device"), "auto omitted");
            var old = new EncodeSettings { Device = DecisionDevice.Cpu };
            Check.Equal("cpu", old.DeviceValue, "older settings: Device");
            old.DeviceArg = "gpu";
            Check.Equal("gpu", old.DeviceValue, "DeviceArg wins");
        }

        static void BitDepth8Or10()
        {
            var f = new OutputFormat { BitDepth = 12 };
            Throws<NotSupportedException>(() => CommandBuilder.PixelFormatFor(Sdr(), f), "PixelFormatFor");
            Throws<NotSupportedException>(() => CommandBuilder.DecodeArgs("a.mp4", Sdr(), f), "DecodeArgs");
            Throws<NotSupportedException>(() => CommandBuilder.EncodeArgs(new EncodeSettings { Output = f }, Sdr(), "o.ivf"), "EncodeArgs");
            var ten = Sdr(); ten.BitDepth = 10;
            var twelve = Sdr(); twelve.BitDepth = 12;
            Check.Equal("yuv420p", CommandBuilder.PixelFormatFor(Sdr(), Fmt), "8-bit source at 10: celv2enc widens it");
            Check.Equal("yuv420p10le", CommandBuilder.PixelFormatFor(ten, Fmt), "10-bit source at 10");
            Check.Equal("yuv420p10le", CommandBuilder.PixelFormatFor(twelve, Fmt), "12-bit source at 10");
            Check.Equal("yuv420p", CommandBuilder.PixelFormatFor(ten, Eight), "10-bit source at 8");
            var a = CommandBuilder.EncodeArgs(new EncodeSettings { Output = Eight }, Sdr(), "o.ivf");
            Check.Equal("8", a[Array.IndexOf(a, "--bit-depth") + 1], "--bit-depth 8");
            Check.That(!CommandBuilder.EncodeArgs(new EncodeSettings(), Sdr(), "o.ivf").Contains("--bit-depth"), "10 is celv2enc's default");
        }

        static void FiltersForInputs()
        {
            Check.That(CommandBuilder.VideoFilterFor(Sdr(), Fmt) == null, "sdr8 none");
            var ten = Sdr(); ten.BitDepth = 10;
            Check.Equal("format=yuv420p10le", CommandBuilder.VideoFilterFor(ten, Fmt), "10-bit sdr");
            Check.Equal("format=yuv420p", CommandBuilder.VideoFilterFor(ten, Eight), "10-bit sdr at 8");
            var hdr = Sdr(); hdr.BitDepth = 10; hdr.Hdr = true;
            Check.Equal("format=yuv420p10le", CommandBuilder.VideoFilterFor(hdr, Fmt), "hdr at 10: no tonemap");
            string h = CommandBuilder.VideoFilterFor(hdr, Eight);
            Check.That(h.Contains("zscale") && h.Contains("tonemap") && h.EndsWith(",format=yuv420p"), "hdr: " + h);
            var vfr = Sdr(); vfr.VariableFrameRate = true; vfr.FpsNum = 76; vfr.FpsDen = 3;
            Check.Equal("fps=76/3", CommandBuilder.VideoFilterFor(vfr, Fmt), "vfr");
            var il = Sdr(); il.Interlaced = true; il.VariableFrameRate = true; il.Hdr = true;
            string c = CommandBuilder.VideoFilterFor(il, Eight);
            Check.That(c.StartsWith("bwdif=mode=send_frame,fps=24/1,zscale"), "order: " + c);
            Check.Equal("-hide_banner -nostdin -loglevel error -i in.mp4 -map 0:0 -vf fps=76/3 -pix_fmt yuv420p -f yuv4mpegpipe -",
                string.Join(" ", CommandBuilder.DecodeArgs("in.mp4", vfr, Fmt)), "decode args");
            Check.Equal("-hide_banner -nostdin -loglevel error -i in.mp4 -map 0:0 -pix_fmt yuv420p -f yuv4mpegpipe -",
                string.Join(" ", CommandBuilder.DecodeArgs("in.mp4", Sdr(), Fmt)), "decode args, no filter");
        }

        static void AudioExtensions()
        {
            string[,] t = { { "aac", ".m4a" }, { "mp3", ".mp3" }, { "opus", ".opus" }, { "ac3", ".ac3" }, { "flac", ".flac" }, { "pcm_s16le", ".m4a" } };
            for (int k = 0; k < t.GetLength(0); k++)
            {
                Check.Equal(t[k, 1], CommandBuilder.AudioExtensionFor(t[k, 0]), t[k, 0]);
                string a = string.Join(" ", CommandBuilder.AudioArgs("in.mkv", t[k, 0], "o" + t[k, 1]));
                Check.That(a.Contains(k < 5 ? "-vn -c:a copy o" : "-vn -c:a aac -b:a 192k o"), t[k, 0] + ": " + a);
            }
            Check.That(string.Join(" ", CommandBuilder.AudioArgs("in.mkv", "pcm_s16le", "o.m4a", 256)).Contains("-b:a 256k"), "audio kbps");
            Check.That(string.Join(" ", CommandBuilder.Mp4AudioArgs("in.mkv", "opus", 1, "o.m4a", 96)).Contains("-b:a 96k"), "mp4 audio kbps");
        }

        static void TargetSizeArgs()
        {
            var s = new EncodeSettings();
            string plain = string.Join(" ", CommandBuilder.EncodeArgs(s, Sdr(), "o.ivf"));
            Check.Equal(plain, string.Join(" ", CommandBuilder.EncodeArgs(s, Sdr(), "o.ivf", null)), "no pass: as before");
            Check.Equal(plain + " --pass 1 --stats s.txt",
                string.Join(" ", CommandBuilder.EncodeArgs(s, Sdr(), "o.ivf", new PassArgs { Pass = 1, Stats = "s.txt" })), "pass 1");
            Check.Equal(plain + " --pass 2 --stats s.txt --target-size 9000000",
                string.Join(" ", CommandBuilder.EncodeArgs(s, Sdr(), "o.ivf", new PassArgs { Pass = 2, Stats = "s.txt", TargetBytes = 9000000 })), "pass 2");
        }

        static void VideoBudget()
        {
            Check.Equal(9000000L, CommandBuilder.VideoBudget(10000000, 1000000, 300, false), "separate files: size less the audio");
            Check.Equal(10000000L - 1000000 - CommandBuilder.ContainerBytes(300, true), CommandBuilder.VideoBudget(10000000, 1000000, 300, true), "mp4: less its boxes too");
            Check.That(CommandBuilder.ContainerBytes(300, true) > 0 && CommandBuilder.ContainerBytes(300, false) == 0, "only mp4 has boxes");
        }

        static void HeAacV2Args()
        {
            Check.Equal("-hide_banner -nostdin -loglevel error -i in.mkv -map 0:a:0 -vn -ac 2 -c:a pcm_s16le -t 2.5 -f wav a.wav",
                string.Join(" ", CommandBuilder.HeAacWavArgs("in.mkv", 2.5, "a.wav")), "wav for mp4: stereo, cut to the video");
            Check.Equal("-hide_banner -nostdin -loglevel error -i in.mkv -map 0:a:0 -vn -ac 2 -c:a pcm_s16le -f wav a.wav",
                string.Join(" ", CommandBuilder.HeAacWavArgs("in.mkv", null, "a.wav")), "wav for separate files");
            Check.Equal("-S -I -p 29 -b 32 -o o.m4a a.wav", string.Join(" ", CommandBuilder.FdkaacArgs("a.wav", "o.m4a", 32)), "fdkaac");
            Check.That(string.Join(" ", CommandBuilder.FdkaacArgs("a.wav", "o.m4a", 5)).Contains("-b 12 "), "kbps floor 12");
            Check.That(string.Join(" ", CommandBuilder.FdkaacArgs("a.wav", "o.m4a", 500)).Contains("-b 96 "), "kbps cap 96");
        }

        [DllImport("shell32.dll", SetLastError = true)]
        static extern IntPtr CommandLineToArgvW([MarshalAs(UnmanagedType.LPWStr)] string cmd, out int argc);
        [DllImport("kernel32.dll")]
        static extern IntPtr LocalFree(IntPtr p);

        static string[] ArgvOf(string cmdLine)
        {
            int n;
            IntPtr p = CommandLineToArgvW("prog.exe " + cmdLine, out n);
            if (p == IntPtr.Zero) throw new CheckFailure("CommandLineToArgvW failed");
            try
            {
                var r = new List<string>();
                for (int k = 1; k < n; k++) r.Add(Marshal.PtrToStringUni(Marshal.ReadIntPtr(p, k * IntPtr.Size)));
                return r.ToArray();
            }
            finally { LocalFree(p); }
        }

        static void QuotesRoundTrip()
        {
            string[] args =
            {
                @"C:\動画 フォルダ\a b.mp4", @"C:\動画 フォルダ\", @"C:\x\", @"say ""hi""", @"a\""b", @"a\\", "", "plain", @"back\slash", "tab\there", @"end\\""",
            };
            string[] back = ArgvOf(CommandBuilder.Join(args));
            Check.Equal(args.Length, back.Length, "argc");
            for (int k = 0; k < args.Length; k++) Check.Equal(args[k], back[k], "arg " + k);
            Check.Equal("plain", CommandBuilder.Quote("plain"), "unquoted when safe");
            Check.Equal("\"\"", CommandBuilder.Quote(""), "empty");
        }

        static void NamerAvoidsExisting()
        {
            string dir = Path.Combine(Path.GetTempPath(), "celv2gui_test_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                string input = Path.Combine(dir, "a.mkv");
                Check.Equal(Path.Combine(dir, "a"), OutputNamer.Stem(dir, input, ".m4a"), "free");
                File.WriteAllText(Path.Combine(dir, "a.ivf"), "");
                Check.Equal(Path.Combine(dir, "a (2)"), OutputNamer.Stem(dir, input, ".m4a"), "ivf exists");
                File.Delete(Path.Combine(dir, "a.ivf"));
                File.WriteAllText(Path.Combine(dir, "a.m4a"), "");
                Check.Equal(Path.Combine(dir, "a (2)"), OutputNamer.Stem(dir, input, ".m4a"), "audio exists");
                Check.Equal(Path.Combine(dir, "a"), OutputNamer.Stem(dir, input, ".mp3"), "other audio ext is free");
                File.Delete(Path.Combine(dir, "a.m4a"));
                File.WriteAllText(Path.Combine(dir, "a.mp4"), "");
                Check.Equal(Path.Combine(dir, "a (2)"), OutputNamer.Stem(dir, input, ".m4a"), "mp4 exists");
                File.WriteAllText(Path.Combine(dir, "a (2).ivf"), "");
                Check.Equal(Path.Combine(dir, "a (3)"), OutputNamer.Stem(dir, input, ".m4a"), "(2) taken");
            }
            finally { Directory.Delete(dir, true); }
        }

        static void NamerUsesGivenName()
        {
            string dir = Path.Combine(Path.GetTempPath(), "celv2gui_test_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                string input = Path.Combine(dir, "a.mkv");
                Check.Equal(Path.Combine(dir, "別の 名前"), OutputNamer.Stem(dir, input, ".m4a", "別の 名前"), "given name");
                Check.Equal(Path.Combine(dir, "b"), OutputNamer.Stem(dir, input, ".m4a", "  b  "), "trimmed");
                Check.Equal(Path.Combine(dir, "b"), OutputNamer.Stem(dir, input, ".m4a", "b.MP4"), "typed .mp4 dropped");
                Check.Equal(Path.Combine(dir, "b"), OutputNamer.Stem(dir, input, ".m4a", "b.ivf"), "typed .ivf dropped");
                Check.Equal(Path.Combine(dir, "b.v2"), OutputNamer.Stem(dir, input, ".m4a", "b.v2"), "other dots kept");
                Check.Equal(Path.Combine(dir, "a"), OutputNamer.Stem(dir, input, ".m4a", " "), "blank -> input name");
                Check.Equal(Path.Combine(dir, "a"), OutputNamer.Stem(dir, input, ".m4a", null), "null -> input name");
                File.WriteAllText(Path.Combine(dir, "b.mp4"), "");
                Check.Equal(Path.Combine(dir, "b (2)"), OutputNamer.Stem(dir, input, ".m4a", "b"), "given name never overwrites");
            }
            finally { Directory.Delete(dir, true); }
        }

        static void NamerRejectsBadNames()
        {
            foreach (string ok in new[] { null, "", " ", "動画 1", "a.b", "con1", "LPT", "b.mp4" })
                Check.Equal(null, OutputNamer.BadName(ok), "ok: " + ok);
            foreach (string bad in new[] { "a/b", @"a\b", "a:b", "a?", "a*", "a\"b", "<a>", "a|b", "a\tb", "a.", ".mp4", "CON", "nul", "Com1", "lpt9.txt", "aux.mp4" })
                Check.That(OutputNamer.BadName(bad) != null, "bad: " + bad);
        }

        static void Mp4CommandLines()
        {
            Check.Equal("x.obu -o y.mp4 --fps 24", string.Join(" ", CommandBuilder.Av2MuxArgs("x.obu", "y.mp4", 24, 1, null)), "mux");
            Check.Equal("x.obu -o y.mp4 --fps 23.976023976023978", string.Join(" ", CommandBuilder.Av2MuxArgs("x.obu", "y.mp4", 24000, 1001, null)), "mux ntsc");
            Check.Equal("x.obu -o y.mp4 --fps 24 --colr-override 1:1:1:0",
                string.Join(" ", CommandBuilder.Av2MuxArgs("x.obu", "y.mp4", 24, 1, "1:1:1:0")), "mux colr");
            Check.Equal("-noprog -add a.m4a y.mp4", string.Join(" ", CommandBuilder.Mp4BoxAddArgs("a.m4a", "y.mp4")), "mp4box");
            var old = System.Threading.Thread.CurrentThread.CurrentCulture;
            System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("de-DE");
            try
            {
                Check.Equal("-hide_banner -nostdin -loglevel error -i in.mkv -map 0:a:0 -vn -c:a copy -t 2.5 o.m4a",
                    string.Join(" ", CommandBuilder.Mp4AudioArgs("in.mkv", "aac", 2.5, "o.m4a")), "aac copy");
                Check.Equal("-hide_banner -nostdin -loglevel error -i in.mkv -map 0:a:0 -vn -c:a aac -b:a 192k -t 2.5 o.m4a",
                    string.Join(" ", CommandBuilder.Mp4AudioArgs("in.mkv", "opus", 2.5, "o.m4a")), "opus to aac");
            }
            finally { System.Threading.Thread.CurrentThread.CurrentCulture = old; }
        }

        static void ColrForInputs()
        {
            var i = Sdr();
            i.ColorPrimaries = "bt709"; i.ColorTransfer = "bt709"; i.ColorSpace = "bt709"; i.ColorRange = "tv";
            Check.Equal("1:1:1:0", CommandBuilder.ColrFor(i, Fmt), "bt709");
            var h = Sdr(); h.Hdr = true; h.ColorPrimaries = "bt2020"; h.ColorTransfer = "smpte2084"; h.ColorSpace = "bt2020nc"; h.ColorRange = "tv";
            Check.Equal("1:1:1:0", CommandBuilder.ColrFor(h, Eight), "hdr10 after tonemap");
            Check.Equal("9:16:9:0", CommandBuilder.ColrFor(h, Fmt), "hdr10 kept");
            Check.Equal("9:16:9:0", CommandBuilder.ColrFor(Probe("probe_hdr10.json"), Fmt), "hdr10 sample");
            Check.That(CommandBuilder.ColrFor(Sdr(), Fmt) == null, "unknown");
            Check.That(CommandBuilder.ColrFor(Probe("probe_no_frame_count.json"), Fmt) == null, "partial info");
            var s = Sdr();
            s.ColorPrimaries = "smpte170m"; s.ColorTransfer = "smpte170m"; s.ColorSpace = "smpte170m"; s.ColorRange = "tv";
            Check.Equal("6:6:6:0", CommandBuilder.ColrFor(s, Fmt), "bt601");
            s.ColorRange = "pc";
            Check.Equal("6:6:6:1", CommandBuilder.ColrFor(s, Fmt), "full range");
        }
    }
}
