using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using CelV2.Runner;
using CelV2.Settings;

namespace CelV2.Tests
{
    public static class RunnerTests
    {
        public static readonly (string name, Action run)[] All =
        {
            ("runner_stdin_has_no_bom_after_console_encoding_read", StdinNoBom),
            ("tool_locator_order", ToolLocatorOrder),
            ("runner_lists_devices", ListsDevices),
            ("hdr_needs_zscale_and_tonemap", HdrNeedsFilters),
            ("ivf_to_obu_strips_headers", IvfToObuStrips),
            ("runner_encodes_video_and_audio", EncodesVideoAndAudio),
            ("runner_he_aac_v2_audio", HeAacV2Audio),
            ("runner_target_size_separate", TargetSizeSeparate),
            ("runner_target_size_mp4", TargetSizeMp4),
            ("runner_target_size_too_small", TargetSizeTooSmall),
            ("runner_target_size_retries_after_mux", TargetSizeRetriesAfterMux),
            ("pass2_over_needs_celv2encs_line", Pass2OverNeedsLine),
            ("runner_target_size_audio_too_big", TargetSizeAudioTooBig),
            ("runner_he_aac_v2_falls_back_without_fdkaac", HeAacV2FallsBack),
            ("runner_video_only_input", VideoOnlyInput),
            ("runner_does_not_overwrite", DoesNotOverwrite),
            ("runner_uses_output_name", UsesOutputName),
            ("runner_japanese_paths", JapanesePaths),
            ("probe_reads_utf8_under_cp932_console", ProbeUtf8UnderCp932),
            ("runner_non_ansi_output_paths", NonAnsiPaths),
            ("path_guard_for_old_windows", PathGuard),
            ("runner_missing_output_is_not_ok", MissingOutputNotOk),
            ("runner_reports_unreadable_input", UnreadableInput),
            ("runner_cancel_stops_quickly", CancelStopsQuickly),
            ("runner_mp4_falls_back_without_mp4box", FallsBackWithoutMp4Box),
            ("runner_mp4_with_audio", Mp4WithAudio),
            ("runner_mp4_he_aac_v2", Mp4HeAacV2),
            ("runner_mp4_video_only", Mp4VideoOnly),
            ("runner_hdr10_source_passes_through", Hdr10PassesThrough),
            ("runner_mp4_mux_failure", Mp4MuxFailure),
            ("runner_mp4_audio_length_without_frame_count", Mp4AudioLength),
            ("runner_mp4_cancel_during_mux", Mp4CancelDuringMux),
            ("runner_mp4_non_ansi_output_paths", Mp4NonAnsiPaths),
        };

        // ---- helpers ----

        static string Env(string name)
        {
            string v = Environment.GetEnvironmentVariable(name);
            return string.IsNullOrEmpty(v) ? null : v;
        }

        static string BuildDir()
        {
            // gui\bin\CelV2.Tests.exe -> repo root is two levels up
            return Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "build"));
        }

        // av2_mux is the copy build.ps1 puts in guiin (with its UTF-8 manifest), the one the zip ships;
        // AVM_BIN only supplies avmdec / av2_demux.
        static ToolPaths Tools()
        {
            var t = ToolLocator.Find(null, Env("MP4BOX_DIR"), BuildDir(), AppDomain.CurrentDomain.BaseDirectory);
            Check.That(t.Ffmpeg != null && t.Ffprobe != null, "ffmpeg / ffprobe not found on PATH");
            Check.That(t.CelV2enc != null, "celv2enc.exe not found in " + BuildDir());
            return t;
        }

        static ToolPaths Mp4Tools()
        {
            if (Env("AVM_BIN") == null) throw new SkipTest("AVM_BIN not set");
            if (Env("MP4BOX_DIR") == null) throw new SkipTest("MP4BOX_DIR not set");
            var t = Tools();
            Check.That(t.Av2Mux != null, "av2_mux.exe not found in gui\\bin (build.ps1 copies it from AV2MUX_DIR, else AVM_BIN)");
            Check.That(t.Mp4Box != null, "MP4Box.exe not found in MP4BOX_DIR");
            return t;
        }

        static void WithTemp(Action<string> body)
        {
            string d = Path.Combine(Path.GetTempPath(), "celv2rt_" + Guid.NewGuid().ToString("N").Substring(0, 8));
            Directory.CreateDirectory(d);
            try { body(d); }
            finally { try { Directory.Delete(d, true); } catch (IOException) { } catch (UnauthorizedAccessException) { } }
        }

        static string Sub(string d, string name)
        {
            string p = Path.Combine(d, name);
            Directory.CreateDirectory(p);
            return p;
        }

        // Runs a tool and returns its exit code; stdout and stderr are merged into output.
        static int Exec(string exe, string argLine, out string output)
        {
            var psi = new ProcessStartInfo(exe, argLine)
            {
                UseShellExecute = false, CreateNoWindow = true,
                RedirectStandardOutput = true, RedirectStandardError = true,
            };
            var sb = new StringBuilder();
            using (var p = Process.Start(psi))
            {
                DataReceivedEventHandler h = (s, e) => { if (e.Data != null) lock (sb) sb.AppendLine(e.Data); };
                p.OutputDataReceived += h; p.ErrorDataReceived += h;
                p.BeginOutputReadLine(); p.BeginErrorReadLine();
                p.WaitForExit();
                output = sb.ToString();
                return p.ExitCode;
            }
        }

        static void ExecOk(string exe, string argLine)
        {
            string o;
            int rc = Exec(exe, argLine, out o);
            Check.That(rc == 0, Path.GetFileName(exe) + " exited " + rc + ": " + o);
        }

        static string Inv(double d) { return d.ToString("R", System.Globalization.CultureInfo.InvariantCulture); }

        // testsrc2 at 24 fps, optionally with a 440 Hz AAC track; 2 s = 48 frames.
        static string MakeMedia(string folder, string name, double seconds, bool audio, string size = "320x240")
        {
            string path = Path.Combine(folder, name);
            var a = new List<string> { "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=" + size + ":rate=24:duration=" + Inv(seconds) };
            if (audio) a.AddRange(new[] { "-f", "lavfi", "-i", "sine=frequency=440:duration=" + Inv(seconds) });
            a.AddRange(new[] { "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p" });
            if (audio) a.AddRange(new[] { "-c:a", "aac" });
            a.Add(path);
            ExecOk(Tools().Ffmpeg, CommandBuilder.Join(a));
            return path;
        }

        // Reference: the same command line the job runs, but through a plain cmd pipe.
        static void RefEncode(ToolPaths t, string input, InputInfo info, EncodeSettings s, string outIvf)
        {
            string line = CommandBuilder.Join(new[] { t.Ffmpeg }.Concat(CommandBuilder.DecodeArgs(input, info, s.Output)))
                + " | " + CommandBuilder.Join(new[] { t.CelV2enc }.Concat(CommandBuilder.EncodeArgs(s, info, outIvf)));
            ExecOk(Environment.GetEnvironmentVariable("ComSpec"), "/s /c \"" + line + "\"");
        }

        sealed class JobRun
        {
            public JobResult Result;
            public readonly List<JobProgress> Progress = new List<JobProgress>();
            public string Engine;
        }

        static JobRun RunJob(ToolPaths t, string input, EncodeSettings s, string outDir, InputInfo info = null, string outputName = null, IMuxStep mux = null)
        {
            info = info ?? Prober.Probe(t, input);
            var job = new EncodeJob(t, input, info, s, outDir, mux, outputName);
            var r = new JobRun();
            job.Progress += p => { lock (r.Progress) r.Progress.Add(p); };
            job.EngineLine += l => r.Engine = l;
            r.Result = job.Run(CancellationToken.None);
            return r;
        }

        static void Succeeded(JobRun r)
        {
            Check.That(r.Result.Ok, "job failed: " + r.Result.Error);
            Check.That(!r.Result.Cancelled, "job reported Cancelled");
        }

        static string[] Names(string dir)
        {
            return Directory.GetFiles(dir).Select(Path.GetFileName).OrderBy(n => n, StringComparer.Ordinal).ToArray();
        }

        static string Show(IEnumerable<string> names) { return "[" + string.Join(", ", names) + "]"; }

        static EncodeSettings Separate() { return new EncodeSettings { Container = ContainerKind.Separate }; }

        // ffprobe -show_streams, keys of one kind in stream order (e.g. all codec_tag_string values).
        static List<string> StreamKeys(ToolPaths t, string file, string key, string selectStreams = null)
        {
            string o;
            string sel = selectStreams == null ? "" : " -select_streams " + selectStreams;
            int rc = Exec(t.Ffprobe, "-v error" + sel + " -show_streams " + CommandBuilder.Quote(file), out o);
            Check.That(rc == 0, "ffprobe failed: " + o);
            return o.Split('\n').Select(l => l.Trim()).Where(l => l.StartsWith(key + "=", StringComparison.Ordinal))
                .Select(l => l.Substring(key.Length + 1)).ToList();
        }

        static void Throws(Action a, string msg)
        {
            try { a(); }
            catch (Exception) { return; }
            throw new CheckFailure(msg + ": expected an exception");
        }

        static byte[] Ivf(int frames, int payload, int rate, int scale)
        {
            var b = new List<byte>();
            Action<long, int> le = (v, n) => { for (int i = 0; i < n; i++) b.Add((byte)(v >> (8 * i))); };
            b.AddRange(Encoding.ASCII.GetBytes("DKIF")); le(0, 2); le(32, 2); b.AddRange(Encoding.ASCII.GetBytes("AV02"));
            le(320, 2); le(240, 2); le(rate, 4); le(scale, 4); le(frames, 4); le(0, 4);
            for (int f = 0; f < frames; f++) { le(payload, 4); le(f, 8); b.AddRange(new byte[payload]); }
            return b.ToArray();
        }

        // ---- tests ----

        static void StdinNoBom()
        {
            // .NET caches Console.InputEncoding; Process builds the redirected stdin writer from it (and a UTF-8 one
            // writes a BOM into the pipe). Read it first, as an app that touched the console would have.
            var enc = Console.InputEncoding;
            Check.That(enc != null, "console input encoding");
            var t = Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 0.25, false);
                string outDir = Sub(d, "out");
                var s = Separate();
                var info = Prober.Probe(t, input);
                Succeeded(RunJob(t, input, s, outDir, info));
                string refIvf = Path.Combine(d, "ref.ivf");
                RefEncode(t, input, info, s, refIvf);
                Check.That(File.ReadAllBytes(refIvf).SequenceEqual(File.ReadAllBytes(Path.Combine(outDir, "in.ivf"))), "IVF identical to the cmd pipe's");
            });
        }

        // The real celv2enc's list: 自動 first, CPU last; a missing celv2enc gives the fixed choices.
        static void ListsDevices()
        {
            var l = Prober.ListDevices(Tools());
            Check.That(l.Count >= 2, "auto and at least the CPU");
            Check.Equal("auto", l[0].Arg, "auto first");
            Check.Equal("cpu", l[l.Count - 1].Arg, "cpu last");
            Check.Equal(3, Prober.ListDevices(new ToolPaths()).Count, "no celv2enc: fallback");
        }

        static void ToolLocatorOrder()
        {
            WithTemp(d =>
            {
                string a = Sub(d, "a"), b = Sub(d, "b"), empty = Sub(d, "empty"), empty2 = Sub(d, "empty2");
                foreach (string dir in new[] { a, b })
                    foreach (string exe in new[] { "ffmpeg.exe", "ffprobe.exe", "MP4Box.exe", "celv2enc.exe", "av2_mux.exe", "fdkaac.exe" })
                        File.WriteAllBytes(Path.Combine(dir, exe), new byte[0]);

                var t = ToolLocator.Find(a, a, b, b);
                Check.Equal(Path.Combine(a, "ffmpeg.exe"), t.Ffmpeg, "configured ffmpeg wins over the exe folder");
                Check.Equal(Path.Combine(a, "ffprobe.exe"), t.Ffprobe, "configured ffprobe");
                Check.Equal(Path.Combine(a, "MP4Box.exe"), t.Mp4Box, "configured MP4Box");
                Check.Equal(Path.Combine(a, "fdkaac.exe"), t.Fdkaac, "fdkaac from the ffmpeg folder");
                Check.Equal(Path.Combine(b, "celv2enc.exe"), t.CelV2enc, "celv2enc from the tools folder");
                Check.Equal(Path.Combine(b, "av2_mux.exe"), t.Av2Mux, "av2_mux from its folder");

                t = ToolLocator.Find(null, null, b, null);
                Check.Equal(Path.Combine(b, "ffmpeg.exe"), t.Ffmpeg, "exe folder ffmpeg");
                Check.Equal(Path.Combine(b, "MP4Box.exe"), t.Mp4Box, "exe folder MP4Box");
                Check.Equal(Path.Combine(b, "fdkaac.exe"), t.Fdkaac, "exe folder fdkaac");
                Check.Equal(Path.Combine(b, "av2_mux.exe"), t.Av2Mux, "av2_mux falls back to the tools folder");

                t = ToolLocator.Find(empty, empty, empty2, null);
                Check.That(t.Ffmpeg != null && Path.GetFileName(t.Ffmpeg).Equals("ffmpeg.exe", StringComparison.OrdinalIgnoreCase)
                    && !t.Ffmpeg.StartsWith(d, StringComparison.OrdinalIgnoreCase), "ffmpeg found on PATH: " + t.Ffmpeg);
                Check.That(t.CelV2enc == null, "celv2enc is only looked for in the tools folder");
                Check.That(t.Av2Mux == null, "av2_mux is only looked for in its folders");
            });
        }

        static void HdrNeedsFilters()
        {
            var t = Tools();
            string filters;
            ExecOk(t.Ffmpeg, "-hide_banner -filters");
            Exec(t.Ffmpeg, "-hide_banner -filters", out filters);
            Check.That(Prober.ParseHasFilters(filters, "zscale", "tonemap"), "this PC's ffmpeg has zscale and tonemap");
            string without = string.Join("\n", filters.Split('\n').Where(l => !System.Text.RegularExpressions.Regex.IsMatch(l, @"^\s*\S+\s+zscale\s")));
            Check.That(without.Length < filters.Length, "sample really lost its zscale line");
            Check.That(!Prober.ParseHasFilters(without, "zscale", "tonemap"), "no zscale -> false");
            Check.That(Prober.ParseHasFilters(without, "tonemap"), "tonemap alone still true");
            Check.That(!Prober.ParseHasFilters(" ... tonemap_opencl     V->V       OpenCL tone mapping\n", "tonemap"), "prefix is not a match");

            WithTemp(d =>
            {
                string fake = Path.Combine(d, "ffmpeg.bat"), marker = Path.Combine(d, "started.txt"), enc = Path.Combine(d, "celv2enc.bat");
                File.WriteAllText(fake, "@echo off\r\necho Filters:\r\necho  T.. tonemap           V-^>V       Conversion between dynamic ranges.\r\n");
                File.WriteAllText(enc, "@echo off\r\necho x> \"" + marker + "\"\r\n");
                var tools = new ToolPaths { Ffmpeg = fake, Ffprobe = t.Ffprobe, CelV2enc = enc };
                string outDir = Sub(d, "out");
                var info = new InputInfo { Width = 320, Height = 240, FpsNum = 24, FpsDen = 1, BitDepth = 10, Hdr = true, FrameCount = 48 };
                var eight = Separate(); eight.Output.BitDepth = 8; // at 10 bits HDR passes through and needs neither
                var r = new EncodeJob(tools, Path.Combine(d, "in.mkv"), info, eight, outDir, null).Run(CancellationToken.None);
                Check.That(!r.Ok, "HDR without zscale must fail");
                Check.That(r.Error != null && r.Error.Contains("zscale"), "Error mentions zscale: " + r.Error);
                Check.That(!File.Exists(marker), "no process was started");
                Check.Equal(0, Names(outDir).Length, "output files");
            });
        }

        static void IvfToObuStrips()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 0.125, false);   // exactly 3 frames
                var info = Prober.Probe(t, input);
                string ivf = Path.Combine(d, "a.ivf"), obu = Path.Combine(d, "a.obu");
                RefEncode(t, input, info, Separate(), ivf);
                int rate, scale;
                IvfToObu.Convert(ivf, obu, out rate, out scale);
                Check.Equal(24, rate, "rate");
                Check.Equal(1, scale, "scale");
                byte[] raw = File.ReadAllBytes(ivf);
                Check.Equal(raw.Length - 32 - 12 * 3, (int)new FileInfo(obu).Length, "obu size");
                Check.Equal(3, BitConverter.ToInt32(raw, 24), "frame count in the header (sanity)");

                // truncated payload and truncated frame header both throw
                string cut = Path.Combine(d, "cut.ivf");
                File.WriteAllBytes(cut, raw.Take(raw.Length - 5).ToArray());
                Throws(() => { int r2, s2; IvfToObu.Convert(cut, Path.Combine(d, "cut.obu"), out r2, out s2); }, "truncated payload");
                File.WriteAllBytes(cut, raw.Take(32 + 12 + BitConverter.ToInt32(raw, 32) + 7).ToArray());
                Throws(() => { int r2, s2; IvfToObu.Convert(cut, Path.Combine(d, "cut.obu"), out r2, out s2); }, "truncated frame header");
                File.WriteAllText(cut, "not an ivf file at all, just text padding.......");
                Throws(() => { int r2, s2; IvfToObu.Convert(cut, Path.Combine(d, "cut.obu"), out r2, out s2); }, "not IVF");
            });
        }

        // FDKAAC_DIR: the folder of fdkaac.exe (nu774/fdkaac with libfdk-aac).
        static ToolPaths FdkaacTools()
        {
            if (Env("FDKAAC_DIR") == null) throw new SkipTest("FDKAAC_DIR not set");
            var t = Tools();
            t.Fdkaac = Path.Combine(Env("FDKAAC_DIR"), "fdkaac.exe");
            Check.That(File.Exists(t.Fdkaac), "fdkaac.exe not found in FDKAAC_DIR");
            return t;
        }

        // A mono AAC source: re-encoded (never copied) to stereo HE-AACv2 at the set kbps, and no temporary file left.
        static void HeAacV2Audio()
        {
            var t = FdkaacTools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 2, true);
                string outDir = Sub(d, "out");
                var s = Separate();
                s.HeAacV2 = true;
                s.HeAacKbps = 24;
                int temps = Directory.GetFiles(Path.GetTempPath(), "celv2_*.wav").Length;
                var r = RunJob(t, input, s, outDir);
                Succeeded(r);
                Check.That(r.Result.Note == null, "no note: " + r.Result.Note);
                Check.Equal(Show(new[] { "in.ivf", "in.m4a" }), Show(Names(outDir)), "files");
                string m4a = Path.Combine(outDir, "in.m4a");
                Check.Equal("HE-AACv2", StreamKeys(t, m4a, "profile").Single(), "profile");
                Check.Equal("2", StreamKeys(t, m4a, "channels").Single(), "stereo");
                long kbps = long.Parse(StreamKeys(t, m4a, "bit_rate").Single()) / 1000;
                Check.That(kbps >= 20 && kbps <= 28, "about 24 kbps: " + kbps);
                Check.Equal(temps, Directory.GetFiles(Path.GetTempPath(), "celv2_*.wav").Length, "temporary WAV deleted");
            });
        }

        static void Mp4HeAacV2()
        {
            var t = Mp4Tools();
            t.Fdkaac = FdkaacTools().Fdkaac;
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 2, true);
                string outDir = Sub(d, "out");
                var r = RunJob(t, input, new EncodeSettings { HeAacV2 = true }, outDir);
                Succeeded(r);
                string mp4 = Path.Combine(outDir, "in.mp4");
                Check.Equal(Show(new[] { "in.mp4" }), Show(Names(outDir)), "files");
                Check.Equal("HE-AACv2", StreamKeys(t, mp4, "profile", "a").Single(), "audio profile");
                double sec = double.Parse(StreamKeys(t, mp4, "duration", "a").Single(), System.Globalization.CultureInfo.InvariantCulture);
                Check.That(Math.Abs(sec - 2) < 0.2, "audio cut to the video's 2 s: " + sec);
            });
        }

        static int TempLeftovers()
        {
            return Directory.GetFiles(Path.GetTempPath(), "celv2_*.stats").Length + Directory.GetFiles(Path.GetTempPath(), "celv2_*_pass1.ivf").Length;
        }

        static long TotalBytes(string dir) { return Directory.GetFiles(dir).Sum(f => new FileInfo(f).Length); }

        // The audio plus 3/4 of the video's plain size: the whole output fits, near the target, and no
        // temporary file is left.
        static void TargetSizeSeparate()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 2, true);
                string plain = Sub(d, "plain"), outDir = Sub(d, "out");
                Succeeded(RunJob(t, input, Separate(), plain));
                long target = new FileInfo(Path.Combine(plain, "in.m4a")).Length + new FileInfo(Path.Combine(plain, "in.ivf")).Length * 3 / 4;
                var s = Separate();
                s.TargetSizeMB = target / 1e6;
                int temps = TempLeftovers();
                var r = RunJob(t, input, s, outDir);
                Succeeded(r);
                Check.Equal(Show(new[] { "in.ivf", "in.m4a" }), Show(Names(outDir)), "files");
                long total = TotalBytes(outDir);
                Check.That(total <= target, "fits: " + total + " of " + target);
                Check.That(total >= target * 7 / 10, "near the target: " + total + " of " + target);
                Check.That(r.Result.Note != null && r.Result.Note.Contains("サイズ指定"), "note: " + r.Result.Note);
                Check.Equal(temps, TempLeftovers(), "temporary files deleted");
            });
        }

        static void TargetSizeMp4()
        {
            var t = Mp4Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 2, true);
                string plain = Sub(d, "plain"), outDir = Sub(d, "out");
                Succeeded(RunJob(t, input, new EncodeSettings(), plain));
                long target = new FileInfo(Path.Combine(plain, "in.mp4")).Length * 85 / 100;
                var r = RunJob(t, input, new EncodeSettings { TargetSizeMB = target / 1e6 }, outDir);
                Succeeded(r);
                Check.Equal(Show(new[] { "in.mp4" }), Show(Names(outDir)), "files");
                Check.That(TotalBytes(outDir) <= target, "fits: " + TotalBytes(outDir) + " of " + target);
            });
        }

        // 1,000 bytes for 48 frames: celv2enc goes over every time; the job fails with the reason and leaves nothing.
        static void TargetSizeTooSmall()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 2, false);
                string outDir = Sub(d, "out");
                var s = Separate();
                s.TargetSizeMB = 0.001;
                int temps = TempLeftovers();
                var r = RunJob(t, input, s, outDir);
                Check.That(!r.Result.Ok && r.Result.Error != null && r.Result.Error.Contains("指定サイズ") && r.Result.Error.Contains("qindex 255"),
                    "error names qindex 255: " + r.Result.Error);
                Check.Equal(0, Names(outDir).Length, "nothing left in the output folder");
                Check.Equal(temps, TempLeftovers(), "temporary files deleted");
            });
        }

        // Separate files whose first mux comes out 5,000 bytes over (as an MP4's boxes might): the outputs go,
        // the audio is made again, pass 2 runs at a smaller budget, and the second whole fits.
        sealed class OverOnceMux : IMuxStep
        {
            public int Calls;
            public IReadOnlyList<string> Finish(string videoIvf, string audioOrNull, CancellationToken cancel)
            {
                if (Calls++ == 0) using (var f = new FileStream(videoIvf, FileMode.Append)) f.Write(new byte[5000], 0, 5000);
                return new SeparateFilesMux().Finish(videoIvf, audioOrNull, cancel);
            }
        }

        static void TargetSizeRetriesAfterMux()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 2, true);
                string plain = Sub(d, "plain"), outDir = Sub(d, "out");
                Succeeded(RunJob(t, input, Separate(), plain));
                long target = new FileInfo(Path.Combine(plain, "in.m4a")).Length + new FileInfo(Path.Combine(plain, "in.ivf")).Length * 3 / 4;
                var s = Separate();
                s.TargetSizeMB = target / 1e6;
                var mux = new OverOnceMux();
                var r = RunJob(t, input, s, outDir, null, null, mux);
                Succeeded(r);
                Check.Equal(2, mux.Calls, "muxed twice");
                Check.Equal(Show(new[] { "in.ivf", "in.m4a" }), Show(Names(outDir)), "the audio made again");
                Check.That(TotalBytes(outDir) <= target, "fits: " + TotalBytes(outDir) + " of " + target);
            });
        }

        // celv2enc's exit 3 means over only with its line: an abort() also exits with 3 on Windows.
        static void Pass2OverNeedsLine()
        {
            var two = new PassArgs { Pass = 2, Stats = "s", TargetBytes = 1000 };
            Check.Equal(OverKind.Over, EncodeJob.Pass2Over(two, 3, new[] { "speed: 1", "celv2enc: over --target-size by 5 bytes" }), "over");
            Check.Equal(OverKind.Coarsest, EncodeJob.Pass2Over(two, 3, new[] { "celv2enc: over --target-size by 5 bytes even at qindex 255" }), "at 255");
            Check.Equal(OverKind.None, EncodeJob.Pass2Over(two, 3, new[] { "abnormal program termination" }), "a crash is not over");
            Check.Equal(OverKind.None, EncodeJob.Pass2Over(new PassArgs { Pass = 1, Stats = "s" }, 3, new[] { "celv2enc: over --target-size by 5 bytes" }), "pass 1");
            Check.Equal(OverKind.None, EncodeJob.Pass2Over(two, 0, new string[0]), "fits");
            // Over a retry's smaller budget but within the target itself: muxed and checked, not retried again.
            Check.That(EncodeJob.MayFit(900, 50, 10, false, 1000), "separate files within the target");
            Check.That(!EncodeJob.MayFit(960, 50, 10, false, 1000), "over the target");
            Check.That(!EncodeJob.MayFit(900, 50, 10, true, 1000), "MP4's boxes push it over");
        }

        static void TargetSizeAudioTooBig()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 2, true);
                string outDir = Sub(d, "out");
                var s = Separate();
                s.TargetSizeMB = 0.01;  // the 2 s of AAC alone are over 30,000 bytes
                var r = RunJob(t, input, s, outDir);
                Check.That(!r.Result.Ok && r.Result.Error != null && r.Result.Error.Contains("音声だけで指定サイズを超えます"), "error: " + r.Result.Error);
                Check.Equal(0, Names(outDir).Length, "nothing left in the output folder");
            });
        }

        static void HeAacV2FallsBack()
        {
            var t = Tools();
            t.Fdkaac = null;
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 2, true);
                string outDir = Sub(d, "out");
                var s = Separate();
                s.HeAacV2 = true;
                var r = RunJob(t, input, s, outDir);
                Succeeded(r);
                Check.That(r.Result.Note != null && r.Result.Note.Contains("fdkaac"), "note names fdkaac: " + r.Result.Note);
                Check.Equal("LC", StreamKeys(t, Path.Combine(outDir, "in.m4a"), "profile").Single(), "the AAC copied as before");
            });
        }

        static void EncodesVideoAndAudio()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 2, true);
                string outDir = Sub(d, "out");
                var s = Separate();
                var info = Prober.Probe(t, input);
                Check.Equal(48L, info.FrameCount, "probed frames");
                var r = RunJob(t, input, s, outDir, info);
                Succeeded(r);
                Check.Equal(Show(new[] { "in.ivf", "in.m4a" }), Show(Names(outDir)), "files in the output folder");
                Check.Equal(2, r.Result.Outputs.Count, "outputs");
                Check.Equal(Path.Combine(outDir, "in.ivf"), r.Result.Outputs[0], "first output");
                Check.Equal(Path.Combine(outDir, "in.m4a"), r.Result.Outputs[1], "second output");
                Check.Equal(48L, r.Progress.Last().Frame, "last progress frame");
                Check.Equal(48L, r.Progress.Last().Total, "progress total");
                Check.That(r.Engine != null && r.Engine.StartsWith("celv2: decisions on"), "EngineLine: " + r.Engine);
                Check.Equal(r.Engine, r.Result.EngineLine, "result EngineLine");

                string refIvf = Path.Combine(d, "ref.ivf");
                RefEncode(t, input, info, s, refIvf);
                Check.That(File.ReadAllBytes(refIvf).SequenceEqual(File.ReadAllBytes(Path.Combine(outDir, "in.ivf"))), "IVF identical to the cmd pipe's");
                Check.Equal("aac", StreamKeys(t, Path.Combine(outDir, "in.m4a"), "codec_name").Single(), "audio codec");
            });
        }

        static void VideoOnlyInput()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 2, false);
                string outDir = Sub(d, "out");
                var r = RunJob(t, input, Separate(), outDir);
                Succeeded(r);
                Check.Equal(Show(new[] { "in.ivf" }), Show(Names(outDir)), "files");
                Check.Equal(1, r.Result.Outputs.Count, "outputs");
            });
        }

        static void DoesNotOverwrite()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 0.5, false);
                string outDir = Sub(d, "out");
                byte[] keep = Encoding.ASCII.GetBytes("keep me");
                File.WriteAllBytes(Path.Combine(outDir, "in.ivf"), keep);
                var r = RunJob(t, input, Separate(), outDir);
                Succeeded(r);
                Check.Equal(Path.Combine(outDir, "in (2).ivf"), r.Result.Outputs.Single(), "shifted name");
                Check.That(File.ReadAllBytes(Path.Combine(outDir, "in.ivf")).SequenceEqual(keep), "original untouched");
            });
        }

        static void UsesOutputName()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 0.5, true);
                string outDir = Sub(d, "out");
                var r = RunJob(t, input, Separate(), outDir, null, "別名 テスト");
                Succeeded(r);
                Check.Equal(Show(new[] { "別名 テスト.ivf", "別名 テスト.m4a" }), Show(Names(outDir)), "files");
            });
        }

        static void JapanesePaths()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string inDir = Sub(d, "テスト 動画"), outDir = Sub(d, Path.Combine("out", "テスト 動画"));
                string input = MakeMedia(inDir, "in.mp4", 1, true);
                var r = RunJob(t, input, Separate(), outDir);
                Succeeded(r);
                Check.Equal(Show(new[] { "in.ivf", "in.m4a" }), Show(Names(outDir)), "files");
            });
        }

        // A GUI process without a console decodes tool output with CP932 unless told otherwise; ffprobe writes UTF-8.
        static void ProbeUtf8UnderCp932()
        {
            var t = Tools();
            var saved = Console.OutputEncoding;
            try
            {
                Console.OutputEncoding = Encoding.GetEncoding(932);
                WithTemp(d =>
                {
                    string input = Path.Combine(Sub(d, "テスト"), "in.mp4");
                    ExecOk(t.Ffmpeg, CommandBuilder.Join(new[] { "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:duration=0.25",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-metadata", "title=テスト", input }));
                    var info = Prober.Probe(t, input);
                    Check.Equal(320, info.Width, "width");
                });
            }
            finally { Console.OutputEncoding = saved; }
        }

        // Names outside the ANSI code page (CP932 here): Windows best-fit would turn Pokémon into Pokemon for an ANSI exe.
        static void NonAnsiCases(ToolPaths t, EncodeSettings s, string ext)
        {
            WithTemp(d =>
            {
                string input = MakeMedia(d, "Pokémon.mp4", 1, true);
                string outDir = Sub(d, "out");
                byte[] keep = Encoding.ASCII.GetBytes("best-fit name, keep me");
                string decoy = Path.Combine(outDir, "Pokemon" + ext);
                File.WriteAllBytes(decoy, keep);
                var r = RunJob(t, input, s, outDir);
                Succeeded(r);
                string want = Path.Combine(outDir, "Pokémon" + ext);
                Check.That(File.Exists(want), "exact-name output exists: " + Show(Names(outDir)));
                Check.That(File.ReadAllBytes(decoy).SequenceEqual(keep), "Pokemon" + ext + " untouched");
                Check.That(r.Result.Outputs.Contains(want), "Outputs lists the exact name: " + Show(r.Result.Outputs));

                string uniDir = Sub(d, "Pokémon 🎬");
                r = RunJob(t, input, s, uniDir);
                Succeeded(r);
                Check.That(r.Result.Outputs.All(File.Exists), "outputs exist: " + Show(r.Result.Outputs));
                Check.That(Names(uniDir).Contains("Pokémon" + ext), "in the Unicode folder: " + Show(Names(uniDir)));
                if (ext == ".mp4")
                {
                    Check.Equal(Show(new[] { "Pokemon.mp4", "Pokémon.mp4" }), Show(Names(outDir)), "no orphans");
                    Check.Equal(Show(new[] { "Pokémon.mp4" }), Show(Names(uniDir)), "no orphans in the Unicode folder");
                    foreach (string mp4 in new[] { want, Path.Combine(uniDir, "Pokémon.mp4") })
                        Check.Equal("av02,mp4a", string.Join(",", StreamKeys(t, mp4, "codec_tag_string").OrderBy(x => x, StringComparer.Ordinal)), "tracks");
                }
            });
        }

        static void NonAnsiPaths() { NonAnsiCases(Tools(), Separate(), ".ivf"); }

        static void Mp4NonAnsiPaths() { NonAnsiCases(Mp4Tools(), new EncodeSettings(), ".mp4"); }

        static void PathGuard()
        {
            var sjis = Encoding.GetEncoding(932);
            Check.Equal("Pokémon 🎬", EncodeJob.UnsupportedName(17763, sjis, @"C:\Pokémon 🎬\in"), "old Windows, é in a folder");
            Check.Equal("Pokémon", EncodeJob.UnsupportedName(17763, sjis, @"C:\テスト\Pokémon"), "old Windows, é in the stem");
            Check.Equal(null, EncodeJob.UnsupportedName(17763, sjis, @"C:\テスト 動画\in"), "Japanese fits CP932");
            Check.Equal(null, EncodeJob.UnsupportedName(18362, sjis, @"C:\Pokémon 🎬\in"), "1903+: the UTF-8 manifest applies");
            Check.Equal("テスト", EncodeJob.UnsupportedName(17763, Encoding.GetEncoding(1252), @"C:\テスト\in"), "Japanese on a CP1252 PC");
        }

        // A mux step that claims a file it never wrote.
        sealed class LyingMux : IMuxStep
        {
            public IReadOnlyList<string> Finish(string videoIvf, string audioOrNull, CancellationToken cancel)
            {
                return new[] { videoIvf, videoIvf + ".missing" };
            }
        }

        static void MissingOutputNotOk()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 0.25, false);
                string outDir = Sub(d, "out");
                var res = new EncodeJob(t, input, Prober.Probe(t, input), Separate(), outDir, new LyingMux()).Run(CancellationToken.None);
                Check.That(!res.Ok && !res.Cancelled, "Ok=" + res.Ok + " Cancelled=" + res.Cancelled);
                Check.That(res.Error != null && res.Error.Contains("in.ivf.missing"), "Error names the missing file: " + res.Error);
                Check.Equal(0, res.Outputs.Count, "outputs");
                Check.Equal(0, Names(outDir).Length, "partials deleted: " + Show(Names(outDir)));
            });
        }

        static void UnreadableInput()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string bad = Path.Combine(d, "bad.mp4");
                File.WriteAllText(bad, "this is not a video, just some text\n");
                string msg = null;
                try { Prober.Probe(t, bad); }
                catch (Exception e) { msg = e.Message; }
                Check.That(!string.IsNullOrWhiteSpace(msg), "Probe throws with a reason");

                string outDir = Sub(d, "out");
                var fake = new InputInfo { Width = 320, Height = 240, FpsNum = 24, FpsDen = 1, BitDepth = 8, FrameCount = 48 };
                foreach (var s in new[] { Separate(), new EncodeSettings() })
                {
                    var r = RunJob(t, bad, s, outDir, fake);
                    Check.That(!r.Result.Ok, "undecodable input must not be Ok");
                    Check.That(!r.Result.Cancelled, "not cancelled");
                    Check.That(!string.IsNullOrWhiteSpace(r.Result.Error), "Error is set");
                    Check.Equal(0, Names(outDir).Length, "no output files left: " + Show(Names(outDir)));
                }
            });
        }

        static HashSet<int> ChildPids()
        {
            return new HashSet<int>(Process.GetProcessesByName("celv2enc").Concat(Process.GetProcessesByName("ffmpeg")).Select(p => p.Id));
        }

        static void CancelStopsQuickly()
        {
            var t = Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "long.mp4", 20, true, "1280x720");
                string outDir = Sub(d, "out");
                var info = Prober.Probe(t, input);
                var before = ChildPids();
                var job = new EncodeJob(t, input, info, Separate(), outDir, null);
                var first = new ManualResetEvent(false);
                job.Progress += p => first.Set();
                var cts = new CancellationTokenSource();
                JobResult res = null; Exception ex = null;
                var th = new Thread(() => { try { res = job.Run(cts.Token); } catch (Exception e) { ex = e; } });
                th.Start();
                long ms = -1;
                try
                {
                    Check.That(first.WaitOne(120000), "no progress within 120 s");
                    var sw = Stopwatch.StartNew();
                    cts.Cancel();
                    Check.That(th.Join(10000), "Run did not return after cancel");
                    ms = sw.ElapsedMilliseconds;
                }
                finally { cts.Cancel(); th.Join(); }
                Check.That(ex == null, "Run threw: " + ex);
                Check.That(ms < 2000, "Run took " + ms + " ms to return");
                Check.That(res.Cancelled, "Cancelled");
                Check.That(!res.Ok, "not Ok");
                Check.That(ChildPids().Except(before).Count() == 0, "child processes still running");
                Check.Equal(0, Names(outDir).Length, "partial outputs left: " + Show(Names(outDir)));
            });
        }

        static void FallsBackWithoutMp4Box()
        {
            var t0 = Tools();
            var t = new ToolPaths { Ffmpeg = t0.Ffmpeg, Ffprobe = t0.Ffprobe, CelV2enc = t0.CelV2enc, Av2Mux = t0.Av2Mux, Mp4Box = null };
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 1, true);
                string outDir = Sub(d, "out");
                var r = RunJob(t, input, new EncodeSettings(), outDir);   // Container defaults to Mp4
                Succeeded(r);
                Check.Equal(Show(new[] { "in.ivf", "in.m4a" }), Show(Names(outDir)), "files");
                Check.Equal(2, r.Result.Outputs.Count, "outputs");
                Check.That(r.Result.Note != null && r.Result.Note.Contains("別ファイル"), "Note: " + r.Result.Note);
            });
        }

        static void Mp4WithAudio()
        {
            var t = Mp4Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 2, true);
                string outDir = Sub(d, "out");
                var s = new EncodeSettings();
                var info = Prober.Probe(t, input);
                var r = RunJob(t, input, s, outDir, info);
                Succeeded(r);
                Check.That(r.Result.Note == null, "no fallback note: " + r.Result.Note);
                string mp4 = Path.Combine(outDir, "in.mp4");
                Check.Equal(Show(new[] { "in.mp4" }), Show(Names(outDir)), "files");
                Check.Equal(mp4, r.Result.Outputs.Single(), "outputs");
                Check.Equal("av02,mp4a", string.Join(",", StreamKeys(t, mp4, "codec_tag_string").OrderBy(x => x, StringComparer.Ordinal)), "tracks");
                var fps = StreamKeys(t, mp4, "r_frame_rate", "v").Concat(StreamKeys(t, mp4, "avg_frame_rate", "v"));
                Check.That(fps.Contains("24/1"), "video frame rate 24/1: " + string.Join(" ", fps));

                // the muxed stream decodes to the same pictures as the plain IVF
                string avm = Env("AVM_BIN");
                string demuxed = Path.Combine(d, "back.obu"), refIvf = Path.Combine(d, "ref.ivf");
                ExecOk(Path.Combine(avm, "av2_demux.exe"), CommandBuilder.Join(new[] { mp4, "-o", demuxed }));
                RefEncode(t, input, info, s, refIvf);
                string yuvA = Path.Combine(d, "a.yuv"), yuvB = Path.Combine(d, "b.yuv");
                ExecOk(Path.Combine(avm, "avmdec.exe"), CommandBuilder.Join(new[] { "--i420", "-o", yuvA, demuxed }));
                ExecOk(Path.Combine(avm, "avmdec.exe"), CommandBuilder.Join(new[] { "--i420", "-o", yuvB, refIvf }));
                var a = File.ReadAllBytes(yuvA);
                Check.Equal(320 * 240 * 3 / 2 * 48 * 2, a.Length, "decoded size (16-bit samples: celv2enc codes 10 bits by default)");
                Check.That(a.SequenceEqual(File.ReadAllBytes(yuvB)), "MP4 decodes identically to the IVF");
            });
        }

        // An HDR10 source (x265, PQ, with mastering display and content light level SEI) stays HDR at 10 bits.
        static void Hdr10PassesThrough()
        {
            var t = Mp4Tools();
            WithTemp(d =>
            {
                string input = Path.Combine(d, "pq.mkv");
                ExecOk(t.Ffmpeg, "-hide_banner -loglevel error -y -f lavfi -i testsrc2=size=64x48:rate=24:duration=0.5 -c:v libx265 -preset ultrafast " +
                       "-pix_fmt yuv420p10le -x265-params " + // the colour tags through x265's own parameters: ffmpeg 9 drops -color_trc for libx265
                       "\"log-level=error:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400\" " +
                       CommandBuilder.Quote(input));
                var info = Prober.Probe(t, input);
                Check.That(info.Hdr && info.BitDepth == 10, "probed as HDR10");
                Check.Equal("G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)", info.MasteringDisplay, "mastering display from the first frame");
                Check.Equal("1000,400", info.ContentLight, "content light level");
                string outDir = Sub(d, "out");
                var r = RunJob(t, input, new EncodeSettings(), outDir, info);
                Succeeded(r);
                string mp4 = Path.Combine(outDir, "pq.mp4");
                Check.Equal("smpte2084", StreamKeys(t, mp4, "color_transfer", "v").Single(), "mp4 colr: PQ");
                Check.Equal("bt2020", StreamKeys(t, mp4, "color_primaries", "v").Single(), "mp4 colr: BT.2020");
                // the CLL metadata OBU payload (1000, 400 big-endian) is in the stream
                var bytes = File.ReadAllBytes(mp4);
                byte[] cll = { 0x03, 0xE8, 0x01, 0x90 };
                Check.That(Enumerable.Range(0, bytes.Length - 3).Any(k => bytes[k] == cll[0] && bytes.Skip(k).Take(4).SequenceEqual(cll)), "CLL metadata in the stream");

                string avm = Env("AVM_BIN");
                string obu = Path.Combine(d, "back.obu"), yuv = Path.Combine(d, "back.yuv");
                ExecOk(Path.Combine(avm, "av2_demux.exe"), CommandBuilder.Join(new[] { mp4, "-o", obu }));
                ExecOk(Path.Combine(avm, "avmdec.exe"), CommandBuilder.Join(new[] { "--i420", "-o", yuv, obu }));
                Check.Equal(64 * 48 * 3 / 2 * 12 * 2, File.ReadAllBytes(yuv).Length, "12 frames of 16-bit samples");
            });
        }

        static void Mp4VideoOnly()
        {
            var t = Mp4Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 1, false);
                string outDir = Sub(d, "out");
                var r = RunJob(t, input, new EncodeSettings(), outDir);
                Succeeded(r);
                Check.Equal(Show(new[] { "in.mp4" }), Show(Names(outDir)), "files");
                Check.Equal("av02", string.Join(",", StreamKeys(t, Path.Combine(outDir, "in.mp4"), "codec_tag_string")), "tracks");
            });
        }

        static void Mp4MuxFailure()
        {
            var t = Mp4Tools();
            WithTemp(d =>
            {
                string ivf = Path.Combine(d, "fake.ivf");
                File.WriteAllBytes(ivf, Ivf(2, 64, 24, 1));            // valid IVF, payload all zeros -> not an AV2 stream
                string mp4 = Path.Combine(d, "x.mp4");
                Throws(() => new Mp4Mux(t, null, mp4).Finish(ivf, null, CancellationToken.None), "zero-filled bitstream");
                Check.Equal(Show(new[] { "fake.ivf" }), Show(Names(d)), "no .mp4 / .obu left behind");
            });
        }

        static void Mp4AudioLength()
        {
            var t = Mp4Tools();
            WithTemp(d =>
            {
                string input = MakeMedia(d, "in.mp4", 2, true);
                string outDir = Sub(d, "out");
                var info = Prober.Probe(t, input);
                info.FrameCount = 0;                                   // unknown: the cut must come from the frames actually coded
                var r = RunJob(t, input, new EncodeSettings(), outDir, info);
                Succeeded(r);
                string mp4 = Path.Combine(outDir, "in.mp4");
                double v = double.Parse(StreamKeys(t, mp4, "duration", "v").Single(), System.Globalization.CultureInfo.InvariantCulture);
                double a = double.Parse(StreamKeys(t, mp4, "duration", "a").Single(), System.Globalization.CultureInfo.InvariantCulture);
                Check.That(a > 1.5 && Math.Abs(a - v) < 0.1, "audio " + a + " s vs video " + v + " s");
            });
        }

        // Cancels the job just before or just after the real mux step runs.
        sealed class CancelingMux : IMuxStep
        {
            readonly IMuxStep inner; readonly CancellationTokenSource cts; readonly bool before;
            public CancelingMux(IMuxStep inner, CancellationTokenSource cts, bool before) { this.inner = inner; this.cts = cts; this.before = before; }
            public IReadOnlyList<string> Finish(string videoIvf, string audioOrNull, CancellationToken cancel)
            {
                if (before) cts.Cancel();
                var r = inner.Finish(videoIvf, audioOrNull, cancel);
                if (!before) cts.Cancel();
                return r;
            }
        }

        static void Mp4CancelDuringMux()
        {
            var t = Mp4Tools();
            foreach (bool before in new[] { true, false })
                WithTemp(d =>
                {
                    string input = MakeMedia(d, "in.mp4", 1, true);
                    string outDir = Sub(d, "out");
                    var info = Prober.Probe(t, input);
                    var cts = new CancellationTokenSource();
                    var mux = new CancelingMux(new Mp4Mux(t, null, Path.Combine(outDir, "in.mp4")), cts, before);
                    var res = new EncodeJob(t, input, info, new EncodeSettings(), outDir, mux).Run(cts.Token);
                    string what = before ? "cancel before the mux tools ran" : "cancel after the mux finished";
                    Check.That(res.Cancelled && !res.Ok, what + ": Cancelled (Ok=" + res.Ok + ", Error=" + res.Error + ")");
                    Check.Equal(0, res.Outputs.Count, what + ": outputs");
                    Check.Equal(0, Names(outDir).Length, what + ": files left " + Show(Names(outDir)));
                });
        }
    }
}
