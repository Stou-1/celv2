using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using CelV2.Settings;

namespace CelV2.Runner
{
    // Total == 0 means the frame count is unknown (Remaining is then meaningless).
    public sealed class JobProgress { public long Frame, Total; public double Fps; public TimeSpan Remaining; }

    public sealed class JobResult
    {
        public bool Ok, Cancelled;
        public string Error;
        public IReadOnlyList<string> Outputs = new string[0];
        public string EngineLine;
        public string Note;                 // set when the job did something other than asked (MP4 -> separate files)
    }

    // ffmpeg decodes into celv2enc through a pipe; then the audio is extracted and the mux step runs.
    public enum OverKind { None, Over, Coarsest }  // Coarsest: over even with every frame at qindex 255

    public sealed class EncodeJob
    {
        const string EngineMarker = "celv2: decisions on";

        readonly ToolPaths tools;
        readonly string input, outputFolder, outputName;
        readonly InputInfo info;
        EncodeSettings settings;
        IMuxStep mux;
        long framesCoded;                       // last `progress frame=N` seen

        public event Action<JobProgress> Progress;            // raised on the thread that calls Run
        public event Action<string> EngineLine;               // "celv2: decisions on ..." once
        public event Action<string> Stage;                    // what runs before the encode (QualityProbe)

        // mux == null: MP4 settings get an Mp4Mux named after the output stem, anything else SeparateFilesMux.
        // outputName: the file name without extension (OutputNamer.Stem); null uses the input's name.
        public EncodeJob(ToolPaths tools, string input, InputInfo info, EncodeSettings s, string outputFolder, IMuxStep mux, string outputName = null)
        {
            this.tools = tools; this.input = input; this.info = info; settings = s; this.outputFolder = outputFolder; this.mux = mux;
            this.outputName = outputName;
        }

        // celv2enc and av2_mux read narrow paths: UTF-8 through their manifest from Windows 10 1903 (build 18362),
        // the ANSI code page before it, where best-fit would silently turn é into e. Returns the first path
        // component that would not survive, or null.
        public static string UnsupportedName(int osBuild, Encoding ansi, string path)
        {
            if (osBuild >= 18362) return null;
            return path.Split('\\').FirstOrDefault(n => ansi.GetString(ansi.GetBytes(n)) != n);
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        struct OsVersionInfo
        {
            public int Size, Major, Minor, Build, Platform;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string CsdVersion;
        }

        [DllImport("ntdll.dll")] static extern int RtlGetVersion(ref OsVersionInfo v);

        // Environment.OSVersion reports 6.2 to an exe without a compatibility manifest.
        static int OsBuild()
        {
            var v = new OsVersionInfo { Size = Marshal.SizeOf(typeof(OsVersionInfo)) };
            return RtlGetVersion(ref v) == 0 ? v.Build : 0;
        }

        sealed class Cancelled : Exception { }

        public JobResult Run(CancellationToken cancel)
        {
            var result = new JobResult();
            var partial = new List<string>();
            try
            {
                RunCore(cancel, result, partial);
                result.Ok = true;
            }
            catch (Cancelled) { result.Cancelled = true; }
            catch (OperationCanceledException) { result.Cancelled = true; }
            catch (Exception e)
            {
                if (cancel.IsCancellationRequested) result.Cancelled = true; // a tool killed by the cancel failed on purpose
                else result.Error = e.Message;
            }
            if (!result.Ok)
            {
                foreach (string p in partial) DeleteWithRetry(p);
                result.Outputs = new string[0];
            }
            return result;
        }

        void RunCore(CancellationToken cancel, JobResult result, List<string> partial)
        {
            if (tools.Ffmpeg == null) throw new InvalidOperationException("ffmpeg が見つかりません");
            if (tools.CelV2enc == null) throw new InvalidOperationException("celv2enc が見つかりません");
            if (info.Hdr && !CommandBuilder.KeepsHdr(info, settings.Output) && !Prober.HasFilters(tools, "zscale", "tonemap"))
                throw new InvalidOperationException("HDR の変換に必要な ffmpeg の機能（zscale / tonemap）がありません");

            bool mp4 = settings.Container == ContainerKind.Mp4;
            if (mp4 && (tools.Av2Mux == null || tools.Mp4Box == null))
            {
                mp4 = false;
                mux = new SeparateFilesMux();
                result.Note = "MP4 に必要な道具（av2_mux / MP4Box）が見つからないため、別ファイルで保存しました";
            }

            bool heAac = settings.HeAacV2 && info.AudioCodec != null && tools.Fdkaac != null;
            if (settings.HeAacV2 && info.AudioCodec != null && tools.Fdkaac == null)
                result.Note = (result.Note == null ? "" : result.Note + "、") + "fdkaac が見つからないため、音声は HE-AACv2 にせず通常どおり保存しました";
            string audioExt = info.AudioCodec == null ? null : mp4 || heAac ? ".m4a" : CommandBuilder.AudioExtensionFor(info.AudioCodec);
            string badName = OutputNamer.BadName(outputName);
            if (badName != null) throw new InvalidOperationException("出力名「" + outputName + "」: " + badName);
            Directory.CreateDirectory(outputFolder);
            string stem = OutputNamer.Stem(outputFolder, input, audioExt, outputName);
            string ivf = stem + ".ivf";
            string audio = audioExt == null ? null : stem + audioExt;
            string bad = UnsupportedName(OsBuild(), Encoding.Default, stem); // celv2enc / av2_mux get stem + an ASCII extension
            if (bad != null)
                throw new InvalidOperationException("この Windows では、ファイル名やフォルダ名に使えない文字が含まれています: " + bad);
            if (mux == null)
                mux = mp4 ? (IMuxStep)new Mp4Mux(tools, CommandBuilder.ColrFor(info, settings.Output), stem + ".mp4") : new SeparateFilesMux();
            if (mp4) partial.Add(stem + ".mp4");
            partial.Add(ivf);

            if (settings.KeepUnderSource && settings.TargetSizeMB <= 0) // a target size is the stronger limit
            {
                int q = QualityProbe.Pick(tools, input, info, settings, cancel, m => { var h = Stage; if (h != null) h(m); });
                if (q != settings.QIndex)
                {
                    result.Note = (result.Note == null ? "" : result.Note + "、") + "qindex " + settings.QIndex + "→" + q + "（元の動画より大きくならないよう調整）";
                    settings = settings.Clone();
                    settings.QIndex = q;
                }
            }
            if (audio != null) partial.Add(audio);
            if (settings.TargetSizeMB > 0)
            {
                result.Outputs = EncodeToSize(cancel, result, partial, ivf, audio, mp4, heAac);
                return;
            }
            EncodeVideo(cancel, result, ivf);
            cancel.ThrowIfCancellationRequested();
            if (audio != null) MakeAudio(cancel, audio, mp4, heAac);
            cancel.ThrowIfCancellationRequested();
            result.Outputs = Mux(cancel, partial, ivf, audio);
        }

        // サイズ指定: pass 1 into %TEMP%, the audio, then pass 2 at the video's budget. Over the target (celv2enc's
        // exit 3, or the muxed whole) the budget goes down by the excess and 1% and pass 2 runs again, 3 times at most.
        IReadOnlyList<string> EncodeToSize(CancellationToken cancel, JobResult result, List<string> partial, string ivf, string audio, bool mp4, bool heAac)
        {
            long target = (long)Math.Floor(settings.TargetSizeMB * 1e6);
            string tag = Path.Combine(Path.GetTempPath(), "celv2_" + Guid.NewGuid().ToString("N"));
            string stats = tag + ".stats", pass1 = tag + "_pass1.ivf";
            try
            {
                RaiseStage("下調べ（1 回目）");
                EncodeVideo(cancel, result, pass1, new PassArgs { Pass = 1, Stats = stats });
                DeleteWithRetry(pass1);
                cancel.ThrowIfCancellationRequested();
                long frames = framesCoded, audioBytes = 0;
                if (audio != null)
                {
                    MakeAudio(cancel, audio, mp4, heAac);
                    audioBytes = new FileInfo(audio).Length;
                }
                if (audioBytes >= target)
                    throw new InvalidOperationException("音声だけで指定サイズを超えます（音声 " + audioBytes + " バイト、指定 " + target + " バイト）");
                long budget = CommandBuilder.VideoBudget(target, audioBytes, frames, mp4), last = 0;
                for (int retry = 0; retry <= 3 && budget > 0; retry++)
                {
                    RaiseStage(retry == 0 ? "本番（2 回目）" : "本番（2 回目、やり直し " + retry + "）");
                    long over;
                    var kind = EncodeVideo(cancel, result, ivf, new PassArgs { Pass = 2, Stats = stats, TargetBytes = budget });
                    if (kind == OverKind.Coarsest)
                        throw new InvalidOperationException("指定サイズに収まりませんでした: いちばん粗い画質（qindex 255）でも映像が " +
                            new FileInfo(ivf).Length + " バイトになり、予算の " + budget + " バイトを超えます");
                    if (kind == OverKind.Over && !MayFit(new FileInfo(ivf).Length, audioBytes, frames, mp4, target))
                    {
                        long video = new FileInfo(ivf).Length;
                        last = video + audioBytes;
                        over = video - budget;
                    }
                    else
                    {
                        if (audio != null && !File.Exists(audio)) MakeAudio(cancel, audio, mp4, heAac); // the last try's mux took it
                        var outputs = Mux(cancel, partial, ivf, audio);
                        last = outputs.Sum(o => new FileInfo(o).Length);
                        if (last <= target)
                        {
                            result.Note = (result.Note == null ? "" : result.Note + "、") + "サイズ指定: " +
                                (last / 1e6).ToString("0.00", CultureInfo.InvariantCulture) + " MB（指定の " +
                                (100.0 * last / target).ToString("0.0", CultureInfo.InvariantCulture) + "%）";
                            return outputs;
                        }
                        over = last - target;
                        foreach (string o in outputs) DeleteWithRetry(o);
                    }
                    budget -= over + budget / 100;
                }
                throw new InvalidOperationException("指定サイズに収まりませんでした（最後: " + last + " バイト、指定 " + target + " バイト）");
            }
            finally
            {
                DeleteWithRetry(stats);
                DeleteWithRetry(pass1);
            }
        }

        // The audio as the settings say, as long as the video really encoded (not the probe's estimate).
        void MakeAudio(CancellationToken cancel, string audio, bool mp4, bool heAac)
        {
            long frames = framesCoded > 0 ? framesCoded : info.FrameCount;
            double seconds = info.FpsNum > 0 ? frames * (double)info.FpsDen / info.FpsNum : info.DurationSeconds;
            if (heAac)
            {
                EncodeHeAac(cancel, mp4 ? seconds : (double?)null, audio);
                return;
            }
            var args = mp4 ? CommandBuilder.Mp4AudioArgs(input, info.AudioCodec, seconds, audio, settings.AudioKbps)
                           : CommandBuilder.AudioArgs(input, info.AudioCodec, audio, settings.AudioKbps);
            var r = Proc.Capture(tools.Ffmpeg, args, cancel);
            if (cancel.IsCancellationRequested) throw new Cancelled();
            if (r.ExitCode != 0)
                throw new InvalidOperationException("音声の取り出しに失敗しました: " + Proc.Tail(r.Err, 4));
        }

        IReadOnlyList<string> Mux(CancellationToken cancel, List<string> partial, string ivf, string audio)
        {
            cancel.ThrowIfCancellationRequested();
            var outputs = mux.Finish(ivf, audio, cancel);
            partial.AddRange(outputs);
            if (cancel.IsCancellationRequested) throw new Cancelled(); // cancelled while the last tool ran: drop the result
            foreach (string o in outputs)
                if (!File.Exists(o)) throw new InvalidOperationException("出力ファイルが作られませんでした: " + o);
            return outputs;
        }

        void RaiseStage(string m) { var h = Stage; if (h != null) h(m); }

        // ffmpeg -> a WAV in %TEMP% -> fdkaac; the WAV is deleted whatever happens.
        void EncodeHeAac(CancellationToken cancel, double? seconds, string m4a)
        {
            string wav = Path.Combine(Path.GetTempPath(), "celv2_" + Guid.NewGuid().ToString("N") + ".wav");
            try
            {
                var r = Proc.Capture(tools.Ffmpeg, CommandBuilder.HeAacWavArgs(input, seconds, wav), cancel);
                if (cancel.IsCancellationRequested) throw new Cancelled();
                if (r.ExitCode != 0)
                    throw new InvalidOperationException("音声の取り出しに失敗しました: " + Proc.Tail(r.Err, 4));
                r = Proc.Capture(tools.Fdkaac, CommandBuilder.FdkaacArgs(wav, m4a, settings.HeAacKbps), cancel);
                if (cancel.IsCancellationRequested) throw new Cancelled();
                if (r.ExitCode != 0)
                    throw new InvalidOperationException("HE-AACv2 への変換に失敗しました: " + Proc.Tail(r.Err, 4));
            }
            finally { DeleteWithRetry(wav); }
        }

        // A retry's video over its smaller budget may still fit the target: then it is muxed and the whole checked.
        public static bool MayFit(long video, long audioBytes, long frames, bool mp4, long target)
        {
            return video + audioBytes + CommandBuilder.ContainerBytes(frames, mp4) <= target;
        }

        // Over the --target-size: pass 2's exit 3 with celv2enc's own line (an abort() exits with 3 on Windows too).
        public static OverKind Pass2Over(PassArgs pass, int exitCode, IEnumerable<string> tail)
        {
            if (pass == null || pass.Pass != 2 || exitCode != 3) return OverKind.None;
            string line = tail.LastOrDefault(l => l.Contains("over --target-size"));
            return line == null ? OverKind.None : line.Contains("even at qindex 255") ? OverKind.Coarsest : OverKind.Over;
        }

        // pass: celv2enc --pass 1 / 2 (null: one pass). Not None when pass 2 went over its --target-size.
        OverKind EncodeVideo(CancellationToken cancel, JobResult result, string ivf, PassArgs pass = null)
        {
            var ffTail = new Queue<string>();
            var encTail = new Queue<string>();
            Process ff = null, enc = null;
            Thread copier = null;
            bool killedFf = false;
            try
            {
                framesCoded = 0;
                ff = Proc.Start(tools.Ffmpeg, CommandBuilder.DecodeArgs(input, info, settings.Output), false, true, true);
                // ffmpeg's stderr must be drained or a chatty ffmpeg blocks on a full pipe
                ff.ErrorDataReceived += (s, e) => { if (e.Data != null) Keep(ffTail, e.Data); };
                ff.BeginErrorReadLine();
                enc = Proc.Start(tools.CelV2enc, CommandBuilder.EncodeArgs(settings, info, ivf, pass), true, true, true);
                enc.OutputDataReceived += (s, e) => { if (e.Data != null) Keep(encTail, e.Data); };
                enc.BeginOutputReadLine();

                var ffRef = ff; var encRef = enc;
                // raw bytes, never the text reader/writer, so the y4m stream arrives untouched
                copier = new Thread(() => Pump(ffRef, encRef)) { IsBackground = true };
                copier.Start();

                using (cancel.Register(() => { Proc.Kill(ffRef); Proc.Kill(encRef); }))
                {
                    var clock = Stopwatch.StartNew();
                    string line;
                    while ((line = enc.StandardError.ReadLine()) != null)
                    {
                        int frame;
                        if (line.StartsWith("progress frame=", StringComparison.Ordinal)
                            && int.TryParse(line.Substring(15), out frame))
                        {
                            framesCoded = frame;
                            RaiseProgress(frame, clock.Elapsed.TotalSeconds);
                        }
                        else if (line.StartsWith(EngineMarker, StringComparison.Ordinal) && result.EngineLine == null)
                        {
                            result.EngineLine = line;
                            var h = EngineLine; if (h != null) h(line);
                        }
                        else Keep(encTail, line);
                    }
                    enc.WaitForExit();
                    if (ff.WaitForExit(enc.ExitCode == 0 ? 10000 : 2000))
                        ff.WaitForExit(); // the untimed overload also lets the async stderr reader finish
                    else
                    {
                        // celv2enc is gone but ffmpeg is stuck writing into the pipe
                        killedFf = true;
                        Proc.Kill(ff);
                        ff.WaitForExit();
                    }
                    copier.Join();
                }
                if (cancel.IsCancellationRequested) throw new Cancelled();

                OverKind over;
                lock (encTail) over = Pass2Over(pass, enc.ExitCode, encTail.ToList());
                var why = new List<string>();
                if (!killedFf && ff.ExitCode != 0) why.Add(Describe("ffmpeg", ff.ExitCode, ffTail));
                if (enc.ExitCode != 0 && over == OverKind.None) why.Add(Describe("celv2enc", enc.ExitCode, encTail));
                if (why.Count > 0) throw new InvalidOperationException(string.Join("\n", why));
                return over;
            }
            finally
            {
                // only reached with live processes on an exception path: leave nothing running
                if (enc != null) { Proc.Kill(enc); enc.WaitForExit(); }
                if (ff != null) { Proc.Kill(ff); ff.WaitForExit(); }
                if (copier != null) copier.Join();
                if (enc != null) enc.Dispose();
                if (ff != null) ff.Dispose();
            }
        }

        static void Pump(Process ff, Process enc)
        {
            try
            {
                var src = ff.StandardOutput.BaseStream;
                var dst = enc.StandardInput.BaseStream;
                var buf = new byte[1 << 20];
                int n;
                while ((n = src.Read(buf, 0, buf.Length)) > 0) dst.Write(buf, 0, n);
                dst.Flush();
            }
            catch (IOException) { }               // the other side was killed or exited
            catch (ObjectDisposedException) { }
            finally
            {
                try { enc.StandardInput.Close(); } catch (IOException) { } catch (ObjectDisposedException) { }
            }
        }

        void RaiseProgress(long frame, double seconds)
        {
            var h = Progress;
            if (h == null) return;
            double fps = seconds > 0 ? frame / seconds : 0;
            double left = fps > 0 ? Math.Max(0, info.FrameCount - frame) / fps : 0;
            h(new JobProgress
            {
                Frame = frame, Total = info.FrameCount, Fps = fps,
                Remaining = TimeSpan.FromSeconds(Math.Min(left, TimeSpan.MaxValue.TotalSeconds / 2)),
            });
        }

        static void Keep(Queue<string> q, string line)
        {
            lock (q) { q.Enqueue(line); while (q.Count > 20) q.Dequeue(); }
        }

        static string Describe(string name, int code, Queue<string> tail)
        {
            string text;
            lock (tail) { var l = tail.Where(x => x.Trim().Length > 0).ToList(); text = string.Join("\n", l.Skip(Math.Max(0, l.Count - 6))); }
            return name + " が失敗しました (exit " + code + ")" + (text.Length > 0 ? ": " + text : "");
        }

        static void DeleteWithRetry(string path)
        {
            for (int i = 0; i < 5; i++)
            {
                try { File.Delete(path); return; }
                catch (IOException) { Thread.Sleep(50); }
                catch (UnauthorizedAccessException) { Thread.Sleep(50); }
            }
        }
    }
}
