using System;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using CelV2.Settings;

namespace CelV2.Runner
{
    public static class Prober
    {
        // How long ffprobe / ffmpeg may take to answer before they are stopped (a stalled network share,
        // a stream that never sends): the input then counts as unreadable instead of waiting forever.
        static readonly TimeSpan Timeout = TimeSpan.FromSeconds(60);

        static Proc.Output CaptureWithTimeout(string exe, string[] args)
        {
            using (var cts = new CancellationTokenSource(Timeout))
            {
                var r = Proc.Capture(exe, args, cts.Token);
                if (cts.IsCancellationRequested)
                    throw new TimeoutException(System.IO.Path.GetFileName(exe) + " did not finish in " + Timeout.TotalSeconds + " s");
                return r;
            }
        }

        // Throws with ffprobe's own message when the file cannot be read.
        public static InputInfo Probe(ToolPaths t, string input)
        {
            var r = CaptureWithTimeout(t.Ffprobe, ProbeParser.ProbeArgs(input));
            if (r.ExitCode != 0)
            {
                string why = Proc.Tail(r.Err, 3);
                throw new InvalidOperationException(why.Length > 0 ? why : "ffprobe exited with code " + r.ExitCode);
            }
            var i = ProbeParser.Parse(r.Out);
            if (i.Hdr)
            {
                // HDR10 metadata is frame side data (SEI, or the container's carried onto frames): decode the first frame.
                var f = CaptureWithTimeout(t.Ffprobe, ProbeParser.HdrMetadataArgs(input, i.VideoStreamIndex));
                if (f.ExitCode == 0) ProbeParser.ParseHdrMetadata(f.Out, i); // without it the video is still HDR, only unsignalled
            }
            return i;
        }

        // The choices of where the decisions run, from celv2enc --list-devices; the earlier fixed ones
        // (自動 / GPU / CPU) when celv2enc is missing or fails.
        public static System.Collections.Generic.List<DeviceOption> ListDevices(ToolPaths t)
        {
            if (t == null || t.CelV2enc == null) return DeviceList.Fallback();
            try
            {
                var r = CaptureWithTimeout(t.CelV2enc, new[] { "--list-devices" });
                var list = r.ExitCode == 0 ? DeviceList.Parse(r.Out) : null;
                return list != null && list.Count > 1 ? list : DeviceList.Fallback();
            }
            catch (Exception) { return DeviceList.Fallback(); }  // a broken celv2enc must not keep the dialog from opening
        }

        public static bool HasFilters(ToolPaths t, params string[] names)
        {
            var r = CaptureWithTimeout(t.Ffmpeg, new[] { "-hide_banner", "-filters" });
            return r.ExitCode == 0 && ParseHasFilters(r.Out, names);
        }

        // `ffmpeg -filters` lines look like " T.. zscale   V->V   description"; the name is the second token.
        public static bool ParseHasFilters(string filtersOutput, params string[] names)
        {
            return names.All(n => Regex.IsMatch(filtersOutput ?? "", @"^\s*\S+\s+" + Regex.Escape(n) + @"\s", RegexOptions.Multiline));
        }
    }
}
