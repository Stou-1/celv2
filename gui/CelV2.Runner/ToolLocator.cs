using System;
using System.IO;

namespace CelV2.Runner
{
    public sealed class ToolPaths { public string Ffmpeg, Ffprobe, CelV2enc, Av2Mux, Mp4Box, Fdkaac; }

    public static class ToolLocator
    {
        // ffmpeg/ffprobe: configured folder -> toolsFolder (the app's exe folder) -> PATH.
        // MP4Box: same order. fdkaac: its configured folder, the ffmpeg folder, then the same. celv2enc: toolsFolder only. av2_mux: av2MuxFolder -> toolsFolder.
        // A member is null when the tool is not found.
        public static ToolPaths Find(string configuredFfmpegFolder, string configuredMp4BoxFolder, string toolsFolder,
                                     string av2MuxFolder, string configuredFdkaacFolder = null)
        {
            return new ToolPaths
            {
                Ffmpeg = Locate("ffmpeg.exe", true, configuredFfmpegFolder, toolsFolder),
                Ffprobe = Locate("ffprobe.exe", true, configuredFfmpegFolder, toolsFolder),
                Mp4Box = Locate("MP4Box.exe", true, configuredMp4BoxFolder, toolsFolder),
                Fdkaac = Locate("fdkaac.exe", true, configuredFdkaacFolder, configuredFfmpegFolder, toolsFolder),
                CelV2enc = Locate("celv2enc.exe", false, toolsFolder),
                Av2Mux = Locate("av2_mux.exe", false, av2MuxFolder, toolsFolder),
            };
        }

        static string Locate(string exe, bool searchPath, params string[] folders)
        {
            foreach (string f in folders)
            {
                string p = InFolder(f, exe);
                if (p != null) return p;
            }
            if (!searchPath) return null;
            foreach (string f in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(';'))
            {
                string p = InFolder(f, exe);
                if (p != null) return p;
            }
            return null;
        }

        static string InFolder(string folder, string exe)
        {
            if (string.IsNullOrWhiteSpace(folder)) return null;
            try
            {
                string p = Path.Combine(folder.Trim().Trim('"'), exe);
                return File.Exists(p) ? p : null;
            }
            catch (ArgumentException) { return null; } // an invalid PATH entry
        }
    }
}
