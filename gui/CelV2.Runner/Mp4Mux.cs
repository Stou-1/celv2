using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using CelV2.Settings;

namespace CelV2.Runner
{
    // av2_mux for the video, then MP4Box -add when there is audio.
    public sealed class Mp4Mux : IMuxStep
    {
        readonly ToolPaths tools;
        readonly string colr, outputMp4;

        public Mp4Mux(ToolPaths tools, string colrOrNull, string outputMp4)
        {
            this.tools = tools; colr = colrOrNull; this.outputMp4 = outputMp4;
        }

        public IReadOnlyList<string> Finish(string videoIvf, string audioOrNull, CancellationToken cancel)
        {
            // a name nobody else owns, so a user's own .obu next to the output is never touched
            string obu = Path.Combine(Path.GetDirectoryName(outputMp4) ?? "", ".celv2-" + Guid.NewGuid().ToString("N") + ".obu");
            try
            {
                int rate, scale;
                IvfToObu.Convert(videoIvf, obu, out rate, out scale);
                Run(tools.Av2Mux, "av2_mux", CommandBuilder.Av2MuxArgs(obu, outputMp4, rate, scale, colr), cancel);
                if (!File.Exists(outputMp4)) throw new InvalidOperationException("av2_mux did not write " + outputMp4);
                if (audioOrNull != null) Run(tools.Mp4Box, "MP4Box", CommandBuilder.Mp4BoxAddArgs(audioOrNull, outputMp4), cancel);
            }
            catch
            {
                Delete(outputMp4); Delete(obu);
                throw;
            }
            Delete(obu); Delete(videoIvf);
            if (audioOrNull != null) Delete(audioOrNull);
            return new[] { outputMp4 };
        }

        static void Run(string exe, string name, string[] args, CancellationToken cancel)
        {
            var r = Proc.Capture(exe, args, cancel);
            if (cancel.IsCancellationRequested) throw new OperationCanceledException(cancel);
            if (r.ExitCode != 0)
            {
                string why = Proc.Tail(r.Err + "\n" + r.Out, 4);
                throw new InvalidOperationException(name + " failed (exit " + r.ExitCode + ")" + (why.Length > 0 ? ": " + why : ""));
            }
        }

        static void Delete(string path)
        {
            try { File.Delete(path); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }
}
