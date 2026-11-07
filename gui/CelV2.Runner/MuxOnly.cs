using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using CelV2.Settings;

namespace CelV2.Runner
{
    // 「MP4 にまとめる」: a separate-file run's .ivf, with the audio saved beside it, into one .mp4. Nothing is
    // re-encoded and the inputs stay.
    public static class MuxOnly
    {
        // The extensions a separate-file run gives its audio (CommandBuilder.AudioExtensionFor), .m4a first.
        static readonly string[] AudioExtensions = { ".m4a", ".mp3", ".opus", ".ac3", ".flac" };

        // The audio beside ivf with the same name, or null.
        public static string FindAudio(string ivf)
        {
            string stem = Path.Combine(Path.GetDirectoryName(ivf) ?? "", Path.GetFileNameWithoutExtension(ivf));
            return AudioExtensions.Select(e => stem + e).FirstOrDefault(File.Exists);
        }

        // outputName: as EncodeJob's (null: the video's name); the .mp4 never replaces an existing file.
        public static JobResult Run(ToolPaths tools, string ivf, string outputFolder, string outputName, CancellationToken cancel,
                                    Action<string> log = null)
        {
            var result = new JobResult();
            var outerLog = Proc.Log;
            Proc.Log = log;
            try
            {
                if (log != null) log("MP4 にまとめる: " + ivf);
                string audio = FindAudio(ivf);
                if (tools.Av2Mux == null) throw new InvalidOperationException("av2_mux が見つかりません（CelV2Gui.exe と同じフォルダに置いてください）");
                if (audio != null && tools.Mp4Box == null)
                    throw new InvalidOperationException("音声（" + Path.GetFileName(audio) + "）をまとめる MP4Box が見つかりません。「ツール設定...」で MP4Box.exe を選んでください");
                string badName = OutputNamer.BadName(outputName);
                if (badName != null) throw new InvalidOperationException("出力名「" + outputName + "」: " + badName);
                Directory.CreateDirectory(outputFolder);
                string mp4 = OutputNamer.FreeFile(outputFolder, ivf, ".mp4", outputName);
                string bad = EncodeJob.UnsupportedName(EncodeJob.OsBuild(), Encoding.Default, mp4);
                if (bad != null) throw new InvalidOperationException("この Windows では、ファイル名やフォルダ名に使えない文字が含まれています: " + bad);
                result.Outputs = new Mp4Mux(tools, null, mp4, true).Finish(ivf, audio, cancel);
                result.Ok = true;
                result.Note = "MP4 にまとめました（音声: " + (audio == null ? "なし" : Path.GetFileName(audio)) + "）";
                if (log != null) log("結果: 成功 " + mp4);
            }
            catch (Exception e)
            {
                if (cancel.IsCancellationRequested) result.Cancelled = true;
                else result.Error = e.Message;
                if (log != null) log(result.Cancelled ? "結果: 中止" : "結果: 失敗: " + e.Message);
            }
            finally { Proc.Log = outerLog; }
            return result;
        }
    }
}
