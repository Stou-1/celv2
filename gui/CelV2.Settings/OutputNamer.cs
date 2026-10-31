using System;
using System.IO;
using System.Linq;

namespace CelV2.Settings
{
    public static class OutputNamer
    {
        // Returns <folder>\<stem> without extension; "<stem> (2)", "(3)" ... while <stem>.ivf, <stem><audioExt> or <stem>.mp4 exists.
        // name: the user's output name; null or blank means the input's file name.
        public static string Stem(string folder, string inputPath, string audioExt, string name = null)
        {
            string clean = Clean(name);
            if (clean.Length == 0) clean = Path.GetFileNameWithoutExtension(inputPath);
            for (int n = 1; ; n++)
            {
                string stem = Path.Combine(folder, n == 1 ? clean : clean + " (" + n + ")");
                if (!File.Exists(stem + ".ivf") && !File.Exists(stem + ".mp4")
                    && (string.IsNullOrEmpty(audioExt) || !File.Exists(stem + audioExt)))
                    return stem;
            }
        }

        // Why a user's output name cannot be a Windows file name, or null when it can (blank = the input's name, also fine).
        public static string BadName(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) return null;
            string clean = Clean(name);
            if (clean.Length == 0) return "名前がありません";
            if (clean.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0) return "使えない文字があります（\\ / : * ? \" < > | など）";
            if (clean.EndsWith(".")) return "最後を「.」にはできません";
            string device = clean.Split('.')[0].TrimEnd().ToUpperInvariant();
            if (new[] { "CON", "PRN", "AUX", "NUL" }.Contains(device)
                || (device.Length == 4 && (device.StartsWith("COM") || device.StartsWith("LPT")) && device[3] >= '1' && device[3] <= '9'))
                return "Windows が予約している名前です";
            return null;
        }

        // Trimmed, without a typed .mp4 / .ivf (the job adds the extension itself).
        static string Clean(string name)
        {
            string s = (name ?? "").Trim();
            foreach (string ext in new[] { ".mp4", ".ivf" })
                if (s.EndsWith(ext, StringComparison.OrdinalIgnoreCase)) s = s.Substring(0, s.Length - ext.Length).TrimEnd();
            return s;
        }
    }
}
