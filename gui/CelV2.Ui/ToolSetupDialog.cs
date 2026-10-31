using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace CelV2.Ui
{
    // Shown when ffmpeg / ffprobe cannot be found; also lets the user point at MP4Box.
    public sealed class ToolSetupDialog : Form
    {
        const string FfmpegUrl = "https://ffmpeg.org/download.html";
        const string GpacUrl = "https://gpac.io/downloads/gpac-nightly-builds/";

        readonly Label ffmpegState = new Label { Name = "ffmpegState", AutoSize = true };
        readonly Label mp4boxState = new Label { Name = "mp4boxState", AutoSize = true };

        // Folders the user picked; null while nothing was picked.
        public string FfmpegFolder { get; private set; }
        public string Mp4BoxFolder { get; private set; }

        public ToolSetupDialog(string currentFfmpegFolder, string currentMp4BoxFolder)
        {
            FfmpegFolder = currentFfmpegFolder;
            Mp4BoxFolder = currentMp4BoxFolder;
            Text = "必要なツールの設定";
            AutoScaleMode = AutoScaleMode.Dpi;
            Font = SystemFonts.MessageBoxFont;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = MinimizeBox = false;
            ShowInTaskbar = false;
            StartPosition = FormStartPosition.CenterParent;
            AutoSize = true;
            AutoSizeMode = AutoSizeMode.GrowAndShrink;

            var panel = new FlowLayoutPanel { FlowDirection = FlowDirection.TopDown, WrapContents = false, AutoSize = true, Padding = new Padding(14), Dock = DockStyle.Fill, MaximumSize = new Size(560, 0) };
            panel.Controls.Add(Text_("ffmpeg と ffprobe が見つかりません", true));
            panel.Controls.Add(Text_("動画の読み込みに ffmpeg が必要です（このアプリには同梱していません）。入手して展開し、ffmpeg.exe と ffprobe.exe が入ったフォルダを選んでください。", false));
            panel.Controls.Add(Link("ffmpeg の入手先: " + FfmpegUrl, FfmpegUrl));
            panel.Controls.Add(Pick("ffmpeg.exe を選ぶ", "ffmpeg.exe", ffmpegState, f => FfmpegFolder = f, true));
            panel.Controls.Add(ffmpegState);
            panel.Controls.Add(Text_("MP4Box（任意）", true));
            panel.Controls.Add(Text_("MP4 で保存するのに使います。なくても変換は続けられ、映像（.ivf）と音声を別ファイルで保存します。", false));
            panel.Controls.Add(Link("GPAC（MP4Box）の入手先: " + GpacUrl, GpacUrl));
            panel.Controls.Add(Pick("mp4box.exe を選ぶ", "MP4Box.exe", mp4boxState, f => Mp4BoxFolder = f, false));
            panel.Controls.Add(mp4boxState);
            var ok = new Button { Text = "OK", DialogResult = DialogResult.OK, AutoSize = true, Margin = new Padding(3, 14, 3, 3) };
            var cancel = new Button { Text = "キャンセル", DialogResult = DialogResult.Cancel, AutoSize = true, Margin = new Padding(3, 14, 3, 3) };
            var row = new FlowLayoutPanel { AutoSize = true };
            row.Controls.AddRange(new Control[] { ok, cancel });
            panel.Controls.Add(row);
            Controls.Add(panel);
            AcceptButton = ok;
            CancelButton = cancel;
            ShowState(ffmpegState, FfmpegFolder, "ffmpeg.exe");
            ShowState(mp4boxState, Mp4BoxFolder, "MP4Box.exe");
        }

        static Label Text_(string text, bool bold)
        {
            var l = new Label { Text = text, AutoSize = true, MaximumSize = new Size(530, 0), Margin = new Padding(3, bold ? 10 : 3, 3, 3) };
            if (bold) l.Font = new Font(l.Font, FontStyle.Bold);
            return l;
        }

        static LinkLabel Link(string text, string url)
        {
            var l = new LinkLabel { Text = text, AutoSize = true, MaximumSize = new Size(530, 0) };
            l.LinkClicked += (a, b) =>
            {
                try { Process.Start(new ProcessStartInfo(url) { UseShellExecute = true }); }
                catch (Exception e) when (e is Win32Exception || e is InvalidOperationException)
                {
                    MessageBox.Show(l.FindForm(), "ブラウザーを開けませんでした。次の場所を手で開いてください: " + url, "必要なツールの設定", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
            };
            return l;
        }

        Button Pick(string caption, string exe, Label state, Action<string> set, bool needsFfprobe)
        {
            var b = new Button { Text = caption, AutoSize = true, Name = exe };
            b.Click += (a, e) =>
            {
                using (var d = new OpenFileDialog { Title = caption, Filter = exe + "|" + exe, FileName = exe })
                {
                    if (d.ShowDialog(this) != DialogResult.OK) return;
                    string folder = Path.GetDirectoryName(d.FileName);
                    if (needsFfprobe && !File.Exists(Path.Combine(folder, "ffprobe.exe")))
                    {
                        MessageBox.Show(this, "同じフォルダに ffprobe.exe がありません。ffmpeg.exe と ffprobe.exe が入ったフォルダを選んでください。", Text, MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return;
                    }
                    set(folder);
                    ShowState(state, folder, exe);
                }
            };
            return b;
        }

        static void ShowState(Label l, string folder, string exe)
        {
            l.Text = string.IsNullOrWhiteSpace(folder) ? "（未設定）" : "選択: " + folder;
        }
    }
}
