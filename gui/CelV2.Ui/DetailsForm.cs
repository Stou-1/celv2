using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using CelV2.Runner;

namespace CelV2.Ui
{
    // The running (or last) encode in detail: its stage, engine and times; the video's size, bitrate and projected
    // size; each coded frame; the log. Reads EncodeDetails twice a second while shown.
    public sealed class DetailsForm : Form
    {
        readonly EncodeDetails details;
        readonly Timer timer = new Timer { Interval = 500 };
        readonly Label statusLabel = new Label { Name = "statusLabel", AutoSize = true };
        readonly Label sizeLabel = new Label { Name = "sizeLabel", AutoSize = true };
        readonly ListView frameList = new ListView
        {
            Name = "frameList", View = View.Details, FullRowSelect = true, GridLines = true, Dock = DockStyle.Fill, VirtualMode = true,
        };
        readonly CheckBox followBox = new CheckBox { Name = "followBox", Text = "最新のフレームを表示", AutoSize = true, Checked = true };
        readonly TextBox logBox = new TextBox
        {
            Name = "logBox", Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Both, WordWrap = false, Dock = DockStyle.Fill,
            MaxLength = 0, Font = new Font(FontFamily.GenericMonospace, 9f),
        };
        EncodeDetails.View view = new EncodeDetails.View();
        long logShown;      // EncodeDetails' LogLinesSeen the box holds up to
        int logInBox;       // lines in the box, to start it over from the kept ones when it grows past them

        public DetailsForm(EncodeDetails details)
        {
            this.details = details;
            Text = "エンコードの詳細";
            AutoScaleDimensions = new SizeF(96F, 96F);
            AutoScaleMode = AutoScaleMode.Dpi;
            Font = SystemFonts.MessageBoxFont;
            ClientSize = new Size(760, 620);
            MinimumSize = new Size(480, 400);

            frameList.Columns.Add("番号", Dp(70));
            frameList.Columns.Add("種類", Dp(80));
            frameList.Columns.Add("qindex", Dp(70), HorizontalAlignment.Right);
            frameList.Columns.Add("サイズ（KB）", Dp(100), HorizontalAlignment.Right);
            frameList.RetrieveVirtualItem += (s, e) => e.Item = Row(e.ItemIndex);

            var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 6, Padding = new Padding(Dp(8)) };
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 55));
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 45));
            root.Controls.Add(statusLabel, 0, 0);
            root.Controls.Add(sizeLabel, 0, 1);
            root.Controls.Add(followBox, 0, 2);
            root.Controls.Add(frameList, 0, 3);
            root.Controls.Add(new Label { Text = "ログ", AutoSize = true, Margin = new Padding(0, Dp(6), 0, 0) }, 0, 4);
            root.Controls.Add(logBox, 0, 5);
            Controls.Add(root);

            timer.Tick += (s, e) => RefreshNow();
            Shown += (s, e) => { RefreshNow(); timer.Start(); };
            FormClosed += (s, e) => timer.Stop();
        }

        int Dp(int px) { return LogicalToDeviceUnits(px); }

        public void RefreshNow()
        {
            view = details.Snapshot();
            statusLabel.Text = StatusText(view);
            sizeLabel.Text = SizeText(view);
            int n = view.Frames.Count;
            if (frameList.VirtualListSize != n)
            {
                frameList.VirtualListSize = n;
                if (followBox.Checked && n > 0) frameList.EnsureVisible(n - 1);
            }
            else frameList.Invalidate();
            ShowLog(view);
        }

        public ListViewItem Row(int i)
        {
            var f = i < view.Frames.Count ? view.Frames[i] : new FrameInfo();
            var row = new ListViewItem(f.Frame.ToString());
            row.SubItems.Add(TypeName(f.Type));
            row.SubItems.Add(f.Qindex > 0 ? f.Qindex.ToString() : "-");
            row.SubItems.Add(f.Bytes >= 0 ? (f.Bytes / 1024.0).ToString("0.0") : "-");
            return row;
        }

        // K an I frame (keyframe); P a lone P frame; A a mini-GOP's anchor, coded from the past only like a P frame
        // (hidden, shown after its B frames); B the others.
        public static string TypeName(char t)
        {
            switch (t)
            {
                case 'K': return "I（キー）";
                case 'P': return "P";
                case 'A': return "P（アンカー）";
                case 'B': return "B";
                default: return "-";
            }
        }

        static string StatusText(EncodeDetails.View v)
        {
            if (v.Name.Length == 0) return "まだ変換していません";
            int n = v.Frames.Count;
            double seconds = v.Elapsed.TotalSeconds;
            string engine = v.Engine;
            int on = engine.IndexOf("decisions on ", StringComparison.Ordinal);
            if (on >= 0) engine = engine.Substring(on + "decisions on ".Length);
            return v.Name + (v.Running ? "　変換中" : "　終了") + (v.Stage.Length > 0 ? "　" + v.Stage : n > 0 && v.Running ? "　符号化" : "")
                   + (v.Pass > 1 ? "　" + v.Pass + " 回目" : "")
                   + Environment.NewLine
                   + (v.Total > 0 ? n + " / " + v.Total + " 枚" : n + " 枚")
                   + string.Format("　{0:0.0} fps　経過 {1}", seconds > 0 ? n / seconds : 0, Clock(v.Elapsed))
                   + (engine.Length > 0 ? "　判定: " + engine : "");
        }

        static string SizeText(EncodeDetails.View v)
        {
            if (v.Frames.Count == 0) return " ";
            return string.Format("映像 {0:0.00} MB　平均 {1:0} kbps", v.Bytes / 1e6, v.Kbps)
                   + (v.ProjectedBytes >= 0 ? string.Format("　完成時の予想 {0:0.0} MB（映像のみ）", v.ProjectedBytes / 1e6) : "");
        }

        static string Clock(TimeSpan t)
        {
            return t.TotalHours >= 1 ? string.Format("{0}:{1:00}:{2:00}", (int)t.TotalHours, t.Minutes, t.Seconds)
                                     : string.Format("{0}:{1:00}", t.Minutes, t.Seconds);
        }

        void ShowLog(EncodeDetails.View v)
        {
            long fresh = v.LogLinesSeen - logShown;
            if (fresh <= 0 && v.LogLinesSeen >= logShown) return;
            if (fresh < 0 || fresh > v.Log.Count || logInBox + fresh > 2 * EncodeDetails.MaxLogLines)
            {
                // a new run, or more than the kept lines since the last look: start over from what is kept
                logBox.Text = string.Join(Environment.NewLine, v.Log) + Environment.NewLine;
                logInBox = v.Log.Count;
            }
            else
            {
                logBox.AppendText(string.Join(Environment.NewLine, v.Log.Skip(v.Log.Count - (int)fresh)) + Environment.NewLine);
                logInBox += (int)fresh;
            }
            logShown = v.LogLinesSeen;
            logBox.SelectionStart = logBox.TextLength;
            logBox.ScrollToCaret();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) timer.Dispose();
            base.Dispose(disposing);
        }
    }
}
