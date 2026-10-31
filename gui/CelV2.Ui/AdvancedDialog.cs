using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using CelV2.Settings;

namespace CelV2.Ui
{
    // Edits a copy of an EncodeSettings; the one passed in is never changed.
    public sealed class AdvancedDialog : Form
    {
        readonly EncodeSettings original;
        readonly PresetStore store;
        readonly TrackBar qSlider = new TrackBar { Minimum = 1, Maximum = 255, TickFrequency = 25, Dock = DockStyle.Fill };
        readonly NumericUpDown qBox = Num("qindex", 1, 255);
        readonly CheckBox keepUnder = Check("keepUnderSource", "元の動画より大きくしない");
        readonly NumericUpDown sizeCap = Num("sizeCapPercent", 10, 100);
        readonly NumericUpDown keyInterval = Num("keyInterval", 0, 100000);
        readonly CheckBox sceneCut = Check("sceneCut", "場面の切り替わりをキーフレームにする");
        readonly NumericUpDown bframes = Num("bframes", -1, 15);
        readonly NumericUpDown refs = Num("refs", 0, 7);
        readonly NumericUpDown bq = Num("bq", -100, 100);
        readonly NumericUpDown aq = Num("aq", -100, 100);
        readonly NumericUpDown kq = Num("kq", -100, 100);
        readonly NumericUpDown deblock = Num("deblockDelta", -16, 15);
        readonly CheckBox cdef = Check("cdef", "CDEF（輪郭のにじみ取り）を使う");
        readonly ComboBox device = Combo("device");  // filled from deviceOptions
        readonly List<DeviceOption> deviceOptions;
        readonly ComboBox tiles = Combo("tiles", "自動", "1", "2", "4");
        readonly ComboBox tileRows = Combo("tileRows", "自動", "1", "2", "4");
        readonly NumericUpDown threads = Num("threads", 0, 256);
        readonly ComboBox parallelB = Combo("parallelB", "自動", "する", "しない");
        readonly ComboBox workers = Combo("workers", "自動", "1", "2", "4");
        static readonly int[] WorkerCounts = { 0, 1, 2, 4 };  // workers' items as EncodeSettings.Workers
        readonly ComboBox container = Combo("container", "MP4", "別ファイル");
        readonly NumericUpDown audioKbps = Num("audioKbps", 32, 512);
        readonly CheckBox heAac = Check("heAacV2", "HE-AACv2 に変換する");
        readonly NumericUpDown heAacKbps = Num("heAacKbps", 12, 96);

        // Set once a preset was saved from this dialog (also when the dialog is then cancelled).
        public string SavedPresetName { get; private set; }

        // store == null disables "プリセットとして保存". devices: the choices of where the decisions run
        // (Prober.ListDevices()); null: the fixed 自動 / GPU / CPU.
        public AdvancedDialog(EncodeSettings s, PresetStore store = null, IList<DeviceOption> devices = null)
        {
            original = s;
            this.store = store;
            // A device the settings name but this machine no longer lists stays selectable, marked.
            deviceOptions = new List<DeviceOption>(devices ?? DeviceList.Fallback());
            string want = s.DeviceValue;
            if (!deviceOptions.Any(o => o.Arg == want)) deviceOptions.Add(new DeviceOption { Arg = want, Label = "（未検出）" + want });
            foreach (var o in deviceOptions) device.Items.Add(o.Label);
            device.Width = 320;  // GPU names are long
            Text = "詳細設定";
            AutoScaleMode = AutoScaleMode.Dpi;
            Font = SystemFonts.MessageBoxFont;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = MinimizeBox = false;
            ShowInTaskbar = false;
            StartPosition = FormStartPosition.CenterParent;
            ClientSize = new Size(760, 400);

            // Every celv2enc option, grouped by what it trades.
            var tabs = new TabControl { Dock = DockStyle.Fill, Name = "tabs" };
            qSlider.ValueChanged += (a, b) => qBox.Value = qSlider.Value;
            qBox.ValueChanged += (a, b) => qSlider.Value = (int)qBox.Value;
            keepUnder.CheckedChanged += (a, b) => sizeCap.Enabled = keepUnder.Checked;
            heAac.CheckedChanged += (a, b) => { heAacKbps.Enabled = heAac.Checked; audioKbps.Enabled = !heAac.Checked; };

            var g = Page(tabs, "画質");
            AddRow(g, "qindex（1〜255、小さいほど高画質）", qSlider, qBox);
            AddRow(g, "圧縮済みの動画", keepUnder, Note("画質（qindex）を自動で下げて合わせる"));
            AddRow(g, "上限（元の動画の映像ビットレート比 %）", sizeCap, Note("標準 90"));

            g = Page(tabs, "フレーム構成");
            AddRow(g, "キーフレーム間隔（枚）", keyInterval, Note("0 = 最初の1枚だけ"));
            AddRow(g, "場面の切り替わり", sceneCut, null);
            AddRow(g, "B フレーム数", bframes, Note("-1 = 自動（参照 5 以上・qindex 100 以上で 7、ほかは 3）。0 = なし"));
            AddRow(g, "参照フレーム数", refs, Note("0 = 標準（4）。多いほど小さく、少し遅い"));
            AddRow(g, "B フレームの qindex 差", bq, Note("標準 +16（B フレームを粗く）"));
            AddRow(g, "隠し参照フレームの qindex 差", aq, Note("標準 -16（参照される先を細かく）"));
            AddRow(g, "キーフレームの qindex 差", kq, Note("標準 0"));

            g = Page(tabs, "フィルター");
            AddRow(g, "デブロック強度", deblock, Note("-16〜15、標準 2。大きいほどブロック境界を強く消す"));
            AddRow(g, "CDEF", cdef, Note("標準オン（約 10% 小さく）"));

            g = Page(tabs, "速度・処理");
            AddRow(g, "判定の場所", device, Note("自動 = CUDA の GPU、なければ Vulkan の GPU、なければ CPU"));
            AddRow(g, "タイル（横）", tiles, Note("自動 = 画面サイズから決める"));
            AddRow(g, "タイル（縦）", tileRows, Note("自動 = 画面サイズから決める"));
            AddRow(g, "スレッド数", threads, Note("0 = 自動"));
            AddRow(g, "B フレームの並列処理", parallelB, Note("自動 = 4K の 2 倍の画素まで（速くなる範囲）"));
            AddRow(g, "場面ごとの並列", workers, Note("同時に変換する場面の数。自動 = 使えるメモリと CPU から"));

            g = Page(tabs, "出力");
            AddRow(g, "出力形式", container, Note("MP4 は MP4Box などが必要"));
            AddRow(g, "音声（AAC）の kbps", audioKbps, Note("音声をそのままコピーできないときに使う"));
            AddRow(g, "音声の形式", heAac, Note("いつも変換し、ステレオにする。fdkaac.exe が必要"));
            AddRow(g, "HE-AACv2 の kbps", heAacKbps, Note("12〜96、標準 32。48 くらいまでが向く"));

            var save = new Button { Text = "プリセットとして保存...", AutoSize = true, Enabled = store != null };
            save.Click += (a, b) => PromptAndSave();
            var ok = new Button { Text = "OK", DialogResult = DialogResult.OK, AutoSize = true };
            var cancel = new Button { Text = "キャンセル", DialogResult = DialogResult.Cancel, AutoSize = true };
            var buttons = new FlowLayoutPanel { Dock = DockStyle.Bottom, AutoSize = true, Padding = new Padding(12, 6, 12, 12), FlowDirection = FlowDirection.LeftToRight };
            buttons.Controls.AddRange(new Control[] { save, ok, cancel });
            Controls.Add(tabs);
            Controls.Add(buttons);
            AcceptButton = ok;
            CancelButton = cancel;

            qBox.Value = Math.Max(1, Math.Min(255, s.QIndex));
            qSlider.Value = (int)qBox.Value;
            keepUnder.Checked = s.KeepUnderSource;
            sizeCap.Enabled = s.KeepUnderSource;
            Set(sizeCap, s.SizeCapPercent);
            Set(keyInterval, s.KeyInterval);
            sceneCut.Checked = s.SceneCut;
            Set(bframes, s.BFrames);
            Set(refs, s.Refs);
            Set(bq, s.BQ);
            Set(aq, s.AQ);
            Set(kq, s.KQ);
            Set(deblock, s.DeblockDelta);
            cdef.Checked = s.Cdef;
            device.SelectedIndex = deviceOptions.FindIndex(o => o.Arg == s.DeviceValue);
            tiles.SelectedIndex = InRange((int)s.Tiles, tiles);
            tileRows.SelectedIndex = InRange((int)s.TileRows, tileRows);
            Set(threads, s.Threads);
            parallelB.SelectedIndex = InRange((int)s.ParallelB, parallelB);
            workers.SelectedIndex = Math.Max(0, Array.IndexOf(WorkerCounts, s.Workers));
            container.SelectedIndex = InRange((int)s.Container, container);
            Set(audioKbps, s.AudioKbps);
            heAac.Checked = s.HeAacV2;
            heAacKbps.Enabled = s.HeAacV2;
            audioKbps.Enabled = !s.HeAacV2;
            Set(heAacKbps, s.HeAacKbps);
        }

        static int InRange(int i, ComboBox c) { return i >= 0 && i < c.Items.Count ? i : 0; }

        static void Set(NumericUpDown n, int v) { n.Value = Math.Max(n.Minimum, Math.Min(n.Maximum, v)); }

        static ComboBox Combo(string name, params string[] items)
        {
            var c = new ComboBox { Name = name, DropDownStyle = ComboBoxStyle.DropDownList, Width = 120 };
            c.Items.AddRange(items);
            return c;
        }

        static NumericUpDown Num(string name, int min, int max)
        {
            return new NumericUpDown { Name = name, Minimum = min, Maximum = max, Width = 90 };
        }

        static CheckBox Check(string name, string text) { return new CheckBox { Name = name, Text = text, AutoSize = true }; }

        static Label Note(string text) { return new Label { Text = text, AutoSize = true, Anchor = AnchorStyles.Left }; }

        // A tab holding a label / control / note grid.
        static TableLayoutPanel Page(TabControl tabs, string title)
        {
            var page = new TabPage(title) { AutoScroll = true };
            var grid = new TableLayoutPanel { ColumnCount = 3, AutoSize = true, Padding = new Padding(12), Dock = DockStyle.Top };
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 200));
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            page.Controls.Add(grid);
            tabs.TabPages.Add(page);
            return grid;
        }

        static void AddRow(TableLayoutPanel g, string label, Control main, Control side)
        {
            int r = g.RowCount++;
            g.Controls.Add(new Label { Text = label, AutoSize = true, Anchor = AnchorStyles.Left, Margin = new Padding(3, 8, 12, 8) }, 0, r);
            g.Controls.Add(main, 1, r);
            if (side != null) g.Controls.Add(side, 2, r);
            else g.SetColumnSpan(main, 2);  // a long check box text runs past the control column
        }

        // A fresh copy built from the controls each time.
        public EncodeSettings Result
        {
            get
            {
                var r = original.Clone();
                r.QIndex = (int)qBox.Value;
                r.KeepUnderSource = keepUnder.Checked;
                r.SizeCapPercent = (int)sizeCap.Value;
                r.KeyInterval = (int)keyInterval.Value;
                r.SceneCut = sceneCut.Checked;
                r.BFrames = (int)bframes.Value;
                r.Refs = (int)refs.Value;
                r.BQ = (int)bq.Value;
                r.AQ = (int)aq.Value;
                r.KQ = (int)kq.Value;
                r.DeblockDelta = (int)deblock.Value;
                r.Cdef = cdef.Checked;
                string arg = deviceOptions[Math.Max(0, device.SelectedIndex)].Arg;
                r.DeviceArg = arg;
                r.Device = arg == "auto" ? DecisionDevice.Auto : arg == "cpu" ? DecisionDevice.Cpu : DecisionDevice.Gpu;  // for older versions
                r.Tiles = (TileMode)tiles.SelectedIndex;
                r.TileRows = (TileMode)tileRows.SelectedIndex;
                r.Threads = (int)threads.Value;
                r.ParallelB = (ParallelMode)parallelB.SelectedIndex;
                r.Workers = WorkerCounts[workers.SelectedIndex];
                r.Container = (ContainerKind)container.SelectedIndex;
                r.AudioKbps = (int)audioKbps.Value;
                r.HeAacV2 = heAac.Checked;
                r.HeAacKbps = (int)heAacKbps.Value;
                return r;
            }
        }

        // Saves Result under `name`; refuses empty names and the built-in ones (ruling 8). Returns whether it was saved.
        public bool TrySavePreset(string name)
        {
            name = (name ?? "").Trim();
            if (store == null) return false;
            if (name.Length == 0) { Warn("プリセット名を入力してください。"); return false; }
            if (PresetStore.BuiltIn.Any(p => string.Equals(p.Name, name, StringComparison.OrdinalIgnoreCase)))
            {
                Warn("「" + name + "」は組み込みのプリセット名です。別の名前で保存してください。");
                return false;
            }
            // preset files are named after the preset: the one already in that file is replaced
            Preset same;
            try { same = store.SameFileAs(name); }
            catch (ArgumentException e) { Warn("保存できませんでした: " + e.Message); return false; }
            if (same != null
                && !(ConfirmOverwrite != null ? ConfirmOverwrite(same.Name)
                     : MessageBox.Show(this, "「" + same.Name + "」を上書きしますか？", Text, MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes))
                return false;
            try { store.Save(new Preset { Name = name, Settings = Result }); }
            catch (Exception e) when (e is System.IO.IOException || e is UnauthorizedAccessException || e is ArgumentException)
            {
                Warn("保存できませんでした: " + e.Message);
                return false;
            }
            SavedPresetName = name;
            return true;
        }

        // Replaceable like ShowWarning.
        public Func<string, bool> ConfirmOverwrite;

        // Replaceable so a test can capture the message instead of blocking on a message box.
        public Action<string> ShowWarning;

        void Warn(string msg)
        {
            if (ShowWarning != null) ShowWarning(msg);
            else MessageBox.Show(this, msg, Text, MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }

        void PromptAndSave()
        {
            using (var f = new Form { Text = "プリセットとして保存", AutoScaleMode = AutoScaleMode.Dpi, Font = Font, FormBorderStyle = FormBorderStyle.FixedDialog,
                                      MaximizeBox = false, MinimizeBox = false, ShowInTaskbar = false, StartPosition = FormStartPosition.CenterParent,
                                      ClientSize = new Size(360, 110) })
            {
                var box = new TextBox { Left = 12, Top = 32, Width = 336 };
                var ok = new Button { Text = "保存", DialogResult = DialogResult.OK, Left = 192, Top = 70, Width = 75 };
                var cancel = new Button { Text = "キャンセル", DialogResult = DialogResult.Cancel, Left = 273, Top = 70, Width = 75 };
                f.Controls.AddRange(new Control[] { new Label { Text = "プリセット名", Left = 12, Top = 10, AutoSize = true }, box, ok, cancel });
                f.AcceptButton = ok; f.CancelButton = cancel;
                while (f.ShowDialog(this) == DialogResult.OK)
                    if (TrySavePreset(box.Text)) break;
            }
        }
    }
}
