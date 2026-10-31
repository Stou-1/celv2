using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Media;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using CelV2.Runner;
using CelV2.Settings;

namespace CelV2.Ui
{
    // The single window: drop files, pick a quality, convert one file at a time.
    // Everything environment-specific comes through the constructor so another host (e.g. a plugin) can create it.
    public sealed class MainForm : Form
    {
        sealed class Item
        {
            public string Path;
            public InputInfo Info;          // null until probed, or when probing failed
            public string Error;            // probe or conversion failure
            public bool Done;
            public ListViewItem Row;
            public string OutputName;       // null: the input's file name
        }

        const int ColSize = 1, ColFps = 2, ColLength = 3, ColAudio = 4, ColNote = 5, ColStatus = 6;

        readonly AppSettings app;
        readonly string settingsPath, toolsFolder;
        readonly PresetStore store;
        readonly List<Item> items = new List<Item>();
        IReadOnlyList<Preset> presets;
        EncodeSettings current;
        ToolPaths tools;
        List<DeviceOption> devices;  // OpenAdvanced(): where the decisions can run
        bool busy, updatingPresets, showingName;
        int pendingProbes;
        CancellationTokenSource cts;
        Thread worker;
        volatile string lastOutput;
        string engine = "";
        JobProgress latest;
        int progressPosted;

        readonly Panel dropZone = new Panel { BorderStyle = BorderStyle.FixedSingle, Dock = DockStyle.Fill, AllowDrop = true };
        readonly Button pickButton = new Button { Name = "pickButton", Text = "ファイルを選ぶ", AutoSize = true };
        readonly ListView list = new ListView { Name = "list", View = View.Details, FullRowSelect = true, GridLines = true, Dock = DockStyle.Fill, HideSelection = false, MultiSelect = false };
        readonly TextBox nameBox = new TextBox { Name = "outputNameBox", Enabled = false };
        readonly Label namePath = new Label { Name = "outputNamePath", AutoSize = true };
        readonly Label nameError = new Label { Name = "outputNameError", AutoSize = true, ForeColor = Color.Firebrick };
        readonly ComboBox presetBox = new ComboBox { Name = "presetBox", DropDownStyle = ComboBoxStyle.DropDownList, Width = 160 };
        // celv2enc --tune, in TuneKind's order; kept when the preset changes.
        readonly ComboBox tuneBox = new ComboBox { Name = "tuneBox", DropDownStyle = ComboBoxStyle.DropDownList, Width = 120 };
        readonly CheckBox maxBox = new CheckBox { Name = "maxBox", Text = "とても遅いが最高画質", AutoSize = true };
        // The whole output at most this many MB (1 MB = 1,000,000 bytes), in 2 passes; 0 = off. Per session.
        readonly NumericUpDown targetSize = new NumericUpDown { Name = "targetSize", Minimum = 0, Maximum = 100000, DecimalPlaces = 1, Width = 80 };
        readonly Label customLabel = new Label { Name = "customLabel", Text = "（詳細設定で変更中）", AutoSize = true, Visible = false, Anchor = AnchorStyles.Left };
        readonly Button advancedButton = new Button { Name = "advancedButton", Text = "詳細設定...", AutoSize = true };
        readonly TextBox outputBox = new TextBox { Name = "outputBox", ReadOnly = true, Width = 330 };
        readonly Button outputButton = new Button { Name = "outputButton", Text = "変更...", AutoSize = true };
        readonly Button outputResetButton = new Button { Name = "outputResetButton", Text = "入力と同じ", AutoSize = true };
        readonly ProgressBar bar = new ProgressBar { Name = "progressBar", Dock = DockStyle.Fill, Minimum = 0, Maximum = 100 };
        readonly Label progressLabel = new Label { Name = "progressLabel", AutoSize = true, Text = " " };
        readonly Button startButton = new Button { Name = "startButton", Text = "変換開始", AutoSize = true };
        readonly Button cancelButton = new Button { Name = "cancelButton", Text = "中止", AutoSize = true, Enabled = false };
        readonly Button clearButton = new Button { Name = "clearButton", Text = "一覧を空にする", AutoSize = true };
        readonly Button toolsButton = new Button { Name = "toolsButton", Text = "ツール設定...", AutoSize = true };
        readonly Label resultLabel = new Label { Name = "resultLabel", AutoSize = true, Anchor = AnchorStyles.Left, Visible = false };
        readonly Button openFolderButton = new Button { Name = "openFolderButton", Text = "フォルダを開く", AutoSize = true, Visible = false };

        // settingsPath: where AppSettings is saved; the preset folder is "presets" next to it.
        // toolsFolder: where celv2enc.exe / av2_mux.exe (and optionally ffmpeg) live; default is this program's folder.
        public MainForm(AppSettings s, string settingsPath, string toolsFolder = null)
        {
            app = s;
            this.settingsPath = settingsPath;
            this.toolsFolder = toolsFolder ?? AppDomain.CurrentDomain.BaseDirectory;
            store = new PresetStore(System.IO.Path.Combine(System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(settingsPath)), "presets"));

            Text = "celv2 - 動画を AV2 に変換";
            AutoScaleDimensions = new SizeF(96F, 96F);
            AutoScaleMode = AutoScaleMode.Dpi;
            Font = SystemFonts.MessageBoxFont;
            ClientSize = new Size(1100, 560);
            MinimumSize = new Size(640, 420);
            AllowDrop = true;
            tuneBox.Items.AddRange(new object[] { "なし", "アニメ", "実写", "静止画多め" });
            tuneBox.SelectedIndex = (int)app.LastTune >= 0 && (int)app.LastTune < tuneBox.Items.Count ? (int)app.LastTune : 0;
            BuildLayout();
            maxBox.Checked = app.LastMaxEffort;  // before ReloadPresets, which copies it into the settings
            ReloadPresets(app.LastPreset);
            ShowOutputFolder();
            UpdateButtons();

            DragEnter += OnDragEnter; DragDrop += OnDragDrop;
            pickButton.Click += (a, b) => PickFiles();
            presetBox.SelectedIndexChanged += (a, b) => OnPresetChanged();
            tuneBox.SelectedIndexChanged += (a, b) => { current.Tune = SelectedTune; app.LastTune = SelectedTune; SaveSettings(); };
            maxBox.CheckedChanged += (a, b) => { current.MaxEffort = maxBox.Checked; app.LastMaxEffort = maxBox.Checked; SaveSettings(); };
            targetSize.ValueChanged += (a, b) => current.TargetSizeMB = (double)targetSize.Value;
            advancedButton.Click += (a, b) => OpenAdvanced();
            outputButton.Click += (a, b) => PickOutputFolder();
            outputResetButton.Click += (a, b) => { app.OutputFolder = null; ShowOutputFolder(); SaveSettings(); };
            startButton.Click += (a, b) => StartEncode();
            cancelButton.Click += (a, b) => CancelEncode();
            clearButton.Click += (a, b) => { items.Clear(); list.Items.Clear(); ShowName(); };
            toolsButton.Click += (a, b) => ShowToolSetup();
            openFolderButton.Click += (a, b) => OpenResultFolder();
            list.DoubleClick += (a, b) => ShowDetail();
            list.SelectedIndexChanged += (a, b) => ShowName();
            nameBox.TextChanged += (a, b) => OnNameEdited();
            Shown += (a, b) => EnsureTools();
            FormClosing += OnClosing;
        }

        // ---- public surface (what the drop handler, a test or another host drives) ----

        public IReadOnlyList<string> PresetNames { get { return presets.Select(p => p.Name).ToList(); } }
        public string SelectedPresetName { get { return presetBox.SelectedItem as string; } }
        public EncodeSettings EncodeSettingsForTest { get { return current.Clone(); } }
        public TuneKind SelectedTune { get { return tuneBox.SelectedIndex > 0 ? (TuneKind)tuneBox.SelectedIndex : TuneKind.None; } }
        public bool IsBusy { get { return busy; } }
        public int PendingProbes { get { return pendingProbes; } }

        // The drop handler and the file picker both end here. Probing runs off the UI thread.
        public void AddFiles(IEnumerable<string> paths)
        {
            if (busy || !EnsureTools()) return;
            var probeTools = tools;
            foreach (string path in paths.Where(File.Exists))
            {
                // a file still waiting is not added twice; a converted one can be added again (another name or settings)
                if (items.Any(i => !i.Done && string.Equals(i.Path, path, StringComparison.OrdinalIgnoreCase))) continue;
                var item = new Item { Path = path };
                item.Row = new ListViewItem(new[] { System.IO.Path.GetFileName(path), "", "", "", "", "", "調べ中" });
                items.Add(item);
                list.Items.Add(item.Row);
                pendingProbes++;
                Task.Run(() =>
                {
                    InputInfo info = null; string err = null;
                    try { info = Prober.Probe(probeTools, path); }
                    catch (Exception e) { err = e.Message; } // any failure just marks this row
                    Ui(() => { item.Info = info; item.Error = err; pendingProbes--; ShowProbe(item); UpdateButtons(); });
                });
            }
            UpdateButtons();
        }

        public void StartEncode()
        {
            if (busy || pendingProbes > 0 || !EnsureTools()) return;
            if (tools.CelV2enc == null)
            {
                MessageBox.Show(this, "celv2enc.exe が見つかりません。CelV2Gui.exe と同じフォルダに置いてください。\n探した場所: " + toolsFolder,
                    Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            var queue = items.Where(i => i.Info != null && !i.Done).ToList();
            if (queue.Count == 0) return;
            var unused = Handle; // make sure BeginInvoke has a window to post to
            busy = true;
            cts = new CancellationTokenSource();
            resultLabel.Visible = openFolderButton.Visible = false;
            UpdateButtons();
            var t = tools; var settings = current.Clone(); var outFolder = app.OutputFolder; var token = cts.Token;
            worker = new Thread(() => RunQueue(queue, t, settings, outFolder, token)) { IsBackground = true, Name = "celv2 encode queue" };
            worker.Start();
        }

        public void CancelEncode()
        {
            var c = cts;
            if (busy && c != null) c.Cancel();
        }

        // ---- layout ----

        void BuildLayout()
        {
            var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 7, Padding = new Padding(10) };
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 84));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            for (int i = 0; i < 5; i++) root.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var drop = new Label
            {
                Text = "ここに動画をドロップ", AutoSize = false, TextAlign = ContentAlignment.MiddleCenter, Dock = DockStyle.Fill,
                Font = new Font(Font.FontFamily, Font.Size * 1.5f), AllowDrop = true, ForeColor = SystemColors.GrayText,
            };
            // the child controls cover the panel, so they take the drops too
            drop.DragEnter += OnDragEnter; drop.DragDrop += OnDragDrop;
            dropZone.DragEnter += OnDragEnter; dropZone.DragDrop += OnDragDrop;
            list.DragEnter += OnDragEnter; list.DragDrop += OnDragDrop;
            list.AllowDrop = true;
            dropZone.Controls.Add(drop);
            var top = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1, Margin = new Padding(0) };
            top.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            top.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            top.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            pickButton.Anchor = AnchorStyles.None;
            top.Controls.Add(dropZone, 0, 0);
            top.Controls.Add(pickButton, 1, 0);
            root.Controls.Add(top, 0, 0);

            list.Columns.Add("ファイル名", Dp(220)); list.Columns.Add("解像度", Dp(80)); list.Columns.Add("fps", Dp(55));
            list.Columns.Add("長さ", Dp(65)); list.Columns.Add("音声", Dp(55)); list.Columns.Add("注意", Dp(250)); list.Columns.Add("状態", Dp(100));
            // the output name of the selected row sits to the right of the list
            var middle = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1, Margin = new Padding(0) };
            middle.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            middle.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, Dp(240)));
            middle.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            nameBox.Width = Dp(225);
            namePath.MaximumSize = nameError.MaximumSize = new Size(Dp(225), 0);
            var side = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.TopDown, WrapContents = false, Padding = new Padding(Dp(6), 0, 0, 0) };
            side.Controls.Add(Caption("出力名（一覧で選んだファイル）"));
            side.Controls.Add(nameBox);
            side.Controls.Add(new Label { Text = "空欄にすると入力と同じ名前", AutoSize = true, ForeColor = SystemColors.GrayText });
            side.Controls.Add(nameError);
            side.Controls.Add(namePath);
            middle.Controls.Add(list, 0, 0);
            middle.Controls.Add(side, 1, 0);
            root.Controls.Add(middle, 0, 1);

            root.Controls.Add(Row(Caption("画質"), presetBox, Caption("内容"), tuneBox, maxBox, Caption("サイズ指定（MB、0 = オフ）"), targetSize, advancedButton, customLabel), 0, 2);
            root.Controls.Add(Row(Caption("出力先"), outputBox, outputButton, outputResetButton), 0, 3);
            bar.Height = Dp(20);
            root.Controls.Add(bar, 0, 4);
            root.Controls.Add(progressLabel, 0, 5);
            root.Controls.Add(Row(startButton, cancelButton, clearButton, toolsButton, resultLabel, openFolderButton), 0, 6);
            Controls.Add(root);
        }

        int Dp(int px) { return LogicalToDeviceUnits(px); }

        static Label Caption(string text) { return new Label { Text = text, AutoSize = true, Anchor = AnchorStyles.Left }; }

        static FlowLayoutPanel Row(params Control[] cs)
        {
            var f = new FlowLayoutPanel { AutoSize = true, Dock = DockStyle.Fill, WrapContents = false };
            f.Controls.AddRange(cs);
            return f;
        }

        // ---- drag and drop / pickers ----

        void OnDragEnter(object s, DragEventArgs e)
        {
            e.Effect = !busy && e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.Copy : DragDropEffects.None;
        }

        void OnDragDrop(object s, DragEventArgs e)
        {
            var files = e.Data.GetData(DataFormats.FileDrop) as string[];
            if (files == null) return;
            // leave Explorer's drag loop first: AddFiles may open a dialog
            BeginInvoke((Action)(() => AddFiles(files)));
        }

        void PickFiles()
        {
            using (var d = new OpenFileDialog { Title = "変換する動画を選ぶ", Multiselect = true, Filter = "すべてのファイル|*.*" })
                if (d.ShowDialog(this) == DialogResult.OK) AddFiles(d.FileNames);
        }

        void PickOutputFolder()
        {
            using (var d = new FolderBrowserDialog { Description = "出力先のフォルダを選ぶ", SelectedPath = app.OutputFolder ?? "" })
                if (d.ShowDialog(this) == DialogResult.OK) { app.OutputFolder = d.SelectedPath; ShowOutputFolder(); SaveSettings(); }
        }

        void ShowOutputFolder() { outputBox.Text = app.OutputFolder ?? "（入力と同じフォルダ）"; ShowName(); }

        // ---- output name ----

        Item Selected()
        {
            if (list.SelectedItems.Count == 0) return null;
            var row = list.SelectedItems[0];
            return items.FirstOrDefault(i => i.Row == row);
        }

        void ShowName()
        {
            var item = Selected();
            showingName = true;
            nameBox.Text = item == null ? "" : item.OutputName ?? System.IO.Path.GetFileNameWithoutExtension(item.Path);
            showingName = false;
            ShowNamePreview(item);
            UpdateButtons();
        }

        void OnNameEdited()
        {
            var item = Selected();
            if (showingName || item == null) return;
            string t = nameBox.Text.Trim();
            item.OutputName = t.Length == 0 || t == System.IO.Path.GetFileNameWithoutExtension(item.Path) ? null : t;
            ShowNamePreview(item);
            UpdateButtons();
        }

        // Where the file will go, with the " (2)" the job would add today; the job checks again when it starts.
        void ShowNamePreview(Item item)
        {
            string bad = item == null ? null : OutputNamer.BadName(item.OutputName);
            nameError.Text = bad ?? "";
            if (item == null || bad != null) { namePath.Text = ""; return; }
            string folder = app.OutputFolder ?? System.IO.Path.GetDirectoryName(item.Path);
            namePath.Text = "保存先: " + OutputNamer.Stem(folder, item.Path, null, item.OutputName)
                            + (current.Container == ContainerKind.Mp4 ? ".mp4" : ".ivf");
        }

        // ---- presets ----

        void ReloadPresets(string select)
        {
            updatingPresets = true;
            presets = store.LoadAll();
            presetBox.Items.Clear();
            foreach (var p in presets) presetBox.Items.Add(p.Name);
            int i = presets.ToList().FindIndex(p => p.Name == select);
            presetBox.SelectedIndex = i >= 0 ? i : 1; // 標準
            current = presets[presetBox.SelectedIndex].Settings.Clone();
            ApplyBoxes();
            customLabel.Visible = false;
            updatingPresets = false;
        }

        // What the main window sets on top of the preset or the 詳細設定 result.
        void ApplyBoxes()
        {
            current.Tune = SelectedTune;
            current.MaxEffort = maxBox.Checked;
            current.TargetSizeMB = (double)targetSize.Value;
        }

        void OnPresetChanged()
        {
            if (updatingPresets || presetBox.SelectedIndex < 0) return;
            current = presets[presetBox.SelectedIndex].Settings.Clone();
            ApplyBoxes();
            customLabel.Visible = false;
            app.LastPreset = presets[presetBox.SelectedIndex].Name;
            SaveSettings();
            ShowName();
        }

        void OpenAdvanced()
        {
            if (devices == null) devices = Prober.ListDevices(tools);  // once a session: about 0.5 s
            using (var d = new AdvancedDialog(current, store, devices))
            {
                bool ok = d.ShowDialog(this) == DialogResult.OK;
                string saved = d.SavedPresetName;
                var edited = d.Result;
                if (saved != null) { ReloadPresets(saved); app.LastPreset = saved; SaveSettings(); } // current = the saved preset
                if (ok)
                {
                    current = edited;
                    ApplyBoxes();
                    customLabel.Visible = saved == null;
                }
            }
            ShowName();
        }

        // ---- tools ----

        ToolPaths FindTools()
        {
            return ToolLocator.Find(app.FfmpegFolder, app.Mp4BoxFolder, toolsFolder, toolsFolder);
        }

        void ShowToolSetup()
        {
            using (var d = new ToolSetupDialog(app.FfmpegFolder, app.Mp4BoxFolder))
            {
                if (d.ShowDialog(this) != DialogResult.OK) return;
                app.FfmpegFolder = d.FfmpegFolder;
                app.Mp4BoxFolder = d.Mp4BoxFolder;
                SaveSettings();
            }
        }

        // Finds the tools; when ffmpeg / ffprobe are missing, asks the user once and looks again.
        bool EnsureTools()
        {
            tools = FindTools();
            if (tools.Ffmpeg == null || tools.Ffprobe == null)
            {
                ShowToolSetup();
                tools = FindTools();
            }
            return tools.Ffmpeg != null && tools.Ffprobe != null;
        }

        void SaveSettings()
        {
            try { app.Save(settingsPath); }
            catch (Exception e) when (e is IOException || e is UnauthorizedAccessException) { } // remembering is best effort
        }

        // ---- list ----

        void ShowProbe(Item item)
        {
            var sub = item.Row.SubItems;
            if (item.Info == null)
            {
                sub[ColStatus].Text = "失敗";
                sub[ColNote].Text = "読み込めません: " + ShortReason(item.Error, item.Path);
                return;
            }
            var i = item.Info;
            sub[ColSize].Text = i.Width + "x" + i.Height;
            sub[ColFps].Text = i.FpsDen > 0 ? ((double)i.FpsNum / i.FpsDen).ToString("0.##") : "?";
            sub[ColLength].Text = Clock(TimeSpan.FromSeconds(i.DurationSeconds));
            sub[ColAudio].Text = i.AudioCodec ?? "なし";
            sub[ColNote].Text = NotesFor(i, current.Output, null);
            sub[ColStatus].Text = "待機";
        }

        // The 注意 text of a row that is not failed: the automatic conversions plus what the job did differently.
        // Built from scratch each time, so nothing from an earlier run survives.
        public static string NotesFor(InputInfo i, OutputFormat f, string jobNote)
        {
            var n = new List<string>();
            if (i.BitDepth > f.BitDepth) n.Add(i.BitDepth + "bit→" + f.BitDepth + "bit");
            if (i.Hdr && !CommandBuilder.KeepsHdr(i, f)) n.Add("HDR→SDR（トーンマップ）");
            if (i.VariableFrameRate) n.Add("可変fps→固定");
            if (i.Interlaced) n.Add("インターレース解除");
            if (!string.IsNullOrEmpty(jobNote)) n.Add(jobNote);
            return string.Join("、", n);
        }

        // One line for the list: no newlines and without the input's path (the full text is on double-click).
        public static string ShortReason(string error, string inputPath)
        {
            string s = Flat(error);
            if (s.Length == 0) return s;
            string name = System.IO.Path.GetFileName(inputPath);
            s = s.Replace(inputPath, name);
            if (s.StartsWith(name + ": ", StringComparison.Ordinal)) s = s.Substring(name.Length + 2);
            return s;
        }

        static string Flat(string s) { return (s ?? "").Replace("\r", "").Replace('\n', ' ').Trim(); }

        static string Clock(TimeSpan t)
        {
            return t.TotalHours >= 1 ? string.Format("{0}:{1:00}:{2:00}", (int)t.TotalHours, t.Minutes, t.Seconds)
                                     : string.Format("{0}:{1:00}", t.Minutes, t.Seconds);
        }

        // Double-click shows the whole reason, which the narrow column cuts off.
        void ShowDetail()
        {
            if (list.SelectedItems.Count == 0) return;
            var row = list.SelectedItems[0];
            var item = items.FirstOrDefault(i => i.Row == row);
            string full = item != null && item.Error != null ? item.Error : row.SubItems[ColNote].Text;
            MessageBox.Show(this, row.Text + "\n状態: " + row.SubItems[ColStatus].Text + "\n\n" + full, Text);
        }

        void UpdateButtons()
        {
            startButton.Enabled = !busy && pendingProbes == 0 && items.Any(i => i.Info != null && !i.Done)
                                  && items.Where(i => i.Info != null && !i.Done).All(i => OutputNamer.BadName(i.OutputName) == null);
            nameBox.Enabled = !busy && Selected() != null;
            cancelButton.Enabled = busy;
            clearButton.Enabled = !busy;
            presetBox.Enabled = tuneBox.Enabled = maxBox.Enabled = advancedButton.Enabled = outputButton.Enabled = outputResetButton.Enabled = pickButton.Enabled = !busy;
        }

        // ---- the queue ----

        void Ui(Action a)
        {
            try { if (!IsDisposed && IsHandleCreated) BeginInvoke(a); }
            catch (Exception e) when (e is InvalidOperationException || e is ObjectDisposedException) { } // the window went away meanwhile
        }

        void RunQueue(List<Item> queue, ToolPaths t, EncodeSettings settings, string outFolder, CancellationToken cancel)
        {
            int ok = 0, failed = 0;
            bool cancelled = false;
            foreach (var item in queue)
            {
                if (cancel.IsCancellationRequested) { cancelled = true; break; }
                Ui(() =>
                {
                    engine = "";
                    item.Error = null; // a retry starts clean
                    item.Row.SubItems[ColNote].Text = NotesFor(item.Info, settings.Output, null);
                    item.Row.SubItems[ColStatus].Text = "変換中";
                    item.Row.EnsureVisible();
                    progressLabel.Text = item.Row.Text;
                    bar.Style = ProgressBarStyle.Marquee;
                });
                JobResult r;
                try
                {
                    var job = new EncodeJob(t, item.Path, item.Info, settings.Clone(), outFolder ?? System.IO.Path.GetDirectoryName(item.Path), null, item.OutputName);
                    job.Progress += p =>
                    {
                        latest = p; // the window shows the newest value; older ones are dropped
                        if (Interlocked.Exchange(ref progressPosted, 1) == 0)
                            Ui(() => { Interlocked.Exchange(ref progressPosted, 0); ShowProgress(item, latest); });
                    };
                    job.EngineLine += l => Ui(() => ShowEngine(l));
                    job.Stage += st => Ui(() => { bar.Style = ProgressBarStyle.Marquee; progressLabel.Text = item.Row.Text + "　" + st; });
                    r = job.Run(cancel);
                }
                catch (Exception e) { r = new JobResult { Error = e.Message }; } // keep the queue going
                var res = r;
                if (res.Cancelled)
                {
                    cancelled = true;
                    Ui(() => item.Row.SubItems[ColStatus].Text = "中止");
                    break;
                }
                if (res.Ok)
                {
                    ok++;
                    if (res.Outputs.Count > 0) lastOutput = res.Outputs[0];
                }
                else failed++;
                Ui(() =>
                {
                    item.Row.SubItems[ColStatus].Text = res.Ok ? "完了" : "失敗";
                    item.Done = res.Ok;
                    if (!res.Ok) { item.Error = res.Error; item.Row.SubItems[ColNote].Text = ShortReason(res.Error, item.Path); }
                    else item.Row.SubItems[ColNote].Text = NotesFor(item.Info, settings.Output, res.Note);
                });
            }
            Ui(() => Finished(ok, failed, cancelled));
        }

        void ShowEngine(string line)
        {
            engine = line.Contains("decisions on GPU") ? "GPU" : line.Contains("decisions on CPU") ? "CPU" : "";
        }

        void ShowProgress(Item item, JobProgress p)
        {
            string tail = string.Format("{0:0.0} fps　残り {1}", p.Fps, p.Total > 0 ? Clock(p.Remaining) : "?")
                          + (engine.Length > 0 ? "　判定: " + engine : "");
            if (p.Total > 0)
            {
                bar.Style = ProgressBarStyle.Continuous;
                bar.Maximum = (int)Math.Min(int.MaxValue, p.Total);
                bar.Value = (int)Math.Min(p.Frame, bar.Maximum);
                progressLabel.Text = item.Row.Text + "　" + p.Frame + " / " + p.Total + " 枚　" + tail;
            }
            else
            {
                bar.Style = ProgressBarStyle.Marquee;
                progressLabel.Text = item.Row.Text + "　" + p.Frame + " 枚　" + tail;
            }
        }

        void Finished(int ok, int failed, bool cancelled)
        {
            busy = false;
            bar.Style = ProgressBarStyle.Continuous;
            bar.Value = 0;
            progressLabel.Text = " ";
            engine = "";
            resultLabel.Text = (cancelled ? "中止しました。" : "完了しました。") + "成功 " + ok + " 件" + (failed > 0 ? "、失敗 " + failed + " 件" : "");
            resultLabel.Visible = true;
            openFolderButton.Visible = lastOutput != null && ok > 0;
            UpdateButtons();
            if (!cancelled) SystemSounds.Asterisk.Play();
        }

        void OpenResultFolder()
        {
            string o = lastOutput;
            if (o == null) return;
            try { Process.Start("explorer.exe", File.Exists(o) ? "/select,\"" + o + "\"" : "\"" + System.IO.Path.GetDirectoryName(o) + "\""); }
            catch (Exception e) when (e is System.ComponentModel.Win32Exception || e is InvalidOperationException)
            {
                MessageBox.Show(this, "フォルダを開けませんでした: " + e.Message, Text, MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        // Replaceable so a test can answer the "stop and quit?" question without a message box.
        public Func<bool> ConfirmClose;

        void OnClosing(object s, FormClosingEventArgs e)
        {
            if (!busy) return;
            bool yes = ConfirmClose != null ? ConfirmClose()
                : MessageBox.Show(this, "変換中です。中止して終了しますか？", Text, MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes;
            if (!yes)
            {
                e.Cancel = true;
                return;
            }
            cts.Cancel();
            worker.Join(5000); // the job kills its tools and removes partial files on cancel
        }
    }
}
