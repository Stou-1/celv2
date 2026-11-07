using System;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using CelV2.Settings;
using CelV2.Ui;

namespace CelV2.Tests
{
    public static class UiTests
    {
        public static readonly (string name, Action run)[] All =
        {
            ("app_settings_round_trip", AppSettingsRoundTrip),
            ("app_settings_missing_and_corrupt_give_defaults", AppSettingsDefaults),
            ("advanced_dialog_returns_edits", AdvancedDialogReturnsEdits),
            ("advanced_dialog_refuses_builtin_preset_name", AdvancedDialogRefusesBuiltInName),
            ("main_form_builds", MainFormBuilds),
            ("main_form_restores_max_effort", MainFormRestoresMaxEffort),
            ("main_form_notes_target_size", MainFormNotesTargetSize),
            ("main_form_target_size", MainFormTargetSize),
            ("main_form_target_vmaf", MainFormTargetVmaf),
            ("main_form_takes_ivf_for_muxing", MainFormTakesIvfForMuxing),
            ("notes_for_inputs", NotesForInputs),
            ("mux_warning_before_start", MuxWarningBeforeStart),
            ("failure_reason_is_short", FailureReasonIsShort),
            ("advanced_dialog_clamps_bad_enums_and_matches_names_ignoring_case", AdvancedDialogRobust),
        };

        static string TempDir()
        {
            string d = Path.Combine(Path.GetTempPath(), "celv2gui_ui_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(d);
            return d;
        }

        static void AppSettingsRoundTrip()
        {
            string dir = TempDir();
            try
            {
                string path = Path.Combine(dir, "sub", "settings.json");
                new AppSettings { LastPreset = "小容量", OutputFolder = @"D:\出力 先", FfmpegFolder = @"C:\ff mpeg", Mp4BoxFolder = @"C:\gpac", FdkaacFolder = @"C:\fdk aac" }.Save(path);
                var s = AppSettings.Load(path);
                Check.Equal("小容量", s.LastPreset, "LastPreset");
                Check.Equal(@"D:\出力 先", s.OutputFolder, "OutputFolder");
                Check.Equal(@"C:\ff mpeg", s.FfmpegFolder, "FfmpegFolder");
                Check.Equal(@"C:\gpac", s.Mp4BoxFolder, "Mp4BoxFolder");
                Check.Equal(@"C:\fdk aac", s.FdkaacFolder, "FdkaacFolder");
            }
            finally { Directory.Delete(dir, true); }
        }

        static void AppSettingsDefaults()
        {
            string dir = TempDir();
            try
            {
                var missing = AppSettings.Load(Path.Combine(dir, "none.json"));
                Check.Equal("標準", missing.LastPreset, "missing file default preset");
                Check.That(missing.OutputFolder == null && missing.FfmpegFolder == null && missing.Mp4BoxFolder == null, "missing file folders null");

                string corrupt = Path.Combine(dir, "bad.json");
                File.WriteAllText(corrupt, "{ not json");
                Check.Equal("標準", AppSettings.Load(corrupt).LastPreset, "corrupt file default preset");

                string partial = Path.Combine(dir, "partial.json");
                File.WriteAllText(partial, "{\"OutputFolder\":\"X:/o\"}");
                var p = AppSettings.Load(partial);
                Check.Equal("標準", p.LastPreset, "field missing from json keeps its default");
                Check.Equal("X:/o", p.OutputFolder, "present field is read");

                string renamed = Path.Combine(dir, "renamed.json");
                File.WriteAllText(renamed, "{\"LastPreset\":\"小さめ\"}");
                Check.Equal("小容量", AppSettings.Load(renamed).LastPreset, "the old name of 小容量 is carried over");
            }
            finally { Directory.Delete(dir, true); }
        }

        static NumericUpDown Num(Form f, string name)
        {
            return (NumericUpDown)f.Controls.Find(name, true).Single();
        }

        static void AdvancedDialogReturnsEdits()
        {
            var input = new EncodeSettings { QIndex = 150 };
            using (var d = new AdvancedDialog(input))
            {
                Num(d, "qindex").Value = 99;
                Num(d, "keyInterval").Value = 60;
                Num(d, "refs").Value = 5;
                Num(d, "bframes").Value = 4;
                ((CheckBox)d.Controls.Find("keepUnderSource", true).Single()).Checked = false;
                ((ComboBox)d.Controls.Find("container", true).Single()).SelectedIndex = 1;
                Num(d, "sizeCapPercent").Value = 80;
                ((CheckBox)d.Controls.Find("sceneCut", true).Single()).Checked = false;
                Num(d, "bq").Value = 10;
                Num(d, "aq").Value = -20;
                Num(d, "kq").Value = -8;
                Num(d, "deblockDelta").Value = -3;
                ((CheckBox)d.Controls.Find("cdef", true).Single()).Checked = false;
                Check.Equal(-1m, Num(d, "grain").Value, "film grain left to celv2enc by default");
                Num(d, "grain").Value = 25;
                ((ComboBox)d.Controls.Find("tileRows", true).Single()).SelectedIndex = 2;
                ((ComboBox)d.Controls.Find("parallelB", true).Single()).SelectedIndex = 2;
                Num(d, "audioKbps").Value = 256;
                var he = (CheckBox)d.Controls.Find("heAacV2", true).Single();
                Check.That(!Num(d, "heAacKbps").Enabled, "HE-AACv2 kbps off while HE-AACv2 is off");
                he.Checked = true;
                Check.That(Num(d, "heAacKbps").Enabled && !Num(d, "audioKbps").Enabled, "HE-AACv2 kbps on, AAC kbps off");
                Num(d, "heAacKbps").Value = 48;
                var workers = (ComboBox)d.Controls.Find("workers", true).Single();
                Check.Equal(0, workers.SelectedIndex, "workers: 自動 by default");
                workers.SelectedIndex = 2;  // 2
                Check.Equal(5, ((TabControl)d.Controls.Find("tabs", true).Single()).TabPages.Count, "tabs");
                var r = d.Result;
                Check.Equal(80, r.SizeCapPercent, "Result.SizeCapPercent");
                Check.That(!r.SceneCut, "Result.SceneCut");
                Check.Equal(10, r.BQ, "Result.BQ");
                Check.Equal(-20, r.AQ, "Result.AQ");
                Check.Equal(-8, r.KQ, "Result.KQ");
                Check.Equal(-3, r.DeblockDelta, "Result.DeblockDelta");
                Check.That(!r.Cdef, "Result.Cdef");
                Check.Equal(25, r.Grain, "Result.Grain");
                Check.Equal(TileMode.Two, r.TileRows, "Result.TileRows");
                Check.Equal(ParallelMode.Off, r.ParallelB, "Result.ParallelB");
                Check.Equal(256, r.AudioKbps, "Result.AudioKbps");
                Check.That(r.HeAacV2, "Result.HeAacV2");
                Check.Equal(48, r.HeAacKbps, "Result.HeAacKbps");
                Check.Equal(2, r.Workers, "Result.Workers");
                Check.Equal(99, r.QIndex, "Result.QIndex");
                Check.Equal(60, r.KeyInterval, "Result.KeyInterval");
                Check.Equal(5, r.Refs, "Result.Refs");
                Check.Equal(4, r.BFrames, "Result.BFrames");
                Check.That(!r.KeepUnderSource, "Result.KeepUnderSource");
                Check.Equal(ContainerKind.Separate, r.Container, "Result.Container");
                Check.Equal(150, input.QIndex, "input QIndex untouched");
                Check.Equal(0, input.KeyInterval, "input KeyInterval untouched");
                Check.Equal(ContainerKind.Mp4, input.Container, "input Container untouched");
                Check.That(!ReferenceEquals(input, r), "Result is a copy");
            }
        }

        static void AdvancedDialogRefusesBuiltInName()
        {
            string dir = TempDir();
            try
            {
                var store = new PresetStore(dir);
                using (var d = new AdvancedDialog(new EncodeSettings(), store))
                {
                    var warnings = new System.Collections.Generic.List<string>();
                    d.ShowWarning = warnings.Add;
                    Check.That(!d.TrySavePreset("標準"), "built-in name refused");
                    Check.That(!d.TrySavePreset("  "), "empty name refused");
                    Check.Equal(2, warnings.Count, "both refusals explained");
                    Check.Equal(3, store.LoadAll().Count, "nothing saved");
                    Check.That(d.TrySavePreset("mine"), "user name accepted");
                    Check.Equal("mine", d.SavedPresetName, "SavedPresetName");
                    Check.Equal(4, store.LoadAll().Count, "saved");
                }
            }
            finally { Directory.Delete(dir, true); }
        }

        static void MainFormBuilds()
        {
            string dir = TempDir();
            try
            {
                using (var f = new MainForm(new AppSettings(), Path.Combine(dir, "settings.json")))
                {
                    var names = f.PresetNames;
                    Check.That(names.Count >= 3, "at least the built-ins");
                    Check.Equal("高画質", names[0], "preset 0");
                    Check.Equal("標準", names[1], "preset 1");
                    Check.Equal("小容量", names[2], "preset 2");
                    Check.Equal("標準", f.SelectedPresetName, "default selection");
                    Check.That(!f.Controls.Find("outputNameBox", true).Single().Enabled, "name box off with nothing selected");
                }
            }
            finally { Directory.Delete(dir, true); }
        }

        // 速度「とても遅い」 saved on: after a restart the encode settings have it too, not only the box.
        static void MainFormRestoresMaxEffort()
        {
            string dir = TempDir();
            try
            {
                using (var f = new MainForm(new AppSettings { LastMaxEffort = true }, Path.Combine(dir, "settings.json")))
                {
                    var speed = (ComboBox)f.Controls.Find("speedBox", true).Single();
                    Check.Equal(1, speed.SelectedIndex, "とても遅い chosen");
                    Check.That(f.EncodeSettingsForTest.MaxEffort, "settings say --preset max");
                    speed.SelectedIndex = 0;
                    Check.That(!f.EncodeSettingsForTest.MaxEffort, "標準: no --preset max");
                }
            }
            finally { Directory.Delete(dir, true); }
        }

        // サイズ指定 on: the note says the preset's qindex gives way to the size.
        static void MainFormNotesTargetSize()
        {
            string dir = TempDir();
            try
            {
                using (var f = new MainForm(new AppSettings(), Path.Combine(dir, "settings.json")))
                {
                    var note = f.Controls.Find("targetSizeNote", true).Single();
                    Check.Equal("", note.Text, "empty with サイズ指定 off");
                    ((NumericUpDown)f.Controls.Find("targetSize", true).Single()).Value = 1.4m;
                    Check.That(note.Text.Contains("qindex"), "note: " + note.Text);
                }
            }
            finally { Directory.Delete(dir, true); }
        }

        // A dropped .ivf is not probed: its row waits to be put into an MP4 with the audio beside it.
        static void MainFormTakesIvfForMuxing()
        {
            string dir = TempDir();
            try
            {
                string ivf = Path.Combine(dir, "clip.ivf");
                File.WriteAllBytes(ivf, new byte[1]);
                File.WriteAllBytes(Path.Combine(dir, "clip.m4a"), new byte[1]);
                using (var f = new MainForm(new AppSettings(), Path.Combine(dir, "settings.json")))
                {
                    f.AddFiles(new[] { ivf });
                    Check.Equal(0, f.PendingProbes, "no probe");
                    var row = ((ListView)f.Controls.Find("list", true).Single()).Items.Cast<ListViewItem>().Single();
                    Check.Equal("待機", row.SubItems[6].Text, "status");
                    Check.Equal("clip.m4a", row.SubItems[4].Text, "audio column");
                    Check.That(row.SubItems[5].Text.Contains("MP4 にまとめる"), "note: " + row.SubItems[5].Text);
                    Check.That(f.Controls.Find("startButton", true).Single().Enabled, "start enabled");
                }
            }
            finally { Directory.Delete(dir, true); }
        }

        // 画質指定（VMAF）: its value is in the encode settings, also after the preset changes; it and サイズ指定
        // exclude each other (setting one clears the other); the note says how the qindex is chosen.
        static void MainFormTargetVmaf()
        {
            string dir = TempDir();
            try
            {
                using (var f = new MainForm(new AppSettings(), Path.Combine(dir, "settings.json")))
                {
                    var box = (NumericUpDown)f.Controls.Find("targetVmaf", true).Single();
                    var size = (NumericUpDown)f.Controls.Find("targetSize", true).Single();
                    var note = f.Controls.Find("targetSizeNote", true).Single();
                    Check.Equal(0m, box.Value, "off at first");
                    Check.Equal(100m, box.Maximum, "maximum");
                    box.Value = 93m;
                    Check.Equal(93.0, f.EncodeSettingsForTest.TargetVmaf, "93");
                    Check.That(note.Text.Contains("VMAF"), "note: " + note.Text);
                    ((ComboBox)f.Controls.Find("presetBox", true).Single()).SelectedIndex = 0;
                    Check.Equal(93.0, f.EncodeSettingsForTest.TargetVmaf, "kept when the preset changes");
                    size.Value = 5m;
                    Check.Equal(0m, box.Value, "サイズ指定 clears it");
                    Check.Equal(0.0, f.EncodeSettingsForTest.TargetVmaf, "off");
                    box.Value = 95m;
                    Check.Equal(0m, size.Value, "it clears サイズ指定");
                    Check.Equal(0.0, f.EncodeSettingsForTest.TargetSizeMB, "size off");
                }
            }
            finally { Directory.Delete(dir, true); }
        }

        // サイズ指定（MB）: its value is in the encode settings, also after the 画質 preset changes; 0 is off.
        static void MainFormTargetSize()
        {
            string dir = TempDir();
            try
            {
                using (var f = new MainForm(new AppSettings(), Path.Combine(dir, "settings.json")))
                {
                    var box = (NumericUpDown)f.Controls.Find("targetSize", true).Single();
                    Check.Equal(0m, box.Value, "off at first");
                    Check.Equal(1, box.DecimalPlaces, "tenths of a MB");
                    Check.Equal(100000m, box.Maximum, "maximum");
                    box.Value = 9.5m;
                    Check.Equal(9.5, f.EncodeSettingsForTest.TargetSizeMB, "9.5 MB");
                    ((ComboBox)f.Controls.Find("presetBox", true).Single()).SelectedIndex = 0;
                    Check.Equal(9.5, f.EncodeSettingsForTest.TargetSizeMB, "kept when the preset changes");
                    box.Value = 0;
                    Check.Equal(0.0, f.EncodeSettingsForTest.TargetSizeMB, "0 is off");
                }
            }
            finally { Directory.Delete(dir, true); }
        }

        static void NotesForInputs()
        {
            var ten = new OutputFormat();
            var eight = new OutputFormat { BitDepth = 8 };
            Check.Equal("", MainForm.NotesFor(new InputInfo { BitDepth = 8 }, ten, null), "plain 8-bit has no notes");
            Check.Equal("", MainForm.NotesFor(new InputInfo { BitDepth = 10 }, ten, null), "10-bit kept");
            Check.Equal("10bit→8bit", MainForm.NotesFor(new InputInfo { BitDepth = 10 }, eight, null), "10-bit coded at 8");
            Check.Equal("12bit→10bit", MainForm.NotesFor(new InputInfo { BitDepth = 12 }, ten, null), "12-bit");
            Check.Equal("", MainForm.NotesFor(new InputInfo { BitDepth = 10, Hdr = true }, ten, null), "HDR kept");
            Check.Equal("10bit→8bit、HDR→SDR（トーンマップ）", MainForm.NotesFor(new InputInfo { BitDepth = 10, Hdr = true }, eight, null), "HDR at 8 bits");
            Check.Equal("可変fps→固定", MainForm.NotesFor(new InputInfo { BitDepth = 8, VariableFrameRate = true }, ten, null), "VFR");
            Check.Equal("インターレース解除", MainForm.NotesFor(new InputInfo { BitDepth = 8, Interlaced = true }, ten, null), "interlaced");
            Check.Equal("可変fps→固定、別ファイルで保存", MainForm.NotesFor(new InputInfo { BitDepth = 8, VariableFrameRate = true }, ten, "別ファイルで保存"), "job note appended");
            Check.Equal("別ファイルで保存", MainForm.NotesFor(new InputInfo { BitDepth = 8 }, ten, "別ファイルで保存"), "job note alone");
        }

        static void MuxWarningBeforeStart()
        {
            var all = new CelV2.Runner.ToolPaths { Av2Mux = "av2_mux.exe", Mp4Box = "MP4Box.exe" };
            var mp4 = new EncodeSettings { Container = ContainerKind.Mp4 };
            Check.Equal<string>(null, MainForm.MuxWarning(all, mp4), "both tools: no warning");
            string w = MainForm.MuxWarning(new CelV2.Runner.ToolPaths { Av2Mux = "av2_mux.exe" }, mp4);
            Check.That(w != null && w.Contains("MP4Box") && !w.Contains("av2_mux") && w.Contains("別ファイル"), "MP4Box missing: " + w);
            w = MainForm.MuxWarning(new CelV2.Runner.ToolPaths { Mp4Box = "MP4Box.exe" }, mp4);
            Check.That(w != null && w.Contains("av2_mux") && !w.Contains("MP4Box が"), "av2_mux missing: " + w);
            Check.Equal<string>(null, MainForm.MuxWarning(new CelV2.Runner.ToolPaths(), new EncodeSettings { Container = ContainerKind.Separate }),
                "separate files need no muxer");
        }

        static void FailureReasonIsShort()
        {
            string path = Path.Combine(Path.GetTempPath(), "some folder", "bad.txt");
            Check.Equal("Invalid data found when processing input",
                MainForm.ShortReason(path + ": Invalid data found when processing input", path), "ffprobe message loses the path");
            string r = MainForm.ShortReason("ffmpeg が失敗しました (exit -2): Error opening input file " + path + ".\nNo such file", path);
            Check.That(!r.Contains(Path.GetTempPath()) && !r.Contains("\n"), "no path or newline left: " + r);
            Check.Equal("", MainForm.ShortReason(null, path), "null reason");
        }

        static void AdvancedDialogRobust()
        {
            string dir = TempDir();
            try
            {
                var bad = new EncodeSettings { Tiles = (TileMode)9, Device = (DecisionDevice)(-1), Container = (ContainerKind)7 };
                using (var d = new AdvancedDialog(bad))
                {
                    var r = d.Result;
                    Check.Equal(TileMode.Auto, r.Tiles, "tiles clamped");
                    Check.Equal(DecisionDevice.Auto, r.Device, "device clamped");
                    Check.Equal(ContainerKind.Mp4, r.Container, "container clamped");
                }
                var listed = DeviceList.Parse("vulkan\tA (Vulkan)\nvulkan2\tB (Vulkan)\ncpu\tCPU\n");
                using (var d = new AdvancedDialog(new EncodeSettings { DeviceArg = "vulkan2" }, null, listed))
                    Check.Equal("vulkan2", d.Result.DeviceArg, "listed device kept");
                using (var d = new AdvancedDialog(new EncodeSettings { DeviceArg = "vulkan2" }))
                {
                    Check.Equal("vulkan2", d.Result.DeviceArg, "a device not listed stays chosen");
                    Check.Equal(DecisionDevice.Gpu, d.Result.Device, "older versions read a GPU");
                }
                var store = new PresetStore(dir);
                using (var d = new AdvancedDialog(new EncodeSettings(), store))
                {
                    d.ShowWarning = m => { };
                    Check.That(d.TrySavePreset("Mine"), "saved");
                    d.ConfirmOverwrite = n => false;
                    Check.That(!d.TrySavePreset("mine"), "overwrite compared ignoring case and declined");
                    Check.That(!d.TrySavePreset("標準"), "built-in refused");
                }
            }
            finally { Directory.Delete(dir, true); }
        }
    }
}
