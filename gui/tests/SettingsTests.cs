using System;
using System.IO;
using System.Linq;
using CelV2.Settings;

namespace CelV2.Tests
{
    public static class SettingsTests
    {
        public static readonly (string name, Action run)[] All =
        {
            ("builtin_presets", BuiltinPresets),
            ("preset_round_trip", PresetRoundTrip),
            ("preset_names_are_file_safe", PresetNamesAreFileSafe),
            ("preset_missing_fields_get_defaults", PresetMissingFieldsGetDefaults),
            ("builtins_are_not_shared", BuiltinsAreNotShared),
            ("preset_same_file_found", PresetSameFileFound),
        };

        // "a/b" is stored as a_b.json, the file of "a_b": saving it must see that preset.
        static void PresetSameFileFound()
        {
            var dir = TempDir();
            try
            {
                var store = new PresetStore(dir);
                store.Save(new Preset { Name = "a_b", Settings = new EncodeSettings { QIndex = 111 } });
                Check.Equal("a_b", store.SameFileAs("a/b")?.Name, "a/b shares a_b's file");
                Check.Equal("a_b", store.SameFileAs("A_B")?.Name, "case ignored");
                Check.Equal(null, store.SameFileAs("other"), "no clash");
                Check.Equal(null, store.SameFileAs("標準"), "built-ins have no file");
            }
            finally { if (Directory.Exists(dir)) Directory.Delete(dir, true); }
        }

        static string TempDir()
        {
            return Path.Combine(Path.GetTempPath(), "celv2gui_test_" + Guid.NewGuid().ToString("N"));
        }

        static void BuiltinPresets()
        {
            var b = PresetStore.BuiltIn;
            Check.Equal(3, b.Count, "count");
            Check.Equal("高画質", b[0].Name, "name0");
            Check.Equal("標準", b[1].Name, "name1");
            Check.Equal("小容量", b[2].Name, "name2");
            int[] q = { 60, 170, 200 };
            for (int i = 0; i < 3; i++)
            {
                var d = new EncodeSettings { QIndex = q[i] };
                if (i == 0) { d.KeyInterval = 240; d.Refs = 7; } // 高画質: a keyframe every 240 frames, 7 references
                else { d.Refs = 7; d.BFrames = -1; }               // 標準 / 小容量: 7 references, celv2enc's B frames
                Check.Equal(i == 0 ? 3 : -1, b[i].Settings.BFrames, "B-frames" + i);
                var s = b[i].Settings;
                Check.Equal(q[i], s.QIndex, "qindex" + i);
                Check.Equal(d.Tiles, s.Tiles, "tiles" + i);
                Check.Equal(d.Device, s.Device, "device" + i);
                Check.Equal(d.KeyInterval, s.KeyInterval, "key" + i);
                Check.Equal(d.Threads, s.Threads, "threads" + i);
                Check.Equal(d.Refs, s.Refs, "refs" + i);
                Check.Equal(d.BFrames, s.BFrames, "bframes" + i);
                Check.Equal(d.Output.BitDepth, s.Output.BitDepth, "bitdepth" + i);
                Check.Equal(d.Container, s.Container, "container" + i);
            }
        }

        static void PresetRoundTrip()
        {
            var dir = TempDir();
            try
            {
                var p = new Preset
                {
                    Name = "テスト 1",
                    Settings = new EncodeSettings
                    {
                        QIndex = 99, Tiles = TileMode.Two, Device = DecisionDevice.Cpu, DeviceArg = "vulkan2",
                        KeyInterval = 30, Threads = 4, Refs = 3, BFrames = 3, Container = ContainerKind.Separate,
                    },
                };
                new PresetStore(dir).Save(p);
                var all = new PresetStore(dir).LoadAll();
                Check.Equal(4, all.Count, "count after save");
                var s = all[3].Settings;
                Check.Equal("テスト 1", all[3].Name, "name");
                Check.Equal(99, s.QIndex, "qindex");
                Check.Equal(TileMode.Two, s.Tiles, "tiles");
                Check.Equal(DecisionDevice.Cpu, s.Device, "device");
                Check.Equal("vulkan2", s.DeviceArg, "device arg");
                Check.Equal(30, s.KeyInterval, "key");
                Check.Equal(4, s.Threads, "threads");
                Check.Equal(3, s.Refs, "refs");
                Check.Equal(3, s.BFrames, "bframes");
                Check.Equal(ContainerKind.Separate, s.Container, "container");
                Check.Equal(10, s.Output.BitDepth, "bitdepth");

                p.Settings.QIndex = 77;
                var store = new PresetStore(dir);
                store.Save(p);
                all = store.LoadAll();
                Check.Equal(4, all.Count, "count after overwrite");
                Check.Equal(77, all[3].Settings.QIndex, "overwritten qindex");

                store.Delete("テスト 1");
                Check.Equal(3, store.LoadAll().Count, "count after delete");
            }
            finally { if (Directory.Exists(dir)) Directory.Delete(dir, true); }
        }

        static void PresetNamesAreFileSafe()
        {
            var root = TempDir();
            var dir = Path.Combine(root, "presets");
            try
            {
                var cases = new[]
                {
                    ("a/b:c*?\"<>|", "a_b_c______.json"),
                    ("..", "_.json"),
                    (".", "_.json"),
                    ("CON", "_CON.json"),
                    ("con.txt", "_con.txt.json"),
                    ("Lpt1", "_Lpt1.json"),
                };
                foreach (var (name, file) in cases)
                {
                    var store = new PresetStore(dir);
                    store.Save(new Preset { Name = name, Settings = new EncodeSettings { QIndex = 55 } });
                    var all = store.LoadAll();
                    Check.Equal(4, all.Count, "count " + name);
                    Check.Equal(name, all[3].Name, "original name kept");
                    Check.Equal(55, all[3].Settings.QIndex, "qindex");
                    Check.Equal(0, Directory.GetDirectories(dir).Length, "no subfolders");
                    Check.Equal(1, Directory.GetFileSystemEntries(root).Length, "nothing outside folder");
                    var files = Directory.GetFiles(dir).Select(Path.GetFileName).ToArray();
                    Check.Equal(1, files.Length, "one file in folder");
                    Check.Equal(file, files[0], "file name for " + name);
                    store.Delete(name);
                    Check.Equal(3, store.LoadAll().Count, "deleted " + name);
                }
            }
            finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        }

        static void PresetMissingFieldsGetDefaults()
        {
            var dir = TempDir();
            try
            {
                Directory.CreateDirectory(dir);
                File.WriteAllText(Path.Combine(dir, "x.json"),
                    "{\"Name\":\"x\",\"Settings\":{\"QIndex\":99}}", new System.Text.UTF8Encoding(false));
                var all = new PresetStore(dir).LoadAll();
                Check.Equal(4, all.Count, "count");
                var s = all[3].Settings;
                Check.Equal(99, s.QIndex, "qindex");
                Check.That(s.Output != null, "Output not null");
                Check.Equal(10, s.Output.BitDepth, "bitdepth");
                Check.Equal(TileMode.Auto, s.Tiles, "tiles");
                Check.Equal(DecisionDevice.Auto, s.Device, "device");
                Check.Equal(ContainerKind.Mp4, s.Container, "container");
                // a preset saved before the options existed codes as celv2enc's defaults
                Check.That(s.SceneCut && s.Cdef && s.BQ == 16 && s.AQ == -16 && s.KQ == 0 && s.DeblockDelta == 2, "encoder option defaults");
                Check.Equal(ParallelMode.Auto, s.ParallelB, "parallel default");
                Check.Equal(TileMode.Auto, s.TileRows, "tile rows default");
                Check.Equal(90, s.SizeCapPercent, "size cap default");
                Check.Equal(192, s.AudioKbps, "audio default");
                Check.That(!s.HeAacV2, "HE-AACv2 off by default");
                Check.Equal(32, s.HeAacKbps, "HE-AACv2 kbps default");
                Check.Equal(0.0, s.TargetSizeMB, "target size off by default");
                Check.Equal(0, s.Workers, "workers auto by default");
                Check.Equal(10, s.Clone().Output.BitDepth, "clone works");
                // nested object present but BitDepth missing
                File.WriteAllText(Path.Combine(dir, "x.json"),
                    "{\"Name\":\"x\",\"Settings\":{\"Output\":{}}}", new System.Text.UTF8Encoding(false));
                var t = new PresetStore(dir).LoadAll()[3].Settings;
                Check.Equal(10, t.Output.BitDepth, "bitdepth default in empty Output");
                Check.Equal(170, t.QIndex, "qindex default");
                File.WriteAllText(Path.Combine(dir, "x.json"),
                    "{\"Name\":\"x\",\"Settings\":{\"Output\":null}}", new System.Text.UTF8Encoding(false));
                Check.Equal(10, new PresetStore(dir).LoadAll()[3].Settings.Output.BitDepth, "explicit null Output");
                // saved before 10-bit coding, when 8 was the only depth: codes at 10 now
                File.WriteAllText(Path.Combine(dir, "x.json"),
                    "{\"Name\":\"x\",\"Settings\":{\"Output\":{\"BitDepth\":8}}}", new System.Text.UTF8Encoding(false));
                Check.Equal(10, new PresetStore(dir).LoadAll()[3].Settings.Output.BitDepth, "old 8-bit preset");
                File.WriteAllText(Path.Combine(dir, "x.json"),
                    "{\"Name\":\"x\",\"Settings\":{\"Output\":{\"CodedBitDepth\":8}}}", new System.Text.UTF8Encoding(false));
                Check.Equal(8, new PresetStore(dir).LoadAll()[3].Settings.Output.BitDepth, "8 chosen after 10-bit coding stays 8");
            }
            finally { if (Directory.Exists(dir)) Directory.Delete(dir, true); }
        }

        static void BuiltinsAreNotShared()
        {
            var dir = TempDir(); // never created: LoadAll yields only built-ins
            PresetStore.BuiltIn[0].Settings.QIndex = 1;
            new PresetStore(dir).LoadAll()[0].Settings.QIndex = 2;
            Check.Equal(60, PresetStore.BuiltIn[0].Settings.QIndex, "BuiltIn untouched");
            Check.Equal(60, new PresetStore(dir).LoadAll()[0].Settings.QIndex, "LoadAll fresh");
        }
    }
}
