using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Json;
using System.Text;

namespace CelV2.Settings
{
    public sealed class PresetStore
    {
        readonly string folder;

        public PresetStore(string folder) { this.folder = folder; }

        public static IReadOnlyList<Preset> BuiltIn =>
            new[] { HighQuality(), Long("標準", 170), Long("小容量", 200) }; // fresh copies each call, callers may mutate

        static Preset Make(string name, int q)
        {
            return new Preset { Name = name, Settings = new EncodeSettings { QIndex = q } };
        }

        // Keeps as much of the source as celv2 can: on an anime test clip qindex 45 was the point where
        // zoomed frames matched the source, and lower values only added bits. Trellis coded quantization
        // moved that quality to qindex 60 (all three presets keep their PSNR from before it: 45/150/180
        // became 60/165/195 on the anime clips and anime A 1080p, then 60/170/200 after the RD mode
        // checks). A keyframe every 240 frames makes the file seekable for +0.1% bits; 7 references
        // save ~2% more bits than celv2enc's default 4 for ~6% of the speed.
        static Preset HighQuality()
        {
            var p = Make("高画質", 60);
            p.Settings.KeyInterval = 240;
            p.Settings.Refs = 7;
            return p;
        }

        // 7 references and celv2enc's B frames: from qindex 100 it codes 8-frame mini-GOPs as a
        // pyramid. 120 frames 1080p q170-230 against 4 references and 3 B frames: anime B
        // -7.3 / -9.9%, live action -9.1%, anime A -9.3%, anime C -1.4%, anime D -0.9%, Xiph
        // crowd_run / park_joy -4.0 / -6.3% BD-rate, anime A q170 8.9 -> 9.2 fps; ducks_take_off (splashing
        // water) +4.5%. 高画質 keeps 3: below qindex 100 the long mini-GOPs lose.
        static Preset Long(string name, int q)
        {
            var p = Make(name, q);
            p.Settings.Refs = 7;
            p.Settings.BFrames = -1;
            return p;
        }

        public IReadOnlyList<Preset> LoadAll()
        {
            var all = new List<Preset>(BuiltIn);
            if (!Directory.Exists(folder)) return all;
            var ser = new DataContractJsonSerializer(typeof(Preset));
            var user = new List<Preset>();
            string[] files;
            try { files = Directory.GetFiles(folder, "*.json"); }
            catch (Exception e) when (e is IOException || e is UnauthorizedAccessException) { return all; }
            foreach (var f in files)
            {
                try
                {
                    using (var s = File.OpenRead(f))
                    {
                        var p = (Preset)ser.ReadObject(s);
                        if (p != null && p.Name != null && p.Settings != null) user.Add(p);
                    }
                }
                catch (Exception e) when (e is IOException || e is UnauthorizedAccessException ||
                                          e is System.Runtime.Serialization.SerializationException)
                {
                    // shortcut: a corrupt or unreadable preset file is skipped silently, surface it in the UI if that matters
                }
            }
            user.Sort((a, b) => string.CompareOrdinal(a.Name, b.Name));
            all.AddRange(user);
            return all;
        }

        public void Save(Preset p)
        {
            Directory.CreateDirectory(folder);
            var ser = new DataContractJsonSerializer(typeof(Preset));
            using (var s = File.Create(PathFor(p.Name)))
            using (var w = JsonReaderWriterFactory.CreateJsonWriter(s, Encoding.UTF8, false, true))
                ser.WriteObject(w, p);
        }

        // The user preset stored in the file `name` would be saved to, or null: names that differ only
        // in characters a file name cannot hold (or in case) share one file.
        public Preset SameFileAs(string name)
        {
            var path = PathFor(name);
            return LoadAll().Skip(BuiltIn.Count)
                .FirstOrDefault(p => string.Equals(PathFor(p.Name), path, StringComparison.OrdinalIgnoreCase));
        }

        public void Delete(string name)
        {
            File.Delete(PathFor(name)); // no-op if absent
        }

        static readonly HashSet<string> reserved = new HashSet<string>(
            new[] { "CON", "PRN", "AUX", "NUL" }
                .Concat(Enumerable.Range(1, 9).SelectMany(i => new[] { "COM" + i, "LPT" + i })));

        string PathFor(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("preset name is empty");
            var bad = Path.GetInvalidFileNameChars().Concat(new[] { '/', '\\', ':' }).ToArray();
            var safe = new string(name.Select(c => bad.Contains(c) ? '_' : c).ToArray()).TrimEnd('.', ' ');
            // Names that differ only in invalid characters share a file (so do "CON" and "_CON", "." and "_"):
            // saving asks before replacing the other one (SameFileAs).
            var stem = safe.Split('.')[0].ToUpperInvariant();
            if (safe.Length == 0 || reserved.Contains(stem)) safe = "_" + safe;
            return Path.Combine(folder, safe + ".json");
        }
    }
}
