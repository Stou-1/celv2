using System;
using System.IO;
using CelV2.Runner;
using CelV2.Settings;

namespace CelV2.Tests
{
    public static class QualityTests
    {
        public static readonly (string name, Action run)[] All =
        {
            ("quality_probe_finds_the_size_cap", FindsSizeCap),
            ("quality_probe_keeps_a_small_enough_qindex", KeepsStart),
            ("quality_probe_kbps_leaves_out_sample_starts", KbpsWithoutStarts),
            ("probe_reads_video_bit_rate", ReadsBitRate),
            ("keep_under_source_defaults_on", DefaultsOn),
        };

        // A bitrate halving every 20 qindex from 20000 kbps at 45: 1000 kbps is first met at
        // 45 + 20 * log2(20) = 131.4, so 132; the search lands within 2 of it in at most 7 trials.
        static void FindsSizeCap()
        {
            int trials = 0;
            Func<int, double> kbps = qi => { trials++; return 20000 * Math.Pow(2, -(qi - 45) / 20.0); };
            int q = QualityProbe.Choose(45, 1000, kbps);
            Check.That(trials <= 7, "trials: " + trials);
            Check.That(kbps(q) <= 1000, "chosen qindex is within the cap: " + q);
            Check.That(q >= 132 && q <= 134, "chosen qindex near the cap: " + q);
        }

        static void KeepsStart()
        {
            Check.Equal(150, QualityProbe.Choose(150, 1000, q => 900), "already under the cap");
            Check.Equal(255, QualityProbe.Choose(45, 10, q => 1000), "never under the cap: the coarsest");
        }

        // Two samples of 3 frames: sizes 1000 (keyframe), 100, 100, 1000, 100, 100 at 30 fps: 400 bytes in
        // 4 frames = 4 * 8 * 100 / (4 / 30) bits per second.
        static void KbpsWithoutStarts()
        {
            string path = Path.Combine(Path.GetTempPath(), "celv2_quality_test.ivf");
            using (var w = new BinaryWriter(File.Create(path)))
            {
                w.Write(new byte[32]);
                foreach (int size in new[] { 1000, 100, 100, 1000, 100, 100 })
                {
                    w.Write((uint)size);
                    w.Write(0UL);
                    w.Write(new byte[size]);
                }
            }
            try
            {
                double k = QualityProbe.KbpsWithoutSampleStarts(path, 3, 30, 1);
                Check.That(Math.Abs(k - 400 * 8 / 1000.0 / (4 / 30.0)) < 1e-9, "kbps " + k);
            }
            finally { File.Delete(path); }
        }

        static void ReadsBitRate()
        {
            const string stream = "{\"streams\":[{\"index\":0,\"codec_type\":\"video\",\"width\":1920,\"height\":1080,\"r_frame_rate\":\"30/1\",\"avg_frame_rate\":\"30/1\",\"pix_fmt\":\"yuv420p\",\"bit_rate\":\"1145276\"}],\"format\":{\"duration\":\"10\",\"bit_rate\":\"1300000\"}}";
            Check.Equal(1145276L, ProbeParser.Parse(stream).VideoBitRate, "stream bit_rate");
            const string mkv = "{\"streams\":[{\"index\":0,\"codec_type\":\"video\",\"width\":1920,\"height\":1080,\"r_frame_rate\":\"30/1\",\"avg_frame_rate\":\"30/1\",\"pix_fmt\":\"yuv420p\"},{\"index\":1,\"codec_type\":\"audio\",\"codec_name\":\"aac\",\"bit_rate\":\"128000\"}],\"format\":{\"duration\":\"10\",\"bit_rate\":\"1300000\"}}";
            Check.Equal(1172000L, ProbeParser.Parse(mkv).VideoBitRate, "container less audio");
            const string none = "{\"streams\":[{\"index\":0,\"codec_type\":\"video\",\"width\":64,\"height\":48,\"r_frame_rate\":\"30/1\",\"avg_frame_rate\":\"30/1\",\"pix_fmt\":\"yuv420p\"}],\"format\":{\"duration\":\"1\"}}";
            Check.Equal(0L, ProbeParser.Parse(none).VideoBitRate, "unknown");
        }

        static void DefaultsOn()
        {
            Check.That(new EncodeSettings().KeepUnderSource, "default on");
            foreach (var p in PresetStore.BuiltIn) Check.That(p.Settings.KeepUnderSource, "preset " + p.Name);
        }
    }
}
