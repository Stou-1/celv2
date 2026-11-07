using System;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using CelV2.Runner;
using CelV2.Ui;

namespace CelV2.Tests
{
    // The encode details: celv2enc's progress lines, their sums, and the window that shows them.
    public static class DetailsTests
    {
        public static readonly (string name, Action run)[] All =
        {
            ("progress_line_reads_type_qindex_bytes", ProgressLineReadsFields),
            ("progress_line_reads_the_old_form", ProgressLineReadsOldForm),
            ("encode_details_sums_and_projects", DetailsSumsAndProjects),
            ("encode_details_new_pass_restarts_the_list", DetailsNewPass),
            ("encode_details_keeps_the_last_log_lines", DetailsKeepsLog),
            ("details_window_shows_the_frames", DetailsWindowShowsFrames),
        };

        static void ProgressLineReadsFields()
        {
            var f = ProgressLine.Parse("progress frame=12 type=B qindex=142 bytes=3810");
            Check.That(f != null, "parsed");
            Check.Equal(12L, f.Frame, "frame");
            Check.Equal('B', f.Type, "type");
            Check.Equal(142, f.Qindex, "qindex");
            Check.Equal(3810L, f.Bytes, "bytes");
            Check.That(ProgressLine.Parse("celv2: decisions on GPU") == null, "other lines are not progress");
            Check.That(ProgressLine.Parse("progress frame=x") == null, "a bad number is not progress");
        }

        // celv2enc before the fields: the frame alone, the rest unknown.
        static void ProgressLineReadsOldForm()
        {
            var f = ProgressLine.Parse("progress frame=7");
            Check.That(f != null, "parsed");
            Check.Equal(7L, f.Frame, "frame");
            Check.Equal('?', f.Type, "type");
            Check.Equal(-1L, f.Bytes, "bytes unknown");
        }

        static void DetailsSumsAndProjects()
        {
            var d = new EncodeDetails();
            d.Begin("a.mp4", 100, 30, 1);
            d.AddFrame(new FrameInfo { Frame = 1, Type = 'K', Qindex = 120, Bytes = 30000 });
            d.AddFrame(new FrameInfo { Frame = 2, Type = 'B', Qindex = 140, Bytes = 6000 });
            d.AddFrame(new FrameInfo { Frame = 3, Type = 'A', Qindex = 130, Bytes = 9000 });
            var s = d.Snapshot();
            Check.Equal(3, s.Frames.Count, "frames");
            Check.Equal(45000L, s.Bytes, "bytes so far");
            // 45000 bytes over 3 frames at 30 fps = 0.1 s: 3600 kbps
            Check.That(Math.Abs(s.Kbps - 3600) < 1e-6, "kbps " + s.Kbps);
            Check.Equal(1500000L, s.ProjectedBytes, "45000 / 3 x 100 frames");
            Check.Equal("a.mp4", s.Name, "name");
        }

        // Target size codes the video twice: frame 1 again starts the list over as the next pass.
        static void DetailsNewPass()
        {
            var d = new EncodeDetails();
            d.Begin("a.mp4", 0, 24, 1);
            d.AddFrame(new FrameInfo { Frame = 1, Type = 'K', Bytes = 100 });
            d.AddFrame(new FrameInfo { Frame = 2, Type = 'B', Bytes = 50 });
            d.AddFrame(new FrameInfo { Frame = 1, Type = 'K', Bytes = 80 });
            var s = d.Snapshot();
            Check.Equal(1, s.Frames.Count, "the second pass's frames only");
            Check.Equal(80L, s.Bytes, "its bytes");
            Check.Equal(2, s.Pass, "pass 2");
            Check.Equal(-1L, s.ProjectedBytes, "no projection without a frame count");
        }

        static void DetailsKeepsLog()
        {
            var d = new EncodeDetails();
            d.Begin("a.mp4", 0, 30, 1);
            for (int i = 0; i < EncodeDetails.MaxLogLines + 10; i++) d.AddLog("line " + i);
            var s = d.Snapshot();
            Check.Equal(EncodeDetails.MaxLogLines, s.Log.Count, "capped");
            Check.Equal("line " + (EncodeDetails.MaxLogLines + 9), s.Log.Last(), "newest kept");
            Check.Equal(10L + EncodeDetails.MaxLogLines, s.LogLinesSeen, "all lines counted");
        }

        static void DetailsWindowShowsFrames()
        {
            var d = new EncodeDetails();
            d.Begin("a.mp4", 10, 30, 1);
            d.SetStage("符号化");
            d.SetEngine("celv2: decisions on GPU (NVIDIA)");
            d.AddFrame(new FrameInfo { Frame = 1, Type = 'K', Qindex = 120, Bytes = 2048 });
            d.AddLog("celv2enc -i -");
            using (var f = new DetailsForm(d))
            {
                f.RefreshNow();
                var list = (ListView)f.Controls.Find("frameList", true).Single();
                Check.Equal(1, list.VirtualListSize, "one row");
                var row = f.Row(0);
                Check.Equal("1", row.Text, "frame number");
                Check.Equal("I（キー）", row.SubItems[1].Text, "type");
                Check.Equal("P", DetailsForm.TypeName('P'), "lone frame");
                Check.Equal("P（アンカー）", DetailsForm.TypeName('A'), "anchor");
                Check.Equal("120", row.SubItems[2].Text, "qindex");
                Check.Equal("2.0", row.SubItems[3].Text, "KB");
                var log = (TextBox)f.Controls.Find("logBox", true).Single();
                Check.That(log.Text.Contains("celv2enc -i -"), "log shown");
                var status = (Label)f.Controls.Find("statusLabel", true).Single();
                Check.That(status.Text.Contains("符号化") && status.Text.Contains("GPU"), "status: " + status.Text);
            }
        }
    }
}
