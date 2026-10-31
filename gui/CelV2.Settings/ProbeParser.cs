#pragma warning disable 0649 // DataContract fields are filled by the serializer
using System;
using System.Globalization;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;
using System.Text.RegularExpressions;

namespace CelV2.Settings
{
    public static class ProbeParser
    {
        [DataContract] class ProbeDisposition { [DataMember(Name = "attached_pic")] public int AttachedPic; }

        [DataContract] class ProbeStream
        {
            [DataMember(Name = "index")] public int Index;
            [DataMember(Name = "codec_type")] public string CodecType;
            [DataMember(Name = "codec_name")] public string CodecName;
            [DataMember(Name = "width")] public int Width;
            [DataMember(Name = "height")] public int Height;
            [DataMember(Name = "pix_fmt")] public string PixFmt;
            [DataMember(Name = "r_frame_rate")] public string RFrameRate;
            [DataMember(Name = "avg_frame_rate")] public string AvgFrameRate;
            [DataMember(Name = "field_order")] public string FieldOrder;
            [DataMember(Name = "color_primaries")] public string ColorPrimaries;
            [DataMember(Name = "color_transfer")] public string ColorTransfer;
            [DataMember(Name = "color_space")] public string ColorSpace;
            [DataMember(Name = "color_range")] public string ColorRange;
            [DataMember(Name = "nb_frames")] public string NbFrames;               // ffprobe prints numbers as strings
            [DataMember(Name = "duration")] public string Duration;
            [DataMember(Name = "bits_per_raw_sample")] public string BitsPerRawSample;
            [DataMember(Name = "bit_rate")] public string BitRate;
            [DataMember(Name = "disposition")] public ProbeDisposition Disposition;
        }

        [DataContract] class ProbeFormat
        {
            [DataMember(Name = "duration")] public string Duration;
            [DataMember(Name = "bit_rate")] public string BitRate;
        }

        [DataContract] class ProbeResult
        {
            [DataMember(Name = "streams")] public ProbeStream[] Streams;
            [DataMember(Name = "format")] public ProbeFormat Format;
        }

        // One side data entry of a frame; only the HDR10 ones are read. Rationals are strings ("34000/50000").
        [DataContract] class ProbeSideData
        {
            [DataMember(Name = "side_data_type")] public string Type;
            [DataMember(Name = "red_x")] public string RedX;
            [DataMember(Name = "red_y")] public string RedY;
            [DataMember(Name = "green_x")] public string GreenX;
            [DataMember(Name = "green_y")] public string GreenY;
            [DataMember(Name = "blue_x")] public string BlueX;
            [DataMember(Name = "blue_y")] public string BlueY;
            [DataMember(Name = "white_point_x")] public string WhiteX;
            [DataMember(Name = "white_point_y")] public string WhiteY;
            [DataMember(Name = "min_luminance")] public string MinLuminance;
            [DataMember(Name = "max_luminance")] public string MaxLuminance;
            [DataMember(Name = "max_content")] public int? MaxContent;
            [DataMember(Name = "max_average")] public int? MaxAverage;
        }

        [DataContract] class ProbeFrame { [DataMember(Name = "side_data_list")] public ProbeSideData[] SideData; }
        [DataContract] class ProbeFrames { [DataMember(Name = "frames")] public ProbeFrame[] Frames; }

        public static string[] ProbeArgs(string input)
        {
            return new[] { "-v", "error", "-print_format", "json", "-show_streams", "-show_format", input };
        }

        // The first frame of the video stream, for its HDR10 side data.
        public static string[] HdrMetadataArgs(string input, int streamIndex)
        {
            return new[] { "-v", "error", "-print_format", "json", "-select_streams", streamIndex.ToString(CultureInfo.InvariantCulture),
                           "-read_intervals", "%+#1", "-show_frames", "-show_entries", "frame=side_data_list", input };
        }

        // Fills i.MasteringDisplay / i.ContentLight from HdrMetadataArgs' output; values celv2enc would refuse
        // (or unreadable output) leave them null, so the video is still coded as HDR, only without that metadata.
        public static void ParseHdrMetadata(string ffprobeJson, InputInfo i)
        {
            ProbeFrames r;
            try
            {
                var ser = new DataContractJsonSerializer(typeof(ProbeFrames));
                using (var ms = new MemoryStream(Encoding.UTF8.GetBytes(ffprobeJson ?? "")))
                    r = (ProbeFrames)ser.ReadObject(ms);
            }
            catch (SerializationException) { return; }
            catch (ArgumentException) { return; }
            if (r == null || r.Frames == null || r.Frames.Length == 0 || r.Frames[0] == null || r.Frames[0].SideData == null) return;
            foreach (var d in r.Frames[0].SideData)
            {
                if (d == null) continue;
                if (d.Type == "Mastering display metadata") i.MasteringDisplay = Mdcv(d);
                else if (d.Type == "Content light level metadata" && d.MaxContent is int cll && d.MaxAverage is int fall
                         && cll >= 0 && fall >= 0 && cll <= 65535 && fall <= 65535)
                    i.ContentLight = cll.ToString(CultureInfo.InvariantCulture) + "," + fall.ToString(CultureInfo.InvariantCulture);
            }
        }

        // x265's --master-display form: chromaticity in 0.00002, luminance in 0.0001 cd/m2; null when incomplete or out of range.
        static string Mdcv(ProbeSideData d)
        {
            long[] c = new long[8];
            string[] xy = { d.GreenX, d.GreenY, d.BlueX, d.BlueY, d.RedX, d.RedY, d.WhiteX, d.WhiteY };
            for (int k = 0; k < 8; k++)
                if (!Units(xy[k], 50000, out c[k]) || c[k] > 50000) return null;
            long lmax, lmin;
            if (!Units(d.MaxLuminance, 10000, out lmax) || !Units(d.MinLuminance, 10000, out lmin) || lmax <= lmin || lmax > 100000000) return null;
            Func<long, string> s = v => v.ToString(CultureInfo.InvariantCulture);
            return "G(" + s(c[0]) + "," + s(c[1]) + ")B(" + s(c[2]) + "," + s(c[3]) + ")R(" + s(c[4]) + "," + s(c[5]) +
                   ")WP(" + s(c[6]) + "," + s(c[7]) + ")L(" + s(lmax) + "," + s(lmin) + ")";
        }

        // "num/den" in units of 1/scale, rounded; false for anything else or a negative value.
        static bool Units(string rational, long scale, out long units)
        {
            units = 0;
            var p = (rational ?? "").Split('/');
            long num, den;
            if (p.Length != 2 || !long.TryParse(p[0], NumberStyles.None, CultureInfo.InvariantCulture, out num)
                || !long.TryParse(p[1], NumberStyles.None, CultureInfo.InvariantCulture, out den) || den <= 0 || num > long.MaxValue / scale) return false;
            units = (num * scale + den / 2) / den;
            return true;
        }

        public static InputInfo Parse(string ffprobeJson)
        {
            ProbeResult r;
            try
            {
                var ser = new DataContractJsonSerializer(typeof(ProbeResult));
                using (var ms = new MemoryStream(Encoding.UTF8.GetBytes(ffprobeJson ?? "")))
                    r = (ProbeResult)ser.ReadObject(ms);
            }
            catch (SerializationException e) { throw new FormatException("ffprobe output is not valid JSON: " + e.Message); }
            catch (ArgumentException e) { throw new FormatException("ffprobe output is not valid JSON: " + e.Message); }

            ProbeStream v = null, a = null;
            if (r != null && r.Streams != null)
                foreach (var s in r.Streams)
                {
                    if (s == null) continue;
                    bool cover = s.Disposition != null && s.Disposition.AttachedPic == 1; // album art, a single picture
                    if (v == null && s.CodecType == "video" && !cover) v = s;
                    else if (a == null && s.CodecType == "audio") a = s;
                }
            if (v == null) throw new FormatException("映像がありません");

            var i = new InputInfo { VideoStreamIndex = v.Index, Width = v.Width, Height = v.Height, PixelFormat = v.PixFmt };
            int rn, rd, an, ad;
            bool rOk = TryRate(v.RFrameRate, out rn, out rd), aOk = TryRate(v.AvgFrameRate, out an, out ad);
            if (!rOk && !aOk) throw new FormatException("the video stream has no frame rate");
            if (aOk) { i.FpsNum = an; i.FpsDen = ad; } else { i.FpsNum = rn; i.FpsDen = rd; }
            // shortcut: 1 % tolerance so 30000/1001 vs 2997/100 is not "variable"; tighten if a real VFR file slips through.
            i.VariableFrameRate = rOk && aOk && Math.Abs((double)rn / rd / ((double)an / ad) - 1) > 0.01;

            string fo = v.FieldOrder;
            i.Interlaced = !string.IsNullOrEmpty(fo) && fo != "progressive" && fo != "unknown";
            i.BitDepth = DepthOf(v);
            i.ColorPrimaries = Blank(v.ColorPrimaries);
            i.ColorTransfer = Blank(v.ColorTransfer);
            i.ColorSpace = Blank(v.ColorSpace);
            i.ColorRange = Blank(v.ColorRange);
            i.Hdr = i.ColorTransfer == "smpte2084" || i.ColorTransfer == "arib-std-b67";
            i.AudioCodec = a == null ? null : Blank(a.CodecName);

            double dur;
            if (!TryDouble(v.Duration, out dur) && !TryDouble(r.Format == null ? null : r.Format.Duration, out dur)) dur = 0;
            i.DurationSeconds = dur;
            long n;
            if (!long.TryParse(v.NbFrames, NumberStyles.None, CultureInfo.InvariantCulture, out n) || n <= 0)
                n = (long)Math.Round(dur * i.FpsNum / i.FpsDen, MidpointRounding.AwayFromZero);
            i.FrameCount = n;
            // The video stream's bit rate, else the container's less the audio's (MKV has no stream rates).
            double vb, fb, ab;
            if (TryDouble(v.BitRate, out vb) && vb > 0) i.VideoBitRate = (long)vb;
            else if (TryDouble(r.Format == null ? null : r.Format.BitRate, out fb) && fb > 0)
                i.VideoBitRate = (long)Math.Max(0, fb - (a != null && TryDouble(a.BitRate, out ab) ? ab : 0));
            return i;
        }

        static string Blank(string s) { return string.IsNullOrEmpty(s) || s == "unknown" ? null : s; }

        static bool TryDouble(string s, out double d)
        {
            return double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out d);
        }

        // "24000/1001" -> 24000, 1001; "0/0" and garbage are unusable.
        static bool TryRate(string s, out int num, out int den)
        {
            num = den = 0;
            if (string.IsNullOrEmpty(s)) return false;
            var p = s.Split('/');
            if (p.Length != 2) return false;
            return int.TryParse(p[0], NumberStyles.None, CultureInfo.InvariantCulture, out num)
                && int.TryParse(p[1], NumberStyles.None, CultureInfo.InvariantCulture, out den) && num > 0 && den > 0;
        }

        static int DepthOf(ProbeStream v)
        {
            var m = Regex.Match(v.PixFmt ?? "", @"(?:p0?|[a-z])(10|12)(?:le|be)?$");
            if (m.Success) return int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
            int b;
            if (int.TryParse(v.BitsPerRawSample, NumberStyles.None, CultureInfo.InvariantCulture, out b) && (b == 10 || b == 12)) return b;
            return 8;
        }
    }
}
