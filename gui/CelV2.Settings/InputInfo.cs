namespace CelV2.Settings
{
    // What ffprobe says about an input file; filled by ProbeParser.Parse.
    public sealed class InputInfo
    {
        public int VideoStreamIndex;           // ffprobe's stream index of the video picked (cover art is never picked)
        public int Width, Height;
        public int FpsNum, FpsDen;             // avg_frame_rate (r_frame_rate when avg is unusable)
        public bool VariableFrameRate;         // r_frame_rate and avg_frame_rate differ by more than 1 %
        public bool Interlaced;                // field_order is anything but progressive / unknown / missing
        public int BitDepth;                   // 8, 10, 12
        public string PixelFormat, ColorPrimaries, ColorTransfer, ColorSpace, ColorRange; // ffprobe names; null when absent
        public bool Hdr;                       // transfer smpte2084 or arib-std-b67
        public string MasteringDisplay;        // celv2enc --mdcv "G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)"; null when absent (ProbeParser.ParseHdrMetadata)
        public string ContentLight;            // celv2enc --cll "MaxCLL,MaxFALL"; null when absent
        public long FrameCount;                // nb_frames, else round(duration * fps)
        public string AudioCodec;              // null when there is no audio stream
        public double DurationSeconds;
        public long VideoBitRate;              // bits per second of the coded video; 0 when unknown
    }
}
