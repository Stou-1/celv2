using System.Runtime.Serialization;

namespace CelV2.Settings
{
    public enum DecisionDevice { Auto, Gpu, Cpu }
    public enum TileMode { Auto, One, Two, Four }          // One/Two/Four: --tile-columns 0/1/2
    public enum ContainerKind { Mp4, Separate }            // Mp4 = av2_mux + MP4Box; Separate = IVF + audio file
    public enum ParallelMode { Auto, On, Off }             // celv2enc --parallel-b auto / 1 / 0
    public enum TuneKind { None, Animation, Film, Still }   // celv2enc --tune: none / animation / film / still

    [DataContract]
    public sealed class OutputFormat                        // the coded depth: 10 (celv2enc's default) or 8
    {
        // Saved as "CodedBitDepth": files from before 10-bit coding hold "BitDepth": 8, the only value then, and load as 10.
        [DataMember(Name = "CodedBitDepth")] public int BitDepth;

        public OutputFormat() { SetDefaults(); }
        [OnDeserializing] void OnDeserializing(StreamingContext c) { SetDefaults(); } // the serializer skips constructors
        void SetDefaults() { BitDepth = 10; }
    }

    [DataContract]
    public sealed class EncodeSettings
    {
        [DataMember] public int QIndex;                     // 1..255
        [DataMember] public TileMode Tiles;
        [DataMember] public DecisionDevice Device;          // older settings; DeviceArg wins when set
        [DataMember] public string DeviceArg;               // celv2enc --device value from --list-devices ("gpu", "vulkan2", ...)
        [DataMember] public int KeyInterval;                // 0 = only the first frame
        [DataMember] public int Threads;                    // 0 = celv2enc default
        [DataMember] public int Refs;                       // past frames to predict from, 1..7; 0 = celv2enc default (4)
        [DataMember] public int BFrames;                    // frames coded after the next one, 0..15; 0 = none (default 3); -1 = celv2enc default (7 with 5+ references at qindex 100+, else 3)
        [DataMember] public bool KeepUnderSource;           // never more video bits than an already compressed source (QualityProbe)
        [DataMember] public int SizeCapPercent;             // KeepUnderSource's limit, % of the source's video bitrate (10..100)
        [DataMember] public OutputFormat Output;
        [DataMember] public ContainerKind Container;
        // The rest map to celv2enc options; the defaults are celv2enc's, so they are passed only when changed.
        [DataMember] public TileMode TileRows;              // One/Two/Four: --tile-rows 0/1/2
        [DataMember] public bool SceneCut;                  // --scenecut
        [DataMember] public int BQ;                         // --bq: qindex offset of B frames
        [DataMember] public int AQ;                         // --aq: qindex offset of hidden (ALT) frames
        [DataMember] public int KQ;                         // --kq: qindex offset of keyframes
        [DataMember] public int DeblockDelta;               // --deblock-delta, -16..15
        [DataMember] public bool Cdef;                      // --cdef
        [DataMember] public ParallelMode ParallelB;         // --parallel-b
        [DataMember] public int AudioKbps;                  // AAC bitrate when the audio is re-encoded
        [DataMember] public bool HeAacV2;                   // always re-encode the audio to stereo HE-AACv2 (fdkaac)
        [DataMember] public int HeAacKbps;                  // its bitrate, 12..96
        [DataMember] public int Workers;                    // celv2enc --workers: scene chunks coded at once; 0 = auto
        [DataMember] public double TargetSizeMB;            // the whole output at most this (1 MB = 1,000,000 bytes), 2 passes; 0 = off
        [DataMember] public TuneKind Tune;                  // --tune: settings by content
        [DataMember] public bool MaxEffort;                 // --preset max: speed ignored for quality and size

        public EncodeSettings() { SetDefaults(); }
        [OnDeserializing] void OnDeserializing(StreamingContext c) { SetDefaults(); } // JSON missing a field keeps its default

        [OnDeserialized] void OnDeserialized(StreamingContext c) { if (Output == null) Output = new OutputFormat(); } // explicit "Output": null

        void SetDefaults()
        {
            QIndex = 170;
            Tiles = TileMode.Auto;
            Device = DecisionDevice.Auto;
            KeyInterval = 0;
            Threads = 0;
            Refs = 0;
            BFrames = 3;
            KeepUnderSource = true;
            SizeCapPercent = 90;
            Output = new OutputFormat();
            Container = ContainerKind.Mp4;
            TileRows = TileMode.Auto;
            SceneCut = true;
            BQ = 16;
            AQ = -16;
            KQ = 0;
            DeblockDelta = 2;
            Cdef = true;
            ParallelB = ParallelMode.Auto;
            AudioKbps = 192;
            HeAacKbps = 32;
        }

        // The --device value to pass: DeviceArg, else what Device meant before devices were listed.
        public string DeviceValue
        {
            get
            {
                if (!string.IsNullOrEmpty(DeviceArg)) return DeviceArg;
                return Device == DecisionDevice.Gpu ? "gpu" : Device == DecisionDevice.Cpu ? "cpu" : "auto";
            }
        }

        public EncodeSettings Clone()
        {
            var c = (EncodeSettings)MemberwiseClone();
            c.Output = new OutputFormat { BitDepth = Output.BitDepth };
            return c;
        }
    }
}
