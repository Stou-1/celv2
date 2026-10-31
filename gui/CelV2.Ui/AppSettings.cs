using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;

namespace CelV2.Ui
{
    // The window's remembered choices (%APPDATA%\celv2gui\settings.json when hosted by CelV2Gui.exe).
    [DataContract]
    public sealed class AppSettings
    {
        [DataMember] public string LastPreset;
        [DataMember] public CelV2.Settings.TuneKind LastTune;  // the 内容 choice, kept across presets
        [DataMember] public bool LastMaxEffort;  // 「とても遅いが最高画質」, kept across presets
        [DataMember] public string OutputFolder;    // null = next to each input
        [DataMember] public string FfmpegFolder;    // null = search
        [DataMember] public string Mp4BoxFolder;    // null = search

        public AppSettings() { SetDefaults(); }
        [OnDeserializing] void OnDeserializing(StreamingContext c) { SetDefaults(); } // the serializer skips constructors
        void SetDefaults() { LastPreset = "標準"; }

        // A missing or unreadable file gives the defaults.
        public static AppSettings Load(string path)
        {
            try
            {
                AppSettings a;
                using (var s = File.OpenRead(path))
                    a = (AppSettings)new DataContractJsonSerializer(typeof(AppSettings)).ReadObject(s) ?? new AppSettings();
                if (a.LastPreset == "小さめ") a.LastPreset = "小容量"; // the built-in preset's old name
                return a;
            }
            catch (Exception e) when (e is IOException || e is UnauthorizedAccessException || e is SerializationException || e is System.Xml.XmlException)
            {
                return new AppSettings();
            }
        }

        public void Save(string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
            using (var s = File.Create(path))
            using (var w = JsonReaderWriterFactory.CreateJsonWriter(s, Encoding.UTF8, false, true))
                new DataContractJsonSerializer(typeof(AppSettings)).WriteObject(w, this);
        }
    }
}
