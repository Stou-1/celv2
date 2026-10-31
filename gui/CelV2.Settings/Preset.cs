using System.Runtime.Serialization;

namespace CelV2.Settings
{
    [DataContract]
    public sealed class Preset
    {
        [DataMember] public string Name;
        [DataMember] public EncodeSettings Settings;
    }
}
