using System.Collections.Generic;

namespace CelV2.Settings
{
    // One choice of where the decisions run: Arg is celv2enc's --device value ("auto", "gpu", "vulkan2",
    // "cpu", ...), Label what the dialog shows.
    public sealed class DeviceOption
    {
        public string Arg, Label;
    }

    public static class DeviceList
    {
        // celv2enc --list-devices output ("<value>\t<name>" lines) as the dialog's choices: 自動 first, then
        // each device ("NVIDIA GeForce RTX 5090（CUDA）"); lines without a tab are skipped.
        public static List<DeviceOption> Parse(string output)
        {
            var list = new List<DeviceOption> { new DeviceOption { Arg = "auto", Label = "自動" } };
            foreach (string raw in (output ?? "").Split('\n'))
            {
                string line = raw.TrimEnd('\r');
                int tab = line.IndexOf('\t');
                if (tab <= 0 || tab + 1 >= line.Length) continue;
                string arg = line.Substring(0, tab), name = line.Substring(tab + 1);
                if (name.EndsWith(")") && name.LastIndexOf(" (") > 0)  // "<name> (CUDA)" -> "<name>（CUDA）"
                {
                    int p = name.LastIndexOf(" (");
                    name = name.Substring(0, p) + "（" + name.Substring(p + 2, name.Length - p - 3) + "）";
                }
                list.Add(new DeviceOption { Arg = arg, Label = name });
            }
            return list;
        }

        // Without celv2enc's list (not found, or it failed): the choices of earlier versions.
        public static List<DeviceOption> Fallback()
        {
            return new List<DeviceOption>
            {
                new DeviceOption { Arg = "auto", Label = "自動" },
                new DeviceOption { Arg = "gpu", Label = "GPU（CUDA）" },
                new DeviceOption { Arg = "cpu", Label = "CPU" },
            };
        }
    }
}
