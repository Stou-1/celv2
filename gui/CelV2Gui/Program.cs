using System;
using System.IO;
using System.Windows.Forms;
using CelV2.Ui;

namespace CelV2Gui
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            string settingsPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "celv2gui", "settings.json");
            Application.Run(new MainForm(AppSettings.Load(settingsPath), settingsPath));
        }
    }
}
