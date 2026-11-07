using System;
using System.Collections.Generic;

namespace CelV2.Tests
{
    public static class TestMain
    {
        // Later tasks append their test lists here.
        static IEnumerable<(string name, Action run)> All()
        {
            foreach (var t in SettingsTests.All) yield return t;
            foreach (var t in CommandTests.All) yield return t;
            foreach (var t in RunnerTests.All) yield return t;
            foreach (var t in UiTests.All) yield return t;
            foreach (var t in QualityTests.All) yield return t;
            foreach (var t in DetailsTests.All) yield return t;
        }

        [STAThread]
        public static int Main()
        {
            int failed = 0, total = 0, skipped = 0;
            foreach (var t in All())
            {
                total++;
                try { t.run(); Console.WriteLine("PASS " + t.name); }
                catch (SkipTest e) { skipped++; Console.WriteLine("SKIP " + t.name + " (" + e.Message + ")"); }
                catch (Exception e) { failed++; Console.WriteLine("FAIL " + t.name + ": " + e.Message); }
            }
            Console.WriteLine((total - failed - skipped) + "/" + total + " passed" + (skipped > 0 ? ", " + skipped + " skipped" : ""));
            return failed == 0 ? 0 : 1;
        }
    }
}
