using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using CelV2.Settings;

namespace CelV2.Runner
{
    // Process plumbing shared by the runner classes.
    internal static class Proc
    {
        public sealed class Output { public int ExitCode; public string Out, Err; }

        // Where this thread's tool command lines and messages go (EncodeJob.Log while a job runs); null drops them.
        [ThreadStatic] public static Action<string> Log;

        public static Process Start(string exe, IEnumerable<string> args, bool stdin, bool stdout, bool stderr)
        {
            if (exe == null) throw new InvalidOperationException("tool not found");
            string joined = CommandBuilder.Join(args);
            var log = Log; if (log != null) log("> " + exe + " " + joined);
            var psi = new ProcessStartInfo(exe, joined)
            {
                UseShellExecute = false, CreateNoWindow = true,
                RedirectStandardInput = stdin, RedirectStandardOutput = stdout, RedirectStandardError = stderr,
            };
            // Every tool writes UTF-8 (celv2enc and av2_mux through their UTF-8 active code page); the default would be
            // Console.OutputEncoding, CP932 in a GUI process. Setting it on a stream that is not redirected throws.
            if (stdout) psi.StandardOutputEncoding = Utf8;
            if (stderr) psi.StandardErrorEncoding = Utf8;
            EnsureBomFreeStdin();
            return Process.Start(psi);
        }

        static readonly Encoding Utf8 = new UTF8Encoding(false);
        static bool stdinChecked;
        static readonly object StdinLock = new object();

        // .NET Framework builds the redirected stdin writer from Console.InputEncoding (cached on first read) and,
        // for UTF-8, writes a BOM into the pipe at start-up, which corrupts a binary y4m stream. Replace a UTF-8 input
        // encoding with a BOM-free one once, before the first Start. A GUI process without a console has code page 0.
        static void EnsureBomFreeStdin()
        {
            lock (StdinLock)
            {
                if (stdinChecked) return;
                stdinChecked = true;
                try
                {
                    if (Console.InputEncoding.GetPreamble().Length > 0) Console.InputEncoding = new UTF8Encoding(false);
                }
                catch (IOException) { } // no console attached: nothing to fix
            }
        }

        public static void Kill(Process p)
        {
            try { p.Kill(); }
            catch (InvalidOperationException) { } // already exited
            catch (Win32Exception) { }            // exiting right now
        }

        // Runs to completion, draining both streams. Cancelling kills the process (the caller checks the token).
        public static Output Capture(string exe, IEnumerable<string> args, CancellationToken cancel = default(CancellationToken))
        {
            var o = new StringBuilder();
            var e = new StringBuilder();
            using (var p = Start(exe, args, false, true, true))
            {
                p.OutputDataReceived += (s, ev) => { if (ev.Data != null) lock (o) o.AppendLine(ev.Data); };
                p.ErrorDataReceived += (s, ev) => { if (ev.Data != null) lock (e) e.AppendLine(ev.Data); };
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
                using (cancel.Register(() => Kill(p))) p.WaitForExit();
                var log = Log;
                if (log != null)
                {
                    foreach (string l in (e + "\n" + o).Split('\n'))  // celv2enc's per-frame progress lines left out
                        if (l.Trim().Length > 0 && !l.StartsWith("progress frame=", StringComparison.Ordinal)) log(l.TrimEnd('\r'));
                    log("(終了コード " + p.ExitCode + ")");
                }
                return new Output { ExitCode = p.ExitCode, Out = o.ToString(), Err = e.ToString() };
            }
        }

        // The last non-empty lines of text, for error messages.
        public static string Tail(string text, int lines)
        {
            var l = (text ?? "").Split('\n').Select(x => x.TrimEnd('\r')).Where(x => x.Trim().Length > 0).ToList();
            return string.Join("\n", l.Skip(Math.Max(0, l.Count - lines)));
        }
    }
}
