using System.Collections.Generic;
using System.Threading;

namespace CelV2.Runner
{
    public interface IMuxStep
    {
        // Called after video and audio are written; returns the final output files. Cancelling the token kills the
        // tool being run and makes Finish throw (after removing what it created).
        IReadOnlyList<string> Finish(string videoIvf, string audioOrNull, CancellationToken cancel);
    }

    // Keeps the IVF and the audio file as they are.
    public sealed class SeparateFilesMux : IMuxStep
    {
        public IReadOnlyList<string> Finish(string videoIvf, string audioOrNull, CancellationToken cancel)
        {
            var l = new List<string> { videoIvf };
            if (audioOrNull != null) l.Add(audioOrNull);
            return l;
        }
    }
}
