using System;

namespace CelV2.Tests
{
    public sealed class CheckFailure : Exception
    {
        public CheckFailure(string msg) : base(msg) { }
    }

    // Thrown by a test whose prerequisite (a tool, an env var) is absent; TestMain reports it as SKIP, not FAIL.
    public sealed class SkipTest : Exception
    {
        public SkipTest(string reason) : base(reason) { }
    }

    public static class Check
    {
        public static void That(bool cond, string msg)
        {
            if (!cond) throw new CheckFailure(msg);
        }

        public static void Equal<T>(T expected, T actual, string msg)
        {
            if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(expected, actual))
                throw new CheckFailure(msg + ": expected <" + expected + "> but was <" + actual + ">");
        }
    }
}
