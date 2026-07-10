using Microsoft.Windows.EventTracing;
using Microsoft.Windows.EventTracing.Cpu;
using System.Text;

string etlPath = args.Length > 0 ? args[0] : @"C:\VSC_Project\ConvoPeq\doc\work63\ConvoPeqTrace_v2.etl";
int targetTid = args.Length > 1 ? int.Parse(args[1]) : 32352;
string outputPath = args.Length > 2 ? args[2] : @"C:\VSC_Project\ConvoPeq\doc\work63\cswitch_wait_time.csv";

Console.WriteLine($"Loading ETL: {etlPath}");
Console.WriteLine($"Target TID: {targetTid}");

using var processor = TraceProcessor.Create(etlPath);
var schedulingData = processor.UseCpuSchedulingData();
processor.Process();

var slices = schedulingData.Result.CpuTimeSlices;
Console.WriteLine($"Total CPU time slices: {slices.Count}");

var waitTimes = new List<int>();
var preempted = new List<(double Ts, int WaitUs, int Cpu, int NewTid)>();

foreach (var slice in slices)
{
    uint sliceTid = slice.SwitchIn?.ThreadId ?? slice.SwitchOut?.ThreadId ?? 0;
    if (sliceTid != targetTid)
        continue;

    var swIn = slice.SwitchIn;
    var swOut = slice.SwitchOut;

    if (swIn?.ThreadId == targetTid && swIn.WaitTime.HasValue)
    {
        int waitUs = (int)swIn.WaitTime.Value.TotalMicroseconds;
        waitTimes.Add(waitUs);
    }

    if (swOut?.ThreadId == targetTid)
    {
        int waitUs = (int)(swIn?.WaitTime?.TotalMicroseconds ?? 0);
        preempted.Add(((double)slice.StartTime.TotalSeconds, waitUs, (int)slice.Processor, (int)(swIn?.ThreadId ?? 0)));
    }
}

waitTimes.Sort();
int n = waitTimes.Count;
Console.WriteLine($"\n=== SwitchIn WaitTime for TID {targetTid} (n={n}) ===");
if (n > 0)
{
    Console.WriteLine($"Min:  {waitTimes[0],5}us");
    Console.WriteLine($"P50:  {waitTimes[n / 2],5}us");
    Console.WriteLine($"P90:  {waitTimes[(int)(n * 0.9)],5}us");
    Console.WriteLine($"P95:  {waitTimes[(int)(n * 0.95)],5}us");
    Console.WriteLine($"P99:  {waitTimes[(int)(n * 0.99)],5}us");
    Console.WriteLine($"Max:  {waitTimes[n - 1],5}us");
    Console.WriteLine($"Avg:  {(int)waitTimes.Average(),5}us");
    Console.WriteLine($" >1ms: {waitTimes.Count(x => x > 1000),5} ({100.0 * waitTimes.Count(x => x > 1000) / n:F1}%)");
    Console.WriteLine($" >5ms: {waitTimes.Count(x => x > 5000),5} ({100.0 * waitTimes.Count(x => x > 5000) / n:F1}%)");
}

Console.WriteLine($"\n=== Preempted events (n={preempted.Count}) ===");
var sorted = preempted.OrderByDescending(x => x.WaitUs).ToList();
foreach (var p in sorted.Take(10))
    Console.WriteLine($"  t={p.Ts:F3}s: CPU={p.Cpu} Wait={p.WaitUs,5}us -> TID={p.NewTid}");

// Save CSV
var sb = new StringBuilder();
sb.AppendLine("TimestampS,WaitTimeUs,CPU");
foreach (var p in sorted.Take(10000))
    sb.AppendLine($"{p.Ts:F3},{p.WaitUs},{p.Cpu}");
File.WriteAllText(outputPath, sb.ToString());
Console.WriteLine($"\nSaved: {outputPath}");
