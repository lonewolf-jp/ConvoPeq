#!/bin/bash
# D116-7 Shutdown/Restart cycles: start -> operate(publication) -> shutdown -> repeat
cd "C:/VSC_Project/ConvoPeq" || exit 1
SUM=evidence/D116_restart_summary.txt
echo "cycle,exitcode,publishes,perf_samples,anomalies,max_private_mb,first_cb_avg_us,last_cb_avg_us" > "$SUM"
for i in 1 2 3 4 5 6; do
  LOG="evidence/D116_restart_cycle${i}.log"
  MEMCSV="evidence/D116_restart_cycle${i}_mem.csv"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File evidence/D116_memory_sampler_poll.ps1 "C:\\VSC_Project\\ConvoPeq\\$MEMCSV" 40000 > /dev/null 2>&1 &
  SAMPLER=$!
  APPEXIT=$(cmd //c "build\\ConvoPeq_artefacts\\Release\\ConvoPeq.exe --cli-run --cli-log-file $LOG --cli-ir evidence\\D116_active.wav --cli-intent-burst-count 3 --cli-intent-burst-interval-ms 2000 --cli-exit-ms 15000 & echo %ERRORLEVEL%" | tr -d '\r')
  wait $SAMPLER 2>/dev/null
  PUB=$(grep -cE "^\[PUBLISH\]" "$LOG")
  PERF=$(grep -cE "CLI_PERF_RAW" "$LOG")
  ANOM=$(grep -cE "AUTH_CONTRACT\) FAIL|stall|XRUN|underrun|EMERGENCY|overflow|quarantine" "$LOG")
  MAXMEM=$(awk -F, 'NR>1 && $3+0>max {max=$3+0} END {print max}' "$MEMCSV")
  AVG1=$(grep -oE "procTimeUsAvg=[0-9.]+" "$LOG" | head -1 | cut -d= -f2)
  AVGN=$(grep -oE "procTimeUsAvg=[0-9.]+" "$LOG" | tail -1 | cut -d= -f2)
  echo "$i,$APPEXIT,$PUB,$PERF,$ANOM,$MAXMEM,$AVG1,$AVGN" >> "$SUM"
  echo "cycle $i done: exit=$APPEXIT publish=$PUB"
  sleep 2
done
echo "ALL CYCLES COMPLETE"
cat "$SUM"
