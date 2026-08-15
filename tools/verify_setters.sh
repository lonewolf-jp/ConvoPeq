#!/usr/bin/env bash
# Verify §13.3 classification of AudioEngine.Parameters.cpp setters.
# Category A: direct submitRebuildIntent; Category B: uiConvolverProcessor.set* (indirect rebuild);
# Category C: publish-only (no rebuild).
cd "$(dirname "$0")/.." || exit 1
F=src/audioengine/AudioEngine.Parameters.cpp

cat_a="setEqBypassRequested setConvolverBypassRequested setInputHeadroomDb setOutputMakeupDb setProcessingOrder setConvolverInputTrimDb setDitherBitDepth setNoiseShaperType setSoftClipEnabled setSaturationAmount setOversamplingFactor setOversamplingType"
cat_b="setConvolverPhaseMode setConvolverTargetIRLength setConvolverRebuildDebounceMs setConvolverStateTree setConvolverTargetUpgradeFFTSize setConvolverEnableProgressiveUpgrade setConvolverMaxCacheEntries clearConvolverCache"
cat_c="setFixedNoiseLogIntervalMs setFixedNoiseWindowSamples setAudioThreadPriorityMode"

echo "=== Category A: expect submitRebuildIntent present ==="
for s in $cat_a; do
  n=$(rg -n "void AudioEngine::${s}\(" "$F" | head -1 | cut -d: -f1)
  if [ -z "$n" ]; then echo "  $s: NOT FOUND"; continue; fi
  r=$(sed -n "${n},$((n+45))p" "$F" | rg -c 'submitRebuildIntent')
  echo "  $s def:$n rebuild:$r"
done

echo "=== Category B: expect uiConvolverProcessor.set* (no direct rebuild) ==="
for s in $cat_b; do
  n=$(rg -n "void AudioEngine::${s}\(" "$F" | head -1 | cut -d: -f1)
  if [ -z "$n" ]; then echo "  $s: NOT FOUND"; continue; fi
  body=$(sed -n "${n},$((n+45))p" "$F")
  r=$(echo "$body" | rg -c 'submitRebuildIntent')
  u=$(echo "$body" | rg -c 'uiConvolverProcessor\.set|uiConvolverProcessor\.clear')
  echo "  $s def:$n rebuild:$r uiConvolver:$u"
done

echo "=== Category C: expect publish-only (no rebuild) ==="
for s in $cat_c; do
  n=$(rg -n "void AudioEngine::${s}\(" "$F" | head -1 | cut -d: -f1)
  if [ -z "$n" ]; then echo "  $s: NOT FOUND"; continue; fi
  body=$(sed -n "${n},$((n+45))p" "$F")
  r=$(echo "$body" | rg -c 'submitRebuildIntent')
  echo "  $s def:$n rebuild:$r"
done

echo "=== setEqLPFFilterMode (D row - withdrawn) ==="
n=$(rg -n "void AudioEngine::setEqLPFFilterMode\(" "$F" | head -1 | cut -d: -f1)
body=$(sed -n "${n},$((n+20))p" "$F")
r=$(echo "$body" | rg -c 'submitRebuildIntent')
echo "  setEqLPFFilterMode def:$n rebuild:$r"
