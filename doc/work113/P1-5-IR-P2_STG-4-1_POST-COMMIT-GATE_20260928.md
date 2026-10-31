# STG-4-1 Post-Commit Gate — PASS

## 1. Commit identity

```text
SHA     : 901a6c336bbd9ec83db546028e749cc868f1a26a
message : fix(convolver): reject stale async IR completion via request-fixed generation
scope   : 7 files (+563/-39)
  ConvoPeq.md / src/ConvolverProcessor.h /
  src/convolver/ConvolverProcessor.LoadPipeline.cpp /
  src/convolver/ConvolverProcessor.LoaderThread.cpp /
  src/convolver/ConvolverProcessor.LoaderThreadInline.h /
  src/convolver/ConvolverProcessor.Rebuild.cpp /
  src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp
```

## 2. Cleanliness / scope

```text
residual Ms：AGENTS.md／headroom-proxy-start.ps1 のみ（pre-existing・commit 外・revert なし）。
src/・ConvoPeq.md：残差なし。STG-4-1 外 production change：なし。
```

## 3. Authority consistency

```text
ConvoPeq.md SHA-256 : 4FF62E0712F1751E2BEAB5E9183920D64E15ADA4F6CBEE1B3DA3C1DA06DE0FC0
size                : 5,580,623 B
Generated           : 2026-09-28 10:02:13
HEAD                : 901a6c33
--check             : FRESH（post-commit 再確認済み・再生成不要のため同 file が authority）
```

## 4. Post-commit regression

```text
再実行（post-commit Debug harness）：
  STG-4-1 PASS（phase1 flight 1/1→phase2 B effective→phase3 B維持・staleOverwrite=0）
  STG-1 6/6＋StateIO 3/3 PASS／STG-2-A/B/C＋R PASS／全先行 checks PASS
維持（commit は記録のみ・同一 sources／同一 binaries）：
  Debug PASS／Release 551/551 PASS／CTest 42/42 PASS
```

## 5. Architectural sanity（commit 前報告と同一）

```text
RT／RuntimeWorld／RuntimeStore／Publication／Crossfade／Retire-Epoch／
Coordinator／atomic wrapper／AutoGain：すべて unchanged。新規 raw atomic 0。
```

## push

```text
NOT AUTHORIZED（未実行）
```
