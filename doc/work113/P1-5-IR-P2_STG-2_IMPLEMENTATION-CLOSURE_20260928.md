# STG-2 Implementation Closure（commit 前停止）

```text
Authority base : 6dc03213 / ConvoPeq.md 3C61996501C6DA27AD85F36998E746301BC6CBF1BBED48F338C11015A508CD7A
STG-1          : CLOSED のまま変更なし
```

## Repair 構造

```text
1. pure helpers（AudioEngine.Parameters.cpp・無名 namespace・副作用なし）:
     inputHeadroomCeilingDb(convBypassed, eqBypassed, order) → -6.0 / 0.0
     clampInputHeadroomDb(db, ceiling) → jlimit(-12.0, ceiling, db)
2. setInputHeadroomDb() を helpers 使用に refactor（挙動同一）。
3. 新 private member revalidateInputHeadroomForCurrentMode()
     （AudioEngine.h 宣言 1 行＋Parameters.cpp 定義）:
     現 mode ceiling で clamp → 変化時のみ 3 atomic publish。rebuild submit なし。
4. 3 mode-changing path から revalidate を呼出し（各 setter の既存 submit の直前）:
     setProcessingOrder / setEqBypassRequested / setConvolverBypassRequested
```

## Rebuild duplication gate（§7）

```text
静的（確定）:
  setEqBypassRequested        submit 1 / revalidate 1
  setConvolverBypassRequested submit 1 / revalidate 1
  setProcessingOrder          submit 1 / revalidate 1
  setInputHeadroomDb          submit 1 / revalidate 0
  revalidate 自体             submit 0
  → 1 logical mode change = 1 submitRebuildIntent call site。二重経路なし（構造保証）。
動的（情報記録・断定なし）:
  [STG-2-R] order transition build generations 2 -> 2
  → 既存 merge/coalesce が吸収。revalidate 側の追加 submit は 0 のため、
    余分な intent は発生し得ない。
```

## 検証結果

```text
修正前（同一 test・production 変更前）:
  STG-2-A FAIL: got 0.000000 want <= -6.0
  STG-2-B FAIL: got 0.000000 want <= -6.0
  STG-2-C FAIL: saved 0.000000 loaded -6.000000
修正後:
  0.0 → -6.0（A/B: ceiling 適用・C: saved -6.0/loaded -6.0 固定点）
  STG-2-A/B/C PASS、Extended R4/R5/R6 PASS（-8.0 維持・-6.0 境界維持・引上なし・save/load/save 安定）
  STG-1 6/6＋StateIO 3/3 PASS（preservation 契約維持）
  Debug build PASS（harness exit 0・all publish pipeline PASS）
  Release build PASS（573/573）
  CTest PASS（42/42・Release AudioEngineHarness 含む）
```

## Stop 条件（§11）该当なし

```text
1-8 いずれも不該当。AutoGain 挙動不変（Builder plan 経路 untouched）。
RT/ISR/RuntimeWorld/Publication/Crossfade/Retire-Epoch/Rebuild ownership/atomic 契約 untouched。
scope: Parameters.cpp＋AudioEngine.h 宣言1行＋test のみ。
```

## Phase gate

```text
production source:
  M src/audioengine/AudioEngine.Parameters.cpp（helpers＋revalidate＋3 call sites＋setter refactor）
  M src/audioengine/AudioEngine.h（private 宣言 1 行）
test:
  M src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp（STG-2-A/B/C＋STG-2-R＋runner）
authority: NOT YET COMMITTED（ConvoPeq.md 再生成は commit GO 後）
commit: NOT AUTHORIZED
push: NOT AUTHORIZED
```
