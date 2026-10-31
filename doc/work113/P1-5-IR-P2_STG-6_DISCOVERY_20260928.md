# STG-6 Discovery — STG-6 STOP (Concrete Defect found: STG-6-D1)

```text
Authority HEAD : 901a6c336bbd9ec83db546028e749cc868f1a26a
Authority file : ConvoPeq.md 5,580,623 B
SHA-256        : 69963A9BF920EAEDDD73A85EF7D57BC36E55A29DB349ADFB6A42EF352BF6FB7B
Generated      : 2026-09-28 10:43:22 / --check FRESH
Phase          : Discovery only（production 0 / test 0 / commit 禁止）
```

## STG-6-D1：sync preset load の silent drop（short IR × upsampling）

### Defect
```text
loadIR() 経由のプリセット要求が、resample 失敗→fallback→sr 不一致却下の連鎖で
適用されず、かつ一切の failure が UI エラー経路に報告されない。
ユーザー操作（IR 選択）が無言で無視される。
```

### Source
```text
src/IRDSP.cpp L67-103（flush loop：短入力で generated=0 のまま break→outLen=0）
src/IRConverter.cpp L258-280（outLen<=0→元 IR fallback、target SR のまま継続）
src/convolver/ConvolverProcessor.LoadPipeline.cpp L403-409（sr 不一致で return）
src/convolver/ConvolverProcessor.LoadPipeline.cpp L230-319 loadIR（全 failure
  branch に handleLoadError／setLastError なし。getLastError は空のまま）
対照：async path は LoaderThread L100／finalize L687-707 で handleLoadError 報告。
```

### Trigger
```text
1. default state（Auto OS→48k×8＝384k processing。OversamplingPolicy::resolve）。
2. 短尺だが valid な WAV（実測：128-sample float32 stereo。UI picker は長さ無検証）。
3. requestConvolverPreset(file)（production entry）。
```

### Execution chain（動的実測済み）
```text
[JUCELOG] resampleIR: trimmed to 0 (maxOutLen=1024)
[JUCELOG] resampleIR: success ch=2 outLen=0
[JUCELOG] convertFile: resampleIR failed, falling back to original IR
          (srcSr=48000.0 targetSr=384000.0)
[JUCELOG] applyComputedIR: sample rate mismatch sr=384000.0 preparedSr=48000.0
→ name 空・len 0・err 空・loading 0・finalized 0 のまま return。
```

### Invariant
```text
load 要求は適用されるか、失敗時は UI エラー経路（getLastError／handleLoadError）
に報告されること。handleLoadError／getLastError 機構の存在と async path の
使用がこの契約の根拠。sync path は同種 failure を一切報告しない。
```

### Counterexample
```text
128-sample valid WAV＋default 8x state で requestConvolverPreset→無変化・無報告。
（STG-4-1 作業中の Debug harness で2回実測。既存 seam のみ使用。）
```

### Impact
```text
user-visible：プリセット選択が無言で無視される（表示・音とも不変、エラーなし）。
波及なし（old state 保持・metadata 不変・leak なし）。default state で到達可能。
```

### Reproduction status
```text
動的確定済み（test-only LOG で全 chain 取得）。専用 regression test は未追加
（STG-6 は discovery only のため。test 追加は Owner GO 待ち）。
```

### Proposed repair boundary
```text
loadIR の failure branches（!exists／convert null／no-FFT）に
handleLoadError／setLastError 報告を追加。DSP・authority・RT 変更なし。
resample 自体の rework は対象外（報告が契約の中心）。
```

## その他の監査結果（clear／既知重複なし）

```text
§1 整数境界：FC-FORM admission（WORK102 テスト済み）／convertFile size_t 演算＋
    null check／computeKey CRC＋checksum／trim 動作（設計どおり）。
§2 DSP validation：LoaderThread chain（テスト済み）／upgrade 三重 guard／
    convert cancellation。B=4096 適用成功で正常系確認。
§3 partial init：PendingCommit／PreparedIRState／cache の組立は null・validation
    チェック付き。所有権移動は RAII。
§4 numerics：peak>10 却下＋jump check／bad-sample sanitization＋denormal zeroing／
    EQ denorm epsilon。scaleFactor 極値は clamp 内。
§5 SIMD：loadu 使用・bounds 事前検証＋corruption fallback（CustomInputOversampler
    L640-652）。既知 store 欠陥と重複なし。
§6 publish integrity：frozen／sealed／monotonicity guard＋admission tests。
STG-1／STG-2／STG-4-1／既知 D105・RCU・FFT overflow との重複なし。
```

## Phase gate

```text
Production changes: NONE
Test changes: NONE
commit: NOT AUTHORIZED
push: NOT AUTHORIZED
```

## STOP 状態

```text
STG-6 STOP — Concrete Defect found（STG-6-D1）。Owner 承認待ち。修正へ移行しない。
```
