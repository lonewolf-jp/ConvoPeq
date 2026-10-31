# STG-7 Discovery — STG-7 STOP (Concrete Defect found: STG-7-D1)

```text
Authority HEAD : 9afc71deb161b0fd862b21b4f19b4189856cc3b7
Authority file : ConvoPeq.md 5,586,461 B
SHA-256        : A7C03CD4FA02311BD1A29F5D9940F944BD26F00431DF0E98C1F59DCF025239DC
Generated      : 2026-09-28 11:58:08 / --check FRESH / NEWER_SRC_COUNT 0
Phase          : Discovery only（production 0 / test 0 / commit 禁止）
```

## STG-7-D1：requestLoadState が autoGainStagingEnabled フラグを復元しない

### Defect
```text
getCurrentState() は autoGainStagingEnabled を保存するが、
requestLoadState() は同値を手動ゲイン復元の要否判定にのみ使い、
setAutoGainStagingEnabled() を呼ばない。save→load で flag が黙って旧値のまま残る。
```

### Source
```text
保存：AudioEngine.StateIO.cpp L201（autoGainStagingEnabled を保存）
読取：同 L70-72（判定のみに使用）
欠落：同 L32-187 全域に setAutoGainStagingEnabled() 呼出しなし（実読確認）。
対照：DeviceSettings.cpp L1229 は device XML から同 flag を復元する。
```

### Trigger
```text
1. AutoGain=ON で save（S1：autoGain=true）。
2. live を OFF に変更（正規 UI 操作／device 設定読込）。
3. requestLoadState(S1)。
```

### Execution chain
```text
load：autoGainEnabled=true→手動ゲイン復元 skip→flag 書換えなし→終了。
live=(OFF, stale manual gains)。保存時=(ON, planner 駆動)。
```

### Expected state
```text
live autoGain=true（保存値と一致）。保存する値は復元されるか除外文書化されるべき。
oversamplingFactor は保存自体しない（StateIO L204-205 除外明記）ため対象外。
autoGain は保存するのに復元しない。
```

### Actual state
```text
live autoGain=false のまま。DSP は manual staging 駆動（保存時と異なる）。
save→load→save で 2 回目 save が autoGain=false に変異（semantic 非冪等）。
逆方向（保存 OFF→live ON→load）も同様に flag 不一致。
```

### Invariant violation
```text
保存した値は復元される（除外は明記）。手動ゲインは STG-1/STG-2 で保護済みだが、
その適用可否を決める flag 自体が復元されない。
```

### Impact
```text
user-visible：プリセット load 後に AutoGain toggle 状態が保存時と異なる。
runtime-visible：planner 駆動⇔manual 駆動の切替わり（可聴差）。
persistence：save/load/save 非冪等。
```

### Reproduction status
```text
直線コード（同期 public API のみ）のため code-level で deterministic。
setAutoGainStagingEnabled(true)→save→setAutoGainStagingEnabled(false)→
requestLoadState→isAutoGainStagingEnabled() が false のまま。
動的 test 追加は Owner GO 待ち（discovery-only のため未実施）。
```

### Proposed minimal repair boundary
```text
requestLoadState の Step 2 付近で setAutoGainStagingEnabled(autoGainEnabled) を
復元（1 行＋bulk guard 配慮）。StateIO.cpp のみ。DSP・authority・RT 不変。
```

## 反証・clear 済み事項

```text
- cache key 不完全性（mixed/tail）：convertToHighRes は convertFile 委譲で
  mixed/tail 非依存。key は格納内容を完全カバー。clear。
- scale の current-IR 相対 staleness：convert 経路は 1-arg（絶対）scale。
  3-arg は LoaderThread 経路専用。clear。
- upgrade 三重 guard／trash cap／defer lifecycle／RCU／handleLoadError／
  incremental：STG-5 で clear 済み。変更なし。
- SIMD／整数境界／publish integrity：STG-6 で clear 済み。変更なし。
- STG-1／STG-2／STG-4-1／STG-6-D1／既知問題：再提出なし。
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
STG-7 STOP — Concrete Defect found（STG-7-D1）。Owner 承認待ち。修正へ移行しない。
```
