# P1-5-IR-P2 — Step 5-R / P3-5-R10: Accessor Implementation + Build Gate

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R10）
- **判定**: **R10-A IMPLEMENTED + BUILD PASS**。本 Step で停止する。
  F/R 実行・比較・P3-1-D には進まない。
- **変更**: 2ファイルのみ（§2）。production 機能変更なし。

---

## 1. State Freeze（実装前・PASS）

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
AudioEngine.h                untouched（実装前確認）
production/CMake/JUCE        0 diff
```

## 2. test-only diff（実装内容）

### A. `src/audioengine/AudioEngine.h`（+51行）

- `struct RuntimeActiveSnapshot`（16 fields・R9 §3完全対応・plain POD）。
  除外遵守：sampleRate／pointer／handle／smart pointer／string／logger state なし。
- `getActiveRuntimeSnapshot() const noexcept`（value return）。
  経路：acquireReadToken → consumeWorldHandle → null check → field copy →
  scope終了 → return（R9 §4どおり。`observeCurrentRuntime` 不使用）。
- 配置：`hasPublishedRuntimeDSP()` 直後（同一 public 領域・同一 token pattern）。
- 命名決定を記録：`RuntimeActiveSnapshot`／`getActiveRuntimeSnapshot`
  （R7/R9 で未定だった事項の本 Step 確定）。

### B. `src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp`

- `CaseOut` に `snapBefore／snapAfter`（値コピー）追加。
- `snapTokens(prefix, snap)` formatter（既存 kv/kvi 形式踏襲・新 logger 機構なし）。
- T6（post-settle 後・capture 前）、T9（clearTap 後）に瞬間 read。
- runPair twoState emit に `ab0／aa0／ab1／aa1` の4ブロック追加。
  requested==active の成功判定コードなし。seq-only 断定なし。

## 3. G1–G5 diff gate（実装直後・PASS）

```text
G1 source diff： AudioEngine.h＋P1 TU のみ（他は P3-3 以前の持ち越し）
G2 forbidden：   CMake／JUCE／RuntimeBuilder／Commit／Publish／Retire／
                 RCU実装／Rebuild dispatch／measurement／logger／CLI／wait-sleep 0件
G4 API surface： new POD＝1／new public method＝1
G5 ownership：   RuntimeWorld*／&／RuntimeReadHandle／GlobalSnapshot*／
                 shared_ptr／unique_ptr の露出0件
     signature： () const noexcept・value return（確認）
```

## 4. G6 build（PASS）

```text
target   AudioEngineHarness／Release／OFF（vcvars64＋oneAPI・A3先例準拠）
result   BUILD_EXIT=0（73 targets・header変更に伴う再compile含む・link成功）
```

注：clangd は実装中に `No type named 'RuntimeActiveSnapshot'` と誤診したが、
同 tool は JUCE header 自体を parse 不能（既存設定不備）であり stale／誤診である。
MSVC 実 build の成功が正本である。

## 5. G7 binary identity

```text
path     build/Release/AudioEngineHarness.exe → tmp/p35_R10.exe に保存
SHA-256  a7bfb513a7d7eea7d2bc198530f7d2b55125a207d00a0dfd1f76d278228e18ba
prefix   a7bfb513a7d7eea7（F f4723815…／R1 a00d140d… と意図どおり相違）
p15ir string  1（含有）
CONVOPEQ_CORRECT_POLYPHASE_GAIN  OFF
HEAD 1e9e63e3／kP15FullMatrix true／production diff 0（再確認）
```

## 6. R10判定

```text
R10-A IMPLEMENTED + BUILD PASS
  → 次の execution validation gate へ（別Step）。
  → 本 Step で停止。F/R実行・比較・P3-1-Dなし。
R10-B／R10-C： 該当なし。
```

## 7. 次gate申送り

- 初回 execution validation では T6／T9 の `ab*／aa*` 行が出ること、
  `generation／worldId／sequence` の三つ組がゼロでないこと（publish 済み world 観測）、
  requested との照合は監査側で行うことを確認する。
- no-world 時は zero-POD（caller 側三つ組ゼロ判定）。新規 validity 機構なし。
- R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
