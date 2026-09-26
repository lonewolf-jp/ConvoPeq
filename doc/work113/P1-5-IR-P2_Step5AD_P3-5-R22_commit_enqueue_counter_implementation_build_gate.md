# P1-5-IR-P2 — Step 5-AD / P3-5-R22: Commit-Enqueue Counter Implementation + Build Gate

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R22）
- **判定**: **R22-A IMPLEMENTED + BUILD PASS**。本 Step で停止する。
  F/R 実行・比較・P3-1-D には進まない。
- **変更**: commit-enqueue counter＋getter＋main-site increment（§2）＋
  test delta `cmt` field（§3）のみ。production 機能変更なし。

---

## 1. State Freeze（実装前・PASS）

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
R16 take counter＋R19 B2 counter＋R10 accessor・T6/T9 保持
R21 設計（main-site commit-enqueue・単一位置）をそのまま実装
```

## 2. 実装内容（3 edits＋test拡張1）

### A. `src/audioengine/AudioEngine.h`

1. counter 宣言（rebuildTakeCount_／rebuildBuildResultCount_ 近傍）：
```cpp
std::atomic<std::uint64_t> rebuildCommitEnqueueCount_ { 0 };
```
命名衝突なし（exhaustive grep 確定）。

2. getter（getRebuildBuildResultCount 直後・public 領域）：
```cpp
[[nodiscard]] std::uint64_t getRebuildCommitEnqueueCount() const noexcept
{
    return convo::consumeAtomic(rebuildCommitEnqueueCount_, std::memory_order_acquire);
}
```
`RebuildDispatchDiagnostics` struct 本体不変（Candidate C 棄却の維持）。
dead field 群の復活なし。

### B. `src/audioengine/AudioEngine.RebuildDispatch.cpp`（main-site・:1420）

```cpp
// ★ P3-5-R22: main-site commit-enqueue 到達のみ加算する counter。
//   recovery enqueue（:1085／:1165）は含めない。queue／take／build／
//   commit／publish では加算しない。
convo::fetchAddAtomic(rebuildCommitEnqueueCount_, static_cast<std::uint64_t>(1),
                      std::memory_order_acq_rel);
enqueuePublicationIntentForRuntimeCommit(dspToCommit, task.generation, ...);
```

- 関数本体内ではなく main-site 呼出し到達直前に配置（R21 §7決定どおり）。
- memory order は同ファイル idiom（acq_rel）に合わせた。

### C. P1 TU test delta 拡張（`emitQDelta` に `cmt` 追加）

```cpp
+ kvi("cmt", (long long)e.getRebuildCommitEnqueueCount())
```

- 新規 logger／prefix family なし。build-result／commit／health／pressure の
  新規計装なし（R17 §5遵守）。
- B2 位置は R18 定義どおり（§2）。R vehicle build／実行は実施していない。

## 3. G1–G5相当の確認

```text
変更ファイル： AudioEngine.h＋RebuildDispatch.cpp（R22分）＋P1 TU（test観測分）。
  他は P3-3 以前の持ち越しのみ。
forbidden diff： CMake／JUCE／RuntimeBuilder／Commit／Publish／Retire／
  RCU実装／measurement／logger／CLI／wait-sleep 0件。
API surface： 新 POD なし・新 public method＝getter 1
  （counter 本体は private member）。
ownership： RuntimeWorld*／&／handle／smart pointer の露出0件。
signature： () const noexcept・value return（確認）。
```

## 4. Build（G6・PASS）

```text
target AudioEngineHarness／Release／OFF（vcvars64＋oneAPI・A3先例準拠）
result BUILD_EXIT=0（73 targets・link成功）
```

- 全target build は R19 と同一の pre-existing MKL include 条件で不可のため、
  harness target を authoritative gate とする（R19 Step5AA の分離記録を継承）。
- clangd の `fetchAddAtomic` 全面誤診は JUCE parse不能の既存不具合であり、
  MSVC 実 build が正本（R10／R16／R19 と同一結論）。

## 5. Binary identity（G7）

```text
path build/Release/AudioEngineHarness.exe → tmp/p35_R22.exe に保存
SHA-256 220902dab7677ee0a0f3fa1e78df180fb8e6339888543b0fd69cb488c53e823c
prefix 220902dab7677ee0（R19 cd6f87f6… と意図どおり相違）
p15ir string 1（含有）／CONVOPEQ_CORRECT OFF
HEAD／kP15／production差分（R22承認範囲のみ）再確認
```

## 6. R22判定

```text
R22-A IMPLEMENTED + BUILD PASS
  counter 1／getter 1／main-site writer 1／R16-R19 counters 不変／
  test delta reader 更新／新規観測機構なし／source reconciliation PASS／
  Release harness build PASS。
  → 次は P3-5-R23（take／bld／commit／seq の4軸実測）。
R22-B／R22-C： 非該当。
```

## 7. STOP（本 Step 終了）

- F/R実行・30/60/120待機・U6断定・commit計装追加・health/pressure判定・
  buzz調査・P3-1-D・帰属結論なし。
- source は R22 vehicle（F順序＋delta観測＋cmt）のまま保持する。revert なし。
- R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
