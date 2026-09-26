# P1-5-IR-P2 — Step 5-X / P3-5-R16: Take Counter Implementation + Build Gate

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R16）
- **判定**: **R16-A IMPLEMENTED + BUILD PASS**。本 Step で停止する。
  F/R 実行・比較・P3-1-D には進まない。
- **変更**: production 最小（counter＋getter＋take点）＋ build のみ。

---

## 1. State Freeze（実装前・PASS）

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
R14=C／R15=A／R10 accessor・T6/T9 不変／P3-1-D 未着手
```

## 2. 実装内容（3 edits・2ファイル中の実質1ファイル）

### A. `src/audioengine/AudioEngine.h`

1. counter 宣言（rebuildBacklog_ 近傍・:5040）：
```cpp
std::atomic<std::uint64_t> rebuildTakeCount_ { 0 };
```
命名衝突なし（exhaustive grep 確定）。process lifetime 累積・reset なし。

2. getter（getRebuildDispatchDiagnostics 直後・public 領域）：
```cpp
[[nodiscard]] std::uint64_t getRebuildTakeCount() const noexcept
{
    return convo::consumeAtomic(rebuildTakeCount_, std::memory_order_acquire);
}
```
`RebuildDispatchDiagnostics` struct 本体には触れない（R15-Candidate C 棄却の維持）。
dead field 群の復活なし。

### B. `src/audioengine/AudioEngine.RebuildDispatch.cpp`（take点・:911）

```cpp
task = pendingTask;
pendingTask.currentDSP = nullptr;
hasPendingTask = false;
// ★ P3-5-R16（take完了直後のみ）
convo::fetchAddAtomic(rebuildTakeCount_, static_cast<std::uint64_t>(1),
                      std::memory_order_acq_rel);
```

- queue成功・wake・backlog clear・build・commit・publish では加算しない。
- `rebuildBacklog_` semantics 不変（:921 維持）。
- memory order は同ファイル idiom（acq_rel）に合わせた。

## 3. 変更レビュー表（§6・PASS）

```text
counter 1個／getter 1個／increment 1箇所（take直後）: PASS（site列挙で確定）
audio thread 非関与（writerはworker threadのみ）: PASS
RT lock／allocation なし（atomicのみ）: PASS
ownership exposure なし（scalarのみ）: PASS
RuntimeWorld exposure なし: PASS
publish／retire／rebuild-semantics／shutdown-semantics 変更なし: PASS
existing diagnostics 不変（struct＋他counter untouched）: PASS
R10 accessor 不変／T6／T9 不変: PASS
```

## 4. Build（G6・PASS）

```text
target AudioEngineHarness／Release／OFF（vcvars64＋oneAPI・A3先例準拠）
result BUILD_EXIT=0（73 targets・link成功）
```

注：clangd は `fetchAddAtomic` を既存行含め全面誤診する（JUCE parse不能の既存不具合）。
MSVC 実 build の成功が正本である（R10 Step5R と同一結論）。

## 5. Binary identity（G7）

```text
path build/Release/AudioEngineHarness.exe → tmp/p35_R16.exe に保存
SHA-256 1651d2ab1a84686c7c2f0b32a0b19f0de729b41c71209b5d376dc095279c3316
prefix 1651d2ab1a84686c（R10 a7bfb513… と意図どおり相違）
p15ir string 1（含有）／CONVOPEQ_CORRECT OFF
HEAD／kP15／production diff 0（再確認）
```

## 6. A/B/C 検証

```text
counter 1／getter 1／take writer 1／take位置正当（:906-909 直後）／
queue semantics不変／publication semantics不変／RT path不変／
ownership非露出／build PASS → R16-A
R16-B／R16-C： 該当なし
```

## 7. R16-A 後の扱い（STOP）

- 本 Step で停止する。F/R実行・30/60/120待機・build-result追加・
  commit-attempt追加・health/pressure計装・P3-1-D・帰属結論なし。
- 次段（R17）では `queued Δ／taken Δ／published Δ` の
  `1/0/0 vs 1/1/0` 判定が初めて可能になる。
  `taken=1, published=0` が実測された場合のみ build-result boundary を設計する。
- R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
