# D169-2-4 — Minimal Implementation（Work Record）

```text
D169-2-4 — Duplicate-Prepare Collapse Minimal Implementation
Date:     2026-09-07
Contract: evidence/D169/D169_2_2_REPAIR_CONTRACT.md（RC-D169-2-1〜7）
Preflight: evidence/D169/D169_2_3_PREFLIGHT_SOURCE_AUDIT.md（P1〜P5 全 GO）
Verdict:  **IMPLEMENTED**
```

## 実装内容

**変更ファイル: `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` のみ（+17 行・functional change 1 箇所）。**

`enterPrepare()` 戻り値の直後に collapse detection を挿入:

```cpp
auto lifecycleToken = lifecycleRuntime_.enterPrepare(samplesPerBlockExpected, static_cast<int>(sampleRate));

// ★ D169-2-4 (RC-D169-2-1/3/5): duplicate-prepare collapse = no-op。...
if (lifecycleToken.expectedPhase == convo::isr::LifecyclePhase::Prepared)
{
    diagLog("[DIAG] prepareToPlay: duplicate-prepare collapsed (same SR/BS, already Prepared)");
    return;
}
```

- 判別子: 既存 `LifecycleToken::expectedPhase == LifecyclePhase::Prepared`
  （collapse 経路のみが生成する一意判別子 — D169-2-3 P1/P5 実証済み・新規 state 0）
- collapse 時: 診断ログ 1 行のみで return（prepare body 全 side effect 不実行 — RC-5）
- コメント内に行番号参照を置かない（D113-A 行番号自己ずれ教訓の適用）

## 完了確認 [1]〜[6]

| # | 確認項目 | 結果 |
| --- | --- | --- |
| [1] | 変更ファイルが PrepareToPlay.cpp のみ | ✓ 本 track の diff は 1 ファイルのみ（working tree の他の変更分は D167/D170 の既存未 commit 分） |
| [2] | `expectedPhase == LifecyclePhase::Prepared` 判定が enterPrepare 戻り値直後 | ✓ :20 呼出 → :33 判定 |
| [3] | collapse branch から leavePrepare() 到達不能 | ✓ return（:36）→ leavePrepare（:344）に到達経路なし |
| [4] | collapse branch から prepare body 全 side effect に到達不能 | ✓ 分岐は D169-2-3 P4 の 25 side effect 項目すべてより前に位置 |
| [5] | ISRLifecycle.cpp / .h 無変更 | ✓ `git diff --stat` empty |
| [6] | RC-1〜RC-10 違反なし | ✓ 下表 |

## RC-1〜RC-10 照合

| RC | 照合 |
| --- | --- |
| RC-1 collapse 維持 | ✓ enterPrepare の duplicate 判定を変更していない |
| RC-2 真の no-op | ✓ body 非実行・即 return |
| RC-3 leavePrepare 非呼出 | ✓ [3] のとおり |
| RC-4 leavePrepare 前提不変 | ✓ ISRLifecycle.cpp 変更 0 |
| RC-5 side effect = 0 | ✓ [4] のとおり（診断ログ 1 行のみ許容内） |
| RC-6 非collapse 不変 | ✓ 分岐は collapse のみで return・通常経路の制御フロー無変更 |
| RC-7 単一 authority | ✓ collapse 判定は LifecycleIsolationRuntime に残留・prepareToPlay は token を読むのみ |
| RC-8 新規 state 禁止 | ✓ 新規 atomic/mutex/generation/transaction ID 0・既存 field の読み取りのみ |
| RC-9 blocked-return scope 外 | ✓ lifecycleState CAS block 経路（:67-84）は無変更 |
| RC-10 protocol 不変 | ✓ placeholder / world / rebuild / publish / retire / shutdown FSM 触らず |

## Build

| Config | 結果 |
| --- | --- |
| Debug（build-diag） | ✓ complete（exit 0） |
| Release（build-diag） | ✓ complete（exit 0） |
| RelWithDebInfo（build-diag） | ✓ complete（exit 0） |

logs: evidence/D169/d169_2_4_build_{debug_release,rwdi}.log

## Verdict

```text
D169-2-4
Verdict: IMPLEMENTED

Changed files:
  src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp（+17 行）

Diff summary:
  enterPrepare() 戻り値直後に collapse detection 1 箇所を挿入
  （expectedPhase == Prepared → diagLog 1 行 → return）+ RC 意味論コメント

Collapse path:
  enterPrepare → expectedPhase==Prepared → return

Non-collapse path:
  unchanged

leavePrepare:
  unchanged / unreachable from collapse

New state/synchronization:
  0

Contract violations:
  NONE

Build:
  Debug / Release / RelWithDebInfo 全 config ✓
```

次: **D169-2-5 targeted collapse regression**（same-SR/BS 連続 prepare の abort 非発火・
side effect 不実行・非 collapse 経路の既存テスト無変更確認）→ D169-2-6 → D169-2-7。
