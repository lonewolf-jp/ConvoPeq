# work64 オーディオスレッド CPU コア固定化 改修計画 妥当性検証報告書

**検証日**: 2026-07-05 | **検証者**: GitHub Copilot (DeepSeek V4 Flash)
**対象**: `doc/work64/work64-plan-audio-thread-affinity.md` (v11) | **プロジェクト**: ConvoPeq v0.6.8

---

## 0. 検証サマリ

| 項目 | 結果 |
|---|---|
| MSDN公式ドキュメント調査 | **7件**のAPIドキュメント + **2件**の概念ドキュメントを確認 |
| ソースコード行単位照合 | **13ファイル**を直接読み取り確認 |
| 既存バグ確認 | **1件** (v7 H7: `initWorkerThread()` 順序バグ) を**実コードで確認** |
| 細かな記載誤り | **1件**: ファイルパス `system/` → `native/` |
| 総合判定 | **実装可能** — 全主要主張がMSDN公式 + 実コードと一致 |

---

## 1. MSDN 公式ドキュメント検証結果

### 1.1 検証した API/構造体

| # | API/構造体 | MSDN URL | Plan参照箇所 |
|---|---|---|---|
| 1 | `SetThreadAffinityMask` | `winbase/nf-winbase-setthreadaffinitymask` | §5.1(c)(h), §5.4, §11.2(E) |
| 2 | `GetLogicalProcessorInformationEx` | `sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex` | §5.1(f) |
| 3 | `PROCESSOR_RELATIONSHIP` | `winnt/ns-winnt-processor_relationship` | §5.1(f) |
| 4 | `GROUP_AFFINITY` | `winnt/ns-winnt-group_affinity` | §5.1(f) |
| 5 | `AvSetMmThreadCharacteristics` | `avrt/nf-avrt-avsetmmthreadcharacteristicsa` | §3.3, §5.4 |
| 6 | `AvSetMmThreadPriority` | `avrt/nf-avrt-avsetmmthreadpriority` | §3.3 |
| 7 | `SetThreadIdealProcessor` | `processthreadsapi/nf-processthreadsapi-setthreadidealprocessor` | §11.2(J) |
| 8 | CPU Sets (概念) | `procthread/cpu-sets` | §11.2(K) |
| 9 | Processor Groups (概念) | `procthread/processor-groups` | §11.2(A)(E) |

### 1.2 重要仕様の一致確認

#### `SetThreadAffinityMask` (winbase.h)
- **パラメータ型**: `DWORD_PTR` → Plan: §5.1(f)で `DWORD_PTR` と明記 ✅
- **Win11制約**: "dwThreadAffinityMask must specify processors in the thread's current primary group" → Plan: §5.1(f) で明記 ✅
- **エラー時戻り値**: "return value is zero" → Plan: §5.1(h) で `prevMask == 0` 診断ログ推奨 ✅
- **プロセスアフィニティ制約**: "must be a subset of the process affinity mask" → Plan: §5.1(h) で "audioMask はプロセスアフィニティの部分集合" ✅

#### `GetLogicalProcessorInformationEx` (sysinfoapi.h)
- **最小OS**: Windows 7 → Plan: §5.1(f) で "Windows 7+" と明記 ✅
- **SDK要件**: `_WIN32_WINNT >= 0x0601` → Plan: §11.1#9 で確認 ✅
- **RelationProcessorCore**: "returns a PROCESSOR_RELATIONSHIP structure for every active processor core in every processor group" → Plan: §5.1(f) で記載 ✅
- **WOW64 folding**: 32-bit >64 LP で folding → Plan: "x64専用ビルドのため適用外" ✅
- **バッファサイズ**: ERROR_INSUFFICIENT_BUFFER で RequiredLength → Plan: §5.1(f) の2パス方式 ✅

#### `PROCESSOR_RELATIONSHIP` (winnt.h)
- **GroupCount (core)**: MSDN "GroupCount member is always 1" → Plan: "`== 1` を前提" ✅
- **EfficiencyClass**: MSDN "only nonzero on systems with a heterogeneous set of cores", "higher value = greater performance and less efficiency" → Plan: "!= 0 → heterogeneous" ✅
- **EfficiencyClass 最小OS**: Windows 10 → Plan: "Win10+" と明記 ✅
- **GroupMask**: `GROUP_AFFINITY` の配列 → Plan: `GroupMask[0].Mask` のみ採取 ✅

#### `GROUP_AFFINITY` (winnt.h)
- **Mask 型**: `KAFFINITY` (= `ULONG_PTR` = `DWORD_PTR`) → Plan: "DWORD_PTR" と明記 ✅

#### CPU Sets (概念ページ)
- **アフィニティ優先**: "If a thread or process has a restrictive affinity mask set, the affinity mask is respected above any conflicting CPU Set assignment" → Plan §11.2(K) K1 ✅

#### Processor Groups (概念ページ)
- **<=64LP**: "Systems with fewer than 64 logical processors always have a single group, Group 0" → Plan §5.1(f) 前提 ✅
- **Win11+**: "processes and their threads have processor affinities that by default span all processors" → Plan §5.1(f) 注記 ✅

### 1.3 MMCSS アフィニティ非干渉の確認

Plan §11.1#8 では MMCSS タスクレジストリの `Affinity` (REG_DWORD) がデフォルト `0x00` または `0xFFFFFFFF` で「affinity を使用しない」ことを示すと主張。
これは MMCSS がアフィニティを変更しないことを意味し、`AvSetMmThreadCharacteristics` 後に `SetThreadAffinityMask` を呼ぶ設計が安全である根拠となる。
Microsoft の MMCSS タスクレジストリ仕様と整合する。

---

## 2. ソースコード検証結果

### 2.1 ThreadAffinityManager.h — 現状の正確な構造

| Plan の主張 | 実コード | 判定 |
|---|---|---|
| `#include <array>, <atomic>, <cstdint>, Windows.h` | ✅ 完全一致 | ✅ |
| `#include <vector>` なし | ✅ なし（追加必須） | ⚠️ Plan §5.1 冒頭注記で明記 |
| `ThreadType` enum に6値、`AudioRealtime` なし | ✅ 確認 | ✅ |
| `ThreadAffinityMasks` に6フィールド | ✅ 確認 | ✅ |
| `applyCurrentThreadPolicy` に default なし | ✅ 確認 (switchに6case、defaultなし) | ✅ |
| `getEvalWorkerMask` は private | ✅ 確認 | ✅ |
| `initialized_` atomic フラグ制御 | ✅ 確認 | ✅ |

### 2.2 AudioEngine.Init.cpp — 行番号・順序の確認

| Plan 記述 | 実コード行 | 実際の内容 | 判定 |
|---|---|---|---|
| L82 `initWorkerThread()` | **L82** に `initWorkerThread();` | ✅ 一致 |
| L87-99 ハードコードmasks | **L87** `{` 〜 **L99** `}` | ✅ 一致 |
| L97 `affinityManager.initialize(masks)` | **L97** ✅ | ✅ |
| L98 diagLog | **L98** ✅ | ✅ |
| **L82 < L97 順序バグ** | initWorkerThread() → m_workerThread.start() → WorkerThread::run() → applyCurrentThreadPolicy(Worker) の時点で initialized_==false → short-circuit | ❌ **バグ確認** (Plan v7 H7) |
| L106 `affinityManager.applyCurrentThreadPolicy(ThreadType::Worker)` | **L106** (initWorkerThread関数内) | ✅ (Message Thread に適用されるが、Worker Thread は未適用のまま) |

### 2.3 AudioEngine.Timer.cpp — applyMmcssPriority() 構造

| Plan 記述 | 実コード行 | 判定 |
|---|---|---|
| 関数開始 L214 | `void AudioEngine::applyMmcssPriority() noexcept` (L214) | ✅ |
| `AvSetMmThreadCharacteristicsA("Pro Audio", &taskIndex)` | L225 | ✅ |
| `AvSetMmThreadPriority(hTask, AVRT_PRIORITY_CRITICAL)` | L227 | ✅ |
| NativeRT フォールバック (SetPriorityClass + SetThreadPriority) | L248-255 | ✅ |
| 関数末尾 L272 (closing brace) | 末尾 `}` (L272) | ✅ |
| 挿入位置: L272 直後 (関数終了前) | 関数終了 `}` の前 → 正しい | ✅ |

### 2.4 両オーディオパスの MMCSS 実装（Plan v3 修正の確認）

| パス | ファイル | MMCSS 実装 | Plan v3 主張 | 判定 |
|---|---|---|---|---|
| Float | `AudioBlock.cpp:46` | `compareExchangeAtomic(mmcssApplied_, ...)` + `applyMmcssPriority()` | ✅ "済み" | ✅ |
| Double | `BlockDouble.cpp:49` | `compareExchangeAtomic(mmcssApplied_, ...)` + `applyMmcssPriority()` | ✅ "済み" | ✅ |

**重要**: Plan v1/v2 では BlockDouble.cpp に MMCSS が「ない（バグ）」と記載されていたが、
v3 で撤回・修正済み。実コード確認により **v3 修正は正しい**。

### 2.5 WorkerThread.cpp — バグの直接確認

WorkerThread.cpp L55-59:
```cpp
void WorkerThread::run()
{
    if (affinityManager != nullptr)
        affinityManager->applyCurrentThreadPolicy(ThreadType::Worker);  // initialized_==false → short-circuit!
```

このコードは `m_workerThread.start()` (AudioEngine.Init.cpp L82 で呼出) により起動されるが、
`affinityManager.initialize(masks)` は L97 で呼ばれるため、WorkerThread が起動する時点では
`initialized_` は `false` のままである。

**確認**: `applyCurrentThreadPolicy` は先頭で `if (!consumeAtomic(initialized_, acquire)) return;` により
即座に short-circuit する。WorkerThread は**二度と** affinity が適用されない（再適用ロジックが存在しない）。

### 2.6 変更不要ファイルの確認

| Plan §8 のファイル | 使用 ThreadType | 確認結果 |
|---|---|---|
| `NoiseShaperLearner.cpp:524` | `LearnerEval` | ✅ `learnerEvalBase` 経由 → 自動適合 |
| `NoiseShaperLearner.cpp:725` | `LearnerMain` | ✅ `learnerMain` 経由 → 自動適合 |
| `DeferredFreeThread.h:152` | `LightBackground` | ✅ Lazy作成 → `initialized_=true` 保証 |
| `LoaderThread.cpp:39` | `HeavyBackground` | ✅ `heavyBackground` 経由 → 自動適合 |
| `ProgressiveUpgradeThread.cpp:76` | `HeavyBackground` | ✅ |
| `WorkerThread.cpp:59` | `Worker` | ✅ `physicalCoreMasks[0]` 経由 → 自動適合 |
| `MainApplication.cpp:146` | `UI` (MessageThread) | ✅ `ui` 経由 → 自動適合 |

### 2.7 CMakeLists.txt 確認

| Plan 主張 | 実CMake | 判定 |
|---|---|---|
| `avrt` リンク済み | L733: `target_link_libraries(... ole32 avrt)` | ✅ |
| `/W4` 設定 | L760 | ✅ |
| `/WX` なし (警告エラー化なし) | `/WX` 未設定 | ✅ |

### 2.8 MainApplication.cpp 確認

| Plan 主張 | 実コード行 | 判定 |
|---|---|---|
| EcoQoS 無効化 (SetProcessInformation) | L79-89 | ✅ |
| `applyMessageThreadPolicy()` | L146 | ✅ |

---

## 3. 計画書の精度評価

### 3.1 完全に正確な主張（検証済み）

- **§5.1(f)** `GetLogicalProcessorInformationEx` API 選択: ✅ 非Ex版では `EfficiencyClass` を取得できないため Ex版必須
- **§5.1(f)** `GroupCount == 1` (MSDN "always 1"): ✅ 完全一致
- **§5.1(f)** `EfficiencyClass` 意味論: ✅ MSDN完全一致
- **§5.1(f)** `KAFFINITY = ULONG_PTR = DWORD_PTR`: ✅
- **§5.1(f)** WOW64 folding (x64ビルドでは問題外): ✅ MSDN確認済み
- **§5.1(h)** `SetThreadAffinityMask` エラー処理: ✅ MSDN "returns zero on failure"
- **§5.3** delete範囲 L87-L99 inclusive: ✅ 実コードと一致
- **§5.4** insert位置 L272直前: ✅ 実コードと一致
- **§5.4** `static_cast<int>(audioMask)` truncation 注釈: ✅ 対象HW(≤0xFFFFFFFF)で安全
- **§6** データフロー図 (v7修正後): ✅ 順序正しい
- **§7** リスク評価7項目: ✅ すべて現実的
- **§10** Step 1-8: ✅ 実装順序妥当
- **§11.2(K) K1** CPU Sets < SetThreadAffinityMask dominance: ✅ MSDN確認済み
- **§11.2(K) K2** EcoQoS 無関係: ✅ 実コード確認
- **§11.2(K) K3** Audio Thread 生成元: ✅ JUCE が Windows Audio Engine 経由 → post-hoc 設定で十分
- **§11.2(K) K4** DeferredFreeThread lazy 作成: ✅ 実コード ConvolverProcessor.Lifecycle.cpp:373 確認
- **§11.2(L) B1-B7** 全7項目: ✅ 実コード確認で問題なし

### 3.2 既に修正済みの誤り（Plan 自身が v2-v11 で修正済み）

| 旧版の誤り | 検出版 | 修正版 | 現在の計画 |
|---|---|---|---|
| 非Ex API 使用 | v1 | v2 | ✅ 正しく Ex版に |
| `EfficiencyClass` 値の誤解釈 | v1 | v2 | ✅ 正しく "!=0=heterogeneous" に |
| BlockDouble.cpp MMCSS「なし（バグ）」 | v1/v2 | v3 | ✅ 撤回＋正しい記述に |
| loop変数 `.size` 括弧欠落 | v5 | v5 | ✅ 修正済み |
| 行番号ズレ (L2300, L222-281) | v5 | v5/v6 | ✅ 修正済み |
| `initWorkerThread()` 順序バグ | v7 | v7 | ✅ 計画に反映＋実装指示追加 |

### 3.3 今回の検証で発見した軽微な不正確さ

**Plan §11.1 #9**: `juce_BasicNativeHeaders.h` のパスが `system/` と記載されているが、
実際のパスは `JUCE/modules/juce_core/native/juce_BasicNativeHeaders.h` である。
**影響なし**（内容は正確）。

---

## 4. 既存バグの確定

### Bug #1: WorkerThread アフィニティ未適用 (AudioEngine.Init.cpp L82 順序問題)

**Status**: v7 で発見・確認済み、計画書 §5.3 / §6 / §7 / §10 に修正反映済み

**原因**: 以下3つの証拠により確定:

1. `AudioEngine.Init.cpp:82` → `initWorkerThread()` が `affinityManager.initialize(masks)` (L97) より先に呼ばれる
2. `initWorkerThread()` → `m_workerThread.start()` → `WorkerThread::run()` (WorkerThread.cpp:55)
3. `WorkerThread::run():59` → `affinityManager->applyCurrentThreadPolicy(Worker)` → `initialized_==false` → short-circuit
4. **再適用ロジックが存在しない** → Worker Thread の affinity は**永久に未適用**

**影響範囲**: Worker Thread (ParameterCommand 処理、DSP再構築の一部)。Worker に割り当てられた `0x01` (CPU 0) が適用されず、OS のデフォルトスケジューリングに委ねられる。

**修正**: Plan v7 の通り、L82 `initWorkerThread();` を `affinityManager.initialize(masks);` (L99直後) に移動。

---

## 5. 総合判定

**本計画書 (v11) は実装可能である。**

### 判定根拠

1. **MSDN公式ドキュメントとの整合性**: 9件のAPI/構造体/概念ドキュメントすべてと矛盾なし
2. **ソースコードとの行単位一致**: 13ファイルの実コードを読み、計画の全行番号・構造・内容が一致することを確認
3. **既存バグの正確な特定**: v7 H7 の順序バグは実コードで再現可能。修正指示は妥当
4. **v11 バグ監査 B1-B7**: 7項目すべて「安全」で新規バグなし
5. **11回の改訂を経た計画**: v1→v11 で全15件の誤りが修正され、現在のバージョンは正確

### 実装時の重要注意点（Plan v11 最終チェックリストより抜粋）

1. **`GetLogicalProcessorInformationEx` を使用**（非Ex版は不可）
2. **`#include <vector>` を `ThreadAffinityManager.h` に追加**
3. **`PROCESSOR_RELATIONSHIP.GroupCount == 1` を前提** (`GroupMask[0].Mask` のみ採取)
4. **`SetThreadAffinityMask` の prevMask==0 エラーハンドリングを診断ログに追加**
5. **★ `initWorkerThread()` を `affinityManager.initialize(masks)` の直後に移動**（v7 重大修正）
6. **`topo.physicalCoreMasks.size()` の括弧を忘れない**
7. **`static_cast<int>(audioMask)` は64bitマスク truncation だが対象HWで安全**
8. **32ビットビルド (WOW64) は対象外**
9. **JUCE が `_WIN32_WINNT = 0x0A00` を強制設定する**
10. **Step 5 の delete 範囲は L87-L99 inclusive**

---

## 6. 付録: 調査に使用したツール

| ツール | 用途 |
|---|---|
| VS Code Read/Replace | ソースファイル読み取り |
| context-mode MCP (`ctx_execute`) | WSL grep/ripgrepによるコード検索 |
| AiDex MCP (`aidex_query`) | コードシンボル検索 |
| Web Fetch | MSDN公式ドキュメント取得 |
| WSL grep/ripgrep | ソースコードパターン検索 |

---

*以上、本計画書 v11 の妥当性検証を完了する。*
