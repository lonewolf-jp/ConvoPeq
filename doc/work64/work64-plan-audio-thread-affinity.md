# オーディオスレッド CPU コア固定化 改修計画書

**work64** | 対象: ConvoPeq v0.6.8 | 環境: Windows 11 x64, Intel Gen4〜Gen12+, AVX2 | 2026-07-05

---

## 1. 目的

オーディオ DSP コールバックスレッドと非オーディオスレッド（UI, 再構築, IRローダー,
CMA-ES学習, ガベージコレクション）の物理コア競合を排除し、
オーディオリアルタイム性能のレイテンシジッタを低減する。

---

## 2. 設計方針

| 条件 | 戦略 |
|---|---|
| P/Eコア非搭載（対称コア環境, 例: 第4世代 4C8T） | Audio = 末尾1物理コアに SetThreadAffinityMask で固定。非Audio = 最初と末尾以外の全物理コアを共有 |
| P/Eコア搭載（非対称環境, 例: 第12世代 8P+8E） | アフィニティ未設定。MMCSS Deadline QoS（既存 AvSetMmThreadCharacteristics("Pro Audio",...) ）に委任。OSが自動的に最速Pコアへスケジュール |
| SMT / Hyper-Threading | 物理コア単位で分割。同一物理コアの全論理スレッド兄弟を同一グループに含める |
| マスク計算 | GetLogicalProcessorInformation による完全自動検出・自動計算。設定ファイル・手動オーバーライドなし |
| コア数 | 動的。2コア〜多コアまで自動適合（N<2 はアフィニティ無効） |

---

## 3. 現状分析

### 3.1 既存の ThreadAffinityManager

`src/core/ThreadAffinityManager.h` 実装済み。`AudioEngine.Init.cpp:87-99` でハードコード:

```
Worker          = 0x01 (CPU 0)
LearnerMain     = 0x02 (CPU 1)
LearnerEvalBase = 0x04 (CPU 2)
HeavyBackground = 0x08 (CPU 3)
LightBackground = 0x0F (全CPU)
UI              = 0x0F (全CPU)
```

AudioRealtime 型・フィールドなし → オーディオスレッドにアフィニティ設定一切なし。

### 3.2 オーディオスレッドエントリポイント

| パス | ファイル | MMCSS初回呼出 |
|---|---|---|
| Float | `AudioEngine.Processing.AudioBlock.cpp:42-48` | あり |
| Double | `AudioEngine.Processing.BlockDouble.cpp` | **なし（バグ）** |

`BlockDouble.cpp` に `applyMmcssPriority()` 呼び出しが存在せず、
初回コールバックが double パス経由だった場合 MMCSS が適用されない。

### 3.3 applyMmcssPriority() (AudioEngine.Timer.cpp:222-281)

- `useMmcssPriority=true`: AvSetMmThreadCharacteristics("Pro Audio") +
  AvSetMmThreadPriority(AVRT_PRIORITY_CRITICAL) → Windows 11 が Deadline QoS を付与
- `useMmcssPriority=false`: SetPriorityClass(REALTIME_PRIORITY_CLASS) +
  SetThreadPriority(THREAD_PRIORITY_TIME_CRITICAL) (NativeRTフォールバック)

### 3.4 MainApplication.cpp の既存 EcoQoS 無効化

L79-89: SetProcessInformation + PROCESS_POWER_THROTTLING でプロセス全体の
EcoQoS を無効化（StateMask=0 = HighQoS）。このコードは維持する。

---

## 4. 変更ファイル一覧

| # | ファイル | 種別 | 変更量 |
|---|---|---|---|
| 1 | `src/core/ThreadAffinityManager.h` | 大幅拡張 | +~140行 |
| 2 | `src/audioengine/AudioEngine.h` | 小変更 | +2行 |
| 3 | `src/audioengine/AudioEngine.Init.cpp` | 中変更 | 1ブロック差替(~20行) |
| 4 | `src/audioengine/AudioEngine.Timer.cpp` | 中変更 | +~15行 |
| 5 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 小変更 | +8行 |

自動適合（変更不要）: NoiseShaperLearner.cpp, DeferredFreeThread.h,
LoaderThread.cpp, ProgressiveUpgradeThread.cpp, WorkerThread.cpp,
MainApplication.cpp

---

## 5. 詳細設計

### 5.1 ThreadAffinityManager.h 拡張

#### (a) ThreadType 列挙型

```cpp
enum class ThreadType {
    Worker, LearnerMain, LearnerEval, HeavyBackground,
    LightBackground, UI,
    AudioRealtime  // ★追加
};
```

#### (b) ThreadAffinityMasks 構造体

heavyBackground と lightBackground の間に audioRealtime を挿入:

```cpp
struct ThreadAffinityMasks {
    DWORD_PTR worker = 0;
    DWORD_PTR learnerMain = 0;
    DWORD_PTR learnerEvalBase = 0;
    DWORD_PTR heavyBackground = 0;
    DWORD_PTR audioRealtime = 0;  // ★追加
    DWORD_PTR lightBackground = 0;
    DWORD_PTR ui = 0;
};
```

#### (c) applyCurrentThreadPolicy() AudioRealtime分岐

UI ケースの直後に追加。MMCSS が優先度管理済みのため SetThreadPriority をスキップ:

```cpp
case ThreadType::AudioRealtime:
    mask = masks_.audioRealtime;
    if (mask != 0)
        ::SetThreadAffinityMask(::GetCurrentThread(), mask);
    return; // ★早期リターン: 優先度設定スキップ
```

#### (d) アクセサ

```cpp
[[nodiscard]] DWORD_PTR getAudioRealtimeMask() const noexcept {
    return masks_.audioRealtime;
}
```

#### (e) CoreTopology 構造体（新規）

```cpp
struct CoreTopology {
    int physicalCoreCount = 0;
    bool hasHeterogeneousArchitecture = false;
    std::vector<DWORD_PTR> physicalCoreMasks;
};
```

#### (f) detectCoreTopology() 静的メソッド（新規）

GetLogicalProcessorInformation で物理コア情報収集。
EfficiencyClass（Win10 1709+）の均一性で P/E 混在判定。

フォールバック: API 失敗時 → GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)
で論理プロセッサ数取得、1ビットずつ physicalCoreMasks 手動構築（SMT無効扱い＝安全側）。

#### (g) computeSymmetricMasks() 静的メソッド（新規）

対称コア環境専用。N = physicalCoreCount:

```
若 N < 2 → ThreadAffinityMasks{}（全ゼロ、アフィニティ無効）

audioMask      = physicalCoreMasks[N-1]
nonAudioMask   = physicalCoreMasks[0] | ... | physicalCoreMasks[N-2]

masks_.audioRealtime   = audioMask
masks_.worker          = physicalCoreMasks[0]
masks_.learnerMain     = physicalCoreMasks[min(1, N-2)]
masks_.learnerEvalBase = nonAudioMask
masks_.heavyBackground = nonAudioMask
masks_.lightBackground = nonAudioMask
masks_.ui              = nonAudioMask
```

**4C8T 計算例**:

```
物理コアマスク: CPU[0]=0x11 CPU[1]=0x22 CPU[2]=0x44 CPU[3]=0x88
audioMask    = 0x88
nonAudioMask = 0x77

worker          = 0x11 (物理コア0)
learnerMain     = 0x22 (物理コア1)
learnerEvalBase = 0x77 (物理コア0-2)
heavyBackground = 0x77
lightBackground = 0x77
ui              = 0x77
audioRealtime   = 0x88 (物理コア3専用)
```

**エッジケース**:

| 物理コア数 | audioMask | nonAudioMask | 備考 |
|---|---|---|---|
| 1 | - | - | 全ゼロ |
| 2 | CPU[1] | CPU[0] | 全非AudioスレッドがCPU[0]集中 |
| 4 | CPU[3] | CPU[0-2] | バランス良好 |
| 6 | CPU[5] | CPU[0-4] | |
| 8 | CPU[7] | CPU[0-6] | |
| 12 | CPU[11] | CPU[0-10] | |

### 5.2 AudioEngine.h — hasHeterogeneousCores_ 追加

ThreadAffinityManager affinityManager; (L2300) の直後:

```cpp
bool hasHeterogeneousCores_ = false; // ★ P/E混在フラグ
```

### 5.3 AudioEngine.Init.cpp — マスク初期化ロジック差替

**削除**: ハードコードマスク設定ブロック (L87-98)

**新規**:

```cpp
{
    auto topo = ThreadAffinityManager::detectCoreTopology();

    if (topo.hasHeterogeneousArchitecture) {
        // P/E混在 → アフィニティ未設定、MMCSS Deadline QoS に委任
        ThreadAffinityMasks noAffinity{};
        affinityManager.initialize(noAffinity);
        hasHeterogeneousCores_ = true;
        diagLog("[AFFINITY] P/E heterogeneous cores (N="
                + juce::String(topo.physicalCoreCount)
                + "). Affinity disabled — MMCSS Deadline QoS active.");
    } else {
        // 対称コア → 末尾1物理コアをAudio専用に
        auto masks = ThreadAffinityManager::computeSymmetricMasks(topo);
        affinityManager.initialize(masks);
        hasHeterogeneousCores_ = false;
        diagLog("[AFFINITY] Symmetric cores (N="
                + juce::String(topo.physicalCoreCount)
                + "). Audio pinned to last physical core.");
    }
}
```

### 5.4 AudioEngine.Timer.cpp — applyMmcssPriority() 拡張

L280 の `}` 直後（関数末尾の `}` の前）に追加:

```cpp
    // ★ [work64] Audioスレッド CPUアフィニティ固定（対称コア環境のみ）
    if (!hasHeterogeneousCores_) {
        const DWORD_PTR audioMask = affinityManager.getAudioRealtimeMask();
        if (audioMask != 0) {
            const DWORD_PTR prevMask = ::SetThreadAffinityMask(
                ::GetCurrentThread(), audioMask);
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
            diagLog("[AFFINITY] AudioThread pinned mask=0x"
                    + juce::String::toHexString(static_cast<int>(audioMask))
                    + " prev=0x" + juce::String::toHexString(static_cast<int64>(prevMask)));
#endif
        }
    }
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    else {
        diagLog("[AFFINITY] P/E cores: AudioThread affinity skipped (MMCSS Deadline QoS)");
    }
#endif
```

### 5.5 AudioEngine.Processing.BlockDouble.cpp — MMCSS初回呼出追加

lifecycle チェック後、numSamples 取得前に追加:

```cpp
    // ★ [work64] MMCSS初回コール（doubleパス独立）
    {
        static std::atomic<bool> s_mmcssDoneDouble{false};
        bool expected = false;
        if (s_mmcssDoneDouble.compare_exchange_strong(expected, true,
                                                      std::memory_order_acq_rel)) {
            applyMmcssPriority();
        }
    }
```

---

## 6. データフロー（対称コア環境・起動〜オーディオ開始）

```
MainApplication::initialise()
  +-> AudioEngine::initialize()
        +-> detectCoreTopology() → CoreTopology
        +-> computeSymmetricMasks(topo) → ThreadAffinityMasks
        +-> affinityManager.initialize(masks)
        +-> hasHeterogeneousCores_ = false
        +-> rebuildThread開始 → applyCurrentThreadPolicy(HeavyBackground)
              → SetThreadAffinityMask(nonAudioMask)
        +-> WorkerThread開始 → applyCurrentThreadPolicy(Worker)
              → SetThreadAffinityMask(物理コア0)
        +-> Timer開始
  +-> getAffinityManager().applyMessageThreadPolicy()
        → SetThreadAffinityMask(nonAudioMask)

JUCE AudioProcessorPlayer → AudioEngineProcessor::processBlock()
  +-> getNextAudioBlock() [Float] または processBlockDouble() [Double]
        +-> [初回のみ] applyMmcssPriority()
              +-> AvSetMmThreadCharacteristics("Pro Audio", ...)
                    → OS が Deadline QoS 付与
              +-> SetThreadAffinityMask(audioMask)
                    → 末尾物理コアに固定
              [NativeRTパス]
              +-> SetPriorityClass(REALTIME) + SetThreadPriority(TIME_CRITICAL)
              +-> SetThreadAffinityMask(audioMask)
```

---

## 7. リスク評価

| リスク | 深刻度 | 対策 |
|---|---|---|
| 古いSDKでEfficiencyClass未対応 → P/E検出不可 | 低 | 対称環境と誤判定 → アフィニティ固定モード（実害なし） |
| BlockDouble.cpp にMMCSS呼出なし（既存バグ） | 中 | 本計画で修正 |
| 対称N=2でWorker/Learner/UIがCPU[0]集中 | 低 | デュアルコアでは不可避。Audio1コア専用化の利益が勝る |
| GetLogicalProcessorInformation API失敗 | 低 | GetActiveProcessorCount フォールバック |
| MMCSS後にSetThreadAffinityMask | 極低 | MMCSSはアフィニティに触れない（SDKドキュメント確認済） |

---

## 8. 変更不要ファイル（自動適合）

| ファイル | 使用マスク | 自動適合理由 |
|---|---|---|
| NoiseShaperLearner.cpp L524,725 | LearnerEval, LearnerMain | learnerEvalBase が複数コア → getEvalWorkerMask() が自然分散 |
| DeferredFreeThread.h L152 | LightBackground | nonAudioMask 自動反映 |
| LoaderThread.cpp L39 | HeavyBackground | nonAudioMask 自動反映 |
| ProgressiveUpgradeThread.cpp L76 | HeavyBackground | 同上 |
| WorkerThread.cpp L59 | Worker | physicalCoreMasks[0] 自動反映 |
| MainApplication.cpp L146 | UI (MessageThread) | nonAudioMask 自動反映 |

---

## 9. 検証計画

| 検証項目 | 環境 | 方法 | 合格基準 |
|---|---|---|---|
| 対称4CでAudio固定 | 第4世代4C8T | Process Explorer + 診断ログ | AudioがCPU3(論理3,7)のみ |
| UIスレッド非Audioコア | 同上 | Process Explorer | CPU0-2範囲内 |
| BGスレッド分散 | 同上 | Process Explorer | DeferredFree等がCPU0-2 |
| P/E環境でQoS委任 | 第12世代+ | 診断ログ `[AFFINITY] P/E heterogeneous` | アフィニティ未設定 |
| BlockDouble MMCSS適用 | 全環境 | `[MMCSS]`ログ | doubleパス初回出力 |
| ドロップアウト変化 | 第4世代 | diagTickDropped前後比較 | 低減or同等 |
| ビルド破壊なし | 全環境 | build.bat Release/Debug | ビルドエラーゼロ |

---

## 10. 実装順序

| Step | 内容 | ファイル | 確認 |
|---|---|---|---|
| 1 | AudioRealtime型+フィールド+分岐+アクセサ | ThreadAffinityManager.h | コンパイル |
| 2 | CoreTopology+detectCoreTopology+computeSymmetricMasks | ThreadAffinityManager.h | コンパイル |
| 3 | hasHeterogeneousCores_メンバ追加 | AudioEngine.h L2300付近 | コンパイル |
| 4 | initialize()内マスク設定を動的計算に置換 | AudioEngine.Init.cpp L87-99 | コンパイル |
| 5 | BlockDouble.cppにMMCSS初回呼出ブロック追加 | BlockDouble.cpp L19付近 | コンパイル |
| 6 | applyMmcssPriority()末尾にアフィニティ固定 | AudioEngine.Timer.cpp L280-281間 | コンパイル |
| 7 | Release+DIAGNOSTICSビルド・実行検証 | build.bat | 診断ログ+ProcessExplorer |
| 8 | CTest 全テストスイート通過確認 | ctest | 全PASS |