# オーディオスレッド CPU コア固定化 改修計画書

**work64** | 対象: ConvoPeq v0.6.8 | 環境: Windows 11 x64, Intel Gen4〜Gen12+, AVX2 | 2026-07-05

---

## 0. 改訂履歴

| 改訂 | 日付 | 内容 |
|---|---|---|
| v1 | 2026-07-05 | 初版策定 |
| v2 | 2026-07-05 | 妥当性検証に基づく大幅修正 (API仕様誤りの修正、SDK要件の訂正、コンパイル時依存の明記、64論理プロセッサ上限問題への対処) |

> **重要**: 本文档の§5詳細設計は v2 で修正済みです。旧版の `GetLogicalProcessorInformation`、`SYSTEM_LOGICAL_PROCESSOR_INFORMATION(u.Processor.EfficiencyClass)`、EfficiencyClass 値「1=E,2=P」はいずれも誤りであった。正しくは `GetLogicalProcessorInformationEx` + `PROCESSOR_RELATIONSHIP::EfficiencyClass` を使用する。

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

> **重要**: 本§5.1 では `#include <vector>` の追加が必要。既存 `ThreadAffinityManager.h` は `<array>`, `<atomic>`, `<cstdint>`, `Windows.h` のみを include。`std::vector<DWORD_PTR>` を `CoreTopology` で使うため、ファイル冒頭の include ブロックに `#include <vector>` を追加すること。`<memory>` は `detectCoreTopology()` 内で `std::vector<BYTE>` を使えば `std::unique_ptr` 不要で済むため不要。

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
    bool hasHeterogeneousArchitecture = false; // 全コアのEfficiencyClassが同一値(通常0)でない場合 true
    std::vector<DWORD_PTR> physicalCoreMasks;  // 各物理コアの論理プロセッサ集合マスク（SMT兄弟含む、単一プロセッサグループ前提）
};
```

**前提**: プロセッサグループは単一（全論理プロセッサ数 <= 64、第4世代〜第12世代のシングルソケット環境で常に満たされる）。マルチソケットNUMAや192コア workstation では `GROUP_AFFINITY.Group` 番号も保持する拡張が必要だが、本計画の対象環境外のため非対応。

#### (f) detectCoreTopology() 静的メソッド（新規）

**API**: `GetLogicalProcessorInformationEx`（非Ex版は `EfficiencyClass` を取得できないため使用不可）

```cpp
static CoreTopology detectCoreTopology() noexcept;
```

**実装擬似コード**:

```cpp
CoreTopology ThreadAffinityManager::detectCoreTopology() noexcept
{
    CoreTopology topo;
#ifdef _WIN32
    // 1. RelationshipType = RelationProcessorCore で物理コア情報のみ取得
    DWORD bufLen = 0;
    if (!::GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bufLen)
        && ::GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        // API 全体が未対応（Vista以前）→ フォールバック
        // GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) で論理CPU数を取得し、
        // SMT無効扱いで物理コアマスク = 1ビットずつ手動構築
        DWORD nLogical = ::GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
        topo.physicalCoreCount = static_cast<int>(nLogical);
        topo.hasHeterogeneousArchitecture = false;
        for (DWORD i = 0; i < nLogical; ++i)
            topo.physicalCoreMasks.push_back(DWORD_PTR{1} << i);
        return topo;
    }

    // 2. バッファ確保・取得
    std::vector<BYTE> buf(bufLen);
    if (!::GetLogicalProcessorInformationEx(RelationProcessorCore,
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
        &bufLen)) {
        // 失敗時はフォールバック（同上）
        return topo;
    }

    // 3. 可変長レコードを走査
    DWORD offset = 0;
    std::vector<BYTE> efficiencyClasses;
    while (offset < bufLen) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + offset);
        if (info->Relationship == RelationProcessorCore) {
            const auto& proc = info->Processor;
            // GroupCount は物理コアでは常に 1（SDK 仕様）
            // 単一グループ前提のため GroupMask[0].Mask のみ採取
            if (proc.GroupCount >= 1) {
                topo.physicalCoreMasks.push_back(proc.GroupMask[0].Mask);
                efficiencyClasses.push_back(proc.EfficiencyClass);
            }
        }
        offset += info->Size;  // ★ 非Ex版と異なり可変長レコードのため Size で進める
    }

    topo.physicalCoreCount = static_cast<int>(topo.physicalCoreMasks.size);

    // 4. P/E混在判定: 全コアの EfficiencyClass が同一か検査
    if (topo.physicalCoreCount > 1) {
        const BYTE first = efficiencyClasses.empty() ? 0 : efficiencyClasses[0];
        for (BYTE ec : efficiencyClasses) {
            if (ec != first) {
                topo.hasHeterogeneousArchitecture = true;
                break;
            }
        }
    }
#endif
    return topo;
}
```

**API仕様の要点**（検証により確認）:
- `GetLogicalProcessorInformationEx` は **Windows 7+** で利用可能
- `PROCESSOR_RELATIONSHIP::EfficiencyClass` は **Windows 10+** で有効（それ以前のOSでは常に0）
- `EfficiencyClass == 0` は homogeneous（対称コア環境）を示す
- `EfficiencyClass != 0` は heterogeneous（P/E混在）を示す。Eコアが低い値、Pコアが高い値
- `RelationProcessorCore` で取得した各レコードは **物理コア1つに対応**。`GroupMask[].Mask` に SMT兄弟 を含む全論理プロセッサのビットマスクが格納される。SMT兄弟同一グループ化は naturally satisfied
- `GetLogicalProcessorInformation`（非Ex）は `EfficiencyClass` を取得できず、また 1グループ当たり 64 論理プロセッサまでの制限あり。`GetLogicalProcessorInformationEx` を使うことで両者を回避

**SDK要件**: `_WIN32_WINNT >= 0x0601` (Win7)。ConvoPeq の CMake は明示指定しないが、Win11 SDK は `_WIN32_WINNT = 0x0A00` (Win10) を規定値とするため問題なし。`PROCESSOR_RELATIONSHIP` 構造体の `EfficiencyClass` フィールドは Win10 SDK 以降で定義される。

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

---

## 11. 妥当性検証結果（2026-07-05実施）

本計画の妥当性を実際のコードベースと Windows API 仕様書で検証した結果を以下に示す。

### 11.1 検証対象と結果

| # | 検証項目 | 結果 | 備考 |
|---|---|---|---|
| 1 | `ThreadAffinityManager.h` 既存実装の正確な構造 | ✅ OK | 計画記載の enum/struct/メソッド配置と一致。`getEvalWorkerMask` は private、`applyCurrentThreadPolicy` の switch は未網羅（default なし）→ `AudioRealtime` 追加で警告なし |
| 2 | `AudioEngine.Init.cpp:87-99` のマスク初期化箇所 | ✅ OK | 行番号・文脈完全一致。`diagLog` は無名名前空間の関数。`affinityManager.initialize()` の後の `initWorkerThread()` 内 `applyCurrentThreadPolicy(Worker)` はそのまま動作 |
| 3 | `AudioEngine.Timer.cpp:222-281` の `applyMmcssPriority()` 構造 | ✅ OK | L280 が MMCSS ブロック終了の `}`、L281 が関数終了の `}`。両者の間にアフィニティ適用コードを挿入する方針で正確。`hasHeterogeneousCores_` と `affinityManager` は `AudioEngine` メンバなので、Timer.cpp が `AudioEngine.h` を L2 で include しているためアクセス可能 |
| 4 | `BlockDouble.cpp` の MMCSS 欠落 | ✅ バグ確定 | L1-16 に `avrt.h` の include なし。`AudioBlock.cpp` L42-48 と対称的な初回コールバック用ブロックが存在しない。`static std::atomic<bool>` は関数ローカル static のため、Float/Double パスで共有されない。本計画で追加修正する価値あり。`AudioEngine.h` を include していれば `applyMmcssPriority()` 宣言 visible。`avrt.h` は `Timer.cpp` 側で既に include されリンクも `avrt.lib` で解決済み（CMake L733）のため `BlockDouble.cpp` 側に `avrt.h` 追加不要 |
| 5 | `AudioEngine.h` L2300 の `affinityManager` 宣言位置 | ✅ OK | 直後に `hasHeterogeneousCores_` を宣言する方針で妥当。`ThreadAffinityManager.h` は L85 で include 済み |
| 6 | `GetLogicalProcessorInformation` API仕様 | ❌ 誤り→修正済み | 非Ex版は `SYSTEM_LOGICAL_PROCESSOR_INFORMATION` 共用体に `EfficiencyClass` フィールドなし。`ProcessorCore.Flags` (BYTE型) のみ。P/E検出には **`GetLogicalProcessorInformationEx`** + `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX.Processor.EfficiencyClass` が必須。本計画書§5.1(f) で修正済み |
| 7 | `EfficiencyClass` 値の意味 | ❌ 誤り→修正済み | 旧記載「1=E, 2=P」は誤り。正しくは **`EfficiencyClass == 0` が homogeneous、`!= 0` が heterogeneous**。Eコアが低い値、Pコアが高い値。対称環境（第4世代等）では全コア `0`。Win10+ で有効なフィールド |
| 8 | MMCSS がアフィニティに触れないこと | ✅ OK | MSDN MMCSS ドキュメントで `Tasks\Pro Audio` レジストリ値に `Affinity` (REG_DWORD) が存在。デフォルト `0x00` または `0xFFFFFFFF` は「affinity を使用しない」を示す。MMCSS は優先度ブーストのみでアフィニティは変更しない。`AvSetMmThreadCharacteristics("Pro Audio",...)` 後に `SetThreadAffinityMask` を呼ぶ設計は安全で、MMCSS 設定を上書きしない |
| 9 | Windows SDK EfficiencyClass 互換性 | ✅ OK | JUCE 8.0.12 + Win11 SDK 前提。`_WIN32_WINNT` は CMake で明示指定しないが、Win11 SDK は `0x0A00` (Win10) に設定。`GetLogicalProcessorInformationEx` は Win7+ (`_WIN32_WINNT >= 0x0601`)、`EfficiencyClass` は Win10+ で有効。両者とも充足される |
| 10 | 64論理プロセッサ上限問題 | ✅ 対処済み | 非Ex版 `GetLogicalProcessorInformation` は「呼び出しスレッドが属する単一グループ内の最大64論理プロセッサ」のみ取得。Ex版は全グループを取得。本計画は Ex版を使うため問題ない。ただし計画の前提（単一プロセッサグループ、<=64論理CPU）を明記済み |
| 11 | `NoiseShaperLearner.cpp` の `getEvalWorkerMask` 自動適合 | ✅ OK | L524 で `ThreadType::LearnerEval, workerIndex` を呼出。`learnerEvalBase = nonAudioMask` (例: 4C8T で 0x77 = bits 0,1,2,4,5,6 で 6ビット) となる。`getEvalWorkerMask` は `bits[]` 配列に立っているビットを順に格納し `workerIndex % count` でラウンドロビン。eval ワーカー数が 6 以下なら 1ワーカー=1ビットに自然分散。変更不要 |
| 12 | `<vector>` ヘッダ追加必要性 | ⚠️ 計画に明記済み | `ThreadAffinityManager.h` 既存 include は `<array>`, `<atomic>`, `<cstdint>`, `Windows.h`。`CoreTopology::physicalCoreMasks` と `detectCoreTopology()` 内バッファで `std::vector` を使用するため `#include <vector>` 追加必須。§5.1 の冒頭注記に明記済み |
| 13 | `build.bat` / CMake でのビルド検証可能性 | ✅ OK | CMake L732-734 で `avrt` ライブラリ既にリンク済み。`GetLogicalProcessorInformationEx` は `Kernel32.lib` に含まれ、既存の Win32 リンクで充足。新規ライブラリ追加不要。`build.bat` Release/Debug でビルド可能 |

### 11.2 検証で発見された追加事項

#### (A) プロセッサグループ関連

本計画は**単一プロセッサグループ（全論理プロセッサ <= 64）**を前提とする。対象環境（第4世代 4C8T、第12世代 8P+8E = 16C32T）では常に満たされる。マルチソケットNUMA や Intel Xeon W-3175N (28C56T, 単一グループ) 等は問題ないが、AMD Threadripper 3990X (64C128T, 2グループ) やサーバー NUMA システムでは `GROUP_AFFINITY.Group` 番号の保持と `SetThreadGroupAffinity` API への拡張が必要。本計画の対象外。

#### (B) `BlockDouble.cpp` MMCSS 欠落の修正位置

`AudioEngine.Processing.BlockDouble.cpp` L18-35 を確認:
- L20-25: `lifecycle != EngineLifecycleState::Prepared` の early return
- L27-32: `isShutdownInProgress()` の early return
- L34-35: `numSamples` 取得と `callbackIndex` 開始

`applyMmcssPriority()` の追加ブロックは L33 と L34 の間（`isShutdownInProgress()` チェック直後、`numSamples` 取得前）に挿入するのが最も `AudioBlock.cpp:42-48` と対称な位置。本計画 §5.5 で該当位置を記載済み。

#### (C) `hasHeterogeneousCores_` のフォールバック

`detectCoreTopology()` が API 失敗で空の `CoreTopology` を返した場合、`physicalCoreCount == 0` となり `computeSymmetricMasks()` は `N < 2` の早期 return で `ThreadAffinityMasks{}`（全ゼロ）を返す。この場合 `hasHeterogeneousCores_ = false` だが `audioMask == 0` のため `applyMmcssPriority()` 末尾の `if (audioMask != 0)` でスキップされ、安全にアフィニティ未設定となる。破綻なし。

#### (D) `applyCurrentThreadPolicy` への `AudioRealtime` 追加に関するコンパイル安全性

`ThreadType` enum に `AudioRealtime` を追加した場合、`applyCurrentThreadPolicy()` の `switch(type)` に default ケースが存在しない（現行コード確認済み）。MSVC は enum 網羅性チェックで `/WX` がない限り警告 C4062 を出すがエラーにしない。`/WX` が CMake で設定されているか確認推奨。設定されている場合は `[[fallthrough]]` なしで default を省略するとビルド失敗する可能性。`AudioRealtime` ケースに `return;` を明示し、`SetThreadPriority` 呼出をスキップする設計で対応済み。

### 11.3 結論

本計画は妥当である。ただし、実装にあたり以下の点を遵守すること:

1. **`GetLogicalProcessorInformationEx` を使用すること**（非Ex版は不可）
2. **`#include <vector>` を `ThreadAffinityManager.h` に追加すること**
3. **`BlockDouble.cpp` への MMCSS ブロック追加は `applyMmcssPriority()` 呼出のみ**（`avrt.h` include 不要）
4. **プロセッサグループが複数にまたがる環境（>= 65論理プロセッサ、マルチソケットNUMA）は対象外**と明記
5. **CMake の `/WX` 設定の有無を実装時に確認**すること