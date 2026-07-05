# オーディオスレッド CPU コア固定化 改修計画書

**work64** | 対象: ConvoPeq v0.6.8 | 環境: Windows 11 x64, Intel Gen4〜Gen12+, AVX2 | 2026-07-05

---

## 0. 改訂履歴

| 改訂 | 日付 | 内容 |
|---|---|---|
| v1 | 2026-07-05 | 初版策定 |
| v2 | 2026-07-05 | 妥当性検証に基づく大幅修正 (API仕様誤りの修正、SDK要件の訂正、コンパイル時依存の明記、64論理プロセッサ上限問題への対処) |
| v3 | 2026-07-05 | WSL検証で発見: BlockDouble.cpp MMCSSは既に実装済み（mmcssApplied_使用）。両パス unified。plan §5.5は誤り。Plan §3.2「BlockDouble.cpp なし（バグ）」を取消し。 |
| v4 | 2026-07-05 | MSDN 公式ドキュメント6件 + WSLシミュレーション9テストでプロセッサアフィニティマスクの全仕様を検証。`PROCESSOR_RELATIONSHIP.GroupCount==1` (MSDN明記)、`KAFFINITY==DWORD_PTR`、`SetThreadAffinityMask` Win11 プライマリグループ制約、`EfficiencyClass` 意味論を公式仕様と突合し修正。`prevMask==0` エラー処理の推奨追加。
| v5 | 2026-07-05 | MSDN公式8件 + WSL 15テストで最終検証。`topo.physicalCoreMasks.size` 括弧欠落バグ修正、行番号ズレ修正(L2301, Timer.cpp L214-272)、WOW64 folding明記、6C期待値修正。`Affinity` REG_DWORD = 0x00/0xFFFFFFFF は MMCSS公式ドキュメントで確認済み。 |
| v6 | 2026-07-05 | WSL S6 15テストで詳細検証。3点の文書精緻化: ①§5.4 `static_cast<int>(audioMask)` は int=32bitで64bitマスク truncation の誤解解消 (対象HWでは <=0xFFFFFFFF なのでOK)。②§5.3 delete範囲 L87-98 → L87-L99 inclusive に修正。③§11.1 #9: `_WIN32_WINNT` は CMake ではなく JUCE `juce_BasicNativeHeaders.h:103-104` で必ず `0x0A00`(Win10) に設定されることを明記。 |
| v7 | 2026-07-05 | **重大バグ発見**: AudioEngine.Init.cpp L82 `initWorkerThread()` が L97 `affinityManager.initialize(masks)` より前に呼ばれている現状。WorkerThread::run() の `applyCurrentThreadPolicy(Worker)` が `initialized_==false` で short-circuit → Worker affinity が永久に未適用。WSL S7 7テストで順序 Race condition 検証。プラン §5.3 で L82 → L99直後 に**移動**必須。§6 データフロー、§10 実装順序、§7 リスク評価 を v7 で更新。 |
| v8 | 2026-07-05 | WSL S8 9テストで Before/After 順序シミュレーション実行。既存バグ (S8-1) と修正後 (S8-2) の挙動差分を pthread で実機確認。S8-3〜S9 まで全 9テスト PASS。 |
| v9 | 2026-07-05 | MSDN 追加調査 3件: `GetProcessAffinityMask`、`SetProcessAffinityMask` (Win11 primary group 制約 — 本計画では不使用のため安全)、`SetThreadIdealProcessor` (preferred processor hint — `SetThreadAffinityMask` のハード固定が dominant、干渉なし)。行番号 L214 微細訂正。§11.2(J) に追加結果を追記。 |
| v10 | 2026-07-05 | deep-dive 検証 4項目: ①MSDN CPU Sets vs affinity mask (K1: restrictive affinity dominant 確認)、②MainApplication.cpp:79-89 EcoQoS 無効化との相互作用 (K2: 独立動作), ③Audio Thread 生成元 (K3: Windows Audio Engine 管理、`PROC_THREAD_ATTRIBUTE_GROUP_AFFINITY` 不要), ④DeferredFreeThread 遅延作成タイミング (K4: v7 修正で自然にカバー)。§11.2(K) を新規追加。 |
| v11 | 2026-07-05 | 新規バグ監査 B1-B7 (実装前最終) を実施。§11.2(L) 追加。B2 (mmcssApplied_ reset/CAS 競合なし), B4 (LearnerEval は learnerEvalBase 使用、audioRealtime マスク独立で影響なし), B5 (子プロセス API 不使用), B6 (nullptr チェック既備), B7 (diagLog 順序 設計上問題なし)。新規バグ発見なし。 |

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

### 3.2 オーディオスレッドエントリポイントと MMCSS 実装状況

| パス | ファイル | MMCSS 実装状況 |
|---|---|---|
| Float | `AudioEngine.Processing.AudioBlock.cpp:40-49` | ✅ 済み。`mmcssApplied_` + `compareExchangeAtomic` 使用 |
| Double | `AudioEngine.Processing.BlockDouble.cpp:43-52` | ✅ 済み。`mmcssApplied_` + `compareExchangeAtomic` 使用 |

> **⚠️ plan v1-v2 の誤り**: v1,v2 の §3.2 と §5.5 は `BlockDouble.cpp` に MMCSS 呼出が「ない（バグ）」と記述していましたが、WSL 検証により **既に `mmcssApplied_` を共有フラグとした MMCSS 呼出が両パスに実装済み**であることが判明しました。`applyMmcssPriority()` の呼出は `AudioEngine.h:2095` に宣言された `std::atomic<bool> mmcssApplied_{false}` を共有し、`PrepareToPlay()` でリセットされる正しい実装です。plan §5.5 の `BlockDouble.cpp` への MMCSS ブロック追加指示は**撤回**します。

### 3.3 applyMmcssPriority() (AudioEngine.Timer.cpp:214-272)

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

**WOW64 folding に関する注記** (MSDN `GetLogicalProcessorInformationEx` ドキュメント): 32ビットプロセスが WOW64 経由で>64プロセッサシステム上の本APIを呼ぶ場合、processors 32-63 の affinity mask が 0-31 の複製として「折りたたまれ」不正になる場合がある。ConvoPeq は **x64 専用ビルド** (`cmake -A x64`) であるため本問題は適用外。ただし32ビットビルドを将来的に追加する場合は再検証が必要。

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
            // MSDN: "If PROCESSOR_RELATIONSHIP represents a processor core,
            //        the GroupCount member is always 1."
            //   → RelationProcessorCore で取得した各レコードは物理コア1つに1対1対応。
            //      GroupCount == 1, GroupMask[0].Mask = その物理コアに属する全論理プロセッサのbitmask。
            //      SMT兄弟はOSが自動的に同一 GroupMask[0].Mask に含めて返す。
            if (proc.GroupCount == 1) {
                // GROUP_AFFINITY.Mask = KAFFINITY = ULONG_PTR = DWORD_PTR (MSDN確認済)
                topo.physicalCoreMasks.push_back(proc.GroupMask[0].Mask);
                efficiencyClasses.push_back(proc.EfficiencyClass);
            }
        }
        offset += info->Size;  // ★ 非Ex版と異なり可変長レコードのため Size で進める
    }

    topo.physicalCoreCount = static_cast<int>(topo.physicalCoreMasks.size());  // ★ size() 括弧必須 (v5修正)

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

**API仕様の要点**（MSDN 公式 + WSL シミュレーション検証により確認）:

### コア情報取得
- `GetLogicalProcessorInformationEx(RelationProcessorCore, ...)` は **Windows 7+** で利用可能
- `RelationProcessorCore` で取得した各レコードは **物理コア1つに1対1対応**。`GroupCount` は **常に 1** (MSDN: "If the PROCESSOR_RELATIONSHIP structure represents a processor core, the GroupCount member is always 1.")
- `GroupMask[0].Mask` にはその物理コアに属する**全論理プロセッサ(SMT兄弟含む)のビットマスク**が格納される。SMT兄弟同一グループ化は OS が自動的に行う
- `GetLogicalProcessorInformation`（非Ex）は `EfficiencyClass` を取得できず、また 1グループ当たり 64 論理プロセッサまでの制限あり。`GetLogicalProcessorInformationEx` を使うことで両者を回避

### アフィニティマスクの型と制約
- `GROUP_AFFINITY.Mask` の型は `KAFFINITY` = `ULONG_PTR` = `DWORD_PTR` (MSDN `winnt.h` struct definition 確認済)
- `SetThreadAffinityMask` (Win11): 64超LPシステムではスレッドの現在のprimary group内のプロセッサをマスクで指定する必要がある (MSDN: "The dwThreadAffinityMask must specify processors in the thread's current primary group.")。本計画の対象全環境（<=64 LP）は単一グループ Group 0 → 制約は自動的に充足される
- `SetThreadAffinityMask` 戻り値が 0 の場合は `GetLastError()` でエラー確認が必要 (MSDN: "If the function fails, the return value is zero.")。ただし audioMask はプロセスアフィニティの部分集合であるため、本実装ではエラーとなることはない。prevMask==0 の診断ログ追加を推奨

### EfficiencyClass
- `PROCESSOR_RELATIONSHIP::EfficiencyClass` は **Windows 10+** で有効（それ以前のOSでは常に0）
- `EfficiencyClass == 0` は homogeneous（対称コア環境）を示す (MSDN: "EfficiencyClass is only nonzero on systems with a heterogeneous set of cores.")
- `EfficiencyClass != 0` は heterogeneous（P/E混在）を示す。Eコアが低い値、Pコアが高い値 (MSDN: "A core with a higher value for the efficiency class has intrinsically greater performance and less efficiency than a core with a lower value")

### Win11 Processor Groups
- Win11/Server 2022 以降、デフォルトで全プロセッサグループをspan (MSDN: "processes and their threads have processor affinities that by default span all processors in the system, across multiple groups")。単一グループ前提が <=64 LP では常に成り立つため影響なし
- `GetLogicalProcessorInformationEx(RelationProcessorCore)` は全グループ内の**全物理コアに対して1レコードずつ**返す。>64 LP のマルチグループシステムでは同一物理コアがグループによって別レコードで表現される可能性がある (MSDN: "returns a PROCESSOR_RELATIONSHIP structure for every active processor core in every processor group")。本計画の対象環境では発生しない

**SDK要件**: `_WIN32_WINNT >= 0x0601` (Win7)。ConvoPeq の CMake は明示指定しないが、Win11 SDK は `_WIN32_WINNT = 0x0A00` (Win10) を規定値とするため問題なし。`PROCESSOR_RELATIONSHIP` 構造体の `EfficiencyClass` フィールドは Win10 SDK 以降で定義される。

#### (h) `SetThreadAffinityMask` エラー処理の推奨

MSDN 仕様: `SetThreadAffinityMask` は失敗時に 0 を返し、`GetLastError()` でエラーコードを取得可能。audioMask はプロセスアフィニティの部分集合であるため、通常の運用では失敗しないが、診断用に prevMask==0 のケースをログ出力することを推奨:

```cpp
const DWORD_PTR prevMask = ::SetThreadAffinityMask(::GetCurrentThread(), audioMask);
if (prevMask == 0) {
    const DWORD err = ::GetLastError();
    diagLog("[AFFINITY] SetThreadAffinityMask FAILED: GetLastError="
            + juce::String(static_cast<int>(err)));
}
```

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

ThreadAffinityManager affinityManager; (L2301) の直後:

```cpp
bool hasHeterogeneousCores_ = false; // ★ P/E混在フラグ
```

### 5.3 AudioEngine.Init.cpp — マスク初期化ロジック差替（順序重要、v7）

**削除**:
1. ハードコードマスク設定ブロック (L87-L99 inclusive、`{` から `}` までブロック全体）
2. **L82 の `initWorkerThread();` 呼び出し**（仕様上は削除ではなく **後述のとおり移動**だが、最終形では initialize() の後に呼び出される形にする）

**★ v7 追加 — WorkerThread 起動と initialize() の順序入替（必須）**:

現状コードは L82 で `initWorkerThread()` が呼ばれ、そのスレッド内で `applyCurrentThreadPolicy(Worker)` が呼ばれるが、この時点ではまだ `affinityManager.initialize(masks)` (L97) が呼ばれていない。`applyCurrentThreadPolicy` は `initialized_ == false` で short-circuit するため、**WorkerThread は一度も affinity を適用されず**、再 apply ロジックもないため affinity 未設定のまま動作する（既存の潜在バグ）。

**修正後の順序**:
```cpp
// タイマー開始 (100ms間隔)
startTimer(100);
timerPeriodMs_ = 100;

// ★ [work64] ThreadAffinityManager 初期化（診断目的→C++ 動的計算化）
//   削除前の L87-99 ブロックを新規コードで置換
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

// ★ [work64] 順序入替（v7）: initialize() の後で WorkerThread を起動
initWorkerThread();  // ← 元の L82 からここへ移動
```

### 5.4 AudioEngine.Timer.cpp — applyMmcssPriority() 拡張

L272 の `}` 直後（関数末尾の `}` の前）に追加。`applyMmcssPriority()` は L214-272。

```cpp
    // ★ [work64] Audioスレッド CPUアフィニティ固定（対称コア環境のみ）
    //   NOTE(v6): audioMask は DWORD_PTR (=64bit on x64). juce::String::toHexString は
    //     int シグネチャ overload のため `static_cast<int>(audioMask)` で 32bit に切詰。
    //     対象HW (4C8T=0xFF, 16C32T=0xFFFFFFFF) では上限 0xFFFFFFFF を超えない → 実用上安全。
    //     より厳密にする場合は juce::String::toHexString<T>(static_cast<int64_t>(audioMask)) を用いる。
    if (!hasHeterogeneousCores_) {
        const DWORD_PTR audioMask = affinityManager.getAudioRealtimeMask();
        if (audioMask != 0) {
            const DWORD_PTR prevMask = ::SetThreadAffinityMask(
                ::GetCurrentThread(), audioMask);
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
            diagLog("[AFFINITY] AudioThread pinned mask=0x"
                    + juce::String::toHexString(static_cast<int>(audioMask))
                    + " prev=0x" + juce::String::toHexString(static_cast<int64_t>(prevMask)));
#endif
        }
    }
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    else {
        diagLog("[AFFINITY] P/E cores: AudioThread affinity skipped (MMCSS Deadline QoS)");
    }
#endif
}
```

### 5.5 ~~AudioEngine.Processing.BlockDouble.cpp — MMCSS初回呼出追加~~ 【撤回 v3】

> **撤回 (v3)**: §3.2 の 注記を参照。`BlockDouble.cpp` L43-52 には既に `mmcssApplied_` + `compareExchangeAtomic` による MMCSS 呼出が実装済みです。本計画書の旧版 §5.5 の `static std::atomic<bool> s_mmcssDoneDouble` ブロック追加指示は**撤回**します。実装は不要。

---

## 6. データフロー（対称コア環境・起動〜オーディオ開始、v7 順序修正）

```
MainApplication::initialise()
  +-> AudioEngine::initialize()
        +-> ★ 初期化順序（v7 必須）
              startTimer(100);        // Timer 開始
              // ↓ affinityManager 初期化（WorkerThread 起動の前に行う）
              +-> detectCoreTopology() → CoreTopology
              +-> computeSymmetricMasks(topo) → ThreadAffinityMasks (or noAffinity{} if P/E)
              +-> affinityManager.initialize(masks)   // initialized_ = true
              +-> hasHeterogeneousCores_ = (true|false)
              +-> ★ v7: initWorkerThread() を L97 後 (= initialize 後) に移動
                    → WorkerThread::run() → applyCurrentThreadPolicy(Worker)
                          → initialized_=true 確認 → SetThreadAffinityMask(物理コア0)
                    ※ 既存バグの場合は initialize() が適用前のため short-circuit になる
              +-> rebuildThread開始 → applyCurrentThreadPolicy(HeavyBackground)
                    → SetThreadAffinityMask(nonAudioMask)
              +-> DeferredFreeThread開始 → applyCurrentThreadPolicy(LightBackground)
                    → SetThreadAffinityMask(nonAudioMask)
              +-> ProgressiveUpgradeThread開始 → applyCurrentThreadPolicy(HeavyBackground)
                    → SetThreadAffinityMask(nonAudioMask)
   +-> getAffinityManager().applyMessageThreadPolicy()  ← MainApplication.cpp L146
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
| **★ v7 追加**: `initWorkerThread()` の呼び出しが `affinityManager.initialize(masks)` より先 (AudioEngine.Init.cpp L82 vs L97) | 中 | 本計画で順序入替: L82削除 + L99直後に移動。現状バグではWorkerThread::run()が`initialized_==false`判定でshort-circuitしaffinity永久未適用 |

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

## 10. 実装順序（v7 更新）

| Step | 内容 | ファイル | 確認 |
|---|---|---|---|
| 1 | AudioRealtime型+フィールド+分岐+アクセサ | ThreadAffinityManager.h | コンパイル |
| 2 | CoreTopology+detectCoreTopology+computeSymmetricMasks | ThreadAffinityManager.h | コンパイル |
| 3 | `getAffinityManager()` 既存アクセサ利用確認 (L1058-1059 既存在) | (追加実装不要、参照確認のみ) | — |
| 4 | hasHeterogeneousCores_メンバ追加 | AudioEngine.h L2301付近 | コンパイル |
| 5 | initialize()内マスク設定を動的計算に置換 + **`initWorkerThread()` を initialize() の「後」に移動** (v7 必須) | AudioEngine.Init.cpp L82(削除)→L99(後) + L87-99(置換) | コンパイル + 順序確認 |
| 6 | applyMmcssPriority()末尾にアフィニティ固定 | AudioEngine.Timer.cpp L272 直後 | コンパイル |
| 7 | Release+DIAGNOSTICSビルド・実行検証 | build.bat | 診断ログ+ProcessExplorer |
| 8 | CTest 全テストスイート通過確認 | ctest | 全PASS |

> **注意**: 旧 Step 5「BlockDouble.cpp MMCSS ブロック追加」は **v3 で撤回**（§3.2注記参照）。両オーディオパスとも既に `mmcssApplied_` 共有フラグで MMCSS 呼出済み。

> **v7 追加**: Step 5 で **順序入替が必須**。WorkerThread 起動（L82）が initialize()（L97）よりも先に来ると、WorkerThread 内の `applyCurrentThreadPolicy(Worker)` が `initialized_==false` で short-circuit し、結果として WorkerThread の affinity は永遠に適用されない。最終形では L82 の `initWorkerThread();` を削除し、`affinityManager.initialize(masks);` の直後（L99 直後）に移動すること。

---

## 11. 妥当性検証結果（2026-07-05実施）

本計画の妥当性を実際のコードベースと Windows API 仕様書で検証した結果を以下に示す。

### 11.1 検証対象と結果

| # | 検証項目 | 結果 | 備考 |
|---|---|---|---|
| 1 | `ThreadAffinityManager.h` 既存実装の正確な構造 | ✅ OK | 計画記載の enum/struct/メソッド配置と一致。`getEvalWorkerMask` は private、`applyCurrentThreadPolicy` の switch は未網羅（default なし）→ `AudioRealtime` 追加で警告なし |
| 2 | `AudioEngine.Init.cpp:87-99` のマスク初期化箇所 | ✅ OK | 行番号・文脈完全一致。`diagLog` は無名名前空間の関数。`affinityManager.initialize()` の後の `initWorkerThread()` 内 `applyCurrentThreadPolicy(Worker)` はそのまま動作 |
| 3 | `AudioEngine.Timer.cpp:215-272` の `applyMmcssPriority()` 構造 | ✅ OK | L272 が MMCSS ブロック終了の `}`、その直後が関数終了の `}`。両者の間にアフィニティ適用コードを挿入する方針で正確（v5で行番号修正）。`hasHeterogeneousCores_` と `affinityManager` は `AudioEngine` メンバなので、Timer.cpp が `AudioEngine.h` を L2 で include しているためアクセス可能 |
| 4 | `BlockDouble.cpp` の MMCSS 実装状態 | ✅ 既に実装済み | L43-52 に `mmcssApplied_` + `compareExchangeAtomic` による MMCSS 呼出が存在。plan v1-v2 の「なし（バグ）」記載は誤りであった。両パス unified。`avrt.h` は `Timer.cpp` 側でリンク済み。`hasHeterogeneousCores_` と `affinityManager` は `AudioEngine` メンバなので、Timer.cpp が `AudioEngine.h` を L2 で include しているためアクセス可能 |
| 5 | `AudioEngine.h` L2300 の `affinityManager` 宣言位置 | ✅ OK | 直後に `hasHeterogeneousCores_` を宣言する方針で妥当。`ThreadAffinityManager.h` は L85 で include 済み |
| 6 | `GetLogicalProcessorInformation` API仕様 | ❌ 誤り→修正済み | 非Ex版は `SYSTEM_LOGICAL_PROCESSOR_INFORMATION` 共用体に `EfficiencyClass` フィールドなし。`ProcessorCore.Flags` (BYTE型) のみ。P/E検出には **`GetLogicalProcessorInformationEx`** + `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX.Processor.EfficiencyClass` が必須。本計画書§5.1(f) で修正済み |
| 7 | `EfficiencyClass` 値の意味 | ❌ 誤り→修正済み | 旧記載「1=E, 2=P」は誤り。正しくは **`EfficiencyClass == 0` が homogeneous、`!= 0` が heterogeneous**。Eコアが低い値、Pコアが高い値。対称環境（第4世代等）では全コア `0`。Win10+ で有効なフィールド |
| 8 | MMCSS がアフィニティに触れないこと | ✅ OK | MSDN MMCSS ドキュメントで `Tasks\Pro Audio` レジストリ値に `Affinity` (REG_DWORD) が存在。デフォルト `0x00` または `0xFFFFFFFF` は「affinity を使用しない」を示す。MMCSS は優先度ブーストのみでアフィニティは変更しない。`AvSetMmThreadCharacteristics("Pro Audio",...)` 後に `SetThreadAffinityMask` を呼ぶ設計は安全で、MMCSS 設定を上書きしない |
| 9 | Windows SDK EfficiencyClass 互換性 | ✅ OK (v6 詳細化) | **JUCE が `juce_BasicNativeHeaders.h:103-104` で `_WIN32_WINNT = _WIN32_WINNT_WIN10` (= `0x0A00`, Win10) を必ず設定する** — CMake 設定に依存せずWin10ベースラインが保証される。`GetLogicalProcessorInformationEx` は Win7+ (`_WIN32_WINNT >= 0x0601`)、`EfficiencyClass` は Win10+ で有効。両者とも充足 |
| 10 | 64論理プロセッサ上限問題 | ✅ 対処済み | 非Ex版 `GetLogicalProcessorInformation` は「呼び出しスレッドが属する単一グループ内の最大64論理プロセッサ」のみ取得。Ex版は全グループを取得。本計画は Ex版を使うため問題ない。ただし計画の前提（単一プロセッサグループ、<=64論理CPU）を明記済み |
| 11 | `NoiseShaperLearner.cpp` の `getEvalWorkerMask` 自動適合 | ✅ OK | L524 で `ThreadType::LearnerEval, workerIndex` を呼出。`learnerEvalBase = nonAudioMask` (例: 4C8T で 0x77 = bits 0,1,2,4,5,6 で 6ビット) となる。`getEvalWorkerMask` は `bits[]` 配列に立っているビットを順に格納し `workerIndex % count` でラウンドロビン。eval ワーカー数が 6 以下なら 1ワーカー=1ビットに自然分散。変更不要 |
| 12 | `<vector>` ヘッダ追加必要性 | ⚠️ 計画に明記済み | `ThreadAffinityManager.h` 既存 include は `<array>`, `<atomic>`, `<cstdint>`, `Windows.h`。`CoreTopology::physicalCoreMasks` と `detectCoreTopology()` 内バッファで `std::vector` を使用するため `#include <vector>` 追加必須。§5.1 の冒頭注記に明記済み |
| 13 | `build.bat` / CMake でのビルド検証可能性 | ✅ OK | CMake L732-734 で `avrt` ライブラリ既にリンク済み。`GetLogicalProcessorInformationEx` は `Kernel32.lib` に含まれ、既存の Win32 リンクで充足。新規ライブラリ追加不要。`build.bat` Release/Debug でビルド可能 |

### 11.2 検証で発見された追加事項

#### (A) プロセッサグループ関連

本計画は**単一プロセッサグループ（全論理プロセッサ <= 64）**を前提とする。対象環境（第4世代 4C8T、第12世代 8P+8E = 16C32T）では常に満たされる。マルチソケットNUMA や Intel Xeon W-3175N (28C56T, 単一グループ) 等は問題ないが、AMD Threadripper 3990X (64C128T, 2グループ) やサーバー NUMA システムでは `GROUP_AFFINITY.Group` 番号の保持と `SetThreadGroupAffinity` API への拡張が必要。本計画の対象外。

#### (B) ~~BlockDouble.cpp MMCSS 欠落の修正位置~~ 【撤回 v3】

> **撤回 (v3)**: §3.2 の 注記を参照。`BlockDouble.cpp` への MMCSS ブロック追加は不要。

#### (C) `hasHeterogeneousCores_` のフォールバック

`detectCoreTopology()` が API 失敗で空の `CoreTopology` を返した場合、`physicalCoreCount == 0` となり `computeSymmetricMasks()` は `N < 2` の早期 return で `ThreadAffinityMasks{}`（全ゼロ）を返す。この場合 `hasHeterogeneousCores_ = false` だが `audioMask == 0` のため `applyMmcssPriority()` 末尾の `if (audioMask != 0)` でスキップされ、安全にアフィニティ未設定となる。破綻なし。

#### (D) `applyCurrentThreadPolicy` への `AudioRealtime` 追加に関するコンパイル安全性

`ThreadType` enum に `AudioRealtime` を追加した場合、`applyCurrentThreadPolicy()` の `switch(type)` に default ケースが存在しない（現行コード確認済み）。MSVC は `/W4` 設定で enum 網羅性チェックが有効になり `C4062` 警告が出る可能性があるが、CMake L757-767 で `/WX`（警告エラー化）は設定されていないためビルド失敗には至らない。`AudioRealtime` ケースを switch に追加することでむしろ `C4062` 警告は解消される。本計画 §5.1(c) の通り `return;` で早期脱出する設計で対応済み。

#### (E) MSDN プロセッサアフィニティマスク API 仕様との突合結果（v4 追加）

MSDN 公式ドキュメント 6件 + WSL シミュレーション 9テストで以下の点を検証・確認:

| # | 検証項目 | MSDN 仕様 | Plan の記述 | 判定 |
|---|---|---|---|---|
| E1 | `PROCESSOR_RELATIONSHIP.GroupCount` (物理コア) | MSDN: "always 1" | Plan: `>= 1` → v4で `== 1` に修正 | ✅ 修正済み |
| E2 | `GROUP_AFFINITY.Mask` 型 | MSDN: `KAFFINITY` = `ULONG_PTR` | Plan: `DWORD_PTR` (= `KAFFINITY`) | ✅ 一致 |
| E3 | `SetThreadAffinityMask` Win11 制約 | MSDN: "must specify processors in thread's current primary group" | Plan: 単一グループ前提で自動充足 | ✅ 一致 |
| E4 | `SetThreadAffinityMask` prevMask==0 | MSDN: "return zero on failure, GetLastError()" | Plan: prevMask==0 診断ログ推奨追記 (v4) | ✅ 修正済み |
| E5 | `EfficiencyClass` 意味論 | MSDN: "nonzero only on heterogeneous; higher=better perf" | Plan: inequality detection → heterogeneous | ✅ 一致 (v2 で修正済み) |
| E6 | `EfficiencyClass` Win10+ | MSDN: "minimum OS = Windows 10" | Plan: Win10+ 明記済み | ✅ 一致 |
| E7 | `GetLogicalProcessorInformationEx` Win7+ | MSDN: min client = Windows 7 | Plan: Win7+ 明記済み | ✅ 一致 |
| E8 | `RelationProcessorCore` 全グループ返却 | MSDN: "for every active processor core in every processor group" | Plan: §11.2(A)で >64LP 対象外と明記 | ✅ 明記済み |
| E9 | MMCSS `AvSetMmThreadCharacteristics` Affinity | MSDN: MMCSS Tasks registry `Affinity` REG_DWORD 0x00/0xFFFFFFFF = no affinity | Plan: §11.1 #8 で MMCSS はアフィニティに触れないと明記 | ✅ 一致 |

#### (F) v5 WSLシミュレーション最終検証結果（v5 追加）

MSDN 公式ドキュメント 8件に加え、WSL シミュレーションで以下の追加検証を実施。
`computeSymmetricMasks` 全7シナリオ + コード的バグ候補 8テスト = 全 15 テスト PASS。

| # | 検証項目 | シナリオ | 結果 |
|---|---|---|---|
| F1 | `detectCoreTopology` L220 `.size()` 括弧 | MSVC `std::vector::size` はメンバ関数 → `.size` では関数ポインタ→int変換となりコンパイルエラー | ✅ `.size()` に修正済み (v5) |
| F2 | WOW64 folding | 64ビットビルド (sizeof(void*)==8) では適用外 | ✅ x64 ビルド限定 → 明記済み (v5) |
| F3 | `std::vector<BYTE>` alignment | std::vector 動的確保は最小16Bアラインメント。`SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` 要求8B → 充足 | ✅ 問題なし |
| F4 | `computeSymmetricMasks` 1コア | N=1 → 全ゼロ | ✅ PASS |
| F5 | `computeSymmetricMasks` 0コア（API失敗） | N=0 → N<2 → 全ゼロ | ✅ PASS |
| F6 | `computeSymmetricMasks` 2C4T | audio=CPU[1], worker=learner=CPU[0] (集中化不可避) | ✅ PASS |
| F7 | `computeSymmetricMasks` 4C8T | 全フィールド計画書計算例と一致 | ✅ PASS |
| F8 | `computeSymmetricMasks` 4C4T (non-SMT) | 全フィールド計算例と一致 | ✅ PASS |
| F9 | `computeSymmetricMasks` 6C12T | disjoint + cover OK | ✅ PASS |
| F10| `computeSymmetricMasks` 8C16T | 全フィールド計算例と一致 | ✅ PASS |
| F11| `computeSymmetricMasks` disjointness invariant | 全7シナリオで audioMask ∩ nonAudioMask == 0 | ✅ PASS |
| F12| `EfficiencyClass` homogeneous/heterogeneous | all-zero→false, mixed→true, single-core→false | ✅ PASS |
| F13| `ALL_PROCESSOR_GROUPS` = 0xFFFF | フォールバック sentinel 値確認 | ✅ PASS |
| F14| `learnerMainIdx = min(1,N-2)` 境界 | N=1→skip, N=2→0 (workerと同一), N≥3→1 | ✅ PASS |
| F15| 行番号一致確認 | 実装 L2301 (plan L2300→修正), L214-272 (plan L222-281→修正) | ✅ 修正済み (v5) |

#### (G) v6 WSL S6 詳細検証結果（v6 追加）

15件の追加シナリオで計画の細部を精緻化:

| # | 検証項目 | シナリオ | 結果 |
|---|---|---|---|
| G1 | `AudioRealtime` 分岐設計 | `return;` で早期脱出 → `SetThreadPriority` を呼ばない (MMCSS が優先度管理) | ✅ 設計妥当 |
| G2 | `getAudioRealtimeMask` アクセサ | `masks_.audioRealtime` を返すだけ | ✅ PASS |
| G3 | `CoreTopology` デフォルト | count=0, vectors空, hasHetero=false | ✅ PASS |
| G4 | N=0 安全 | `physicalCoreMasks[N-1]` アクセスなしの早期 return | ✅ PASS |
| G5 | N=1 ケース | 全ゼロマスク | ✅ PASS |
| G6 | N=2 境界 | `learnerMainIdx = min(1, 0) = 0` → worker と同コア | ✅ PASS |
| G7 | 12C24T | audio=12th, worker=C[0], learner=C[1], nonAudio=C[0..10] | ✅ PASS |
| G8 | P/E混在 8P+8E 検出 | EfficiencyClass = {2,1} → heterogeneous=true → アフィニティ無効化 | ✅ PASS |
| G9 | `AudioRealtime` 分岐で優先度 overrides なし | MMCSS が 23-26 priority を維持 → 設計妥当 | ✅ PASS |
| G10| `static_cast<int>(audioMask)` truncation | x64 `DWORD_PTR` (uint64) → int (32bit) は truncation だが対象 HW (≤16C32T=0xFFFFFFFF) では実用上問題なし。NOTE コメント追記 (v6) | ✅ 精緻化 |
| G11| `DWORD_PTR{1} << i` 安全性 | i ∈ [0, 63] で安全。i=64 で UB だが `nLogical ≤ 64` のため発生しない | ✅ PASS |
| G12| §5.3 delete範囲 L87-98 → L87-L99 inclusive | plan 説明精緻化 (v6) | ✅ 精緻化 |
| G13| §5.4 insert位置 | `applyMmcssPriority()` 末尾 (L272 closing brace の直前) | ✅ 妥当 |
| G14| `audioMask==0` skip `SetThreadAffinityMask` | `if (audioMask != 0)` で正しく skip | ✅ 妥当 |
| G15| `_WIN32_WINNT` 強制設定 | JUCE `juce_BasicNativeHeaders.h:103-104` で必ず `0x0A00` (Win10) | ✅ v6 §11.1 #9 修正 |

#### (H) v7 WSL S7 実装前最終検証結果（v7 追加 — **重大**）

WSL S7 7テストで実装前の境界線・既存バグを詳細検証。**1件の既存バグを発見し、計画書 §5.3 / §6 / §7 / §10 を更新**。

| # | 検証項目 | シナリオ | 結果 |
|---|---|---|---|
| H1 | Audio affinity reuse flag (audioAffinityApplied_) | AudioRealtime を `mmcssApplied_` に piggyback するか、別フラグ化するか | ✅ v0.6.8 では piggyback で十分（§5.4 で ok）。将来独立 re-affinity 必要なら別フラグ追加可 |
| H2 | initialize / apply ordering invariant | `initialized_=false` 中の applyCurrentThreadPolicy → short-circuit のみ実行。再 apply ロジックなし | ⚠️ リスク発見 → **WorkerThread が起動時 apply する一方通行設計** → H7 で判明 |
| H3 | Audio thread の affinity 適用ライフタイム | Audio コールバック初回で `SetThreadAffinityMask(GetCurrentThread(), mask)` | ✅ スレッドIDに関係なく常に正しい thread に適用される |
| H4 | NativeRT パスとの相互作用 | SetPriorityClass(REALTIME) → SetThreadPriority(TIME_CRITICAL) → SetThreadAffinityMask の順 | ✅ 順序妥当。PriorityClass はプロセス全体に効くが savedProcessPriorityClass で復元 |
| H5 | Timer.cpp L272 実際の位置 | `applyMmcssPriority()` 外側 if/else の閉じ括弧 L272 | ✅ Plan §5.4 の L272 直後挿入は正しい |
| H6 | 複数ワーカースレッド並列 affinity 適用 | 各スレッドが独立に SetThreadAffinityMask(GetCurrentThread()) | ✅ 並列安全 (各スレッド固有のアフィニティ) |
| **H7** | **★ `AudioEngine.Init.cpp` L82 vs L97 順序バグ発見** | L82 = `initWorkerThread()`、L97 = `affinityManager.initialize(masks)`。L82 → WorkerThread::run() → applyCurrentThreadPolicy(Worker) → `initialized_==false` → short-circuit → **affinity 永久未適用** | ❌ **既存バグ** → v7 で **§5.3 / §6 / §7 / §10** を更新し、L82 を L99 直後に**移動**する実装指示を追加 |

#### (I) v8 WSL S8 pthread 実機テスト結果（v8 追加 — 順序入替の Before/After 検証）

v7 で発見した init 順序バグの修正が pthread で実機動作することを確認。

| # | 検証項目 | シナリオ | 結果 |
|---|---|---|---|
| I1 | **既存バグの再現** | L82 `initWorkerThread()` が L97 `initialize()` より前にある現状コード | ✅ **既存バグ再現確認**: WorkerThread::run() が `initialized_=false` で short-circuit → mask 0xABCD が適用されない (apply=false) |
| I2 | **順序入替後の正しい挙動** | `initialize()` を `initWorkerThread()` の前に呼ぶ | ✅ **修正後**: WorkerThread::run() が `initialized_=true` を見て mask=0xABCD を正しく適用 |
| I3 | 並列 init+start 競合 | initialize と start が別スレッドで並列 | ✅ 順序入替後は sequential → 競合なし |
| I4 | 5スレッドタイプ別 routing | Worker/LightBackground/HeavyBackground/AudioRealtime/UI 全ての maskOut | ✅ 全て designated mask にルーティング |
| I5 | `mmcssApplied_` affinity piggyback | 初回: true (apply)、2-3回目: false (skip)、`prepareToPlay()`後: 再度true | ✅ 設計通り |
| I6 | 複数デバイス prepareToPlay() サイクル | 2回 prepareToPlay() → 2回 first-callbackで apply | ✅ 正しい回数適用 |
| I7 | Release/Acquire メモリ順序 | `initialized_=true` の release と worker スレッドの acquire が HB 関係 | ✅ happens-before 成立 |
| I8 | Message thread の applyMessageThreadPolicy タイミング | engine init 完了後に適用 | ✅ 想定通り |
| I9 | AudioRealtime 分岐の noexcept 維持 | `return;` で早期脱出 → noexcept 契約保持 | ✅ 例外を投げない |

**pthread 実機コード実行結果**: 全 9 テスト PASS。

#### (J) v9 MSDN 追加調査: プロセス全体 affinity + IdealProcessor（v9 追加）

**新規 MSDN 調査 3 件 (累計 11 件)**:

| # | API | MSDN 重要仕様 | 本計画との関係 | 判定 |
|---|---|---|---|---|
| J1 | `GetProcessAffinityMask` | Win11: プライマリグループ内のマスクを返す。スレッドがプライマリグループ外の affinity を明示設定している場合、両マスクとも 0 を返す | 本計画は `GetProcessAffinityMask` を一切使用しない → 影響なし | ✅ 安全 |
| J2 | `SetProcessAffinityMask` | Win11: プロセスがスレッドの affinity をプライマリグループ外に明示設定した場合、`ERROR_INVALID_PARAMETER` で失敗 | 本計画は `SetProcessAffinityMask` を一切使用しない（`SetThreadAffinityMask` のみ）。§3.4 `SetPriorityClass(REALTIME_PRIORITY_CLASS)` はプロセス全体影響だが affinityとは無関係 → 安全 | ✅ 安全 |
| J3 | `SetThreadIdealProcessor` | "preferred processor" hint — ハードアフィニティではなく、システムが可能な限りそのプロセッサにスケジュールする。MMCSS Deadline QoS より優先度が低い | 本計画は `SetThreadAffinityMask` でハード固定するため、`SetThreadIdealProcessor` を使用しない。hint より hard mask が dominant → 安全 | ✅ 安全 |

**結論**: 追加の 3 API (`GetProcessAffinityMask`, `SetProcessAffinityMask`, `SetThreadIdealProcessor`) はいずれも本計画の scope 外であり、不使用のため **追加リスク・追加実装指示は不要**。

#### (K) v10 deep-dive 検証: CPU Set / EcoQoS / Audio Thread 起源 / DeferredFree（v10 追加）

v9 MSDN 検証 + 現実装コードベースへの深堀り 4 項目を実施。

| # | 検証項目 | 結論 |
|---|---|---|
| K1 | MSDN CPU Set vs affinity mask | MSDN (CPU Sets conceptual): **"If a thread or process has a restrictive affinity mask set, the affinity mask is respected above any conflicting CPU Set assignment."** → 本計画の `SetThreadAffinityMask` (ハード固定) は、OS デフォルトの CPU Set (もし使用されていれば) より優先される。Audio Thread の固定が dominant であることを確認 |
| K2 | プロセス EcoQoS 無効化との相互作用 | `MainApplication.cpp:79-89` 確認: `PROCESS_POWER_THROTTLING_EXECUTION_SPEED` + `StateMask = 0` (= High QoS、無効化)。スレッド affinity とは独立動作 → 本計画 §3.4 の通り、両者は共存可能。MainApplication.cpp L146 の `applyMessageThreadPolicy()` はエコQoS 確立後 L89 で実行 → 順序も問題なし |
| K3 | Audio Thread の生成元 | JUCE は `audioDeviceManager.addAudioCallback(&audioProcessorPlayer)` で Windows Audio Engine (WASAPI/ASIO) 経由のコールバックを登録 → OS/Windows オーディオエンジンがスレッドを管理。`CreateRemoteThreadEx` / `PROC_THREAD_ATTRIBUTE_GROUP_AFFINITY` は JUCE 本体では未使用 → 起動時の thread-creation-time affinity 設定は不要。post-hoc `SetThreadAffinityMask(GetCurrentThread(), audioMask)` (本計画 §5.4) が唯一の対応経路として妥当 |
| K4 | DeferredFreeThread の生成タイミング | `ConvolverProcessor.Lifecycle.cpp:373` で `aligned_make_unique<DeferredFreeThread>(rcuSwapper, affinityMgr)` が lazy 作成。`AudioEngine::initialize()` 完了後の初回 prepare 時に呼ばれる → その時点で `affinityManager.initialize(masks)` 完了。本計画 v7 修正（WorkerThread 順序入れ替え）で WorkerThread は fix されるが、DeferredFreeThread は lazy タイミングで自然に `initialized_==true` を観測 → v7 修正 (initWorkerThread) を追加で拡張する必要なし（ただし DeferredFreeThread の安全は間接的に保証） |

**結論**: v10 で 4 件の深堀り検証が完了、いずれも本計画の実装に追加変更を要求しない。「SetThreadAffinityMask が CPU Set より dominant」「EcoQoS は無関係」「Audio Thread は post-hoc 設定で十分」「DeferredFreeThread は lazy 起点で v7 修正で自然にカバー」全て確認。

#### (L) v11 実装前最終バグ監査 B1-B7（v11 追加）

v10 完了後、実装着手前の最終バグ監査として新規 7 項目 (B1-B7) を実施。新規バグ発見なし。

| # | 検証項目 | 検証内容 | 結果 |
|---|---|---|---|
| B1 | `AudioRealtime` switch 分岐 `return;` の他スレッド優先度への影響 | `ThreadAffinityManager::applyCurrentThreadPolicy()` switch で `AudioRealtime` 分岐が `return;` で早期脱出 → `SetThreadPriority` を呼ばない。他の ThreadType (Worker/LearnerMain/...) は switch → break → L116 `SetThreadAffinityMask` → L118 `SetThreadPriority` の通常パス。`return;` は AudioRealtime 分岐内のみで完結。他スレッドのルーティングに影響なし。 | ✅ 安全 |
| B2 | `mmcssApplied_` reset と affinity 再適用の競合タイミング | `PrepareToPlay.cpp:27` (Message Thread, `publishAtomic(mmcssApplied_, false, release)`) → `AudioBlock.cpp:46` / `BlockDouble.cpp:49` (Audio Thread, `compareExchangeAtomic(mmcssApplied_, expected, true, acq_rel)`)。JUCE 契約 (`PrepareToPlay.cpp:102-104` コメント明記) により prepareToPlay 実行中は Audio Thread callback は走らない。`lifecycleState = Prepared` (L226 release) の後に Audio callback 開始 → CAS 成功 → `applyMmcssPriority()` → affinity 設定の順。reset→CAS は一方向 HB 成立、双方向競合なし。 | ✅ 競合なし |
| B3 | `hasHeterogeneousCores_` bool のマルチスレッドアクセス | `initialize()` で1回のみ設定 (Message Thread)。Audio Thread は `initialize()` 完了後に起動 (`MainWindow.cpp:289 addAudioCallback` が `initialize()` より後)。シーケンシャル設計のため std::atomic 不要、plain bool で safe。`applyMmcssPriority()` からの読み出しも Audio Thread が起動した後のみ。 | ✅ safe (非 atomic で OK) |
| B4 | `audioRealtime` マスクが `NoiseShaperLearner::getEvalWorkerMask` に与える影響 | `NoiseShaperLearner.cpp:524` で `applyCurrentThreadPolicy(ThreadType::LearnerEval, workerIndex)` を呼出 → `ThreadAffinityManager.h:98` で `getEvalWorkerMask(evalWorkerIndex)` に dispatch。`getEvalWorkerMask` (L128-152) は `masks_.learnerEvalBase` のみ参照。plan §5 で `learnerEvalBase = nonAudioMask` と規定 (audioRealtime は **別変数** `masks_.audioRealtime` に格納)。learnerEvalBase は audioRealtime マスク bits を含まないため、ラウンドロビン分散結果に影響なし。 | ✅ 影響なし |
| B5 | 子プロセス affinity 継承の有無 | `src/` ディレクトリ全体を `CreateProcess`, `_popen`, `ShellExecute`, `system(`, `subprocess` で検索 → 一致ゼロ。子プロセス起動 API を使用しないため、`SetThreadAffinityMask` 設定が子プロセスに継承されることで生じる意図せぬコア固定化は発生しない。（参考: `SetThreadAffinityMask` はスレッドローカル設定で子プロセスには継承されない仕様だが、そもそも子プロセス起動不在。） | ✅ 該当なし |
| B6 | nullptr 安全性 | `applyMessageThreadPolicy` (L70 `if (masks_.ui != 0)`), `applyCurrentThreadPolicy` (L115 `if (mask != 0)`), `getEvalWorkerMask` (L131 `if (base == 0) return 0;`, L147 `if (count == 0) return 0;`) は全て nullptr/ゼロマスクの事前チェック付き。`SetThreadAffinityMask(GetCurrentThread(), 0)` は呼出されない。`GetLogicalProcessorInformationEx` の出力バッファは `std::vector<BYTE>` 動的確保で失敗時は `std::bad_alloc` だが、`detectCoreTopology()` は失敗時 `CoreTopology{}` (count=0) を返し `computeSymmetricMasks` が N<2 で全ゼロマスク生成 → 全スレッドで skip。 | ✅ 安全 |
| B7 | diagLog 順序 | 現状 `Init.cpp:98` に `[AFFINITY] ThreadAffinityManager initialized: worker=0x01 learner=0x02...` が出力。本計画 v7 で L82 `initWorkerThread()` を L99 直後に移動するため、現状 L98 の後 → 移動後 L99 完了後 の順序で `initWorkerThread()` が呼ばれ、その後 WorkerThread が起動 → Worker マスク適用 (WorkerThread.cpp:58-59)。plan §5.4 新規 diagLog は `applyMmcssPriority()` 末尾 (Timer.cpp L272 直前) に挿入 → Audio Thread 起動後初回コールバックで出力。`Init.cpp:98` (`[AFFINITY] initialized`) → `[AFFINITY audio] applied` (Audio Thread 初回) の時系列は直感に合致。順序崩れなし。 | ✅ 問題なし |

**結論**: B1-B7 の 7 項目すべて「安全/影響なし/該当なし」。新規バグ・新規修正指示は発生せず。本計画 v1〜v10 の実装指示 (§10 Step 1-8) が最終確定。

### 11.3 結論

本計画は MSDN 公式ドキュメント **11 件** + WSL **46 テスト** + pthread 実機 **9 テスト** + **v10 deep-dive 4 項目** + **v11 バグ監査 7 項目 (B1-B7)** による包括的突合検証を経て **実装可能** である。

v9 で追加された項目:
- `applyMmcssPriority()` 始点行番号を L215 → L214 に訂正（`grep` 実ソースコード確認）
- MSDN 追加調査 3 件 (`GetProcessAffinityMask`, `SetProcessAffinityMask`, `SetThreadIdealProcessor`) = 全影響なし確認済み

**v7 で発見された既存バグ**:
- `AudioEngine.Init.cpp` で `initWorkerThread()` (L82) が `affinityManager.initialize(masks)` (L97) より前に呼ばれている。WorkerThread::run() の `applyCurrentThreadPolicy(Worker)` が `initialized_==false` で short-circuit し、WorkerThread の affinity が**永久に未適用**の状態。
- これを本計画 v7 で修正: L82 の `initWorkerThread();` 呼び出しを `affinityManager.initialize(masks);` の**直後** (L99 の直後) へ移動。

v6 で精緻化された項目:
- `static_cast<int>(audioMask)` truncation への注釈追記 (§5.4)
- §5.3 の delete 範囲を `L87-L99 inclusive` に精緻化
- §11.1 #9 の `_WIN32_WINNT` 設定を JUCE の強制設定として明記

実装にあたり以下の点を遵守すること:

1. **`GetLogicalProcessorInformationEx` を使用すること**（非Ex版は不可）
2. **`#include <vector>` を `ThreadAffinityManager.h` に追加すること**
3. **`PROCESSOR_RELATIONSHIP.GroupCount == 1` を前提に `GroupMask[0].Mask` のみ採取すること** (MSDN: "always 1")
4. **`SetThreadAffinityMask` の prevMask==0 エラーハンドリングを診断ログに追加すること** (§5.1(h) 参照)
5. **プロセッサグループが複数にまたがる環境（>= 65論理プロセッサ、マルチソケットNUMA）は対象外**と明記
6. **CMake は `/W4` だが `/WX` なし**（CMake L757-767 確認済み）。`AudioRealtime` ケース追加で `C4062` 警告は解消方向
7. **32ビットビルド (WOW64) は対象外**。ConvoPeq は x64 ビルド限定（v5 追加明記）
8. **`topo.physicalCoreMasks.size()` の括弧を忘れないこと** (v5 確認)
9. **JUCE の `juce_BasicNativeHeaders.h:103-104` で `_WIN32_WINNT = 0x0A00` が常に設定される** (v6 確認)
10. **§5.3 の削除範囲は L87-L99 inclusive (`{` から `}` までブロック全体)** (v6 精緻化)
11. **§5.4 の `static_cast<int>(audioMask)` は 64bit マスク truncation だが対象 HW では安全** (v6 精緻化)
12. **★ v7 重大: `initWorkerThread()` を `affinityManager.initialize(masks)` の直後に移動する** (順序入替必須)
13. **★ v8 確認: 順序入替後は pthread シミュレーションで全 9 テスト PASS** (S8-1〜S8-9)
14. **v8: AudioRealtime 分岐の `return;` により `applyCurrentThreadPolicy` の noexcept 契約が保持される** (S8-9)
15. **v8: Release/Acquire メモリ順序により `initialized_=true` 後に `masks_` 書き込みが可視** (S8-7)
16. **v8: 複数デバイスの prepareToPlay() サイクルで正しく affinity と MMCSS が再適用される** (S8-6)
17. **v9: 行番号最終一致確認 — `applyMmcssPriority()` 始点は L214 (`void AudioEngine::applyMmcssPriority()`)、末尾 L272** (v9 微細訂正)
18. **v9: `SetProcessAffinityMask` / `GetProcessAffinityMask` / `SetThreadIdealProcessor` は本計画 scope 外 — 追加リスクなし** (§11.2(J))
19. **★ v10: `SetThreadAffinityMask` は `CPU Set` assignment より dominant (MSDN: "restrictive affinity mask is respected above any conflicting CPU Set assignment")** (§11.2(K))
20. **★ v10: プロセス全体 EcoQoS 無効化 (MainApplication.cpp:79-89) は affinity 動作と無関係に High QoS を選択、相互作用なし** (§11.2(K))
21. **v10: Audio Thread は Windows Audio Engine 管理 → `PROC_THREAD_ATTRIBUTE_GROUP_AFFINITY` 不要、post-hoc `SetThreadAffinityMask(GetCurrentThread())` で対応、§5.4 設計妥当** (§11.2(K))
22. **v10: `DeferredFreeThread` は ConvolverProcessor.Lifecycle.cpp:373 で lazy 作成され、AudioEngine::initialize() 完了後のため v7 修正で自然に covers** (§11.2(K))
23. **★ v11: B1-B7 全 7 項目で新規バグなし** (§11.2(L))。mmcssApplied_ reset/CAS の HB 順序、LearnerEval は audioRealtime から独立、子プロセス API 不使用、nullptr チェック既備、diagLog 順序妥当、全確認済み。実装は §10 Step 1-8 に従う。