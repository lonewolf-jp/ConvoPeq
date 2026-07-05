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

### 4.1 コードベース全スレッド棚卸し（v19 確定）

v19 で実施したコードベース全体のスレッド棚卸し結果。全スレッドが計画のアフィニティ管理範囲内であることを確認済み。

| # | スレッド | 生成方式 | ThreadType | アフィニティ適用場所 | 計画対応 |
|---|---|---|---|---|---|
| 1 | WorkerThread | `std::thread` | Worker | `WorkerThread::run()` L59 | ✅ 自動適合 |
| 2 | RebuildThread | `std::thread` | HeavyBackground | `rebuildThreadLoop()` L723 | ✅ 自動適合 |
| 3 | LoaderThread | `juce::Thread` | HeavyBackground | `LoaderThread::run()` L39 | ✅ 自動適合 |
| 4 | ProgressiveUpgradeThread | `juce::Thread` | HeavyBackground | `ProgressiveUpgradeThread::run()` L76 | ✅ 自動適合 |
| 5 | DeferredFreeThread | `std::thread` (lazy) | LightBackground | `DeferredFreeThread::run()` L152 | ✅ 自動適合 |
| 6 | NoiseShaperLearner worker | `std::jthread` | LearnerMain | `workerThreadMain()` L725 | ✅ 自動適合 |
| 7 | NoiseShaperLearner eval | `std::jthread[]` | LearnerEval | `evaluationWorkerMain()` L524 | ✅ 自動適合 |
| 8 | Audio callback | Win Audio Engine | AudioRealtime | `applyMmcssPriority()` (初回) | ✅ §5.4 |
| 9 | IR Preview ThreadPool | `juce::ThreadPool(1)` | — | 未設定（非RT、短命ジョブのため不要） | ✅ 対象外 |
| 10 | Save ThreadPool | `juce::ThreadPool(1)` | — | 未設定（非RT、短命ジョブのため不要） | ✅ 対象外 |
| 11 | Message/UI Thread | Main thread | UI | `applyMessageThreadPolicy()` L146 | ✅ 自動適合 |

**確認日**: 2026-07-05 | **ツール**: grep/rg (WSL), AiDex MCP

---

## 5. 詳細設計

### 5.1 ThreadAffinityManager.h 拡張

> **重要**: 本§5.1 では `#include <vector>` の追加が必要。既存 `ThreadAffinityManager.h` は `<array>`, `<atomic>`, `<cstdint>`, `Windows.h` のみを include。`std::vector<DWORD_PTR>` を `CoreTopology` で使うため、ファイル冒頭の include ブロックに `#include <vector>` を追加すること。`<memory>` は `detectCoreTopology()` 内で `std::vector<BYTE>` を使えば `std::unique_ptr` 不要で済むため不要。
> **v15 追加**: `CoreTopology::cores` に `PhysicalCoreInfo` 構造体を使用するため、`#include <algorithm>`（`std::sort`）を追加。`cores` には mask と efficiencyClass がペアで格納されるため、ソート時のずれが生じない。
> **v16 追加**: `ThreadAffinityManager` は純粋ユーティリティクラスとして `diagLog` に依存しない。ログ出力用の include (`JuceHeader.h` 等) は追加しない。`detectCoreTopology()` は `CoreTopology` を返すだけでログは呼び出し側で行う。
> **v20 追加**: ソート用の `lowestBit()` に `std::countr_zero` (C++20, `<bit>`) を使用する。`<bit>` は `DspNumericPolicy.h` で既に include 済み。`while` ループより可読性が高く、コンパイラにより1命令 (BSF/TZCNT) に最適化される。

#### (a) ThreadType 列挙型

```cpp
enum class ThreadType {
    Worker, LearnerMain, LearnerEval, HeavyBackground,
    LightBackground, UI,
    AudioRealtime  // ★追加（将来の拡張性のため。現在 AudioThread の affinity は
                   //   applyMmcssPriority() 末尾で直接 SetThreadAffinityMask しているため、
                   //   本 enum を applyCurrentThreadPolicy() で使うと二重適用になる。
                   //   二重適用自体は無害（同一マスクの再設定）だが、責務は
                   //   applyMmcssPriority() (Timer.cpp) 側にあり、
                   //   applyCurrentThreadPolicy は将来のリファクタリング用に用意。
                   // ★ v19: convo::numeric_policy::ThreadRole::AudioRealtime (DspNumericPolicy.h)
                   //   とは別概念。そちらはランタイムスレッド検出（assertion用）であり、
                   //   CPUアフィニティ管理とは無関係。名前空間が異なるため衝突なし。
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

また、既存全ケースの `SetThreadAffinityMask` 呼び出しを `prevMask` でラップし、診断ログ有効時に各スレッドの適用結果を追跡可能にする（v17, v20）:

```cpp
// ★ v17: SetThreadAffinityMask の結果を prevMask で取得（非診断ビルドでは unused）
if (mask != 0) {
    const DWORD_PTR prevMask = ::SetThreadAffinityMask(::GetCurrentThread(), mask);
    juce::ignoreUnused(prevMask);
}
```

#### (d) アクセサ

```cpp
[[nodiscard]] DWORD_PTR getAudioRealtimeMask() const noexcept {
    return masks_.audioRealtime;
}
```

#### (e) CoreTopology 構造体（新規）

```cpp
struct PhysicalCoreInfo {
    DWORD_PTR mask;            // 論理プロセッサ集合マスク（SMT兄弟含む、単一プロセッサグループ前提）
    BYTE efficiencyClass;      // EfficiencyClass（P/E判定用、Win10+で有効）
};

struct CoreTopology {
    int physicalCoreCount = 0;
    bool hasHeterogeneousArchitecture = false;
    std::vector<PhysicalCoreInfo> cores;  // 全物理コアの情報（マスク最下位ビット順にソート済み）
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
    // ★ v20: noexcept を維持する。起動時初期化であり、
    //   メモリ確保失敗は回復不能と判断している。
    //   万が一 std::bad_alloc が発生した場合は std::terminate() によりプロセス終了。
    CoreTopology topo;
#ifdef _WIN32
    // 1. RelationshipType = RelationProcessorCore で物理コア情報のみ取得
    DWORD bufLen = 0;
    if (!::GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bufLen)
        && ::GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        // ★ v15/v16: API が全体的に未対応 → アフィニティ無効（全ゼロマスク）
        //   ThreadAffinityManager は純粋なユーティリティクラスとして設計するため、
        //   ログ出力は行わない。呼び出し側（AudioEngine.Init.cpp）が
        //   physicalCoreCount == 0 を検出して diagLog する。
        //   本計画の対象環境は Windows 10+ のため、このパスは実質的に到達しない。
        return topo;
    }

    // 2. バッファ確保・取得
    std::vector<BYTE> buf(bufLen);
    if (!::GetLogicalProcessorInformationEx(RelationProcessorCore,
        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
        &bufLen)) {
        // ★ v16: 2回目API失敗 → 全ゼロマスク（アフィニティ無効）。ログは呼び出し側で行う。
        return topo;
    }

    // 3. 可変長レコードを走査
    DWORD offset = 0;
    while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= bufLen) {
        // ★ v20: 最低限のヘッダサイズを確保してから info->Size を読む
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + offset);
        // ★ v18: 防御的ガード — API 破損やメモリ破壊で Size==0 になると無限ループとなるため
        if (info->Size == 0) break;
        // ★ v18: 境界チェック — Size がバッファ残より大きい場合は破損とみなす
        if (offset + info->Size > bufLen) break;
        if (info->Relationship == RelationProcessorCore) {
            const auto& proc = info->Processor;
            // MSDN: "If PROCESSOR_RELATIONSHIP represents a processor core,
            //        the GroupCount member is always 1."
            //   → RelationProcessorCore で取得した各レコードは物理コア1つに1対1対応。
            //      GroupCount == 1, GroupMask[0].Mask = その物理コアに属する全論理プロセッサのbitmask。
            //      SMT兄弟はOSが自動的に同一 GroupMask[0].Mask に含めて返す。
            // ★ v21: mask==0 の物理コアレコードは破損データとみなして除外
            if (proc.GroupCount == 1 && proc.GroupMask[0].Mask != 0) {
                // ★ v15: PhysicalCoreInfo 構造体に mask と efficiencyClass をペアで格納
                topo.cores.push_back({proc.GroupMask[0].Mask, proc.EfficiencyClass});
            }
        }
        offset += info->Size;
    }

    topo.physicalCoreCount = static_cast<int>(topo.cores.size());

    // ★ v15: MSDN は GetLogicalProcessorInformationEx の列挙順を保証しないため、
    //    cores[] を mask の最下位ビット（論理CPU番号）順にソートする。
    //    PhysicalCoreInfo 構造体に mask と efficiencyClass が一体化しているため、
    //    ソート時に効率クラスがずれることはない。
    std::sort(topo.cores.begin(), topo.cores.end(),
        [](const PhysicalCoreInfo& a, const PhysicalCoreInfo& b) noexcept {
            // ★ v20: C++20 std::countr_zero を使用（<bit> ヘッダ）
            //   while ループより可読性が高く、BSF/TZCNT 命令に最適化される。
            //   mask==0 の場合は 64 を返す（countr_zero の仕様により 64 以上）。
            const auto lowestBit = [](DWORD_PTR mask) noexcept -> int {
                return mask == 0 ? 64 : static_cast<int>(std::countr_zero(mask));
            };
            return lowestBit(a.mask) < lowestBit(b.mask);
        });

    // 4. P/E混在判定: 全コアの EfficiencyClass が同一か検査
    if (topo.physicalCoreCount > 1) {
        const BYTE first = topo.cores.empty() ? 0 : topo.cores[0].efficiencyClass;
        for (const auto& core : topo.cores) {
            if (core.efficiencyClass != first) {
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

> **v13 P/E heterogeneous ポリシー覚書**: `hasHeterogeneousArchitecture == true` のとき、本設計は「アフィニティを設定しない」という**意図的なポリシー選択**を行う。MMCSS Deadline QoS (Pro Audio) + Windows Thread Director に委任することで、OS が最適な P-core を動的に選択する。`EfficiencyClass` が非ゼロになる将来の CPU (AMD Zen5c, Intel LPE 等) でも同一ロジックで正しく heterogeneous と判定される。このポリシーは「EfficiencyClass が異なる ≠ affinity 無効化による性能劣化」ではなく、「OS スケジューラに委任する方が P/E 環境では有利」という能動的判断である。

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

#### (h) `SetThreadAffinityMask` エラー処理の推奨（v13 強化、v14 維持）

MSDN 仕様: `SetThreadAffinityMask` は失敗時に 0 を返し、`GetLastError()` でエラーコードを取得可能。最も一般的な失敗原因は **`ERROR_INVALID_PARAMETER`** — thread affinity mask が process affinity mask の部分集合でない場合に発生する。診断用に prevMask==0 のケースでエラーコードをログ出力することを推奨:

```cpp
const DWORD_PTR prevMask = ::SetThreadAffinityMask(::GetCurrentThread(), audioMask);
if (prevMask == 0) {
    const DWORD err = ::GetLastError();
    diagLog("[AFFINITY] SetThreadAffinityMask FAILED: mask=0x"
            + juce::String::toHexString(static_cast<int>(audioMask))
            + " GetLastError=" + juce::String(static_cast<int>(err)));
}
```

> **v14 補足**: v13 で記載していた `ERROR_INVALID_PARAMETER` 定数値のログ出力は除去（デバッガで確認可能なため冗長）。mask 値 + `GetLastError` で十分。

> **v14 補足**: `audioMask` は常にプロセスアフィニティの部分集合であるため (`computeSymmetricMasks` が `GetLogicalProcessorInformationEx` の実在コアマスクのみ使用)、理論上 `ERROR_INVALID_PARAMETER` は発生しない。しかし診断ログは念のため実装する。

#### (i) 起動時診断ログ（v14 追加）

`AudioEngine.Init.cpp` の `affinityManager.initialize(masks)` 直後に、以下の情報を一括出力する。これにより Process Explorer を使わずともログから全アフィニティ状態を確認できる:

```cpp
// ★ [work64 v14] 起動時アフィニティ診断ログ（一括出力）
diagLog("[AFFINITY] coreTopology: physical=" + juce::String(topo.physicalCoreCount)
    + " logical=" + juce::String(::GetActiveProcessorCount(ALL_PROCESSOR_GROUPS))
    + " heterogeneous=" + juce::String(hasHeterogeneousCores_ ? "true" : "false"));
diagLog("[AFFINITY] audioMask=0x" + juce::String::toHexString(static_cast<uint64_t>(masks.audioRealtime))
    + " nonAudio=0x" + juce::String::toHexString(static_cast<uint64_t>(nonAudioMask))
    + " worker=0x" + juce::String::toHexString(static_cast<uint64_t>(masks.worker))
    + " learner=0x" + juce::String::toHexString(static_cast<uint64_t>(masks.learnerMain))
    + " heavyBG=0x" + juce::String::toHexString(static_cast<uint64_t>(masks.heavyBackground))
    + " lightBG=0x" + juce::String::toHexString(static_cast<uint64_t>(masks.lightBackground))
    + " ui=0x" + juce::String::toHexString(static_cast<uint64_t>(masks.ui)));
```

**出力例 (4C8T)** :
```
[AFFINITY] coreTopology: physical=4 logical=8 heterogeneous=false
[AFFINITY] audioMask=0x88 nonAudio=0x77 worker=0x11 learner=0x22 heavyBG=0x77 lightBG=0x77 ui=0x77
```

**出力例 (P/E混在)** :
```
[AFFINITY] coreTopology: physical=16 logical=24 heterogeneous=true
[AFFINITY] audioMask=0x00 nonAudio=0x00 worker=0x00 learner=0x00 heavyBG=0x00 lightBG=0x00 ui=0x00
```

> v14: 全マスクがゼロの場合は P/E 環境でアフィニティが無効化されていることを示す。起動時のログだけで状況を完全に把握できる。

#### (g) computeSymmetricMasks() 静的メソッド（新規）

対称コア環境専用。`CoreTopology` 全体を引数に受けることで、将来の拡張（NUMA, processor group 等）にもインターフェース変更が不要。

```cpp
static ThreadAffinityMasks computeSymmetricMasks(const CoreTopology& topo) noexcept;
```

```
// ★ v20: cores.size() を基準に計算（physicalCoreCount はログ表示用）
const size_t N = topo.cores.size();
若 N < 2 → ThreadAffinityMasks{}（全ゼロ、アフィニティ無効）

audioMask      = topo.cores[N-1].mask
nonAudioMask   = topo.cores[0].mask | ... | topo.cores[N-2].mask

masks_.audioRealtime   = audioMask
masks_.worker          = topo.cores[0].mask
masks_.learnerMain     = topo.cores[min(1, N-2)].mask
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

    if (topo.physicalCoreCount == 0) {
        // ★ v16: detectCoreTopology が空を返した（API失敗）→ アフィニティ無効
        ThreadAffinityMasks noAffinity{};
        affinityManager.initialize(noAffinity);
        hasHeterogeneousCores_ = false;
        diagLog("[AFFINITY] GetLogicalProcessorInformationEx failed: Affinity disabled.");
    } else if (topo.hasHeterogeneousArchitecture) {
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

### 5.4 AudioEngine.Timer.cpp — applyMmcssPriority() 拡張（v14 最終）

L272 の `}` 直後（関数末尾の `}` の前）に追加。`applyMmcssPriority()` は L214-272。

**v14 で `SetThreadIdealProcessorEx` を削除**: audioMask は1物理コアのみ（SMT兄弟のみ）であり、Windows Scheduler が SMT sibling を自動選択するため、IdealProcessor の実質効果は無い。`_BitScanForward64` / `<intrin.h>` の依存も同時に削除。

```cpp
    // ★ [work64 v16] Audioスレッド CPUアフィニティ固定（対称コア環境のみ）
    //   NOTE(v16): toHexString は JUCE template のため uint64_t を直接受け付ける
    //     (JUCE/modules/juce_core/text/juce_String.h:1124 確認済)。
    //   v14: SetThreadIdealProcessorEx は削除 (audioMask=1物理コアのみで効果無し)。
    if (!hasHeterogeneousCores_) {
        const DWORD_PTR audioMask = affinityManager.getAudioRealtimeMask();
        if (audioMask != 0) {
            const DWORD_PTR prevMask = ::SetThreadAffinityMask(
                ::GetCurrentThread(), audioMask);
            if (prevMask == 0) {
                const DWORD err = ::GetLastError();
                diagLog("[AFFINITY] FAILED: mask=0x"
                        + juce::String::toHexString(static_cast<uint64_t>(audioMask))
                        + " GetLastError=" + juce::String(static_cast<int>(err)));
            }
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
            else {
                diagLog("[AFFINITY] AudioThread pinned mask=0x"
                        + juce::String::toHexString(static_cast<uint64_t>(audioMask))
                        + " prev=0x" + juce::String::toHexString(static_cast<uint64_t>(prevMask)));
            }
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

**v16 補足**: `SetThreadAffinityMask` のみ使用。`SetThreadIdealProcessorEx` / `_BitScanForward64` / `PROCESSOR_NUMBER` は削除済み。`toHexString` は `static_cast<uint64_t>` に統一。JUCE 8.0.12 の `toHexString` は template (`juce_String.h:1124`) で任意の整数型を受け付けるため、`uint64_t` は曖昧さなく解決される。

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
| 対称N=2でWorker/Learner/UIがCPU[0]集中 | 低 | デュアルコアでは不可避。Audio1コア専用化の利益が勝る |
| GetLogicalProcessorInformation API失敗 | 低 | 全ゼロマスク（アフィニティ無効）— SMT topology 検出不能時にマスクを仮定するリスクを回避 |
| MMCSS後にSetThreadAffinityMask | 極低 | MMCSSはアフィニティに触れない（SDKドキュメント確認済） |
| **★ v7 追加**: `initWorkerThread()` の呼び出しが `affinityManager.initialize(masks)` より先 (AudioEngine.Init.cpp L82 vs L97) | 中 | 本計画で順序入替: L82削除 + L99直後に移動。現状バグではWorkerThread::run()が`initialized_==false`判定でshort-circuitしaffinity永久未適用 |
| **★ v13 追加**: ISR/DPC が CPU 0 に集中する影響で Worker スレッド (CPU0) が割込みを受ける可能性 | 低 | 対称コア環境で worker に CPU[0] を割り当てている。Windows 11 は旧OSよりインテリジェントに割込みを分散するが、一部レガシードライバでは CPU0 に DPC が集中し得る。worker は軽量コマンド処理のみのため実害は軽微。将来の拡張として worker を CPU[0] 以外に割り当て可能な設計 (例: `cores[1].mask`) をコメントに残す。 |
| **★ v15 追加**: API 失敗フォールバック時は全ゼロマスク（アフィニティ無効）を返す。SMT topology が検出不能な状態でマスクを仮定する危険性を回避。 | 極低 | 対象環境 (Windows 10+) では API は正常動作するため到達しないパス。安全側への倒し方として妥当。 |

---

## 8. 変更不要ファイル（自動適合）

| ファイル | 使用マスク | 自動適合理由 |
|---|---|---|
| NoiseShaperLearner.cpp L524,725 | LearnerEval, LearnerMain | learnerEvalBase が複数コア → getEvalWorkerMask() が自然分散 |
| DeferredFreeThread.h L152 | LightBackground | nonAudioMask 自動反映 |
| LoaderThread.cpp L39 | HeavyBackground | nonAudioMask 自動反映 |
| ProgressiveUpgradeThread.cpp L76 | HeavyBackground | 同上 |
| WorkerThread.cpp L59 | Worker | cores[0].mask 自動反映 |
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

## 10. 実装順序（v18 更新）

| Step | 内容 | ファイル | 確認 |
|---|---|---|---|
| 1 | AudioRealtime型+フィールド+分岐+アクセサ（二重適用注記含む） | ThreadAffinityManager.h | コンパイル |
| 2 | PhysicalCoreInfo構造体+CoreTopology+detectCoreTopology（diagLog非依存 / Size==0ガード / offset+Size<=bufLen境界 / noexcept+bad_alloc注釈）+computeSymmetricMasks(const CoreTopology&)（整合性チェック含む） | ThreadAffinityManager.h | コンパイル |
| 3 | `getAffinityManager()` 既存アクセサ利用確認 (L1058-1059 既存在) | (追加実装不要、参照確認のみ) | — |
| 4 | hasHeterogeneousCores_メンバ追加 | AudioEngine.h L2301付近 | コンパイル |
| 5 | initialize()内マスク設定を動的計算に置換 + **`initWorkerThread()` を initialize() の「後」に移動** (v7 必須) + **`physicalCoreCount==0` 分岐を追加** (v16) + 起動時診断ログ追加 (v14/v15) | AudioEngine.Init.cpp L82(削除)→L99(後) + L87-99(置換) | コンパイル + 順序確認 |
| 6 | applyCurrentThreadPolicy()のSetThreadAffinityMaskをprevMaskラップ（v17） | ThreadAffinityManager.h | コンパイル |
| 7 | applyMmcssPriority()末尾にアフィニティ固定（v14: IdealProcessor/BitScan削除、v16: toHexString uint64_t確認） | AudioEngine.Timer.cpp L272 直後 | コンパイル |
| 8 | Release+DIAGNOSTICSビルド・実行検証 | build.bat | 診断ログ+ProcessExplorer |
| 9 | CTest 全テストスイート通過確認 | ctest | 全PASS |

> **注意**: 旧 Step 5「BlockDouble.cpp MMCSS ブロック追加」は **v3 で撤回**（§3.2注記参照）。両オーディオパスとも既に `mmcssApplied_` 共有フラグで MMCSS 呼出済み。

> **v7 追加**: Step 5 で **順序入替が必須**。WorkerThread 起動（L82）が initialize()（L97）よりも先に来ると、WorkerThread 内の `applyCurrentThreadPolicy(Worker)` が `initialized_==false` で short-circuit し、結果として WorkerThread の affinity は永遠に適用されない。最終形では L82 の `initWorkerThread();` を削除し、`affinityManager.initialize(masks);` の直後（L99 直後）に移動すること。

---

## 付録

## A. 改訂履歴

| 改訂 | 日付 | 内容 |
|---|---|---|
| v1 | 2026-07-05 | 初版策定 |
| v2 | 2026-07-05 | 妥当性検証に基づく大幅修正 (API仕様誤りの修正、SDK要件の訂正、コンパイル時依存の明記、64論理プロセッサ上限問題への対処) |
| v3 | 2026-07-05 | WSL検証で発見: BlockDouble.cpp MMCSSは既に実装済み（mmcssApplied_使用）。両パス unified。plan §5.5は誤り。Plan §3.2「BlockDouble.cpp なし（バグ）」を取消し。 |
| v4 | 2026-07-05 | MSDN 公式ドキュメント6件 + WSLシミュレーション9テストでプロセッサアフィニティマスクの全仕様を検証。`PROCESSOR_RELATIONSHIP.GroupCount==1` (MSDN明記)、`KAFFINITY==DWORD_PTR`、`SetThreadAffinityMask` Win11 プライマリグループ制約、`EfficiencyClass` 意味論を公式仕様と突合し修正。`prevMask==0` エラー処理の推奨追加。
| v5 | 2026-07-05 | MSDN公式8件 + WSL 15テストで最終検証。`topo.physicalCoreMasks.size` 括弧欠落バグ修正、行番号ズレ修正(L2301, Timer.cpp L214-272)、WOW64 folding明記、6C期待値修正。`Affinity` REG_DWORD = 0x00/0xFFFFFFFF は MMCSS公式ドキュメントで確認済み。 |
| v6 | 2026-07-05 | WSL S6 15テストで詳細検証。3点の文書精緻化: ①§5.4 `static_cast<int>(audioMask)` は int=32bitで64bitマスク truncation の誤解解消 (対象HWでは <=0xFFFFFFFF なのでOK)。②§5.3 delete範囲 L87-98 → L87-L99 inclusive に修正。③§11.1 #9: `_WIN32_WINNT` は CMake ではなく JUCE `JUCE/modules/juce_core/native/juce_BasicNativeHeaders.h:103-104` で必ず `0x0A00`(Win10) に設定されることを明記。 |
| v7 | 2026-07-05 | **重大バグ発見**: AudioEngine.Init.cpp L82 `initWorkerThread()` が L97 `affinityManager.initialize(masks)` より前に呼ばれている現状。WorkerThread::run() の `applyCurrentThreadPolicy(Worker)` が `initialized_==false` で short-circuit → Worker affinity が永久に未適用。WSL S7 7テストで順序 Race condition 検証。プラン §5.3 で L82 → L99直後 に**移動**必須。§6 データフロー、§10 実装順序、§7 リスク評価 を v7 で更新。 |
| v8 | 2026-07-05 | WSL S8 9テストで Before/After 順序シミュレーション実行。既存バグ (S8-1) と修正後 (S8-2) の挙動差分を pthread で実機確認。S8-3〜S9 まで全 9テスト PASS。 |
| v9 | 2026-07-05 | MSDN 追加調査 3件: `GetProcessAffinityMask`、`SetProcessAffinityMask` (Win11 primary group 制約 — 本計画では不使用のため安全)、`SetThreadIdealProcessor` (preferred processor hint — `SetThreadAffinityMask` のハード固定が dominant、干渉なし)。行番号 L214 微細訂正。§11.2(J) に追加結果を追記。 |
| v10 | 2026-07-05 | deep-dive 検証 4項目: ①MSDN CPU Sets vs affinity mask (K1: restrictive affinity dominant 確認)、②MainApplication.cpp:79-89 EcoQoS 無効化との相互作用 (K2: 独立動作), ③Audio Thread 生成元 (K3: Windows Audio Engine 管理、`PROC_THREAD_ATTRIBUTE_GROUP_AFFINITY` 不要), ④DeferredFreeThread 遅延作成タイミング (K4: v7 修正で自然にカバー)。§11.2(K) を新規追加。 |
| v11 | 2026-07-05 | 新規バグ監査 B1-B7 (実装前最終) を実施。§11.2(L) 追加。B2 (mmcssApplied_ reset/CAS 競合なし), B4 (LearnerEval は learnerEvalBase 使用、audioRealtime マスク独立で影響なし), B5 (子プロセス API 不使用), B6 (nullptr チェック既備), B7 (diagLog 順序 設計上問題なし)。新規バグ発見なし。 |
| v12 | 2026-07-05 | MSDN公式9件 + 実コード13ファイルの外部妥当性検証完了。`juce_BasicNativeHeaders.h` のファイルパスを `native/` に訂正 (旧 `system/`)。§11.3 に検証報告書への参照を追加。全主要主張がMSDN公式＋実コードと一致することを確認。 |
| v13 | 2026-07-05 | ユーザーレビュー評価に基づく改善: §2 設計方針にP/E heterogeneous ポリシーの将来拡張性コメントを明記。§5.4 に `SetThreadIdealProcessorEx` 併用の推奨を追加。§5.1(h) に `ERROR_INVALID_PARAMETER` の診断ログを強化。§7 に ISR/DPC CPU0 集中リスクを追加。§5.1 に `applyAudioThreadPolicy()` 独立関数化の代替案を追記。§B.2(M)(N)(O)(P) に new deep-dive 4項目を追加。全改訂点のMSDN調査＋実コード検証完了。 |
| v14 | 2026-07-05 | ユーザーレビュー評価に基づく全面的改善: ①`SetThreadIdealProcessorEx`/`_BitScanForward64`/`applyAudioThreadPolicy()` を全て削除（audioMaskは1物理コアのみでIdealProcessorの効果が無いため）。②`detectCoreTopology` の2回目 API 失敗時のフォールバックを `GetActiveProcessorCount` に修正（空の topo を返さない）。③`physicalCoreMasks` を最下位ビット（論理CPU番号）順にソートする処理を追加（MSDNが列挙順を保証しないため）。④起動時に物理コア数・論理コア数・P/E有無・全マスク値を一括出力する診断ログを追加。 |
| v15 | 2026-07-05 | ユーザーレビュー評価に基づく構造的改善: ①`CoreTopology` の `physicalCoreMasks`＋`efficiencyClasses` 別vectorを廃止し、`std::vector<PhysicalCoreInfo> cores` に統合（ソート時の同期ずれバグを根絶）。②API 失敗フォールバックを全ゼロマスク（アフィニティ無効）に変更。③`toHexString` のキャストを `static_cast<uint64_t>` に統一。④`AudioRealtime` ThreadType の用途注記を追加。 |
| v16 | 2026-07-05 | ユーザーレビュー評価に基づく依存関係整理: ①`ThreadAffinityManager::detectCoreTopology()` から `diagLog` 呼び出しを削除（ThreadAffinityManager は純粋ユーティリティとして設計、ログは呼び出し側の AudioEngine.Init.cpp で行う）。②`toHexString(uint64_t)` の安全性を JUCE 8.0.12 ソースコードで確認（template のため曖昧さなし）。③`lowestBit()` while ループは `_BitScanForward64` 相当の代替として妥当性確認。 |
| v17 | 2026-07-05 | ユーザーレビュー評価に基づく堅牢性改善: ①`detectCoreTopology()` のレコード走査に `info->Size==0` ガードと `sizeof` 境界チェックを追加（無限ループ防止）。②`computeSymmetricMasks()` に `cores.size()!=physicalCoreCount` 入力整合性チェックを追加。③`applyCurrentThreadPolicy()` の `SetThreadAffinityMask` 呼び出しを `prevMask` でラップし診断追跡可能に。④`detectCoreTopology() noexcept` に `bad_alloc→terminate` の注釈を追加。 |
| v18 | 2026-07-05 | ユーザーレビュー評価に基づくコードパターン改善: ①`detectCoreTopology()` の while ループ条件を `offset+Size<=bufLen` 方式に変更（Windows SDK 可変長レコード走査の標準パターンに一致）。②`computeSymmetricMasks()` の引数を `const CoreTopology& topo` に一本化（将来拡張に強い）。③`AudioRealtime` ThreadType のコメントに二重適用の注記を追加（責務は applyMmcssPriority 側にあり、enum は将来リファクタリング用）。 |
| v19 | 2026-07-05 | コードベース全スレッド棚卸し完了 report. 全8スレッドカテゴリの網羅性を確認。IUCE ThreadPool 2件（IR preview / save）は短命バックグラウンド処理のためアフィニティ設定不要と確定。`convo::numeric_policy::ThreadRole::AudioRealtime` と `ThreadType::AudioRealtime` の名前衝突がないことを確認（異なる名前空間/列挙型）。`GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)` のMSDN仕様を再確認。 |
| v20 | 2026-07-05 | ユーザーレビュー評価に基づくコード改善: ①`lowestBit()` を `std::countr_zero` (C++20) に変更（whileループより可読性向上）。②`while(offset<bufLen)` に最低限のヘッダサイズ保証 `sizeof(...)` を追加。③リスク表のフォールバック記述を「全ゼロマスク」に修正（本文と統一）。④`computeSymmetricMasks()` の計算基準を `physicalCoreCount` → `cores.size()` に変更。⑤`noexcept` コメントを改善。`<bit>` のインクルード注記を追加。 |
| v21 | 2026-07-05 | **最終承認レビュー (A-)**。最終推奨点: `detectCoreTopology()` に `proc.GroupMask[0].Mask != 0` の防御チェックを追加（破損データ対策）。全21回の改訂を経て、設計書は実装可能な水準に達した。 |

> **重要**: §5詳細設計は v2 で修正済みです。旧版の `GetLogicalProcessorInformation`、`SYSTEM_LOGICAL_PROCESSOR_INFORMATION(u.Processor.EfficiencyClass)`、EfficiencyClass 値「1=E,2=P」はいずれも誤りであった。正しくは `GetLogicalProcessorInformationEx` + `PROCESSOR_RELATIONSHIP::EfficiencyClass` を使用する。

---

## B. 妥当性検証結果

本計画の妥当性を実際のコードベースと Windows API 仕様書で検証した結果を以下に示す。

### B.1 検証対象と結果

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
| 9 | Windows SDK EfficiencyClass 互換性 | ✅ OK (v6 詳細化) | **JUCE が `JUCE/modules/juce_core/native/juce_BasicNativeHeaders.h:103-104` で `_WIN32_WINNT = _WIN32_WINNT_WIN10` (= `0x0A00`, Win10) を必ず設定する** — CMake 設定に依存せずWin10ベースラインが保証される。`GetLogicalProcessorInformationEx` は Win7+ (`_WIN32_WINNT >= 0x0601`)、`EfficiencyClass` は Win10+ で有効。両者とも充足 |
| 10 | 64論理プロセッサ上限問題 | ✅ 対処済み | 非Ex版 `GetLogicalProcessorInformation` は「呼び出しスレッドが属する単一グループ内の最大64論理プロセッサ」のみ取得。Ex版は全グループを取得。本計画は Ex版を使うため問題ない。ただし計画の前提（単一プロセッサグループ、<=64論理CPU）を明記済み |
| 11 | `NoiseShaperLearner.cpp` の `getEvalWorkerMask` 自動適合 | ✅ OK | L524 で `ThreadType::LearnerEval, workerIndex` を呼出。`learnerEvalBase = nonAudioMask` (例: 4C8T で 0x77 = bits 0,1,2,4,5,6 で 6ビット) となる。`getEvalWorkerMask` は `bits[]` 配列に立っているビットを順に格納し `workerIndex % count` でラウンドロビン。eval ワーカー数が 6 以下なら 1ワーカー=1ビットに自然分散。変更不要 |
| 12 | `<vector>` / `<algorithm>` ヘッダ追加必要性 | ⚠️ 計画に明記済み | `ThreadAffinityManager.h` には `#include <vector>`（`std::vector<PhysicalCoreInfo>`）と `#include <algorithm>`（`std::sort`）の追加が必要。§5.1 冒頭注記に明記済み |
| 13 | `build.bat` / CMake でのビルド検証可能性 | ✅ OK | CMake L732-734 で `avrt` ライブラリ既にリンク済み。`GetLogicalProcessorInformationEx` は `Kernel32.lib` に含まれ、既存の Win32 リンクで充足。新規ライブラリ追加不要。`build.bat` Release/Debug でビルド可能 |

### B.2 検証で発見された追加事項

#### (A) プロセッサグループ関連

本計画は**単一プロセッサグループ（全論理プロセッサ <= 64）**を前提とする。対象環境（第4世代 4C8T、第12世代 8P+8E = 16C32T）では常に満たされる。マルチソケットNUMA や Intel Xeon W-3175N (28C56T, 単一グループ) 等は問題ないが、AMD Threadripper 3990X (64C128T, 2グループ) やサーバー NUMA システムでは `GROUP_AFFINITY.Group` 番号の保持と `SetThreadGroupAffinity` API への拡張が必要。本計画の対象外。

#### (B) ~~BlockDouble.cpp MMCSS 欠落の修正位置~~ 【撤回 v3】

> **撤回 (v3)**: §3.2 の 注記を参照。`BlockDouble.cpp` への MMCSS ブロック追加は不要。

#### (C) `hasHeterogeneousCores_` のフォールバック

`detectCoreTopology()` が API 失敗で空の `CoreTopology` を返した場合（v15 フォールバック変更済み）、`physicalCoreCount == 0` となり `computeSymmetricMasks()` は `N < 2` の早期 return で `ThreadAffinityMasks{}`（全ゼロ）を返す。この場合 `hasHeterogeneousCores_ = false` だが `audioMask == 0` のため `applyMmcssPriority()` 末尾の `if (audioMask != 0)` でスキップされ、安全にアフィニティ未設定となる。破綻なし。

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
| E8 | `RelationProcessorCore` 全グループ返却 | MSDN: "for every active processor core in every processor group" | Plan: §B.2(A)で >64LP 対象外と明記 | ✅ 明記済み |
| E9 | MMCSS `AvSetMmThreadCharacteristics` Affinity | MSDN: MMCSS Tasks registry `Affinity` REG_DWORD 0x00/0xFFFFFFFF = no affinity | Plan: §B.1 #8 で MMCSS はアフィニティに触れないと明記 | ✅ 一致 |

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
| G4 | N=0 安全 | `cores[N-1].mask` アクセスなしの早期 return | ✅ PASS |
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
| G15| `_WIN32_WINNT` 強制設定 | JUCE `JUCE/modules/juce_core/native/juce_BasicNativeHeaders.h:103-104` で必ず `0x0A00` (Win10) | ✅ v6 §B.1 #9 修正 |

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

#### (M) v13 P/E heterogeneous ポリシー詳細検証（v13 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| M1 | AMD Zen5c が EfficiencyClass を使用する可能性 | MSDN: "EfficiencyClass is only nonzero on systems with a heterogeneous set of cores." — AMD/Intel/ARM いずれの CPU でも heterogeneous 構成では非ゼロになる。Zen5c で EfficiencyClass を使用する場合、本設計の不等号検出 (`ec != first`) で正しく heterogeneous と判定される。`0` 対 `1` でも `2` 対 `3` でも原理は不変。 | ✅ 互換性維持 |
| M2 | Intel LPE (Low Power E-core) が追加された場合の影響 | LPE が E-core と異なる EfficiencyClass 値を持った場合も、単一値グループ判定で heterogeneous になる。追加コアタイプが増えてもロジック変更不要。 | ✅ 拡張性十分 |
| M3 | 「heterogeneous = affinity 無効化 = 性能劣化」という誤解の防止 | P/E 環境でアフィニティを設定しないのは「OS に委任する方が有利」というポリシー選択である。MSDN CPU Set ドキュメントで `SetThreadAffinityMask` が CPU Set より dominant であることが確認されており、むしろ affinity を設定することで Thread Director の判断を阻害するリスクを回避する。 | ✅ コメント明記で対応 |
| M4 | EfficiencyClass 0 の homogeneous 環境混在リスク | 第4世代〜第10世代まですべて EfficiencyClass=0。M1 以降の Apple Silicon では非ゼロだが本計画の対象外 (Windows x64)。同一値グループ判定で全コア同一ならば正しく homogeneous。全コア同一で非ゼロ（hypothetical な将来 CPU）も homogeneous として扱われ、末尾固定される。問題なし。 | ✅ 安全 |

#### (N) v13 SetThreadIdealProcessorEx 調査（v13 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| N1 | `SetThreadIdealProcessorEx` API シグネチャ | `BOOL SetThreadIdealProcessorEx(HANDLE, PPROCESSOR_NUMBER, PPROCESSOR_NUMBER)` — Win7+ (`_WIN32_WINNT >= 0x0601`)。DLL: Kernel32.dll。 | ✅ 利用可能 |
| N2 | ソフトヒント vs ハードアフィニティの関係 | MSDN: "provides a hint to the scheduler about the preferred processor for a thread." — hint であり強制力はない。`SetThreadAffinityMask` がハード制約、`SetThreadIdealProcessorEx` がソフトヒント。両者の併用は MSDN で禁止されておらず、むしろスケジューラに「このコアを優先的に使いたい」意図を伝える相補的関係。 | ✅ 併用推奨 |
| N3 | `SetThreadAffinityMask` との呼び出し順序 | `SetThreadAffinityMask` によるハード制約 → `SetThreadIdealProcessorEx` によるソフトヒント の順が自然。逆順でも動作に差異はない (MSDN に順序依存の記述なし)。 | ✅ §5.4 の順序で問題なし |
| N4 | `_BitScanForward64` の互換性 | MSVC: `<intrin.h>` の `_BitScanForward64`。x64 ビルド限定。GCC/Clang: `__builtin_ctzll`。ConvoPeq は MSVC ビルド (CMake `-DCMAKE_C_COMPILER=cl`) のため MSVC 組み込み関数で十分。 | ✅ §5.1(g) に注意書き |
| N5 | `PROCESSOR_NUMBER` 構造体 | `{WORD Group, BYTE Number, BYTE Reserved}` — Win7+ で利用可能。`Group=0` 前提。`Number` は group-relative 0-based。 | ✅ 問題なし |

#### (O) v13 CPU0 ISR/DPC 集中リスク検証（v13 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| O1 | Windows 11 での DPC ルーティング | MSDN DPC ドキュメント: DPC は「arbitrary DPC context」で実行される。旧来の「CPU0 優先」動作は Windows 8+ で改善され、Windows 11 では割込みコントローラがインテリジェントに分散。ただし特定のレガシードライバ (特にオーディオ/NIC) では CPU0 に DPC が偏る可能性がある。 | ✅ リスク認識済み — §7 に追記 |
| O2 | Worker スレッドが CPU0 固定の影響 | Worker は `ParameterCommand` の軽量ポップ処理のみ (WorkerThread.cpp:65-98)。DPC による瞬間的な割込みが発生しても、Worker の処理遅延がオーディオに直接影響することはない。ただし重い rebuild 処理は `rebuildThread` (HeavyBackground) が担当し、CPU[0] ではない。 | ✅ 影響軽微 |
| O3 | 将来の CPU0 除外設計拡張 | `computeSymmetricMasks` で `worker` に `cores[0].mask` ではなく `cores[1].mask` を割り当てるよう変更可能。現設計でもコメントで `worker` 割り当てポリシーが可変であることを明記すれば十分。 | ✅ §5.1(g) で注記 |

#### (P) v13 applyAudioThreadPolicy() 独立関数化の評価（v13 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| P1 | `ThreadType::AudioRealtime` enum 追加方式の利点 | 既存の `applyCurrentThreadPolicy` switch 文に 1 case 追加するだけで済む。`getEvalWorkerMask` との統一性がある。コード変更量最小。 | ✅ 主実装として採用 |
| P2 | 独立 `applyAudioThreadPolicy()` 方式の利点 | AudioRealtime に特化したエラーハンドリング (`SetThreadIdealProcessorEx` 併用、優先度設定スキップの明示) を独立管理できる。可読性が高い。enum 追加方式の上位ラッパーとして提供可能。 | ✅ 両立可能 — §5.1(g) で両方式を定義 |
| P3 | 両方式の共存方法 | `applyCurrentThreadPolicy(AudioRealtime)` は switch で `return;` し、`applyAudioThreadPolicy()` はその上位ラッパーとして `SetThreadIdealProcessorEx` のロジックを含める。enum ケースは `return;` で優先度設定をスキップ、専用メソッドは IdealProcessor を追加設定。排他ではなく直列。 | ✅ 設計統合完了 |


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

#### (Q) v14 SetThreadIdealProcessorEx 削除の根拠（v14 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| Q1 | `SetThreadIdealProcessorEx` の効果範囲 | MSDN: "provides a hint to the scheduler" — hint のみ。ハード制約 (`SetThreadAffinityMask`) で CPU 集合が 1 物理コアに制限されている場合、IdealProcessor の候補は SMT 兄弟の範囲のみ。Windows Scheduler が自動選択するため付加価値は極めて低い。✅ 削除妥当 |
| Q2 | `_BitScanForward64` 不要 → `<intrin.h>` 依存除去 | `SetThreadIdealProcessorEx` を削除すれば `_BitScanForward64` も不要。MSVC 依存が減る。✅ 依存削除 |
| Q3 | `PROCESSOR_NUMBER` 構造体の不使用 | 単一グループ前提 (`Group=0`) のコードが削除される。マルチグループ対応の将来拡張に使う可能性はあるが、本計画の対象外。✅ 削除妥当 |
| Q4 | v13 からの変更によるデグレードリスク | `SetThreadAffinityMask` は維持。IdealProcessor 削除で Audio スレッドのコア固定動作に影響なし。✅ リスクなし |

#### (R) v14 detectCoreTopology 2回目API失敗フォールバック修正（v14 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| R1 | 2回目 API 失敗の原因 | `GetLogicalProcessorInformationEx` の2回目呼出 (実バッファ版) が失敗する可能性: メモリ不足、バッファサイズ競合など。`bufLen` が1回目の呼出後から変化した場合に発生し得る。✅ レアケースだが考慮 |
| R2 | 修正前の挙動 | `return topo;` (physicalCoreCount=0) → `computeSymmetricMasks(N<2)` → 全ゼロマスク → 全Affinity無効。API の一時的失敗で全アフィニティが機能しなくなる。❌ 不完全 |
| R3 | 修正後の挙動 | `GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)` で論理CPU数取得、1ビットずつマスク生成。SMT 検出不可だがデフォルトよりマシ。✅ 改善 |
| R4 | `ALL_PROCESSOR_GROUPS` (=0xFFFF) | MSDN: "returns the total number of active processors across all processor groups." ✅ 正しい |

#### (S) v14 physicalCoreMasks ソート追加の根拠（v14 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| S1 | MSDN 列挙順保証の有無 | `GetLogicalProcessorInformationEx` の MSDN ドキュメント: 列挙順に関する保証なし。「returns a structure for every active processor core」のみ。✅ ソート必須 |
| S2 | ソートなしのリスク | Windows Update やファームウェア更新で列挙順が変わった場合、「末尾=最大番号」の前提が崩れ、Audio スレッドが不適切なコアに固定される可能性がある。✅ リスク顕在化 |
| S3 | ソート計算量 | cores.size() ≤ 64。`std::sort` O(N log N) ≈ 384 比較 → 起動時1回のみ。性能影響ゼロ。✅ 問題なし |
| S4 | efficiencyClasses との同期（v15 で構造体化により解決） | v14 では「pair で保持」としていたが、v15 で `PhysicalCoreInfo` 構造体を導入し `cores[]` に統合。`mask` と `efficiencyClass` が常に同一オブジェクト内で保持されるため、ソート時の同期ずれは原理的に発生しない。❌ 別vector → ✅ 構造体化 |

#### (T) v14 起動時診断ログ追加の評価（v14 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| T1 | 出力内容 | 物理コア数・論理コア数・P/E有無・全7マスク値。P/E環境では全ゼロで明確表示。✅ 十分 |
| T2 | 出力タイミング | `AudioEngine.Init.cpp` の `affinityManager.initialize(masks)` 直後。WorkerThread 起動前に確定。✅ 適切 |
| T3 | 既存 diagLog との統合 | 現在 `Init.cpp:98` のハードコードマスク `[AFFINITY] initialized` ログを v14 診断ログで置換。削除漏れ防止。✅ 統合済み |

#### (U) v15 PhysicalCoreInfo 構造体導入（v15 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| U1 | `PhysicalCoreInfo` 構造体の必要性 | v14 までの別vector方式ではソート時に物理コアマスクと効率クラスの対応が崩れるリスクがあった。`PhysicalCoreInfo` 構造体で両者を一体化し、ソート時の同期ずれを原理的に防止。✅ 必須対応 |
| U2 | 別vector方式の具体的なバグシナリオ | 列挙順が `0x22,0x11,0x88,0x44` だった場合、別vector方式ではソート後に mask と ec の対応がずれる。構造体方式では常に同一オブジェクト内で保持されるためズレが発生しない。✅ 構造体で解決 |
| U3 | API 失敗フォールバックの変更根拠 | v14 までは `GetActiveProcessorCount` で論理CPU数を取得し1ビットずつマスクを生成 → SMTなし扱い。v15 では全ゼロマスク（アフィニティ無効）に変更。SMT topology 検出不能時にマスクを仮定する危険性を回避。Windows 10+ では API が失敗することは事実上ないため、フォールバックは安全網。✅ 安全側に倒す |
| U4 | `toHexString` の `uint64_t` 統一 | v14 までは `static_cast<int>` と `static_cast<int64_t>` が混在。`uint64_t` に統一。JUCE 8.0.12 の `toHexString` は template (`juce_String.h:1124`) で任意の整数型を受け付けるため、`uint64_t` は曖昧さなく解決される。`juce_String.cpp:1942` で `createHex(uint64)` が呼ばれる。✅ 可読性向上・検証済み |

#### (V) v16 ThreadAffinityManager 純粋ユーティリティ化と依存関係整理（v16 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| V1 | `ThreadAffinityManager` から `diagLog` 呼び出しを削除する理由 | `ThreadAffinityManager.h` は `src/core/` に属する汎用ユーティリティクラス。`diagLog` は `AudioEngine` の無名名前空間で定義された内部関数であり、`ThreadAffinityManager` がこれに依存することは不適切（循環依存ではないが、責務分離に反する）。`detectCoreTopology()` は `CoreTopology` を返すだけの純粋関数とし、ログは呼び出し側 (`AudioEngine.Init.cpp`) で行う。 | ✅ 責務分離改善 |
| V2 | 呼び出し側の `physicalCoreCount == 0` チェック | `AudioEngine.Init.cpp` の初期化ブロックで `topo.physicalCoreCount == 0` を判定し、その場合も `noAffinity{}` で `initialize()` を呼ぶ。これにより `initialized_ = true` が保証され、他のスレッドが `applyCurrentThreadPolicy` を呼んでも short-circuit しない。 | ✅ 安全性維持 |
| V3 | JUCE `toHexString(uint64_t)` の互換性検証 | JUCE 8.0.12 ソースコード確認: `juce_String.h:1124` で `template <typename IntegerType> static String toHexString(IntegerType number)` として宣言。`juce_String.cpp:1942` で `String::createHex(uint64 n)` が実装済み。`uint64_t`（MSVC では `unsigned long long`）はテンプレート実引数として曖昧さなく解決される。ユニットテスト (`juce_String.cpp:2628`) でも `String::toHexString((size_t) 0x12ab)` が使用されており、任意の整数型の受け入れが確認済み。 | ✅ 安全確認 |
| V4 | `lowestBit()` while ループの評価 | while ループによる最下位ビット検出は最大64回の走査。`_BitScanForward64` (MSVC intrinsic) と比較して性能差は無視できる (64回 vs 1命令、起動時1回のみ)。`<intrin.h>` 依存を避けられる利点がある。現状の while ループで問題なし。 | ✅ 現状維持で十分 |

#### (W) v17 堅牢性改善 — Size==0 ガード・境界チェック・整合性チェック・診断統一（v17 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| W1 | `info->Size==0` ガードの必要性 | `GetLogicalProcessorInformationEx` が正常なら Size≥sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)=8 を返す。将来の API 変更やメモリ破損で Size=0 になった場合、`while(offset<bufLen)` が無限ループになる。`if(info->Size==0) break;` で防御。MSDN に Size==0 の定義はないが、防御的コーディングとして妥当。 | ✅ 予防的防御 |
| W2 | `while(offset+sizeof(...)<=bufLen)` 境界チェック | 元の `while(offset<bufLen)` は末尾1バイト未満の状態でもループ内に入る可能性がある。`sizeof` を加算することで、最低1レコード分の領域があることを保証。Windows API は通常正しいが、メモリ破損対策として有効。 | ✅ 堅牢性向上 |
| W3 | `computeSymmetricMasks()` 入力整合性チェック | `cores.size()` と `physicalCoreCount` の不一致は `GroupCount!=1` skip 等で発生する。放っておくと `cores[N-1]` が out-of-range になる可能性がある。`return ThreadAffinityMasks{}`（全ゼロ）で安全側に倒す。 | ✅ 安全 |
| W4 | `applyCurrentThreadPolicy()` の `prevMask` ラップ | v16 までは `SetThreadAffinityMask` の戻り値を捨てていた。AudioThread 側では `prevMask==0` チェック済み。`applyCurrentThreadPolicy` 側でも `prevMask` を取得することで、診断ログ有効時 (`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`) に各スレッドのアフィニティ適用結果を追跡可能になる。`juce::ignoreUnused(prevMask)` で非診断ビルドでは最適化消去される。 | ✅ 統一 |

#### (X) v18 コードパターン改善 — while 条件・computeSymmetricMasks 引数・AudioRealtime 注記（v18 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| X1 | while 条件を `offset+Size<=bufLen` 方式に変更 | 元の `while(offset+sizeof(...)<=bufLen)` は Size=0 ガードと組み合わせて防御できていたが、Windows SDK の可変長レコード走査では`while(offset<bufLen)` を条件に、ループ内で `Size==0` と `offset+Size>bufLen` を個別にチェックするパターンが標準。可読性と一貫性の観点から変更。 | ✅ 標準パターンに一致 |
| X2 | `computeSymmetricMasks(const CoreTopology&)` への変更 | `CoreTopology` 構造体が既にあるため、引数を構造体全体に一本化。呼び出し側 (`AudioEngine.Init.cpp`) では既に `topo` を渡しているため変更不要。将来 NUMA/processor group 情報を追加しても関数シグネチャ不変。 | ✅ 将来拡張性向上 |
| X3 | AudioRealtime 二重適用のコメント明記 | `applyMmcssPriority()` 末尾で直接 `SetThreadAffinityMask` している現状では、`applyCurrentThreadPolicy(AudioRealtime)` は二重適用となる。二重適用は無害（同一マスクの再設定は SetThreadAffinityMask に副作用なし）だが、保守性のためにコメントで責務を明記。将来 AudioThread の affinity 設定を一本化する場合のガイドにもなる。 | ✅ 将来のリファクタリングを考慮 |

#### (Y) v19 コードベース全スレッド棚卸し（v19 追加）

#### (Z) v20 コード改善 — lowestBit・while条件・リスク表統一・cores.size()基準・noexceptコメント（v20 追加）

| # | 検証項目 | 結論 |
|---|---|---|
| Z1 | `lowestBit()` → `std::countr_zero` の妥当性 | C++20 `<bit>` の `std::countr_zero` は符号なし整数の最下位ビットからの連続ゼロビット数を返す。`DWORD_PTR` は64ビット符号なし整数のため `countr_zero` の引数として完全に適合。mask==0 の戻り値は 64 以上（実装依存）。while ループより可読性が高く、MSVC/GCC/Clang すべてで BSF/TZCNT 命令に最適化される。`<bit>` は既に `DspNumericPolicy.h` で使用済み（新規依存追加なし）。 | ✅ 改善
| Z2 | `while(offset+sizeof(...)<=bufLen)` の再導入 | v17 から v18 で `while(offset<bufLen)` ＋ 内部チェックに変更したが、`info->Size` を読む前に最低限のヘッダサイズが存在することを保証するには `sizeof(...)` 条件の方が安全。内部の `Size==0` ガードと `offset+Size>bufLen` チェックは維持。3段階の防御（sizeof条件 → Size==0 → Size+bound check）で堅牢性最大化。 | ✅ 改善
| Z3 | リスク表フォールバック記述の統一 | 従来「GetActiveProcessorCount フォールバック」と記載していたが、本文は v15 で全ゼロマスクに変更済み。リスク表のみ未更新だったため「全ゼロマスク（アフィニティ無効）」に修正。`GetActiveProcessorCount` は診断ログ用としてのみ使用。 | ✅ 統一
| Z4 | `cores.size()` 基準への変更 | `physicalCoreCount` と `cores.size()` は原則一致するが、`computeSymmetricMasks` は cores の実データを基準に計算すべき。`physicalCoreCount` はログ表示用・整合性確認用として維持。計算に `cores.size()` を使うことで将来のメンバ追加時の不一致を予防。 | ✅ 堅牢性向上
| Z5 | `noexcept` コメント改善 | 従来の「bad_alloc→terminate」は実装依存の表現。`noexcept` 指定の設計意図は「起動時初期化でありメモリ確保失敗は回復不能」の一点。例外の種類に言及せず、コメントを純粋な設計意図の説明に変更。 | ✅ 可読性向上

v19 ではコードベース全体のスレッド生成箇所を grep/rg (WSL) + AiDex MCP で fully inventory し、計画のアフィニティ管理網羅性を確認した。

| # | 検証項目 | 結論 |
|---|---|---|
| Y1 | 全スレッドカテゴリの網羅性 | WorkerThread / RebuildThread / LoaderThread / ProgressiveUpgradeThread / DeferredFreeThread / NoiseShaper Learner (x2) / Audio callback / Message+UI = **11種**。全スレッドのアフィニティ適用箇所を確認。 | ✅ 完全網羅 |
| Y2 | JUCE ThreadPool (IR preview) | `g_irPreviewThreadPool(1)` (`ConvolverControlPanel.cpp:15`) — IR ファイル解析の短命ジョブ。アフィニティ未設定だが、非RT・軽量・ユーザ操作時のみのため不要。`ThreadAffinityManager.h` を include しておらず、適用コスト>利益。 | ✅ 対象外で妥当 |
| Y3 | JUCE ThreadPool (save) | `g_saveThreadPool(1)` (`NoiseShaperLearner.cpp:21`) — 学習済み状態のファイル保存。非RT・低頻度・短命。同上。 | ✅ 対象外で妥当 |
| Y4 | `LoaderThread` / `ProgressiveUpgradeThread` 基底クラス | 両者とも `juce::Thread` (JUCEスレッドラッパー) を継承。内部で `std::thread` を使用。`applyCurrentThreadPolicy(HeavyBackground)` を自身の `run()` で呼ぶため、初期化後に起動されれば正しく適用される。AudioEngine::initialize() 完後のユーザ操作でのみ起動。 | ✅ 安全 |
| Y5 | `DeferredFreeThread` lazy 作成タイミング | `ConvolverProcessor.Lifecycle.cpp:373` で初回 `prepareToPlay()` 時に `aligned_make_unique<DeferredFreeThread>(rcuSwapper, affinityMgr)` として作成。この時点で `affinityManager.initialize()` 完了済みのため `initialized_==true` を観測可能。 | ✅ 安全 (v7 修正で間接的カバー) |
| Y6 | `convo::numeric_policy::ThreadRole::AudioRealtime` との命名衝突 | `DspNumericPolicy.h` の `ThreadRole::AudioRealtime` はランタイムスレッド検出用（`ScopedThreadRole` でコンストラクタ/デストラクタにより AudioThread slot を acquire/release）。`ThreadType::AudioRealtime` は CPU アフィニティポリシー用。名前空間が異なり (`convo::numeric_policy` vs global enum)、列挙型も異なるため衝突なし。 | ✅ 安全確認 |
| Y7 | `GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)` MSDN 再確認 | MSDN: "If this parameter is ALL_PROCESSOR_GROUPS, the function returns the number of active processors in the system." — 本設計の診断ログ用に使用。0xFFFF=ALL_PROCESSOR_GROUPS。Win7+で利用可能。 | ✅ MSDN一致 |
| W5 | `detectCoreTopology() noexcept` と `bad_alloc` | `noexcept` 関数内で `std::vector` を使用しているが、`std::bad_alloc` はスタック巻き戻しを行わず `std::terminate()` を呼びプロセス終了。起動時に1度だけ呼ばれ、メモリ不足は回復不能のため許容。コメントで明示済み。 | ✅ 明示 |

### B.3 結論

本計画は MSDN 公式ドキュメント **11 件** + WSL **46 テスト** + pthread 実機 **9 テスト** + **v10 deep-dive 4 項目** + **v11 バグ監査 7 項目 (B1-B7)** + **v12 外部妥当性検証 (MSDN 9件 + 実コード 13 ファイル)** + **v14 deep-dive 4 項目 (Q-T)** + **v15 deep-dive 1 項目 (U)** + **v16 deep-dive 1 項目 (V)** + **v17 deep-dive 1 項目 (W)** + **v18 deep-dive 1 項目 (X)** + **v19 コードベース全スレッド棚卸し (Y)** + **v20 コード改善 5 項目 (Z)** による包括的突合検証を経て **実装可能** である。

**最終判定: 承認可 (A-)**, 2026-07-05。設計思想・API整合性・コード品質いずれも実装に進められる水準と評価する。唯一の追加推奨点 (`Mask!=0` 防御) は v21 で反映済み。

v12 外部妥当性検証の詳細報告書:
`doc/work64/work64-plan-validation-report.md`

v14 で確定した最終設計の要点:
- `SetThreadIdealProcessorEx` / `_BitScanForward64` / `applyAudioThreadPolicy()` は全て削除
- `detectCoreTopology()` の2回目 API 失敗時も `GetActiveProcessorCount` フォールバック
- 起動時に物理コア数・論理コア数・P/E有無・全マスク値を一括出力する診断ログ追加

v15 で確定した最終設計の要点:
- `CoreTopology::cores` を `PhysicalCoreInfo` 構造体の配列に変更 → ソート時の同期ずれを根絶
- API 失敗フォールバックを全ゼロマスク（アフィニティ無効）に変更 → 安全側に倒す
- `toHexString` のキャストを `static_cast<uint64_t>` に統一 → JUCE template で問題なし確認
- `AudioRealtime` ThreadType に用途注記を追加

v16 で確定した最終設計の要点:
- `ThreadAffinityManager::detectCoreTopology()` から `diagLog` 呼び出しを全削除 → 純粋ユーティリティ化
- ログは呼び出し側 `AudioEngine.Init.cpp` で `physicalCoreCount == 0` をチェックして出力
- JUCE `toHexString(uint64_t)` の互換性を JUCE 8.0.12 ソースコードで確認済み

v17 で確定した最終設計の要点:
- `detectCoreTopology()` のレコード走査に `info->Size==0` ガード＋`sizeof` 境界チェックを追加（無限ループ防止）
- `computeSymmetricMasks()` に `cores.size()!=physicalCoreCount` 整合性チェックを追加（out-of-range防止）
- `applyCurrentThreadPolicy()` の `SetThreadAffinityMask` 呼び出しを `prevMask` でラップ（診断統一）
- `detectCoreTopology() noexcept` に `bad_alloc→terminate` の注釈を明示

v18 で確定した最終設計の要点:
- `detectCoreTopology()` の while ループを `offset+Size<=bufLen` パターンに変更（Windows SDK 可変長レコード走査の標準パターン）
- `computeSymmetricMasks(const CoreTopology&)` に引数一本化（将来拡張に強いインターフェース）
- `AudioRealtime` ThreadType に二重適用の注記を追加（責務は applyMmcssPriority 側）

v19 で確定した最終設計の要点:
- コードベース全11スレッドカテゴリの棚卸し完了。全スレッドのアフィニティ網羅性確認済み
- JUCE ThreadPool 2件は非RT短命ジョブのためアフィニティ設定不要と確定
- `ThreadRole::AudioRealtime` vs `ThreadType::AudioRealtime` の名前衝突なし確認

v20 で確定した最終設計の要点:
- `lowestBit()` を `std::countr_zero` (C++20) に変更 → 可読性・最適化向上
- `while(offset<bufLen)` に `sizeof(...)` 境界保証を追加 → 破損データ対策
- リスク表のフォールバック記述を「全ゼロマスク」に統一
- `computeSymmetricMasks()` の計算基準を `cores.size()` に変更 → 将来拡張堅牢
- `noexcept` コメントを改善（実装依存を避け、設計意図のみ記載）

v9 で追加された項目:
- `applyMmcssPriority()` 始点行番号を L215 → L214 に訂正（`grep` 実ソースコード確認）
- MSDN 追加調査 3 件 (`GetProcessAffinityMask`, `SetProcessAffinityMask`, `SetThreadIdealProcessor`) = 全影響なし確認済み

**v7 で発見された既存バグ**:
- `AudioEngine.Init.cpp` で `initWorkerThread()` (L82) が `affinityManager.initialize(masks)` (L97) より前に呼ばれている。WorkerThread::run() の `applyCurrentThreadPolicy(Worker)` が `initialized_==false` で short-circuit し、WorkerThread の affinity が**永久に未適用**の状態。
- これを本計画 v7 で修正: L82 の `initWorkerThread();` 呼び出しを `affinityManager.initialize(masks);` の**直後** (L99 の直後) へ移動。

v6 で精緻化された項目:
- `static_cast<int>(audioMask)` truncation への注釈追記 (§5.4)
- §5.3 の delete 範囲を `L87-L99 inclusive` に精緻化
- §B.1 #9 の `_WIN32_WINNT` 設定を JUCE の強制設定として明記 (`JUCE/modules/juce_core/native/juce_BasicNativeHeaders.h:103-104`)

v12 で修正された項目:
- `juce_BasicNativeHeaders.h` のファイルパスを `JUCE/modules/juce_core/native/` に訂正 (旧 `system/` パスから完全パスに統一)
- §B.3 に外部妥当性検証報告書 (`doc/work64/work64-plan-validation-report.md`) への参照を追加

実装にあたり以下の点を遵守すること:

1. **`GetLogicalProcessorInformationEx` を使用すること**（非Ex版は不可）
2. **`#include <vector>` を `ThreadAffinityManager.h` に追加すること**
3. **`#include <algorithm>` を `ThreadAffinityManager.h` に追加すること**（`std::sort` のため）
4. **`PROCESSOR_RELATIONSHIP.GroupCount == 1` を前提に `GroupMask[0].Mask` のみ採取すること** (MSDN: "always 1")
5. **`SetThreadAffinityMask` の prevMask==0 エラーハンドリングを診断ログに追加すること** (§5.1 参照) — mask + GetLastError 出力
6. **プロセッサグループが複数にまたがる環境（>= 65論理プロセッサ、マルチソケットNUMA）は対象外**と明記
7. **CMake は `/W4` だが `/WX` なし**（CMake L757-767 確認済み）。`AudioRealtime` ケース追加で `C4062` 警告は解消方向
8. **32ビットビルド (WOW64) は対象外**。ConvoPeq は x64 ビルド限定（v5 追加明記）
9. **`topo.cores.size()` の括弧を忘れないこと** (v5 確認、`cores` に名称変更)
10. **JUCE の `JUCE/modules/juce_core/native/juce_BasicNativeHeaders.h:103-104` で `_WIN32_WINNT = 0x0A00` が常に設定される** (v6 確認)
11. **§5.3 の削除範囲は L87-L99 inclusive (`{` から `}` までブロック全体)** (v6 精緻化)
12. **§5.4 の `static_cast<int>(audioMask)` は 64bit マスク truncation だが対象 HW では安全** (v6 精緻化)
13. **★ v7 重大: `initWorkerThread()` を `affinityManager.initialize(masks)` の直後に移動する** (順序入替必須)
14. **★ v8 確認: 順序入替後は pthread シミュレーションで全 9 テスト PASS** (S8-1〜S8-9)
15. **v8: AudioRealtime 分岐の `return;` により `applyCurrentThreadPolicy` の noexcept 契約が保持される** (S8-9)
16. **v8: Release/Acquire メモリ順序により `initialized_=true` 後に `masks_` 書き込みが可視** (S8-7)
17. **v8: 複数デバイスの prepareToPlay() サイクルで正しく affinity と MMCSS が再適用される** (S8-6)
18. **v9: 行番号最終一致確認 — `applyMmcssPriority()` 始点は L214 (`void AudioEngine::applyMmcssPriority()`)、末尾 L272** (v9 微細訂正)
19. **v9: `SetProcessAffinityMask` / `GetProcessAffinityMask` / `SetThreadIdealProcessor` は本計画 scope 外 — 追加リスクなし** (§B.2(J))
20. **★ v10: `SetThreadAffinityMask` は `CPU Set` assignment より dominant (MSDN: "restrictive affinity mask is respected above any conflicting CPU Set assignment")** (§B.2(K))
21. **★ v10: プロセス全体 EcoQoS 無効化 (MainApplication.cpp:79-89) は affinity 動作と無関係に High QoS を選択、相互作用なし** (§B.2(K))
22. **v10: Audio Thread は Windows Audio Engine 管理 → `PROC_THREAD_ATTRIBUTE_GROUP_AFFINITY` 不要、post-hoc `SetThreadAffinityMask(GetCurrentThread())` で対応、§5.4 設計妥当** (§B.2(K))
23. **v10: `DeferredFreeThread` は ConvolverProcessor.Lifecycle.cpp:373 で lazy 作成され、AudioEngine::initialize() 完了後のため v7 修正で自然に covers** (§B.2(K))
24. **★ v11: B1-B7 全 7 項目で新規バグなし** (§B.2(L))。mmcssApplied_ reset/CAS の HB 順序、LearnerEval は audioRealtime から独立、子プロセス API 不使用、nullptr チェック既備、diagLog 順序妥当、全確認済み。実装は §10 Step 1-8 に従う。
25. **★ v14: `SetThreadIdealProcessorEx` / `_BitScanForward64` / `<intrin.h>` / `PROCESSOR_NUMBER` は不使用。audioMask は1物理コアのみのため IdealProcessor の効果が無い。**
26. **★ v14: `SetThreadAffinityMask` 失敗時のエラーログに mask 値 + GetLastError を含めること**
27. **★ v14: `detectCoreTopology()` の2回目 API 失敗時も `GetActiveProcessorCount` フォールバックすること**（空の topo を返さない）\n→ **v15 で変更: フォールバックは全ゼロマスク（アフィニティ無効）を返す。SMT topology 検出不能時にマスクを仮定すると不適切なコア固定になるリスクがあるため。**
28. **★ v15: `CoreTopology::cores[]` に `PhysicalCoreInfo` 構造体を使用すること** — `mask` と `efficiencyClass` を同一オブジェクト内で保持し、別vectorの同期ずれを根絶
29. **★ v15: `cores[]` を mask の最下位ビット（論理CPU番号）順にソートすること** — `std::sort` + ビット位置比較ラムダ
30. **★ v14: P/E heterogeneous は「affinity を設定しない」という意図的なポリシー選択であることをコメントに明記する**
31. **★ v14: §5.3 の `AudioEngine.Init.cpp` 初期化ブロック末尾に起動時診断ログを追加すること** — 物理コア数・論理コア数・P/E有無・全マスク値を一括出力
32. **v15: `#include <algorithm>` を `ThreadAffinityManager.h` に追加すること**（`std::sort` 使用のため）
33. **v14: `applyAudioThreadPolicy()` 専用メソッドは不要 — `ThreadType::AudioRealtime` + `applyCurrentThreadPolicy()` で十分**
34. **★ v16: `ThreadAffinityManager::detectCoreTopology()` 内で `diagLog` を呼ばないこと** — ログは呼び出し側 (`AudioEngine.Init.cpp`) で `topo.physicalCoreCount == 0` をチェックして出力
35. **★ v16: `AudioEngine.Init.cpp` の初期化ブロックに `physicalCoreCount == 0` の分岐を追加すること** — `noAffinity{}` で `initialize()` を呼び、全スレッドが安全に short-circuit できる状態を保証
36. **v16: `toHexString(static_cast<uint64_t>(...))` は JUCE 8.0.12 で有効** — template のため任意の整数型を直接受け付ける (`juce_String.h:1124` 確認済み)
37. **★ v17: `detectCoreTopology()` の `while(offset<bufLen)` を `while(offset+sizeof(...)<=bufLen)` に変更すること** — 最低1レコード分の領域があることを保証
38. **★ v17: `info->Size==0` のガードチェックを追加すること** — API 破損やメモリ破壊による無限ループ防止
39. **★ v17: `computeSymmetricMasks()` に `cores.size()!=physicalCoreCount` の整合性チェックを先頭に追加すること** — 不一致時は `return ThreadAffinityMasks{}`（全ゼロ）
40. **★ v17: `applyCurrentThreadPolicy()` の `SetThreadAffinityMask` を `const DWORD_PTR prevMask = ::SetThreadAffinityMask(...); juce::ignoreUnused(prevMask);` でラップすること** — 診断ログ有効時に各スレッドの適用結果を追跡可能に
41. **★ v17: `detectCoreTopology() noexcept` のコメントで `bad_alloc→terminate` を明示すること** — `std::vector` 使用時の `noexcept` 契約を文書化
42. **★ v18: `detectCoreTopology()` の while 条件を `offset<bufLen` に変更し、ループ内で `info->Size==0` と `offset+info->Size>bufLen` を個別チェックすること** — Windows SDK 可変長レコード走査の標準パターン
43. **★ v18: `computeSymmetricMasks()` の引数を `const CoreTopology& topo` に一本化すること** — 呼び出し側で `topo.cores` 等でアクセス
44. **v18: `AudioRealtime` ThreadType のコメントに二重適用の注記を追加すること** — 現状は `applyMmcssPriority()` 側で直接設定しているため
45. **★ v19: `convo::numeric_policy::ThreadRole::AudioRealtime` と `ThreadType::AudioRealtime` は別概念であることをコメントに明記すること** — 名前空間が異なり衝突しないが、コードレビュー時の混乱を防止
46. **v19: §8 のスレッド棚卸し表を参照し、JUCE ThreadPool 2件がアフィニティ管理範囲外であることを確認すること** — 変更不要
47. **★ v20: `lowestBit()` に `std::countr_zero` (C++20, `<bit>`) を使用すること** — while ループより可読性が高く、コンパイラにより1命令に最適化
48. **★ v20: `detectCoreTopology()` の while 条件を `offset+sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)<=bufLen` にすること** — 最低限のヘッダサイズが存在することを保証
49. **★ v20: `computeSymmetricMasks()` では `topo.physicalCoreCount` ではなく `topo.cores.size()` を計算基準にすること** — 将来のメンバ追加時の不一致を防止
50. **v20: `noexcept` のコメントは「起動時初期化、回復不能」のみに留め、`bad_alloc` 実装依存の記述は避けること**
51. **★ v21: `detectCoreTopology()` の `if (proc.GroupCount == 1)` に `&& proc.GroupMask[0].Mask != 0` を追加すること** — 破損データ対策。正常系では発生しないが、防御コードとして有効
