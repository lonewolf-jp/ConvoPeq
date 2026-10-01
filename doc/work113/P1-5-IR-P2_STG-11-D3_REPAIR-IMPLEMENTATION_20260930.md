# STG-11-D3 Repair Implementation Record

- Document: `doc/work113/P1-5-IR-P2_STG-11-D3_REPAIR-IMPLEMENTATION_20260930.md`
- Work item: **STG-11-D3 / Candidate A** — MMCSS / CPU-affinity failure path の RT-unsafe logging 除去
- Predecessor: `P1-5-IR-P2_STG-11-D3_REPAIR-CONTRACT-AUDIT_20260929.md`（Candidate A 確定）
- Date: 2026-09-29 → 2026-09-30
- Commit / push: **未実施**（Owner の個別 GO 待ち。作業ツリーに保持）

---

## 1. Authority（実ファイルから再取得・転記禁止）

implementation 開始時（2026-09-29 作業開始時点）に実ファイルから取得:

| 項目 | 値 |
| --- | --- |
| authority file | `ConvoPeq(20260929-124301).md` == `ConvoPeq.md` |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `9565A3D546BF256F75C1A2B97FB1E9B99CFF8D6DED11A3673642EADFF4F8F5BE` |
| size | 5,710,250 B |
| Generated | `2026-09-29 21:16:18` |
| NEWER_SRC_COUNT | 0 |
| `--check` | `STATUS : FRESH — snapshot は現行ソースを反映しています` |
| 不一致 | なし → implementation 停止条件に該当せず proceed |

実装完了後に再生成した値は Post-Implementation Gate 记录的参照（§Gate）。

---

## 2. 対象 defect

`AudioEngine::applyMmcssPriority()`（`src/audioengine/AudioEngine.Timer.cpp`）の 3 つの failure 分岐が
logging backend を RT（audio thread）上から直接呼んでいた。

| 分岐 | OS 呼び出し | 修正前の failure 処理 | 修正前 guard |
| --- | --- | --- | --- |
| F1 | `SetPriorityClass(REALTIME_PRIORITY_CLASS)` | `diagLog(...)` | `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` |
| F2 | `SetThreadPriority(THREAD_PRIORITY_TIME_CRITICAL)` | `diagLog(...)` | 同上 |
| **F4** | `SetThreadAffinityMask(GetCurrentThread(), audioMask)` | `diagLog(...)` | **guard 無し（無条件）** |

`diagLog` は `AudioEngine.Timer.cpp:136` の非ガード free function で、

```
diagLog -> DBG(message) + asyncSink(message)
asyncSink -> std::lock_guard<std::mutex> (s_logMutex) -> LockFreeRingBuffer::pushWithWriter -> juce::String
```

のとおり、**mutex 取得 + ヒープ確保**を伴う。F4 は guard が無いため
**Diagnostics OFF でも RT → mutex → heap に到達していた**（= INV-D3-3 違反の核心）。

F1/F2 は Diagnostics ON のときだけ同経路に到達していたため、D3 の修復範囲に含めた
（同一 RT 到達欠陥・同一輸送机构。Owner 指示「Diagnostics ON の failure path でも
RT から mutex/heap/logging backend に到達しないことを structural に保証」に該当）。

---

## 3. 実装（Candidate A）

### 3.1 RT 側: lock-free atomic record のみ

`src/audioengine/AudioEngine.h` に private atomic member **6 件**を追加（`mmcssShutdownRequested` 直後）:

```cpp
// STG-11-D3: affinity failure observation (RT to NonRT diagnostic transport).
//   RT-safe: lock-free atomics only. lossy-coalescing (last-wins + monotonic count).
//   No mutex / allocation / logging backend on RT. No new authority.
//   Readout and diagnosis by NonRT reportAffinityFailureIfRecorded().
std::atomic<bool>            affinityFailureObserved_{false};   // (3) monotonic flag, last store
std::atomic<std::uint64_t>   affinityFailureCount_{0};          // (2) monotonic + publication marker
std::atomic<std::uint64_t>   affinityFailureLastMask_{0};       // (1) payload, last-wins
std::atomic<std::uint32_t>   affinityFailureLastError_{0};      // (1) payload, last-wins
std::atomic<std::uint32_t>   affinityFailureLastKind_{0};   // 0=affinity, 1=SetPriorityClass, 2=SetThreadPriority
std::atomic<std::uint64_t>   affinityFailureReportedCount_{0};  // NonRT-only once-only bookkeeping
```

(1)(2)(3) は §3.1.1 の publication 順序に対応。

private 宣言 2 件:

```cpp
void recordAffinityFailure(std::uint32_t kind, DWORD_PTR audioMask, DWORD error) noexcept;
void reportAffinityFailureIfRecorded() noexcept;
```

`AudioEngine.Timer.cpp` の recorder（**RT 到達**）:

```cpp
void AudioEngine::recordAffinityFailure(
    std::uint32_t kind, DWORD_PTR audioMask, DWORD error) noexcept // NOLINT(bugprone-easily-swappable-parameters)
{
    // (1) payload: publish the snapshot BEFORE anything marks the record.
    convo::publishAtomic(affinityFailureLastKind_, kind, std::memory_order_release);
    convo::publishAtomic(affinityFailureLastMask_, static_cast<std::uint64_t>(audioMask),
                         std::memory_order_release);
    convo::publishAtomic(affinityFailureLastError_, static_cast<std::uint32_t>(error),
                         std::memory_order_release);
    // (2) count: monotonic AND the per-record publication marker.
    convo::fetchAddAtomic(affinityFailureCount_, std::uint64_t{1}, std::memory_order_acq_rel);
    // (3) observed: monotonic flag, always the last store.
    convo::publishAtomic(affinityFailureObserved_, true, std::memory_order_release);
}
```

### 3.1.1 publication 順序（2026-09-30 ordering correction）

**旧順序が誤っていた理由.** release store が publish するのは「その store より
sequenced-before された書き込み」だけである。旧実装は

```
RT:     observed(release) → count(fa, acq_rel) → payload(release)
NonRT:  observed(acquire) → count(acquire) → payload(acquire)
```

の順で、`observed` の release が payload の store より**前**にあったため、
`observed` の acquire には payload の happens-before edge が含まれず、
INV-D3-5 の「payload 可視性が保証される」が成立していなかった。

**修正後の順序と各位置の役割.**

| 位置 | 対象 | wrapper / order | 役割 |
| --- | --- | --- | --- |
| (1) | `LastKind_` / `LastMask_` / `LastError_` | `convo::publishAtomic` / `release` | payload。**先に publish** する |
| (2) | `Count_` | `convo::fetchAddAtomic` / `acq_rel` | 単調増加。かつ **per-record の publication marker** |
| (3) | `Observed_` | `convo::publishAtomic` / `release` | 単調 bool flag（fast-out 専用）。payload を運ばない。**常に最後の store** |

(2) が per-record の可視性保証を担う理由: `fetchAddAtomic` は既定 `acq_rel`
（release operation）であり、reader の `count` acquire が値 `N` を観測すると、
`N` を生じさせた加算と synchronizes-with し、したがって **record #N の payload が可視**になる。
(3) の `observed` は「一度でも failure があったか」の単調 flag であり、fast-out にのみ使う。
`observed` の acquire から payload の可視性を期待する設計は行っていない。

**reader 側の必須順序**（snapshot の read は count の acquire より**後**）:

```
observed(acquire, fast-out) → count(acquire) → reported(acquire) → mask/error/kind(acquire)
```

これにより:

- `count == N`  ⇒  record #N の payload は可視（**古い snapshot を新しい count として確定しない**）
- snapshot が `count` より**新しい**場合は、RT が 2 つの load の間に再度記録したケース。
  これが lossy-coalescing として許容、ownership を運ばないため silent loss は起きない
- `observed` 単独では payload の可視性を保証しないため、**fast-out としてのみ**使用

failure 分岐の置換（3 箇所）:

- `if (pcResult == 0) { recordAffinityFailure(1, 0, ::GetLastError()); }`
- `if (tpResult  == 0) { recordAffinityFailure(2, 0, ::GetLastError()); }`
- `if (prevMask  == 0) { recordAffinityFailure(0, audioMask, ::GetLastError()); }`

**すべて `convo::` wrapper のみ使用**（raw `std::atomic` API 追加 0）。

### 3.2 NonRT 側: 既存 NonRT execution point から診断

`reportAffinityFailureIfRecorded()` は `convo::` wrapper で snapshot を読み、**既存** `diagLog` backend に出力する。
新規 timer / thread / worker / authority は**作っていない**。呼び出し点は既存の `timerCallback()`
（`processDeferredReleases()` 直後、`m_coordinator.tryCompleteFade()` の直前）1 箇所のみ。

```cpp
void AudioEngine::reportAffinityFailureIfRecorded() noexcept
{
    // fast-out only: monotonic "any failure ever" flag, not a payload carrier.
    if (!convo::consumeAtomic(affinityFailureObserved_, std::memory_order_acquire))
        return;
    // publication marker: acquire of count == N publishes record #N's payload.
    const std::uint64_t count = convo::consumeAtomic(affinityFailureCount_, std::memory_order_acquire);
    if (count == 0)
        return;
    const std::uint64_t reported =
        convo::consumeAtomic(affinityFailureReportedCount_, std::memory_order_acquire);
    if (reported >= count)
        return;  // already diagnosed (count is monotonic: nothing new to report)
    ... // last-wins snapshot を読む
    diagLog(juce::String(tag) + ": mask=0x" + ... " count=" + ...);
    convo::publishAtomic(affinityFailureReportedCount_, count, std::memory_order_release);
}
```

### 3.3 変更していない failure policy（§7 遵守）

以下は**すべて不変**（diff で該当行の挙動が変わっていないことを §Gate の diff で確認）:

- affinity failure = informational（`priorityApplied` に影響しない）
- `priorityApplied` の意味
- `applyMmcssPriority()` の failure policy（`if (!nativeRtOk) priorityApplied = false;` を保持）
- caller の return value handling
- `t_mmcssTried` による初回実行 semantics
- fallback policy
- MMCSS policy 自体（再設計していない）

### 3.4 変更範囲（`git diff --numstat`）

| ファイル | + | − | 帰属 |
| --- | --- | --- | --- |
| `src/audioengine/AudioEngine.Timer.cpp` | 99 | 11 | **D3**（ordering correction 込み） |
| `src/audioengine/AudioEngine.h` | 15 | 0 | **D3** |
| `src/tests/AudioEngineHarness/STG11D3AffinityFailureTests.cpp` | 新規 | — | **D3**（untracked） |
| `src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h` | 50 | 0 | D1 + **D3** |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | 16 | 0 | D1 + D2 + **D3** |
| `CMakeLists.txt` | 133 | 0 | D1 + D2 + **D3**（1 行） |
| `src/core/SnapshotCoordinator.cpp` | 21 | 3 | D2（既存） |
| `src/eqprocessor/EQProcessor.Core.cpp` | 37 | 10 | D1（既存） |
| `src/eqprocessor/EQProcessor.h` | 39 | 0 | D1（既存） |
| `ConvoPeq.md` | 2299 | 25 | 再生成 |

production 変更は **2 ファイルのみ**（`AudioEngine.Timer.cpp` / `AudioEngine.h`）。

### 3.5 非変更の確認（§2 禁止領域）

以下は `git status --porcelain` で**完全に未変更**（空）であることを確認:

`ISRRetireRouter.h/.cpp`, `EpochDomain.h/.cpp`, `ISRRuntimePublicationCoordinator.h/.cpp`,
`ISRCoordinatorLoop.cpp`, `PublicationAdmission.h`, `RuntimeStore.h`

さらに `Crossfade` / `Publish` / `Retire` / `Recovery` の各ファイルにも差分なし。

---

## 4. 不変条件（implementation contract）

| ID | 内容 | 担保手段 |
| --- | --- | --- |
| INV-D3-1 | RT-reachable failure logging shall not acquire a mutex. | F1/F2/F4 の brace 抽出で `std::mutex` / `lock_guard` / `asyncSink` / `s_logMutex` 不在を構造的に検証。recorder 本体も同様 |
| INV-D3-2 | RT-reachable failure logging shall not allocate or free memory. | 同 brace 抽出で `juce::String` / `new ` / `make_unique` / `make_shared` / `malloc` / `calloc` / `realloc` / `free(` 不在 |
| INV-D3-3 | Diagnostics OFF shall not be required as a precondition for RT safety. | (a) failure 分岐の禁止 token を **guard に関係なく**検査、(b) `recordAffinityFailure` 呼び出しが diagnostics guard 内に無いことを検査。success 側は全 `diagLog`/`juce::String` が guard 内であることを検査 |
| INV-D3-4 | Diagnostic observation shall not create a new publication, retire, or recovery authority. | recorder body に `std::thread` / `std::jthread` / `juce::Timer` / `LockFreeRingBuffer` / `enqueueDeferredDelete` / `enqueueRetire` / `DeferredDeletionQueue` / `RuntimePublication` / `Recovery` が無いことを検査 |
| INV-D3-5 | RT diagnostic transport shall use explicitly defined lossy-coalescing semantics; silent ownership loss is prohibited. | 輸送するのは `count`（単調増加）+ last-wins snapshot（mask / error / kind）のみ。`RuntimeWorld` / `Snapshot` / `Retire object` / `Publication object` / `Recovery obligation` は**輸送しない**ため coalescing しても ownership を失わない。`reportedCount` による once-only 診断で coalescing が可視化される。**publication 順序**（payload → count → observed）と **reader 順序**（count acquire → snapshot acquire）を D3-T5 が構造的に検証する |

---

## 5. テスト

`src/tests/AudioEngineHarness/STG11D3AffinityFailureTests.cpp`（新規）。
新規 CTest target は**作らない**（`AudioEngineHarness.exe` のサブテスト。STG-8 / STG-9 と同じ形）。
harness main から `runSTG11D3AffinityFailureTests()` として呼び出す。

OS failure injection 用の新しい production mechanism は作っていない（Owner 指定）。
production と同じ private `recordAffinityFailure()` を合成値（mask / error / kind）で駆動し、
輸送の両端（記録・診断）を独立に検証する。fixture は harness が所有し TU 側 stack 確保を小さくしている。

| Test | 象限 | 方式 | 検証内容 |
| --- | --- | --- | --- |
| **D3-T1** | OFF + success / **T4** ON + failure | structural | F1/F2/F4 の failure 分岐を brace-match で厳密抽出し、backend / lock / allocation token の不在。guard の有無に依存しない |
| **D3-T1b** | OFF + success | structural | `applyMmcssPriority` 内の全 `diagLog(` / `juce::String` が diagnostics guard 内。関数全体が mutex / SPSC sink を直接触らない |
| **D3-T2** | OFF + failure / **T4** ON + failure | functional | 記録 → state 検証（observed / count / kind / mask / error）→ NonRT 診断で `reportedCount` 進展 → 再診断で**非重複** → 2 回目で last-wins + count 増加 → 3 回目で kind 伝播 |
| **D3-T3** | ON + success | structural | 既存 guarded 成功ログ 3 種（`[AFFINITY] AudioThread pinned` / `[AFFINITY] P/E cores` / `[NATIVE_RT] applied`）の保持と guard 維持 |
| **D3-T5a** | ordering | structural | recorder の atomic access 順序。コメント除去後に `convo::` wrapper 呼び出しを source 順に列挙し、**payload 3 種すべてが count より前**、**count が observed より前**、**observed が最後の access** であることを検証。`count` は `convo::fetchAddAtomic`（単調 + release）、`observed` は `convo::publishAtomic` |
| **D3-T5b** | ordering | structural | report の read 順序。`count` の read が snapshot 3 種の read より**前**であることを検証。`count` / snapshot はすべて `convo::consumeAtomic`、`memory_order_acquire` が明記されていること、raw atomic API がないこと |
| **D3-T5c** | ordering | deterministic | 5 回の連続 record。毎回 `observed == true`、`count == k`（単調・欠落/幽灵 increment なし）、snapshot が **record #k 自身の値**（last-wins、古い snapshot を新しい count として確定しない）、`reportedCount` が **ちょうど +1** 進展すること。record が増えない情形で report が進展しないこと |

補助検査（同じ TU 内）:

- recorder body に raw `std::atomic` API（`.load/.store/.fetch_add/.fetch_sub/.fetch_or/.fetch_xor/.exchange/.compare_exchange`）が無いこと
- recorder が `convo::publishAtomic` / `convo::fetchAddAtomic` を使っていること
- NonRT report が `diagLog` backend を保持していること
- `reportAffinityFailureIfRecorded()` が `timerCallback()` から呼ばれること
- coalescing 状態 4 フィールドと `reportedCount` の存在

structural helper として `stripComments()`（quote-aware、`//` / `/* */` を除去）を
`atomicCallOrder()` の前段に置いた。これにより順序 oracle は**コメント文字列に
依存せず実 code のみ**を見る。 Owner 指示の「古い snapshot を新しい count の
snapshot として誤って確定しない」を Comments 経由で誤判定しない。

preprocessor 判定は単純な深さカウンタではなく **条件付きスタック**で行う
（`#else` は diagnostics guard を反転、`#elif` は保守的に無効化）。
別 `#if` / `#endif` ペアや body-relative offset で誤判定しないことを実装時に実測修正した。

---

## 6. 実施した検証

| 検証 | 結果 |
| --- | --- |
| Debug build | BUILD_EXIT=0 |
| Release build | BUILD_EXIT=0 |
| D3 sub-test（Debug, harness 直接実行） | rc=0 / D3-T1, T1b, T2, T3, T5a, T5b, T5c 全 PASS |
| D3 sub-test（Release, harness 直接実行） | rc=0 / 同上 |
| **Negative control #1**（F4 を原始欠陥へ一時復元） | D3-T1 `F4 ... reaches RT-forbidden backend 'diagLog('` → **FAIL**、D3-T1b `'diagLog(' reachable with Diagnostics OFF` → **FAIL**、harness rc=**1**。⇒ oracle は実際に欠陥を捕捉する（test-first 成立） |
| **Negative control #2**（recorder を旧順序 `observed` 先頭へ一時復元） | D3-T5 `payload affinityFailureLastKind_ is published AFTER the count increment (index 2 > 1): the release/acquire pair would not cover it` → **FAIL**、harness rc=**1**。⇒ ordering oracle は実際に publication 逆順を捕捉する |
| Negative control 後の復元 | 両 negative control とも復元 → 再 build → 全 PASS。recorder の順序が `LastKind_ → LastMask_ → LastError_ → Count_(fetchAdd) → Observed_` であることを実ファイルで確認 |
| D1 回帰（`STG11EQRetireTests` Debug） | rc=0 / TD1-1 Q=304、TD1-2 D/Q/E/T = 4096/512/512/280 total 5400、TD1-3 / TD1-4a / TD1-4b PASS（不変） |
| D2 回帰（`STG11D2SnapshotRetireTests` Debug） | rc=0 / TD2-1 E=3、TD2-2 T=3、TD2-3 全 0（不変） |
| full Debug CTest | **45/45 PASS**（最終 run、他に 5/5 連続 PASS） |
| full Release CTest | **45/45 PASS**（最終 run） |
| raw `std::atomic` audit | D3 追加行に raw API **0**。atomic 操作 12 箇所すべて `convo::` wrapper |
| RT allocation / lock audit | D3 追加行の RT 到達部（recorder）に `diagLog` / `juce::String` / mutex / heap **0**。該当トークンは NonRT report 関数とコメントのみ |
| authority audit | 新規 publication / retire / recovery / thread / timer / queue authority **0** |
| clang-tidy（`AudioEngine.Timer.cpp`, JSON-array DB） | D3 追加行の finding **0**。残存 `bugprone-branch-clone` ×2（1260, 1291）と `bugprone-unchecked-optional-access` ×6（2046-2073）はいずれも **D3 の diff hunk 外**（D3 は 252-394 と 1011-1015）の既存事象 |
| cppcheck（project DB, `-i` で D3 TU 限定） | project 全体 135 件すべて既存ノイズ、**D3 production 2 ファイル 0 件** |
| ASAN | 環境 block（`0xC0000139`、pre-main の `ntdll!LdrGetProcedureAddressForCaller` forwarder 解決失敗）。ASAN failure と implementation failure を分離し、構造的 oracle + negative control + Debug/Release 双構成 45/45 で代替 evidence を取得 |

---

## 7. D3 scope 外で発見した事項（**修正せず記録のみ**）

Owner 指示「実装途中で D3 scope 外の問題を発見しても、勝手に修正範囲を拡張せず、問題を記録して停止」に従い、以下は**未修正**で記録する。

### OBS-1: Diagnostics ON 時の success path は依然として RT から logging backend を呼ぶ

`applyMmcssPriority()` の以下の 3 経路は `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内で `diagLog` を呼ぶ:

- `[NATIVE_RT] applied: win32Prio=...`（NativeRT 成功時）
- `[AFFINITY] AudioThread pinned mask=0x...`（affinity 成功時）
- `[AFFINITY] P/E cores: AudioThread affinity skipped`（異質コア時）

これらは RT → `diagLog` → `asyncSink` → `std::mutex` → heap に到達する。
ただし Owner §8 は「Diagnostics ON / success → **existing behavior**」と規定しており、
§7 は failure policy のみ不変を要求し success logging は変更対象外。
D3 の failure 修復とは**別 defect** であり、同一の「RT-safe diagnostic transport」構造で
修可能（成功ATCCを record して NonRT 診断する）。**次回の Discovery 投入候補**とする。
本実装では guard 構造（`#if` / `#endif` の位置）を一切動かしていない。

### OBS-3: `recordAffinityFailure` の 3 引数が交換可能（clang-tidy が指摘）

clang-tidy が `bugprone-easily-swappable-parameters` で
`recordAffinityFailure(kind, audioMask, error)` を指摘した。3 引数すべて
`std::uint32_t` / `DWORD_PTR` / `DWORD` で互いに暗黙変換可能であり、
入れ違えてもコンパイルが通る。

**対応**: 根拠コメント付き `// NOLINT(bugprone-easily-swappable-parameters)` を付与。
**引数 変更していない**（Owner の publication 順序指示と独立した判断。
順序変更は diff を拡大するため行わない）。

安全性は呼び出し側で担保されている: production の呼び出しは failure 分岐 3 箇所のみで、
いずれも第 1 引数にリテラル（`0` / `1` / `2`）を渡すため、交換しても
`kind` 値が 0/1/2 以外になりえない。加えて D3-T5c は 5 ラウンドの各回で
`(kind, mask, error)` の対応を一意に検証している。

### OBS-4: Owner 提示の authority スナップショット名 `ConvoPeq(20260929-153558).md` は存在しない

Owner 指示 §1 の authority 名 `ConvoPeq(20260929-153558).md` は
リポジトリ内に存在しない。`doc/ConvoPeqMD/` の実在スナップショットは
`20260928104322` / `20260928143345` / `20260928155853` / `20260928161811` /
`20260928221921` / `20260929001943` / `20260929193552` / `20260929211618` /
`20260930001956` の 9 件である。

Owner は同時に「`ConvoPeq(20260929-153558).md` はプロジェクトルートの
`ConvoPeq.md` と統一ファイルである」と規定しているため、
**リポジトリルート `ConvoPeq.md` を唯一の source authority として作業した**。
その内容は Owner が §2 で描述した実装（`recordAffinityFailure` が
`observed` を先に publish する順序）と完全に一致していた。
事実の記録のみ。追加の采取了行っていない。

### OBS-2: Debug `AudioEngineHarness` の intermittent（STG-6 area の access violation）

Details は Post-Implementation Gate §6 参照。STG-6 / STG-8-D2b の既存 test 側で発生し、
**D3 sub-test は `main()` 上でこれ）より後に実行されるため D3 test コードでは説明できない**。
D3 除去ベースラインで 5/5 PASS、D3 構築で後続 5/5 PASS であったが、
初期窓（run 1〜5）で 3 件の失敗が観測された。**再現できず、証拠からは D3 への帰属は立証できない。**
R8（intermittent 3 件）の一部と同一の phenomenology である可能性がある。

---

## 8. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`
- staged = 0
- **commit / push は未実施**。Post-Implementation Gate は `READY FOR COMMIT`。
- D1 / D2 の未 commit 変更はそのまま保持している。
- pre-existing worktree 変更（`AGENTS.md` / `headroom-proxy-start.ps1` /
  `doc/work113/P1-5-IR-P2_STG-8-D1-D3_REPAIR-GATE_20260928.md`）には触れていない。
