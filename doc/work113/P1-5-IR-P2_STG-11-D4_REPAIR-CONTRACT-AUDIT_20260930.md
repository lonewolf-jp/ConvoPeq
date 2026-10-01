# STG-11-D4 Repair Contract Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D4_REPAIR-CONTRACT-AUDIT_20260930.md`
- Work item: **STG-11-D4 / OBS-1** — Diagnostics ON 時の `applyMmcssPriority()` success path の RT-unsafe logging 除去
- Predecessor: D3 Post-Implementation Gate §8 OBS-1（記録のみ、未修正）
- Date: 2026-09-30
- Authority: リポジトリルート `ConvoPeq.md`（Owner 指示の統一ファイル規定による。`ConvoPeq(20260930-111230).md` という名前のファイルは存在しない — §1 参照）
- Commit / push: **禁止**（作業ツリーに保持）

---

## 0. 判定

```
GO（Contract 成立。実装へ進む）
```

§6 の停止条件はいずれも発生しない（§7 参照）。

---

## 1. Authority（実ファイルから再取得・転記禁止）

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md` |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `F441875191E30FA864B06C0538AA4E5A2B775470E0FA49990A59F65E726CD4C0` |
| size | 5,761,584 B |
| Generated | `2026-09-30 01:42:22` |
| NEWER_SRC_COUNT | 0 |
| `--check` | `STATUS : FRESH — snapshot は現行ソースを反映しています`（exit 0） |

Owner 指示の authority 名 `ConvoPeq(20260930-111230).md` はリポジトリ内に存在しない。
`doc/ConvoPeqMD/` の実在スナップショットは `20260928104322` / `20260928143345` /
`20260928155853` / `20260928161811` / `20260928221921` / `20260929001943` /
`20260929193552` / `20260929211618` / `20260930001956` / `20260930014222` の 10 件であり、
`111230` はいずれにも該当しない。D3 第 2 版の OBS-4 と同一の phenomenology である。
Owner は同時に「プロジェクトルートの `ConvoPeq.md` と統一ファイルである」と規定しているため、
**リポジトリルート `ConvoPeq.md` を唯一の source authority とした**。
古い `doc/ConvoPeqMD/` snapshot は authority にしていない。

D1 + D2 + D3 の未 commit 変更は保持されている（`git status -sb` = `main...origin/main [ahead 9]`、
staged 0、tracked 変更 9 ファイル + untracked 新規 test 3 TU）。
pre-existing 変更（`AGENTS.md` / `headroom-proxy-start.ps1` / STG-8 gate 文書）には触れていない。

---

## 2. 対象 defect（最新 source から実確認）

### 2.1 RT 到達 chain（call chain まで確認）

```
Audio callback (RT)
  src/audioengine/AudioEngine.Processing.AudioBlock.cpp:62
  src/audioengine/AudioEngine.Processing.BlockDouble.cpp:66
        │  static_cast<void>(tryApplyMmcssForSelfManagedThread());   ← audio thread 上
        ▼
  src/audioengine/AudioEngine.Mmcss.cpp:72
  tryApplyMmcssForSelfManagedThread()
        │  if (t_mmcssTried) return ...;   ← thread_local, 初回のみ通過
        │  t_mmcssTried = true;
        │  applyMmcssPriority();   (:81, 無条件 — useMmcssPriority の判定より前)
        ▼
  src/audioengine/AudioEngine.Timer.cpp:233
  applyMmcssPriority()
        │  Diagnostics ON 時の success 分岐 3 箇所で diagLog() を直接呼ぶ
        ▼
  diagLog (Timer.cpp:196, 非ガード free function)
        │  DBG(message);
        │  asyncSink(message);
        ▼
  asyncSink (Timer.cpp:144)
        │  std::lock_guard<std::mutex> (s_logMutex)
        │  s_logBuffer.pushWithWriter(...) + juce::String
        ▼
  mutex 取得 + ヒープ確保（RT 上）
```

`applyMmcssPriority()` は audio thread 上で実行される（`noexcept` だが RT-safe ではない）。
`t_mmcssTried`（`Mmcss.cpp:29`、thread_local）により **audio thread ごとに初回 1 回だけ**
実行される。device reopen / driver-owned ASIO thread 生成のたびに再実行され得る
（`revertMmcssOnAudioThread` が `t_mmcssTried = false` に戻す。`Mmcss.cpp:205`）。

### 2.2 3 つの success site（最新 source の実位置）

| # | 位置 | 条件 | 現行 code（Diagnostics ON 時のみ compile） |
| --- | --- | --- | --- |
| S1 | Timer.cpp:264-272 | `!useMmcssPriority && nativeRtOk` | `diagLog("[NATIVE_RT] applied: win32Prio=" + juce::String(win32Priority) + " procClass=" + ... + " savedClass=" + ...)` |
| S2 | Timer.cpp:296-302 | `!hasHeterogeneousCores_ && audioMask != 0 && prevMask != 0` | `diagLog("[AFFINITY] AudioThread pinned mask=0x" + toHexString(audioMask) + " prev=0x" + toHexString(prevMask))` |
| S3 | Timer.cpp:305-309 | `hasHeterogeneousCores_`（`if/else` の else 側） | `diagLog("[AFFINITY] P/E cores: AudioThread affinity skipped (MMCSS Deadline QoS)")` |

いずれも `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` の内側にある。
CMake の既定は `option(... OFF)`（`CMakeLists.txt:129`）のため既定 build では
compile されないが、**Diagnostics ON の success path では RT が mutex / heap に到達する**。
これが OBS-1 の核心であり、D3 の F1/F2/F4 と同一の defect class である。

### 2.3 既存ログの消費者（推測でなく実査）

- S1/S2/S3 の文字列を program 上で parse / assert する consumer は存在しない。
  リポジトリ全体の出現は `AudioEngine.Timer.cpp`（production）、
  `STG11D3AffinityFailureTests.cpp`（D3-T3 の structural 存在＋guard 確認）、
  過去の `ConvoPeqMD` snapshot と work64 計画文書のみ。
- 出力先は `asyncSink` → `LockFreeRingBuffer` → `flushLogBuffer` → `juce::Logger::writeToLog`。
  すなわち人間向け診断ログであり、machine-readable な契約は存在しない。
- 既存ログは sequence 番号も thread ID も持たない。per-event の identity を
  下流が消費している証拠はない。

---

## 3. 修正方針（D3 Candidate A と同一思想の適用可否）

### 3.1 適用可否の判定: **適用できる**

```
RT（applyMmcssPriority の success 分岐）
  ↓  lock-free atomic observation（convo:: wrapper のみ）
既存 NonRT execution point（timerCallback — D3 の failure report と同一箇所）
  ↓  既存 diagLog backend（文言は §3.3 のとおり維持）
```

- RT 側の禁止物（mutex / allocation / free / `juce::String` / `diagLog` / `asyncSink` /
  logging backend / blocking / new authority）はすべて排除できる。
  記録するのは整数・bool の atomic のみであり、OS 呼び出し結果（`GetThreadPriority` 等の
  **戻り値**）を整数として写すだけである。`::GetThreadPriority` / `::GetPriorityClass`
  自体の呼び出しは RT 上に残るが、これは既存の OS query であり logging backend ではない
  （現行 code も同一呼び出しを行っている。変更しない）。
- D3 の failure transport とは**別の member / 関数群**を新設するため、D3 の boundary を
  変更しない（§5）。

### 3.2 Observation identity（§3.1 — 3 種類の区別）

| kind | 意味 | 値 |
| --- | --- | --- |
| 1 | NativeRT applied（S1） | `successKind_ == 1` |
| 2 | AudioThread pinned（S2） | `successKind_ == 2` |
| 3 | P/E cores affinity skipped（S3） | `successKind_ == 3` |
| 0 | 未記録（初期値） | `successKind_ == 0` |

NonRT report は kind ごとに §3.3 の原文言で `diagLog` する。kind が未知値の場合は
出力を抑止する（将来の kind 追加時の fail-safe。現行 3 種以外は到達しない）。

### 3.3 payload（§3.2 — 実際に必要な情報だけ）

既存ログの文言から逆算した最小 payload（すべて整数）：

| kind | 既存ログの情報 | payload field | 型 |
| --- | --- | --- | --- |
| 1 | `win32Prio`（`::GetThreadPriority` の戻り値） | `successA_` | uint64（int を格納） |
| 1 | `procClass`（`::GetPriorityClass` の戻り値） | `successB_` | uint64（DWORD を格納） |
| 1 | `savedClass`（`savedProcessPriorityClass`） | `successC_` | uint64（DWORD を格納） |
| 2 | `audioMask` | `successA_` | uint64 |
| 2 | `prevMask` | `successB_` | uint64 |
| 2 | —（`successC_` は未使用 = 0） | `successC_` | uint64 |
| 3 | なし（固定文言のみ） | 全 field 未使用 = 0 | — |

3 kind で field を共有する（union 的使用）。S3 は kind のみが情報である。
`juce::String::toHexString` の整形は NonRT report 側で行う（RT では整数だけを写す）。

成功ログの意味・内容は維持する。NonRT report の文言は既存の 3 文言と同一とし、
D3 の failure report と同じく `count=` suffix を付す（coalescing の可視化。
D3-T3 の substring 検査を壊さない）。

### 3.4 coalescing semantics（§3.3 — 推測で決めない。以下は実 code からの導出）

**判定: last-wins + monotonic count で足りる。各成功イベントの個別保持は不要。**

導出根拠（いずれも実 code / 実査）：

1. **頻度**: `applyMmcssPriority()` は audio thread ごとに初回 1 回だけ実行される
   （`t_mmcssTried` guard。§2.1）。per-callback の高頻度イベントではない。
   複数 thread の Driver 所有スレッドが生成されても、各 event は「その thread の
   setup 結果」という冪等な状態報告であり、event ごとに固有の identity を持たない。
2. **consumer**: machine-readable な consumer は存在しない（§2.3）。
   失われると困る per-event 情報は存在しない。
3. **文言**: 既存ログに sequence / thread ID がない。順序つき event stream としての
   意味論を持っていない。
4. **failure transport との対称性**: D3 の failure は同一頻度・同一 consumer 構造で
   last-wins + count が成立している（Gate 承認済み）。success も同一構造である。

したがって capacity proof は「固定 7 atomic（§4）」で足り、queue の新設は不要。
`count` は単調増加（`fetchAddAtomic` / acq_rel）、snapshot は last-wins、
ownership は輸送しない（整数のみ）。

**複数 record の race**（RT record × N → NonRT report）に対する契約は D3 と同一：

- `count` は単調増加（RMW のため lost increment なし）
- snapshot は last-wins（新旧の混ざりは「新しい count に対する古い snapshot」にはならない。
  publication 順序 payload → count が保証する。D3 ordering correction と同一の論証）
- `count == N` ⇒ record #N の payload が可視
- snapshot が `count` より新しい場合は coalescing として許容（ownership なし）

### 3.5 report point（§3.4 — 新規 thread / timer / queue / authority なし）

`timerCallback()` の既存 hook（D3 の `reportAffinityFailureIfRecorded()` 呼び出しと
同一箇所、Timer.cpp:1014 付近）に `reportSuccessIfRecorded()` を 1 行追加する。
診断 backend は既存 `diagLog()`。新規 execution point は作らない。

Diagnostics OFF semantics の維持：report 関数内の `diagLog` 発行部を
`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で囲む（現行の guard 位置と同一の意味）。
OFF 時は record（RT-safe な atomic のみ）が実行され、report は bookkeeping
（`reportedCount` 進展）のみ行い backend に到達しない。ON 時の success は既存文言で出力。

---

## 4. Contract（不変条件）

| ID | 内容 |
| --- | --- |
| INV-D4-1 | RT-reachable success logging shall not acquire a mutex. |
| INV-D4-2 | RT-reachable success logging shall not allocate or free memory. |
| INV-D4-3 | Diagnostics OFF shall not be required as a precondition for RT safety. ON の success path でも RT から backend に到達しない |
| INV-D4-4 | Success observation shall not create a new publication, retire, or recovery authority. ownership を輸送しない |
| INV-D4-5 | Success transport shall use explicitly defined lossy-coalescing semantics (payload → count → observed, count == N ⇒ record #N visible). silent ownership loss is prohibited |

既存 invariant（D1 / D2 / D3-T1〜T5 / D3 publication ordering）は維持する。

### D4 atomic member（7 件）

| # | member | 役割 | writer | reader |
| --- | --- | --- | --- | --- |
| 1 | `successObserved_`（bool） | 単調 flag（fast-out 専用。常に最後の store） | RT | NonRT |
| 2 | `successCount_`（uint64） | 単調 count ＋ per-record publication marker | RT | NonRT |
| 3 | `successKind_`（uint32） | 1/2/3（§3.2） | RT | NonRT |
| 4 | `successA_`（uint64） | payload（§3.3） | RT | NonRT |
| 5 | `successB_`（uint64） | payload（§3.3） | RT | NonRT |
| 6 | `successC_`（uint64） | payload（§3.3。S2/S3 では 0） | RT | NonRT |
| 7 | `successReportedCount_`（uint64） | NonRT 専用 once-only 記録 | NonRT のみ | NonRT |

### D4 関数（2 件。D3 と同名衝突なし）

- `recordSuccessObserved(std::uint32_t kind, std::uint64_t a, std::uint64_t b, std::uint64_t c) noexcept`
  — RT 到達。payload → count(`fetchAddAtomic`/acq_rel) → observed の順で publish。
  `convo::` wrapper のみ。4 引数は無符号整数で交換可能なため D3 OBS-3 と同一理由で
  根拠付き NOLINT を付す（引数順序の変更はしない）。
- `reportSuccessIfRecorded() noexcept`
  — NonRT 専用。observed fast-out → count acquire → snapshot acquire → kind 別 `diagLog`
  （guard 内）→ `reportedCount` 進展。`timerCallback()` から呼ぶ。

---

## 5. D3 との境界（変更禁止物の実在確認）

以下は最新 source 上に実在し、**いずれも変更しない**：

- `recordAffinityFailure()`（Timer.cpp:335 付近）— publication 順序 payload → count → observed を維持
- `reportAffinityFailureIfRecorded()`（Timer.cpp:368 付近）— read 順序を維持
- D3 atomic member 6 件（`AudioEngine.h` の `affinityFailure*_`）
- D3-T1〜T5（`STG11D3AffinityFailureTests.cpp`）— 特に **D3-T3** は success 文言の存在＋guard を
  検査する。D4 は success 文言を report 関数内（guard 内）に移動するため、
  D3-T3 の `src.find(...)` と guard 検査は引き続き成立する（§8 の D4-T6 で実証する）
- D3 failure policy / `priorityApplied` / caller return-value handling / `t_mmcssTried`

D3-T1b（`applyMmcssPriority` body 内の全 `diagLog`/`juce::String` が guard 内）は、
D4 後に body 内の `diagLog` が 0 件になるため引き続き成立する。

---

## 6. Scope 外として触れないもの（記録）

- `AudioEngine.Mmcss.cpp` の file-local `diagLog` による MMCSS 登録ログ
  （`[MMCSS-*] registered` / `already registered` / `FAILED` / `reverted`）。
  同一の first-call RT 到達性を持つが、Owner の限定（`applyMmcssPriority()` の
  success path 3 件）に含まれないため触れない。次回 Discovery 候補。
- `revertMmcssPriorityOnAudioThread()` / `finalizeMmcssShutdown()` の guarded `diagLog`。
  同じく範囲外のため触れない。
- `diagLog` backend 自体（`asyncSink` / `s_logMutex` / `LockFreeRingBuffer`）は変更しない。
- Publish / Retire / Recovery / Coordinator authority は変更しない。

---

## 7. 停止条件の判定（§6 — いずれも非該当のため GO）

| 停止条件 | 判定 |
| --- | --- |
| success event の保持数に明確な capacity proof が必要になる | 非該当。固定 7 atomic で足りる（§3.4 の導出 1-4） |
| queue を新設しないと semantics を維持できない | 非該当。last-wins で semantics 維持（§3.4） |
| 既存 NonRT execution point では診断を安全に処理できない | 非該当。`timerCallback` は D3 report と同一形状の処理を既に hosting している |
| event coalescing により診断上重要な情報が失われる | 非該当。per-event consumer が存在せず（§2.3）、冪等な状態報告である（§3.4） |
| D3 の authority boundary を変更する必要がある | 非該当。別 member / 別関数（§5） |
| `diagLog` の backend 自体を変更する必要がある | 非該当。既存 backend をそのまま使う |
| Publish / Retire / Recovery authority に触れる必要がある | 非該当 |
| RT/nonRT の ownership ambiguity が発生する | 非該当。整数のみ輸送し ownership を持たない |

```
GO（Contract 成立。実装へ進む）
```
