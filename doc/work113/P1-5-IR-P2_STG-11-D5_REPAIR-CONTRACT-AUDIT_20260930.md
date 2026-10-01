# STG-11-D5 Repair Contract Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D5_REPAIR-CONTRACT-AUDIT_20260930.md`
- Work item: **STG-11-D5** — MMCSS RT 到達経路の残存 RT 契約違反修正
- Predecessor: D4 Gate OBS-D4-1 / OBS-D4-2（scope 外として記録）
- Date: 2026-09-30
- Authority: リポジトリルート `ConvoPeq.md`（Owner 指示により `ConvoPeq(20260930-121735).md` と同一の統一 authority として扱う。ファイル名の存在確認によるフォールバックは行わない）
- Commit / push: **禁止**

---

## 0. 判定

```
GO（Contract 成立。実装へ進む）
```

ただし §4 の結論が重要: **logging は D4 と同一思想で移設するが、MMCSS OS API 自体は
移動しない**。OS API を移動しない理由は「50-200μs だから」ではなく、
同一 thread 要求＋driver 所有スレッドという移動不可能性であり、
安全な設計契約（§4.3）として明示する。

---

## 1. Authority（実ファイルから再取得）

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md` |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `678B5BA15658A7AEDCED97D0E6A59EC2D3AF4C9BBE107350EF4710CC9B3DF57C` |
| size | 5,805,842 B |
| Generated | `2026-09-30 21:09:22` |
| NEWER_SRC_COUNT | 0 |
| `--check` | `STATUS : FRESH`（exit 0） |

D1 + D2 + D3 + D4 の未 commit 変更を保持（`main...origin/main [ahead 9]`、staged 0）。
`AudioEngine.Mmcss.cpp` は D1〜D4 のいずれも未変更であることを確認
（forbidden-area check で空。§7 参照）。

---

## 2. D5-1 — RT call chain audit（実証）

### A1. Registration 経路

```
Audio callback (RT)
  AudioEngine.Processing.AudioBlock.cpp:55-70 / BlockDouble.cpp:同等箇所
  （policy が SelfManagedProAudio / SelfManagedPlayback のときのみ）
        │  static_cast<void>(tryApplyMmcssForSelfManagedThread());
        ▼
  AudioEngine.Mmcss.cpp:72 tryApplyMmcssForSelfManagedThread()
        │  if (t_mmcssTried) return ...;   ← thread_local、thread 初回のみ通過
        │  t_mmcssTried = true;
        │  applyMmcssPriority();            ← D3/D4 で修正済み
        │  policy 判定（JuceManaged / None → return true）
        ▼
  AvSetMmThreadCharacteristicsW + AvSetMmThreadPriority + diagLog ×4 箇所
```

`diagLog` 5 箇所（全て `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内）：

| # | 位置 | 文言 |
| --- | --- | --- |
| M1 | Mmcss.cpp:127-129 | `[MMCSS-<tag>] registered: task=<name> priority=<str> taskIndex=<n>` |
| M2 | Mmcss.cpp:147-149 | `[MMCSS-<tag>] already registered by JUCE/driver (err=<n>) task=<name>` |
| M3 | Mmcss.cpp:171-173 | `[MMCSS-<tag>] registered (fallback): task=<name> priority=<str> taskIndex=<n>` |
| M4 | Mmcss.cpp:186-188 | `[MMCSS-<tag>] FAILED: primary err=<n> task=<name>` |
| M5 | Mmcss.cpp:200 | `[MMCSS] reverted on Audio Thread`（`revertMmcssOnAudioThread` 内） |

Mmcss.cpp の file-local `diagLog`（:32-36）は `DBG + juce::Logger::writeToLog` を
**直接**呼ぶ（Timer.cpp の `asyncSink` 経由ではないが、`juce::String` 構築＋
Logger I/O を RT 上で実行する点で同種の RT 契約違反である）。

### A2. Shutdown 経路

```
Message Thread (NonRT)
  ReleaseResources.cpp:113-123
        │  mmcssShutdownRequested = true（flag のみ）
        │  finalizeMmcssShutdown()（NonRT 上で直接実行 — RT 到達なし）
        ▼
Audio callback (RT, 次回 callback)
  AudioBlock.cpp:64-67 / BlockDouble.cpp:同等箇所
        │  if (mmcssShutdownRequested) revertMmcssOnAudioThread();
        ▼
  Mmcss.cpp:195 revertMmcssOnAudioThread()
        │  ::AvRevertMmThreadCharacteristics(t_mmcssHandle);  ← 同一 thread 要求
        │  diagLog("[MMCSS] reverted on Audio Thread");        ← M5
        │  t_mmcssTried = false（次回 device open / thread 生成時に再登録可）
```

`revertMmcssPriorityOnAudioThread()`（Timer.cpp:474）は caller が存在しない
dead code であり、scope 外（触れない）。`finalizeMmcssShutdown()` は
Message Thread 上で実行され RT 到達しないため scope 外（触れない）。

---

## 3. D5-2 — OS API RT-safety audit

### 3.1 対象 API と実行場所（実 code から）

| API | 実行場所 | thread | 頻度 |
| --- | --- | --- | --- |
| `AvSetMmThreadCharacteristicsW` | `tryTask` ← registration 経路 | Audio（RT） | thread 初回 1 回（+ fallback 再試行） |
| `AvSetMmThreadPriority` | registration 成功時 | Audio（RT） | 同上 |
| `AvRevertMmThreadCharacteristics` | `revertMmcssOnAudioThread` | Audio（RT） | shutdown 時に 1 回 |
| `GetLastError` | failure 解析 | Audio（RT） | 同上（TLS read、cheap） |
| `SetPriorityClass` / `SetThreadPriority` / `SetThreadAffinityMask` / `GetThreadPriority` / `GetPriorityClass` | `applyMmcssPriority`（D3/D4 済み） | Audio（RT） | thread 初回 1 回（direct syscall、LPC なし） |

### 3.2 Microsoft Learn による裏付け（2026-09-30 取得）

- `AvSetMmThreadCharacteristicsW`: "**Associates the calling thread** with the specified task."
  （https://learn.microsoft.com/en-us/windows/win32/api/avrt/nf-avrt-avsetmmthreadcharacteristicsw）
  すなわち登録は本質的に **calling-thread-local** であり、他 thread からの代理実行は
  API 契約上不可能である。
- `AvRevertMmThreadCharacteristics` は task 完了時に呼ぶ逆操作であり、handle は
  per-thread の登録に対応する（in-repo の `// MUST be called from same thread` と一致）。
- task 名は registry 由来（`Pro Audio` / `Audio` / `Playback` が既定で存在。
  in-repo の fallback chain と一致）。

### 3.3 判定: OS API は移動しない（理由は時間ではなく移動不可能性）

**「50-200μs だから許容」を根拠に採用しない。** 以下の architectural な理由により
NonRT への移動は不可能であり、残置を設計契約として明示する：

1. **同一 thread 要求**: `AvSetMmThreadCharacteristicsW` は calling thread を登録する。
   Message Thread / Timer からの代理登録は API 契約上不可能
   （Microsoft Learn の "Associates the calling thread"）。
   `AvRevertMmThreadCharacteristics` も同一 thread 要求（in-repo コメント）。
   Owner 指示のとおり Message Thread への単純移動は**禁止**どおり行わない。
2. **driver 所有スレッド**: ASIO の callback thread は driver が所有する。
   その thread 上で実行する最初の機会が初回 callback 自体である
   （in-repo: `thread_local ensures safety across driver-owned threads`、
   `AudioBlock.cpp:53`）。事前に別 thread で setup する選択肢が存在しない。
3. **実行時点**: 初回 callback（steady state 前）および shutdown teardown 時。
   いずれも glitch-free 保証の定常区間ではない。ただしこれを「許容の根拠」にはせず、
   「移動不可能性」の付帯事実としてのみ記録する。
4. **業界標準**: JUCE 自身も audio thread 上で MMCSS 登録を行う（WASAPI path）。
   本設計はそれと同一である。

`SetPriorityClass` / `SetThreadPriority` / `SetThreadAffinityMask` /
`Get*` 系は direct syscall（service call なし）であり、D3/D4 でも呼び出し自体は
残置している（logging のみ移設）。D5 でも同様に呼び出しは残す。

### 3.4 D5-1〜D5-7 への回答

| 項目 | 回答 |
| --- | --- |
| D5-1 registration の実行場所 | **変更なし**（audio thread 初回 callback。§3.3-1/2 により移動不可） |
| D5-2 priority 設定の実行場所 | **変更なし**（同上。`AvSetMmThreadPriority` は登録 handle に対する即時操作） |
| D5-3 revert の実行場所 | **変更なし**（audio thread の shutdown callback。同一 thread 要求） |
| D5-4 `thread_local` lifetime | **変更なし**（`t_mmcssHandle` / `t_mmcssTaskIndex` / `t_mmcssTried` は thread_local のまま。device switch で thread が死ねば TLS が消え、新 thread で再登録される既存設計を維持） |
| D5-5 ASIO driver-owned thread の制約 | 初回 callback が最初の実行機会であることを契約として明示（§4.3）。`thread_local` + lock-free observation のみで対応し、lock を持ち込まない |
| D5-6 device reopen / thread recreation 時の再登録 | `t_mmcssTried = false` による再登録経路を維持（変更なし）。再登録時は再度 observation が記録される（last-wins で最新状態を反映） |
| D5-7 shutdown ordering | Message Thread は flag のみ（現行維持）。Audio thread が次回 callback で revert＋record。`finalizeMmcssShutdown()`（NonRT）は触れない |

---

## 4. D5-3 — Repair Contract

### 4.1 修正範囲: logging のみ移設（D4 と同一思想。機械的コピーではなく §4.2 の適応あり）

```
RT（tryApply... / revertMmcssOnAudioThread）
  ↓  lock-free atomic observation（convo:: wrapper のみ。新設 transport）
既存 NonRT execution point（timerCallback — D3/D4 の report と同一箇所）
  ↓  既存 diagLog backend（Mmcss.cpp の file-local diagLog。当該 TU の既存 backend）
```

### 4.2 D4 設計からの適応点（機械的コピーでない部分）

- **task 名は wide string のため整数化する**: policy（0=ProAudio, 1=Playback）と
  task 選択子（0=primary, 1=fallback1, 2=fallback2）を記録し、task 名・
  priority 文字列・`[MMCSS-ASIO]` / `[MMCSS-DS]` tag は NonRT report 側で復元する。
  RT では整数だけを写す。
- **err / index の二重使用**: `C` field は priority（成功時）または err（既登録/失敗時）、
  `D` field は taskIndex（成功時のみ。else 0）。kind が解釈を決定する
  （D4 の S1/S2 field 共有と同一手法）。
- **report 関数は Mmcss.cpp に置く**（site と同一 TU。D3/D4 の TU に触れない）。
  backend は当該 TU の既存 file-local `diagLog`。

### 4.3 安全な設計契約（OS API 残置の明示）

```
C-D5-OS1  AvSetMmThreadCharacteristicsW / AvSetMmThreadPriority は
          audio thread の初回 callback でのみ実行する（移動しない）。
          理由: calling-thread 登録（MS Learn）＋ driver 所有スレッド。
C-D5-OS2  AvRevertMmThreadCharacteristics は登録と同一 thread 上の
          shutdown callback でのみ実行する（移動しない）。
C-D5-OS3  上記 OS API の実行結果（handle / err / index）は整数として
          lock-free observation に写し、NonRT で診断する。
          RT 上では logging backend に到達しない。
C-D5-OS4  Message Thread は従来どおり flag 設定のみ行い、Av* を実行しない。
```

### 4.4 不変条件

| ID | 内容 |
| --- | --- |
| INV-D5-1 | RT-reachable MMCSS logging shall not acquire a mutex.（Mmcss.cpp の file-local diagLog は Logger 直接 I/O のため RT から到達させない） |
| INV-D5-2 | RT-reachable MMCSS logging shall not allocate or free memory (`juce::String` 構築を含む）。 |
| INV-D5-3 | Diagnostics OFF shall not be required as a precondition. record は guard 外、report の発行部は guard 内 |
| INV-D5-4 | MMCSS observation shall not create a new authority; ownership を輸送しない（整数のみ） |
| INV-D5-5 | Lossy-coalescing（payload → count → observed、count == N ⇒ record #N visible）。D3/D4 と同一の順序契約 |

### 4.5 D5 atomic member（8 件）

| # | member | 役割 |
| --- | --- | --- |
| 1 | `mmcssObserved_`（bool） | 単調 flag（fast-out 専用。常に最後の store） |
| 2 | `mmcssCount_`（uint64） | 単調 count ＋ per-record publication marker |
| 3 | `mmcssKind_`（uint32） | 1=registered, 2=already-registered, 3=fallback, 4=FAILED, 5=reverted |
| 4 | `mmcssA_`（uint64） | policy（0=ProAudio, 1=Playback） |
| 5 | `mmcssB_`（uint64） | task 選択子（0=primary, 1=fallback1, 2=fallback2。kind 5 では 0） |
| 6 | `mmcssC_`（uint64） | priority（成功時）または err（kind 2/4。kind 5 では 0） |
| 7 | `mmcssD_`（uint64） | taskIndex（成功時のみ。else 0） |
| 8 | `mmcssReportedCount_`（uint64） | NonRT 専用 once-only 記録 |

### 4.6 D5 関数（2 件）

- `recordMmcssEventObserved(kind, a, b, c, d) noexcept` — RT 到達。
  payload → count(`fetchAddAtomic`/acq_rel) → observed の順。`convo::` のみ。
  5 引数は無符号整数のため D3 OBS-3 / D4 と同一理由で根拠付き NOLINT（順序変更なし）。
- `reportMmcssEventIfRecorded() noexcept` — NonRT 専用。`timerCallback()` から呼ぶ。
  kind 別に原文言で `diagLog`（guard 内）。未知 kind は抑止。

---

## 5. D1〜D4 との境界（変更禁止物の実在確認）

- D3: `recordAffinityFailure()` / `reportAffinityFailureIfRecorded()` / member 6 件 /
  T1〜T5 / publication ordering / failure policy — **不変**
- D4: `recordSuccessObserved()` / `reportSuccessIfRecorded()` / member 7 件 /
  T1〜T6 / ordering — **不変**
- `priorityApplied` / caller return / `t_mmcssTried` / fallback chain / error code 判定
  （5/183/1552 → true、1531 → fallback）/ return true/false sites — **不変**
- `thread_local` 3 変数・same-thread revert・Message-Thread-flag 設計 — **不変**
- `diagLog` backend 自体（両 TU）— **不変**

## 6. Scope 外として触れないもの

- `revertMmcssPriorityOnAudioThread()`（caller なしの dead code）
- `finalizeMmcssShutdown()`（NonRT。RT 到達なし）
- `AudioEngine.Timer.cpp` の D3/D4 定義（hook 呼び出しの追加 1 行を除く）
- Publish / Retire / Recovery / Coordinator / EpochDomain / RuntimeStore / Crossfade

---

## 7. 停止条件の判定（Owner §6 — いずれも非該当のため GO）

capacity proof は固定 8 atomic で足りる（once-per-thread の冪等な状態報告＋
machine consumer なし。M1〜M4 の文言に sequence/thread ID はなく、per-event 保持の
必要性を示す証拠はない）。queue 新設不要。既存 `timerCallback` で処理可能
（D3/D4 と同一形状）。coalescing で失われる診断上重要な情報なし。
D3/D4 boundary 変更なし。backend 変更なし。authority 変更なし。ownership ambiguity なし
（整数のみ）。

```
GO（Contract 成立。実装へ進む）
```
