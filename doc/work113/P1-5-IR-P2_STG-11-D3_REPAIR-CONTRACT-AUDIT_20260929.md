# STG-11-D3 Repair Contract Audit（2026-09-29）

> **Verdict: D3 = CONFIRMED / Repair Contract = PROVEN（候補 A を推奨）**
> **本書は read-only 監査である。production = 0 / test = 0 / CMake = 0 / ConvoPeq.md = 0 / commit = 0 / push = 0。**
> **D1/D2 の変更に触れていない。D1/D2 commit なし。D3 implementation には進まない。**
> **D4 / N4 / dead code / STG-8〜10 / D1 / D2 / ASAN 環境には scope を広げていない。**

---

## 1. Authority

Owner 指定: `ConvoPeq(20260929-124301).md == ConvoPeq.md`（統一 authority）。
実 source 検証は repo の現行 `ConvoPeq.md` を基準に実施した。

| 項目 | 値 |
| --- | --- |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `9565A3D546BF256F75C1A2B97FB1E9B99CFF8D6DED11A3673642EADFF4F8F5BE` |
| size | 5,710,250 B |
| Generated | 2026-09-29 21:16:18 |
| NEWER_SRC_COUNT | 0 |
| STATUS | FRESH（`--check` exit 0） |

---

## 2. Current source state

| 要素 | 状態 |
| --- | --- |
| D1（EQ member router＋double-ownership 除去） | working tree に保持。revert なし |
| D2（sink Q→E→T） | working tree に保持。revert なし |
| D3 対象（`applyMmcssPriority`／`diagLog`） | **未修正**（本 Audit は read-only） |
| staged／commit／push | 0／0／0 |

---

## 3. D3 finding

```text
D3 = diagnostics OFF 時も実行される failure logging が RT safety contract に違反する。
```

正確な欠陥文:

> `AudioEngine::applyMmcssPriority()` の affinity 失敗分岐
> （`AudioEngine.Timer.cpp:288-293`）は diagnostics guard の外側にあり、
> Release／既定 OFF ビルドでも audio thread 上で `diagLog()` を実行する。
> `diagLog()` は `std::mutex` 取得（`asyncSink`）＋ヒープ確保
> （`juce::String` 連結一時オブジェクト）を伴う。
> したがって RT 到達可能な failure path が mutex／allocation に到達する。

**到達可能性は推測ではなく source で証明する（§6）。**
「現在は実際に failure が起きにくい」ことは safety proof に使わない。

---

## 4. applyMmcssPriority call graph

### 4.1 定義・caller・thread context

| 項目 | source evidence |
| --- | --- |
| 定義 | `AudioEngine.Timer.cpp:233` `bool AudioEngine::applyMmcssPriority() noexcept` |
| 唯一の caller | `AudioEngine.Mmcss.cpp:81`（`tryApplyMmcssForSelfManagedThread` 内） |
| caller の caller | `AudioEngine.Processing.BlockDouble.cpp:66`／`AudioEngine.Processing.AudioBlock.cpp:62`（**RT audio callback**） |
| 実行頻度 | 初回 callback のみ（`t_mmcssTried` — `Mmcss.cpp:29,74-76`、`thread_local`） |
| 戻り値の扱い | 両 call site とも `static_cast<void>(...)` で**破棄** |
| その他の参照 | コメントのみ（`AudioEngine.h:2773-2781`、`Timer.cpp:229`、`ThreadAffinityManager.h:30-33`）。production 呼び出しは上記 1 件のみ |

### 4.2 関数内の failure 分岐（全列挙）

| # | 分岐 | 条件 | logging | guard |
| --- | --- | --- | --- | --- |
| F1 | NativeRT `SetPriorityClass` 失敗 | `useMmcssPriority==false` かつ `pcResult==0` | `diagLog`（`:253-257`） | **内側**（diagnostics ON のみ） |
| F2 | NativeRT `SetThreadPriority` 失敗 | 同上かつ `tpResult==0` | `diagLog`（`:258-262`） | **内側** |
| F3 | NativeRT 成功ログ | `nativeRtOk` | `diagLog`（`:263-269`） | **内側** |
| **F4** | **affinity 失敗** | `!hasHeterogeneousCores_` かつ `mask!=0` かつ `SetThreadAffinityMask==0` | **`diagLog`（`:290-292`）** | **外側（無条件実行）** |
| F5 | affinity 成功ログ | 同上かつ成功 | `diagLog`（`:296-298`） | 内側 |
| F6 | P/E cores skip ログ | `hasHeterogeneousCores_` | `diagLog`（`:305`） | 内側 |

**F4 のみが無条件実行である。** 成功時ログ（F5）との非対称は、
`BlockDouble.cpp:56` のコメント「Logging is guarded ... (zero cost in Release)」
とも矛盾する（comment-vs-code 乖離。欠陥自体ではないが guard 欠落が
意図的でないことの状況証拠）。

### 4.3 既定値（到達条件の live 確認）

| 条件 | 既定値 | source |
| --- | --- | --- |
| `useMmcssPriority` | `true` | `AudioEngine.h:2635`（`Parameters.cpp:547` で変更可） |
| `hasHeterogeneousCores_` | `false` | `AudioEngine.h:2939`（initialize 時に設定） |
| `audioMask` | topology 最終コア mask（`ThreadAffinityManager.h:249`）。`initialize`（`AudioEngine.Init.cpp:149`）後に非ゼロ | `:142,161-162` |
| diagnostics flag | OFF | `CMakeLists.txt:129` `option(... OFF)` |

したがって通常ビルド・対称コア環境では F4 の到達条件
（`!hasHeterogeneousCores_ && mask!=0`）が live であり、
残る条件は OS 呼び出しの失敗のみである。

### 4.4 failure semantics（§11）

| 問い | 答え | 根拠 |
| --- | --- | --- |
| failure は fatal か | **No**（informational） | affinity 失敗は `priorityApplied` に影響しない（`:274-276` は NativeRT のみ） |
| failure は recoverable か | N/A（回復動作なし。次回 callback で再試行もしない。`t_mmcssTried` により初回のみ） | `Mmcss.cpp:74-76` |
| caller は return value を使用するか | **No**（両 call site で破棄） | `BlockDouble.cpp:66`、`AudioBlock.cpp:62` |
| failure 時に fallback があるか | **No**（affinity 未設定のまま続行） | — |
| fallback は RT-safe か | N/A | — |

**結論: D3 は logging の安全性のみを扱い、failure handling policy を変更しない。**
`priorityApplied` の意味・戻り値の破棄・fallback なしは全て維持する。

---

## 5. diagLog call graph

### 5.1 定義・caller・guard 位置

| 項目 | source evidence |
| --- | --- |
| 定義 | `AudioEngine.Timer.cpp:136-140`: `void diagLog(const juce::String& message) { DBG(message); asyncSink(message); }` |
| diagnostics guard の位置 | **`diagLog() の内部にはない。呼び出し側にある**（または無い）。F4 には呼び出し側 guard もない |
| RT reachable caller | **F4（`:290-292`）**。他の `diagLog` 呼び出し（Mmcss 成功／失敗／revert 等）は全て `#if` 内側または NonRT 文脈 |

### 5.2 backend の到達内容（RT 視点）

| 段 | file:line | 内容 | RT 違反種別 |
| --- | --- | --- | --- |
| 1 | `:138` `DBG(message)` | Debug のみ有効（Release では消滅）。Debug では Logger／debugger 出力 | Debug 限定の backend 到達 |
| 2 | `:139` `asyncSink(message)` | **無条件実行** | — |
| 3 | `:87` `std::lock_guard<std::mutex> lock(s_logMutex)` | Message／Rebuild thread と contention しうる（`:82`）。audio callback をブロックしうる | **mutex（RT 待機）** |
| 4 | `:88-91` `pushWithWriter`＋`message.copyToUTF8` | SPSC ring（§5.4）への格納自体は lock-free だが `:87` の mutex 下で実行される | mutex 下 |
| 5 | 呼び出し式全体 | `juce::String` 連結一時オブジェクト（`+` chain＋`toHexString`）の構築・破棄 | **heap allocation＋free（RT 上）** |

### 5.3 ownership / lifetime（§12）

- `juce::String` 一時オブジェクト: 呼び出し元の式内で構築され、
  full-expression 終了時に **RT（audio thread）上で破棄**される（heap free）。
- `const char* → juce::String` の暗黙変換を含む全ての `+` が allocation を伴う
  （`"..." + juce::String::toHexString(...) + " GetLastError=" + juce::String(int)` —
  最低 3 個の一時オブジェクト）。
- `s_logBuffer`（固定 4096×256B）・`s_logMutex` は function-local static
  （`Timer.cpp:80,82`）。lifetime は process と同命で問題ないが、
  **mutex 自体が RT 到達する**ことが問題である。
- `message.copyToUTF8(entry.text, ...)`（`:89`）は固定バッファへの複写で
  allocation しない。**allocation は全て呼び出し側の `juce::String` 構築にある。**

### 5.4 既存 bridge の RT 安全性（Candidate C の前提検証）

| bridge | RT-safe か | 根拠 |
| --- | --- | --- |
| `asyncSink` | **No**（`:87` mutex） | — |
| `s_logBuffer.pushWithWriter` 直接呼び出し | **No**。`LockFreeRingBuffer` は **SPSC**（`LockFreeRingBuffer.h:2,67`「Do NOT rely ... for multi-producer」）。既存 producer が Message＋Rebuild の 2 者（`Timer.cpp:82`）であり、audio thread を第 3 producer として追加できない | SPSC 契約違反 |
| `DBG` backend | **No**（Debug backend 到達。Release では消滅するが、OFF/ON 両対応にならない） | — |

**結論: 既存 bridge に RT-safe なものは存在しない。Candidate C の前提は不成立。**

---

## 6. RT reachability proof

```text
Audio Thread
    ↓  processBlock(double/float) — BlockDouble.cpp:66 / AudioBlock.cpp:62
caller: tryApplyMmcssForSelfManagedThread() — Mmcss.cpp:72
    ↓  初回のみ（t_mmcssTried, thread_local）
applyMmcssPriority() — Timer.cpp:233
    ↓  affinity branch（既定 live: !hasHeterogeneousCores_ && mask!=0）
failure: SetThreadAffinityMask == 0 — Timer.cpp:286-288（環境依存の失敗）
    ↓  無条件 diagLog — Timer.cpp:290-292（guard 外）
mutex: s_logMutex 取得 — Timer.cpp:87（Message/Rebuild と contention 可）
allocation: juce::String 一時オブジェクト構築・破棄 — Timer.cpp:290-292 の式全体
logging backend: DBG（Debug）＋ asyncSink（常時）
```

**判定: RT unsafe（到達可能）。**

到達性の各段は source で確定した（caller chain・既定値・guard 位置・backend 実体）。
「failure が起きにくい」ことは safety proof に使っていない。
唯一の環境依存条件は OS 呼び出しの失敗であり、到達可能性の否定材料にならない。

---

## 7. Failure semantics（§11 の結論）

§4.4 のとおり。D3 は `diagnostic logging を安全にする` ことのみを扱い、
`failure handling policy を変更しない`。以下を全て維持する:

- `priorityApplied` の意味（affinity 失敗はinformational のまま）
- 戻り値破棄（両 call site）
- fallback なし
- 初回のみ実行（`t_mmcssTried`）

---

## 8. Candidate A/B/C comparison

### 8.1 定義

| 候補 | 内容 |
| --- | --- |
| **A** | RT failure path では logging を行わず、failure state／metric を lock-free な RT-safe mechanism（atomic）に記録し、NonRT 側で診断する |
| **B** | diagnostics guard を caller 側へ移動（`#if` で F4 を囲む） |
| **C** | 既存 diagnostics infrastructure の bridge を RT から直接使う |

### 8.2 比較（Owner §8 の採用基準で評価）

| 評価軸 | A（推奨） | B | C |
| --- | --- | --- | --- |
| RT lock = 0 | **0**（atomic store のみ。既存 `convo::` wrapper 流儀） | 0（OFF 時のみ） | ×（asyncSink mutex） |
| RT allocation = 0 | **0**（`juce::String` 構築を RT から除去） | 0（OFF 時のみ） | ×（同左） |
| RT blocking = 0 | **0** | 0（OFF 時のみ） | × |
| RT filesystem = 0 | **0**（変更なし） | 0 | ×（DBG backend は Debug で到達） |
| RT logging backend = 0 | **0**（RT 側は backend に触れない） | 0（OFF 時のみ） | × |
| raw std::atomic = 0 | **0**（`convo::` wrapper 使用を契約） | 0 | 0 |
| new retire authority = 0 | **0**（atomic のみ。queue なし） | 0 | 0 |
| new publication authority = 0 | **0** | 0 | 0 |
| semantic behavior change | 最小（logging 位置の移動のみ。failure policy 不変） | 最小だが不十分 | — |
| diagnostics ON でも安全か | **Yes** | **No**（ON 時は現状のまま危険） | No |
| diagnostics OFF でも安全か | **Yes** | Yes | No |

### 8.3 判定

| 候補 | 判定 | 理由 |
| --- | --- | --- |
| **A** | **ADOPT（推奨）** | ON/OFF 両対応の唯一の候補。新 authority なし。failure policy 不変 |
| B | REJECT（単独では） | OFF 時のみ安全。Owner §8「OFF なら安全だけでは不十分」に該当 |
| C | REJECT | §5.4 のとおり既存 bridge に RT-safe なものが存在しない（Owner の採用条件を満たさない） |

### 8.4 Selected repair contract（A の具体形）

```text
RT 側（applyMmcssPriority の F4 分岐）:
  - juce::String 構築＋diagLog 呼び出しを除去する。
  - 代わりに failure state を lock-free atomic に記録する
    （例: sticky flag＋counter＋mask/err 値。convo:: wrapper のみ使用）。
  - return 値・分岐構造・failure policy は変更しない。

NonRT 側（既存 NonRT execution point。Timer tick／health-monitor 経路から選択。
          implementation stage で確定）:
  - atomic を読み、設定されていれば既存 diagLog backend で診断出力する。
  - RT 側の記録は上書き coalescing（last-wins＋cumulative count）とし、
    loss semantics を明示する（INV-D3-5）。
```

変更範囲の見込み（implementation stage 用）:

```text
src/audioengine/AudioEngine.Timer.cpp（F4 分岐のみ）
src/audioengine/AudioEngine.h（atomic member 宣言のみ）
診断読み出し側 1 箇所（既存 NonRT tick。 신규 timer/thread/authority なし）
```

---

## 9. Invariant impact

### 9.1 採用する D3 invariant（必要性を確認の上で採用）

| ID | 内容 | 必要性の根拠 |
| --- | --- | --- |
| INV-D3-1 | RT-reachable failure logging shall not acquire a mutex. | §5.2-3（`s_logMutex` 到達が核心） |
| INV-D3-2 | RT-reachable failure logging shall not allocate or free memory. | §5.2-5、§5.3（一時オブジェクトの RT 破棄） |
| INV-D3-3 | Diagnostics OFF shall not be required as a precondition for RT safety. | §8.2（B の却下理由。ON/OFF 両対応） |
| INV-D3-4 | Diagnostic observation shall not create a new publication, retire, or recovery authority. | A が atomic のみで authority を作らないことの固定 |
| INV-D3-5 | All RT diagnostic transport shall use explicitly defined lossy-coalescing semantics; silent ownership loss is prohibited. | A の記録方式（last-wins＋count）の意味を固定。D1/D2 の教訓（silent loss 禁止）を診断 transport に適用 |

既存 invariant の番号体系は変更しない（D3 用に新規採番のみ）。

### 9.2 Practical Stable ISR Bridge Runtime との整合（§9）

- RT は待たない／解放しない／判断しない: A は RT 側に atomic store のみを追加し、
  判断・解放・待機を追加しない。**維持**。
- 新 authority なし: queue／retire／publish／recovery のいずれも作らない。**維持**。
- `RT failure → diagnostic observation → NonRT diagnosis` のみ許可し、
  `policy decision → repair/retire/publish` は作らない。**維持**。

---

## 10. Test contract（§15。実装なし）

### 10.1 D3-T1〜T4 の4象限

| ID | 条件 | oracle の方向 |
| --- | --- | --- |
| D3-T1 | diagnostics OFF＋MMCSS success | RT path に logging backend 到達なし（structural）＋ metric 非設定 |
| D3-T2 | diagnostics OFF＋MMCSS failure | **RT path に mutex/alloc なし（structural）**＋ metric 設定＋ NonRT 診断出力 |
| D3-T3 | diagnostics ON＋MMCSS success | 既存 guarded log の非回帰（NonRT 診断の内容維持） |
| D3-T4 | diagnostics ON＋MMCSS failure | **RT path に mutex/alloc なし（structural）**＋ metric 設定＋ NonRT 診断出力 |

### 10.2 RT／NonRT の分離

- **RT 側**: audio thread を起こさず検証する。RT 到達可能な failure path が
  `diagLog`／mutex／`juce::String` を含まないことの **structural regression test**
  （`tools/retire_authority_verifier.py` の前例に倣う source-pattern 検証）。
  決定論的・全 configuration で実行可能。
- **NonRT 側**: metric record→read→diagnose の機能 test。
  recorder を単体 test 可能な単位にすること（implementation stage の要件）。
- failure 注入（OS 呼び出し失敗の再現）は行わず、metric transport の両端を
  独立に検証する（注入機構の新設は scope 外）。

### 10.3 ASAN（§16）

- `existing ASAN limitation`（D1/D2 Gate で記録した `0xC0000139` 環境 block）と
  `D3-specific memory safety evidence` を分離する。
- D3 の修正は RT path から heap object を**除去**する方向であり、
  新規 heap 確保を追加しない。memory-safety surface は縮小する。
- evidence は structural test（mutex/alloc 不在）＋ unit test（metric transport）で代替する。

---

## 11. Scope boundary（§14）

D3 の scope は D3 failure path（`applyMmcssPriority` の F4 分岐＋RT 側記録＋
NonRT 側診断読み出し）に限定する。以下に触れない:

```text
D4／N4／resetFadeStateAndRetireTarget／STG-8／STG-9／STG-10／D1／D2／
D2 Gate の remaining risk（N4／dead code／harness CWD／ASAN environment）
```

特に NativeRT failure ログ（F1/F2。既に guarded）・MMCSS 成功ログ・
`revertMmcssOnAudioThread`・`ThreadAffinityManager` の責務再編は scope 外。

---

## 12. Remaining uncertainty

| # | 項目 | 扱い |
| --- | --- | --- |
| 1 | NonRT 診断読み出しの exact hook 点（Timer tick vs health-monitor） | implementation stage で既存 NonRT execution point から選択する。本 Audit では特定しない |
| 2 | metric の exact field 構成（flag＋count＋mask/err） | 上記 1 と同時に確定する。lossy-coalescing（INV-D3-5）を満たす範囲で最小にする |
| 3 | guarded 成功ログ（F5）の扱い | 現状維持（NonRT 診断として無害）。変更しない |
| 4 | `juce::String` の暗黙 alloc の完全列挙 | RT から `diagLog` 呼び出し自体を除去するため、列挙は不要になる |

---

## 13. Implementation prerequisites

```text
1. §8.4 の contract を満たす最小 diff（Timer.cpp F4＋AudioEngine.h member＋NonRT 読み出し）
2. failure policy・戻り値・初回のみ実行の不変（§4.4、§7）
3. raw atomic 追加なし（convo:: wrapper のみ。§13）
4. 新 authority なし（§9）
5. §10 の test contract（structural＋unit。test implementation は GO 後）
```

---

## 14. Verdict

```text
STG-11-D3 Repair Contract Audit = COMPLETE

production source changes = 0
test source changes = 0
CMake changes = 0
commit = 0
push = 0
staged = 0

D3 status:
  CONFIRMED

RT reachability proof:
  PASS（§6: Audio Thread → 初回 callback → applyMmcssPriority →
        affinity 失敗 → 無条件 diagLog → mutex＋heap。reachable）

failure semantics:
  PASS（§4.4、§7: informational・policy 不変。logging の安全性のみが対象）

repair candidate:
  A（RT 側は atomic 記録のみ、NonRT 側で診断。ON/OFF 両対応）
  B は REJECT（OFF 時のみ安全）。C は REJECT（proven-safe bridge なし）。

minimum repair scope:
  AudioEngine.Timer.cpp（F4 分岐のみ）
  AudioEngine.h（atomic member 宣言のみ）
  診断読み出し側 1 箇所（既存 NonRT tick）
  ISRRetireRouter／Epoch／RT／Coordinator／authority／EQ 系は不変

new invariant required:
  INV-D3-1〜INV-D3-5（§9.1。必要性を確認の上で採用）

implementation readiness:
  GO（候補 A。§12 の hook 点・field 構成を implementation stage で確定すること）

Verdict = READY FOR IMPLEMENTATION
```

**`Repair Contract = PROVEN`（候補 A）に収束した。ただし implementation には進まない。**
Owner の implementation GO を待つ。
