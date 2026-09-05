# D162-2-I3-0 — Remaining Issues Triage（read-only・production/test/CMake/build 0 変更）

```text
Date:     2026-09-05
Type:     read-only triage（I1 Final PASS 確定後の残課題仕分け）
Baseline: ConvoPeq.md Generated 2026-09-04 23:38:10・I2 修復込み binary（RWDI Sep 5 10:12）
前提:     D162-2-I1-A/B/C/D = PASS（正式確定）・D162-2-I2 = PASS
判定:     **I3-0-A/B（reconfigure publication 契約）= Case C（契約曖昧・実装禁止・ADR 先行）**
          P0 課題の着手条件と全 6 課題の帰属を本記録で確定
```

---

## I3-0-A/B. reconfigure 後 `Active=0` — 4 lifetime 状態遷移グラフと契約判定

### 4 lifetime の分離評価

```text
L1: AudioEngine lifetime
    ctor（MainWindow member・h:65 AudioEngine audioEngine — unique_ptr ではない・
    interactive で再生成されない）→ initialize() → [prepare/release 反復] → dtor

L2: ShutdownRuntime admission lifetime（ISRShutdown.h・D101-30 locked contract）
    Open（ctor）→ closeAdmission()（releaseResources:87・release 1 回ごとに発火）
    → Closing → joinProducers()（:214・count==0）→ Closed
    → **Closed→Open は存在しない（INV-LIFE-9・ISRShutdown.h:299・reopen API 不存在）**
    → packedState_ は phase_（:335）と別 word（:373）で「phase は戻せるが admission は永久」

L3: RuntimeIntentCoordinator lifetime（CoordinatorState・ISRShutdown 非同期）
    Bootstrapping → Ready → ShuttingDown（requestShutdown・release:75）
    → markShutdownComplete()（release:684 — **release ごとに呼ばれる**）
    → isFullyDrained() 成立時 Bootstrapping へ復帰（Coordinator.cpp:569-573）
    → **L3 だけは再 bootstrap を想定した設計（L2 と非対称）**

L4: JUCE AudioProcessor prepare/release lifetime
    JUCE は同一 Processor に device open/close・sample rate・buffer 変更のたび
    releaseResources → prepareToPlay を反復（AudioEngineProcessor.cpp:55-67 が
    JUCE callback をそのまま委譲・「audio device 列挙時に releaseResources() を
    複数回呼ぶことがある」comment あり）
```

### reconfigure 時の state-transition（I2 実測ログと突合済み）

```text
JUCE setCurrentAudioDeviceType
  → AudioEngine::releaseResources
      L4: lifecycleState Prepared→Releasing
      L2: closeAdmission() Open→Closing（:87）→ joinProducers Closing→Closed（:214・永久）
      L3: ShuttingDown
      world clear（:503-509）・V-D（bootstrap DSP retire）・各 drain
      L3: markShutdownComplete → Bootstrapping（:684・復帰あり）
      L4: lifecycleState → Unprepared（:709）
  → AudioEngine::prepareToPlay
      L4: Unprepared→Preparing→Prepared（CAS 設計どおり・re-prepare は第一級遷移）
      L2: **無関係（admission は Closed のまま）**
      L3: 再 bootstrap 可能だが依頼側（publish）が全部 gate で落ちる
      placeholder 構築 → commitRuntimePublication → tryAdmit fail（Closed）
      → {Failed, CallerDestroy} → I2 修復で破壊（orphan 0・I2 で閉包済み）
  → session 継続: L4=Prepared・L2=Closed 永久・publish 不可能
      → [WORLD] Active=0・bypass（I1-D/I2 実測 6/6）
```

### Case A/B/C 判定（source evidence）

| 証拠 | 内容 | 指す方向 |
| --- | --- | --- |
| T8 契約（REPAIR_PLAN2-dash2.md:2811） | 「Path B resurrection: shutdown admission close 後の `enqueuePublicationIntent()` → **rejected**」— dash2 A2 の機械的 acceptance test 項目 | **Case B 寄り** |
| INV-LIFE-9（同書:3423） | 「Closed → Open は存在しない（no-resurrection）＝ **第五者レビュー #31 Phase 0 凍結対象**」 | **Case B 寄り** |
| isShutdownInProgress（AudioEngine.h:1523-1533） | 「OR 判定を**永久維持**…ShutdownRuntime のみへの完全委譲は行わない」 | **Case B 寄り** |
| publishIdleWorldOnly（Transition.cpp:15-16） | `if (isShutdownInProgress()) return false;` — idle publish に明示 guard。shutdown 後の publish 抑制は設計済み | **Case B 寄り** |
| prepareToPlay re-entrancy（:50-87） | Unprepared→Preparing は第一級遷移・rebuild thread 再起動・generation reset を実装。**re-prepare 自体は設計内** | Case A 寄り |
| prepareToPlay:217-220 comment | 「Runtime publication admission は lifecycle=Prepared を前提」— re-prepare 後も publish が機能する前提の記述 | Case A 寄り |
| markShutdownComplete を release 毎に実施（:684） | L3 の再 bootstrap 準備を release ごとに行う | Case A 寄り |
| AudioEngineProcessor comment（:58） | release 複数回は想定（ただし device **列挙**時の話） | 中立 |
| **dash2 設計文書全体** | JUCE host 契約（同一 engine の reconfigure 反復）と admission lifetime の関係に触れる記述 **ゼロ** | **Case C 要素** |

### 判定: **Case C（契約曖昧）— ADR / contract clarification を先に行うこと**

- **de-facto の design（dash2 凍結契約）は Case B**: admission close 後の publish 拒否は
  テスト項目（T8）として凍結されており、「同一 engine の reconfigure 後に publication を
  復活させる」設計は dash2 に存在しない。
- **一方 JUCE host 層の実装は Case A 前提で書かれている**: prepareToPlay は完全に
  re-entrant で、Releasing 以外の状態からは必ず再 prepare できる。Two レイヤーが
  同一 engine 上で永続的に共存する構造なのに、その組合せの契約文が**どこにもない**。
- 実務的帰結（interactive 実機）: `AudioEngine` は MainWindow member（再生成なし）のため、
  **interactive で device type / sample rate / buffer を 1 回でも変えると、以後 engine は
  publication を復活できない（bypass 継続）**。これは Case B を「意図」と呼ぶには
  機能的にあまりに重い（device 変更 = 永久 bypass）。
- よって: **実装禁止（I2/R0 結論どおり reopen・C/D 案は着手しない）**。
  次工程は ADR（契約明確化）: (i) engine lifetime = admission lifetime（Case B 正式化・
  reconfigure 時は engine 再生成を host 側契約とする）か、(ii) reconfigure-aware admission
  設計（Case A 正式化・INV-LIFE-9 再審）かを文書で決める。I2 で確認された
  ownership closure（未登録 DSP → CallerDestroy → destroyRolledBackDSP）は
  **I1 lifecycle closure として完了済み・今後この経路を不用意に再変更しない**。

## I3-0-C. Debug `prepare→release→prepare→release` crash（P1）

- **crash 位置**: 2 回目 releaseResources の `ABOUT_TO_EXIT_SCOPE`（ReleaseResources.cpp:710）
  の直後 〜 `~AudioEngine: enter`（CtorDtor.cpp:106）の直前。I2FileLogger（1 行 flush）で
  engine diag を全捕獲したが、最終行が ABOUT_TO_EXIT_SCOPE で ~AudioEngine enter が後続しない。
- **候補区間**: (a) `lifecycleRuntime_.leaveRelease()`（ISRLifecycle.cpp:108-118 —
  phase 不一致時は `std::abort()` だが abort は exit 3、観測は 139 = SIGSEGV なので
  abort ではない）、(b) releaseResources return 直後の teardown、(c) `~AudioEngine` の
  diagLog より前の区間。
- **二重処理の確定状況**:
  - isr ShutdownRuntime FSM（ShutdownComplete(8) → AudioStopped(1) の逆行）:
    `transitionTo` は backward を `transitionViolations_++` で**握り潰し**（ISRShutdown.cpp:145-148）
    — crash ではない。
  - LifecycleIsolationRuntime: Released→Preparing→Released は**遷移表どおり合法**
    （ISRLifecycle.cpp:171）。
  - slot/handle/registry/world/deferred: I2 後は slot=null・registry 空の整合状態
    （I1-D で dangling だった点は解消済み）。BISECT（修復 destroy 無効化）でも crash するため
    **I2 修復とは無関係**。
  - 残り候補: coordinator 停止状態での DeferredDeletionQueue/OwnerChannel drain、
    ~AudioEngine の re-prepare 後 teardown。**確定には debugger attach（x64dbg/WER dump）が要る**
    → I3-1 タスク（crash audit・独立）。
- **I2 影響**: なし（BISECT 済み）。harness 側は `abandonEngine()` + `_exit(0)` seam で回避済み。

## I3-0-D. Release AudioEngineHarness 0xc0000374（P2）

- 発生記録: G2（3.49s）/ G3（3.60s）/ G4（4.95s）/ I1-R2（4.25s）の ctest_release.log が
  すべて `0xc0000374`（heap corruption）で AudioEngineHarness のみ失敗（**39/40 が
  G2 以来の実績値**）。
- I2 後は即時 SEGFAULT（0.07s）に変化 — ただし production/test revert bisect でも再現するため
  **I2 無関係**。違いはリンク順（object 追加）による heap layout 差と推定（未検証）。
- 帰属（production source / test seam / link order / CRT / destruction order）は未確定。
  crash audit は Debug double-release と同時に x64dbg で実施するのが効率的 → I3-1。
- I2 の regression gate としての取り扱い: Debug 40/40 を主 gate、Release 39/40 は
  **I1-R2 baseline と同値**（追加劣化なし）として記録済み。

## I3-0-E. `commitRuntimePublication` ownership=None conservation（P3）

### 返却点 × caller 表（全走査）

| # | 返却点 | ownership | 登録状態 | CallerDSP 会計 |
| --- | --- | --- | --- | --- |
| 1 | h:4613-4614 tryAdmit fail | **CallerDestroy** | 未登録（:4631 未到達） | caller が唯一 owner → 破壊義務（I2 が履行） |
| 2 | h:4636-4637 register 失敗 | **None** | 未登録（jassertfalse） | caller が唯一 owner → **破壊義務だが None では通知されない** |
| 3 | h:4646-4647 world==nullptr | **None** | 登録済み→ScopeExit rollback 済み（new DSP）/ 温存（existing DSP） | **caller 形状依存で状態が分岐** |
| 4 | h:4650-4651 seqId==0 | **None** | 同上 | 同上 |
| 5 | h:4663-4668 OwnerChannel full | **CallerDestroy** | rollback 済み | caller が破壊（Orchestrator:291-292 前例） |
| 6 | h:4682-4692 intent queue full | **CallerDestroy** | rollback 済み | 同上 |
| 7 | h:4703 success | **Transferred** | 登録確定 | registry/world authority が所有 |

### conservation 判定（None が返ったとき DSPCore ownership は成立するか）

- **既存 registered DSP（pubResult1/Timer/Transition 形）**: rollback CAS
  （Constructing→Reclaimed のみ成功）が Active で失敗 → registration 温存 →
  **registry が ownership 保持 = conservation 成立**。caller が破壊すると UAF（I2-0-3 記録）。
- **新規 placeholder（pubResult2 形）**: rollback 成功 → 未登録 → caller（release 済みの
  生 pointer 保持者）が owner。しかし `ownership=None` は**破壊義務を通知しない**ため、
  I2 分岐（CallerDestroy 判定）を素通りして leak が成立する = **latent ownership hole**。
- **現行到達性**: #3/#4 は production では不発 —
  (a) world==nullptr は caller が build 失敗の world を渡した場合のみ（buildRuntimePublishWorld
  失敗 = OOM 時）で、呼び出し側が null world を握って呼ぶ実装は存在しない
  （T-I2-3 はテスト的に null world を注入したのみ）、
  (b) seqId は `reserveRuntimePublicationIdentity` が fetchAdd+1 で常に非 0（h:3510-3512）。
- **結論**: None は「caller が既定の ownership を保持し続ける（何も移譲されていない）」という
  意味では conservation は**形式的に成立**するが、「誰が破壊すべきか」の契約が
  new-DSP/existing-DSP で**暗黙分岐しており enum に符号化されていない**点が hole。
  → 修復は enum cleanup ではなく **ADR で None の意味を確定**（例: None = 「ownership 変更なし・
  caller が事前状態を維持」+ 新規 DSP を渡す caller は CallerDestroy 以外を
  「呼び出し側契約違反」として扱う）してから I2 分岐の拡張可否を判断する。
  現行不発のため緊急度は低（I3-1 以降）。

## I3-0-F. MMCSS / FFTBackendTests（P4/P5）

### MMCSS `err=1552`（P4）

- **定数の確定**: 1552 は winerror.h:10219 `ERROR_THREAD_ALREADY_IN_TASK`（1551=
  ERROR_INVALID_TASK_INDEX と隣接）。コードの comment（Mmcss.cpp:139「1552
  (ERROR_NO_MORE_ITEMS)」）は誤り — `ERROR_NO_MORE_ITEMS` は **259L**（winerror.h:1927）。
  `:143` の expected 分岐（5/183/259）は **1552 を catch しない**ため FAILED に到達する。
- 意味: 「当該 thread は既に別 MMCSS task に所属」= JUCE WASAPI 側が登録済みの可能性が高く、
  **正常系として success 扱いすべき事象**（comment の意図どおりなら :143 に 1552 を足す
  だけの一事）。機能影響なし（false → NativeRT/no priority fallback・AFFINITY pin は実行済み）。
- **ASIO tag 副課題の根因**: `setAudioDeviceTypeName` は session 初期化時のみ呼ばれる
  （DeviceSettings.cpp:1133/1260）ため、**device switch 後に policy が再 publish されず残値
  （SelfManagedProAudio）が使われる**。初期化時の device type 名と switch 後の device type が
  異なれば tag は不正確になる。`policyTag` は diag 表示のみのため gate 影響なし。
  修復候補（I3-1 以降）: DEV_SWITCH success 後の setAudioDeviceTypeName 再呼び出し +
  :143 の定数修正（259→正しい意味付け + 1552 追加）。

### FFTBackendTests CMake（P5）

- RWDI full build が `AlignedAllocation.h(19)` 系で失敗する原因: **`/utf-8` 未付与 target での
  CP932 誤解釈**。FFTBackendTests は CMakeLists.txt:614-627 で
  `JUCE_DSP_USE_INTEL_MKL` / `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` の compile definition と
  `/utf-8` compile option の双方が未付与（AudioEngineHarness は :1876 で `/utf-8` あり）。
  日本語 Windows（CP932）+ BOM 無し UTF-8 日本語コメント → C4819 → DIAG 枝（windows.h include 含む）
  の parse 破損 → `DIAG_MKL_MALLOC` 未定義（C3861）+ 仮引数解決失敗（C2065）。
- **stale-obj マスキング**: Debug/Release は過去に build した .obj が残っているため I2-3 で
  「通った」ように見えたが、RWDI は FFTBackendTests.dir が存在しない初回 compile で顕在化。
  **clean Debug/Release rebuild でも同様に壊れる潜在 build 破損**（lifecycle 系と完全に独立した
  CMake 課題）。修復候補: `target_compile_options(FFTBackendTests PRIVATE /utf-8)` + 
 必要 compile definition の明示（P5・CMake 修復候補）。

## 仕分け結果（I3-1 以降の着手順）

| 優先 | 課題 | 状態 | 次の扱い |
| --- | --- | --- | --- |
| P0 | reconfigure 後 publication 不可 | **Case C 確定（契約曖昧）** | **ADR / contract clarification を先に実施（実装禁止）** |
| P1 | Debug double-release crash | crash 区間特定（ABOUT_TO_EXIT_SCOPE〜~AudioEngine 間） | I3-1: x64dbg crash audit（Release crash と同時実施可） |
| P2 | Release harness 0xc0000374 | G2 以来 pre-existing・I2 無関係確定 | I3-1: Debug crash audit に併合 |
| P3 | ownership=None | 形式的 conservation 成立・new-DSP 経路に latent hole（現行不発） | ADR で None 契約確定 → 必要なら I2 分岐拡張 |
| P4 | MMCSS 1552 | **ERROR_THREAD_ALREADY_IN_TASK（誤分類確定）** + stale policy | diagnostic 修正候補（小） |
| P5 | FFTBackendTests | `/utf-8` 不足 + stale-obj マスキング確定 | CMake 修復候補（小・lifecycle 系から分離） |

本 triage は read-only（production/test/CMake/build 変更 0・build 実行は RWDI 原因特定の
ための compile 試行のみ・生成物変更なし）。
