# D162-2-I1-D-R0 — Reconfigure Orphan Ownership / Admission-State Audit（read-only）

```text
Date:      2026-09-05
Type:      read-only root-cause audit（production 変更 0 / test 0 / CMake 0 / build 0）
Baseline:  ConvoPeq.md Generated 2026-09-04 23:38:10（R2 A′ 込み）+ R2 binary Sep 4 23:09
Input:     D162-2-I1-D Profile D 実測（D-2〜D-7 6/6 決定論的 orphan）+ PROFILE_D_COMPLETION.md
判定:      **GO（A 案 = prepareToPlay 内 CallerDestroy 応答の destroy 追加を最小修復単位として確定可）**
           I1 Final HOLD 理由も含め §10 参照
```

---

## 1. `tryAdmit()` / `CallerDestroy` の契約確定

### 1.1 CallerDestroy の契約文（実コード）

`AudioEngine.h:3632-3641`:

```cpp
// ★ work70 Phase2: OwnershipDisposition を追加 — publish 失敗後の DSPCore
//   所有権状態を呼び出し元に通知し、リークを防止する。
//   Transferred: publish 成功、所有権移譲済み
//   CallerDestroy: publish 失敗、rollback 完了、呼び出し元が物理解放すべき
//     （実際の破棄は DSPLifetimeManager::destroyRolledBackDSP() 経由）
enum class OwnershipDisposition : uint8_t { None, Transferred, CallerDestroy };
```

**契約は明示的**: `CallerDestroy` は「rollback 完了後、**呼び出し元が DSPCore を物理解放すべき**」
という義務の通知であり、破棄手段まで指定されている（`destroyRolledBackDSP()` 経由）。
PublicationExecutor.cpp:68-71 も同一解釈の二次文献:

```cpp
// ★ work70 Phase2: OwnershipDisposition::CallerDestroy の場合、
//   呼び出し元が DSPLifetimeManager::destroyRolledBackDSP() を呼んで
//   物理解放する必要がある。DSPHandle は既に rollback 済み（Reclaimed）。
```

### 1.2 tryAdmit の位置づけ（ISRShutdown.cpp:501-522 / ISRShutdown.h:293-311）

- `tryAdmit(n)`: `packedState_`（Open/Closing/Closed/Faulted + version + reservation count の
  32bit CAS）で Open のときのみ reservation 取得。**Closed の場合は side-effect zero で false**。
- `closeAdmission()`（:429-457）: Open→Closing（version++・shutdownGeneration++）。
- `joinProducers()`（:464-485）: Closing→Closed（count==0 必須・不可逆）。
- **`ISRShutdown.h:299`: 「Closed→Open は存在しない（INV-LIFE-9 no-resurrection）」**。
  reopen / reset / version-rollback の API はクラスに存在しない（全メンバ走査で確認済み）。

### 1.3 CallerDestroy が返る失敗点（enqueueRuntimePublicationFireAndForget 内）

| 失敗点 | 戻り値 | DSPCore 状態 |
| --- | --- | --- |
| `:4613` tryAdmit 失敗 | `{Failed, CallerDestroy}` | **`registerDSPHandleForRuntime`（:4633）に到達せず・map 未登録** |
| `:4651` seqId==0 | `{Failed, CallerDestroy}` | 未登録（同上） |
| `:4667` OwnerChannel enqueue 失敗 | `{Failed, CallerDestroy}` | 登録済み → ScopeExit で rollback 済み |
| `:4691` ISR intent queue full | `{Failed, CallerDestroy}` | 同上 |

**Profile D orphan は最初の行**（tryAdmit 失敗）で、registration 前に CallerDestroy が返る。

## 2. ownership graph（aligned_unique_ptr::release() → 失敗帰還）

### 2.1 Profile D（reconfigure・admission Closed）の実測 chain

```text
T0  PrepareToPlay.cpp:241  aligned_unique_ptr<DSPCore> placeholderDSP = aligned_make_unique<DSPCore>()
    → 所有者: placeholderDSP（unique・1 つだけ）
T1  :264  setActiveRuntimeDSP(placeholderDSP.release())
    → 所有者: なし（release() は所有権放棄のみ・activeRuntimeDSPSlot は所有権契約を
      持たないレガシー topology slot — h:2245 comment「placeholder 専用のレガシースロット」・
      「寿命は publish/retire の順序に従い、非所有で管理」(fading slot 同様の記述 :2210)）
T2  :277  commitRuntimePublication(RegistrationContext::needsRegistration(getActiveRuntimeDSP()), null)
T3  AudioEngine.h:4613  tryAdmit(1) → **false（admission Closed）**
    → :4614 return { Failed, CallerDestroy }  — DSPCore は T1 以降 owner なし
T4  PrepareToPlay.cpp:280  juce::ignoreUnused(pubResult2)
    → **CallerDestroy 義務を誰も履行しない**
T5  session 終了まで: world publish されない（world.engine.current は該当 DSP を参照しない）。
      activeRuntimeDSPSlot は保持するが RT は resolveActiveRuntimeDSPFromRuntimeWorldOnly
      （world first・AudioBlock.cpp:306-311 dsp==nullptr で silent clear）のため参照しない。
      registry 上の状態: DSPState::Constructing のまま（create 時の初期値・activate されない）
T6  shutdown: ReleaseResources.cpp:331 retire(slot ptr=placeholder) →
      retireDSPHandleForRuntime → map.find MISS → retired=0 → enqueue せず return
      （DSPLifetimeManager.cpp:45-49 early return）
T7  V-D（:465-479）: activeHandle/fadingHandle とも null（activate された handle が存在しない）
      → 不発。~AudioEngine（CtorDtor.cpp:204-214）: retireByHandle も同一 handle authority で null
      → 不発。E-2 により pointer-value retirement は構造的に廃止済み（:190-203 comment）
T8  process exit → OS 回収（~108MB）
```

**「3番目（どちらにも戻らない）」が確定** — T1 で unique owner が消滅し、T4 の
CallerDestroy 契約が未履行のまま、以後いかなる authority も DSPCore への
identity（handle）を再構築できない。**ownership violation の直接実装点は
`PrepareToPlay.cpp:264` の release()（所有権放棄が publish 成功確認より先行）+
`:280` の ignoreUnused（失敗応答の破棄）の組**。

### 2.2 他 caller が leak しない理由（対比確定・§2.1 の chain が D 固有である根拠）

| caller | DSP 源 | tryAdmit 失敗時の帰趨 | leak |
| --- | --- | --- | --- |
| prepareToPlay:155/277・Timer:994・Transition:25 | **既に map 登録済み DSP**（`needsRegistration(既存 DSP)` → register は既存 handle 返却・h:4327-4329 idempotent） | rollback CAS（`rollbackRegistration`: Constructing→Reclaimed、ISRDSPHandle.cpp:157-173）が **Active 状態の DSP では失敗** → 登録温存 → EBR/authority で回収可能 | **なし** |
| rebuild/recovery 経路（Commit.cpp:804 → Orchestrator trySubmit） | **事前登録（Constructing）** | Orchestrator.cpp:70 tryAdmit 失敗 → RejectedShutdown かつ submitPublishRequest の RejectedShutdown case で `retireRegisteredDSP`（:444） | **なし** |
| PublicationExecutor（CoordinatorLoop 上再送） | `alreadyRegistered` | rollback 対象外（登録済み）・失敗時 Orchestrator:291-292 `destroyRolledBackDSP` | **なし** |

つまり **registration が先行している caller は全て回収経路が存在し、唯一
`PrepareToPlay.cpp:277` の「未登録 DSP を needsRegistration で投げて失敗する」経路だけが
owner なしに到達する**。孤立性が確認できた。

## 3. prepareToPlay の failure return 処理（全 caller 走査）

`commitRuntimePublication` の caller 5 箇所（h:4706 同期ラッパ経由含む）:

| caller | 戻り値処理 |
| --- | --- |
| PrepareToPlay.cpp:155-158（pubResult1・idle #2） | `juce::ignoreUnused` |
| **PrepareToPlay.cpp:277-280（pubResult2・idle #3・orphan 元）** | `juce::ignoreUnused` |
| Timer.cpp:994-997（pubResultTimer・idle #5） | `juce::ignoreUnused` |
| Transition.cpp:25-28（publishIdleWorldOnly 内） | `juce::ignoreUnused` |
| PublicationExecutor.cpp:52-72（executePublish） | **stage 検査あり**（`!isCommitted` → log + PublishFailed 通知・destroy は Orchestrator 側 :291-292 が担当） |

prepareToPlay 系 4 caller はいずれも結果を検査しない。Design 意図上は
「registration が先行していて失敗しても registered のまま = 回収可能」が暗黙前提だったが、
**PrepareToPlay.cpp:277 だけはその前提が破れている**（未登録 DSP を渡す唯一の呼び出し）。

## 4. 未登録 DSP terminal disposition site 全列挙（既存 authority の適用可否）

| # | site | 対象要件 | 未登録 DSP へ適用可否 |
| --- | --- | --- | --- |
| 1 | `DSPLifetimeManager::retire(void*)`（cpp:35-77） | runtimeDSPHandleMap_ 登録必須（find MISS → false → **early return・破壊しない**） | ✗ |
| 2 | `DSPLifetimeManager::retireByHandle(handle)`（cpp:79-137） | handle 必須 | ✗ |
| 3 | `RuntimePublicationOrchestrator::retireRegisteredDSP(req)`（:742-759） | `resolveDSPHandle(req.newDSP)` 非 null 必須（:748-749 no-op） | ✗ |
| 4 | S3 `clearDeferredForShutdown`（Orchestrator.cpp:588-649） | deferredSlot_ 内 req → 上記 3 経由 | ✗（slot 対象のみ・orphan は slot に不在） |
| 5 | V-D（ReleaseResources.cpp:534-554） | active/fading handle 非 null | ✗（handle が存在しない） |
| 6 | pendingTask retire（CtorDtor.cpp:178-179） | 上記 1 経由 | ✗ |
| 7 | `DSPLifetimeManager::destroyRolledBackDSP(void*)`（cpp:149-157） | **void* 直接・EBR 不要** | ✓（呼び出し site なし = 今回の gap） |
| 8 | DSPGuard（RebuildDispatch.cpp:932-964） | 未登録確認（lookup isNull jassert）+ retire false → `destroyDSPCoreNode` 直破壊 | ✓（未登録 DSP の破壊前例・rebuild-obsolete 専用のスコープ） |
| 9 | recovery warmup 失敗破壊（RebuildDispatch.cpp:1034-1038/1116-1120） | 未コミット DSP 直破壊 | ✓（同前例） |

**結論**: 未登録 DSP を terminal disposition できる既存 primitive は
`destroyDSPCoreNode`（直破壊・D0 で allocator mismatch 修正済み）に存在し、
未登録 DSP に対する EBR はそもそも不要という契約が `AudioEngine.h:4314-4315` に明記
（「未登録 DSPCore（handle map 不在・publish されない）→ DSPGuard 契約の destroyDSPCoreNode
直接破壊（EBR epoch 保護不要）」）。**D162-2 の「registered DSP は authority→EBR 一本化」
原則（h:4310-4313・INV-D162-1/3）は registered DSP にのみ適用される規約であり、
未登録 DSP を destroyDSPCoreNode で破壊することは原則違反ではない**（むしろ契約の
そう定める正規経路）。

## 5. shutdownRuntime_ Closed と device reconfigure の境界（INV-LIFE-9 と再開の非対称）

```text
layer 1: ShutdownRuntime (admission FSM・ISRShutdown.h)
         Open → Closing → Closed 一方向。reopen API 不存在（INV-LIFE-9 no-resurrection）。
         対象 reservation: Publication / Recovery / Build の 3 経路（Retire は含まない :308）。
         → AudioEngine 1 instance の生涯で publication admission は実質 1 回限り。

layer 2: RuntimeIntentCoordinator::CoordinatorState (intent loop・ISRRuntimePublicationCoordinator.h:119-127)
         Bootstrapping → Ready → Publishing → ... → ShuttingDown
         markShutdownComplete() で ShuttingDown → **Bootstrapping へ復帰あり**（cpp:563-574、
         isFullyDrained 成立時）。こちらは再 bootstrap を想定した設計。
```

**2 layer 間の非対称が本件の設計上の境界**:

- JUCE 契約上、同一 AudioProcessor（=同一 AudioEngine）に対して
  `releaseResources()` + `prepareToPlay()` は **device open/close・sample rate 変更・
  buffer 変更のたびに反復される**（AudioEngineProcessor.cpp:55-67/35-53 が JUCE callback を
  そのまま委譲）。engine lifetime ≠ admission lifetime の非対称が構造的背景。
- D116 の restart cycle（修正前 D116-7・6/6 cycle PASS）は「**AudioEngine 自体の
  生成し直し**」を伴う restart であり、admission reopen と矛盾しない。今回の orphan は
  **同一 engine instance 内での re-prepare** で初めて顕在化（A/B/C/D profile は
  engine 1 世代 + reconfigure なしで完遂）。
- `setShutdownPhase(ShutdownPhase::Running)`（PrepareToPlay.cpp:72）は shutdownPhase
  （表示・診断用 sequential state）のみの復帰であり、**admission FSM（packedState_）とは
  独立**。つまり現行実装は「phase は Running に戻るのに admission は Closed のまま」
  という状態を許容しており、prepareToPlay の publish が全て RejectedShutdown に落ちる
  静的不可視状態を作る（実測 `[WORLD] Active=0`・publish 0・rebuild queued 5/5消化 0）。

## 6. ~AudioEngine が orphan を回収できないことの再確認

- CtorDtor.cpp:132-135/190-214: `activeToRelease` は観測用変数のみ
  （E-2 で pointer-value retirement を構造的に廃止 — dangling address-reuse 対策）。
  回収は `retireByHandle(getActiveRuntimeDSPHandle())` の **handle authority 単独**。
- orphan は `activate()` されたことがないため `activeRuntimeDSPHandle_` は null
  （ISRDSPHandle.cpp:100 のみが set）→ `:207 isNull() → no-op`。
- graceful drain / quarantine sweep / overflow drain はすべて **intent queue・registry・
  quarantine store に存在するエントリ**が対象で、**どこにも登録されていない裸の DSPCore\*
  を列挙する手段が存在しない**（slot 値は所有権 authority に昇格させないという E-2 契約の
  帰結）。
- よって ~AudioEngine での救済は現契約下で不能。**orphan は process exit のみが回収点**。

## 7. 修復候補 A/B/C/D 比較（決定はしない・比較のみ）

| 案 | 内容 | 所有権整合 | EBR/authority 原則との整合 | 副作用リスク | 評価 |
| --- | --- | --- | --- | --- | --- |
| **A** | prepareToPlay が CallerDestroy を検査して `destroyRolledBackDSP`（= destroyDSPCoreNode）を呼ぶ（PrepareToPlay.cpp:277 のみ・失敗時のみ） | **◎** — release() で放棄した owner 責務を契約どおり caller が履行。CallerDestroy 契約（h:3635 + PublicationExecutor.cpp:68-71）の文面上の正しい履行 | ◎ — 未登録 DSP への直破壊は h:4314-4315 の正規契約。registered DSP には触れない。EBR 対象外 | 低。破壊対象は「admission Closed で publish 失敗した未登録 placeholder」のみ。Audio Thread 停止中（prepareToPlay 契約内）で RT 参照なし。destroyDSPCoreNode は D0/F で破壊安全性確認済み | **最小・契約履行型**。tryAdmit 失敗点（:4613-4614）と registration 前失敗点（:4651）の 2 点で帰る CallerDestroy を同一処理で吸収可 |
| B | admission failure 時に authority が ownership を引き取る（commitRuntimePublication 側で失敗 DSP を破壊） | △ — engine が失敗した API が自己の入力引数（未登録 raw ptr）の lifetime を引き受ける責務分担。ただし needsRegistration 契約は「caller が事前状態を保証」ではなく「engine が登録責任」であり、rollback（Reclaimed 化）前提の cleanup は Orchestrator 既存パターン（:291-292 destroyRolledBackDSP）と同型 | ○ | 中。commitRuntimePublication は 5 caller 共通口であり、呼び出し側の事前状態（登録済み/未登録）が混在。失敗点によって DSP が登録済み（ScopeExit rollback と二重破壊競合）/未登録が分かれるため、**破壊責任の分岐を API 内に埋め込むことになり現行の ScopeExit 契約との共存が複雑化**（h:4626-4630 rollback guard との順序設計が必要） | A と同効果だが contract 面は A の方が局所。既存 Orchestrator:291-292 パターンと合わせるなら「caller 側 destroy」で一貫 |
| C | device reconfigure 後に shutdownRuntime_ を再 Open する設計（admission reopen / 新 FSM phase） | ○（発行側は復活する） | — | **大**。INV-LIFE-9 no-resurrection は D101-30 locked contract（G-H race は packedState_ CAS で閉じている・version ABA 防御含む）。reopen を導入すると QuiescenceObservation Q7（NoResurrection = Closed 固定）・Proof/Permit identity 契約・I0-G4 の全 audit chain に波及。**ISRLifetimeProof 層の再設計に等しい** | 修復コスト・監査面積が最小単位と釣り合わない。admission lifetime = engine lifetime という現在の設計判断を反転させる大改修。**推奨しない**（将来的に engine 内 re-prepare を本気で publication 対応させるなら検討余地あり） |
| D | tryAdmit 自体が reconfigure 後の lifecycle state を想定していない問題を前提化（例: prepareToPlay で admission Closed を検知したら placeholder publish 自体を skip） | △ — publish しないので orphan は生まれない。ただし placeholder 生成も skip すると RT パスが nulldsp で silent clear（実測済みの挙動と同じ）になるため実害は生じない | ◎ | 低〜中。ただし **hasPublishedCurrent==false の系（startup・reconfigure 共通）で placeholder publish を skip する」という条件分岐は、admission 状態を engine 外（prepareToPlay）が読む設計になり、PrepareToPlay が shutdownRuntime_ を参照する新規依存を生む**。また skip 時に build 意図（rebuild 5/5）が admission gate で静止したままという本質（session 全体 unpublished）は残る | orphan は消えるが「reconfigure 後 engine が生きない」設計矛盾を温存・隠蔽する。リスク低だが問題の修理としては不完全 |

**比較結論**: 所有権契約の観点では **A が唯一「契約に書かれた義務を履行していない caller に
履行させる」最小変更**であり、B は A と同効果で contract 拡散が大きい、C は locked contract
（INV-LIFE-9）への反転で監査面積が桁違い、D は orphan の隠蔽に留まる。

## 8. 最小修復単位の決定は可能か

**可能（確定）**。最小修復単位 = **A 案の 1 箇所**:

```text
PrepareToPlay.cpp:277-280（pubResult2 = idle publish #3・placeholder）
  if (!PublishStageResultTraits::isCommitted(pubResult2.stage)
      && pubResult2.ownership == OwnershipDisposition::CallerDestroy)
      → destroyRolledBackDSP(getActiveRuntimeDSP()) 等で該当 DSP を 1 回破壊
      → setActiveRuntimeDSP(nullptr)（slot クリア）
```

根拠:

1. 失敗帰還 CallerDestroy の義務者 = caller（§1.1 契約文・§4 既存前例）。
2. 未登録 DSP の直破壊 = h:4314-4315 の正規契約（D162-2 原則に非違なし）。
3. 対象 DSP は `release()` 直後で RT 参照ゼロ（prepareToPlay は Audio Thread 停止中実行）。
4. destroyDSPCoreNode は D162-2-F で allocator 契約修正済み（D0 signature 消失済み）。
5. 範囲が PrepareToPlay.cpp 1 関数内・production diff 数行。pubResult1（:155）は
   既存登録済み DSP の idle publish で rollback CAS 失敗 → 登録温存（§2.2）のため
   CallerDestroy 到達時の leak が原理的に発生しないが、**契約履行の一貫性**として
   同型 check を入れるか否かは I2 設計時の裁量（最小単位では :277 のみで十分・
   :155 は失敗しても leak しないことが §2.2 で証明済み）。

**付帯確定事項（I2 設計の入力）**: A 案でも「reconfigure 後の session が publication 不可
（Active=0・bypass 継続）」という §5 の境界問題は残る（A は memory 契約のみ修理）。
publication 復活は C/D 案級の設計変更を要するため、**I2 の scope を
「ownership repair（A）」と明確に切り、publication 復活は別 track** とすることが安全。

## 9. GO / NO-GO

**GO** — 以下を根拠に D162-2-I2（reconfigure orphan repair・A 案最小単位）の
実装可否判断に進めてよい:

- 契約・ownership・孤立性・既存 primitive・破壊安全性・最小単位の 6 点がすべて
  source 実装で確定（§1-§4・§8）。
- D162-2 の authority 原則（registered=EBR 一本化）と未登録直破壊契約（h:4314-4315）
  が衝突しない。
- production 変更は PrepareToPlay.cpp 1 関数に限定可能（B/C/D 案の広範囲変更と対照的）。

## 10. I1 Final を HOLD する理由（記録用の正式文言）

```text
D162-2-I1-A  PASS
D162-2-I1-B  PASS
D162-2-I1-C  PASS
D162-2-I1-D  STOP（lifecycle gate: constructed DSP = destroyed DSP が pointer identity で
             6/6 不成立。crash / ordering / disposition / guard 系 12 gate は全 PASS であり
             A′ 修復自体の regression ではない）

A′ deferred terminal disposition: proven effective（B 6/6 retired=1 再現・C/D no-op 冪等）

New independent finding (D-R0 確定):
    device reconfigure → prepareToPlay → admission Closed（INV-LIFE-9）
    → needsRegistration publish が tryAdmit 失敗 → CallerDestroy 未履行
    → unregistered DSPCore ~108MB が terminal disposition なしで process-exit reclamation
    （root cause: PrepareToPlay.cpp:264 release() + :280 ignoreUnused の組・§2.1）

I1 Final HOLD 理由:
    I1 の合格は「A′ 修復込み binary で shutdown lifecycle 会計が崩れないこと」の運用検証
    であるが、D profile が同一 binary 上に独立した lifecycle 会計違反（orphan）を
    6/6 決定論的に暴露した。I1 Final の判定対象が「A′ の効果」だけなら A/B/C PASS で
    満たされるが、I1 は R2 binary の shutdown 会計全体の closure 証明を含むため、
    orphan 修復（D162-2-I2）を入れた binary で D profile を再試験して初めて
    「全 profile で constructed = destroyed」が成立する。よって I1 Final は HOLD。
```

## 11. 成果物・検証手段

- 本監査は実コード直読（ISRShutdown.h/cpp・AudioEngine.h・PrepareToPlay.cpp・
  ReleaseResources.cpp・DSPLifetimeManager.cpp/h・RebuildDispatch.cpp・
  RuntimePublicationOrchestrator.cpp・PublicationExecutor.cpp・ISRDSPHandle.cpp/h・
  AudioEngineProcessor.cpp・AudioBlock.cpp・RuntimeTransition.h・Init.cpp）+
  Profile D 6 log 実測との突合で実施。production/test/CMake/build 変更 0。
- D log 側の整合確認: `retired=0` 行が DSPLifetimeManager.cpp:46 出力（map MISS early
  return）と一致・`[WORLD] Active=0` が §5 の unpublished session 結論と一致・
  CB silent clear が AudioBlock.cpp:306-311 と一致。
