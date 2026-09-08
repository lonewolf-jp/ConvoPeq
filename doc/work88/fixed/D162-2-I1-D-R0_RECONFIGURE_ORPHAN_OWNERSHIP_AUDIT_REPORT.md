# D162-2-I1-D-R0 Reconfigure Orphan Ownership / Admission-State Audit Report

```text
Date:      2026-09-05
Type:      read-only root-cause audit report（production 変更 0 / test 0 / CMake 0 / build 0）
Evidence:  evidence/D162-2-I1/D_R0_RECONFIGURE_ORPHAN_OWNERSHIP_AUDIT.md（詳細版）
Input:     D162-2-I1-D Profile D 実測（D-2〜D-7・6/6 決定論的 orphan）
判定:      **GO（A 案最小単位確定）・I1 Final HOLD 継続**
```

---

## 1. 監査結論（10 項目応答）

### (1) tryAdmit() の CallerDestroy 契約

- `AudioEngine.h:3632-3641` に契約が明文化されている:
  **「CallerDestroy: publish 失敗、rollback 完了、呼び出し元が物理解放すべき（実際の破棄は
  DSPLifetimeManager::destroyRolledBackDSP() 経由）」**。
- つまり CallerDestroy は「呼び出し側が所有 DSP を破壊する」義務の通知であり、
  work70 Phase2 で「リーク防止」を目的に導入された。PublicationExecutor.cpp:68-71 も
  同一契約の二次文献。rebuild 経路（Orchestrator.cpp:291-292）はこの契約を履行済み。
- tryAdmit 失敗点（h:4613-4614）は **registration の前に返る**ため、この時点で DSPCore は
  map 未登録・registry 未登録・world 非参照（void* mirror のみ）であり、
  破壊義務は caller に 100% 残る。

### (2) aligned_unique_ptr::release() から失敗帰還までの ownership graph

```text
:241 aligned_make_unique → 所有者 = unique_ptr（1 のみ）
:264 setActiveRuntimeDSP(release()) → 所有者 = **なし**（slot は非所有 topology mirror）
:277 commitRuntimePublication(needsRegistration(未登録 DSP))
h:4613 tryAdmit → false（admission Closed）→ {Failed, CallerDestroy}
:280 ignoreUnused → 破壊義務が消滅
以後: registry も world も EBR も shutdown sweep も対象 DSP を識別できない
```

**「unique owner に戻る」でも「caller が raw pointer を destroy すべき」でもなく、
第三のケース「どちらにも戻らない」が確定** — ownership violation そのもの。
根本は `:264` の release()（所有権放棄が publish 成否確認より先行）と `:280` の
ignoreUnused（失敗応答の握り潰し）の組。

### (3) prepareToPlay の failure return 処理

`commitRuntimePublication` 全 5 caller のうち 4 caller（prepareToPlay ×2・Timer・
Transition）が `juce::ignoreUnused` で結果を廃棄。PublicationExecutor のみ stage 検査あり
（破壊は Orchestrator 側）。PrepareToPlay.cpp:217-220 の comment
「Runtime publication admission は lifecycle=Prepared を前提」が
**admission が常に open であるという誤前提**を残しており、設計上の盲点の直接証拠。

### (4) 未登録 DSP の terminal disposition site 全列挙

既存 disposition primitive 9 種を全走査した結果、**未登録 DSP を terminal disposition
できる site は存在しない**（詳細は evidence 版 §4 の表）:

| 不可（registration/handle 必須） | 可能だが今回の経路に未接続 |
| --- | --- |
| retire(void*) / retireByHandle / retireRegisteredDSP / S3 / V-D / pendingTask retire / ~AudioEngine handle authority | **destroyRolledBackDSP**（void* 直破壊・EBR 不要・cpp:149-157）、**DSPGuard**（RebuildDispatch.cpp:932-964・未登録確認 + destroyDSPCoreNode）、recovery warmup 失敗破壊（:1034-1038/1116-1120） |

h:4314-4315 に「未登録 DSPCore（handle map 不在・publish されない）→ DSPGuard 契約の
destroyDSPCoreNode 直接破壊（EBR epoch 保護不要）」という正規契約が明記されており、
**D162-2 の authority 原則（registered→EBR 一本化）と未登録直破壊は矛盾しない**。

### (5) shutdownRuntime_ Closed と device reconfigure の関係

- JUCE は同一 AudioProcessor に対し device open/close・sample rate・buffer 変更のたびに
  releaseResources + prepareToPlay を反復する（AudioEngineProcessor.cpp がそのまま委譲）。
- しかし admission FSM（ShutdownRuntime::packedState_）は **engine 1 生涯で 1 回限りの
  Open→Closing→Closed**（INV-LIFE-9 no-resurrection・reopen API 不存在）。
- prepareToPlay.cpp:72 が shutdownPhase（表示用 sequential state）を Running に戻すため
  「phase は Running・admission は Closed 永久」という不可視状態が成立し、
  reconfigure 後の publish が全て RejectedShutdown になる（実測 `[WORLD] Active=0`・
  session 全体 unpublished・bypass 継続）。
- 非対称: intent loop 側の CoordinatorState は ShuttingDown→**Bootstrapping 復帰あり**
  （markShutdownComplete・ISRRuntimePublicationCoordinator.cpp:563-574）である一方、
  admission FSM に復帰概念がない。**admission lifetime = engine lifetime という設計と
  JUCE re-prepare 契約の衝突が本件の構造的背景**。

### (6) ~AudioEngine が orphan を回収できないことの再確認

CtorDtor.cpp:132-214 のとおり、activeToRelease/fadingToRelease は観測用のみ
（E-2 で pointer-value retirement 廃止）、回収は handle authority 単独。orphan は
activate されたことがなく handle が存在しないため dtor の全経路が no-op。
graceful drain / quarantine sweep も intent queue・registry・store の登録物が対象で、
**未登録の裸 DSPCore* を列挙する手段が存在しない**。回収点は process exit のみ。

### (7) 修復候補 A/B/C/D 比較

| 案 | 要旨 | 評価 |
| --- | --- | --- |
| **A** | prepareToPlay が CallerDestroy を検査し destroyRolledBackDSP（未登録 → destroyDSPCoreNode 直破壊）+ slot クリア | **最小・契約履行型**。h:4314-4315 の正規契約内。RT 参照ゼロ（prepareToPlay 契約内）。diff 数行 |
| B | admission failure 時に authority が ownership を引き取る | 同効果だが 5 caller 共通口に破壊責任分岐を埋め込み ScopeExit 契約と競合。contract 拡散が大 |
| C | shutdownRuntime_ 再 Open（reconfigure-aware reopen） | **INV-LIFE-9（D101-30 locked contract・G-H race closure・Q7 NoResurrection proof）の反転**。ISRLifetimeProof 層の再設計級。監査面積が桁違い |
| D | prepareToPlay が admission Closed を検知して placeholder publish 自体を skip | orphan は消えるが「reconfigure 後 engine が生きない」設計矛盾の隠蔽に留まる。shutdownRuntime_ への新規依存を生む |

### (8) 最小修復単位の決定は可能か

**可能**。最小単位 = **PrepareToPlay.cpp:277（pubResult2・idle #3）のみに A 案を適用**。

- 他 3 caller（pubResult1/Timer/Transition）は既に登録済み DSP を needsRegistration で渡す
  ため、失敗時の rollback CAS（Constructing→Reclaimed）が Active 状態で失敗し
  **登録が温存 = leak が原理的に発生しない**（ISRDSPHandle.cpp:157-173 の CAS 条件が
  保証）。rebuild 経路は Commit.cpp:804 事前登録 + Orchestrator:291-292 destroyRolledBackDSP
  で自己完結。
- よって leak する唯一の呼び出しは :277。1 関数内・数行 diff で閉じる。
- I2 設計時の裁量事項: :155 への同型 check 追加（契約履行の一貫性・leak はしないので必須
  ではない）。

### (9) GO / NO-GO

**GO（条件付き）** — D162-2-I2（reconfigure orphan repair・A 案最小単位）の実装可否判断へ
進めてよい。条件:

1. I2 scope を「ownership repair（A）」に限定し、publication 復活（C/D 案級）を
   別 track として切り離すこと。
2. I2 完了後の検証: Debug/Release CTest・D profile 再試験（orphan 0・
   constructed = destroyed 6/6）・E-4 会計・D123 zone 3 行の再確認。
3. C/D（publication 復活・admission reopen 設計）は別監査（G-series 相当）を要する
   ため本 R0 では判断しない。

### (10) I1 Final を HOLD する理由

```text
I1-A/B/C = PASS / I1-D = STOP（lifecycle gate 6/6 FAIL・crash 系 12 gate 全 PASS）
A′ deferred terminal disposition: proven effective（B 6/6 retired=1・C/D no-op 冪等）
New independent finding（R0 確定）: reconfigure → admission Closed → 未登録 DSPCore
    ~108MB が terminal disposition なしで process-exit reclamation

I1 Final HOLD:
    I1 は「A′ 効果の運用検証」に加え「R2 binary の shutdown 会計全体の closure 証明」
    を含む。orphan は同一 binary 上の独立した lifecycle 会計違反であり、
    修復 binary（I2）で D profile を再試験して constructed = destroyed 6/6 が成立して
    初めて I1 の会計 closure 証明が完遂する。よって HOLD。
```

## 2. 検証の基礎（実測との突合）

| R0 の確定 | D log 実測との整合 |
| --- | --- |
| tryAdmit 失敗 → 未登録 | `retired=0`（map MISS early return・DSPLifetimeManager.cpp:46）6/6 |
| session unpublished | `[WORLD] Active=0`・`dspReady=0`・procTimeUs 3-5μs 6/6 |
| V-D 不発（handle null） | `[D162-2G2_VD_RETIRE]` shutdown 時出力なし 6/6 |
| dtor handle MISS | `RETIRE_BY_HANDLE handle=254 lookup=MISS` 6/6 |
| bootstrap DSP は正常閉包 | VD_RETIRE retired=1 → epoch=9 → DESTROY → remaining=0 6/6 |

production/test/CMake/build 変更: **0**（本監査は read-only）。
