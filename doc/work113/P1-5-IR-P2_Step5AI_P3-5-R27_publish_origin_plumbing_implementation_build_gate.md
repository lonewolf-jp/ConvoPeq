# P1-5-IR-P2 — Step 5-AI / P3-5-R27: Publish Intent Origin Plumbing Implementation / Build Gate

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R27）
- **判定**: **R27-A GO**。origin plumbing（R26 Option A）を bounded 範囲で実装し、
  D105 completion 再監査 PASS・INV-X5-1 維持・Release/OFF harness＋full build PASS・
  targeted run PASS。R28 へ進行可能。F/R 本測定・原因帰属は行わない。
- **種別**: production implementation＋test-only instrumentation＋build/run gate。

---

## 1. State Freeze（実装前・PASS）

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
  （CMakeCache.txt:232 CONVOPEQ_CORRECT_POLYPHASE_GAIN:BOOL=OFF 実測一致）
R16 take＋R19 B2＋R22 commit＋R10 accessor＋T6/T9＋R23 delta観測 保持
R24 design（Step5AF）＋R25-B finding（Step5AG）＋R26-A design（Step5AH）保持
ConvoPeq.md fresh（2026-09-23 21:32:56・5,521,528 B・最新src編集 20:40 より後）
coordinatorTakeCount_ 出現数 = 0（実装前・R25-B 時のまま）
```

### R26前提の6点再検証（実装前・すべて live source 一致）

```text
1. PublishPayload に既存 recoveryObligationId（Coordinator.h:732・{0} NSDMI）      → 一致
2. AudioEngine.h:4872 intent.payload.publish.recoveryObligationId = 0 固定          → 一致
3. trySubmitImpl は req.recoveryObligationId を保持（Commit.cpp:831 で req に設定） → 一致
4. trySubmitImpl は executor_.publish 成功後に Route-A completion（Orchestrator.cpp:321）→ 一致
5. executePublish → onPublishCommitted(seq, intent.payload.publish.recoveryObligationId)（RuntimePublishExecutor.h:114）→ 一致
6. 旧: Route-B intent は id=0 のため async completion は recovery を resolve しない     → 一致
```

- R26報告と相違なし。実装を続行した。

## 2. 実装（R26 Option A の bounded 5点・新 struct field なし）

### A1. `PublicationExecutor`（h＋cpp）

`publish`／`publishImpl` に `std::uint64_t recoveryObligationId = 0` を追加（default により
既存 caller 不変）。`publishImpl` は commit／fire-and-forget 両方へ中継する。

```cpp
PublishResult PublicationExecutor::publish(..., convo::isr::DSPHandle oldHandle,
                                           std::uint64_t recoveryObligationId) noexcept
{ return publishImpl(..., /*waitForReceipt=*/true, recoveryObligationId); }
```

### A2. `AudioEngine::commitRuntimePublication`（AudioEngine.h:4900）

`std::uint64_t recoveryObligationId = 0` を追加し、`enqueueRuntimePublicationFireAndForget`
へ中継。既存 caller（Timer.cpp:995／Transition.cpp:25／PrepareToPlay×2／ReleaseResources:223／
test 5箇所）は default で不変。

### A3. `AudioEngine::enqueueRuntimePublicationFireAndForget`（AudioEngine.h:4787）

`std::uint64_t recoveryObligationId = 0` を追加。固定値代入を搬送値へ置換：

```cpp
-  intent.payload.publish.recoveryObligationId = 0;   // 旧: Route B (non-recovery) ⇒ no obligation
+  intent.payload.publish.recoveryObligationId = recoveryObligationId;
```

### A4. `trySubmitImpl`（RuntimePublicationOrchestrator.cpp:281）から起点伝搬

```cpp
auto result = executor_.publish(engine_, std::move(frozen), req.newDSP, oldHandle,
                                req.recoveryObligationId);
```

- これで `trySubmitImpl → executor_.publish → publishImpl → commitRuntimePublication /
  enqueueRuntimePublicationFireAndForget → Intent.payload.publish.recoveryObligationId`
  の一本の伝搬が成立（R27 §3 の要求どおり）。

### A5. Coordinator counter（R26-A path A）

- **declaration 1**: `ISRRuntimePublicationCoordinator.h:974`
  `std::atomic<std::uint64_t> coordinatorTakeCount_{0};`（residency 直後・対称配置）
- **getter 1**: `ISRRuntimePublicationCoordinator.cpp:440`＋decl（h:189）＋
  AudioEngine passthrough（AudioEngine.h:2093）。
- **writer 1（唯一）**: `ISRRuntimePublicationCoordinator_ProcessIntent.cpp:61-62`

```cpp
case IntentType::Publish:
    convo::fetchSubAtomic(publicationIntentResidencyCount_, ...);   // 既存・不変
    if (commonIntent.payload.publish.recoveryObligationId == 0)
        convo::fetchAddAtomic(coordinatorTakeCount_, std::uint64_t{1}, std::memory_order_acq_rel);
    break;
```

- origin 2値（Main = obligationId==0／Recovery = !=0）。deferred は origin でない
  （resubmit は元 req を move-out するため起源保存・§5）。
- Decision／World の書換なし（read-only 参照のみ・HANDLER-1）。

### A6. Test-only（`P1PolyphaseGainCharacterization.cpp`）

`emitQDelta` に `coord` を追加（`cmt` と `drp` の間）。新 logger／prefix family なし。
新 counter なし（retry／defer／drop-reason／admission／per-task 一切なし）。

```text
req／que／dup／take／bld／cmt／coord／drp／seq／blo
```

## 3. D105 completion authority 再監査（R27 §6 の9 check・PASS）

```text
[1] resolveRecoveryObligation の Live→terminal full-word CAS が唯一の状態遷移 authority
    → PASS（本体無変更・Coordinator.h:508-531）
[2] async が先に winner でも Coordinator 側の後続 resolve は no-op
    → PASS（非 Live／id 不一致は false・:528 idempotent no-op）
[3] liveCount_ 二重減算なし
    → PASS（fetchSub は CAS winner のみ・:523）
[4] terminal state から別状態へ再遷移しない
    → PASS（CAS は w.state==Live のときのみ・terminal は不変）
[5] postRecoveryFailureSignal が成功 publish 後に terminal failure を生成しない
    → PASS（同 signal は publish 失敗時のみ・Orchestrator.cpp:312・成功経路になし）
[6] D152 T3 の discardedPending 分類に変化なし
    → PASS（resolve の *discardedPendingOut 無変更・:521-522）
[7] postRecoveryFailureSignal／retry／rearm 経路に意味変更なし
    → PASS（無変更。Retry は ΔL=0 維持・:1038-1039／:426-428）
[8] shutdown drain 中の二重 completion なし
    → PASS（ShutdownDiscarded 終端は CAS 単一 winner・:1060-1061）
[9] HANDLER-1: origin を「読むだけ」で Decision/World を書き換えない
    → PASS（ProcessIntent の writer は read-only 参照のみ）
```

### winner 反転の帰結（contract violation ではなく invariant 維持）

- `PublicationExecutor::publish` は常に `waitForReceipt=true`（publishImpl の唯一 caller）。
  実測: `publishImpl` の caller は `publish()` のみ、`publish()` の caller は
  `RuntimePublicationOrchestrator.cpp:281` のみ。したがって `waitForReceipt=false` 経路は
  現状存在しない（Orchestrator.cpp:277 のコメントは stale）。
- よって `commitRuntimePublication` は receipt まで待ち、receipt 配送元
  `onPublishCommitted`（RuntimePublishExecutor.h:114）内の resolve（Orchestrator.cpp:358）が
  先に走り、trySubmitImpl:321 の sync resolve は後着の no-op になる。
- R26 §4.3 の予測どおり **winner が async 側へ反転**するが、両 call site とも `won` 戻り値を
  無視（:321／:358 は文として呼ぶだけ）であり、終端状態は `ResolvedSuccess` で同一。
  CAS-based single completion invariant は維持される。
- Main origin は obligationId==0 のため async resolve は early-return no-op（:1033）—
  非 recovery 経路の挙動は完全不変。

## 4. INV-X5-1 の維持確認（PASS）

```text
publicationIntentResidencyCount_ の writer／reader／意味 = 不変。
R27 diff で residency に触れたのはコメント3行のみ（コード変更 0）。
coordinatorTakeCount_ は residency から導出しない（別 atomic・別意味: cumulative take）。
  residency = gauge（enqueue reservation + queue − pop）
  coord     = Main-origin pop の累積
```

## 5. Build Gate（PASS・harness＋full の二段）

```text
条件: Release／CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF／vcvarsall x64＋oneAPI setvars intel64
       generator=Ninja Multi-Config／compiler=cl

(a) authoritative harness target: cmake --build build --config Release --target AudioEngineHarness
    → BUILD_EXIT=0（[74/75] Linking CXX executable Release\AudioEngineHarness.exe・error 0 行）
(b) full target（R27 で追加実施）: cmake --build build --config Release
    → FULL_BUILD_EXIT=0（[241/242] Linking … 242 targets・error 0 行）
```

- **MKL include-path 問題は R27 では再発しなかった**（full target も 0 error）。
  R19/R22 で記録した pre-existing 条件は本環境では顕在化せず、R27 差分起因の
  defect ではないことを分離記録する（implementation defect 0／環境条件は非再現）。
- production compile（AudioEngine.h／PublicationExecutor／Coordinator 各 TU）、
  test compile（AudioEngineHarness＝P1 TU 含む）、origin plumbing compile、
  coordinator counter compile、getter compile、emitQDelta compile をいずれも通過。
- clangd の JUCE parse 不能由来の誤診（fetchAddAtomic／optional 等）は既存不具合であり、
  MSVC 実 build が正本（R10／R16／R19／R22 と同一結論）。

### Binary identity（G7）

```text
build/Release/AudioEngineHarness.exe → tmp/p35_R27.exe
SHA-256 425158012318dde394340b9118b23ffeedff22d15c617ea1a08b17743f7dbd55
mtime 2026-09-23 22:21:35／size 41,180,672／p15ir string 1（含有）
（R22 220902dab7677ee0…／R23 から意図どおり相違）
```

## 6. Run Gate（PASS・実装健全性のみ／F/R 測定ではない）

### 6a. default harness（publish pipeline＋deferred 回帰）

```text
build\Release\AudioEngineHarness.exe（引数なし）
→ PASS: "AudioEngineHarness: all publish pipeline tests PASS"
   runDeferredFlowIntegrationTests / runDeferredPublishViewStateMachineTests を含む
   （deferred 経路が新 plumbing 下で成立・origin 保存の実装健全性を確認）
```

### 6b. targeted origin 観測（`--p1-char`・R23 vehicle 1 回のみ）

```text
flag_macro=0（OFF build）／summary flag_macro=0 cases=1 failures=1

qdelta（6点）:
  pair1-sc0  pre:  req=2  que=2  dup=0 take=2  bld=2 cmt=2 coord=3 drp=0 seq=5 blo=0
             post: req=5  que=5  dup=0 take=5  bld=3 cmt=2 coord=3 drp=0 seq=5 blo=0
  pair2-sc0  pre:  req=6  que=6  dup=0 take=6  bld=4 cmt=3 coord=5 drp=0 seq=8 blo=0
             post: req=8  que=8  dup=0 take=7  bld=4 cmt=3 coord=5 drp=0 seq=8 blo=0
  pair2-sc1  pre:  req=10 que=10 dup=0 take=9  bld=7 cmt=5 coord=5 drp=0 seq=8 blo=0
             post: req=12 que=12 dup=0 take=10 bld=7 cmt=5 coord=5 drp=0 seq=8 blo=0
```

delta：

| window | reqΔ | queΔ | takeΔ | bldΔ | cmtΔ | **coordΔ** | drpΔ | seqΔ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pair1-sc0 | +3 | +3 | +3 | +1 | 0 | **0** | 0 | 0 |
| pair2-sc0 | +2 | +2 | +1 | 0 | 0 | **0** | 0 | 0 |
| pair2-sc1 | +2 | +2 | +1 | 0 | 0 | **0** | 0 | 0 |
| gap(post1→pre2) | +1 | +1 | +1 | +1 | +1 | **+2** | 0 | +3 |

不変条件の確認（R27 §10）：

```text
- origin 2値: coord は obligationId==0 のみ計数。Main/Recovery の分類は payload 由来。
- coord monotonic: 3 → 3 → 5 → 5 → 5 → 5（非減少・reset なし）。
- coord >= cmt: 3>=2, 3>=2, 5>=3, 5>=3, 5>=5, 5>=5（Main bucket ⊇ main-site rebuild）。
- residency: 観測していないが writer 不変（§4）。coord と独立。
- seq: 5 → 8（origin 非依存・publish 成功で従来どおり前進）。
- 形状保存: gainDb_sc0=sc1=-13.0276（R23 と同一・8回目の再現）／
  gen=8・seq=8／ab0_fade=0.0600／"publish not confirmed os=1 sc=0"／failures=1
  → plumbing は publish 挙動を変化させていない。
- crash／assert／exception 0（clean 終了・log 末尾まで正常）。
```

### Main bucket の既知混在（R26 §4.3 の実証）

```text
gap で cmtΔ=+1 に対し coordΔ=+2（差 +1）。
→ obligationId==0 の Main bucket は main-site rebuild 以外（idle／bootstrap publish）も含む。
  R26 の注意書きどおり「main-site counter だから Coordinator でも main-site だけ」は成立しない。
  cmt との対応は R28 の window-level 相関で扱う（R27 では原因推定しない）。
```

### Run Gate の限界（明示）

```text
本 sanity run は Recovery-origin（obligationId != 0）の publish を 1 件も発生させていない
（coord は Main bucket のみ・recovery publish なし）。したがって D105 の winner 反転は
実走では未検証であり、静的解析＋既存 unit（default harness の publish pipeline／deferred 全 PASS）
に依拠する。Recovery-origin 実走は R28 で確認する（R27 §11 の次 Step 送り）。
```

## 7. Source reconciliation（PASS）

```text
coordinatorTakeCount_ declaration = 1（Coordinator.h:974）
getCoordinatorTakeCount()        = decl 1（Coordinator.h:189）＋def 1（Coordinator.cpp:440）
                                    ＋AudioEngine passthrough 1（AudioEngine.h:2093）
writer（fetchAddAtomic）          = 1（ProcessIntent.cpp:62・Publish case 内・Main-only）
PublicationExecutor 伝搬         = publish→publishImpl→commit/fireAndForget（中継のみ）
commitRuntimePublication          = default 0 により既存 caller 不変
enqueuePublicationFireAndForget   = 固定値 0 を搬送値へ置換（唯一の代入点）
Intent／PublishPayload／OwnerChannel／deferredSlot_ = 変更 0（git grep 一致）
residency writer = 不変（コメントのみ）
working tree 差分 = 承認範囲のみ（production 5 file＋test 1 file・§2）
```

- writer 位置の順序： `intentQueue_.pop(Publish) → residency-- → coord++(Main のみ) →
  DispatchTable → executePublish(ownerChannel take → publish → seq bump)`。
  「ownership transfer 後・next Coordinator processing 前」を満たす。

## 8. R27 Gate

```text
R27-A GO： 全 PASS
[PASS] origin plumbing が R26 の bounded 範囲内（default 引数＋中継＋1 代入）
[PASS] Intent/OwnerChannel/deferred 構造を変更していない（差分 0）
[PASS] Main/Recovery の2値 origin が保持される
[PASS] deferred 前後で origin が保持される（default harness deferred PASS）
[PASS] coordinatorTakeCount_ は Main-only（recoveryObligationId==0 条件）
[PASS] writer は processIntent の 1 箇所
[PASS] residency semantics 不変（INV-X5-1）
[PASS] D105 single-completion invariant PASS（§3 の9 check・winner 反転は no-op 吸収）
[PASS] HANDLER-1 PASS（read-only 参照のみ）
[PASS] D152 T3 関連 PASS
[PASS] Release/OFF build PASS（harness 75 targets＋full 242 targets）
[PASS] targeted run PASS（default harness 全 PASS＋coord 観測・形状保存）
[PASS] working tree 差分が承認範囲内
R27-B／R27-C： 非該当（bounded 逸脱 0／D105 未解決矛盾 0／二重 resolve・liveCount 二重減算 0／
  deferred origin 消失 0／分類不能 0／writer 複数化 0／実装起因 build failure 0）。
```

- R28 への申送り： Recovery-origin 実走（D105 winner 反転の動的確認）／
  `cmt / coord / seq / drp` の origin-aware window 相関（C-1／C-2 observability closure）。
- F/R 本測定・C-1/C-2 分類・retry/defer/drop-reason/admission subtype・seq attribution・
  P3-1-D・buzz/limiter/NUC 帰属はすべて未実施（R27 §11 遵守）。
- source は R27 vehicle として保持する。revert なし。
- R4 境界・保留事項・P3-5 §7解釈制約・H-B 対象外を維持する。
