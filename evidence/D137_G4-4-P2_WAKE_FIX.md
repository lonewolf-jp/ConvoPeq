# D137 — G-4.4-P2 Builder Wake / Liveness 修復 (Audit + Implementation Record)

**Date:** 2026-08-31 01:0x–01:3x (+09:00)
**Type:** read-only audit → 最小 wake 配線実装 → build/CTest。P3 対象（double representation）は未touch。
**基準:** `ConvoPeq.md`（実装前 `Generated: 2026-08-31 00:52:36` = P1 後ツリーと一致を確認して監査、実装後 `01:33:50` に再生成）
**Build evidence:** `evidence/g44p2_ctest.log` — Debug full build→CTest `100% tests passed out of 40`（DBG_CTEST_EXIT=0）/ Release 同（REL_CTEST_EXIT=0）
**Verdict: P2 実装完了・AC-P2-1..5 成立（下記証明）**

---

## 1. Read-only audit（実装前）— 欠陥確定

### 追跡対象と事実
- `redriveDeferredRecoveryObligations`（cpp:1110-1120）: Live ∧ delivery==None のみに `redriveDeferredRecovery` を適用。付着成功サイトは durable（cpp:1158-1170）と transport（cpp:1173-1178）の 2 つ。
- `runCoordinatorPhase`（Threading.cpp:258-325）: processIntent → **redrive（:270）** → deferred-publish watchdog（hasDeferredRequest 限定）→ overflow drain → retire drain。**redrive の後に Builder へ向かう wake なし**（D136-A 確定事項）。
- `rebuildThreadLoop` wait predicate（RebuildDispatch:854-859）: `hasPendingTask || publishRetryReady || recoveryPending || exit`。`recoveryPending` の set は submitRecoveryIntent（AudioEngine.h:4500-4504）のみ、clear は recovery 区間開始時（RebuildDispatch:972-975、rebuildMutex 下）。
- `markTransientFailure`（cpp:1066-1092）: Live 維持・delivery=None・counter+1・枯渇時のみ ResolvedFailed。**wake は発火しない**（CoordinatorLoop 専用、rebuildMutex 非取得）。
- `settlePendingRecoveryAdmission`（cpp:1237-1248）: Building→DurablePending / クリア。wake なし。
- `release()`/reservation: submitRecoveryIntent の `shutdownRuntime_.tryAdmit(1)/release(1)`（h:4494/4507）は submit 経路専用。redrive 付着は既存 admission の住処変更のみで新規予約を要さない（obligation は既に admitted・CoordinatorLoop は shutdown 時に停止 — ISRCoordinatorLoop:35）。
- Builder transport loop（RebuildDispatch:980-1041）/ durable loop（:1058-1124）: wake 後に全数をドレイン（pop/take ループ）。**flag は coalescing「仕事あり」意味** — 1 wake で複数表現を処理する。

### 4 遷移の証明（実装前 → 実装後）
```text
A. transport failure:  pop → markTransientFailure(None) → [P1] → redrive → 付着 → (P2 前: wake なし=FAIL / P2 後: latch→recoveryPending→notify=OK)
B. durable failure:    take → settle(true) → markTransientFailure(None) → redrive → transport 再付着（durable busy 同 oblId）→ (同上)
C. redrive→transport:  None→Transport（cpp:1175-76）→ P2: redriveWakePending_=true → 同一 tick consume → wake
D. redrive→durable:    None→Durable（cpp:1158-69）→ P2: 同上
```
実装前: A-D とも「付着は起きるが Builder は眠ったまま」— 消費は次の無関係 wake 依存（D136-A FAIL の再確認）。**欠陥確定 → 実装へ**。

## 2. Lost-wake proof（predicate state × signal linearization）

生産 triple: `rebuildMutex` + plain bool `recoveryPending`（mutex 下でのみ read/write）+ `rebuildCV.wait(lock, pred)` + `notify_all`。線形化点: producer の `recoveryPending=true` は lock(m) クリティカルセクション内（h:4501-4503 / Threading 新配線）、consumer の predicate 評価は同一 mutex 下。

- **redrive → notify → wait**: 着信時 Builder は未 wait。以後 wait 進入時に predicate が `recoveryPending==true` を読み即復帰。取りこぼしなし（state が信号を保持）。
- **redrive → wait → notify**: Builder は cv.wait 中。producer の set（mutex 下）→ notify_all → Builder 再起動、predicate true。取りこぼしなし。
- **wait → redrive → notify**: 上記と同型（wait 進入順序が逆なだけ）。
- **Builder が recoveryPending を clear（:974）した後の付着**: clear は recovery 区間開始時。付着が clear 後・pop/take ドレイン中に起きても、latch は同一 tick の consume で recoveryPending=true を再設定 → Builder は当該サイクルのドレインで拾うか、次の wait 進入時に predicate 真で即復帰。**notify の有無に関わらず state が保持**するため lost-wake 不成立。
- 重要: 本配線は「notify したか」でなく「predicate 状態 + mutex 下 set」に依存 — 既存 submitRecoveryIntent と同一の証明済みプロトコルを再利用しており、新しい順序論は導入しない。

## 3. 実装（最小・wake/liveness のみ）

| ファイル | 変更 | 内容 |
|---|---|---|
| `ISRRuntimePublicationCoordinator.h` | +13 | `consumeRedriveWake()`（public、[[nodiscard]]）+ private `bool redriveWakePending_ = false`（CoordinatorLoop 専用 — 同一スレッド set/consume、新規 cross-thread ordering なし） |
| `ISRRuntimePublicationCoordinator.cpp` | +2 | 付着成功 2 箇所で `redriveWakePending_ = true`（durable cpp:1170 / transport cpp:1177）— **None→X の実付着のみ**（both-busy・no-op・terminal スキップでは立たない） |
| `AudioEngine.Threading.cpp` | +16 | runCoordinatorPhase の redrive 直後: `consumeRedriveWake()` 真 → `{lock(rebuildMutex); recoveryPending=true;} rebuildCV.notify_all()` — submitRecoveryIntent（h:4500-4504）と同一プロトコル。processIntent 内の opportunistic redrive（capacity reject で admitted=false となり wake が飛ばされる経路）も同一 latch で拾われる |
| `ISRSemanticValidationTests.cpp` | +150 | T-P2-1..5（下記） |

禁止事項の遵守: RecoveryAdmission/durable slot 設計・take/settle・memory-order 契約・recoveryIntentQueue_・coalesce/supersession・RecoveryEpisodeId・capacity・retry 回数・markTransientFailure 意味・**P3 double representation**・retry scheduler・shutdown — 全て無変更（diff = 上表のみ）。

### AC 検証
- **AC-P2-1**: 付着（None→Transport/Durable）→ 同一 tick consume → recoveryPending 成立 → notify → Builder 復帰。コード証明 + T-P2-1（latch 実付着のみ発火）+ T-P2-3（wait 中 notify で復帰）。
- **AC-P2-2**: 3 順序すべて上記 lost-wake 証明 + T-P2-2（signal-before-wait）/ T-P2-3（signal-after-wait）/ T-P2-4（64 ラウンド混合順序・1 outstanding ハンドシェイク・全数観測）。
- **AC-P2-3**: redrive 1 回の付着は durable（return 後 transport 非実行）または transport の排他 1 本（cpp:1158-1182 構造不変）。latch は「付着があった」事実のみ伝え、住処の意味論を変えない — **P3 の二重住処窓は意図的に温存**（B 遷移の durable busy 同 oblId→transport 経路は従来どおり）。
- **AC-P2-4**: wake は実付着時のみ（毎 tick notify への逆戻りなし — F6-5 意図維持）。failure ループは obligation counter K=4 で ResolvedFailed 終端 → 以後 redrive 候補でなく latch 不発 → wake 源消滅。T-P2-5 が終端と「no wake source」を実証。idle 系は latch false のまま（ロック/notify ゼロ）。
- **AC-P2-5**: P1 の「恒久→次 wake まで遅延」が、P2 で「redrive 後、Builder が自律的に再実行」へ。残る唯一の非保証は付着**前**の deferred 観測（delivery=None のまま無関係 wake なし）だが、これは redrive 自体が毎 tick 実行されるため付着機会自体は保証済み — 遅延は「次 wake」でなく「次 tick 付着＋即 wake」に短縮。

## 4. テスト（T-P2-1..5）と開発中の発見

- **T-P2-1**（生産 coordinator）: 開始時 latch false → markTransientFailure 単体では false → redrive 付着で true → consume で消灯 → 2 回目 redrive（delivery!=None）は false。**実付着のみ・1 回のみ**を立証。
- **T-P2-2/3/4**（実スレッド）: 生産 triple を正確に模した `WakeProtocolHarness`（mutex + plain bool predicate + notify_all）で 3 順序 + 64 ラウンド混合順序を real contention で検証。単なる「notify した」確認でなく consumer の待機競合を含む。
  - **開発中の発見（要記録）**: 初版 T-P2-4 は「signal 数 = consumed 数」を期待する設計で、**coalescing フラグ意味論の誤モデル**によりハングした（複数 signal が 1 フラグに合一 → consumer が永久待機 → join ハング、CTest #21 で 9 分停止を実測・プロセス強制終了）。生産セマンティクス（1 wake で全ドレイン）に合わせ、**1 outstanding + progress ハンドシェイク**（awaitConsumed）へ修正 → 合格。この事象自体が「recoveryPending は状態信号でありイベントカウンタではない」ことの実証になった。
- **T-P2-5**: 3 周期の mark→redrive→wake（各 1 回）→ 4 回目 mark で ResolvedFailed（L=0・exhausted+1）→ 以後 redrive 不発・latch false。**K=4 停止と wake 源消滅**を実証。
- 既存 40 テスト（C1-C16/R13/R18/G43 含む）は両構成で変らず全合格（回帰なし）。

## 5. 残存（次 Gate へ — 本 P2 で意図的に未対応）
- P3: D136-B double representation（B 遷移の durable busy 同 oblId 窓 — 本実装は窓の挙動を一切変えない）。
- P4: D136-D memory-order 契約（latch は CoordinatorLoop 専用で新規エッジなし — 既存 defect は不変・未修正）。
- P5: Building 中 overwrite 契約。P6: T1/T2/T3 final regression（AudioEngineHarness レベルの実 Engine 統合 wake テストは依然未整備 — 本 P2 の実スレッドテストはプロトコル層）。

## 判定
**P2 完了（AC-P2-1..5 成立）。** STOP — P3 には進まない。
