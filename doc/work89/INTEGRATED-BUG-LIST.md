# ConvoPeq 統合バグリスト (work89)

- **作成日**: 2026-08-08
- **対象**: `doc/work89/bugs/` 内の全バグファイル（BUG-047 〜 BUG-065、重複1ファイル含む）
- **検証方法**: 各バグの報告内容（該当ファイル・行番号・症状・コード断片）を **現在のソースコード** と 1:1 で照合。ソース上の `★ BUG-XXX` 修正コメントと実際のコード変更を確認。
- **検証日時点のソース**: 主要対象ファイルは 2026-07-30 〜 2026-08-05 に更新済み（バグ発見日 2026-07-26 より後）
- **再検証（2026-08-12）**: HEAD 3198acc にて別視点調査（§15）を実施。全 19 バグの「修正済み」判定を再確認し、一部の不正確な記述（§3 BUG-048/049/054、§5、§8.1、§14）を修正した。**行番号は §15.2 の表を正とする。**
- **再検証第2パス（2026-08-12）**: serena / AiDex MCP を stdio JSON-RPC で実使用（§16）、類推探索・文献検証を実施。§15 の結論に変更なし。N-1（Emergency Override の exchange）は work88 BUG-029 の意図的設計であることを確定し、CAS-only 化の推奨を撤回。未完了事項はゼロに確定。

---

## 1. サマリ

| カテゴリ | 件数 | 内訳 |
|---------|------|------|
| バグ報告ファイル | 20 | BUG-047 〜 BUG-065（うち1件は BUG-047 の重複ファイル） |
| ユニークバグ | **19** | BUG-047(重複除く), 048〜065 |
| 修正済み（ソースで確認） | **19 / 19** | 全バグに修正コード + `★ BUG-XXX` コメントが残存 |
| 未修正（ソースで確認） | 0 | — |
| 残存リスク（注意事項） | 1 | BUG-065 の一部（`rtSeenAgcResetSerial` 直接書き込み） |

> **最重要所見**: バグレポート（2026-07-26 作成）後に全 19 バグが修正され、各修正箇所に `★ BUG-XXX` コメントが残されている（※ BUG-060/063/064/065 は ★ なし表記、N-9 参照）。README 記載の「work88 全バグ (BUG-011〜BUG-046) は修正されていない」という記述は work89 バグには適用されない。**現行コードに未修正バグはゼロ**。

---

## 2. バグ一覧（修正状態つき）

| ID | タイトル | 報告時重大度 | 修正状態 | 修正内容（ソース確認） |
|----|---------|------------|---------|----------------------|
| **BUG-047** | EQCoeffCache::computeParamsHash に sampleRate 不足 | **HIGH** | ✅ 修正済み | `computeParamsHash(params, sampleRate, maxBlockSize)` に signature 拡張＋ハッシュへ `srBits`/`maxBlockSize` を組込（EQProcessor.ProcessingCache.cpp:23-50【★ :46】、EQProcessor.h:242-244）。`getOrCreate` 側も新 signature で呼出（AudioEngine.Cache.cpp:51） |
| **BUG-048** | detectStuckReaders が最初の一致で break し重度 Stuck を見逃す | LOW | ✅ 修正済み | 単一ループ break-on-first を **3パス評価**（Pass3 Chronic → Pass2 Warning → Pass1 EpochGap）に変更（EpochDomain.h:440-509）。加えて residencyTime を実時間ベースに改善（P4.5） |
| **BUG-049** | quarantineFlags への並行 store 競合（即時 vs 遅延隔離） | MEDIUM | ✅ 修正済み | plain store を **CAS** に変更（0x00→0x02 / 0x00→0x01 / 0x02→0x01 の原子的遷移のみ許容、0x03 生成防止）（EpochDomain.h:255-285 ほか） |
| **BUG-050** | enterReader の HB 順序（epoch store が depth++ の後） | MEDIUM | ✅ 修正済み | **epoch を depth++ より前に store** して HB ギャップを除去（EpochDomain.h:107-115）。`enterReader` 自体は `[[deprecated]]`（RCUReader 移行推奨） |
| **BUG-051** | exchangeFadingRuntimeDSP sentinel mask デッドコード | LOW | ✅ 修正済み | sentinel `(uintptr_t)-1` チェックを全 3 箇所で削除。`activeToRelease = getActiveRuntimeDSP()` 等に簡素化（AudioEngine.CtorDtor.cpp:127-134、ReleaseResources.cpp:136/159）。fading slot は CAS-based `claimFadingRuntimeDSP` に刷新（DSPTransition.h:113-128） |
| **BUG-052** | consumeDeferredRequest の non-atomic アクセス | LOW | ✅ 修正済み | `consumeDeferredRequest()` を廃止し **`DeferredPublishView`（move-only View + owner_->finishView() 委譲）** パターンに再設計（design-D4 / ADR-C4）。`hasDeferredRequest()` は `consumeAtomic(hasDeferred_)` に変更（RuntimePublicationOrchestrator.h:52-109, 158-167） |
| **BUG-053** | stopNoiseShaperLearning の stopLearning() 二重呼び出し | MEDIUM | ✅ 修正済み | 直接 `noiseShaperLearner->stopLearning()` 呼び出しを **削除**し、キュー経由（enqueueLearningCommand → processDeferredLearningActions）に一本化（AudioEngine.Learning.cpp:55-63） |
| **BUG-054** | onPublishCompleted の crossfade handle 不一致 | MEDIUM | ✅ 修正済み | `getActiveRuntimeDSPHandle()` をやめ、**enqueue 時に resolve 済みの真の oldHandle を引数で受渡し**。内側変数は `fadingHandle` に改名（DSPTransition.h:89-114）。fading slot は CAS-only `claimFadingRuntimeDSP` に刷新 |
| **BUG-055** | runPublicationPrecheckNonRt の到達不能 else if | LOW | ✅ 修正済み | 到達不能 `else if (fadingRuntimeUuid != 0)` ブロックを **削除**（AudioEngine.Commit.cpp:221-225） |
| **BUG-056** | checkCrossfadeTimeout/Drop の MonitorState 永久 Error 貼り付き | MEDIUM | ✅ 修正済み | 両関数に **Normal 復帰パス**を追加。`isPending()==false` 時と `delta < warning` 時に `emitOnTransition(..., Normal, ...)`（RuntimeHealthMonitor.cpp:534-536, 562-565）。`reset()` でも Normal 化（:1217） |
| **BUG-057** | checkOverflowRate の eventCode に RETIRE_STALL 誤用 | LOW | ✅ 修正済み | **専用イベントコード `EVENT_OVERFLOW_RATE_WARNING=1012` / `EVENT_OVERFLOW_RATE_CRITICAL=1013`** を追加し置換（RuntimeHealthMonitor.h:60-61） |
| **BUG-058** | checkWorldConsistency が m_prevConfigDivergenceState_ を誤流用 | MEDIUM | ✅ 修正済み | **専用 state `m_prevWorldConsistencyState_`** と**専用イベントコード 7000/7001/7002** を追加。Consistent/Suspicious/Broken の 3 状態遷移を実装（RuntimeHealthMonitor.h:63, 372、.cpp:950-985） |
| **BUG-059** | reset() の MonitorState リセット不整合 | MEDIUM | ✅ 修正済み | **全 MonitorState を Normal に統一リセット**（クリーンスタート）。従来「維持」だった retire/publication/overflow/retireAge 等も全て Normal 化（RuntimeHealthMonitor.cpp:1209-1245）。実在異常は次回 tick で再検出 |
| **BUG-060** | quarantineResidentCount の TOCTOU → unsigned underflow | MEDIUM | ✅ 修正済み | check-then-act をやめ **fetchSubAtomic 単一アトミック + previous==0 時は fetchAdd で回復**（ISRRetireRuntimeEx.cpp:215-224）。quarantine 側も `previousLane != Quarantine` 時のみ fetchAdd（:236-237） |
| **BUG-061** | DSPQuarantineManager の std::vector 非保護同時アクセス | MEDIUM | ✅ 修正済み | **`std::mutex` による全アクセス保護**（push_back / getEntry / getMaxEntryAgeSec / reclaimSlot / destroyForShutdown / compactAuditLog / public API）（ISRDSPQuarantine.cpp:35, 84, 118, 157-176、.h:64, 83）。ロック保持済み internal 版と public 版の二重ロック防止 |
| **BUG-062** | checkRetireReclaimLatency uint64_t 版に正常復帰イベント欠如 | MEDIUM | ✅ 修正済み | uint64_t 版にも **`EVENT_RETIRE_AGE_NORMAL` 復帰パスを追加**（double 版と整合）（RuntimeHealthMonitor.cpp:897-901） |
| **BUG-063** | EpochDomain ownerTag 非同期読み書き (data race) | LOW | ✅ 修正済み | `ownerThreadId` (`std::atomic<uint64_t>`) を追加し、**ownerThreadId != 0 の時のみ ownerTag をコピー**（EpochDomain.h:69-71, 490-495, 538-539）。register 側は CAS 排他下で設定 |
| **BUG-064** | Float/Double 出力パスで applyFixedLatencyDelay と clamp の順序不一致 | MEDIUM | ✅ 修正済み | float パス `processOutput` を **clamp → delay の順序**（double パスと統一）に変更。delay buffer に clamp 済み値のみ格納されることを保証（DSPCoreIO.cpp:521-530） |
| **BUG-065** | EQProcessor::reset() が AGC リセットを発火しない（serial を 0 に設定） | MEDIUM | ⚠️ **修正済み＋1点残存リスク** | `publishAtomic(agcResetSerial, 0)` を **`fetchAddAtomic(agcResetSerial, 1)`（increment）** に変更（EQProcessor.Core.cpp:276, 787 = reset()/prepareToPlay()）。→ **残存**: `rtSeenAgcResetSerial = 0` の直接書き込み（:279）は残っている |

---

## 3. バグ別詳細（報告内容 vs 検証結果）

### BUG-047 — EQCoeffCache ハッシュに sampleRate 不足 【HIGH → 修正済み】

- **報告**: `computeParamsHash(eqParams)` に sampleRate/maxBlockSize が含まれず、サンプルレート変更後に 48kHz 用係数が 96kHz 処理に使われる（カットオフ約 2 倍シフト）。
- **検証（修正後）**:
  - `EQProcessor.h:242-244` — signature が `computeParamsHash(const EQParameters&, double sampleRate, int maxBlockSize)` に拡張。
  - `EQProcessor.ProcessingCache.cpp:24-49` — ハッシュ計算末尾に `★ BUG-047` コメント付きで `srBits`（sampleRate のビット列）と `maxBlockSize` を `hashCombine`。
  - `AudioEngine.Cache.cpp:51-53` — `getOrCreate` 側も新 signature `computeParamsHash(params, sampleRate, maxBlockSize)` で呼出。
- **判定**: 修正完了。sampleRate 変更後は旧係数キャッシュがヒットしなくなり、正しい係数が再生成される。修正案 A に相当。

### BUG-048 — detectStuckReaders 最初の一致で break 【LOW → 修正済み】

- **報告**: 条件優先順位を実装しているが最初の一致で break し、重度 Chronic Stuck を見逃す。
- **検証（修正後）**: `EpochDomain.h:434-512`（detectStuckReaders）に `★ BUG-048` コメント（:448）。`for (int pass = 3; pass >= 1; --pass)` の 3 パス構造（Pass3=Chronic、Pass2=Warning、Pass1=EpochGap）に変更。各パス内で `buildReaderInfo(i, severity)` を走査し、最初に合致した Reader を返す。**※ 2026-08-12 再検証: Pass3/Pass2 の合致条件には `info.pendingRetireCount > 0` の AND 条件が含まれる**（:461/:466。pendingRetire==0 の間は Pass1（EpochGap）のみが検出可能）。
- **判定**: 修正完了（修正案 A 相当）。さらに residency を steady_clock 実時間ベースに改善（P4.5）。

### BUG-049 — quarantineFlags 並行 store 競合 【MEDIUM → 修正済み】

- **報告**: 2 Coordinator が 0x02(pending) と 0x01(quarantined) を競合 store → pending & depth==0 の不変条件違反、Debug assert 発火。
- **検証（修正後）**: `EpochDomain.h:255-316`（quarantineReader）に `★ BUG-049` コメント（:267、exitReader 内の昇格 CAS は :169）。即座隔離は CAS（`expected = kPendingQuarantineFlag` または 0 からの昇格）、depth>0 時は **CAS で 0x00→0x02 のみ設定**（:308-315。OR ではない — 既に quarantined(0x01) なら何もしない）。`0x00→0x02 / 0x00→0x01 / 0x02→0x01` のみ許容し 0x03 を生成しない。
- **判定**: 修正完了（修正案 A 相当）。

### BUG-050 — enterReader の HB 順序 【MEDIUM → 修正済み】

- **報告**: epoch store が depth++ の後 → getMinReaderEpoch が depth>0 を観測しても stale epoch（kInactiveEpoch 等）を読む可能性。UAF には至らないが設計上の HB ギャップ。
- **検証（修正後）**: `EpochDomain.h:107-115` に `★ BUG-050` コメント。**epoch を先に store（release）→ その後 depth++（acq_rel）** の順序に変更。nested 時は「epoch は active Reader を反映済みで安全」として戻さない。
- **判定**: 修正完了（修正案 A 相当）。なお `enterReader` は `[[deprecated]]` 付き（RCUReader 移行推奨）。

### BUG-051 — sentinel mask デッドコード 【LOW → 修正済み】

- **報告**: `exchangeFadingRuntimeDSP` 戻り値の `(uintptr_t)-1` チェックが常に false のデッドコード（3 箇所）。
- **検証（修正後）**:
  - `AudioEngine.CtorDtor.cpp:127-134` — `★ BUG-051` コメント付きで `activeToRelease = getActiveRuntimeDSP();` に簡素化（再解釈チェック除去）。
  - `ReleaseResources.cpp:136, 159` — 同様に簡素化。fading slot は CAS-based `claimFadingRuntimeDSP` に刷新（DSPTransition.h）。
- **判定**: 修正完了。

### BUG-052 — consumeDeferredRequest の non-atomic アクセス 【LOW → 修正済み】

- **報告**: `consumeDeferredRequest()` / `hasDeferred_` / `deferredSlot_` が plain アクセス。現状デッドコードだが将来リスク。
- **検証（修正後）**: `RuntimePublicationOrchestrator.h` が大幅再設計。`consumeDeferredRequest` は **`DeferredPublishView`（move-only、consume()/discard() が owner_->finishView() へ所有権解放を委譲）** に置換。`peekDeferred()`（Single Thread Owner 契約）＋ `hasDeferredRequest()` は `consumeAtomic(hasDeferred_)`。design-D4 / ADR-C4 に基づく Phase-1 設計。
- **判定**: 修正完了（設計再構築により解消）。

### BUG-053 — stopLearning() 二重呼び出し 【MEDIUM → 修正済み】

- **報告**: enqueue(Stop) ＋ 直接 `stopLearning()` で 2 回呼び出し。NoiseShaperLearner の実装次第でクラッシュ/UB。
- **検証（修正後）**: `AudioEngine.Learning.cpp:55-63` に `★ BUG-053` コメント。直接呼び出し行は削除済み（コメント内に「`noiseShaperLearner->stopLearning(); ← 削除`」と明記）。Timer が毎回 processLearningCommands → processDeferredLearningActions を drain するため停止は確実に到達。
- **判定**: 修正完了（修正案 A 相当）。

### BUG-054 — onPublishCompleted の crossfade handle 不一致 【MEDIUM → 修正済み】

- **報告**: `getActiveRuntimeDSPHandle()` が activate 後は newDSP を返し、crossfade が newDSP→newDSP として登録される。
- **検証（修正後）**: `DSPTransition.h:89-114` に `★ BUG-054` コメント。**enqueue 時に resolve 済みの真の oldHandle を引数で受渡し**、`registerDSPHandleForRuntime(newDSP)` のみで newHandle を取得。内側変数は `fadingHandle` に改名。fading slot は CAS-only `claimFadingRuntimeDSP(oldDSP)` に刷新。
  - **※ 2026-08-12 再検証で訂正（「exchange 廃止」は不正確）**: `exchangeFadingRuntimeDSP()` は **AudioEngine.h:2092 に残存**し、**Emergency Override パス（DSPTransition.h:63-65）で使用継続**（`// ★ Temporary: exchangeFadingRuntimeDSP (A-6 fix). Will be removed after B-1 CAS-only claimFadingRuntimeDSP().` コメント付き）。通常 crossfade / Timer / shutdown の fading slot クリアは全て CAS-based に刷新済み（Timer.cpp:1018 / DSPTransition.h:139 / CtorDtor.cpp:136 / ReleaseResources.cpp:140）。→ 未完了事項 N-1 として §15.3 に記録。
- **判定**: 修正完了（修正案 A 相当）。

### BUG-055 — 到達不能な else if 【LOW → 修正済み】

- **報告**: `else if (fadingRuntimeUuid != 0)` が常に false のデッドコード。
- **検証（修正後）**: `AudioEngine.Commit.cpp:221-225` に `★ BUG-055` コメント。該当 else if ブロックは削除済み。後続の `hasTransitionNext` 導出（`fadingRuntimeUuid != 0`）は残存（正常）。
- **判定**: 修正完了。

### BUG-056 — crossfade MonitorState 永久 Error 貼り付き 【MEDIUM → 修正済み】

- **報告**: `checkCrossfadeTimeout` / `checkCrossfadeEventDrop` に Normal 復帰パスがなく、一度 Error になると PolicyEngine が Recover を発行し続ける。
- **検証（修正後）**: `RuntimeHealthMonitor.cpp:534-536`（`isPending()==false` → Normal 復帰）と `:562-565`（`delta < warning` → Normal 復帰）に `★ BUG-056` コメント付きで追加。`reset()` でも `m_prevCrossfadeDropState = Normal`（:1217）。
- **判定**: 修正完了（修正案どおり）。

### BUG-057 — overflow rate の eventCode 誤用 【LOW → 修正済み】

- **報告**: `checkOverflowRate()` が `EVENT_RETIRE_STALL`(1001)/`EVENT_RETIRE_STALL_WARNING`(1002) を誤用。
- **検証（修正後）**: `RuntimeHealthMonitor.h:60-61` に `★ BUG-057` コメント付きで **`EVENT_OVERFLOW_RATE_WARNING=1012` / `EVENT_OVERFLOW_RATE_CRITICAL=1013`** を新設。
- **判定**: 修正完了。

### BUG-058 — checkWorldConsistency の state 流用 【MEDIUM → 修正済み】

- **報告**: `m_prevConfigDivergenceState_` を流用 → Config Divergence 状態の上書き / World Consistency 初回検出スキップ。
- **検証（修正後）**: `RuntimeHealthMonitor.h:63`（専用イベントコード 7000/7001/7002）と `:372`（`m_prevWorldConsistencyState_`）を新設。`checkWorldConsistency()`（.cpp:950-985）は Consistent/Suspicious/Broken の 3 状態を専用 state で遷移。
- **判定**: 修正完了（修正案どおり）。

### BUG-059 — reset() の MonitorState リセット不整合 【MEDIUM → 修正済み】

- **報告**: 一部のみ Normal 化され不整合（Config Divergence のみ早期再通知など）。
- **検証（修正後）**: `RuntimeHealthMonitor.cpp:1209-1245` に `★ BUG-059` コメント。**全 MonitorState を Normal に統一リセット**（retire/publication/crossfade/readerSlot/overflowRate/retireAge/configDivergence/snapshotStarvation/structuralDeploy/suppression/worldConsistency/progressFreeze/configDrift/learnerBackpressure）。実在異常は次回 tick で再検出されるため維持不要という設計判断。
- **判定**: 修正完了（修正案 B 相当）。

### BUG-060 — quarantineResidentCount TOCTOU underflow 【MEDIUM → 修正済み】

- **報告**: `consumeAtomic(resident)` + `if (resident > 0)` の間に別スレッドが fetchSub → UINT64_MAX ラップ。
- **検証（修正後）**: `ISRRetireRuntimeEx.cpp:215-224` に `★ BUG-060` コメント（「TOCTOU 排除 — check-then-act ではなく fetchSub 単一アトミック + 回復」）。`fetchSubAtomic(...,1)` の戻り値が 0 だった場合 `fetchAddAtomic(...,1)` で回復。quarantine 側（:236-237）も `previousLane != Quarantine` 時のみ fetchAdd で二重加算防止。
- **判定**: 修正完了（修正案どおり）。

### BUG-061 — DSPQuarantineManager の vector データ競合 【MEDIUM → 修正済み】

- **報告**: `auditLog_` (std::vector) への erase（compactAuditLog）と reverse iteration（getEntry）の同時実行で UB。
- **検証（修正後）**: `ISRDSPQuarantine.cpp:35, 84, 118, 157-176` と `.h:64, 83` に `★ BUG-061` コメント。**`std::mutex` による全アクセス保護**。ロック保持済み internal 版（`compactAuditLogLocked` 等）と public API（二重ロック防止）を分離。
- **判定**: 修正完了（mutex 案どおり）。

### BUG-062 — uint64_t 版に正常復帰イベント欠如 【MEDIUM → 修正済み】

- **報告**: `m_maxRetireAgeRef`（uint64_t）パスで retire age 回復時に `EVENT_RETIRE_AGE_NORMAL` が出ず、state が Warning/Error に貼り付く。
- **検証（修正後）**: `RuntimeHealthMonitor.cpp:897-901` に `★ BUG-062` コメント。uint64_t 版の else 節に `emitOnTransition(..., Normal, ..., EVENT_RETIRE_AGE_NORMAL, ...)` を追加（double 版と整合）。
- **判定**: 修正完了。

### BUG-063 — ownerTag data race 【LOW → 修正済み】

- **報告**: `registerReaderThread`（書込）と `detectStuckReaders`（読取）で非 atomic `ownerTag[32]` が競合（stale read 許容とコメント済みだが UB）。
- **検証（修正後）**: `EpochDomain.h:69-71` — register 時に `ownerThreadId` (`std::atomic<uint64_t>`) を release で publish。`:490-495` — 検出側は `ownerThreadId` を acquire で読み、**非 0 の時のみ ownerTag をコピー**。:538-539 に `std::atomic<uint64_t> ownerThreadId` 宣言。
- **判定**: 修正完了（修正案 B 相当 — ownerThreadId で保護）。

### BUG-064 — Float/Double 出力パスの clamp/delay 順序不一致 【MEDIUM → 修正済み】

- **報告**: float パスが「delay → clamp」、double パスが「clamp → delay」で、delay buffer に unclamped 値が入り得る。
- **検証（修正後）**: `DSPCoreIO.cpp:521-530` に `★ BUG-064` コメント。float パス `processOutput` を **clamp（jlimit）→ applyFixedLatencyDelay** の順序に変更し、double パス（DSPCoreDouble.cpp:739）と統一。delay buffer には clamp 済み値のみ格納。
- **判定**: 修正完了（修正案「clamp-then-delay 側に統一（推奨案）」どおり）。

### BUG-065 — reset() が AGC リセットを発火しない 【MEDIUM → 修正済み（1点残存）】

- **報告**: `publishAtomic(agcResetSerial, 0)` ＋ `rtSeenAgcResetSerial = 0` 直接書き込みで、Audio Thread の比較 `agcResetSerialNow != rtSeenAgcResetSerial` が false になり AGC リセットが発火しない。加えて Message Thread からの `rtSeenAgcResetSerial` 書き込みは data race。
- **検証（修正後）**:
  - `EQProcessor.Core.cpp:276`（reset()）と `:787`（prepareToPlay()）— 共に **`fetchAddAtomic(agcResetSerial, 1)`（increment）** に変更（コメント `// increment (not set to 0): so RT agcResetSerialNow != rtSeenAgcResetSerial triggers AGC reset (BUG-065)`）。
  - `:277-279` — `rtDeferredBandResetMask.store(0, relaxed)` / `rtSeenBandResetSerial = 0` / **`rtSeenAgcResetSerial = 0` は残存**。
- **判定**: **機能修正は完了**（serial が必ず変化するため Audio Thread はリセットを検知する）。
  - **残存リスク ①（data race）**: `rtSeenAgcResetSerial = 0` の直接書き込み（:279）は非 atomic 変数への Message Thread 書き込みであり、Audio Thread 実行中の `process()` と競合し得る（C++ 標準上は UB）。ただし `reset()` は prepareToPlay 停止パス等で呼ばれることが多く、Audio Thread 停止中なら実害なし。安全化するなら `:279` を削除する（increment 方式なら Audio Thread が検知して自身で rtSeen を更新するため不要）。
  - **残存リスク ②（意味論）**: reset() 直後の最初の process で 1 回 AGC リセットが走るのは意図どおりだが、`rtSeenAgcResetSerial=0` が Audio Thread の検知**後**に書き込まれると 2 回目リセットが起こり得る（前後関係に依存）。

---

## 4. 重複ファイルの指摘

- `BUG-047.md` と `BUG-047-EQCoeffCache-ハッシュにsampleRate不足.md` は **同一バグの重複**（後者は内容が短縮版）。実質バグ数は 19。
- 管理上、`BUG-047-EQCoeffCache-ハッシュにsampleRate不足.md` は削除または BUG-047 への相互リンク化を推奨。

## 5. 未確認事項・制約

- 各修正の**動作検証（テスト/ビルド）**は本調査の範囲外。修正コメントとコード差分で「修正実装の存在」を確認した。
- work88 バグ（BUG-011〜BUG-046）は本リストの対象外（README 参照）。
- 修正の入ったコミットは 2026-07-30 〜 2026-08-05 のコミット群に含まれる（`★ BUG-XXX` コメントは現行 HEAD に残存）。**※ 2026-08-12 再検証で特定: 全 19 バグの修正＋★ コメントは単一コミット `444c2f3`（2026-08-05）で一括導入**（`git log -S 'BUG-0XX' -- src/` で全数確認）。
- **※ 2026-08-12 追記（ビルド検証）**: `tools/build-check.log`（2026-08-11 16:15）で **Debug 構成 166/166 ターゲットのビルド成功**を確認（AudioEngineHarness.exe・各テスト exe 含む）。HEAD（3198acc）の src/ はビルド時点（c8ca439）から変更なしのため、現行ソースはコンパイル検証済み。

## 6. 結論

1. **doc/work89/bugs/ の全 19 ユニークバグは、現行ソース上で全て修正済み**（各修正箇所に `★ BUG-XXX` コメントが残存、2026-08-08 時点の HEAD で確認）。※ 表記注: BUG-060/063/064/065 の 4 件はコメントが「BUG-XXX」（★ なし）表記（N-9）。修正実装の有無に影響なし。
2. 修正パターンの内訳:
   - **設計再構築（2）**: BUG-052（DeferredPublishView 化）、BUG-051（CAS ベース fading slot 化と併せて刷新）
   - **原子性修正（3）**: BUG-049（CAS）、BUG-060（fetchSub+回復）、BUG-065（fetchAdd increment）
   - **メモリ順序修正（2）**: BUG-050（epoch 先行 store）、BUG-063（ownerThreadId ガード）
   - **状態遷移補完（4）**: BUG-056、BUG-058、BUG-059、BUG-062（Normal 復帰/専用 state/統一リセット）
   - **ロジック修正（8）**: BUG-047、048、053、054、055、057、061、064
3. **唯一の残存リスク**は BUG-065 の `rtSeenAgcResetSerial = 0` 直接書き込み（data race の可能性）。機能上の AGC リセットは動作するが、完全に解消するには :279 行の削除が望ましい。

---

## 7. BUG-065 改修詳細設計（追記: 2026-08-08）

> 本章は、§3 の検証で特定した「残存リスク」を解消するための改修の詳細設計である。
> 対象ソース: `src/eqprocessor/EQProcessor.Core.cpp`（reset() / prepareToPlay()）、`src/eqprocessor/EQProcessor.h`、`src/eqprocessor/EQProcessor.Processing.cpp`（Audio Thread 側検知）。

### 7.1 対象コードの現状（検証済みスナップショット）

#### reset() — EQProcessor.Core.cpp:274-284（現状）

```cpp
convo::publishAtomic(bandResetPacked, static_cast<std::uint64_t>(0), std::memory_order_release);
convo::fetchAddAtomic(agcResetSerial, static_cast<std::uint64_t>(1), std::memory_order_acq_rel); // ★ BUG-065 修正済み (increment)
rtDeferredBandResetMask.store(0, std::memory_order_relaxed);   // ← Audio Thread 専有変数への Non-RT 書込
rtSeenBandResetSerial = 0;                                     // ← 非 atomic、Audio Thread 専有
rtSeenAgcResetSerial = 0;                                      // ← 非 atomic、Audio Thread 専有（残存リスク）
```

#### prepareToPlay() — EQProcessor.Core.cpp:786-790（現状、同一パターン）

```cpp
convo::publishAtomic(bandResetPacked, static_cast<std::uint64_t>(0), std::memory_order_release);
convo::fetchAddAtomic(agcResetSerial, static_cast<std::uint64_t>(1), std::memory_order_acq_rel); // ★ BUG-065 修正済み (increment)
rtDeferredBandResetMask.store(0, std::memory_order_relaxed);
rtSeenBandResetSerial = 0;
rtSeenAgcResetSerial = 0;
```

#### Audio Thread 側の検知ロジック — EQProcessor.Processing.cpp:586-593（process 3引数版）／ :1070-1077（double 版）

```cpp
const std::uint64_t agcResetSerialNow = convo::consumeAtomic(agcResetSerial, std::memory_order_acquire);
if (agcResetSerialNow != rtSeenAgcResetSerial)
{
    rtSeenAgcResetSerial = agcResetSerialNow;
    rtAgcCurrentGainShadow.store(1.0, std::memory_order_relaxed);
    rtAgcEnvInputShadow.store(0.0, std::memory_order_relaxed);
    rtAgcEnvOutputShadow.store(0.0, std::memory_order_relaxed);
}
```

バンド側も同一パターン（:595-601）: `bandResetSerialNow != rtSeenBandResetSerial` 検知 → `rtDeferredBandResetMask.fetch_or(mask)`。

#### 関連データ構造 — EQProcessor.h

```cpp
// :487-490 — Message Thread publish / Audio Thread consume-only の契約
std::atomic<std::uint64_t> bandResetPacked { 0 };   // high 32-bit = serial, low 32-bit = mask
std::atomic<std::uint64_t> agcResetSerial { 0 };
// :715-721 — Audio Thread 専有シャドウ
std::atomic<std::uint32_t> rtDeferredBandResetMask { 0 };
std::uint64_t rtSeenBandResetSerial = 0;   // 非 atomic（Audio Thread 専有）
std::uint64_t rtSeenAgcResetSerial = 0;    // 非 atomic（Audio Thread 専有）
```

### 7.2 残存リスクの本質（3 点）

| # | リスク | 深刻度 | 根拠 |
|---|-------|--------|------|
| **R1** | `rtSeenAgcResetSerial` / `rtSeenBandResetSerial`（**非 atomic uint64_t**）への Non-RT スレッドからの直接書き込み = **C++ 標準上の data race（UB）** | MEDIUM | Audio Thread が `process()` で同変数を読み書き中に、Worker/Message/UI スレッドから書込みが走り得る。x86-64 では torn read は起きにくいが、標準上は UB であり TSan 検出対象 |
| **R2** | `bandResetPacked` を **serial ごと 0 に publish** する巻き戻し。Audio Thread の「処理済み serial N」が残っている状態で 0 に戻すと、`bandResetSerialNow(0) != rtSeenBandResetSerial(N)` で **無意味な再検知**が走る（mask=0 なのでリセット自体は発生しないが、serial の単調性が破れる） | LOW | `requestBandReset()` は CAS で serial を **前進のみ**（+1）する設計（EQProcessor.h:508-531）。0 への publish はこの単調性に反する |
| **R3** | `rtDeferredBandResetMask.store(0, relaxed)` が Audio Thread の `fetch_or` と競合し、**Audio Thread が累積中のバンドリセット要求が失われる**可能性 | LOW | `fetch_or` と `store(0)` の直後の `exchange(0)` が interleave すると、Audio Thread が拾い損ねる要求が発生し得る。ただし reset() は `filterState` を直接 memset 済みのため、失われても機能上の実害は小さい |

**補足（呼び出し元スレッドの実測調査 — 2026-08-08）**:

- `EQProcessor::prepareToPlay()` は `EQRuntimeState::prepare()` → `DSPCore::prepare()` → `RuntimeBuilder::build()`（RuntimeBuilder.cpp:448）/ `AudioEngine.Processing.PrepareToPlay.cpp:245`（placeholderDSP）から呼ばれる。いずれも **Worker / Message スレッド（Non-RT）**。JUCE の `prepareToPlay` はデバイス開始前に呼ばれるため Audio Thread は通常停止中だが、**Audio Thread 稼働中に呼ばれる経路（rebuild publish 時等）が存在しないことの保証はコード上に明記されていない**。
- `EQProcessor::reset()` は `EQRuntimeState::resetForRuntime()` → `DSPCore::reset()`（AudioEngine.Processing.DSPCoreLifecycle.cpp:335）から呼ばれるが、**DSPCore::reset() の呼び出し元は grep で検出されず、現状デッドコード化の可能性が高い**。ただし将来 publish / rebuild パスで使用される可能性があるため、契約は安全側で設計する。
- 正しいパターン: `resetToDefaults()`（EQProcessor.Core.cpp:206-242）は `requestAllBandReset()` + `requestAgcReset()` のみを使い、**rt シャドウには一切触らない**。これが準拠すべき既存の安全パターン。

### 7.3 スレッド契約（改修後の不変条件）

1. **Non-RT → Audio Thread の通信は atomic の publish のみ**: `agcResetSerial`（fetchAdd）、`bandResetPacked`（CAS）に限定。
2. **rt シャドウ変数（`rtSeenAgcResetSerial` / `rtSeenBandResetSerial` / `rtDeferredBandResetMask` / `rtAgc*Shadow`）は Audio Thread 専有**。Non-RT からは**一切読み書きしない**（EQProcessor.h:487 の既存コメント「Message Thread publish / Audio Thread consume-only」に統一）。
3. **serial は常に単調増加**: `bandResetPacked` の serial を巻き戻さない（0 へ publish しない）。mask のみ 0 にクリアする場合は CAS で serial を +1 した上で mask=0 を publish する。
4. **リセットの検知は Audio Thread 側に委ねる**: Non-RT が「処理済み」を先回りして設定しない。Audio Thread は `serial != rtSeen` を検知した時点で自分自身のシャドウを更新する。

### 7.4 修正案

#### 案 A: rt シャドウ書込の全廃 + bandResetPacked の serial 前進クリア（推奨）

reset() / prepareToPlay() の末尾 3 行を削除し、bandResetPacked の publish を「serial 前進 + mask=0 クリア」に変更する。

```cpp
// ★ BUG-065 改修（案 A）: reset() / prepareToPlay() 共通

// (1) bandResetPacked: 保留中バンドリセット要求をクリア（serial は前進、mask のみ 0 に）
//     0 への publish は serial 巻き戻し（R2）を招くため、requestBandReset と同型の CAS で
//     serial を +1 した状態に forward する。Audio Thread は「serial 変化 + mask=0」を検知し、
//     rtDeferredBandResetMask.fetch_or(0) により何もせず rtSeen を同期する。
std::uint64_t packed = convo::consumeAtomic(bandResetPacked, std::memory_order_acquire);
for (;;)
{
    const std::uint32_t serial = bandResetSerialFromPacked(packed);
    const std::uint64_t desired = makeBandResetPacked(static_cast<std::uint32_t>(serial + 1u), 0u);
    if (convo::compareExchangeAtomic(bandResetPacked, packed, desired,
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire))
        break;
}

// (2) AGC: serial を increment（既存の ★ BUG-065 修正を維持）
convo::fetchAddAtomic(agcResetSerial, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);

// (3) rt シャドウへの書込は全て削除（Audio Thread 専有。Audio Thread が検知時に自身で同期する）
//     rtDeferredBandResetMask.store(0, relaxed);  ← 削除（R3）
//     rtSeenBandResetSerial = 0;                 ← 削除（R1）
//     rtSeenAgcResetSerial = 0;                  ← 削除（R1）
```

**修正前後の動作比較（AGC）**:
- 修正前: `serial = N → N+1`（fetchAdd）→ `rtSeen = 0`。Audio Thread は `N+1 != 0` で検知 → rtSeen を N+1 に更新。機能は動くが、(a) Non-RT の `rtSeen=0` 書込が Audio Thread の読み書きと data race（R1）、(b) Audio Thread が既に N+1 を処理した**後**に `rtSeen=0` が書かれると、次回 process で `N+1 != 0` の **2 回目リセット**が発火し得る（意味論の不正確さ）。
- 修正後: `serial = N → N+1`（fetchAdd）のみ。Audio Thread は `N+1 != rtSeen(=N)` で 1 回だけ検知 → rtSeen を N+1 に更新。**data race なし・二重リセットなし・Audio Thread 自身が単一の真実を維持**。

**修正前後の動作比較（Band）**:
- 修正前: `packed = 0`（serial 巻き戻し）＋ `rtSeenBandResetSerial = 0`。Audio Thread が N を処理済みでも `0 != N` で再検知（無害だが serial 単調性違反・data race）。
- 修正後: `packed = (serial+1 << 32) | 0`。Audio Thread は `serial+1 != N` で 1 回検知 → mask=0 なのでリセットなし → rtSeen を serial+1 に同期。**serial 単調性維持・data race なし**。

#### 案 B: 最小修正（rt シャドウ書込の削除のみ）

案 A の (3) のみ実施し、bandResetPacked の 0 publish は現状維持する。R1（data race）は解消するが、R2（serial 巻き戻し）と、`rtSeenBandResetSerial=0` 削除後の「Audio Thread が 0 を検知して rtSeen=0 に同期する」挙動（無害だが無駄な 1 回の検知）は残る。移行コストを最小にしたい場合の選択肢。

#### 案 C: 現状維持 + 契約のコメント化

実害が低い（x86 では torn なし、Audio Thread 停止中に呼ばれる経路が多い）として現状のままにし、7.3 の契約をコメントとして明文化するのみ。**推奨しない**（将来 Audio Thread 稼働中の reset 経路が追加された場合に UB が顕在化するため）。

### 7.5 推奨案の決定と理由

**案 A を推奨する**。理由:
1. `resetToDefaults()`（:206-242）という既存の安全パターンに完全一致する（rt シャドウ不干渉 + serial 前進）。
2. R1（data race）を標準的に根絶し、TSan クリーンを達成できる。
3. R2（serial 巻き戻し）を排除し、`bandResetPacked` の「CAS による単調増加」設計（EQProcessor.h:508-531）と整合する。
4. 変更量は reset()/prepareToPlay() の各 4〜6 行のみで、Audio Thread 側（Processing.cpp）は**無変更**（検知ロジックが既に正しいため）。

### 7.6 影響範囲とリスク

- **変更ファイル**: `EQProcessor.Core.cpp`（reset() / prepareToPlay() のみ）。`EQProcessor.h` / `EQProcessor.Processing.cpp` は変更不要。
- **機能影響**: reset() / prepareToPlay() 直後の Audio Thread での AGC・バンドリセットは**1 回だけ**確実に発火する（従来と同じ）。二重リセットの可能性が排除される。
- **回帰リスク**: 低。Audio Thread 側の検知ロジックは無変更であり、挙動差は「rtSeen を Audio Thread が自身で管理する」ことのみ。
- **残課題**: `EQProcessor::reset()` の呼び出し元（DSPCore::reset() 経由）が現状 grep で検出されない（7.2 補足）。本改修と同時に、呼び出し経路の有無を確認し、不要なら `DSPCore::reset()` 側もデッドコード整理を検討すること。

### 7.7 検証・テスト計画

1. **コンパイル/型チェック**: 変更後、`EQProcessor.Core.cpp` のビルドが通ること（`bandResetSerialFromPacked` / `makeBandResetPacked` / `compareExchangeAtomic` は EQProcessor.h 内に既存）。
2. **TSan（推奨）**: 既存の sanitizer CI（commit 75c0453 で追加）で reset() / prepareToPlay() と Audio Thread の並行実行シナリオを流し、data race 検出ゼロを確認。
3. **機能テスト**: reset() 後に process() を 1 回実行し、`rtAgcCurrentGainShadow == 1.0` / `rtAgcEnvInputShadow == 0.0` / `rtAgcEnvOutputShadow == 0.0` になること、および連続 reset() で**2 回目以降のリセットが発火しない**こと（serial が進むため `serial == rtSeen` でスキップ）を検証。
4. **単調性テスト**: `bandResetPacked` の serial が reset() 後も巻き戻らず前進のみであることを単体テストで確認。
5. **回帰テスト**: 既存の EQ / AGC 関連テスト（あれば）を全て通過させる。

### 7.8 実装メモ（案 A 適用時）

- `bandResetSerialFromPacked` / `makeBandResetPacked` は EQProcessor.h:492-506 に静的 constexpr として存在するため、Core.cpp から直接利用可能。
- CAS ループの失敗時は `expected` が最新値に更新される（compareExchangeAtomic のセマンティクス）ため、retry で直ちに最新 serial から再計算される。
- 案 A の (1) は reset() と prepareToPlay() の両方に同コードを置く。共通ヘルパー（例: `clearPendingBandResetRequests()`）への抽出も可。
- 削除対象 3 行のうち、`rtDeferredBandResetMask.store(0, relaxed)` の削除が R3 に該当。Audio Thread が累積中の要求をクリアする必要が真にある場合のみ、`bandResetPacked` の mask=0 クリア（案 A (1)）がその代替を担う（Audio Thread は mask=0 検知で何もしない）。

---

## 8. 「重要な新事実」深掘り調査結果（追記: 2026-08-08）

> §7 の「重要な新事実」3 点について、ソースコードの実査（grep 全数検索・git 履歴・呼び出しチェーン追跡）により深掘り調査した結果。

### 8.1 新事実① — prepareToPlay() の呼び出し経路と実行スレッド（完全追跡）

**呼び出しチェーン（実測）**:

```
EQProcessor::prepareToPlay()  [eqState->prepare() → ref().prepareToPlay()]
  ▲ DSPCore::prepare()  (定義: AudioEngine.Processing.DSPCoreLifecycle.cpp:72。:200/:255 は内部の convolverState/eqState->prepare() 呼び出し)
  │   ┌─ 経路 A: AudioEngine::prepareToPlay() (JUCE prepareToPlay) → placeholderDSP->prepare() (PrepareToPlay.cpp:245)
  │   └─ 経路 B: RuntimeBuilder::build() (RuntimeBuilder.cpp:448) → runtime->prepare()
  │            ▲ AudioEngine::rebuildThreadLoop() (定義: RebuildDispatch.cpp:804。RuntimeBuilder::build() 使用は :905 付近)
  │              ▲ rebuildThread 専用スレッド (AudioEngine.Init.cpp:33 / PrepareToPlay.cpp:85 で std::thread 起動)
```

**実行スレッド判定**:

| 経路 | 呼び出し元 | スレッド | Audio Thread 並行稼働? |
|------|-----------|---------|----------------------|
| A | `AudioEngine::prepareToPlay()` (PrepareToPlay.cpp:245, placeholderDSP) | **Message/UI スレッド**（JUCE ライフサイクル） | **通常は No**（JUCE 契約: prepareToPlay 実行中は Audio callback が走らない — コメント明記） |
| B | `RuntimeBuilder::build()` (RuntimeBuilder.cpp:448) | **Rebuild Thread**（専用 std::thread、HeavyBackground アフィニティ） | **Yes（常に並行稼働）** |

**結論**: `EQProcessor::prepareToPlay()` は **Rebuild Thread から Audio Thread 稼働中に呼ばれる**（経路 B）。これは AudioEngine の通常動作（パラメータ変更→submitRebuildIntent→rebuildThreadLoop→build→prepare）で頻繁に発生する。したがって §7.2 の R1（`rtSeenAgcResetSerial = 0` 直接書込の data race）は**理論上の懸念ではなく、実際の実行経路で発生し得る**。

**ただし重要な緩和要素**: 経路 B の `runtime` は**新規構築した DSPCore**（RuntimeBuilder.cpp:443 で `aligned_make_unique<DSPCore>`）であり、`prepare()` は **publish 前（Audio Thread に公開される前）** に呼ばれる。新規 DSPCore の `EQProcessor` インスタンスは Audio Thread がまだ参照していないため、**同一インスタンスへの同時アクセス（data race）は構造的に起きない**。問題が顕在化するのは「既存の公開済み DSPCore に対して reset()/prepareToPlay() が呼ばれる」経路のみ（→ 8.2 で後述）。

### 8.2 新事実② — DSPCore::reset() / EQProcessor::reset() のデッドコード化（確定）

**調査結果: 呼び出し元ゼロを全数確認**

| 検証項目 | 結果 |
|---------|------|
| `DSPCore::reset()` 定義 | DSPCoreLifecycle.cpp:335（`convolverState->resetForRuntime(); eqState->resetForRuntime();` 等を呼ぶ） |
| `DSPCore::reset()` の呼び出し元 | **ゼロ**（リポジトリ全ソース grep：`->reset()` / `.reset()` 全数調査で DSPCore インスタンスに対する呼び出しなし。AudioEngine.h / *.cpp / テストすべて） |
| `eqState->resetForRuntime()` 呼び出し元 | DSPCoreLifecycle.cpp:338（**DSPCore::reset() 内のみ**） |
| `resetForRuntime()` 定義 | AudioEngine.h:726（`ref().reset()`） |
| `EQProcessor::reset()` の呼び出し元 | **ゼロ**（DSPCore::reset() 経由のみであり、その DSPCore::reset() 自体が未呼び出し） |
| virtual 経由の可能性 | `reset()` は非 virtual（EQProcessor.h:259、AudioEngine.h:867）。継承（EQEditProcessor）経由でも直接呼び出しなし |
| git 履歴 | `DSPCore::reset()` 導入コミット e208c26（ファイル分割時）以降、呼び出し元が追加された形跡なし（`git log -S "eqState->resetForRuntime()"` で検出されず） |
| テスト | src/tests/ に DSPCore::reset() / EQProcessor::reset() 呼び出しなし |

**判定: `EQProcessor::reset()` は実質デッドコード（呼び出しパスが存在しない）。**

このことは BUG-065 の評価に二つの影響を与える:

1. **R1 の実害は限定的**: `reset()` 内の `rtSeenAgcResetSerial = 0`（:279）は現在実行されないため、data race は「将来 reset() が使われた場合」に顕在化する潜在リスクである。ただし `prepareToPlay()` 内（:790）は Rebuild Thread 経由で**実際に実行される**（8.1）。ただし publish 前の新規 DSPCore に対してのみなので、現状は Audio Thread と同一インスタンスを共有しない。
2. **修正優先度の再評価**: 「デッドコードの reset() を直す」ことより、「将来 reset() が活性化されたときに data race にならない設計」を保証する方が本質的。案 A（rt シャドウ書込の削除）はまさにそれを満たすため、実施コストが低く将来安全を確保できる。

**推奨アクション**:
- `DSPCore::reset()`（および `EQProcessor::reset()`）の呼び出し経路が本当に不要か確定する（8.3 の調査で UI 経路は resetToDefaults のみであることを確認済み）。
- 不要なら `DSPCore::reset()` を削除（または「将来用に予約」コメントを付けて public を維持）。**少なくとも `EQProcessor::reset()` は rt シャドウ書込を除去した安全な形にしておく**（案 A）ことで、将来の活性化時に即安全。

### 8.3 新事実③ — resetToDefaults() が既存の安全パターンであること（確認）

**呼び出しチェーン（実測）**:

```
EQControlPanel.cpp:433 (Reset ボタン) → engine.resetEQToDefaults()
  → uiEqEditor.resetToDefaults() (AudioEngine.h:1247)
    → EQEditProcessor::resetToDefaults() (EQEditProcessor.cpp:116)
      → EQProcessor::resetToDefaults() (EQProcessor.Core.cpp:206)
        → requestAllBandReset() → requestBandReset(0xFFFFFFFF)   // CAS で serial 前進 + mask 設定
        → requestAgcReset() → fetchAddAtomic(agcResetSerial, 1) // serial 前進
        → (rt シャドウ変数には一切触れない)  ★
```

**確認事項**:
- `resetToDefaults()`（EQProcessor.Core.cpp:206-242）は `agcCurrentGain` / `agcEnvInput` / `agcEnvOutput` を atomic publish で初期化し、`requestAllBandReset()` + `requestAgcReset()` で serial を前進させる。**`rtSeen*` / `rtDeferredBandResetMask` には一切書き込まない**。
- これは §7.3 のスレッド契約（Non-RT は publish のみ・rt シャドウは Audio Thread 専有）を**既に満たす既存の安全パターン**であり、`reset()` が準拠すべき参照実装である。
- UI 経路のリセットはすべて `resetToDefaults()` 経由であり、`reset()` は使われていない。

### 8.4 調査の総括（新事実の精緻化）

| 新事実 | 深掘り後の確定内容 | 影響 |
|--------|-------------------|------|
| ① prepareToPlay の呼び出し経路 | 経路 A（JUCE prepareToPlay, Message スレッド）＋ 経路 B（**Rebuild Thread, Audio Thread 並行稼働**）の 2 経路。B は通常動作で頻繁に発生 | R1 は実際に実行されるコードパスに存在（ただし対象は publish 前の新規 DSPCore のみ） |
| ② DSPCore::reset / EQProcessor::reset のデッドコード化 | **呼び出し元ゼロを全数確認（grep 全ソース + git 履歴 + テスト）。実質デッドコード** | BUG-065 の :279 は現状未実行。将来の活性化リスクとして対処 |
| ③ resetToDefaults が安全パターン | UI リセットは全て resetToDefaults 経由（reset() 不使用）。既に §7.3 契約を満たす参照実装 | 案 A の実装は resetToDefaults のパターンに機械的に合わせればよい |

**結論**:
1. `EQProcessor::reset()` は実質デッドコードだが、**案 A の改修（rt シャドウ書込削除）は将来の活性化に備えて実施する価値がある**（コスト 4〜6 行、resetToDefaults の既存パターンに一致、Audio Thread 側は無変更）。
2. `prepareToPlay()` 側（:790）は Rebuild Thread から**実際に実行される**ため、こちらが現実的な修正対象。publish 前の新規 DSPCore に対してのみ呼ばれるため現状 data race は顕在化しないが、「将来 DSPCore が再利用され得る」可能性を考慮すると安全化が望ましい。
3. 併せて **`DSPCore::reset()`（および不要なら `EQProcessor::reset()`）のデッドコード整理**（削除 or 予約コメント化）を推奨する。

---

## 9. §7 設計の精緻化（追加調査: 2026-08-08）

> ユーザー依頼「情報不足な点を可能な限り詳細に調査し、詳細設計を精緻化」への対応。§7 の BUG-065 改修詳細設計を、追加のコード実査で得た新事実に基づき補強・修正する。

### 9.1 追加調査で確定した新事実（§7 の前提を補強/修正）

#### F-1: `rtSeenAgcResetSerial` への書込は 4 箇所、うち**実行されるのは 1 箇所のみ**

| 書込箇所 | 関数 | 実行される? | 実行スレッド |
|---------|------|------------|------------|
| EQProcessor.Core.cpp:279 | `reset()` | **No**（呼び出し元ゼロ = デッドコード、§8.2 確定） | — |
| EQProcessor.Core.cpp:603 | `syncStateFrom()` | **No**（呼び出し元ゼロ — 追加確定、下記 F-2） | — |
| EQProcessor.Core.cpp:667 | `syncGlobalStateFrom()` | **No**（同上） | — |
| EQProcessor.Core.cpp:790 | `prepareToPlay()` | **Yes** | Rebuild Thread（経路 B）／ Message Thread（経路 A） |

**結論**: 現状実行される Non-RT 側の `rtSeenAgcResetSerial` 書込は `prepareToPlay()`（:790）のみ。reset() 側（:279）は §7 で扱ったが、**実害の主対象は prepareToPlay() 側**であることが確定した。

#### F-2: `syncStateFrom()` / `syncGlobalStateFrom()` もデッドコード（呼び出し元ゼロ）

- `EQProcessor::syncStateFrom`（Core.cpp:561）、`syncGlobalStateFrom`（:624）、`syncBandNodeFrom`（:610）は **リポジトリ全体で呼び出し元ゼロ**（grep 全数確認）。
- EQProcessor.h:697 のコメント「Source of Truth → Worker syncStateFrom() → this shadow」は**設計意図の記述であり、実装されていない**（同期は snapshot → eqParams → coeffCache 経由で行われる — F-3）。
- したがって :603 / :667 の `rtSeenAgcResetSerial = syncedAgcResetSerial` も現状未実行。ただし reset() と同様、**将来の活性化リスク**として §7.3 契約の対象に含めるべき。

#### F-3: Audio Thread は eqParams（snapshot 由来）と coeffCache で処理 — DSPCore 内 eq の内部状態は直接使われない

- Audio Thread の EQ 処理は `state.eqParams`（**snapshot 由来**、AudioEngine.Snapshot.cpp:31 `uiEqEditor.getEQStateSnapshot() → toEQParameters()`）+ `state.eqCache`（coeffCache）を 3引数版 `process(block, eqParams, coeffCache)` に渡す（DSPCoreDouble.cpp:381-400 / DSPCoreFloat.cpp:300）。
- つまり **DSPCore 内の EQProcessor の内部状態（currentState / bandNodes）は、Audio Thread の主処理では直接参照されない**。3引数版は `eqParams` / `coeffCache` のみで処理し、`process(block)`（1引数）へは bypass / cache null / M-S 時のみフォールバックする（Processing.cpp:1035-1045）。
- **影響**: reset() が触る「内部状態（filterState 等）」の Audio Thread での実効的な意味は「フォールバック時」に限定される。一方 **AGC の shadow（rtAgc*Shadow）は processAGC（3引数版内 :1256 / 1引数版内 :951 両方）で使用される**ため、AGC リセット（serial 経由）の重要度は高いまま。

#### F-4: `agcCurrentGain` atomic は Audio Thread から読まれない — 二重管理の実態

- `processAGC`（Processing.cpp:367-463）は **`rtAgcCurrentGainShadow` / `rtAgcEnvInputShadow` / `rtAgcEnvOutputShadow`（RT-local shadow）のみを使用**（:421, :440）。
- `agcCurrentGain` / `agcEnvInput` / `agcEnvOutput` atomic は **Audio Thread からは一度も読まれない**（grep 全数確認）。読み手はデッドコードの `syncStateFrom` / `syncGlobalStateFrom`（Core.cpp:580-582, 638-640）のみ。
- **結論**: 
  - reset() / prepareToPlay() の `publishAtomic(agcCurrentGain, 1.0)` は Audio Thread の実処理には**直接影響しない**（shadow は serial 経由でリセットされる）。
  - 「atomic を 1.0 に戻す」と「shadow を 1.0 に戻す（serial 経由）」の二重管理のうち、**機能上重要なのは shadow 側のみ**。
  - したがって BUG-065 修正の本質は「**serial を increment して shadow リセットを確実に発火させること**」であり、atomic 側の `publishAtomic(agcCurrentGain, 1.0)` は補助的（将来の sync 用スナップショット）。§7.1 の判断と整合。

#### F-5: `canSafelyResetState`（バンドリセット）と AGC リセットの非対称性

- **バンドリセット**: Audio Thread は silent/bypass 時のみ適用し、それ以外は `rtDeferredBandResetMask` に**戻す**（defer）。そのため「serial 巻き戻し」が起きても、**非 silent 中はリセットが実行されず次の silent ブロックまで待つ**（Processing.cpp:603-609, 1083-1091 の `isAudioBlockSilent` チェック）。
- **AGC リセット**: silent 判定なしで**即時適用**（:587-593, :1070-1077）。serial が変われば次の process で必ず shadow が 1.0/0.0/0.0 に戻る。
- **設計への影響**: 
  - AGC は即時性が高いため、「serial を increment する」方式が**唯一の確実なリセット手段**（atomic publish は shadow に効かない）。
  - バンドは defer 機構があるため、`bandResetPacked` の 0 publish による serial 巻き戻し（R2）は「非 silent 中は実害なし」だが、**silent 中の不意の再リセット**（例: 再生停止→再開で意図せず全バンド memset）を誘発し得る。§7.4 案 A の「serial 前進 + mask=0」修正がこれを確実に防ぐ。

#### F-6: prepareToPlay() は filterState を memset する（新規 DSPCore に対してのみ）

- prepareToPlay()（Core.cpp:752 付近）は `std::memset(filterState.data(), 0, sizeof(filterState))` を実行する（§7 で未記載だった点）。
- これは**新規構築・未 publish の DSPCore** に対してのみ実行される（§8.1 経路 A/B とも新規 DSPCore）。Audio Thread は publish 後のみ参照するため、memset 自体は安全。
- **設計への影響**: reset() も同様に filterState を memset するが、こちらはデッドコード。案 A では**rt シャドウ書込のみを削除**し、filterState の memset 等の「新規インスタンス初期化として意味のある処理」は**現状維持**とする（将来 reset() が活性化された場合も、publish 前呼び出しなら memset は安全。publish 後呼び出しは想定外として契約に明記）。

#### F-7: prepareToPlay() の rt シャドウ書込は「新規インスタンスへの初期化」として実質無害

- prepareToPlay() は新規 DSPCore に対してのみ呼ばれるため、`rtSeenAgcResetSerial = 0` / `rtSeenBandResetSerial = 0` / `rtDeferredBandResetMask.store(0)` は**新規インスタンスの初期値（既に 0）への再書込**に等しい（EQProcessor.h:721 のメンバ初期値 0）。
- つまり prepareToPlay() 側の R1/R3 は「実行されるが実質無害」であり、**設計上は「将来の DSPCore 再利用」に備えた安全化**として位置づけられる。

### 9.2 §7 設計の修正・補強（追加調査の反映）

#### §7.2 残存リスクの再評価

| リスク | §7 の評価 | 追加調査後の評価（§9） |
|-------|----------|----------------------|
| R1（rtSeen 書込 data race） | 潜在リスク | **「実行されるのは prepareToPlay() のみ・対象は新規 DSPCore のみ」で現状は実質無害**（F-1, F-7）。ただし将来の reset() / syncStateFrom() 活性化で顕在化し得るため契約として対処 |
| R2（bandResetPacked serial 巻き戻し） | LOW | バンドは defer 機構により非 silent 中は無害だが、**silent 中の不意の全バンド memset を誘発し得る**（F-5）。修正意義がやや増す |
| R3（rtDeferredBandResetMask.store(0)） | LOW | prepareToPlay() では新規インスタンス（初期値 0）への再書込で無害（F-7）。reset() 活性化時の問題 |

#### §7.4 修正案の補強

- **案 A の適用範囲を「reset() / prepareToPlay() 両方」から「prepareToPlay()（実効）＋ reset()（将来安全）＋ syncStateFrom() / syncGlobalStateFrom()（将来安全）」へ拡張**するのが望ましい。
  - 実効修正（必須）: prepareToPlay() の rt シャドウ 3 行削除（:788-790）。
  - 将来安全（推奨）: reset()（:277-279）、syncStateFrom()（:601-603）、syncGlobalStateFrom()（:665-667）の rt シャドウ書込を同一パターンで削除。
- **注意（新規）**: syncStateFrom / syncGlobalStateFrom は「rt シャドウへ同期値を書く」ことが関数の**目的そのもの**（F-2）。単純削除はできない。
  - 正しい扱い: これらの関数は「非 atomic の rtSeen への書込」をやめ、**「agcResetSerial / bandResetPacked を同期値に引き上げる」（serial を追い越させる）方式**に変更する。すなわち `syncedAgcResetSerial` を「直接 rtSeen に書く」のではなく、**対象 EQProcessor の agcResetSerial を `syncedAgcResetSerial` と一致させる fetchAdd** にする。
  - ただし、これらの関数自体がデッドコードのため、**活性化時に再設計する方が現実的**（現段階では「活性化時は §7.3 契約に従う」旨をコメント化）。

#### §7.6 影響範囲の更新

- 変更対象ファイルは **EQProcessor.Core.cpp の 4 関数**（reset / prepareToPlay / syncStateFrom / syncGlobalStateFrom）。
- Audio Thread 側（Processing.cpp）は無変更のまま（検知ロジックは正しい — F-5）。

#### §7.7 検証・テスト計画の補強

- **テスト 6（追加）**: `syncStateFrom` / `syncGlobalStateFrom` 活性化時（呼び出し元が追加された場合）に、rt シャドウ書込ではなく serial 同期方式が使われていることをテスト（またはコンパイル時検証）で確認。
- **テスト 7（追加）**: prepareToPlay() 後に「非 silent ブロック処理中」のバンドリセットが defer され、silent ブロックで正しく 1 回だけ適用されることを確認（F-5 の動作検証）。

### 9.3 精緻化後の最終推奨（§7 結論の更新）

1. **実効修正（必須）**: `prepareToPlay()` の rt シャドウ 3 行（:788-790）を削除。agcResetSerial / bandResetPacked の serial 前進方式は維持。**Audio Thread は serial 差分で確実に shadow をリセットする**（F-4, F-5 により唯一の確実な手段であることが確定）。
2. **将来安全（推奨）**: `reset()` の rt シャドウ 3 行（:277-279）も同一パターンで削除。`syncStateFrom()` / `syncGlobalStateFrom()` は**デッドコードのため現状はコメントで契約を明記**し、活性化時に serial 同期方式へ再設計（F-2）。
3. **デッドコード整理（推奨）**: `DSPCore::reset()` / `EQProcessor::reset()` の呼び出し経路が無いことを確定済み（§8.2）。不要なら削除 or 予約コメント化。
4. **変更範囲**: EQProcessor.Core.cpp のみ（4 関数）。Processing.cpp / EQProcessor.h は無変更。

---

## 10. AGC 二重管理（atomic vs shadow）の検証と shadow 単一管理へのリファクタリング案（追記: 2026-08-08）

> ユーザー依頼: F-4（atomic `agcCurrentGain` 等は Audio Thread から読まれず shadow のみ使用）の設計検証と、atomic 側 publish を廃止して shadow 単一管理に統一するリファクタリング案の設計。

### 10.1 F-4 の完全検証（全参照の網羅調査結果）

#### atomic 側（`agcCurrentGain` / `agcEnvInput` / `agcEnvOutput`）

| 種別 | 箇所 | 関数 | 実行される? |
|------|------|------|------------|
| 書込 | EQProcessor.Core.cpp:241-243 | `resetToDefaults()` | **Yes**（UI Reset ボタン経由、§8.3） |
| 書込 | EQProcessor.Core.cpp:264-266 | `reset()` | No（デッドコード、§8.2） |
| 書込 | EQProcessor.Core.cpp:782-784 | `prepareToPlay()` | Yes（Rebuild/Message Thread、§8.1） |
| 読取 | EQProcessor.Core.cpp:580-582 | `syncStateFrom()` | No（デッドコード、§9 F-2） |
| 読取 | EQProcessor.Core.cpp:638-640 | `syncGlobalStateFrom()` | No（デッドコード、§9 F-2） |

**Audio Thread（Processing.cpp）からの参照: ゼロ**（processAGC は shadow のみ使用 — §9 F-4）。
**UI / AudioEngine / getter / テレメトリからの参照: ゼロ**（`getAgc*Gain` 系 API は存在しない。audioengine / EQEditProcessor / EQControlPanel / SpectrumAnalyzer 全て grep で 0 件）。

#### shadow 側（`rtAgcCurrentGainShadow` / `rtAgcEnvInputShadow` / `rtAgcEnvOutputShadow`）

| 種別 | 箇所 | 関数 | 実行される? |
|------|------|------|------------|
| 読/書 | Processing.cpp:419-421, 438-440 | `processAGC()` | **Yes**（Audio Thread 実処理） |
| 書込 | Processing.cpp:590, 1074 | AGC リセット（serial 検知） | **Yes**（Audio Thread） |
| 書込 | Core.cpp:597-599 | `syncStateFrom()` | No（デッドコード） |
| 書込 | Core.cpp:661-663 | `syncGlobalStateFrom()` | No（デッドコード） |

**shadow は Audio Thread 専有の実データ**（processAGC の唯一の状態源）。

#### 検証結論（F-4 確定）

- **atomic 3 つは「書くだけで誰も読まない」状態**。読み手はデッドコードの sync 関数のみであり、**現状コードベース全体で意味のある読み取りはゼロ**。
- AGC の「現在の状態」（ゲイン・エンベロープ）の**唯一の真実源は shadow**（Audio Thread 専有）。
- atomic は「Non-RT 側から見た AGC 状態スナップショット」として設計された名残であり、**現状は冗長な二重管理**。
- **例外（削除してはならない atomic）**: `agcAttackCoeff` / `agcReleaseCoeff` / `agcSmoothCoeff`（:548-550）は **processAGC が atomic で読む**（Processing.cpp:369-371）ため実使用あり。`agcEnabled` / `m_pendingAGCChange` は UI が読む別系統（有効フラグ）であり対象外。

### 10.2 リファクタリング案

#### 案 A: atomic 3 つを廃止し、shadow 単一管理へ（推奨）

**方針**: `agcCurrentGain` / `agcEnvInput` / `agcEnvOutput` の atomic メンバを削除し、AGC 状態は `rtAgc*Shadow`（Audio Thread 専有）のみで管理する。Non-RT 側の「リセット要求」は既存の serial 機構（`agcResetSerial`）のみで伝える。

**変更点**:

1. **EQProcessor.h:545-547** — メンバ削除:
   ```cpp
   // 削除
   std::atomic<double> agcCurrentGain { 1.0 };
   std::atomic<double> agcEnvInput    { 0.0 };
   std::atomic<double> agcEnvOutput   { 0.0 };
   ```
2. **resetToDefaults()（:241-243）** — publish 3 行削除:
   ```cpp
   // 削除（shadow は resetToDefaults → requestAgcReset() の serial 経由で Audio Thread がリセット）
   convo::publishAtomic(agcCurrentGain, 1.0, ...);
   convo::publishAtomic(agcEnvInput, 0.0, ...);
   convo::publishAtomic(agcEnvOutput, 0.0, ...);
   ```
3. **reset()（:264-266）** — publish 3 行削除（関数自体はデッドコードだが将来安全のため同時修正）。
4. **prepareToPlay()（:782-784）** — publish 3 行削除。
5. **syncStateFrom()（:580-582 読取 / :597-599 shadow 書込）** — 削除（関数自体がデッドコード。活性化時は §9 方針の serial 同期方式で再設計）。
6. **syncGlobalStateFrom()（:638-640 読取 / :661-663 shadow 書込）** — 同上。

**動作の同一性**:
- リセット経路は変更なし: `resetToDefaults()` → `requestAgcReset()` → `fetchAddAtomic(agcResetSerial, 1)` → Audio Thread が `serial != rtSeen` を検知 → `rtAgc*Shadow` を 1.0/0.0/0.0 に（Processing.cpp:587-593 / 1070-1077）。
- `agcAttackCoeff` 等の係数 atomic は維持（processAGC の実使用のため）。
- UI 表示・スナップショット・テレメトリは元々 atomic を参照していないため影響なし。

**削除理由の要約（コメントに残すべき文言）**:
```cpp
// ★ AGC 状態は Audio Thread 専有の rtAgc*Shadow のみで管理する（二重管理廃止）。
//   agcCurrentGain 等の atomic は Audio Thread から読まれず、読み手はデッドコードの
//   syncStateFrom / syncGlobalStateFrom のみだった。リセットは serial 機構で伝える。
```

#### 案 B: atomic を残し、読み手を追加する（将来の同期機能用）

`getAgcCurrentGain()` 等の getter を追加し、UI やテレメトリで AGC ゲインを表示できるようにする。ただし: 
- atomic には**常に最新の AGC 状態が書かれるとは限らない**（processAGC は shadow のみを更新するため、atomic に反映するには processAGC 内で publish が必要 = Audio Thread での atomic store 追加）。
- Audio Thread での atomic store は performance 劣化と RT 規約リスク（publishAtomic の RT 使用は AudioEngine 内では禁止されている — DSPCoreFloat.cpp:286 のコメント参照）。
- **現状の要件（AGC 状態の外部表示）が存在しないため、案 B は採用しない。**

#### 案 C: 現状維持（二重管理のまま）

- 機能上は現状も正しく動作する（atomic は無害な冗長 publish、shadow が実データ）。
- ただし「読まれない atomic への publish」は誤解を招き、将来 sync 関数を活性化した際に §9 と同じ data race リスクを持つ。**推奨しない。**

### 10.3 リスクと影響評価

| 項目 | 評価 |
|------|------|
| 機能影響 | なし。リセット経路（serial）は不変、AGC 実処理（shadow）は不変、UI/スナップショットは atomic 非参照 |
| RT 規約 | 改善。Audio Thread での atomic 操作は増えない（削除のみ） |
| メモリ | 3 atomic × 8 バイト削減（微） |
| 保守性 | 向上。「書くだけで読まれない」変数が消え、真実源が 1 つになる |
| 将来リスク | 同期関数（syncStateFrom 等）を復活させる場合は serial 同期方式（§9）で再設計が必要 |
| 変更範囲 | EQProcessor.h（メンバ削除）+ EQProcessor.Core.cpp（4 関数）。Processing.cpp は無変更 |

### 10.4 検証・テスト計画

1. **コンパイル**: 削除後のビルドが通ること（参照ゼロを確認済みのため、コンパイルエラーは出ないはず）。
2. **機能テスト**: 
   - resetToDefaults() → process() 1 回で `rtAgcCurrentGainShadow == 1.0` / `rtAgcEnvInputShadow == 0.0` / `rtAgcEnvOutputShadow == 0.0` を確認（serial 経由のリセット動作）。
   - AGC 有効時に processAGC が正常にゲインを更新することを確認。
3. **参照ゼロの再確認**: 削除前に `grep -rn "agcCurrentGain" src/` で残存参照がゼロであることをテスト（CI スクリプト化推奨）。
4. **TSan**: 既存 sanitizer CI で data race ゼロを確認。

### 10.5 実施の推奨タイミング

- 単独で実施可能（BUG-065 の serial 修正とは独立）。ただし、**§9 の BUG-065 修正（rt シャドウ書込削除）と同一ファイル（EQProcessor.Core.cpp）を触るため、同じコミット/作業でまとめて実施**するのが安全。
- syncStateFrom / syncGlobalStateFrom の「shadow への直接書込」（:597-599, :661-663）は §9 の R1 と同種の data race 源であるため、本リファクタリングで同時に除去される。

---

## 11. デッドコード関数（reset / sync 系）の最終確認と整理実施案（追記: 2026-08-08）

> ユーザー依頼: DSPCore::reset() / EQProcessor::reset() / syncStateFrom() / syncGlobalStateFrom() の呼び出し元ゼロを最終確認し、デッドコード整理（削除 or 予約コメント化）の実施案を提案。

### 11.1 最終確認結果（呼び出し元ゼロの全数検証）

対象 4 関数に加え、同系統の関数も網羅確認した。検索手段: `grep` 全数（src/ 全 .cpp/.h）、関数ポインタ / `std::mem_fn` / `std::bind` / テンプレート経由、git 履歴（`git log -S` / `-G`）、テスト / tools。

| 関数 | 宣言 | 定義 | 呼び出し元 | 判定 |
|------|------|------|-----------|------|
| `AudioEngine::DSPCore::reset()` | AudioEngine.h:867（**public**） | DSPCoreLifecycle.cpp:335 | **ゼロ**（`->reset()` / `.reset()` 全数検索で DSPCore インスタンスへの呼び出しなし） | **デッドコード確定** |
| `EQProcessor::reset()` | EQProcessor.h:259（**public**） | Core.cpp:259 | **ゼロ**（唯一の経路 DSPCore::reset() → eqState->resetForRuntime() が未呼び出し） | **デッドコード確定** |
| `EQProcessor::syncStateFrom()` | EQProcessor.h:356（**public**） | Core.cpp:561 | **ゼロ**（AudioEngine 側・テスト・tools すべてで 0 件） | **デッドコード確定** |
| `EQProcessor::syncGlobalStateFrom()` | EQProcessor.h:359（**public**） | Core.cpp:635 | **ゼロ** | **デッドコード確定** |
| `EQProcessor::syncBandNodeFrom()` | EQProcessor.h:358（**public**） | Core.cpp:610 | **ゼロ**（§10 調査で併せて判明） | **デッドコード確定** |
| `ConvolverProcessor::syncStateFrom()` | ConvolverProcessor.h:501（DEAD CODE 注記 :498） | StateAndUI.cpp:393（DEAD CODE 注記 :389） | **ゼロ**（AudioEngine.Parameters.cpp:641 のコメントは「次回 rebuild 時に追従」と記載するが、**実装は applyBuildSnapshot + transferIRStateFrom（RuntimeBuilder.cpp:445-447）に置換済み**） | **デッドコード確定（コメントと実装は §12 で修正済み）** |

**補足（git 履歴）**:
- `git log -G ".syncStateFrom("` は過去コミット（e208c26 分割等）で sync 呼び出しが存在した痕跡を示すが、現行 HEAD では全て削除/置換済み（06ad145「bug.md all fixed.」等で撤去）。
- `DSPCore::reset()` は導入コミット e208c26（ファイル分割）以降、呼び出し元が追加された形跡なし。
- 全関数とも **public API として残存**（クラス外部からアクセス可能）だが、利用箇所はゼロ。

### 11.2 デッドコード整理の実施案

#### 案 A: 関数削除（推奨）

対象 6 関数（上表）を削除する。

- **削除内容**: 各 .cpp の定義＋各 .h の宣言。
  - `DSPCore::reset()`: DSPCoreLifecycle.cpp:335-361 の定義＋AudioEngine.h:867 の宣言。ただし**内部の子リセット呼び出し（convolverState->resetForRuntime() / eqState->resetForRuntime() / dcBlockers().reset() / dither.reset() / noiseShaper 系 / oversampling.reset() / outputFilter.reset() / ramps().resetForRuntime() / histories().resetForRuntime()）は個別に別関数へ移すか削除を検討**（DSPCore::reset() が唯一の呼び出し元であるため、これらも間接デッドコード）。
  - `EQProcessor::reset()`: Core.cpp:259-284 の定義＋h:259 の宣言。
  - `EQProcessor::syncStateFrom()` / `syncGlobalStateFrom()` / `syncBandNodeFrom()`: 定義＋宣言。
  - `ConvolverProcessor::syncStateFrom()`: 定義＋宣言。
- **利点**: コード量削減、誤用防止（「存在するが未使用」の public API がなくなる）、§9/§10 で触れた shadow 直接書込（:597-599, :661-663）が同時に消える。
- **リスク**: 万一将来必要になった場合は再実装コスト。ただし§9 の serial 同期方式で再設計する方針が既にあるため許容。
- **注意**: DSPCore::reset() 内部の子リセット（特に convolverState / eqState / ramps / histories の resetForRuntime）が**他からも必要になる可能性**（例: 新規 DSPCore 構築時の初期化）を確認してから削除すること。現状は「新規 DSPCore はコンストラクタ＋prepare() で初期化」され、resetForRuntime は不要（§8.1 の通り prepare 経路で初期化完了）。

#### 案 B: 予約コメント化（安全側）

関数は削除せず、`[[deprecated]]` 属性＋「未使用・将来用」コメントを付けて残す。

- **変更内容**: 各宣言に `[[deprecated("Unused. Re-design with serial sync (§9/§10) if reactivated.")]]` を付与し、定義にコメント追記。
- **利点**: 将来の再実装コストがゼロ、呼び出し元ゼロを CI で監視できる。
- **リスク**: 「存在するが deprecated」の API が残り、コードベースが複雑化する。

#### 案 C: 削除＋CI 監視（推奨は A と C の併用）

案 A で削除しつつ、**「未使用関数の再発防止」として CI/スクリプトで監視**する。

- 例: `tools/` に grep ベースのスクリプトを追加し、「`DSPCore::reset()` / `EQProcessor::sync*` 等の呼び出し元がゼロであること」を assert する（§10.4 の「参照ゼロの再確認」と統合）。
- これにより、将来誰かが「何となく」呼び出して data race を復活させることを防ぐ。

### 11.3 推奨実施プラン

1. **§9 の BUG-065 修正（rt シャドウ書込削除）** と **§10 案 A（AGC atomic 廃止）** をまず適用（同一ファイル EQProcessor.Core.cpp / EQProcessor.h を触るため）。
2. **§11 案 A** で 6 関数を削除（この時点で syncStateFrom / syncGlobalStateFrom の shadow 直接書込も消える）。
3. **§11 案 C** で「呼び出し元ゼロ」監視スクリプトを tools/ に追加。
4. ビルド・テスト・TSan で検証（§7.7 / §10.4 の計画を流用）。

**段階的実施の代替案**: まず案 B（deprecated 化）で 1 リリース分観察し、問題なければ案 A（削除）に進む。リスクを最小化したい場合はこちら。

### 11.4 影響とリスク（削除時）

| 項目 | 評価 |
|------|------|
| 機能影響 | なし（呼び出し元ゼロを確認済み） |
| コンパイル | 削除後もビルドが通ることを確認必須（参照ゼロなので通るはず） |
| RT 規約 | 影響なし |
| 保守性 | 向上（未使用 public API が消え、真実源が明確化） |
| 将来リスク | sync 関数を復活させる場合は serial 同期方式（§9）で再設計。reset は新規 DSPCore 構築＋prepare() で代替（§8.1） |
| 変更範囲 | DSPCoreLifecycle.cpp / AudioEngine.h / EQProcessor.Core.cpp / EQProcessor.h / ConvolverProcessor.StateAndUI.cpp / ConvolverProcessor.h（6 ファイル） |

---

## §12 ConvolverProcessor::syncStateFrom コメント乖離の調査とコメント修正実施案（2026-08-08）

### 12.1 乖離コメントの特定

`src/audioengine/AudioEngine.Parameters.cpp:640-642`（`setConvHCFilterMode()` 内）:

```cpp
// [Mem-Fix] NUC SoA (irFreqReal/irFreqImag) を再適用するため、uiConvolverProcessor を再構築する。
// DSPCore::convolver は次回 requestRebuild 時に syncStateFrom + rebuildAllIRsSynchronous で追従する。
uiConvolverProcessor.setNUCFilterModes(...)
```

- 641 行目の **`syncStateFrom + rebuildAllIRsSynchronous` の記述が乖離**（`syncStateFrom` は §11 で呼び出し元ゼロ確定のデッドコード）。
- 640 行目（`[Mem-Fix] NUC SoA ...`）は 8f00bd5b (2026-07-08) で追記された正確な記述。

### 12.2 乖離の発生経緯（git 検証）

| コミット | 日付 | 内容 |
|---|---|---|
| e208c26a | 2026-05-10 | 分割コミット。**この時点でコメント導入**（当時は syncStateFrom 呼び出しが実在した可能性） |
| 8c9af92 | — | 「Layered Transactional Runtime Architecture」実装。**audioengine 側の syncStateFrom 呼び出しが撤去され、BuildSnapshot / applyBuildSnapshot / transferIRStateFrom 方式に置換**（`git log -S 'syncStateFrom' -- src/audioengine/` で確認） |
| 06ad145 | — | 「commit bug.md all fixed」 |

→ **コメントは置換後の機構を反映せず、当時のまま残存**。

### 12.3 実際の追従機構（全チェーン検証済み）

`setConvHCFilterMode` / `setConvLCFilterMode` 実行後、NUC フィルタモードが新 DSPCore へ伝わる実経路:

```
setConvHCFilterMode (Parameters.cpp:637)
  ├─ publishAtomic(convHCFilterMode)
  └─ uiConvolverProcessor.setNUCFilterModes() (StateAndUI.cpp:812)
       └─ pendingOverride.nucHCMode/nucLCMode 書込 + postCoalescedChangeNotification()
            └─ changeListenerCallback (UIEvents.cpp:12) → convolverParamsChanged()
                 └─ submitRebuildIntent(Structural, ConvolverParamsChanged)
                      └─ rebuildThreadLoop (RebuildDispatch.cpp)
                           ├─ task.convolverBuildSnapshot = uiConvolverProcessor.captureBuildSnapshot() (:589)
                           │     └─ snapshot.nucHCMode/nucLCMode（copyPendingToSnapshotUnlocked, StateAndUI.cpp:142-143）
                           ├─ RuntimeBuilder::build (:432)
                           │     ├─ convolverRt().applyBuildSnapshot(snapshot) (:443)
                           │     │     └─ copySnapshotToPendingUnlocked で新 DSPCore の pendingOverride へ (:194-199)
                           │     └─ convolverRt().transferIRStateFrom(engine.getConvolverProcessor()) (:447)
                           │           └─ 実 IR AudioBuffer データの転送（メタデータのみの applyBuildSnapshot を補完）
                           ├─ convolverRt().rebuildAllIRsSynchronous(isObsolete) (:954) ← NUC SoA 再構築
                           └─ コミット（publish）→ Audio Thread が crossfade で切替
```

**結論**: 実際の機構は **`captureBuildSnapshot → applyBuildSnapshot + transferIRStateFrom → rebuildAllIRsSynchronous`**。`rebuildAllIRsSynchronous` の記述は正確だが、`syncStateFrom` は置換済みの旧機構への言及。

### 12.4 他ファイルの syncStateFrom 参照の全数確認

| 場所 | 種別 | 判定 |
|---|---|---|
| `AudioEngine.Parameters.cpp:641` | コメント | **乖離（修正対象）** |
| `ConvolverProcessor.StateAndUI.cpp:389` | 関数定義 | デッドコード（§11 監視対象） |
| `ConvolverProcessor.h:499` | 宣言 | 同上 |
| `ConvoPeq.md:32243` ほか | 生成ドキュメント | 再抽出時に自動更新（`output_sourcecode_markdown.py`） |
| `doc/work21, work77, work80, mutable_code_reaudit` | 過去監査アーカイブ | 歴史的記録のため**変更しない** |
| `EQProcessor.h:697` コメント | EQ 側「Worker syncStateFrom() → this shadow」 | 別問題（§9 F-2 記載済み: 設計意図コメントで実装なし） |

### 12.5 コメント修正の実施案

**案 A（推奨・最小）: 641 行目を現実の機構に書き換え**

```cpp
// [Mem-Fix] NUC SoA (irFreqReal/irFreqImag) を再適用するため、uiConvolverProcessor を再構築する。
// DSPCore::convolver は次回 rebuild 時に captureBuildSnapshot → applyBuildSnapshot + transferIRStateFrom → rebuildAllIRsSynchronous で追従する。
```

- 変更 1 行のみ、機能影響ゼロ、他スレッド・API に一切影響なし。
- なお `setConvLCFilterMode`（:652-657）側にはそもそもこの追従コメントがなく非対称。LC 側へ同一コメントを追加するかは任意（推奨: 追加せず最小化）。

**案 B（推奨）: 案 A ＋ syncStateFrom 定義/宣言にデッドコード注記**

`StateAndUI.cpp:389`（および `ConvolverProcessor.h:499`）に関数冒頭で:

```cpp
// DEAD CODE（§11 監視対象）: 呼び出し元ゼロ。
// 代替は captureBuildSnapshot → applyBuildSnapshot + transferIRStateFrom。
void ConvolverProcessor::syncStateFrom(const ConvolverProcessor& other)
```

- 関数を読んだ将来の開発者が「代替機構」を迷わず見つけられる。
- `tools/dead_code_callers_verifier.py` が呼び出し再導入を引き続き防止（コメントは検出対象外のため相互干渉なし）。

**案 C: 削除**

§11 案 A（関数削除）を先に実施する場合、コメント修正は不要になる。ただし §11 の削除判断が保留の間は案 A/B が有効。

**推奨**: **案 A + 案 B をセットで実施**（変更 2 ファイル、コメントのみ）。§11 の関数削除（案 A）を実施する場合はその後にコメント削除も含めて一括整理。

### 12.6 検証計画

1. コメント修正後の grep 再確認: `grep -rn 'syncStateFrom' src/audioengine/` が Parameters.cpp:641 に一致しなくなること。
2. `python tools/dead_code_callers_verifier.py` が PASS のままであること（コメント変更は検出ロジックに影響しない）。
3. コメントのみの変更のためビルド・テストは不要（ただし念のため該当 TU のみコンパイル確認可）。
4. `ConvoPeq.md` は次回抽出時に自動追従（手動編集不要）。

### 12.7 実施完了記録（2026-08-08 追記）

案 A + 案 B + 対称化案（案 1）を適用済み。全てコメントのみの変更で機能影響ゼロ。

| 適用 | 対象 | 内容 | 行 |
|---|---|---|---|
| 案 A | `src/audioengine/AudioEngine.Parameters.cpp` | 乖離コメントを実機構に書き換え（`syncStateFrom + rebuildAllIRsSynchronous` → `captureBuildSnapshot → applyBuildSnapshot + transferIRStateFrom → rebuildAllIRsSynchronous`） | :641（`setConvHCFilterMode` 内） |
| 案 B | `src/convolver/ConvolverProcessor.StateAndUI.cpp` | `syncStateFrom` 定義の直前に DEAD CODE 注記（§11 監視対象・代替機構・監視スクリプト連携を明記） | :389 直前 |
| 案 B | `src/ConvolverProcessor.h` | `syncStateFrom` 宣言に DEAD CODE 注記 | :498-499 |
| 案 1 | `src/audioengine/AudioEngine.Parameters.cpp` | `setConvLCFilterMode` に HC と同一の追従機構コメントを追加（対称化。旧コメント `// HC と組み合わせて NUC を再構築` は HC と重複のため置換） | :653-657（`setConvLCFilterMode` 内） |

**検証結果**:

- `python tools/dead_code_callers_verifier.py` → `[PASS]`（DEAD CODE 注記追加は検出ロジックと干渉なし）。
- `grep -rn 'syncStateFrom' src/audioengine/` は Parameters.cpp に一致なし（コメント書換完了）。
- コメントのみの変更のためビルド・テスト不要。
- 追記したコメントは §12.5 の実機構チェーンと完全一致（HC/LC 同一経路）。

**差分サマリ**: 3 ファイル、+11/−2 行（`AudioEngine.Parameters.cpp` +7/−2、`ConvolverProcessor.StateAndUI.cpp` +4、`ConvolverProcessor.h` +2）。

---

## §13 AudioEngine.Parameters.cpp 全 setter の rebuild 追従調査と対称化提案（2026-08-08 追記）

### 13.1 調査目的

§12 で HC/LC フィルタモード setter の追従機構コメントを対称化した後、`src/audioengine/AudioEngine.Parameters.cpp` 全体（737 行）の**他の setter（`setEqLPFFilterMode` 等）にも rebuild 追従機構に関する乖離・不足コメントがないか**一括調査し、必要な対称化を提案する。

### 13.2 調査方法

1. ファイル内の全 setter を列挙し、各 setter の構造（`publishAtomic` / `submitRebuildIntent` / `uiConvolverProcessor.set*()` の有無）を精査。
2. 各パラメータの Audio Thread への伝達経路を判定:
   - **snapshot 経由のみ**（`captureBuildSnapshot` → `ProcessingState` → DSPCore の引数）→ **rebuild 必須**。
   - **Audio Thread が atomic を直読** → rebuild 不要（publish のみで正しい）。
3. UI 呼び出し元（EQControlPanel.cpp 等）で rebuild が別途トリガーされないか確認。
4. EQ 系 setter の正規パターン（`setEqBypassRequested` :153）と比較して欠落を判定。

### 13.3 分類結果（4 カテゴリ）

| カテゴリ | パターン | setter 一覧 | 追従コメントの要否 |
|---|---|---|---|
| **A: 直接 `submitRebuildIntent`** | コードで rebuild を明示的にトリガー | `setEqBypassRequested` / `setConvolverBypassRequested` / `setInputHeadroomDb` / `setOutputMakeupDb` / `setProcessingOrder` / `setConvolverInputTrimDb` / `setDitherBitDepth` / `setNoiseShaperType` / `setSoftClipEnabled` / `setSaturationAmount` / `setOversamplingFactor` / `setOversamplingType` | 不要（コードが明示） |
| **B: `uiConvolverProcessor.set*()` 経由** | 内部で `convolverParamsChanged` → `submitRebuildIntent()` | `setConvolverPhaseMode` / `setConvolverTargetIRLength` / `setConvolverMixedTransition*` / `setConvolverRebuildDebounceMs` / `setConvolverTail*` / `setConvolverStateTree` / `setConvolverTargetUpgradeFFTSize` / `setConvolverEnableProgressiveUpgrade` / `setConvolverMaxCacheEntries` / `clearConvolverCache` | 不要（585 行付近の一括ブロックコメントで説明済み） |
| **C: publish-only だが rebuild 不要** | Audio Thread が atomic を直読 | `setFixedNoiseLogIntervalMs` / `setFixedNoiseWindowSamples`（Timer.cpp:1054 で直読）/ `setAudioThreadPriorityMode`（Mmcss.cpp:89 / Timer.cpp:244 で直読） | 任意（直読箇所へのポインタがあると親切だが過剰） |
| **D: ⚠️ publish-only なのに rebuild が必要** | snapshot 経由のみで伝達（atomic 直読なし） | **`setEqLPFFilterMode`（:669）** | **要修正（実バグの疑い）** |

### 13.4 最重要発見: `setEqLPFFilterMode` の rebuild 欠落（実バグの疑い）

```
setEqLPFFilterMode(mode)  (:669)
  └─ publishAtomic(eqLPFFilterMode, ...) のみ  ← submitRebuildIntent なし！
```

検証で確定した事実（伝達チェーン）:

1. `eqLPFFilterMode` は **snapshot 経由でのみ** DSPCore に伝達される（AudioEngine.h:3733 / 3795 で `captureBuildSnapshot` → `ProcessingState::eqLPFMode` → DSPCoreFloat/Double の `outputFilter.process` の引数）。
2. **Audio Thread からの atomic 直読はゼロ**（DSPCoreFloat/Double は `state.eqLPFMode` のみ使用。`OutputFilter::process` も引数受取で atomic を読まない）。
3. UI ボタン（EQControlPanel.cpp:208/223/238）は `setEqLPFFilterMode` + `updateLPFModeButtons()`（UI 状態更新のみ）で **rebuild を一切トリガーしない**。
4. 一方、EQ 系の正規パターン `setEqBypassRequested`（:153）は `submitRebuildIntent(Structural, ...)` を呼ぶ。
5. プリセット復元経路（StateIO.cpp:151 → `endBulkParameterRestore(true)`）では rebuild が走るため救われるが、**UI ボタン単独操作では反映されない**。

**結論**: UI で LPF モード（Sharp / Natural / Soft）を変更しても、次の rebuild（別パラメータ変更・プリセットロード）まで **Audio Thread に反映されない**潜在バグ。

### 13.5 対称化提案

| 案 | 内容 | 判定 |
|---|---|---|
| **案 A（必須）** | `setEqLPFFilterMode` に `submitRebuildIntent(Structural, EnqueueSnapshotCommand, Snapshot, Replaceable)` を追加（`setEqBypassRequested` :153 と同パターン）— **実バグ修正** | 推奨 |
| **案 B（推奨）** | 併せて追従機構コメントを追加: `// DSPCore の OutputFilter は snapshot.eqLPFMode 経由で追従する (rebuild 必須、Audio Thread 直読なし)` | 推奨 |
| 案 C | カテゴリ C の publish-only setter に「atomic 直読のため rebuild 不要」注記を追加 | 任意（過剰なら見送り） |
| 案 D | カテゴリ B への個別コメント追加 | **不要**（一括コメントで済んでいる。HC/LC は [Mem-Fix] の特別な理由があるための例外） |

### 13.6 推奨と留意点

- **推奨**: 案 A + 案 B のセット実施（変更 1 ファイル、`setEqLPFFilterMode` のみ）。これはコメント対称化を超えた**機能修正（潜在バグ修正）**であり、動作検証（LPF モード切替の実音確認 or スナップショット反映確認）を伴う。
- **留意点**: 案 A の `submitRebuildIntent` 追加は rebuild 頻度を増やす可能性がある。ただし LPF モード切替はユーザー操作起因で低頻度であり、`setEqBypassRequested` と同一パターンのため過度な懸念は不要。
- カテゴリ C への注記（案 C）は「任意」— 直読箇所（Timer.cpp / Mmcss.cpp）の実装が変わるとコメントが再度乖離するため、優先度は低い。

---

> **★ 訂正（2026-08-08 §14）**: §13 の「実バグの疑い」は **誤り**。詳細は §14。§13.4 の結論（「次の rebuild まで反映されない潜在バグ」）は撤回する。**`setEqLPFFilterMode` は publish-only のままが正しい設計**であり、§13 案 A（submitRebuildIntent 追加）は**実施しないこと**。

---

## §14 setEqLPFFilterMode「rebuild 欠落」の検証とテスト手順（2026-08-08 追記）

### 14.1 検証の結論（§13 の結論を撤回）

§13 で「`eqLPFFilterMode` は snapshot 経由でのみ伝達（Audio Thread の atomic 直読ゼロ）」と結論したが、**コード精査の結果これは誤り**。`setEqLPFFilterMode` は **rebuild 不要で次オーディオブロックから即時反映される**。

**伝達チェーン（コード検証済み）**:

```
UI ボタン (EQControlPanel.cpp:208/223/238)
  → setEqLPFFilterMode(mode)            [AudioEngine.Parameters.cpp:672] publishAtomic のみ
  → 次オーディオブロック:
      getNextAudioBlock (AudioBlock.cpp:344) / processBlockDouble (BlockDouble.cpp:296)
      → captureAudioThreadParameterSnapshot(runtimeWorld)   [AudioEngine.h:3809]
          └─ :3854  snapshot.eqLPFMode = consumeAtomic(eqLPFFilterMode)  ← ★ per-block atomic 直読（GlobalSnapshot 版は :3792）
      → buildAudioThreadProcessingState (AudioEngine.h)
          └─ :3892  .eqLPFMode = snapshot.eqLPFMode
      → DSPCore::process (DSPCoreFloat.cpp:361 / DSPCoreDouble.cpp:461)
          └─ outputFilter.process(block, convIsLast, hc, lc, state.eqLPFMode)
      → OutputFilter::process (OutputFilter.cpp:199)
          └─ lpCoeff[lpIdx][0/1] テーブル参照（prepare で全モード事前計算済み）
```

**§13 が誤った理由**: §13 の調査は「AudioEngine.h:3792/3854 の `consumeAtomic` は rebuild 時の captureBuildSnapshot 由来」と想定したが、**:3854 は Audio Thread が毎ブロック呼ぶ `captureAudioThreadParameterSnapshot`（world 版）の実装**であり、`eqLPFMode` は world の有無に関わらず**無条件で** atomic 直読される（routing/automation 由来の他フィールドとは異なり、eqLPFMode は atomic が真実源）。

**なぜ HC/LC と非対称か**: `convHCFilterMode` / `convLCFilterMode` も同じ関数で atomic 直読され OutputFilter 係数は即時切替するが、HC/LC には **NUC SoA（irFreqReal/irFreqImag）の再適用**という convolver 内部の構造変更が伴うため、[Mem-Fix] の rebuild が必要。**EQ LPF は OutputFilter の lpCoeff テーブル参照のみ**で、NUC 相当の内部状態がないため、rebuild 不要が正しい設計。

### 14.2 レベル 1: 静的検証（grep で経路を証明、ビルド不要）

```bash
# ① per-block atomic 直読の存在
grep -n "eqLPFMode = consumeAtomic" src/audioengine/AudioEngine.h
#   期待: 2 箇所 — 2026-08-12 時点の実測は :3792（GlobalSnapshot 版 captureAudioThreadParameterSnapshot）/
#          :3854（RuntimePublishWorld 版）。§14 執筆時の :3733/:3795 は行番号ドリフト
#          （capture 関数定義 :3775/:3809、buildAudioThreadProcessingState の .eqLPFMode 代入 :3892）

# ② Audio Thread が毎ブロック captureAudioThreadParameterSnapshot を呼ぶこと
rg -n "captureAudioThreadParameterSnapshot\\(runtimeWorld" src/audioengine/AudioEngine.Processing.AudioBlock.cpp src/audioengine/AudioEngine.Processing.BlockDouble.cpp
#   期待: AudioBlock.cpp:344 / BlockDouble.cpp:296

# ③ ProcessingState への受け渡し（AudioEngine.h:3833）
grep -n "eqLPFMode = snapshot.eqLPFMode" src/audioengine/AudioEngine.h

# ④ OutputFilter への引数受け渡し
rg -n "state.eqLPFMode" src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp
#   期待: :361 / :461

# ⑤ OutputFilter が prepare で全モード係数を事前計算（再計算不要）
grep -n "lpCoeff\\[" src/OutputFilter.cpp
#   期待: prepare(): Sharp/Natural/Soft の 3 モード × 2 段を事前計算
```

**判定**: ①〜⑤ が全て確認できれば、「publish-only で即時反映される」設計が静的根拠をもって証明される。§13 の案 A は実施しない。

### 14.3 レベル 2: 統合テスト（AudioEngineHarness、rebuild 世代 + 出力ハッシュ検証）

**目的**: 実行時に (a) `setEqLPFFilterMode` が **rebuild を発生させない**こと、(b) それでも **Audio Thread の出力が変化**すること、の 2 点を実証する。

**テスト 2-1: rebuild 非発生の証明（世代カウンタ）**

```cpp
AudioEngineHarness h;
h.start(48000.0, 512);
AudioEngine& e = h.engine();

// 起動 rebuild が落ち着くまで待機（generation が安定するまで）
const auto* w0 = e.observePublishedWorld();
const uint64_t gen0 = w0 ? w0->generation : 0;
const uint64_t seq0 = w0 ? w0->publication.sequenceId : 0;

// 対象 setter（rebuild を期待しない）
e.setEqLPFFilterMode(convo::HCMode::Sharp);
std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 仮に rebuild があれば到達する時間

const auto* w1 = e.observePublishedWorld();
assert(w1->generation == gen0);        // ★ rebuild 非発生
assert(w1->publication.sequenceId == seq0);
assert(e.getEqLPFFilterMode() == convo::HCMode::Sharp); // atomic は更新済み

// 対照群: rebuild を期待する setter（setEqBypassRequested）で検証系が正しいことを確認
const uint64_t genBefore = e.observePublishedWorld()->generation;
e.setEqBypassRequested(!e.isEqBypassRequested());
// waitUntil: generation が増加することを確認 → 観測系の正当性が証明される
```

**期待結果**: `setEqLPFFilterMode` では generation / sequenceId が不変（FAIL メッセージ: 「unexpected rebuild」）、対照群では増加（観測系の健全性）。

**テスト 2-2: 出力変化の証明（テストトーン + 出力ハッシュ）**

前提: ハーネスの `audioLoop` は現在ゼロ入力を回すため、**入力信号注入の拡張が必要**。推奨は AudioEngineHarness に「テストトーン（freq, amp）と出力キャプチャ（出力バッファのハッシュ/振幅累積）」を追加する方式。

```cpp
// ハーネス拡張案（設計のみ、実装は別途）:
//   setInputTone(double freqHz, double amp)  — audioLoop が毎ブロック sin を書き込む
//   getOutputRms() / getOutputHash()         — 出力バッファから統計を取る

AudioEngineHarness h;
h.start(48000.0, 512);       // fc_lp = 19kHz (≤48kHz)
h.setInputTone(18000.0, 0.1); // カットオフ近傍トーン: Q 差が減衰量に現れる

const double rmsNatural = h.getOutputRms();          // 既定 Natural
h.engine().setEqLPFFilterMode(convo::HCMode::Soft);  // Q=0.5×2 は減衰が緩い
std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 数ブロック待機
const double rmsSoft = h.getOutputRms();

// 期待: |rmsSoft − rmsNatural| が有意差（例: >0.01×rmsNatural）
//       ※ rebuild なし（2-1 で証明済み）で出力が変わる = 即時反映の決定的証拠
```

**期待結果**: 出力 RMS/ハッシュがモード間で有意に変化。変化しない場合は LPF 反映経路の欠陥（想定外）を意味するため要調査。

**注意**: トーン周波数はサンプルレートに依存（48kHz 以下 → fc_lp=19kHz、それ以上 → 24kHz）。テスト時は SR に対応する fc の **0.8〜0.95 倍**を選ぶ（例: 48kHz→18kHz、96kHz→22kHz）。

### 14.4 レベル 3: 実機 UI 検証（手動）

1. ConvoPeq を起動し、EQ 最終段構成（EQ 単体 or Convolver→EQ）でオーディオ再生（18kHz 付近のテストトーン推奨: `sampledata/` の test_music でも可）。
2. EQ パネルの HCF（LPF）ボタン Sharp / Natural / Soft をクリック。
3. **即時**（1 オーディオブロック以内、体感 <50ms）に高域の減衰特性が変わることを聴感・スペアナで確認。
4. ログで rebuild が発生していないことを確認（`[REBUILD_TELEMETRY] event=REBUILD_REQUESTED` が出ないこと。CLI telemetry / Debug ログで確認可）。
5. 対照: 他のパラメータ（例: Total Gain）変更時は rebuild ログが出ること。

**期待結果**: LPF ボタン切替が rebuild なしで即時反映（聴感・スペアナ・ログの 3 点で確認）。

### 14.5 実施タイミングと §13 への反映

- **§13 案 A（submitRebuildIntent 追加）は実施しない**（誤った修正であり、無駄な rebuild を増やすだけ）。
- §13 案 B（コメント）も誤った前提（「rebuild 必須・直読なし」）に基づくため、**書かない**。代わりに必要なら `setEqLPFFilterMode` に正しい注記（「OutputFilter は lpCoeff テーブル参照のため rebuild 不要・per-block atomic 直読で即時反映」）を追加できる。
- カテゴリ D の分類表（§13.3）は「D から削除（A〜C のいずれかでなく『publish-only で正しい』）」。
- 検証はレベル 1（静的）を即時実施可。レベル 2 はハーネス拡張（入力トーン + 出力キャプチャ）を伴うため、既存の `PublishPipelineIntegrationTests` に `runEqLpfModePropagationTests()` として追加するのが自然。レベル 3 はリリース前の手動確認として推奨。
- **文書保守**: §13.3 表の D 行と §13.4 の結論には「§14 により撤回」の注記を追加済み（本セクション冒頭）。

### 13.7 実施完了記録（2026-08-08 追記）

案 C（カテゴリ C の publish-only setter への注記追加）を適用済み。コメントのみの変更で機能影響ゼロ。

| 適用 | 対象 | 内容 | 行 |
|---|---|---|---|
| 案 C | `src/audioengine/AudioEngine.Parameters.cpp` — `setFixedNoiseLogIntervalMs` | `// publish-only: Audio Thread は atomic 直読（AudioEngine.Timer.cpp:1058/1089）のため rebuild 不要。` | :477 |
| 案 C | `src/audioengine/AudioEngine.Parameters.cpp` — `setFixedNoiseWindowSamples` | `// publish-only: Audio Thread は atomic 直読（AudioEngine.Timer.cpp:1054/1085）のため rebuild 不要。` | :488 |
| 案 C | `src/audioengine/AudioEngine.Parameters.cpp` — `setAudioThreadPriorityMode` | `// publish-only: 消費側は atomic 直読（AudioEngine.Mmcss.cpp:89 / AudioEngine.Timer.cpp:244/317/347）のため rebuild 不要。` | :511 |

**検証結果**:

- `python tools/dead_code_callers_verifier.py` → `[PASS]`。
- 直読箇所の行番号は grep で照合済み（Timer.cpp:1054/1058/1085/1089/244/317/347、Mmcss.cpp:89）。
- コメントのみの変更のためビルド・テスト不要。
- **注**: §13 案 A（setEqLPFFilterMode への submitRebuildIntent 追加）と案 B（同コメント）は §14 の結論により**実施しない**（publish-only が正しい設計）。

---

## §15 再検証（別視点調査）結果（追記: 2026-08-12）

> ユーザー依頼「ソースコードを別の視点から可能な限り詳細に調査・検証し、不適切な個所は修正」への対応として、
> 本リスト全章（§1〜§14）の主張を **現行ソース（HEAD 3198acc, 2026-08-12）** と 1:1 で再照合した。
> **結論: 全 19 バグの「修正済み」判定は全て妥当**。同時に、不正確な記述の修正（§3/§5/§8.1/§14 に反映済み）、
> 未確定事項の確定、新規所見を以下に記録する。

### 15.1 検証方法と使用ツール

| ツール | 用途 | 本検証での実績 |
|--------|------|---------------|
| WSL rg / sed / awk / fd / fzf / ag | テキスト検索・行番号照合 | `★ BUG-0XX` 全マーカーの所在、各修正位置、呼び出し元ゼロ（デッドコード）の全数確認 |
| WSL ast-grep | AST 構造検索 | `fetchAddAtomic(agcResetSerial, ...)` の全出現（Core.cpp:276/:787 が BUG-065 修正箇所）を構造一致で確認 |
| cocoindex code（`ccc.exe`） | セマンティック検索 | fading slot の CAS 化をクロス確認（Timer.cpp:1018 / DSPTransition.h:139 / CtorDtor.cpp:136 / ReleaseResources.cpp:140） |
| semble（`semble.exe`） | セマンティック検索 | 同上の結果を独立確認（DSPTransition.h / Timer.cpp / AudioEngine.h をヒット） |
| graphify（`graphify.exe`） | ナレッジグラフ | `EQProcessor::reset()` ノード（Core.cpp:259, degree 3）と接続（loadCurrentState/storeTotalGainDb）を確認 |
| serena / AiDex MCP | コード検索 | **本セッションには MCP アクセスが無く、CLI は MCP サーバー起動のみで検索結果を返さない**（aidex は "AiDex MCP server started" 出力のみ）。コード検索は rg / ast-grep / ccc / semble / graphify で代替し、rg の全数検索で網羅性を担保 |
| git log（-S / --stat） | 履歴検証 | 修正導入コミット 444c2f3 の特定、e208c26 / 8c9af92 / 06ad145 の存在確認、resetForRuntime 導入コミットの確認 |
| ビルド検証 | コンパイル確認 | `tools/build-check.log`（2026-08-11 16:15）: **Debug 構成 166/166 ターゲット成功**。HEAD の src/ はビルド時点（c8ca439）から変更なし |

### 15.2 全 19 バグの再照合結果（2026-08-12 時点の行番号つき）

| ID | 確認位置（現行ソース） | 修正内容の再確認 | 判定 |
|----|----------------------|----------------|------|
| BUG-047 | EQProcessor.h:242 / ProcessingCache.cpp:23-50（★ :46）/ AudioEngine.Cache.cpp:51 | signature 拡張＋srBits/maxBlockSize を hashCombine。`getOrCreate` は新 signature で呼出 | ✅ |
| BUG-048 | EpochDomain.h:434-512（★ :448） | 3 パス評価。※ Chronic/Warning は `pendingRetireCount>0` が AND 条件（:461/:466、N-2） | ✅ |
| BUG-049 | EpochDomain.h:255-316（★ :267、exitReader :169） | CAS 遷移のみ（0x00→0x02 / 0x00→0x01 / 0x02→0x01）。0x03 生成なし | ✅ |
| BUG-050 | EpochDomain.h:107-132（★ :120） | epoch 先行 store（release）→ depth++（acq_rel）。ネスト時は return | ✅ |
| BUG-051 | CtorDtor.cpp:132 / ReleaseResources.cpp:136,159 | sentinel `(uintptr_t)-1` チェック削除（残存はコメントのみ） | ✅ |
| BUG-052 | RuntimePublicationOrchestrator.h:47-112, 158-167 | DeferredPublishView（move-only・finishView 委譲）＋ `consumeAtomic(hasDeferred_)`。consumeDeferredRequest はコメントのみ残存 | ✅ |
| BUG-053 | AudioEngine.Learning.cpp:55-63 | 直接 `stopLearning()` 削除（キュー経由に一本化）。stopLearning は joinable ガードで冪等（NoiseShaperLearner.cpp:185-195） | ✅ |
| BUG-054 | DSPTransition.h:89-114 | oldHandle 引数化・fadingHandle 改名・CAS claim（**※ N-1: Emergency Override パスの exchange は残存**） | ✅ |
| BUG-055 | AudioEngine.Commit.cpp:221-223 | 到達不能 else if 削除。後続 `hasTransitionNext` 導出は正常 | ✅ |
| BUG-056 | RuntimeHealthMonitor.cpp:534-537, 563-567, 1217 | Normal 復帰パス追加（isPending()==false / delta<warning）+ reset() で Normal 化 | ✅ |
| BUG-057 | RuntimeHealthMonitor.h:60-61 | EVENT_OVERFLOW_RATE_WARNING=1012 / CRITICAL=1013 新設（.cpp:811/834/853 で使用） | ✅ |
| BUG-058 | RuntimeHealthMonitor.h:63, 372 / .cpp:953-987 | 専用 state（m_prevWorldConsistencyState_）＋ event code 7000/7001/7002。3 状態遷移 | ✅ |
| BUG-059 | RuntimeHealthMonitor.cpp:1209-1249 | 全 MonitorState を Normal 統一リセット | ✅ |
| BUG-060 | ISRRetireRuntimeEx.cpp:209-224, 236-238 | fetchSub 単一アトミック＋previous==0 回復。quarantine 側は previousLane!=Quarantine 時のみ fetchAdd。laneOf/quarantineResidentCount_ は atomic | ✅ |
| BUG-061 | ISRDSPQuarantine.cpp:35,84,118,157,176 / .h:64,83 | mutex 全アクセス保護。ロック保持済み internal 版（compactAuditLogLocked）と public 版を分離 | ✅ |
| BUG-062 | RuntimeHealthMonitor.cpp:897-901 | uint64_t 版にも EVENT_RETIRE_AGE_NORMAL 復帰パス（double 版と整合） | ✅ |
| BUG-063 | EpochDomain.h:74-76, 498-505, 546 | `ownerThreadId`（std::atomic<uint64_t>）を acquire 読取、非 0 時のみ ownerTag コピー。register は CAS 排他下で release publish | ✅ |
| BUG-064 | DSPCoreIO.cpp:521-527（double 版 DSPCoreDouble.cpp:733-739 と同一順序） | float パスを clamp→delay に統一（delay buffer には clamp 済み値のみ） | ✅ |
| BUG-065 | EQProcessor.Core.cpp:276, 787（reset / prepareToPlay） | `fetchAddAtomic(agcResetSerial, 1)`（increment）。**rt シャドウ書込 3 行は残存**（§7 案 A 未適用 — D-1） | ✅ |

### 15.3 新規所見（本検証で判明した不正確・未確定事項）

**N-1: `exchangeFadingRuntimeDSP` は「廃止」されておらず、Emergency Override パスに残存**（§3 BUG-054 の記述を修正済み）

- 定義: AudioEngine.h:2092。**呼び出し元は DSPTransition.h:63-65 の Emergency Override パスのみ**（`// ★ Temporary: exchangeFadingRuntimeDSP (A-6 fix). Will be removed after B-1 CAS-only claimFadingRuntimeDSP().` コメント付き）。
- 通常の crossfade / Timer / shutdown の fading slot クリアは全て CAS-based（Timer.cpp:1018 / DSPTransition.h:139 / CtorDtor.cpp:136 / ReleaseResources.cpp:140 / `claimFadingRuntimeDSP` AudioEngine.h:2100）。
- **※ 2026-08-12 第2パスで訂正（「CAS-only 化が望ましい」は誤り）**: serena MCP 検索と git 履歴（`doc/work88/REPAIR_PLAN.md:1862/2383`、`doc/work88/big_bug/INTEGRATED_BUG_LIST.md:489 R-20`）により、**この exchange は work88 BUG-029 修正で意図的に追加された設計**であることを確認。Emergency Override は「fading slot が既に別 DSP を保持していても新 DSP を displace し、戻り値 `prevRaw`（旧占有者）を retire する」必要があり、`exchange`（任意値 swap）はその displacement セマンティクスを単一原子操作で実現する。**`claimFadingRuntimeDSP`（CAS nullptr→oldDSP）に置換すると占有中は CAS 失敗となり、prevRaw の取得・retire ができず BUG-029 が再発する**ため、置換してはならない。
- したがって本項は「未完了事項」ではなく、**「work88 BUG-029 の正しい修正が残存している状態」**。`★ Temporary` コメントは「B-1 CAS-only 化完了後に撤去予定」という将来の設計意図の注記であり、現状のまま正しい。**残存リスクなし**（sentinel 問題は既に解消済み）。

**N-2: BUG-048 の Chronic/Warning 条件には `pendingRetireCount > 0` が含まれる**（§3 の記述を修正済み）

- EpochDomain.h:461（Chronic）/ :466（Warning）。§3 の「Pass3=Chronic >30s、Pass2=Warning >10s」は条件の一部を省略しており、**pendingRetire==0 の期間は Pass1（EpochGap）のみが検出可能**。

**N-3: BUG-049 の「depth>0 時は pending を OR」は不正確**（§3 の記述を修正済み）

- 実装は CAS（0x00→0x02）であり OR ではない（EpochDomain.h:308-315）。「0x03 を生成しない」という結論は正しい。

**N-4: 全 19 バグの修正は単一コミット 444c2f3（2026-08-05）で一括導入**

- `git log -S 'BUG-0XX' -- src/` の全数確認で、全マーカーが commit 444c2f3（2026-08-05, 34 ファイル, +1569/−519）で追加されたことを確認。§5 の「2026-07-30〜08-05 のコミット群」は範囲として正しいが、単一コミットに特定可能。

**N-5: ビルド検証（§5 の制約の一部解消）**

- `tools/build-check.log`（2026-08-11 16:15）: Debug 構成 **166/166 ターゲット成功**（AudioEngineHarness.exe・各テスト exe 含む）。HEAD（3198acc）の src/ はビルド時点（c8ca439）から変更なしのため、**現行ソースはコンパイル検証済み**。

**N-6: 行番号ドリフト**（§7〜§14 の参照行番号の一部）

- 最大: §14.2 ① `eqLPFMode` consumeAtomic — doc 記載 :3733/:3795 → 実際 **:3792/:3854**（capture 関数定義 :3775/:3809、`.eqLPFMode` 代入 :3892）。**機能主張（publish-only で即時反映）は変わらず正しい**（§14.1 の結論は維持）。
- その他: `DSPCore::prepare()` 定義は DSPCoreLifecycle.cpp:72（:200/:255 は内部の eqState->prepare() 呼び出し）、`rebuildThreadLoop()` 定義は RebuildDispatch.cpp:804（RuntimeBuilder::build() 使用は :905 付近）、`setEqLPFFilterMode` は Parameters.cpp:672、`ConvolverProcessor::syncStateFrom` 定義は StateAndUI.cpp:393（DEAD CODE 注記 :389）、publish-only 注記は Parameters.cpp:478/489/512 等。**§15.2 の表を正とする**。

**N-7: serena / AiDex MCP — 第1パスでは使用不可だったが、第2パスで stdio JSON-RPC により実使用に成功**

- **第1パス（§15 執筆時）**: MCP アクセスが無く、CLI はサーバー起動のみで検索結果を返さなかった。
- **第2パス（2026-08-12）**: `tools/mcp_query.py` / `tools/mcp_list_tools.py`（本セッションで作成）により、MCP stdio サーバーへ JSON-RPC（initialize → tools/call）で直接アクセスし、以下を実行・確認した。
  - **serena**: `activate_project(ConvoPeq)` → `search_for_pattern("exchangeFadingRuntimeDSP")`。work88 の BUG-029 修正記録（REPAIR_PLAN.md / big_bug/INTEGRATED_BUG_LIST.md R-20）をヒットさせ、N-1 の訂正根拠を取得。memories 一覧（task/consolidate-bug-list 等）も取得。
  - **AiDex**: `aidex_query`（既存インデックス .aidex/index.db を使用）で `fetchAddAtomic`（197 件）、`agcResetSerial`（Core.cpp:276/279/787/790 ほか）、`ownerThreadId`（EpochDomain.h:74/498-500/546）、`m_prevWorldConsistencyState_`（RuntimeHealthMonitor.cpp:953/969/1237, .h:372）を取得。**doc の行番号主張（BUG-058/063/065）と完全一致**。
- インデックス整備: AiDex は .aidex/index.db が既存で使用可能（整備済み）。serena は .serena/ が既存（プロジェクト登録済み）。

**N-8: 類推探索（同種バグの他箇所）— バグなしを確認**

| 探索対象 | BUG-XXX との対応 | 結果 |
|---------|-----------------|------|
| 他キャッシュのハッシュ: `CacheManager::computeKey`（convolver IR キャッシュ） | BUG-047（sampleRate 欠落） | **srBits を hash に含む**（CacheManager.cpp:95-98、fftSize/phaseMode/partitionSize と併せて）。同種バグなし |
| `fetchSubAtomic` 全 18 使用箇所 | BUG-060（TOCTOU underflow） | RefCountedDeferred（==1 チェック）、EpochDomain/DspNumericPolicy（depth）、WorldLifecycleAudit.h:51（prev==0 アサート＋fetchAdd 補正）、ProcessIntent.cpp:30-95（reservation 不変条件、work88 で絶対値リセット廃止済み）。**同種 underflow なし** |
| `getMinReaderEpoch` の quarantine 除外 | BUG-049（不変条件） | EpochDomain.h:207-241 で quarantined reader を safe-epoch 計算から除外（depth==0 の防衛的アサート付き）。設計と整合 |
| §13.3 setter 分類の全数再検証 | §13（rebuild 追従） | カテゴリ A 12 件（直接 submitRebuildIntent、setEqBypassRequested :153 含む）/ B 8 件（uiConvolverProcessor 経由）/ C 3 件（publish-only）。**分類は全て正しい** |
| `checkRetireReclaimLatency` の 2 パス | BUG-062（uint64 版 Normal 復帰） | double 版（:864-883、Work38 済み）＋ uint64 版（:884-901、★ BUG-062 済み）の両方が Normal 復帰パスを持つ。**記述は正確** |

**N-9: `★ BUG-XXX` マーカー表記の不整合（表記のみ・内容影響なし）**

- 19 バグ中 15 件は `★ BUG-XXX` 表記だが、**BUG-060 / BUG-063 / BUG-064 / BUG-065 の 4 件はコメントが「BUG-XXX」（★ なし）**（ISRRetireRuntimeEx.cpp:217 / EpochDomain.h:498 / DSPCoreIO.cpp:521 / EQProcessor.Core.cpp:276,787）。
- §1/§2/§6 の「全バグに `★ BUG-XXX` コメントが残存」という記述は厳密には 15/19 のみ ★ 付き。**表記揺れであり修正実装の有無に影響しない**（4 件はコード変更そのものが存在）。

### 15.4 未確定事項の確定結果（要調査・棚卸し・保留 → 確定）

| 項目 | 保留時の内容 | 確定結果（2026-08-12） |
|------|------------|----------------------|
| D-1 | §7 BUG-065 改修（案 A/B/C の選択） | **案 A（rt シャドウ書込の全廃 + bandResetPacked serial 前進クリア）を確定・推奨**。根拠: (a) `EQProcessor::reset()` はデッドコード（§8.2 再確認: 呼び出し元ゼロ）、(b) `prepareToPlay()` は Rebuild Thread 経由で実動するが新規 DSPCore への publish 前呼び出しに限定（§8.1）、(c) 案 A は `resetToDefaults()`（Core.cpp:206-255）の既存安全パターンと完全一致、(d) Audio Thread 側（Processing.cpp:586-593 / 1070-1077）は無変更で済む。**実装はソース変更を伴うため別作業として実施**（本 doc 更新は調査・確定記録のみ。§7.8 の手順・§7.7 の検証計画を適用）。 |
| D-2 | §10 AGC atomic 3 つ（agcCurrentGain/agcEnvInput/agcEnvOutput）の削除 | **案 A（削除）を確定・推奨**。前提の再検証: atomic 3 つは Audio Thread から読まれない（processAGC は rtAgc*Shadow のみ使用 — Processing.cpp:419-421/438-440。atomic 参照は係数 agcAttack/Release/SmoothCoeff の 3 つのみ :369-371）。削除時は §10.4-3 の参照ゼロ確認（grep）を実施。**実装は別作業**（D-1 と同一ファイルのため併せて実施推奨）。 |
| D-3 | §11 デッドコード 6 関数の整理（削除 or 予約コメント化） | **「監視（verifier）下で維持」を確定**。`tools/dead_code_callers_verifier.py` は実装済みで **再実行 [PASS]**（2026-08-12 確認）。関数削除（§11 案 A）は `DSPCore::reset()` の子リセット（ramps/histories 等）が将来の DSPCore 再利用で必要になり得るため即時削除せず、監視下で維持。`[[deprecated]]` 化（案 B）は任意・低優先。削除の実施判断は work89 完了後に再評価。 |
| D-4 | §14 レベル 2 統合テスト（runEqLpfModePropagationTests） | **未実装であることを確認**（src/tests/ に該当テストなし）。レベル 1 静的検証は本セッションで全 5 項目を再実施し **PASS**（① eqLPFMode 直読 :3792/:3854、② capture 呼出 AudioBlock.cpp:344 / BlockDouble.cpp:296、③ `.eqLPFMode` 代入 :3892、④ state.eqLPFMode DSPCoreFloat.cpp:361 / DSPCoreDouble.cpp:461、⑤ lpCoeff 事前計算 OutputFilter.cpp:107-117）。レベル 2/3 は別途実施（§14.3/14.4 の手順どおり）。 |
| D-5 | §4 重複ファイル（BUG-047-EQ...md）の扱い | **相互リンク化を推奨（管理判断）**。削除せず、重複である旨の相互参照を追加する運用を確定（bug ファイル自体の変更は本 doc のスコープ外）。 |
| D-6 | §12.7 / §13.7「適用済み」の検証 | **適用済みであることを実査確認**。Parameters.cpp:640-661（HC/LC 追従コメント）、ConvolverProcessor.StateAndUI.cpp:389-393 / ConvolverProcessor.h:498-501（DEAD CODE 注記）、Parameters.cpp:478/489/512（publish-only 注記）。`dead_code_callers_verifier.py` [PASS]。 |

### 15.5 結論（2026-08-12）

1. **§1〜§6 の核心的結論は全て妥当**: 全 19 ユニークバグは現行ソース上で修正済みであり、`★ BUG-XXX` コメントは HEAD（3198acc）に残存する。未修正バグゼロの主張は維持される（※ N-9: 4 件は ★ なし表記）。
2. **修正した不正確な記述は 6 箇所**: §2 BUG-047（行番号）、§3 BUG-048（条件省略）、§3 BUG-049（OR 誤記）、§3 BUG-054（exchange 廃止誤記）、§5（コミット特定・ビルド検証）、§8.1 / §14（行番号ドリフト）。
3. **未完了事項は実質ゼロ**: 第1パスで唯一の未完了事項とした N-1（Emergency Override の exchange 残存）は、第2パスで **work88 BUG-029 の意図的な修正設計（displacement セマンティクス）** であることを確定し、CAS-only 化の推奨を撤回した（N-1 参照）。
4. **未確定事項 D-1〜D-6 は全て確定**し、§15.4 の表に反映した。ソース変更を伴う D-1 / D-2 は「実施手順確定・実装は別作業」とした。

---

## §16 第2パス再検証（MCP ツール実使用・類推探索）結果（追記: 2026-08-12）

> ユーザー再依頼に対応し、第2パスとして (a) serena / AiDex MCP の実使用、(b) 未検証だった doc 主張の追加検証、(c) 同種バグの類推探索、(d) 技術前提の文献検証を実施した。
> **結論: §15 の検証結果に変更なし。新たなバグ・不正確な記述は 0 件。**

### 16.1 serena / AiDex MCP の実使用（N-7 更新）

- `tools/mcp_query.py` / `tools/mcp_list_tools.py`（本セッションで新規作成）により、MCP stdio サーバーを JSON-RPC で直接駆動。
- **serena**: `activate_project` → `search_for_pattern`。work88 BUG-029 修正記録を取得し、N-1 の訂正根拠となった。
- **AiDex**: `aidex_query`（既存インデックス使用）で BUG-058/063/065 関連行を取得し、doc の行番号主張と完全一致を確認。
- 補足: 本セッションのエージェントツールセットには MCP ツール（serena/AiDex）への直接アクセスが無いため、stdio JSON-RPC ブリッジで代替した（手法の詳細は tools/ のヘルパーに記載）。

### 16.2 追加検証結果

1. **§13.3 分類表の全数再検証（N-8）**: カテゴリ A 12 件・B 8 件・C 3 件を全数確認。`setEqBypassRequested`（:153）・`setConvolverBypassRequested`（:164）は直接 `submitRebuildIntent` 呼び出し（A 分類どおり）。カテゴリ C の 3 setter（:476/:487/:510）は publish-only で `// publish-only:` 注記付き（§13.7 適用済み）。**分類は全て正しい**。
2. **BUG-062 の 2 パス確認**: `checkRetireReclaimLatency`（RuntimeHealthMonitor.cpp:861）は double 版（:864-883、Work38）と uint64 版（:884-901、★ BUG-062）の両方が Normal 復帰パスを持つ。doc の「uint64_t 版にも復帰イベント追加」は正確。
3. **getMinReaderEpoch の quarantine 除外（BUG-049 整合）**: EpochDomain.h:207-241 で quarantined reader を safe-epoch 計算から除外（`continue`）し、防衛的 assert（depth==0）を保持。BUG-049 の「quarantined ⇒ depth==0」不変条件と整合。
4. **類推探索（N-8）**: CacheManager::computeKey は srBits を hash に含む（BUG-047 同種なし）、fetchSubAtomic 全 18 箇所は underflow ガード済み（BUG-060 同種なし）、sentinel `(uintptr_t)-1` はソースに残存なし（コメントのみ、BUG-051 同種なし）。

### 16.3 技術前提の文献検証

- **JUCE prepareToPlay スレッド契約（§8.1 の前提）**: JUCE 公式ドキュメント（docs.juce.com, `juce::AudioProcessor::prepareToPlay`）で「Called before playback starts, to let the processor prepare itself」を確認。ConvoPeq ソースの PrepareToPlay.cpp:219 に「JUCE 契約上 prepareToPlay 実行中は Audio Thread callback が走らないため、ここでの状態公開は安全」というコメントが実在し、doc §8.1 経路 A の前提と一致。
- **BUG-050 のメモリ順序（release/acquire + RMW）**: enterReader の「epoch store（release）→ depth fetchAdd（acq_rel）」は、getMinReaderEpoch 側の「depth load（acquire）→ epoch load（acquire）」と release-acquire synchronizes-with を形成し、`depth>0` 観測後の epoch load が先行する epoch store を必ず観測する（C++ 標準メモリモデル上で正当）。ネスト enter（previousDepth>0 で return）でも外側の epoch store が release チェーンで可視。doc §3 BUG-050 の記述は正しい。

### 16.4 第2パスの結論

1. 第1パス（§15）の検証結果・修正内容に**変更を要する新事実は無し**。
2. **N-1 のみ訂正**: Emergency Override の exchange は work88 BUG-029 の意図的設計であり、「CAS-only 化が望ましい」とする第1パスの推奨を撤回（N-1 に反映済み）。
3. 未完了事項・要調査事項は**ゼロ**に確定。D-1〜D-6 の「実装は別作業」判断は維持。
4. 使用ツール: serena / AiDex MCP（stdio JSON-RPC）、WSL rg/ast-grep/fd/fzf/ag/sed/awk、ccc / semble / graphify、git log、web 検索（JUCE ドキュメント）。
