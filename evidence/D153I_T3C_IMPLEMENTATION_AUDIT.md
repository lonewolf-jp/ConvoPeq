# D153-I — T3c Implementation Audit (post-implementation, read-only)

```text
Production source changes: 本ターンで T3c 実装を適用（下記 file 一覧）
Test source changes:       NT-1..5 追加 + main 登録 + include 2 件
CMake changes:             0
Build:                     NOT RUN（指示どおり audit まで）
CTest:                     NOT RUN

Source baseline:
  実装前 ConvoPeq.md = Generated: 2026-08-31 15:32:58 / newer src 0（実測）
  実装後 worktree: 6 files 変更（うち RebuildDispatch/Threading/tests はセッション開始時既存の
    G-4.x/F6 変更を含む。T3c 差分は下表の追加行。Orchestrator.cpp は本実装で初変更=純 T3c）
```

## 0. 適用した変更（D152 + D152-R1 準拠）

| file | T3c 変更 |
|---|---|
| ISRRuntimePublicationCoordinator.h | `RecoveryLifecycleWord`（alignas(16)+static_assert 5 点・is_always_lock_free assert なし）/ `LogicalRecoveryObligation` = `std::atomic<W> lifecycle` + plain payload（pragma 4324 囲り）/ table ctor（runtime `is_lock_free()` 検証・MSVC `#if` 前例踏襲）/ findByKey・tryInsert（payload 5 引数化・payload→CAS→liveCount++）・resolve（全文 CAS 統合 + discardedPendingOut）・adjudicatedFailureCount / 宣言: `postRecoveryFailureSignal`・`adjudicateRecoveryFailureSignals`・TEST-ONLY `markTransientFailure`・`peekLifecycleForTest`・`resolveRecoveryObligation`→bool・私有 `casDelivery` / telemetry getter 4 + member 4 / delivery 注釈 W 化 |
| ISRRuntimePublicationCoordinator.cpp | wasDeferredBefore・coalesce（T5 再試行ループ）・COALESCE oblId/delivery 読・tryInsert 呼び出し（payload 移動）・attach×3→casDelivery / resolveRecoveryObligation bool 化 + droppedTerminal 計上 / 旧 markT 本体→**T1 postSignal + T2 adjudicate + TEST-ONLY ラッパ + peek** / redrive scan/gate→lifecycle snapshot・attach→casDelivery / shutdown close id 読→lifecycle / casDelivery 定義 |
| AudioEngine.RebuildDispatch.cpp | 4 サイト（:1006/:1033/:1091/:1115）→ `postRecoveryFailureSignal` |
| RuntimePublicationOrchestrator.cpp | 2 サイト（:311/:401）→ `postRecoveryFailureSignal` |
| AudioEngine.Threading.cpp | runCoordinatorPhase: processIntent 直後・redrive 直前に `adjudicateRecoveryFailureSignals()` 挿入（:272） |
| ISRSemanticValidationTests.cpp | NT-1..5（分割原語を直接使用・ラッパ不使用）+ main 登録 + `<chrono>`/`<atomic>` |

## 1. V1-V10 機械検証（実測）

| V | 条件 | target | 実測 | 判定 |
|---|---|---|---|---|
| V1 | production `markTransientFailure(` 呼び出し | 0 | 宣言 h:589 + 定義 cpp:1153 のみ、**呼び出し 0** | **PASS** |
| V2 | `.delivery =` plain write | 0 | `desired.delivery`（ローカル語）2 箇所のみ（cpp:1139/1422）、slot への plain write **0** | **PASS** |
| V3 | 旧 atomic field 宣言 | 0 | `consecutiveFailureCount`/`atomic<ObligationState>`/`atomic<...Id>` **0** | **PASS** |
| V4 | id/state 直接 atomic アクセス | 0 | slot 経由の `.state.`/`.id.` アクセス **0**（`identity` は別フィールド・正当） | **PASS** |
| V5 | intrinsic 直接使用 0 / `std::atomic<W>` 宣言 1 / runtime 検証 1 | — | `_InterlockedCompareExchange128` 使用 **0**（STL 内部のみ）/ storage 宣言 **1**（h:386、他は検証用一時 object h:420・static_assert h:96・コメント）/ `is_lock_free()` 検証 **1**（h:421） | **PASS** |
| V6 | W + alignas + static_assert | 実装済 | h:84 alignas(16) 定義 + h:92-95 assert 4 点 + h:96 alignof(atomic) assert（計 5 点） | **PASS** |
| V7 | adjudicate call site | production 1 + tests | production: Threading.cpp:272（runCoordinatorPhase）**1** + TEST-ONLY ラッパ内 cpp:1156 + tests 4 | **PASS** |
| V8 | `postRecoveryFailureSignal` production | 6 | RebuildDispatch 4 + Orchestrator 2 = **6**（+定義/宣言） | **PASS** |
| V9 | CTest | NOT RUN | **NOT RUN — audit ゲート通過後の次工程** | 保留 |
| V10 | 旧 counter/delivery 構造 | 0 | `fetch_add.*consecutive`/`.consecutiveFailureCount` **0** | **PASS** |

**残存参照 sweep**: 全 src 対象に `recoveryConsecutiveFailureCount`・slot の `.id/.state/.delivery` 直アクセス → 0（`slots_[i].identity` 2 hit は正規表現部分一致の誤検出、identity は W 外 plain 維持が仕様）。tryInsert 呼び出し = 1（cpp:964、新署名）。

## 2. コードトレース 12 項目（最終ソース再読による）

1. **T1〜T7 expected/new**: T1 `{O,Live,p<K,a,d}→{O,Live,p+1,a,d}`（cpp:1094-1097）✓ / T2 非枯渇 `{O,Live,c,a,d}→{O,Live,0,a2,None}` 単一 CAS（cpp:1136-1141）✓ / T2 枯渇 resolve 経由 `{O,Live,..}→{O,Failed,0,0,d}`（cpp:1130-1134→h:515-524）✓ / T3 統合 CAS（h:515-524）✓ / T4 payload→CAS→+1（h:469-494）✓ / T5 全文 identity CAS + 再試行（cpp:938-947）✓ / T6/T7 casDelivery（cpp:1412-1427）✓
2. **T5 contention 再試行**: CAS 失敗→`w` 現在値更新→`state!=Live` のときのみ break（tryInsert 経路）。Live のうちは再試行 — **1 発失敗で NEW を作らない**（cpp:938-947）。D152 R1 充足 ✓
3. **T4 payload-before-Live**: payload 6 書込（h:470-475）→ CAS（h:482）→ liveCount++（h:493）。呼び出し側の post-insert 書込は消滅（cpp:961-972 は oblId 読のみ）✓
4. **liveCount −1 = CAS 勝者のみ**: h:523（resolve CAS 成功ブロック内のみ）。adjudicate 枯渇は resolve 経由なので同一路径 ✓
5. **exhaustion telemetry = 勝者のみ**: cpp:1132-1133（`resolveRecoveryObligation` が true を返したときのみ ++）。旧 cpp:1079 の pre-resolve 増加は消滅 ✓
6. **resolve の唯一 −1**: table.resolve 呼び出し元 = resolveRecoveryObligation のみ（adjudicate 枯渇・Route A/B/C・shutdown は全て同経路に収束）✓
7. **tryInsert の唯一 +1**: h:493、CAS 勝者パスのみ。table 外 +1 なし ✓
8. **delivery writer authority**: 変更経路 = tryInsert 初期化（desired.delivery=0）/ casDelivery / T2 非枯渇 desired — 全て CAS 内・owner=CL（casDelivery 呼び出し元: submit・redrive・いずれも CL）。postSignal/resolve は delivery 保持 ✓
9. **lifecycle 全 writer**: tryInsert(h:482)・resolve(h:520)・postSignal(cpp:1096)・adjudicate(cpp:1140)・coalesce identity CAS(cpp:942)・casDelivery(cpp:1423)。全 6 箇所が `compare_exchange_strong` のみ。非 CAS store なし ✓
10. **lifecycle 全 reader**: findByKey・tryInsert/resolve/postSignal/adjudicate/casDelivery scan・wasDeferredBefore・coalesce oblId/delivery 読・cpp:972・redrive scan/gate・shutdown close・peek・adjudicatedFailureCount — 全て advisory load で、commit は必ず CAS（規律 §3 準守）✓
11. **ctor runtime lock-free 検証**: h:418-430（static 一回・Debug/Release 両構成で ctor 実行）✓
12. **MSVC 前例 `#if` 踏襲**: h:422-426 が ISRDSPHandle.cpp:21-26 と同一分岐形（MSVC=記録のみ／非 MSVC=assert）。`is_always_lock_free` は失敗判定根拠に使用せず ✓

## 3. 監査上の注記

- **LSP（clangd）診断はゲート不採用**: 本環境の clangd はプロジェクトのビルドフラグ（C++17/20 標準・include パス）を持たず、`std::optional`/`std::filesystem`/`is_always_lock_free` のエラーが**既存コード**（popRecoveryRequest 等・実ビルドで 40/40 合格中）にも等しく出る環境ノイズ。新規コード固有の構文エラーは検出されなかった。真の構文検証は次工程の Debug/Release build。
- **実装選択の記録（仕様適合）**: (i) T2 枯渇は独自 CAS を持たず**単一 terminal authority `resolveRecoveryObligation` 経由**（D152 §5 T2「route through the single resolve authority」の文字どおり実装。勝者信号のため resolveRecoveryObligation を bool 化 — D152 §5.8 の単一 authority 要求の最小実現）。(ii) `resolve` の `discardedPendingOut` 省略時既定値で既存呼び出し形を維持。
- **NT-5 の 20ms 窓**: invariant アサートのみ（順序非決定に耐性）。干渉候補は postSignal∥adjudicate∥resolve のみで、全 commit が全文 CAS — 混成世代 commit は構造上不能（D150 §9-D,E,F）。
- **ConvoPeq.md は再生成されていない**（派生スナップショット。実装後の次再生成時に T3c 構造が反映される。監査はソース直読で実施済み）。

## 4. 判定

```text
D153-I implementation audit = ALL PASS（V1-V8, V10 + コードトレース 12/12）
  → build ゲート（Debug build → Debug CTest → Release build → Release CTest）へ進出可。
  → V9 は build/CTest 実施時に確定させる（NOT RUN からの更新）。
本監査ターンでは build/CTest を実施しない（指示どおりの停止点）。
```
