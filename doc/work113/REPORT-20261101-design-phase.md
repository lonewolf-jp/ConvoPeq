# 設計フェーズ報告（2026-11-01 第 2 部）— B-2 / A2→A3-a / D-01 / 最新 ConvoPeq.md 共有

- 基準: `main` = `origin/main` = **`6bb7c8d5`**（`fd92c6e7` のソースから `ConvoPeq.md` を再生成した doc-only コミット）
- 作業拠点: `C:\VSC_Project\ConvoPeq`（main-develop セッションのみ）
- **製品コード・テストは変更していない。ビルド・デバッガ・リプレイも実行していない**（Owner 制約 4）

## 1. 最新 ConvoPeq.md の共有（Owner 制約 1）

| 項目 | 値 |
|---|---|
| 生成元ソース | **`fd92c6e7`**（`src/` はこの時点から不変） |
| 再生成コミット | **`6bb7c8d5`**（doc-only: `ConvoPeq.md` の再生成のみ。`origin/main` と一致） |
| 生成時刻（ファイル内ヘッダ） | 2026-11-01 20:15:00 |
| サイズ | **6,436,359 B**（git blob。`.gitattributes` の `*.md text eol=lf` により正規化後） |
| **SHA-256** | **`f8aab49b49dccd9134ec8a4e777a6ddbfe9ed7c3be7f8a923160049239fa2f08`** |
| 分割版 | **4 ファイル**・合計 6,436,359 B・`identical=True`・全 2 MB 以下 |
| Dropbox | `ConvoPeq.md` ＋ 分割 4 ＋ ハッシュ付き 4 ＝ **9 ファイル**・`mismatch=none`（ALL MATCH） |

### 1.1 既報（`67cb8260…`）との差分

前回共有した 6,436,359 B / SHA-256 `67cb8260…` と、今回の `f8aab49b…` は
**「Generated:」行のタイムスタンプ 1 行だけ**が異なる（内容は同一）。

- 確認方法: 改行正規化後の比較で **145,541 行が完全一致**、差分は 1 行のみ。
- 作業ツリーの `ConvoPeq.md` は CRLF（6,581,900 B）だが、`.gitattributes` の
  `*.md text eol=lf` により **git blob は LF（6,436,359 B）**。共有・分割はこの blob から作っている。
- **生成元ソースは `fd92c6e7` のままで、`src/` に変更は無い。**

## 2. B-2 設計案（`doc/residual-20261101/B2-DIAG-LIFETIME-DESIGN-20261101.md`）

### 2.1 Owner 条件への対応

| 条件 | 設計での担保 |
|---|---|
| `convDiagBuffer` と `eqDiagBuffer` の**両方** | 両方を同一機構（インスタンス所有の `DiagSink`）へ移行 |
| 所有権と参照先の寿命を一致 | バッファ本体は既存メンバのまま**エンジンが所有**（消費者 `Timer.cpp:1736` が同一エンジン内で pop するため）。新設 `DiagSink` は**所有しない参照の束**（4 ポインタ）で、生産者は**インスタンスごとの `std::atomic<DiagSink*>`** で参照する |
| 破棄時に RT が参照していないことを保証 | **停止順序をコードで固定**: ①コールバック停止要求 →②スレッド終了待ち →③`audioCallbackActiveCount == 0` →④`detachDiagSink()` →⑤破棄 |
| 単純なリセット / atomic 化で解決と判断しない | **B-1（クリア＋atomic）は不十分**と明記（detach と破棄の間に RT が load 済みポインタを使う窓が残る） |
| 複数エンジン／反復生成破棄／診断無効 | C1（2 エンジン同時）、C2（114 回構築）、C3（診断 OFF でレイアウト不変）を確認項目に設定 |
| RT でロック・確保・解放なし | RT は `atomic load` ＋ 固定長リングへの push のみ（`DiagEvent` はスタック上の POD） |

### 2.2 実測で裏付けた前提

| 項目 | 実測 |
|---|---|
| `audioCallbackActiveCount` | `AudioEngine.h:1852`、増減は `AudioBlock.cpp:93/98`・`BlockDouble.cpp:88/93` で**コールバック本体を囲む**（静止判定に流用可能。新規 RT コスト無し） |
| `~AudioEngine` の停止順序 | `CtorDtor.cpp:154-171` で StopAcceptingWork → StopAudio → StopWorkers（retryScheduler / Coordinator join / rebuild 停止） |
| 生産者インスタンス | `convolver`（`AudioEngine.h:894`）、`uiConvolverProcessor`（同 2317）、DSPCore 群の `logEqTime` |
| 消費側 | `AudioEngine.Timer.cpp:1736` の `diagBuffer.pop` |

### 2.3 残る確認点

- オーディオコールバックの **join** が `~AudioEngine` / `releaseResources` のどこで保証されるか
  （保証が無ければ `audioCallbackActiveCount == 0` の待機を破棄経路に追加する）。
- `attach` を冪等にする（`prepareToPlay` が複数回呼ばれる経路がある）。

## 3. A2 → A3-a 設計案（`doc/residual-20261101/A2-A3A-ASSERT-DESIGN-20261101.md`）

### 3.1 Owner 条件への対応

| 条件 | 設計での担保 |
|---|---|
| **assert を削除しない** | 述語 `prevState == Quarantined` は **CAS 成功時の不変条件**として保持。加えて**真の異常（`Quarantined` への逆行）**を検出する assert を新設 |
| A1 単独は採用しない | 呼び出し側の再確認は A3-a（再試行範囲の限定）に吸収し、単独案としては採らない |
| A2 と A3-a の責務・順序・競合時の挙動 | A2＝「観測と遷移」の契約、A3-a＝「誰が再試行を起動するか」。**A2 → A3-a** の順で適用 |
| Debug と Release の両方で整合性 | 同一の遷移表を両構成で検証（V3） |
| 二重解放と誤状態処理を**別々**に検証 | V1（フリーリスト push のスロット×世代カウント）と V2（呼び出し側が `false` 時に lane/flag を触らないこと）を独立に実施 |

### 3.2 実測で裏付けた前提

| 項目 | 実測 |
|---|---|
| `MAX_DSP_SLOTS` | **256**（`ISRDSPHandle.h:197`）→ ビットマスクは `uint64` × 4 |
| `deferredQuarantineDestroyCount()` の呼び出し元 | `AudioEngine.Retire.cpp:44`（`quarantineEmpty`）と `:72`、`DeferredPublicationTestAccess.h:175` → **「保留中のスロット数」を返す契約を維持** |

## 4. D-01 の再審査（CLOSE 推奨）

### 4.1 最新ソースでの呼び出し経路（実測）

| 要素 | 実測 |
|---|---|
| 製品の呼び出し元 | `RuntimePublicationOrchestrator` が `PublicationAdmission admission_`（`RuntimePublicationOrchestrator.h:376`）を保持し、`admission_.evaluate(req, engine_, pubCtx)`（`RuntimePublicationOrchestrator.cpp:47`）を呼ぶ |
| 判定条件（製品） | `PublicationAdmission.cpp:50-58` の `hasFading` は **`#if defined(CONVOPEQ_UNIT_TESTS)` の内側**に `engine.testFadingRuntimePresent()` を持ち、製品では **`engine.hasFadingRuntimeInWorld(...)` のみ** |
| 製品の述語 | `AudioEngine.h:3621` の `hasFadingRuntimeInWorld`（`PrepareToPlay.cpp:169`、`Timer.cpp:630` でも使用） |
| 製品での書き手 | `testFadingRuntimePresent_` を書くのはテスト専用クラスのみ（`CONVOPEQ_ALL_SOURCES` に `src/tests` は 0 件） |

### 4.2 判断

**製品実行経路では、このフックは存在しない（コンパイル対象外）**。これは
`AudioEngine.h:3996-4004` のコメントどおり**意図された設計**であり、
製品の判定は `hasFadingRuntimeInWorld` が担う。

したがって「製品 2 行」を追加しても:

- 製品の判定結果は変わらない（書き手が存在しないため常に false）。
- **テスト専用フックが製品の判定ロジックに常駐する**という設計変更だけが残る。

**結論: 2 行は適用しない（D-01 を CLOSE することを推奨）。**
「STG-8-D2 の retention 検証が実質化する」という当初の目的は、
`CONVOPEQ_UNIT_TESTS=1` でビルドされるテスト（`AudioEngineHarness`）で**既に達成**されている。

## 5. 共通制約の遵守

| # | 制約 | 状態 |
|---|---|---|
| 1 | 最新 ConvoPeq.md の取得と commit / size / SHA-256 の確認 | **完了**（§1） |
| 2 | 最新ソースを確認できない場合はコード変更に着手しない | 確認済み。ただし**コード変更自体を実施していない**（設計のみ） |
| 3 | 作業場所・セッション | 遵守 |
| 4 | 製品コード変更・テスト変更・build・debugger/replay は設計承認に含まれない | **いずれも未実施** |
| 5 | U5 FAIL・既定 OFF、U3/U4 HOLD | **維持** |
| 6 | 既存の未追跡 24 件を整理・削除・追跡しない | **既存 24 件は不変**。本フェーズで追加した 3 件（設計書 2 ＋ 本報告書 1）を含め、未追跡は計 **27 件**（追跡・整理・削除はしていない） |
| 7 | 測定結果を理由に合格基準を変えない | **変更なし** |

## 6. 実装承認をお願いしたい範囲（次段）

| 案 | 実装対象 | 事前条件 |
|---|---|---|
| **B-2** | `DiagSink` 導入、グローバル廃止、停止順序の固定、`convDiagBuffer` / `eqDiagBuffer` の両方 | §2.3 の join 保証の確認 |
| **A2 → A3-a** | assert の位置と述語、deferred マーク、再試行範囲 | なし（設計は確定） |

いずれも **コード変更・ビルド・テスト実行の承認が別途必要**です。
本報告では設計案の提出までとし、実装には着手していません。
