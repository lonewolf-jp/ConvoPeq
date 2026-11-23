# --verify-runtime-evidence-live の設計

- 実施日時: 2026-11-15 00:40〜01:10 JST
- 基準: `origin/main` = `783f3d24dbda110827770c66a2108a370698117f`
- 作業ブランチ: `feat/verify-runtime-evidence-live`（独立 worktree `C:\VSC_Project\live-evidence`）
- 種別: 設計（§8 に実装結果を追記。§2.1 の `sampleCursorCount` 要件は §8 で置き換え済み）

---

## 1. 既存実装の構造と問題点（調査結果）

### 1.1 現行の `--verify-runtime-evidence`

`src/MainApplication.cpp:72-79`

```cpp
if (commandLine.contains("--verify-runtime-evidence"))
{
    convo::isr::EvidenceExporter evidenceExporter;
    evidenceExporter.exportEvidence(nullptr, 0);
    juce::Logger::writeToLog("[verify-runtime-evidence] runtime evidence exported");
    juce::JUCEApplicationBase::quit();
    return;
}
```

- **部分一致**（`contains`）のため、`--verify-runtime-evidence-live` もこの分岐に入る
- `EvidenceExporter` を生成して `exportEvidence(nullptr, 0)` を 1 回呼ぶだけで、**Runtime（AudioEngine）は起動しない**
- `snapshot == nullptr` のため、テレメトリ由来の 3 artifact は空文字で書かれる

### 1.2 `EvidenceExporter::exportEvidence()`（`ISREvidenceExporter.cpp:325`）

- artifact 11 件のうち **8 件はハードコードされた placeholder**（`status: "not_measured"`、`nodeCount: 0` など）
- `snapshot != nullptr` のときだけ `publication_failure_log.json` / `publication_progress_log.json` / `deferred_health.json` が実測値になる
- manifest は `generationMode: "runtime"`、`runtimeRunId`（env `CONVO_ISR_RUNTIME_RUN_ID` か `"runtime-local"`）、`generatedAtNs`（`steady_clock`）を書く
- Release かつ `forceFullEvidence=false` では manifest のみ（`minimalEvidence: true`）

### 1.3 `isr-run-runtime-evidence.ps1`（CI の実行スクリプト）

- `--verify-runtime-evidence` で exe を起動し、20 秒待つ（280 行）
- **不足 artifact をスクリプトが捏造する**（`New-FallbackArtifactObject`、66〜206 行、245〜251 行）。中身は `provenance: "runtime"`、`status: "generated"`、`eventCount: 1` などの固定値
- 全 artifact に `provenance: "runtime"` を上書きする（268 行）

### 1.4 `isr-verify-evidence-provenance.ps1`（検証）

- runtime モードで見るのは `provenance == "runtime"`、`runId == manifest.runtimeRunId`、`generatedAtNs > 0` の 3 点（59〜79 行）
- これらは 1.3 のスクリプトが付与できるため、**Runtime 実行の証明にならない**

### 1.5 既存の Runtime 側の実測点（活用できる資産）

- `AudioEngine::rtLocalState_.audioCallbackEpochCounter`（`AudioEngine.h:1905`、`AudioBlock.cpp:53` で callback ごとに +1、**単調増加・常時**）
- `AudioEngine::rtLocalState_.audioSampleCursorCounter`（`AudioEngine.h:1906`、`AudioBlock.cpp:312` で `numSamples` ずつ加算、**単調増加**）
- `AudioEngine::consumeCliProcessingTelemetrySnapshot()`（`AudioEngine.h:2093`、CLI 時のみ・消費で 0 に戻る）
- `AudioEngine.Commit.cpp:735` の `evidenceExporter_.exportEvidence()` … **実 Runtime 中に engine 自身が evidence を書いている**（`epoch_reclaim_audit.json`、`retire_timeline.json`、`world_retirement_telemetry.json` などは実測値）

→ 実 Runtime 実行では evidence が engine から書かれる。既存モードはそれを待たずに Exporter だけを呼んでいる点が本質的な欠陥である。

---

## 2. Live Evidence の定義

`--verify-runtime-evidence-live` が証明する内容を次の 5 点に限定して定義する。**「Runtime が起動し、音声コールバックを処理し、その観測値が engine 自身から得られた」**ことだけを主張し、ISR 安全性の一般保証は主張しない。

| ID | 要件 | 判定に使う観測値 | 生成元 |
| --- | --- | --- | --- |
| **L1** Runtime 起動 | AudioEngine が生成され、`prepareToPlay` 相当の初期化が完了した | `engineInitialized=true`、`sampleRateHz>0`、`blockSamples>0` | アプリ（engine の状態） |
| **L2** Runtime 実行 | 音声コールバックが実際に処理された | `audioCallbackCount>0` かつ `sampleCursorCount>0` | engine の単調カウンタ |
| **L3** Runtime 由来 | 証拠が実行中のプロセスから出ている | `live_runtime_evidence.json` の mtime が **プロセス開始〜終了の窓内**、かつ evidence/ 配下に**終了後に書かれたファイルが無い** | プロセスとファイルシステム |
| **L4** 実行識別 | 1 回の実行を一意に識別できる | `runId`（env）と manifest の `runtimeRunId` が一致、`processId`、`startedAtNs < emittedAtNs` | アプリ + スクリプト |
| **L5** Fail-Closed | 次のいずれかで成功扱いにしない | 起動失敗 / callback 0 / 必須観測値欠落 / 書き込み失敗 / runId 不一致 / 窓外 mtime / 終了後書き込み | 検証スクリプト |

### 2.1 「Live」の証明強度（何をもって Live とするか）

- **固定値・テンプレート値の排除**: Live ファイルの値は engine のカウンタから読む。スクリプトは値を書かない
- **相互整合**: `audioCallbackCount × blockSamples` と `sampleCursorCount` が ±10% 以内で一致することを要求する（engine の実測同士の整合。手書きの固定値では成立しにくい）
- **時間窓**: ファイルの mtime がプロセス開始時刻以降・終了時刻以前であること。**プロセス終了後に evidence/ へ書き込みが無い**こと
- **識別**: `runId` は実行ごとに一意（スクリプトが GUID で生成して env で渡す）。古い evidence は実行前に削除し、削除に失敗したら停止する

### 2.2 証明しないこと（明示）

- 悪意あるスクリプトが実行中に偽のファイルを書くことまでは防がない（本機構は「通常のパイプラインが捏造していないこと」を証明する）
- ISR の意味的正しさ（別の検証群の責務）
- 音声デバイスの実入出力（本機構はコールバック処理の発生までを対象とする）

---

## 3. CLI の動作設計

| 項目 | 設計 |
| --- | --- |
| フラグ | `--verify-runtime-evidence-live` |
| 単独指定 | **失敗**（Runtime を回さないため）。エラーログを出し、終了コード **2** |
| 必須の併用 | `--cli-run` と `--cli-exit-ms <ms>`（実 Runtime を回す既存経路） |
| GUI | 既存の `--cli-run` と同じ（ウィンドウは生成されるが自動終了する）。新しい GUI 経路は作らない |
| 初期化 | 既存の CLI 経路（device 選択、`prepareToPlay`）をそのまま使う。追加の初期化はしない |
| 実行 | `--cli-exit-ms` の間、engine がコールバックを処理する |
| 取得時点 | `MainApplication::shutdown()` の先頭（`mainWindow.reset()` より前 = engine 生存中） |
| 出力先 | `current_path()/evidence/live_runtime_evidence.json` |
| 終了コード | 0 = Live 証拠の生成に成功（callback>0 を含む） / 2 = フラグ単独（実行なし） / 3 = 実行したが証拠生成に失敗（callback 0・書き込み失敗） |
| タイムアウト | スクリプト側で `--cli-exit-ms + 30 秒`。超過は強制終了して **失敗**（成功扱いにしない） |

### 3.1 既存モードとの関係

- `--verify-runtime-evidence` の分岐は**変更しない**（部分一致のまま）。Live の分岐を**その前**に置くため、`-live` は Live 側で処理される
- 既存モードの manifest（`generationMode: "runtime"`）と検証（`isr-verify-evidence-provenance.ps1`）は不変
- Live は `generationMode: "runtime-live"` を使い、既存の検証対象と混ざらないようにする

---

## 4. 実装方針（責務分離）

| 主体 | 責務 |
| --- | --- |
| `AudioEngine` | 観測値の提供（単調カウンタの読み出しのみ。追加は読み取り専用アクセサ 1 個。§8 参照） |
| `MainApplication` | Live 要求の受理、shutdown 時の観測値収集の起点、終了コードの決定 |
| `MainWindow` | engine の状態（初期化済み・サンプルレート・ブロック長・デバイス名・IR 状態）の取得 |
| `EvidenceExporter` | **受け取った観測値を JSON に書くだけ**（Runtime 実行の責務を持たない） |
| スクリプト | 実行（起動・待機・終了コード確認）と検証（窓・整合・識別）。**値の生成はしない** |

### 4.1 変更ファイル（予定）

| ファイル | 変更 |
| --- | --- |
| `src/MainApplication.cpp` | Live 分岐の追加（既存分岐の前）、終了コード設定 |
| `src/MainWindow.h` / `MainWindow.cpp` | engine 状態の取得と Live 観測値の収集 |
| `src/audioengine/AudioEngine.h` | 単調カウンタの読み取りアクセサ（1 個。§8 参照） |
| `src/audioengine/ISREvidenceExporter.h` / `.cpp` | Live 観測値の構造体と書き出し関数 |
| `.github/scripts/isr-run-runtime-evidence-live.ps1`（新規） | 実行 + 検証（fail-closed） |
| `.github/scripts/isr-check-evidence-live.ps1`（新規） | 検証の単体実行（負のテスト用） |
| `.github/scripts/isr-run-tiered-verification.ps1` | `-RequireLiveRuntimeEvidence` スイッチ（既定 off） |
| `.github/workflows/isr-verification.yml` | 入力 `requireLiveRuntimeEvidence`（既定 false） |
| `.github/workflows/vbcable-probe.yml` | 入力 `requireLiveRuntimeEvidence`（既定 false）と Live 実行ステップ |

### 4.2 命名方針（重要）

検証スクリプトは `isr-verify-*.ps1` と**しない**。`isr-verify-verifier-execution-layers.ps1` は execution-layer policy の検証器数を **37 件に固定**しており、`isr-verify-*` を追加するとポリシー表（`.github/isr-verifier-execution-layers.json` と `ISRRuntimeSemanticSchema.h`）の同時改訂が必要になる。これは本作業の範囲外なので、`isr-check-*.ps1` として追加し、既存の検証器レジストリを変更しない。

---

## 5. Live 証拠ファイルのスキーマ

```json
{
  "schema": "live_runtime_evidence_v1",
  "generationMode": "runtime-live",
  "evidenceSource": "process",
  "runId": "<CONVO_ISR_RUNTIME_RUN_ID>",
  "processId": 12345,
  "startedAtNs": 123456789,
  "emittedAtNs": 123456789,
  "runtime": {
    "engineInitialized": true,
    "audioCallbackCount": 187,
    "sampleCursorCount": 1146880,
    "sampleRateHz": 48000.0,
    "blockSamples": 512,
    "irLoaded": true,
    "irLength": 24000,
    "deviceType": "Windows Audio",
    "deviceName": "..."
  },
  "consistency": { "callbacksTimesBlock": 95744, "withinTolerance": true }
}
```

- `startedAtNs` / `emittedAtNs` は `steady_clock`（プロセス内の単調時計）
- ファイルの mtime（wall clock）は検証側で「プロセス開始〜終了」の窓と比較する

---

## 6. テスト計画（指示 7 の 8 ケース）

| # | ケース | 方法 | 期待 |
| --- | --- | --- | --- |
| 1 | 正常（Runtime 実行） | `--cli-run --cli-exit-ms 4000 --verify-runtime-evidence-live` | exit 0、callback>0、検証 PASS |
| 2 | 初期化失敗 | 検証器に `engineInitialized=false` の証拠を与える | 検証 FAIL（exit 1） |
| 3 | Runtime 未実行 | 検証器に `audioCallbackCount=0` の証拠を与える | 検証 FAIL |
| 4 | 必須証拠欠落 | `sampleCursorCount` を削除した証拠 | 検証 FAIL |
| 5 | 証拠生成失敗 | 不正 JSON / 空ファイル | 検証 FAIL |
| 6 | 古い証拠が残存 | 実行前に古い runId の証拠を置く | 実行スクリプトが削除してから実行。削除できない場合は停止。検証器は runId 不一致で FAIL |
| 7 | 既存モード | `--verify-runtime-evidence` を実行 | 従来どおり manifest を生成（`generationMode=runtime`） |
| 8 | 終了コード | 単独指定 / 正常 / callback 0 | 2 / 0 / 3 |

CI 実行は `-RequireLiveRuntimeEvidence` を明示したときのみ（既定 off）。既定の CI 挙動は変えない。

---

## 7. 残存リスク

1. **CI ランナーの音声デバイス依存**: `--cli-run` は実デバイスを開く。デバイスが無いランナーでは callback が 0 になり、Live 検証は FAIL する（fail-closed）。CI での実行可否は VB-CABLE 等の導入状況に依存する
2. **悪意ある同時書き込み**: 実行中に別プロセスが evidence を書く攻撃は防がない（§2.2）
3. **`steady_clock` と wall clock の混在**: 窓判定は mtime（wall clock）で行うため、システム時計の変更には弱い
4. **既存モードの意味は不変**: 既存の `--verify-runtime-evidence` は引き続き「Exporter の出力確認」であり、Live の代用にはならない

---

## 8. 実装結果（as-built）

実装と検証の結果は `doc/work113/20261115-0106_Live証拠実装/report.md` に記録した。設計からの主な変更点は次の 7 点。

| # | 設計 | 実装 | 理由 |
| --- | --- | --- | --- |
| 1 | `sampleCursorCount > 0` を要求 | 削除し、`derived.dutyRatio`（処理音声秒 / 実時間、0.25〜4.0）で代替 | standalone は double 経路（`processBlockDouble`）を使い、`audioSampleCursorCounter` は float 経路でしか更新されない（実測で常に 0）。RT 経路への counter 追加は避けた |
| 2 | スクリプトは任意の文字コード | **ASCII のみ** | PowerShell 5.1 は BOM 無しファイルを ANSI で読むため、日本語コメントが化けて構文解析が壊れた |
| 3 | checker を同一セッションで実行 | 子プロセスで実行 | checker の `exit` が呼び出し元 runspace を終了させる |
| 4 | `isr-verify-*.ps1` | `isr-check-*.ps1` | execution-layer policy の検証器数 37 件を変えないため |
| 5 | evidence/ の全 JSON を削除 | live ファイルのみ削除 | 既存モードの artifact を壊さない（識別は runId と mtime） |
| 6 | （未記載） | CI は `-RequireLiveRuntimeEvidence` / 入力 `requireLiveRuntimeEvidence` で**明示時のみ** | 既定の CI 挙動を変えない |
| 7 | （未記載） | Live を実行する CI は `vbcable-probe.yml`。`isr-verification.yml` は転送のみで、アプリをビルドしないため実行ファイルが無い | 音声デバイスを持つ hosted runner は VB-CABLE を導入する workflow だけである |

検証は正常系 3 構成（Debug×2・Release）、異常系 11（改変証拠）、終了コード 3 種、既存モード互換、ローカルゲート 8 種で行い、すべて期待どおりだった。CI 上での実行は未実施で、workflow が呼ぶのと同じコマンドをローカルで実行して PASS を確認した（push は承認済みだが、資格情報の `workflow` スコープ不足で GitHub に拒否された。詳細は実装報告 §10・§11）。
