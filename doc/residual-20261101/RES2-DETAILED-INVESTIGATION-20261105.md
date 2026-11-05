# RES2 詳細調査報告（2026-11-05）

- 基準 commit: `f90f56be`（`main` = `origin/main`、作業開始時に clean・worktree 1 本を確認）
- 参照: `ConvoPeq.md` 6,493,955 B / `ac2d5f76…`（FRESH, `NEWER_SRC_COUNT: 0`）、ソース、テスト、CI 定義、Git 履歴
- 位置づけ: 本報告は調査結果であり、承認されていない実行・有効化・再開を含まない。

## 1. 全体サマリー

| 区分 | 件数 | 対象 |
|---|---|---|
| 既に修正済み（今回の再確認対象外） | 3 | RES2-02・14・17 |
| 追加調査の結果、前提が古いと判明 | 2 | RES2-08、RES2-03 |
| 技術的に解消（文書・コメント） | 2 | RES2-03（コメント更新）、RES2-08（設計移行の記録） |
| Owner 決定（現状維持・保留） | 9 | RES2-01・04・05・06・07・11・12・15・16 |
| 要件照合待ち | 1 | RES2-13 |
| 監視 | 2 | RES2-09・10 |
| 対応不要 | 1 | RES2-19 |

## 2. 重点項目の調査結果

### 2.1 RES2-01 ポリフェーズ利得（案 E）

**現行コードの再確認（実測）**

現行 OFF ビルドの `PolyphaseGainFidelityTests` を実行し、54 項目すべて PASS（`summary: PASS=54 FAIL=0`）。主な実測値:

| 検証 | 結果 |
|---|---|
| DC（IIR3/LP3 × r=2/4/8） | base = 0.75 / 0.5625 / 0.421875（厳密に 0.75^N）、cand = 1.0 |
| impulse S1(31/90) | Σ base 0.75 / cand 1.0、peak offset 15 |
| P0-F latency ×4 経路 | argmax ∈ [floor(D), floor(D)+1]、base == cand（bitwise） |
| P0-I SoftClip OS DC（A=0.5） | base 0.375 / cand 0.5（±1e-6）、clamp 0、tanh clamp 18200 |
| P0-G float/double | maxAbs 2.58e-08 / 2.96e-08、maxRel 1.84e-05 |
| D2 画像 | −223.31 / −217.69 / −184.26 dB（linear 経路） |

**現行コードの根拠**

- `src/CustomInputOversampler.cpp:556-561`: `convValue *= 2.0` は無条件、`centerValue *= 2.0` は `CONVOPEQ_CORRECT_POLYPHASE_GAIN` の内側。
- `CMakeLists.txt:134`: 既定 OFF。
- `PolyphaseGainFidelityTests` は candidate を shadow で同一バイナリ内に持ち、REF-FIDELITY で production（OFF）と bitwise 一致を要求。したがって candidate の数値は「フラグ ON 時の挙動」の代理として妥当。

**未確定事項（旧資料から継承し、現行でも未確定）**

- 0.75^N が設計意図かどうか（P1-4 N1）。
- 聴感・preset replay・EQ/IR 併用の実データ。

**処置**: 既定 OFF 維持。判断材料は既存の `RES2-DECISION-PACKETS-20261105.md` §1 と本節。ON ビルドは別途取得し、数値を §4 に記録する。

### 2.2 RES2-03 SoftClip 局所 OS の遅延定数

**判明した事実**: 「未実測」は誤り。`PolyphaseGainFidelityTests` の P0-F latency gate が S1(31/90) 経路を D=15 として実測しており、argmax が [15,16] に入ることを要求している。今回の実行でも PASS。

したがって定数 15 は既存テストで裏付けられている。残る問題は `AudioEngine.Processing.Latency.cpp:119` のコメント「要実測確認」が古いことだけである。

**処置**: コメントを「P0-F latency gate（31tap/90dB, D=15）で検証済み」へ更新する（挙動変更なし）。定数は変更しない。

### 2.3 RES2-04 F-4 / D2b 記録消失

**現行コード**: `STG8_ONLY=d2b` のハーネスセレクタ（`PublishPipelineIntegrationTests.cpp:1200-1240`）と、`deferredOverwriteCount`・`recoveryRetryRedriveCount`（`RuntimePublicationOrchestrator.cpp:509`、`ISRRuntimePublicationCoordinator.cpp:1264`）が存在する。旧資料が指す「D2b 診断タプル」はテスト側のセレクタであり、製品コードの記録消失カウンタではない。

**状態**: 38 反復で再現せず、Owner 判断で停止。今回も停止を維持し、再開しない。

**所見**: 再開する場合は「現象の定義（どの記録が、どの条件で、どう失われるか）」を先に固定する必要がある。現在の診断は STG-8 の obligation 系列が中心で、F-4 の現象を直接観測するものではない。

### 2.4 RES2-05 B2 Stage 2/3

**現行**: Stage 1 の計測車両はハーネス内（`B2_STAGE1_MEASURE`、`adoption_criteria_version=v5.3`、`b2_adoption` 出力）。本番経路の計測は無い。

**所見**: Stage 1 は「採用可能性の判定」であり、採用承認ではない。本番計測の必要性は、現行の ASan・テスト・soak で検出できないメモリ回帰の有無に依存する。現時点で具体的な未検出事例は無い。

**処置**: 設計資料のみ（§3.1）。

### 2.5 RES2-06 A2 本番適用

**現行**: `namespace a2probe` は `rtalloc` の別名へ統合され、ハーネス内でのみ動作（`AudioEngineHarness.cpp:37,156`、`PublishPipelineIntegrationTests.cpp:1384-1405`）。本番経路の計測は無い。NO-GO 維持。

### 2.6 RES2-07 U5

**現行**:

- `RtMeasureRing` は既定 `Mode::Off`（`RtMeasureRing.h:97`）。`initializeFromEnvironment()` が `CONVOPEQ_RT_MEASURE` を読む（`1`=Full, `2`=TimeOnly）。
- リング容量 8,192、`dropped` を別カウンタで計上（`:218-269`）。欠落が 10% 超またはサンプル欠落なら INCONCLUSIVE（`:28`）。
- UI・設定からの切替経路は無い（`UI/Panel/Settings/StateIO` に参照 0）。
- 測定は metric v2 で PASS（#119）。ただし PASS は「計測器の判定」であり、製品有効化の承認ではない。

**RT オーバーヘッド**: ブロックごとに 1 回の scope timer（`AudioBlock.cpp:34`）と 1 秒ごとの非 RT 集計（`Threading.cpp:386`）。計測 Off 時は分岐 1 回。今回のセッションでは独立したオーバーヘッド測定は実施していない（未検証として記録）。

### 2.7 RES2-08 EngineRuntime 統合

**判明した事実**: `RuntimePublishState` は現行ソースに存在せず（0 件）、`EngineRuntime` は `[[deprecated]]`（`AudioEngine.h:150`）で、`RuntimeWorld`／`worldAuthority`（428／177 件）が公開モデルとして使われている。旧「dual-write を単一 atomic へ統合する」前提は、その後の RuntimeWorld 移行で置き換えられている。

**処置**: 「設計移行により前提が解消」としてクローズし、残る移行作業（deprecated 型の除去）を技術的負債として記録する。

### 2.8 RES2-11 / 12 / 13

| ID | 事実 | 評価 |
|---|---|---|
| RES2-11 | `sanitizer-ci.yml:255-259` で TSan は `continue-on-error: true`、対象は MKL/IPP 非依存テストのみ | データ競合の検出が必須ゲートではない。必須化には対象拡大と安定性評価が必要 |
| RES2-12 | `soak-ci.yml` は `workflow_dispatch` のみ。S1/S2b/S3/S4/S5 | 夜間実行の設計は可能。頻度・コストは Owner 判断 |
| RES2-13 | TX-02/TX-10 は 2026-05 の記録で Waived（`phase1_tx_validation_log_jp.md`） | 現行要件として必須かは未確認。実機環境が必要 |

### 2.9 RES2-15 / 16

- `NoiseShaperLearner.cpp:1450-1456`: CMA は mean のみ復元。共分散・履歴は復元しない。
- 同 `:700-712`: elite 再評価は同じ候補を再実行するだけで、セグメントのランダム化が無い。
- いずれも設計判断（要件次第）。不具合ではない。

### 2.10 RES2-18 / 20

- `AudioEngine.h:3548-3550`: `ObserveChannel::Worker` は Worker スレッド未識別のためアサーションなし。識別方式の設計が必要。
- `.venv`（216 パッケージ）は `C:\Python314\python.exe` 不在で使用不能。venv を使う手順があるかで扱いが変わる。

## 3. 設計資料

### 3.1 B2 Stage 2/3 設計（承認待ち・実装しない）

| 項目 | 内容 |
|---|---|
| 計測対象 | Publish 経路のメモリ確保・解放（deferred retire 含む） |
| 計測方法 | 既存 `deferredOverwriteCount`／`recoveryRetryRedriveCount` に加え、確保サイズ別のカウンタを非 RT 側で集計 |
| RT 制約 | RT 経路は atomic increment のみ。確保・ロック・ログは禁止 |
| オーバーヘッド | 1 イベントあたり 1 回の fetch_add（キャッシュライン競合に注意） |
| 検証 | `RtMeasureRingTests`、ASan、soak、オーバーヘッド A/B |
| rollback | コンパイルフラグ（既定 OFF）で無効化 |
| 必要な承認 | 本番計測の実装・有効化 |

### 3.2 U5 UI 切替設計（承認待ち・実装しない）

| 項目 | 内容 |
|---|---|
| 既定 | Off（現行どおり） |
| 切替 | 設定画面で「実時間計測」を明示選択。再起動不要を目標 |
| 保存 | 既存 `DeviceSettings` と同様の属性保存。Off を既定値 |
| RT 安全性 | 切替は非 RT（メッセージスレッド）で `g_requestedMode` を publish し、RT は次のブロックで読むだけ |
| 影響 | Off 時は分岐 1 回。On 時は scope timer と 1 秒集計 |
| 説明 | 計測値は環境依存であることを UI に明記 |
| 検証 | 切替時のクリック・XRUN が無いこと、On/Off の A/B、ASan |
| 必要な承認 | UI 実装・製品有効化 |

## 4. 検証結果

| 検証 | コマンド / 環境 | 結果 |
|---|---|---|
| PolyphaseGainFidelityTests（OFF） | `build-icx\Release\PolyphaseGainFidelityTests.exe`（icx 2026.2.0, AVX2） | PASS=54 FAIL=0 |
| 既存回帰（今回の変更対象） | `D8_1`・`FFTBackendTests`・`CorruptSampleGuardTests` | すべて Passed |
| ON ビルド | `build-icx-on`（`-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON`） | §4.1 に追記 |

### 4.1 ON ビルドの結果

OFF=54/54 PASS、ON=19/54 PASS（OFF 期待値との差は設計どおり）。ON の DC は全構成 1.0、SoftClip OS DC は 0.5、float/double は 3.31e-08 で一致。既存記録（ON 19/54）を再現した。

## 5. 既存資料と現行コードの不一致

| 資料の記述 | 現行コード | 扱い |
|---|---|---|
| SoftClip 遅延は「要実測確認」 | P0-F が D=15 を検証済み | コメントを更新 |
| `RuntimePublishState` + `EngineRuntime` dual-write が現行 | `RuntimePublishState` 不存在、`EngineRuntime` は deprecated、`RuntimeWorld` が現行 | RES2-08 を前提解消としてクローズ |
| D2b は製品の記録消失カウンタ | ハーネスの STG-8 セレクタ | 再開時の定義を明確化 |
