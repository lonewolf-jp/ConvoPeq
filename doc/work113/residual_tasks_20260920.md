# ConvoPeq 残件棚卸し（2026-09-20 時点）

本ファイルは 2026-09-20 セッション終了時点の残件整理です。前スナップショット `residual_tasks_20260919.md` の
続編として、同ファイルの §B（B-1 帰属）と §F（新規棚卸し所見 F-2〜F-6）の更新を反映し、
**未 commit / 未 push の作業状態**と**未着手の別 work item**を一覧化したものです。

---

## 0. リポジトリ / 作業状態（実測）

| 項目 | 実測 |
| --- | --- |
| HEAD | `8f127bfe` docs(work113): record B-2 closure, post-series deviation and validator audit（2026-09-20 07:06:22） |
| origin/main との差 | **ahead 2 / behind 0**（`c4a08171` B-2 test、`8f127bfe` docs が未 push） |
| production `src/` の未 commit 差分 | **0**（`git diff --name-only -- src/` の非テスト差分は無し） |
| 未 commit エントリ | 6 件（下記 §1） |
| 実行した検証 | AudioEngineHarness Release インクリメンタルビルド、`--buzz-rigcheck=*` 各種、既定スイート |

---

## 1. 未 commit / 判断待ち（最優先で意思決定が必要）

### O-1. B-1 計測用 test-only 計装（未 commit・+642 行）

| ファイル | 差分 |
| --- | --- |
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | **+634 / −2** |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | **+8 / −0** |

内容（詳細は前スナップショット §F-6）:
- rigcheck 診断モード: `eqdiag` / `eqdiagser` / `eqos<digit>`（OS 明示固定）/ `irwet<digit>`（conv 有効の真の wet 対照）
- 観測行: `gainpath`（実効 staging/AGC/totalGain/headroom/makeup/struct）、`[EQ_RTPATH]`（World の eqCoeffHash・eqParams・routing・eqLPFMode・RT cache 実体）、`[XFADE]`（crossfade runtime 実値）、`[EQLEVEL]`（`outputLevelLinear` の測定窓中最大値）
- 直接駆動測定（既定スイート登録・`runEqDirectDriveAttribution()`）: `[EQ_DIRECT]`（実 EQProcessor、base 192k/2048 と RT 768k/4096）、`[OF_DIRECT]`（実 `convo::OutputFilter` ②、3 mode×2 rate）、`[OS_DIRECT]`（実 `CustomInputOversampler` round-trip、ratio 1/2/4/8 × preset、up/down 分離、`prepareSingleStage(31,90)`）

**判断が必要な理由**: B-1 は原因究明を完了してクローズしたため、この診断資産を
(a) そのまま commit して将来の回帰検出に残す、(b) 最小限（`[OS_DIRECT]` のみ等）に絞って commit、
(c) 退役させる、のいずれかを決める必要があります。現状は未 commit のまま保持されています。

なお、既存 `eq` の判定窓 `[0.486,0.496]` は本計装を通じて**不変**です（`eq` は `ratio=0.4912` で PASS 継続）。

### O-2. 台帳更新（未 commit）

`doc/work113/residual_tasks_20260919.md` — B-1 帰属（ATTRIBUTED）と F-2〜F-6 の追記分。
記録操作として独立に commit するか、O-1 と同梱するかの判断が必要です。

### O-3. `ConvoPeq.md`（生成物・未 commit）

`Generated: 2026-09-20 07:00:33`（本セッションで再生成）。その後 test-only 計装を追加したため **stale**。
production 内容は HEAD と一致しています。既存方針どおり **commit しない**（監査用一時ファイル・ユーザー運用）。
再生成は `python output_sourcecode_markdown.py`。

### O-4. `Testing/Temporary/CTestCostData.txt`（tracked が削除状態）

worktree で ` D` 状態。ctest の作業ファイルで、harness ビルド中の CMake/ctest が触ったものと推定。
**現状維持**（ユーザー判断: 勝手に `git checkout` しない）。戻す場合は
`git checkout -- Testing/Temporary/CTestCostData.txt` の 1 行。

### O-5. `.opencode/opencode.json`（未追跡）

作成者・目的不明。**触らない**方針（削除も commit もしない）。

### O-6. push（未承認）

`c4a08171`（B-2 test）と `8f127bfe`（docs）の 2 件が未 push。push は独立運用操作として**未実施**。

---

## 2. B-1 の修正設計（別 work item・未着手）

B-1 の**原因究明は完了・クローズ**していますが、**修正は未着手**です。
`decimateStage()` に `×2` を追加すればよい、という段階ではありません。

### 2.1 確定している事実（前スナップショット §B-1 の要約）

```
主因        CustomInputOversampler の up/down round-trip 利得欠陥
1 stage     round-trip 0.750000（up 1.000000 / down 0.750000）
多段        0.75^N（ratio 2/4/8 = 0.75 / 0.5625 / 0.421875、preset 非依存）
SoftClip局所OS 0.75（prepareSingleStage(31, 90.0) を production 引数で実測）
数学的導出  0.5×0.5 + 0.5×1.0 = 0.75
engine fit  0.98379 × 0.75^max(log2 effOS, 1) で 4 条件 ≤0.05%
契約不整合  isSymmetricUpDown / Latency の static_assert 前提と矛盾
production 変更 0 / commit 0
```

### 2.2 設計時に検討すべき 8 観点

1. `decimateStage()` 側の補正
2. up/down の gain convention 自体の再設計
3. `isSymmetricUpDown` の意味を実装に合わせる（宣言側を直すか）
4. `AudioEngine.Processing.Latency.cpp` の対称 FIR 前提への影響
5. 既存 calibration / rigcheck（窓 `[0.486,0.496]` を含む）の再取得
6. SoftClip 分岐（`effOS==1`）と主 OS 分岐（`effOS>1`）の双方
7. Float / Double 両経路
8. 既存 regression への波及

### 2.3 影響の大きさ（注意）

修正すると**全オーディオ経路のレベルが最大 +7.5 dB 変化**し得ます（ratio 8 で 0.75³ = −7.5 dB の補正）。
したがって推奨順序は
**修正設計レビュー → 影響予測 → test-only validation → production change（明示承認後）** です。

---

## 3. harness 欠陥（open・未修正）

| ID | 内容 | 影響 |
| --- | --- | --- |
| **F-2** | `--buzz-rigcheck=eq` の実効 AGC が意図と一致しない。`configureProbeFlatEQ`（1370）の AGC off を、直後の `setAutoGainStagingEnabled(false)`（1374）が `setAGCEnabled(!enabled)`=`setAGCEnabled(true)` で上書き（`getEQProcessor()` は `uiEqEditor` そのもの） | 「EQ AGC off で測定」という前提が実効状態と一致しない。**B-1 の原因ではない**（AGC 強制 OFF でも結果不変） |
| **F-3** | EQ dry/wet 混合の潜在欠陥。`canBlendDry==false` のとき `out = wet × α`（dry 補償なし）。`dryBypassBuffer` が確保できないと遷移中にレベル低下 | 遷移時のみ。定常の B-1 とは別事象 |
| **F-4** | `--buzz-rigcheck=ir` は `setConvolverBypassRequested(true)` を残したまま IR を load するだけで `convBypassed` を解除しない → 出力は bypass blend の **dry コピー** | **wet 対照実験に使えない**。B-1 C2-D で「OS 段は無損失」と誤結論した原因。既存窓 `[0.880,0.897]` は dry 測定の基準 |

参考: `--buzz-rigcheck=eq` の単一 authority 化（C-5）と `filter_application_implementation_plan_20260918.md` の同期は完了済み（§E）。

---

## 4. テスト / 監査系（open）

### F-1. `src/tests/PublicationValidatorIsolationTests.cpp` — MIGRATE CASES THEN RETIRE

- 520 行・`TEST_F` 34 + `TEST` 4 = 38 ケース。2026-06-03 追加、最終更新 2026-07-12
- **CMake 未登録**（`add_executable` 39 件に該当なし。2026-06-03 に登録され同日削除）
- **gtest 使用はリポジトリ内で本ファイルのみ**。CMakeLists / build.bat / presets に gtest 依存なし
- `checkNoConflictingTransitions` を **9 箇所で外部から呼ぶ**が現行 header では `private`。`FRIEND_TEST` は 0 件 → **コンパイル不可**
- 意味論は現行実装と一致。**`RuntimePublicationValidator` を参照する登録テストは 0 件**、`CrossfadeAuthority` の唯一のカバレッジも本ファイル
- 判定: **MIGRATE CASES THEN RETIRE**（移行方針は前スナップショット §F-1 に記載）
- `tools/build-debug.bat:29` も存在しない target を指定する stale 参照

---

## 5. 別課題（既存記録・本セッション対象外）

| ID | 内容 | 出所 |
| --- | --- | --- |
| **B-3** | timestamp-based capture。平均実効レート binding（`out.size()/2.0s`）は非一様 callback rate を完全補正しない。現行 transition probe の目的には十分（flipIndex 誤差 <0.5% 実測） | §B-3（別課題認定済） |
| **D-1** | build identity gate の M1/M2 欠陥（E-G3-3 既知・未修正）。M2: commit 毎に `source_revision` 変更 → COHERENCE-4 fail-closed（stamp 再作成で回避）。M1: ベアシェル/icx 起動時の cache 素字化 | §D-1 |
| **D-2** | headroom ランタイム統合（4 ランタイム混在）。uv tool `headroom-ai` 0.37.0 は extras=mcp のみで fastapi 無し → proxy 起動不可。Startup の .lnk / bat / vbs 3 起動体の整理 | §D-2 |
| **R-1** | `--buzz-flip-eqgain=` は値を受理して破棄（設計上ステップ固定）。silent-ignore 系の別パターン | §B-2 残観察 |
| **R-2** | `parseHcIdx` / `parseLcIdx` と `stod`/`stoi`/`stof` は不正入力で未捕捉例外 → terminate（診断メッセージ付き fail-closed ではない） | §B-2 残観察 |

---

## 6. 環境（対応済み・参考）

| ID | 内容 | 状態 |
| --- | --- | --- |
| **F-5** | Serena MCP の python LS 起動不能 | **修復済**。`solidlsp` が `uvx --from pyright==1.1.403 pyright-langserver` を起動するが PyPI pyright 1.1.403 の実体名は `pyright-python-langserver` → `program not found`。`.serena/project.yml` の `language_servers` を `python` → **`python_basedpyright`**（`basedpyright==1.39.9`、実行ファイル名一致）に変更し、3 LS（cpp / python_basedpyright / bash）起動完了・例外 0 を検証。`project.yml` は untracked。**稼働中 Serena の再起動が必要** |
| **F-6** | B-1 計測用 test-only 診断資産 | 保持中（O-1 の commit 判断待ち） |

---

## 7. CLOSED（対比列挙・再着手不要）

| 項目 | 根拠 |
| --- | --- |
| **A. push gate** | 全 commit push 済（当時）。以後の差分は O-6 参照 |
| **C / C-2. 未 commit 分の系列別整理** | C-1〜C-14 の commit に系列別分割済（合計 39 ファイルが監査値と一致）。untracked 三点仕分けの帰結も記録済 |
| **B-2. `--buzz-order=` silent fallback** | fail-closed 化 + regression test 8 ケース、`cte` 新規明示値導入。commit `c4a08171` |
| **B-1. EQ-on −5.17 dB の帰属（原因究明）** | **ATTRIBUTED / ROOT CAUSE LOCALIZED**。主因 `CustomInputOversampler`。詳細 §2.1。**修正のみ別 work item** |
| **`rigcheck=eq` criterion 不整合 / mirror 系 / boundary jump 等** | §E の既存 CLOSED 一覧（2026-09-19 以前） |
| **生成物/履歴の扱い** | `06794615`（post-series housekeeping、production 変更なし）は既知の履歴逸脱として固定。rewrite しない |

---

## 8. 推奨する次工程の順序

```
[判断待ち] O-1〜O-6（未 commit / 未 push の扱い）
      ↓
[別 work item] B-1 修正設計レビュー（§2.2 の 8 観点）
      ↓
[別 work item] B-1 修正の影響予測（+7.5dB・rigcheck 窓の再校正計画）
      ↓
[別 work item] harness 欠陥 F-2 / F-3 / F-4 の cleanup
      ↓
[別 work item] F-1 PublicationValidatorIsolationTests の移行・退役
      ↓
[別課題] B-3 / D-1 / D-2 / R-1 / R-2
```

---

## 9. 参照

- 前スナップショット（B-1 帰属の全測定記録・F-1〜F-3 詳細）: `doc/work113/residual_tasks_20260919.md`
- 本セッションの検証記録: 同ファイル §B-1（stage 別棄却 / 主因の実装式 / engine fit / 契約不整合）、§F-4〜§F-6
- 基準ソースの扱い: **B-1 の実測・検証は HEAD（C 系列適用後）の production source と HEAD からビルドした binary を対象**。`ConvoPeq(20260919-220828).md` は C-5 / C-8 / C-9 を含まないため B-1 の再監査基準としては不適切
