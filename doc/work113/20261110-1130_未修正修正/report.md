# 未修正不具合の修正実装と再検証（B-07 / B-11 / B-12 / B-15 / B-01・B-02）

## §0 作業前提と範囲

| 項目 | 内容 |
|---|---|
| 対象 HEAD | `c94d376772ee22916f8350cdaa33f8d3787d16bd`（ブランチ `main`） |
| 作業開始時の作業ツリー | 追跡 26 ファイル変更（+1,541 / −219）、未追跡 5 ファイル（B-01〜B-15 の先行実装分） |
| 今回の変更 | 上記に追加して 9 ファイル変更・6 ファイル新規（§8。B-11 の実測モード追加を含む） |
| 未コミット | すべて未コミットのまま。マージ・コミット・リリース・配布・公開告知は行っていない |
| 実装順序 | B-07 → B-11 → B-12 → B-15 → B-01／B-02（指示どおり） |

各項目は独立に切り分けられる単位で実装した。B-07 と B-15 は同じファイル（`NoiseShaperLearner`）に触れるが、B-07 はスレッド所有権、B-15 は保存形式であり、変更箇所は重ならない。

---

## §1 B-07 — 学習器のデータレース

### 1.1 変更対象

| ファイル | 種別 | 内容 |
|---|---|---|
| `src/learner/LearnerStateAuthority.h` | 新規（109 行） | 復旧要求キューと公開スナップショットの同期契約。JUCE 非依存 |
| `src/NoiseShaperLearner.h` | 変更 | `stateAuthority_`、`pendingModeRaw`（atomic）、`learnedStateLegacyUnknown_`、内部メソッド宣言 |
| `src/NoiseShaperLearner.cpp` | 変更 | `getState` / `setState` / `captureState` / `applyRestore` / `publishCurrentSnapshot` / `isWorkerActive`、世代境界での復旧適用、開始遷移の直列化 |
| `src/tests/LearnerStateAuthorityTests.cpp` | 新規 | 同期契約の実スレッドテスト（R-07-A〜E） |
| `CMakeLists.txt` | 変更 | 新ターゲット `LearnerStateAuthorityTests` |

### 1.2 根拠（設計どおりの点）

- `optimizer` は学習スレッドだけが触る。メッセージスレッドは `setState` で「復旧要求」を渡すだけで、`deserializeFrom` を直接呼ばない。
- 復旧は世代境界（`optimizer.sample()` の直前）で 1 回だけ適用する。これにより「復旧直後の世代が復旧値から始まる」ことが保証される。
- UI 向けの読み取りは `publishGenerationResult` が学習スレッド上で作るスナップショットのみ。`getState` はそれを返し、optimizer を直接読まない（停止中のみ直接読みを許可）。
- `pendingMode` は `std::atomic<int>` にした（書き手はメッセージスレッド、読み手は学習スレッド）。
- 停止判定と `Idle → Starting` の遷移は同じ mutex（`LearnerStateAuthority::withExclusive`）で直列化したため、「停止中と判断して直接適用した直後に学習が始まる」窓が無い。

### 1.3 実装の要点

```cpp
// setState（メッセージスレッド）
if (std::this_thread::get_id() == workerThread.get_id()) { applyRestore(s); ... return; }  // 学習スレッド自身
stateAuthority_.withExclusive([&] {
    if (!isWorkerActive()) { applyRestore(s); appliedNow = true; return; }
    stateAuthority_.enqueueRestoreLocked(s);        // 最新 1 件のみ保持
});
```

```cpp
// 世代境界（workerThreadMain）
if (stateAuthority_.takePendingRestore(restoredState)) {
    applyRestore(restoredState);
    stateAuthority_.publishSnapshot(restoredState);
}
```

学習スレッド自身からの `setState`（multi-start の巻き戻し）は即時適用にした。ここを pending にすると自分の復旧が次世代まで反映されず、既存の再スタート動作が変わるためである。

### 1.4 テストと結果

`build/Release/LearnerStateAuthorityTests.exe`（Release、実スレッド）

| ID | 内容 | 結果 |
|---|---|---|
| R-07-A | 停止中の投入は保持されない | PASS |
| R-07-B1〜B3 | 5 回投入 → 最新 1 件だけ保持、境界で 1 回だけ取得、2 回目は空 | PASS |
| R-07-C1〜C3 | 2 スレッド × 2,000 回投入と境界取得。取りこぼし・二重適用なし、件数一致 | PASS |
| R-07-D | 20,000 回の公開と 200,000 回の読み取りで torn read 0 | PASS |
| R-07-E | `withExclusive` 内で積んだ要求が境界で取得できる | PASS |

`[SUMMARY] failures=0`（exit 0）。

### 1.5 未検証・残存リスク

- TSan は環境に無いため、データ競合の機械検出は未実施。ASan でもこの種の欠陥は検出できない。
- 実際の学習スレッドを走らせた状態での `setState` 同時実行は、テストでは「境界処理を模したスレッド」で代替した。実エンジンでの長時間運転は未実施。
- `notifyHealthyPublication` は依然として呼出元が無い（デッドコード）。将来配線する場合は `getState`（スナップショット）を経由すること。

### 1.6 他項目への影響

- B-15 と同じファイルに触れるが、変更箇所は独立（保存属性と version 判定）。`learnedStateLegacyUnknown_` は B-07 のメンバ追加に相乗りしただけである。

---

## §2 B-11 — ISR 証跡生成元の統合

### 2.1 変更対象

| ファイル | 種別 | 内容 |
|---|---|---|
| `.github/scripts/isr-run-runtime-evidence.ps1` | 全面改訂 | run 単位ディレクトリ、排他ロック、無条件削除の撤去、合成の明示化、manifest と artifactDetail の整合 |
| `.github/scripts/isr-verify-common.ps1` | 変更 | `provenance=synthesized` を不合格にする |
| `.github/scripts/isr-verify-evidence-consistency.ps1` | 新規 | manifest / 実体 / artifactDetail の整合性検査 |
| `.github/scripts/isr-generate-runtime-evidence.ps1` | 変更 | `evidence/*.json` の無条件削除を撤去 |
| `.github/scripts/isr-seed-evidence.ps1` | 変更 | 同上 |
| `.github/scripts/isr-run-tiered-verification.ps1` | 変更 | strict tier を「generate（実測）→ run（アプリ出力）→ consistency」へ変更 |

### 2.2 実装の要点

- **run 単位の出力**: `evidence/runs/<runId>/evidence/` にアプリ出力を集める。`evidence/` 直下は消さない。
- **排他**: `evidence/runs/<runId>/.lock` を `FileMode::CreateNew` で開く。既にあれば「別の run が書いている」として失敗する。
- **合成の識別**: `-AllowSynthesis` を付けたときだけ不足 artifact を作り、`provenance="synthesized"`、`status="not_measured"`、`measured=false`、`synthesizedBy`、`sourceRunId` を付ける。既定は合成せず、不足一覧を出して失敗する。
- **manifest と実体の整合**: `artifacts` を実体から再計算し、`artifactDetail`（name / provenance / status / measured）を同時に更新する。既存 artifact の `provenance` は書き換えない。
- **tiered 検証の経路**: strict tier は「generate（実測シナリオ）→ run（アプリ実測、run 単位ディレクトリ）→ consistency（run ディレクトリ）」の順に変更した。consistency へ渡すパスはアプリの書き出し先（`evidence/runs/<runId>/evidence`）に合わせた。
- **検証器の分離**: 実測を要求する検証器（v5/v6/v7 が使う共通部）は `synthesized` と `not_measured` を不合格にする。整合性検証器は「manifest と実体が一致しているか」だけを見る。両者は別の問いであり、片方の成功をもう片方の証明にしない。
- **アプリの実測モード**: `--verify-runtime-evidence-live` を追加した。エンジンを通常どおり動かし、Release でも HB トレース・HB 違反レポート・shutdown トレース・retire タイムラインを実測で書き出す。従来の `--verify-runtime-evidence` はエンジンを起動せず manifest だけを書いていた。強制発行はフラグ（`convo::isr::setForceEvidenceEmission`）で行い、既定の Release は従来どおり発行しない。
- **HB レポートの status 整合**: 従来は `status` が「グラフの妥当性」だけを反映し、シナリオが fail でも `ok` になっていた。実測で `retire_delay=fail` / `status=ok` を確認したため、シナリオの合否も `status` に反映するよう修正した（fail があれば `violation`）。

### 2.3 テストと結果（ローカル実行）

一時ツリー（`.github/scripts` を複製、`build/` はジャンクション）で実行した。

| 手順 | 結果 |
|---|---|
| generate | exit 0、`evidence/*.json` 10 件 |
| その後 v5 / v6 / v7 | 3 本とも exit 0（実測シナリオ証跡で合格） |
| run（合成なし） | exit 1。不足 artifact を列挙して失敗（Release のアプリ出力は manifest のみ） |
| run（`-AllowSynthesis`） | exit 0、artifacts 10、**measured=0**（すべて合成として識別） |
| consistency（run ディレクトリ） | exit 0、`10 artifacts, measured=0` |
| 負例: `status=not_measured` の artifact を v5 に投入 | exit 1（placeholder として拒否） |
| 負例: `provenance=synthesized`（status は generated）を v5 に投入 | exit 1（`synthesized` として拒否） |
| 負例: 同じ runId の lock が存在する状態で run | exit 1（二重書き込みを拒否） |
| 実測モード（`--verify-runtime-evidence-live`、12 秒） | 21 artifact、**measured=5**（`hb_graph_trace`、`hb_violation_report`、`shutdown_trace`、`retire_timeline`、`mutation_fault_trace`） |
| 実測モードの consistency（run ディレクトリ） | exit 0（`21 artifacts, measured=5`） |
| 実測証跡を v5 に通す | exit 1（`HB violation report indicates violation`。`retire_delay` シナリオが fail） |

### 2.4 未検証・残存リスク

- 実 GitHub Actions での実行は未実施（この環境では CI を走らせられない）。
- strict tier は generate を呼ぶ構成に変えた。CI の実行時間は増える。generate が前提とする環境（runId など）は tiered スクリプトが設定する。
- アプリ自身の実測は 5 件（HB トレース、HB 違反レポート、shutdown、retire タイムライン、mutation トレース）まで。`closure_graph`・`payload_tier_report`・`retire_latency_report`・`runtime_budget_report`・`recovery_trace` は依然 placeholder または来歴なしで、measured には数えない。
- 実測モードの v5 は、この環境では `retire_delay` シナリオの fail により **不合格** になる（`status=violation`）。測定値どおりの結果であり、合格にしていない。その後の調査で、このシナリオは retire 遅延ではなく「HB イベントの隣接間隔 ≤ 50 ms」を測っていることが分かった（`doc/work113/20261110-1313_retire実測調査/report.md`）。判定条件は変更していない。
- 「v5/v6/v7 が通る」のは generate の実測によるもので、アプリ単体の実測ではない。この区別は manifest の `evidenceRunRoot` で辿れる。

### 2.5 他項目への影響

- なし（スクリプトのみ）。

---

## §3 B-12 — block サイズ依存の相対差

### 3.1 変更対象

| ファイル | 内容 |
|---|---|
| `src/MKLNonUniformConvolver.cpp` | テール整形ゲインを IR の時間位置で定義し、IR へ適用。層ゲインは 1.0 に固定（二重適用の回避） |
| `src/tests/TailLayerIntegrityTests.cpp` | H-7（block 不変の独立検査）と中間サイズを追加 |

### 3.2 実装の要点

- 修正前: `layer1Gain = 1.4375`、`layer2Gain = 1.1000` を層の出力に掛けていた。層の被覆区間は `l0Part = nextPowerOfTwo(block)` と `tailL1L2Mult` から決まるため、block が変わると「IR のどの区間がどのゲインで再構成されるか」が変わった。
- 修正後: `t = IR 上のサンプル位置` に対する区分関数 `g(t)` を定義し、IR の各サンプルへ先に掛ける（Direct ヘッドも同じ）。
  - `t < tailStart`（0.12 s）→ 1.0
  - `t < tailStart + kL1MaxParts × 参照 L1 パート長` → `layer1Gain`
  - それ以降 → `layer2Gain`
- 参照 L1 パート長は block 256 のジオメトリから計算する固定値（`kL1MaxParts = 64`、48 kHz で 1.365 秒）。**block に依存しない**。
- 結果として、実効 IR が `g(t)·h(t)` になり、どの分割で畳み込んでも同じ出力になる。
- 読み出し側の `m_tailLayerGain` は 1.0 にした（IR へ掛けたゲインとの二重適用を防ぐ）。
- 期待するゲイン分布は `src/MKLNonUniformConvolver.h` の `FilterSpec` 直後に仕様として明記した（境界は秒と参照ジオメトリで固定、block 非依存）。

### 3.3 修正前後の実測

`TailLayerIntegrityTests` の `eFixed`（出力後半のエネルギー）。

**IR 1.0 秒**

| block | 修正前 | 修正後 |
|---|---|---|
| 256（基準） | +1.176 dB | +1.176 dB |
| 4096 | +1.176 dB | +1.176 dB |
| 32768 | **−1.228 dB** | **+1.176 dB** |
| 524288 | −1.976 dB | +1.176 dB |

**IR 3.0 秒**

| block | 修正前 | 修正後 |
|---|---|---|
| 64〜512 | −11.21 dB | −11.21 dB |
| 1024〜32768 | **−8.89 dB** | **−11.21 dB** |
| 524288 | −12.04 dB（ref と一致） | −11.21 dB |

修正前の差は 2.41 dB（1 秒 IR の 32768）と 3.15 dB（3 秒 IR の 1024〜32768）で、層ゲイン +3.15 dB / +0.83 dB と一致していた。修正後は全ブロックで同一値になった。

### 3.4 テストと結果

| ID | 内容 | 結果 |
|---|---|---|
| H-7 | block 2048〜524288 の 9 点で基準（256）との差 ≤ 0.1 dB | 全点 PASS（差 0.000 dB） |
| H-4 | 3 秒 IR、block 64〜524288 の絶対帯（ref −6 dB 以上） | 全点 PASS |
| H-6 | 1 秒 IR、block 64/256/4096/32768 の絶対帯（±6 dB） | 全点 PASS |
| F-1 / F-2 | Bypass の厳密再構成（maxErr ≤ 1e-10） | PASS（maxErr 2.22e-16、tailDelta 0.000 dB） |
| 再現性 | 修正を一時的に戻すと H-7 が 32768 で 2.40 dB、65536〜524288 で 3.15 dB の差を検出して 5 件 FAIL | 確認済み（修正を戻して実行し、再度修正を適用） |

`[SUMMARY] failures=0`。

### 3.5 未検証・残存リスク

- 実機の聴感確認は未実施（数値のみ）。
- `tailL1L2Mult` を変更した場合、`g(t)` の第 2 境界は参照ジオメトリ由来のままなので、意図した音色と合わない可能性がある。境界を設定から導く改善は別課題。
- `(32768, 524288]` の全範囲を検証したわけではない。H-7 は 9 点（2048〜524288）の確認である。
- 修正前の絶対値（例: 3 秒 IR で −11.21 dB）は、修正前の 64〜512 の値と一致している。テールの聴感は「小さい block の従来挙動」に揃えた。

---

## §4 B-15 — 既存学習係数ファイルの移行

### 4.1 変更対象

| ファイル | 種別 | 内容 |
|---|---|---|
| `src/learner/LearnedStateMigration.h/.cpp` | 新規（41 / 124 行） | version 判定と明示移行（atanh、検証、バックアップ、原子的置換） |
| `src/NoiseShaperLearner.cpp` | 変更 | 保存時に `version="2"` / `conversion="parcor"`、読込時に version 判定と legacyUnknown 警告 |
| `src/NoiseShaperLearner.h` | 変更 | `learnedStateLegacyUnknown_` と getter |
| `src/MainApplication.cpp` | 変更 | `--migrate-learned-state`（明示操作 CLI） |
| `src/tests/LearnedStateMigrationTests.cpp` | 新規 | M-15-1〜13 |
| `CMakeLists.txt` | 変更 | 新ターゲット `LearnedStateMigrationTests`、`CONVOPEQ_ALL_SOURCES` へ追加 |

### 4.2 実装の要点

- 保存: `version="2"`、`conversion="parcor"`。
- 読込: version 無し → `legacyUnknown`（値は変更しない、警告ログ、`wasLearnedStateLegacyUnknown()` で参照可能）。version 1 → 同じく自動変換しない。version 2 → そのまま。version 3 以上 → 読み込み拒否（メモリ上の状態を壊さない）。
- 明示移行: `parcor = atanh(c)`。`|c| ≥ 1` と非有限は拒否。`tanh(変換後) == 元値` を 1e-12 で検証。バックアップ `<file>.bak-<ms>` を取り、一時ファイルへ書いて `replaceFileIn` で原子的に置換。失敗時は元ファイルを残す。
- `CmaMean` は変換しない（二重変換の影響を受けないため）。
- 自動変換は行わない。CLI フラグを明示したときだけ動く。

### 4.3 テストと結果

| ID | 内容 | 結果 |
|---|---|---|
| M-15-1 | 旧形式（version 無し）を移行し、バックアップを作る | PASS |
| M-15-2 | `version=2` / `conversion=parcor` が書かれる | PASS |
| M-15-3 | 変換値が元の parcor と一致（maxErr 0.000） | PASS |
| M-15-4 | バックアップが存在する | PASS |
| M-15-5 | `CmaMean` は変換されない | PASS |
| M-15-6 | 2 回目の移行は no-op（冪等） | PASS |
| M-15-7/8 | version 3 は拒否、元ファイル不変 | PASS |
| M-15-9/10/11 | `|c| ≥ 1` は拒否、元ファイル不変、一時ファイル残存なし | PASS |
| M-15-12/13 | 壊れた XML は拒否、元ファイル不変 | PASS |

`[SUMMARY] failures=0`。

### 4.4 未検証・残存リスク

- `--migrate-learned-state` を実際のユーザーファイルに対しては実行していない（明示操作であり、データを書き換えるため）。CLI 分岐はコードで確認し、移行ロジックは M-15 で検証した。
- 実在する `%APPDATA%\ConvoPeq\learned_state.xml` が旧形式か新形式かは、値だけでは判定できない。`legacyUnknown` として扱われる。
- version 属性を持たない「新形式」ファイル（本修正前に保存されたもの）も `legacyUnknown` になる。ユーザーが明示移行すると誤変換になり得るため、自動化していない。この曖昧性は形式に version が無かったことに起因し、完全には解消できない。

---

## §5 B-01 / B-02 — 同期経路と LoaderThread の受理条件

### 5.1 変更対象

| ファイル | 内容 |
|---|---|
| `src/convolver/IRTrailingSilence.h` | 新規（37 行）。末尾無音トリムの共通規則（JUCE 非依存） |
| `src/IRConverter.cpp` | 読込直後に共通規則でトリムし、トリム後の長さで FC-FORM-5 と作業集合を見積もる |
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | AVX 実装のトリム走査を共通関数へ置換 |
| `src/tests/AudioEngineHarness/IRLoadAdmissionTests.cpp` | T-1〜T-5（規則の単体）と A-01（同期経路の受理・拒否）を追加 |

### 5.2 実装の要点

- 共通規則: 閾値 `1.0e-15` を超える絶対値を持つ最後のサンプル + 1。2ch は大きい方。該当なしは 1 サンプル。
- 同期経路は読み込み直後にトリムし、以降（FC-FORM-5、作業集合、resample、キャッシュ）はトリム後の長さで扱う。
- これにより「プレビュー（同期）は全体長で拒否、本適用（非同期）はトリム後で受理」という差が無くなる。
- キャッシュキーは IR 内容の CRC と構成（fftSize / phaseMode / partitionSize / sampleRate）から作られる。ファイル内容は変わらないため、**意味論バージョン `kCacheKeySchemaVersion = 2` をキーに混ぜた**。version 1（トリム導入前）のキーとは一致しないので、古い無トリム IR を再利用しない。version 1 を明示すると従来キーになる（検証用）。
- 元ファイルは読むだけで書き換えない。

### 5.3 テストと結果

| ID | 内容 | 結果 |
|---|---|---|
| T-1 | 末尾無音を落とす（1000 → 600） | PASS |
| T-2 | 2ch は大きい方のチャネルで決める（800） | PASS |
| T-3 | 全無音でも 1 サンプル残す | PASS |
| T-4 | 無音が無ければ全長 | PASS |
| T-5 | 閾値ちょうどは無音扱い、超えると有音扱い | PASS |
| A-01 | 48k→96k、全体 1,100,000・末尾 100,000 無音 → トリム後で受理（resample 後 1,996,457 サンプル、上限 2,097,153 以下） | PASS |
| A-01b | 同じ長さで末尾に信号 → 全体長で拒否（メッセージは上限 2097153 の診断） | PASS |
| A-02 | 同じ無音テール入力で、プレビュー（`analyzeImpulseResponseFile` 96k）と本適用（`prepareToPlay` 96k）の受理結果が一致 | PASS |
| C-12a/b/c | キャッシュキー: v1 と v2 が異なる、既定は現行版、版が 2 以上 | PASS |
| 回帰 | IRConverter admission I-1〜I-6、既存 PC/PD、ハーネス全体、CacheManager C-1〜C-12 | PASS |

### 5.4 未検証・残存リスク

- 実音声での聴感確認は未実施。
- トリムで表示 IR 長とキャッシュ内容が変わる。既存キャッシュは自動で消えないが、キーが変わるため再利用されない（容量は B-08 の剪定に委ねる）。
- 閾値 1e-15 は絶対値であり、非常に小さい正当な信号を末尾から落とす可能性がある。仕様として明記した。
- 末尾に意図的な無音（遅延）がある IR では、その遅延が落ちる。本適用では従来からそうであり、同期経路をそれに合わせた。

---

## §6 全体検証

### 6.1 ctest（Release、`build/`）

**最終実行（本報告の結論に用いる結果）**

```
100% tests passed, 0 tests failed out of 60
Total Test time (real) = 377.38 sec
```

キャッシュキー版の追加後、および A-02（プレビューと本適用の一致）追加後に実行した結果である。60 件すべてが合格している。

**過去の実行（履歴・結論には用いない）**

同じテストスイートの過去の実行で、`HeadlessAudioPathVerification` が 1 件だけ失敗した回がある。これは判定不一致ではなく、スモークテストのタイムアウトである。

```
[CLI-SMOKE] Launch: .../ConvoPeq.exe
ConvoPeq did not exit within timeout (30s).
```

単独実行でも 30.60 秒で同じ失敗をした回がある。アプリのプロセス寿命は 26〜29 秒（実測 6 回で 25.9〜28.7 秒）で、テストのタイムアウト 30 秒に対して余裕がないため、実行環境の負荷で通過・失敗が分かれる。**最終実行では通過している**が、この不安定さは残っている（原因は調査 B を参照）。

**終了時間の内訳（調査 B の実測）**

`--cli-log-file` を付けずに実行し、`ConvoPeq.log` の各行の出現時刻を測定した。

| 区間 | 実測 |
|---|---|
| 起動（プロセス開始 → `Auto-exit scheduled`） | 12.5 秒 |
| 実行（スケジュール → `Auto-exit flush`） | 1.8 秒 |
| `releaseResources` のドレイン | 2.15 秒 |
| `DeviceSettings::saveSettings` | 10.24 秒 |
| `closeAudioDevice` / `engineProcessor.reset` / UI 破棄 | 41 ms |
| `~AudioEngine` の後始末 | 53 ms |
| `mainWindow.reset()` 完了 → プロセス終了 | 100 ms |
| 合計 | 26.9 秒（実測 26.937 秒。各値は丸めのため単純和とは 0.05 秒差） |

`saveSettings` の 10.24 秒は、`noise_shaper_learn.xml`（278,331 バイト）の読み込み・再構築に起因する。同ファイルを 38 バイトにすると `saveSettings` は 66 ms、プロセス寿命は 5.9 秒になった（起動側も +10.7 秒かかっていた）。デバイスクローズやエンジン破棄ではない。

以前の報告で「`~MainWindow` 後のエンジン破棄に約 18 秒」と書いたが、その後の実測ではエンジン破棄は 53 ms であり、**この記述は誤りだった**。18 秒は計測方法（区間の取り方）に起因する推定であり、実測では起動側の読み込みと `saveSettings` に帰属する。

### 6.2 個別テスト

| テスト | 結果 |
|---|---|
| `LearnerStateAuthorityTests`（新規） | 9/9 PASS |
| `LearnedStateMigrationTests`（新規） | 13/13 PASS |
| `TailLayerIntegrityTests`（H-7 追加） | `failures=0` |
| `AudioEngineHarness`（B-01 T-1..5・A-01 追加） | Passed（302.80 s、単独 ctest） |
| `InputInspectionTests` / `CacheManagerHeaderValidationTests`（C-12 追加） | Passed |

### 6.3 ASan（`build-asan-local`、RelWithDebInfo）

| ターゲット | 結果 |
|---|---|
| `LearnerStateAuthorityTests` | failures=0 |
| `LearnedStateMigrationTests` | failures=0 |
| `TailLayerIntegrityTests` | 27 件失敗、すべて `RES-F`（CPU 閾値）。`RES-F` 以外 0 件 |
| `AudioEngineHarness` | 終了コード 0 |
| `ERROR: AddressSanitizer` | 0 件 |

B-11 の実測モード（`MainApplication`・`ISREvidenceExporter`・`ISRDebugRuntime`・`AudioEngine.Commit`・`ISRHB`）の変更後に、`AudioEngineHarness` を ASan で再実行した（終了コード 0、`AddressSanitizer` 報告 0 件）。キャッシュキー版の変更後は `CacheManagerHeaderValidationTests`（C-12 を含む）を ASan で実行し、failures=0 だった。

A-02（プレビューと本適用の一致）を追加した後の ASan 再実行は、実行時間が長く（ASan 下のハーネスは 30 分超の見込み）、途中で停止した。停止までの出力に `AddressSanitizer` の報告は無い。A-02 はテストコードの追加であり、対象の `IRConverter` の変更は B-11 変更後の ASan 実行に含まれている。

### 6.4 B-11 のローカル検証

§2.3 の表のとおり。要点は次の 3 つ。

- 合成は `provenance=synthesized` / `status=not_measured` として識別され、`measured` には数えない。
- アプリの実測モード（`--verify-runtime-evidence-live`、12 秒）は 21 artifact のうち **5 件を実測**として書き出す（HB トレース、HB 違反レポート、shutdown、retire タイムライン、mutation トレース）。manifest と実体の整合性検証も通る。
- その実測証跡を v5 に通すと **不合格** になる（`retire_delay` シナリオが fail、`status=violation`）。測定結果をそのまま反映しており、合格にはしていない。

---

## §7 未検証事項と残存リスク（まとめ）

- TSan は環境に無く、B-07 のデータ競合検出は未実施。ASan では検出できない。
- 実 GitHub Actions での B-11 の実行は未実施。strict tier の構成変更はローカルで再現した範囲のみ確認した。
- アプリ自身の実測は 5 件まで（§6.4）。残りの artifact は placeholder または来歴なしで、measured には数えない。
- 実測証跡の v5 は `retire_delay` の fail で不合格になる。調査の結果、このシナリオは「retire 遅延」ではなく「HB イベントの隣接間隔 ≤ 50 ms」を測っており、期待と測定対象が一致していない（`doc/work113/20261110-1313_retire実測調査/report.md`）。失敗の事実は維持し、判定条件は変更していない。
- B-15 の CLI 移行は実際のユーザーファイルに対して実行していない。version 無しの「新形式」ファイルは `legacyUnknown` になり、明示移行すると誤変換になり得る（曖昧性は解消不能）。
- B-12 の聴感確認、`tailL1L2Mult` 変更時の境界の妥当性は未評価。
- B-01/B-02 の聴感確認は未実施。トリム閾値の妥当性は仕様として明記した。
- `HeadlessAudioPathVerification` はプロセス寿命が 26〜29 秒（実測 6 回で 25.9〜28.7 秒）で、30 秒のタイムアウトに対して余裕がない（実行によって通過・失敗が分かれる。最終実行では通過）。内訳は起動 12.5 秒、`saveSettings` 10.2 秒、ドレイン 2.15 秒、エンジン破棄 53 ms である（§6.1 と調査 B の報告）。主因は `noise_shaper_learn.xml`（278 KB）の読み込み・再構築で、5 項目の外側であり未修正のまま残す。
- 実行時障害の再現、メモリ使用量の実測、RES-06 は今回も未実施。`(32768, 524288]` の全範囲は検証済みとしない。

## §8 差分と権限

- 追跡ファイル: 36 変更（+2,362 / −541。B-01〜B-15 の先行分を含む）、新規 10 ファイル。
- 今回追加したファイル: `src/learner/LearnerStateAuthority.h`、`src/learner/LearnedStateMigration.h/.cpp`、`src/convolver/IRTrailingSilence.h`、`src/tests/LearnerStateAuthorityTests.cpp`、`src/tests/LearnedStateMigrationTests.cpp`、`.github/scripts/isr-verify-evidence-consistency.ps1`。
- B-01/B-02 のキャッシュキー版: `src/CacheManager.h/.cpp`（`kCacheKeySchemaVersion`）、`src/tests/CacheManagerHeaderValidationTests.cpp`（C-12）。
- B-11 の実測モードで変更したファイル: `src/MainApplication.cpp/.h`、`src/audioengine/ISREvidenceExporter.cpp/.h`、`src/audioengine/ISRDebugRuntime.cpp/.h`、`src/audioengine/AudioEngine.Commit.cpp`、`src/audioengine/ISRHB.cpp`。
- 変更したスクリプト: `isr-run-runtime-evidence.ps1`（全面改訂＋実測モード起動）、`isr-verify-common.ps1`、`isr-run-tiered-verification.ps1`、`isr-generate-runtime-evidence.ps1`、`isr-seed-evidence.ps1`。
- 既存の未コミット変更は破棄していない。すべて未コミットのまま。
- main へのマージ、コミット、リリース、配布、公開告知、README / RELEASE_NOTES / manual の変更、RES-06 の解除は行っていない。
