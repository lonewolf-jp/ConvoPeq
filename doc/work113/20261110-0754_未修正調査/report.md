# 未修正不具合の調査・設計（B-07 / B-11 / B-12 / B-15 / B-01・B-02）

## §0 作業前提

| 項目 | 内容 |
|---|---|
| 対象 HEAD | `c94d376772ee22916f8350cdaa33f8d3787d16bd`（ブランチ `main`） |
| 作業ツリー | 追跡 26 ファイル変更（+1,541 / −219）、未追跡 5 ファイル（`src/InputInspection.*`、テスト 3 本）を保持 |
| 未コミット変更 | 前回の B-03〜B-15 実装分を破棄せず、そのまま維持している |
| 本報告の範囲 | 調査と設計のみ。**ソースコードの変更は行っていない**（新規作成したのは本報告書だけ） |
| 実装の扱い | Owner の指示どおり、設計確認後に段階実装する。本報告はその確認材料である |

本報告では、指示された「問題が現在の HEAD で成立するか」を最初に確認し、成立しない記述があれば根拠付きで訂正する。断定できないものは未確定として残す。

---

## §1 B-07 — 学習器のデータレース

### 1.1 対象ファイルと関数

| ファイル | 関数 | 役割 |
|---|---|---|
| `src/CmaEsOptimizer.h` | `sample` / `update` / `serializeTo` / `deserializeFrom` / `setMean` / `setSeed` / `initFromParcor` | 最適化器の状態（`mean[9]`、`covariance[81]`、`sigma`、`rng`）を読み書きする |
| `src/NoiseShaperLearner.cpp` | `getState`（381–394） | `optimizer.serializeTo` を無ロックで呼び、状態をコピーする |
| 同上 | `setState`（396 以降） | `optimizer.deserializeFrom` で状態を書き戻す |
| 同上 | `workerThreadMain`（736 以降） | `sample`（816／946）、`update`（824／975）、`setSeed`（807）、`initFromParcor`（1123）を実行する |
| 同上 | `publishGenerationResult`（1468） | 学習スレッド上で `getState(snapshot)` を 1 回呼ぶ（B-07 の前回修正） |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | `notifyHealthyPublication`（433） | 441 行で `noiseShaperLearner->getState(current)` を呼ぶ |
| `src/audioengine/AudioEngine.Timer.cpp` | `executeRecoveryAction`（2004） | 2038 行で `noiseShaperLearner->setState(...)` を呼ぶ |

`CmaEsOptimizer` に mutex は無く、`mean` / `covariance` は生ポインタ、`sigma` は `double` である。クラス全体が「単一スレッドから使う」前提で書かれている。

### 1.2 問題の再確認（HEAD 上での到達可能性）

指示にあった「`getState` の定期保存経路が学習スレッドと並行する」という記述は、**HEAD では成立しない**。

- `notifyHealthyPublication` は `AudioEngine.h:1688` の宣言と `AudioEngine.CtorDtor.cpp:433` の定義しか存在せず、**呼出元が本番・テストともに 0 件**である（`grep -rn notifyHealthyPublication src` で 2 件のみ）。
- したがって 441 行の `getState` は現在デッドコードであり、定期保存経路として実行されない。

代わりに、**実際に到達する競合は `setState` 側**である。

- `executeRecoveryAction` は `m_healthMonitor.setActionCallback`（`AudioEngine.CtorDtor.cpp:86`）から呼ばれ、`RuntimeHealthMonitor` は「Message Thread only」と明記されている（同ファイル 23 行、673 行）。つまりメッセージスレッドで走る。
- 学習は `workerThreadMain`（専用スレッド）で走る。
- `setState` は `optimizer.deserializeFrom` で `mean[9]`・`covariance` 45 要素・`sigma` を書き換える。学習スレッドは同時に `update` で同じ配列を読む。両者に同期は無い。

さらに、`pendingMode`（`NoiseShaperLearner.h:274`）も同じ形のレースを持つ。

- `setLearningMode`（217 行）はメッセージスレッドから `pendingMode = mode` と書く（`AudioEngine.Learning.cpp:101/311`、`AudioEngine.Parameters.cpp:505`）。
- 学習スレッドは `handleModeSwitch`（321 行）で `pendingMode` を読む。
- `pendingMode` は `std::atomic` ではなく、`LearningMode` の単純代入である。

### 1.3 成立する実行順序（`setState` と学習）

```
[Message thread]                       [Learner thread]
executeRecoveryAction()
  setState(state)
    deserializeFrom(mean, cov, sigma)
      mean[0..8] ← 復旧値             update(candidates, fitness)
                                        oldMean[i] = mean[i]      // 旧値と新値が混在
                                        ...
                                        covariance[r*9+c] ← 新しい共分散
      covariance[...] ← 45 要素を上書き   // 上書きと計算が交差
      sigma ← 復旧値                     sigma を読む（新旧混在）
```

結果として、(a) `mean` の一部だけが復旧値になる、(b) `covariance` が非対称になる（`deserializeFrom` は対称に書くが、`update` の書き込みと交差する）、(c) `sigma` が `sigmaMin`〜`sigmaMax` の外に出る、という破損が起こり得る。破損した状態は次世代の候補生成に伝播し、学習の収束を乱す。クラッシュはしないため、症状は「学習が進まない」「スコアが跳ねる」として現れる。

`getState` 側も、将来 `notifyHealthyPublication` を配線した時点で同じ破損を読む。現在のデッドコードを根拠に「問題なし」とは扱わない。

### 1.4 根本原因

1. `CmaEsOptimizer` の状態が**スレッド所有の契約を持たない**（どのスレッドが触ってよいかの明文化が無く、実際にメッセージスレッドから触る経路がある）。
2. 復旧（`setState`）が「学習スレッドの安全な地点」ではなく任意のタイミングで適用される。
3. `pendingMode` など、同じ設計の非原子メンバが残っている。

「`getState` に mutex を足す」だけでは、(a) `setState` の競合、(b) `pendingMode` の競合、(c) 学習スレッド内の `sample`→`update` 間の一貫性、のいずれも解決しない。

### 1.5 修正設計

**案 A（推奨）: 学習器を学習スレッドに閉じ込め、復旧はコマンドとして渡す**

- `CmaEsOptimizer` は学習スレッドだけが触る、と契約を明文化する（ヘッダにコメントし、`getState` の呼出条件を「停止中のみ、または公開スナップショット経由」に限定する）。
- `setState`（メッセージスレッド）は optimizer に直接触らず、`std::mutex pendingRestoreMutex_` 配下の `std::optional<State> pendingRestore_` に置く。`std::atomic<bool> restorePending` で学習スレッドへ通知する。
- 学習スレッドは世代境界（`update` の直前）で `restorePending` を確認し、立っていればロックを取って 1 回だけ適用し、フラグを下ろす。適用は学習スレッド自身なので、`update` との交差は起こらない。
- 学習が停止している場合は、同じ mutex の下で「停止確認 → 即時適用」を行う。学習開始（`startLearning`）も同じ mutex を取って `Starting` へ遷移させることで、「停止確認後に学習が始まる」窓を閉じる。
- UI 向けの読み取りは、`publishGenerationResult` が学習スレッド上で作る `State` スナップショットを `std::shared_ptr<const State>` として原子的に公開し、`getState` はそれを返す（optimizer を直接読まない）。
- `pendingMode` は `std::atomic<int>` に変更する（`LearningMode` の値を int で保持）。

**案 B（比較対象）: `CmaEsOptimizer` 内に mutex を置く**

- `sample` / `update` / `serializeTo` / `deserializeFrom` / `setMean` の全体を 1 本の mutex で保護する。
- 実装は単純だが、`sample` は候補 18 個 × 9 次元の生成、`update` は 18 要素のソートと共分散更新を含み、ロック保持時間が長い。学習スレッドがロックを持っている間、メッセージスレッドの `setState` は待つ。復旧は緊急経路なので待たせること自体は許容できるが、**ロック順序**（`lastErrorMutex_` などとの組み合わせ）を新たに管理する必要があり、デッドロックの検討範囲が広がる。
- また、案 B でも「復旧値が次世代の `update` で上書きされる」という意味論の問題（復旧が即座に失われる）は残る。案 A なら世代境界で適用されるため、「復旧を 1 世代は使う」ことが保証できる。

**推奨**: 案 A。リアルタイムスレッド（オーディオコールバック）は `getState` も `setState` も呼ばないため、RT への影響は無い。影響するのはメッセージスレッドの `setState` が最大 1 世代分（現状の計測で数十 ms 程度）待つ可能性だけで、復旧経路としては許容できる。

### 1.6 専用テスト設計

| ID | 内容 | 判定 |
|---|---|---|
| R-07-1 | 学習スレッドを走らせたまま、別スレッドから `setState` を 100 回呼ぶ。各復旧後に `mean` が完全一致すること（適用回数の確認） | 適用回数 = 呼出回数、`mean` の最大誤差 0 |
| R-07-2 | 同上で、`covariance` が常に対称（`|c[i][j] − c[j][i]| ≤ 1e-12`）かつ `sigma ∈ [sigmaMin, sigmaMax]` であること | 全サンプルで成立 |
| R-07-3 | 復旧直後の 1 世代が復旧値から始まること（案 A の意味論） | 次世代の `mean` が復旧値と一致 |
| R-07-4 | `setLearningMode` を別スレッドから連続変更し、`handleModeSwitch` が取りこぼさないこと | 最後の指定値が反映される |
| R-07-5 | 保存→読み込み→再保存で `mean` / `bestCoefficients` がビット一致すること（B-15 と共通） | 完全一致 |

TSan はこの環境に無いため、R-07-1/2 は「破損の検出」ではなく「契約の確認」として実施する。TSan での確認は未実施として残す。

### 1.7 未検証・残存リスク

- TSan による競合検出は未実施（環境に無い）。
- `notifyHealthyPublication` の呼出元が無い理由（未配線か、削除予定か）は特定していない。配線する場合は本設計の公開スナップショット経由にすること。
- ASan では本競合は検出できない（メモリ破壊ではなく値の混在のため）。

---

## §2 B-11 — ISR 証跡生成元の未統合

### 2.1 データフロー（現状）

| 経路 | 生成元 | 出力先 | 内容 |
|---|---|---|---|
| アプリ `--verify-runtime-evidence` | `ISREvidenceExporter::exportEvidence(nullptr, 0)` | `current_path()/evidence` | Release は manifest のみ（実測 `artifacts: []`）。Debug は `debugArtifacts` 8 種 |
| CI generate | `isr-generate-runtime-evidence.ps1` | `evidence/` | シナリオ実行による実測値 |
| CI seed | `isr-seed-evidence.ps1` | `evidence/` | 固定の seed 値（strict では拒否される） |
| CI run | `isr-run-runtime-evidence.ps1` | `evidence/` | アプリを起動し、**足りない artifact を合成で補う** |

### 2.2 問題の再確認

1. **合成値が実測値と同じ来歴を名乗る**。`isr-run-runtime-evidence.ps1` の `New-FallbackArtifactObject` は、すべての合成 artifact に `provenance = "runtime"`、`status = "generated"` を付ける（同スクリプト 80–110 行）。実測との区別が証跡上に残らない。
2. **manifest の来歴が書き換わる**。`generator` が空なら `"isr-run-runtime-evidence.ps1"` を入れる（216–218 行）。アプリが書いた manifest に `generator` が無いため、実際にはアプリが生成したのに CI スクリプト名になる。
3. **manifest と実体が不整合になる**。合成後、`evidence/*.json` を列挙して `artifacts` を丸ごと上書きする（257–260 行）。アプリが書いた `artifactDetail`（`ISREvidenceExporter.cpp:438–444`。`source` と `measured` を持つ）は更新されないため、「`measured:false` なのに `artifacts` に載っている」「合成したのに `measured` が古い」という状態になる。
4. **`evidence/` の衝突**。アプリも CI も `evidence/` 直下に同じ名前で書く。同時実行すると互いに上書きする。run スクリプトは起動前に `*.json` を全削除する（11 行）ため、他プロセスの出力も消す。
5. **v5 / v6 / v7 の失敗理由**。Release のアプリ出力は manifest のみ（実測 267 バイト、`artifacts: []`）なので、`hb_graph_trace.json`・`shutdown_trace.json`・`retire_latency_report.json` が存在せず `Artifact not found` で exit 1 になる。run スクリプトはこの 3 件を合成で埋めるため、検証器は通るが、その証跡は実測ではない。

### 2.3 根本原因

「実測を出す主体」と「証跡を揃える主体」が分離されておらず、揃える役（run スクリプト）が**欠落を合成で埋める**設計になっている。来歴（provenance）の項目はあるが、合成を識別する値が定義されていない。

### 2.4 修正設計

1. **実行ごとのディレクトリ**にする。`evidence/runs/<runId>/` を単位とし、アプリは `--evidence-dir` で受け取る（既定は従来の `evidence/` で後方互換）。CI は `CONVO_ISR_EVIDENCE_ROOT` で切り替える。`evidence/.lock` を取り、同一 runId への同時書き込みを排他する。
2. **合成に識別子を付ける**。合成 artifact は `provenance = "synthesized"`、`status = "not_measured"`、`synthesizedBy = "isr-run-runtime-evidence.ps1"`、`sourceRunId` を持たせる。実測は `provenance = "runtime"`、`measured = true`。
3. **検証器を分ける**。v5／v6／v7 は「`provenance = "runtime"` かつ `measured = true` の artifact のみ合格」とする。合成値で通したい経路（seed・generate）は、それぞれの来歴を明示した別の受理条件にする。
4. **manifest を上書きしない**。`artifacts` は実体から再計算するが、`artifactDetail` も同時に更新し、`measured` は artifact 自身の `provenance` から導く。`generator` はアプリが書いた値を保持し、CI は `additionalGenerators` に追記する。
5. **整合性検証器を追加**する。`isr-verify-evidence-consistency.ps1` を新設し、(a) manifest の全 `artifacts` が存在する、(b) `artifactDetail` と実体の `provenance` が一致する、(c) `runId` が全 artifact で一致する、(d) `not_measured` が合格扱いされていない、を検査する。
6. **run スクリプトは合成をやめる**。足りない artifact は「失敗」として報告し、合成が必要なら明示的な `-AllowSynthesis` を付けた別モードにする（既定は不許可）。

### 2.5 CI 契約への影響

| 変更 | 影響 | 互換性 |
|---|---|---|
| `provenance` に `synthesized` を追加 | 既存の実測値は `runtime` のままで影響なし。合成を使う経路は不合格になる | 意図した破壊的変更。合成依存の CI は赤くなる |
| `--evidence-dir` の追加 | 既定は従来パスなので既存呼出は動く | 後方互換 |
| `isr-verify-v5/v6/v7` の合格条件 | 合成を拒否する分だけ厳しくなる | 実測経路は影響なし |
| 整合性検証器の追加 | 新規ジョブ。既存ジョブの合否は変えない | 追加のみ |

### 2.6 テスト設計

| ID | 内容 |
|---|---|
| E-11-1 | Release のアプリ出力を単独で v5／v6／v7 に通し、`Artifact not found` で失敗することを固定する（現状の失敗を回帰として保持） |
| E-11-2 | 合成 artifact を v5 に通し、`provenance=synthesized` を理由に失敗することを確認する（偽陽性の禁止） |
| E-11-3 | 同一 runId で 2 プロセスを同時に走らせ、排他で片方が失敗すること |
| E-11-4 | manifest の `artifacts` と実体が一致し、`artifactDetail.measured` が実体の `provenance` と一致すること |
| E-11-5 | generate／seed／run の 3 経路で、来歴が正しく記録されること |

### 2.7 未検証・残存リスク

- アプリ側で実測 artifact（HB シナリオ 4 件・shutdown・retire）を出すには、`--verify-runtime-evidence` でエンジンを起動する必要がある。本設計では「アプリは実測を出さない」現状を前提に、合成を明示する方向を採る。エンジン起動モードの追加は別設計とする。
- v5／v6／v7 を「実測でのみ合格」にすると、現状の Release 単体では必ず失敗する。CI の runtime ジョブが generate の実測に依存する形へ組み替える必要がある（ジョブ構成の変更は本設計の範囲外）。

---

## §3 B-12 — block サイズ依存の相対差

### 3.1 測定条件の再確認

- テスト: `src/tests/TailLayerIntegrityTests.cpp`。48 kHz、IR は指数減衰サイン（997 Hz）＋末尾マーカー、入力はインパルス、出力は `conv.Add` / `conv.Get` の倍精度。
- 指標 `eFixedDb` は **出力の後半**（`lag + irLen/2` 〜 `lag + irLen`）のエネルギー。`ref` は IR 自身の同区間のエネルギー。
- H-1 は block 256 を基準にした相対比較（≤16384、許容 0.1 dB）、H-6 は IR エネルギーに対する絶対比較（±6 dB）、H-4 は 3 秒 IR の絶対比較。
- 相対比較の対象は **≤16384 に限定**されており（406 行のループ）、32768 と 524288 は相対比較の外にある。この理由はコードのコメント（404 行）に「32768 では block 依存の差が出るため」と書かれている。

### 3.2 実測（今回の再測定）

`build/TailLayerIntegrityTests_artefacts/Release/TailLayerIntegrityTests.exe` を実行（exit 0、`[SUMMARY] failures=0`）。

**IR 1.0 秒（48000 サンプル）、tailMode=1（既定）**

| block | eFixed | ref | ref との差 |
|---|---|---|---|
| 64 | −1.15 dB | −1.98 dB | +0.83 dB |
| 256 | +1.18 dB | −1.98 dB | +3.16 dB |
| 4096 | +1.18 dB | −1.98 dB | +3.16 dB |
| 32768 | −1.23 dB | −1.98 dB | +0.75 dB |

**IR 3.0 秒（144000 サンプル）**

| block | eFixed | ref | ref との差 |
|---|---|---|---|
| 64〜512 | −11.21 dB | −12.04 dB | +0.83 dB |
| 1024〜32768 | −8.89 dB | −12.04 dB | +3.15 dB |
| 524288 | −12.04 dB | −12.04 dB | 0.00 dB |

F-5-audio（block 524288、IR 1 秒）は `eFixed = −1.98 dB`、`ref = −1.98 dB` で一致している。

### 3.3 既存報告値との相違（訂正）

- 「block 32768 で約 2.4 dB」は再現した。IR 1 秒で block 256（+1.18）と 32768（−1.23）の差が **2.41 dB** である。
- 「block 524288 で約 13 dB」は**再現しない**。524288 の eFixed は ref と完全一致（−12.04 dB、F-5-audio も一致）で、block 256 との差は 0.83 dB である。13 dB という値の出所は特定できていないため、本報告では**未再現の旧観測**として扱い、根拠に使わない。

### 3.4 原因（特定）

差分は層ゲインの値と完全に一致する。

- `MKLNonUniformConvolver.cpp:714–721`（tailMode=1）で `tailStrength = max(userTailStrength, 1.25)`、`strength01 = userTailStrength × 0.5`。
- 既定 `FilterSpec::tailStrength = 1.0` なので `tailStrength = 1.25`、`strength01 = 0.5`。
- `layer1Gain = 1.25 × (1.05 + 0.20 × 0.5) = 1.4375` → **+3.15 dB**
- `layer2Gain = 1.25 × (0.82 + 0.12 × 0.5) = 1.1000` → **+0.83 dB**

実測の差（+3.15 dB、+0.83 dB、0.00 dB）は、この 3 値と一致する。つまり:

1. L1／L2 の出力は「テールを足す信号」ではなく、**IR そのものの再構成の一部**である（`m_tailLayerGain[0]=1.0`、`[1]=1.4375`、`[2]=1.10` を層出力に掛けて加算する）。
2. どの区間が L0／L1／L2 に割り当てられるかは、`partSize`（`nextPowerOfTwo(max(block,64))`）と `tailL1L2Mult` から決まる。したがって **block サイズが層の被覆区間を変える**。
3. 後半区間が L1 に覆われれば +3.15 dB、L2 なら +0.83 dB、L0 だけなら 0 dB になる。3 秒 IR で 1024〜32768 が +3.15 dB、64〜512 が +0.83 dB、524288（L0 が IR 全体を覆う）が 0 dB という実測はこの説明と一致する。

**根本原因**: テール整形のゲインが「層」に固定されており、層の被覆区間が block 依存であるため、同じ IR でも block によって再構成のゲインが変わる。数値誤差ではなく、設計上の block 依存である。

### 3.5 仕様判断

- 「block サイズに依存しない」を仕様とするなら、ゲインを層ではなく**IR 上の時間位置**の関数にする必要がある。
- 「許容される差」を仕様とするなら、その差を明示し、相対比較の範囲と許容値を仕様書に書く必要がある。ただし今回は「原因特定前に許容閾値を緩和して隠す」ことが禁止されているため、この選択は原因を特定したうえでの仕様判断としてのみ許される。

本報告は前者（block 不変を仕様とする）を推奨する。理由は、H-1 の目的が「テール保持がブロック長に依存しないこと」であり、現状はその目的を満たしていないため。

### 3.6 修正設計

**案 A（推奨）: ゲインを時間位置の関数にする**

- 目標ゲイン `g(t)` を定義する。例: `g(t) = 1.0`（`t < tailStart`）、`1.4375`（`tailStart ≤ t < tailStart + L1span`）、`1.10`（それ以降）。境界は秒で固定し、block に依存させない。
- `SetImpulse` で各層の各パーティションについて、その開始サンプル位置から `g(t)` を評価し、**パーティション単位のゲイン**として保持する（現行の `m_tailLayerGain[3]` をパーティション別配列へ拡張）。
- 合成時はパーティションゲインを掛ける。これにより、層の被覆区間が block で変わっても、同じ IR 位置には同じゲインが掛かる。
- 影響: テールの聴感は block によって変わらなくなる。既存の `tailStrength` の意味（テールの強さ）は維持できる。CPU はパーティションごとの乗算が増えるが、既に `vdMul` でゲインを掛けている箇所（396–465 行）の係数を差し替えるだけで済む。

**案 B（比較対象）: 層の境界を block 非依存にする**

- L0 の被覆を「block に依らず 0.12 秒」に固定し、L1／L2 の境界も秒で固定する。パーティション数は block から決まるが、層の割り当ては時間で決める。
- 構造変更が大きい（`partSize` と層の対応が崩れるため、FFT サイズの制約とメモリ見積もりを再計算する必要がある）。

**推奨**: 案 A。

### 3.7 テスト設計

| ID | 内容 | 判定 |
|---|---|---|
| R-12-1 | block 64／256／1024／4096／16384／32768／524288 で、同一 IR の `eFixed` が基準（block 256）と ±0.1 dB 以内 | 全 block で成立 |
| R-12-2 | 中間サイズ（例: 2048、8192、65536、131072、262144）を追加し、境界付近で不連続が無いこと | ±0.1 dB |
| R-12-3 | 3 秒 IR でも同じ不変条件が成立すること | ±0.1 dB |
| R-12-4 | tailMode=2（Bypass）で従来どおり厳密再構成（maxErr ≤ 1e-10）であること | 変化なし |
| R-12-5 | チャネル数 2ch、サンプルレート 44.1／48／96 kHz、信号レベル −20／0 dB で不変条件が成立すること | ±0.1 dB |
| R-12-6 | 修正前の値（±3.15／±0.83 dB）を検出する再現テストを残す（修正後に落ちることを確認する） | 修正前は FAIL、修正後は PASS |

`(32768, 524288]` の全範囲は検証済みとしない。R-12-2 で数点を追加するが、全サンプル点の証明ではない。

### 3.8 未検証・残存リスク

- 実機の音声での確認は行っていない（数値のみ）。
- `tailL1L2Mult` を変更した場合の影響は未評価。
- 層の被覆区間の実測（どの区間がどの層か）は、テストの `NUPCTestAccess` で部分的にしか取れていない。設計実装時に層ごとの区間を記録する計測を追加する。

---

## §4 B-15 — 既存学習係数ファイルの移行

### 4.1 保存形式と変換履歴

- 形式: `<ConvoPeqLearnedState bestScore sampleRate bitDepth phase elapsedPlaybackSeconds>` に `<BestCoefficients c0..c8>` と `<CmaMean m0..m8>`（`saveLearnedState`、1402 行）。**version 属性は無い**。
- 変換履歴: B-15 の修正前は `publishGenerationResult` が `std::tanh(coeffs[i])` を適用していた（`git diff` で確認）。修正後は `parcor` をそのまま公開する。
- 係数の定義域: `CmaEsOptimizer::toParcor`（`CmaEsOptimizer.h:194`）は `parcor = tanh(unconstrained)` なので `|parcor| < 1`。
- したがって旧ファイルの `c_i` は `tanh(parcor) ∈ (−0.7616, 0.7616)`、新ファイルの `c_i` は `parcor ∈ (−1, 1)`。

### 4.2 問題の再確認

- `%APPDATA%\ConvoPeq\learned_state.xml` が実在する（804 バイト、確認済み）。
- `loadLearnedState` は `c_i` をそのまま `bestCoefficients` として公開する（1446 行以降）。旧ファイルなら、実際より小さい係数が復元される。
- version 属性が無いため、旧形式と新形式を確実に区別できない。**`|c| ≥ 0.7616` なら旧形式ではあり得ない**が、それ未満は曖昧である。

### 4.3 根本原因

保存形式に世代識別が無く、変換規則の変更（B-15）を既存データへ適用する手段が定義されていない。曖昧なファイルを「新形式」と決め打ちすると、旧データの意味が変わる。

### 4.4 移行設計

1. **version 2 を導入**する。`<ConvoPeqLearnedState version="2" conversion="parcor">` を保存時に付ける。新形式は `c_i = parcor` を明示する。
2. **version 無しは「legacyUnknown」として扱う**。自動変換しない。読み込み時に次のいずれかを行う。
   - 既定: 値をそのまま読み込み、`legacyUnknown = true` として UI とログに警告を出す（「古い形式の可能性があり、係数が実際より小さい可能性がある」）。
   - 明示操作（CLI フラグまたは UI の 1 回限りの操作）: `parcor = atanh(c)` で変換する。変換前に `learned_state.xml.bak-<timestamp>` を作成し、変換後の値を検証する（`|parcor| < 1`、`tanh(parcor) == c` を 1e-12 以内で確認）。
3. **曖昧性の扱い**: `|c| ≥ 0.7616` を含むファイルは旧形式ではあり得ないので「新形式」と判定してよい。それ以外は legacyUnknown とし、変換は明示操作に限定する。これにより「旧ファイルを無条件に新形式として読み込む」ことも「新ファイルを無条件に旧形式として変換する」ことも避ける。
4. **原子性**: 変換結果は一時ファイルに書き、`rename` で置換する。失敗時は元ファイルを残す。変換は冪等（version 2 を付けた後は再変換しない）。
5. **未知 version**: `version > 2` は読み込みを拒否し、現在のメモリ状態を保持する（既存値を壊さない）。`version < 2`（=1 が明示された場合）は旧形式として変換対象にする。
6. **CmaMean は変換しない**。`mean` は最適化器の生値であり、B-15 の二重変換の影響を受けていない（`publishGenerationResult` の変更は公開係数のみ）。この区別をコードとコメントに明記する。

### 4.5 テスト設計

| ID | 内容 |
|---|---|
| M-15-1 | 旧形式（version 無し、`c_i = tanh(parcor)`）を読み込み、legacyUnknown として警告が出て、値が変更されないこと |
| M-15-2 | 明示変換で `atanh` が適用され、変換後の `tanh` が元値と 1e-12 以内で一致すること |
| M-15-3 | 新形式（version=2）を読み込み、変換されないこと |
| M-15-4 | 変換→保存→再読み込みで値が変わらないこと（冪等） |
| M-15-5 | 未知 version（3）を拒否し、メモリ上の既存状態が保持されること |
| M-15-6 | 不正 XML・非有限値・`|c| ≥ 1` の処理（拒否または 0.0） |
| M-15-7 | 実ファイルのコピーを使い、元ファイルが変更されないこと |

実ファイルでの検証は `%APPDATA%\ConvoPeq\learned_state.xml` のコピーに対して行い、元ファイルは触らない。

### 4.6 未検証・残存リスク

- 実際の旧ファイルが二重変換済みかどうかは、値だけでは確定できない（`|c| < 0.7616` のため）。実ファイルの内容を統計的に見るだけでは決められない。
- 変換を自動化しないため、ユーザーが明示操作をしない限り旧値のまま使われる。これは安全側の選択だが、学習の初期値が小さいままになる。

---

## §5 B-01 / B-02 — 同期経路と LoaderThread の受理条件差

### 5.1 両経路の条件比較

| 条件 | LoaderThread（非同期・本適用） | 同期経路（`IRConverter::convert`） |
|---|---|---|
| チャネル数 0 拒否 | `admitChannelCountNonZero`（423 行付近） | あり（`IRConverter.cpp:279`） |
| 長さ 0 拒否 | `admitFileLengthNonZero` | あり（`:282`） |
| 長さ表現可能性 | `admitFileLengthRepresentable` | あり（`:285`） |
| チャネル数上限 | `admitChannelCount` | あり（`:288`） |
| バイト予算（1 GiB） | `admitByteBudget` | あり（`:291`） |
| 作業集合（同時確保） | あり | あり（`:302`、`admitWorkingSet`） |
| **末尾無音トリム** | `doTrimStep`（571 行、閾値 1e-15）で実行 | **行わない** |
| **FC-FORM-5 の評価対象** | trim 後の長さ（633–642 行） | ファイル全体の長さ（`loadedSamples`、`:396–404`） |
| resample 前の上限評価 | あり | あり |

### 5.2 差異の再確認

差は「トリムの有無」と「FC-FORM-5 の評価対象」の 2 点である。`IRLoadAdmission.h:96` は「FC-FORM-5 は L_trim に対してのみ評価する」と定義しており、LoaderThread はそれに従っている。同期経路はトリムを行わないため、同じ定義を適用できず、生の長さで評価している（コードのコメントにも明記されている）。

### 5.3 意図か不整合か

- LoaderThread がトリム後に評価するのは意図的である（コメントに「raw に適用すると UI は実効長で受理するが loader は拒否する回帰になる」とある）。
- 同期経路がトリムしないのは、プレビュー・キャッシュ用に元の波形を保持する必要があるためと考えられるが、**その結果として FC-FORM-5 の判定が本適用より厳しくなる**。これは意図した差ではなく、トリム未実装による副次的差である。

### 5.4 境界例（数値）

`admitResampleOutput` の上限は `MAX_IR_LATENCY + 1 = 2,097,153` サンプル（`IRLoadAdmission.h:49`）。

例: 48 kHz のファイルを 96 kHz で処理する場合、resample 後の長さは入力長の 2 倍。

- ファイル長 1,100,000、末尾 100,000 が無音（閾値 1e-15 未満）
  - LoaderThread: trim 後 1,000,000 → resample 後 2,000,000 ≤ 2,097,153 → **受理**
  - 同期経路: trim しない → resample 後 2,200,000 > 2,097,153 → **拒否**
- ファイル長 1,048,000（全体有音）: 両経路とも resample 後 2,096,000 ≤ 上限 → 受理
- ファイル長 1,050,000（全体有音）: 両経路とも 2,100,000 > 上限 → 拒否

つまり、末尾に 1e-15 未満の無音が十分長く、かつ全体長が上限を超え、トリム後が上限以下になる場合にだけ、プレビュー（同期）と本適用（非同期）で結論が分かれる。この条件を再現する WAV を作れば、差を実証できる。

### 5.5 設計案

**案 A（推奨）: 同期経路でもトリム後の長さで FC-FORM-5 を評価する**

- ただし同期経路は実際に全体長を resample するため、単に評価対象を変えると resampler の int 上限を守れない。したがって**同期経路でもトリムを実施**し、トリム後のバッファを resample する必要がある。
- 影響: プレビューとキャッシュの IR 長が変わる（無音テールが落ちる）。キャッシュキーにトリム後の長さが入るため、既存キャッシュは無効化される。表示上の IR 長も変わる。本適用と一致するので、遅延計算のずれは解消する。
- リスク: 「末尾マーカー」のような微小信号をトリムで落とす可能性がある（閾値 1e-15 は十分小さいが、無音とみなす基準を仕様として明記する必要がある）。

**案 B: 差を仕様化し、UI に明示する**

- 同期経路は「全体長で判定する厳格モード」と定義し、プレビュー時に「本適用では受理される可能性がある」ことを表示する。あるいは逆に、プレビュー段階で両方の判定を計算し、差がある場合に警告する。
- 変更は UI と判定の追加のみで、音声・キャッシュへの影響が無い。ただし「プレビューで拒否されたものが本適用で通る」という体験は残る。

**推奨**: 案 A。H-1 の目的（経路間の一貫性）に合致し、遅延・音声の一貫性も改善する。ただしキャッシュ無効化と表示長の変化を許容できることが条件である。

### 5.6 テスト設計

| ID | 内容 |
|---|---|
| A-01-1 | 無音テール付き WAV（上記の境界例）で、両経路の受理／拒否が一致すること |
| A-01-2 | テール長を 0／1000／100000 サンプルと変え、境界の移動を確認する |
| A-01-3 | 全体長が上限ちょうど／+1 のケースで両経路が同じ結論になること |
| A-01-4 | トリム後の長さが 1 サンプルになる極端なケース（全無音）で、両経路が同じ結論になること |
| A-01-5 | 正常な IR（無音テールなし）で、出力・遅延・メモリが変わらないこと（回帰） |
| A-01-6 | キャッシュのキーが変わり、古いキャッシュが使われないこと |

### 5.7 未検証・残存リスク

- 実音声での聴感確認は未実施。
- トリム閾値 1e-15 は「絶対値」であり、レベルが非常に小さい正当な IR では末尾が落ちる可能性がある。案 A を採る場合は、閾値を仕様として明記する必要がある。
- 無音テールを落とすことで、IR の末尾に意図的な無音（遅延）がある場合に、その遅延が失われる。これは本適用ですでに起きている挙動であり、案 A は同期経路をそれに合わせる変更である。

---

## §6 実装順序と承認依頼

指示の順序に従い、次の順で段階実装する。各段階は独立にコミット可能な大きさとし、原因の切り分けを保つ。

| 段階 | 対象 | 主な変更 | 検証 |
|---|---|---|---|
| 1 | B-07 | 学習器のスレッド契約、pending restore、スナップショット公開、`pendingMode` の原子化 | R-07-1〜5、既存 ctest、ASan |
| 2 | B-11 | evidence の run 単位化、合成の識別、検証器の分離、整合性検証器 | E-11-1〜5、CI 3 経路 |
| 3 | B-12 | パーティション単位の時間基準ゲイン、比較テストの拡張 | R-12-1〜6、ctest |
| 4 | B-15 | version 2、legacyUnknown、明示変換、バックアップ | M-15-1〜7 |
| 5 | B-01／B-02 | 同期経路のトリムと FC-FORM-5 の統一（案 A）または差の明示（案 B） | A-01-1〜6 |

**承認をお願いしたい点**

1. B-12 は「block 不変を仕様とする」（案 A）か「差を仕様化する」（案 B）か。推奨は案 A。
2. B-01／B-02 は「同期経路もトリムする」（案 A、キャッシュ無効化と表示長の変化を伴う）か「差を UI で明示する」（案 B）か。推奨は案 A。
3. B-15 の暗黙変換は行わず、明示操作に限定する方針でよいか（安全側）。
4. B-11 で v5／v6／v7 を「実測のみ合格」にすると、現状の Release 単体の runtime ジョブは必ず失敗する。CI ジョブ構成の変更まで含めてよいか。

---

## §7 未検証事項と残存リスク

- TSan は環境に無く、B-07 の競合検出は未実施。ASan では検出できない種類の欠陥である。
- B-12 の実機音声確認、`tailL1L2Mult` 変更時の影響は未評価。
- B-15 の実ファイルが旧形式か新形式かは、値だけでは確定できない。
- B-11 は「アプリが実測を出さない」前提の設計であり、エンジンを起動して実測を出す案は別設計。
- B-01／B-02 の聴感確認は未実施。
- 実行時障害の再現、メモリ使用量の実測、RES-06 は今回も未実施。`(32768, 524288]` の全範囲は検証済みとしない。

## §8 今回の変更

- 追加したのは本報告書のみ。ソースコード・CI スクリプト・設定は変更していない。
- 作業ツリーは §0 の状態のまま（追跡 26 ファイル変更＋未追跡 5 ファイル）で、既存の未コミット変更は破棄していない。
- main へのマージ、コミット、リリース、配布、公開告知、README / RELEASE_NOTES / manual の変更、RES-06 の解除は行っていない。
