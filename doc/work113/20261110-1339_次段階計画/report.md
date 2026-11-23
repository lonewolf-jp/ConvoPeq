# 次段階計画: A（retire 遅延の測定設計）と B（関数単位の計測計画）

## §0 前提

- 本報告は**計画の提示のみ**である。ソースコード・テスト・閾値・設定の変更は行っていない。commit、merge、release、配布、公開告知、README／RELEASE_NOTES／manual の変更も行っていない。既存の未コミット変更は保持している。
- 今回の判断材料として、**リポジトリ外の一時ハーネス**（`C:\Users\user\AppData\Local\Temp\opencode\b0612\`）で測定を行った。リポジトリ内のソース・テストは追加・変更していない（追加したのは本報告書のみ）。
- 本報告の数値はすべて実測である。推定は「推定」と明記した。
- C（A-02 追加後の ASan）は前回で目的達成として受領済みであり、追加作業は行っていない（§3）。

## §1 A: `retire_delay` の新しい測定設計

### 1.1 現状の事実（前回調査の再確認）

| 項目 | 事実 |
|---|---|
| 判定式 | `checkRetireDelayBound(50'000'000)`（`src/audioengine/ISRHB.cpp:287`）。HB トレースの**隣接イベント間隔**が 50 ms を超えたら false |
| 観測不足 | イベントが 2 件未満なら `true`（無条件合格） |
| イベント種別 | 100→200（publish）、200→300（退役）、300→400（終了境界） |
| 200→300 の位置 | `AudioEngine::onRuntimeRetiredNonRt`（`src/audioengine/AudioEngine.Commit.cpp:470`）内。`recordHBEdge(200,300,...)`（485 行）の数行後に `runtimePublicationBridge_.retire(...)`（494 行）を呼び、ここで解放待ち行列に積まれる |
| 解放完了の記録 | **無い**。`200→300` は「退役要求（enqueue）の直前」であり、メモリ解放の完了時刻ではない |
| 1 実行の観測数 | 12 秒実行で publish 3 件・退役 1 件（イベント計 4 件）。30 秒実行でも同じ |

要点は 2 つある。1 つは判定式が「隣接イベント間隔」を見ていること。もう 1 つは、**そもそも解放完了の時刻が記録されていない**ため、現行トレースから retire 遅延を計算できないことである。

### 1.2 何を retire 遅延と呼ぶか（定義案）

| 定義 | 内容 | 実測例 | 評価 |
|---|---|---|---|
| **D1（採用案）** | 退役要求（`200→300` の時刻）から、実際の解放（`deleter` 実行）までの時間。epoch で対応付ける | 現行トレースでは**算出不能**（完了時刻が無い） | 解放のコストを直接測る。要求と完了の両方に epoch があるため対応付けが可能 |
| D2（参考値として併記） | 新しい epoch の publish から、旧 epoch の退役要求までの時間 | 0.002 ms（実測 #1→#2） | 「退役の追随時間」であり、解放コストではない |
| D3（不採用） | 同一 epoch の publish から、その epoch の退役要求まで | 150.08 ms（実測 #0→#2） | epoch の寿命を測る値であり、retire 遅延ではない |
| D4（不採用） | 隣接イベント間隔 | 最大 10,758〜13,819 ms | 現行の判定式。retire と無関係な区間を含む |

D1 を採用し、D2 を補助情報として同じレポートに出す。D3・D4 は retire 遅延の指標にしない。

### 1.3 記録点（実装時に変更する箇所。今回は未変更）

| 事象 | 記録位置 | 取得できる epoch |
|---|---|---|
| 退役要求（既存） | `AudioEngine.Commit.cpp:485` の `recordHBEdge(200,300,...)` | `world->generation` |
| 解放完了（同期経路） | `ISRRetireRouter::terminalReclaim(...)` の `deleter(ptr)`（`src/audioengine/ISRRetireRouter.cpp:708` 付近） | 引数 `epoch` |
| 解放完了（drain 経路） | `TerminalReclaimAuthority::drain(...)` の `e.deleter(e.ptr)`（同 `:209` と `:269`） | `Entry::epoch` |
| 集計（既存） | `recordWorldReclaim()`（同 `:710`、`:753`、宣言 `ISRRetireRouter.h:158`） | **epoch を持たない**ため、epoch 付きに拡張する必要がある |

実装時は「要求数」と「完了数」をカウンタでも持ち、実行終了時に一致することを検証する（取りこぼしの検出）。

### 1.4 判定（観測不足を合格にしない）

| 条件 | result |
|---|---|
| ペアが 0 件 | `insufficient` |
| ペアが必要数 `N_req` 未満 | `insufficient` |
| 要求数 > 完了数（未完了の解放が残っている） | `insufficient` |
| 最大遅延が 50 ms を超える | `fail` |
| 上記以外 | `pass` |

- `N_req` は暫定 3 とする。§1.6 の実測で retire 16 件が観測できることを確認済みであり、3 件は現実的な要求である。最終値は実装後のパイロットで確定する。
- **50 ms の閾値は変更しない。**
- `insufficient` を `pass` に読み替えない。`insufficient` の間は「未測定」として扱い、v5 を通過させない。

### 1.5 スキーマと CI への影響

**トレース（`hb_trace_v1`）**

- 追加する情報: 各イベントの `kind`（`publish` / `retireRequest` / `retireComplete` / `shutdownBoundary`）と、対応付け用の `epoch`。
- 検証器 `isr-verify-common.ps1`（122 行）は `schema` の一致と必須キーだけを見るため、**フィールド追加は後方互換**である。
- ただし `schema` 名を固定で照合するスクリプトがある（`isr-run-runtime-evidence.ps1:123`、`isr-verify-evidence-provenance.ps1:11`、`isr-generate-runtime-evidence.ps1:35`、`isr-seed-evidence.ps1:31`、`isr-verify-backlog-specfixed-residual.ps1:69`）。`hb_trace_v2` に上げる場合はこれらの同時更新が必要。

**レポート（`hb_violation_report_v1`）**

| 案 | 内容 | 変更する CI | 評価 |
|---|---|---|---|
| **案 1（推奨）** | v1 のまま。`scenarioResults` の各要素に `observations`・`required`・`reason` を追加し、`insufficient` は `result: "fail"` + `reason: "insufficient_observations"` として表現 | なし（`isr-verify-v5.ps1:72` の `pass`/`fail` 検査をそのまま満たし、`fail` で throw される） | 「合格にしない」要件を満たしつつ、CI 契約を変えない。`fail` に 2 種類の意味が入る点だけが欠点 |
| 案 2（将来） | `hb_violation_report_v2` とし、`result` に `insufficient` を追加 | `isr-verify-v5.ps1`（schema と許容値の 2 箇所）、`isr-run-runtime-evidence.ps1:124`、`isr-verify-evidence-provenance.ps1:12`、`isr-generate-runtime-evidence.ps1:48`、`isr-seed-evidence.ps1:44`、`isr-verify-validator-tiering.ps1:176`、`isr-verify-gate-wiring.ps1:959` | 意味が明確。変更範囲が広い |

推奨は案 1 で開始し、案 2 を別の変更として後続させること。案 1 でも「観測不足は v5 を通らない」という要件は満たせる。

### 1.6 観測数を確保する方法（既存 CLI のみ・コード追加なし・実測済み）

現状は 12 秒で退役が 1 件しか出ないため、`N_req = 3` を満たせない。publish を駆動する既存の CLI オプションを実測した（すべて `--verify-runtime-evidence-live`、変更なし）。

| 実行 | 引数 | イベント | publish | retire |
|---|---|---|---|---|
| 1 | `--cli-ir-reload-count 6 --cli-ir-reload-interval-ms 700 --cli-exit-ms 8000` | 4 | 3 | 1 |
| 2 | `--cli-intent-burst-count 8 --cli-intent-burst-interval-ms 400 --cli-exit-ms 7000` | 4 | 3 | 1 |
| 3 | `--cli-eq-band 0 --cli-eq-gain-db -3 --cli-debounce-ms 50 --cli-exit-ms 7000` | 4 | 3 | 1 |
| 4 | `--cli-bypass-burst-count 8 --cli-bypass-burst-interval-ms 400 --cli-exit-ms 7000` | 8 | 5 | 3 |
| 5 | `--cli-bypass-burst-count 16 --cli-bypass-burst-interval-ms 300 --cli-exit-ms 9000` | 4 | 3 | 1 |
| 6 | `--cli-bypass-burst-count 24 --cli-bypass-burst-interval-ms 250 --cli-exit-ms 12000` | **34** | 18 | **16**（epoch 1〜16） |

- **使える駆動は `--cli-bypass-burst-count` / `--cli-bypass-burst-interval-ms`** である。24 回・250 ms 間隔で **retire 16 件**が観測できた。`N_req = 3` は達成できる。
- IR リロード（1）と intent バースト（2）、EQ 変更（3）は retire を増やさなかった。実行 1 は IR 未ロードのため reload が実質 no-op だったと推定する（推定）。
- 実行 5 が振るわないのは、**起動に約 12.5 秒かかるため `--cli-exit-ms 9000` ではバーストが走る前に終了した**ためである（実行 6 は同じ起動時間で 12 秒待つので走る）。これは §2 の B の遅延と観測窓が干渉することを意味する。B を修正すれば起動が短くなり、観測窓は広がる。
- 実装時は、`--cli-bypass-burst-count` の回数と `N_req` の関係を CI の実行時間内で確認する。v5 の実行時間は延びる（12 秒以上の実行が必要）。

### 1.7 受け入れ基準（この設計自体の検証方法）

1. **パイロット**: §1.6 の実行 6 と同じ条件（`--cli-bypass-burst-count 24 --cli-bypass-burst-interval-ms 250 --cli-exit-ms 12000`）で、要求数と完了数が一致することを確認する。
2. **対応付け**: 要求数と完了数が実行終了時に一致すること。
3. **境界**: 49.9 ms と 50.1 ms の人工遅延で判定が分かれること。
4. **観測不足**: ペア 0 件の入力で `insufficient` になり、`pass` にならないこと。
5. **負例**: 60 ms の人工遅延で `fail` になること。
6. **単調性**: `steady_clock` の時刻が単調であること（既存検査を流用）。

### 1.8 実装時に変更するファイル（承認後）

`src/audioengine/ISRHB.h`、`ISRHB.cpp`（trace v2・判定）、`ISRDebugRuntime.cpp`（記録の追加）、`AudioEngine.Commit.cpp`（要求点に種別を付与）、`ISRRetireRouter.cpp` / `.h`（完了点と epoch 付きカウント）、`src/tests/`（境界・負例・観測不足）、必要なら §1.5 の CI スクリプト。

### 1.9 未確定事項

- 実機負荷時に retire 遅延が 50 ms を超えるかは、この設計でも「実機で測る」まで分からない。
- 解放が「同期経路」と「drain 経路」のどちらを通るかは実行条件で変わる。両方に記録点を置く前提で設計している。
- `N_req` の妥当値（パイロット待ち）。

## §2 B: 関数単位の計測計画

### 2.1 今回追加で行った測定（リポジトリ外・コード変更なし）

**(1) ファイル内容を変えた比較**（同一バイナリ、`--cli-run --cli-exit-ms 2000`、`%APPDATA%\ConvoPeq\noise_shaper_learn.xml` のみ差し替え。測定後に元ファイルを復元し、ハッシュ一致を確認）

| # | 条件 | 起動 (ms) | `saveSettings` (ms) | 総経過 (ms) |
|---|---|---|---|---|
| 1 | 元ファイル（278,331 B、`Bank_` 180 要素） | 13,092 | **10,218** | 27,451 |
| 2 | タグ名だけ変更（`Bank_`→`Bnk__`、同サイズ） | 11,956 | **9,990** | 26,034 |
| 3 | 子タグ名だけ変更（`State`→`Stat_`、同サイズ） | 12,662 | **10,690** | 27,624 |
| 4 | 1 要素・277 KB の単一属性（要素数 1） | 12,047 | **10,170** | 26,571 |
| 5 | 38 B | 1,741 | **41** | 5,912 |

**読み方**: 要素数・タグ名・階層に依存せず、**ファイルの総バイト数だけ**に比例して遅い。180 要素の再構築や名前照合が原因ではない。

**(2) 関数単体のベンチマーク**（リポジトリの `juce_core.cpp.obj` にリンクした一時プログラム。`computeMaxElementDepth` は `src/InputInspection.cpp:13` の実装を複製）

| 処理 | 278,331 B | 277,052 B |
|---|---|---|
| `File::loadFileAsString` | 0.95 ms | 0.77 ms |
| **`computeMaxElementDepth`** | **9,614 ms** | **9,014〜12,175 ms** |
| `containsIgnoreCase` ×2（DOCTYPE／ENTITY 検査） | 4.81 ms | 4.93 ms |
| `XmlDocument::parse` | **1.48 ms** | 0.93 ms |
| `XmlElement::toString` | 1.00 ms | 1.10 ms |

**(3) 文字走査の比較**（277,052 B、同じ文字を 1 回ずつ読むだけのループ。最終行の深さ計算のみ両ファイルで確認）

| 走査方法 | 時間 |
|---|---|
| 生ポインタ `toRawUTF8()` + 添字 | **0.13 ms** |
| `juce::String::operator[]` + 添字 | **9,134 ms** |
| `juce::String::operator[]` + `length()` を毎回評価 | 12,057 ms |
| `getCharPointer()` を進める | **0.14 ms** |
| 生バイトで深さ計算（案 A 相当） | **0.29 ms**（depth=3、文字走査版と一致） |

### 2.2 確定した原因

- 10 秒の実体は **`computeMaxElementDepth`（`src/InputInspection.cpp:13`）** である。ファイル読み込み・XML パース・書き出しはいずれも数 ms 以内で、180 要素の構築を含む全体（38 B 条件）でも 41 ms である。
- 同関数は `text[i]` / `text[j]` のように **`juce::String` を文字位置で 1 文字ずつ走査**している。JUCE の `String::operator[]` は UTF-8 バッファ上の文字位置を先頭から数える実装であり、全文走査が **O(n²)** になる。実測もこの挙動と一致する（277 KB で 9.0〜12.2 秒、生ポインタ走査なら 0.13 ms、`getCharPointer()` の前進なら 0.14 ms）。深さ計測は 2 ファイル・計 7 回で 9,014〜12,208 ms だった（同一バイナリ・同一環境でも 9.0〜12.2 秒の幅がある）。
- したがって、起動の +10.7 秒（`loadNoiseShaperState` → `parseXmlFile` → `inspectXmlFile`）と終了時の `saveSettings` +10.2 秒（`saveNoiseShaperState` → `parseXmlFile` → `inspectXmlFile`）は**同一関数の同一原因**である。
- 前回報告で候補として挙げた「XML パース」「180 要素の再構築」「`NamedValueSet` の線形探索」は、いずれも**実測で否定された**（パース 1.5 ms、再構築 41 ms、要素数に非依存）。前回の記述は候補の列挙であり、原因確定ではなかった。今回の測定で候補は 1 関数に絞られた。

### 2.3 計測対象（承認後にプローブを入れる関数）

| 関数 | 位置 | 計測する内容 |
|---|---|---|
| `computeMaxElementDepth` | `src/InputInspection.cpp:13` | 総時間、`i` の増分回数、`substring`/`indexOf` の呼出回数（計算量の確認） |
| `inspectXmlFile` | 同 `:210` | 読み込み・DOCTYPE 検査・深さ検査の 3 分割 |
| `parseXmlFile` | 同 `:255` | 検証とパースの分割 |
| `DeviceSettings::loadNoiseShaperState` | `src/DeviceSettings.cpp:912` | 起動側の総時間 |
| `DeviceSettings::saveNoiseShaperState` | 同 `:848` | パース・180 要素構築・書き出しの 3 分割 |
| `DeviceSettings::saveSettings` | 同 `:1010` | 総時間（既知の 10.2 秒との一致確認） |

### 2.4 計測ポイント

| # | 位置 | 出力 |
|---|---|---|
| P1 | `saveSettings` 入口／出口 | 総 ms |
| P2 | `saveNoiseShaperState` 入口／`parseXmlFile` 後／`writeTo` 前／出口 | 3 区間の ms |
| P3 | `inspectXmlFile` 入口／`loadFileAsString` 後／`computeMaxElementDepth` 後／出口 | 3 区間の ms |
| P4 | `computeMaxElementDepth` 内のループ回数と文字走査回数 | カウント |
| P5 | `loadNoiseShaperState` 入口／出口 | 総 ms |

### 2.5 計測方法

- `std::chrono::steady_clock` の RAII スコープタイマで、区間の ms を 1 行だけ `stderr` に出す（既存の `[DIAG]` と同じ経路）。
- 環境変数 `CONVOPEQ_DIAG_PROBE=1` のときだけ有効化し、既定は無効（通常運用のログを汚さない）。
- ビルドは `build/`（Release）を用いる。必要なら `RelWithDebInfo`。
- 計測は `--cli-run --cli-exit-ms 2000` で 3 回実施し、中央値を採る。

### 2.6 検証方法

1. P1 の値が、外部測定の `step 6 saveSettings` → `step 7` の差（10.2 秒）と ±100 ms で一致すること。
2. P3 の深さ検査区間が P2 の 99% 以上を占めること（今回の推定の再現）。
3. プローブ有効時と無効時の総経過が ±2% 以内であること（計測のオーバーヘッド確認）。
4. 既存テストの該当 2 ターゲット（`CacheManagerHeaderValidationTests`、`AudioEngineHarness`）が PASS すること（回帰確認）。

### 2.7 変更と復元の手順（承認後）

未コミット変更が多数あるため、`git checkout` や `git stash` は使わない。

1. 編集対象のファイルを `tmp/probe-backup/` にバイトコピーし、SHA-256 を記録する。
2. プローブを追加し、`git diff -- <対象ファイル>` の差分を `tmp/probe.patch` として保存する（計測内容の証跡）。
3. 計測を実行する。
4. 復元はバイトコピーから戻し、SHA-256 の一致を確認する。
5. `git status --porcelain <対象ファイル>` が編集前と同じ出力であることを確認する。
6. `tmp/probe.patch` とタイマ実装はリポジトリに残さない。

### 2.8 修正案（実装は別承認）

| 案 | 内容 | 期待効果 | 影響 |
|---|---|---|---|
| **A（推奨）** | 走査を `getCharPointer()` / `toRawUTF8()` ベースに変える（`computeMaxElementDepth` の局所修正） | 実測: 277 KB で **11.9〜12.2 秒 → 0.29 ms**。同じ 2 ファイルで depth が 3 と 2 に一致した。起動 12.5 秒 → 約 2 秒、`saveSettings` 10.2 秒 → 約 0.1 秒（推定） | 判定結果は変わらない（同じ文字を数える。宣言・コメント・CDATA の分岐も同じ）。テスト追加は境界確認のみ |
| B | 深さ検査をやめ、パース後のツリーで深さを測る | 走査 1 回分の削減 | 検証の意味が「生テキスト」から「パース結果」に変わる。仕様変更を伴う |
| C | スモークテストのタイムアウト延長 | なし | 症状を隠すだけ。採用しない |

保存形式の変更・差分保存・非同期化は提案しない。特に非同期化は終了時のデータ保全に影響するため、単独の高速化として扱わない（Owner 判断に従う）。

## §3 C: A-02 追加後の ASan

追加作業は行っていない。前回報告のとおり、A-02 を含む `AudioEngineHarness` の ASan は完走（終了コード 0、`ERROR: AddressSanitizer` 0 件）。`LearnerStateAuthorityTests` の 9/9、`LearnedStateMigrationTests` の 13/13、`CacheManagerHeaderValidationTests` の `failures=0` も受領済みの内容で変わらない。`TailLayerIntegrityTests` の 27 件は `RES-F`（CPU 閾値）であり、メモリ安全性の失敗でも ASan 全体の失敗でもない。

## §4 `AGENTS.md` への追記（今回の調査結果とは分けて報告）

- **位置**: ファイル末尾。既存の最終行（旧 561 行）の後ろに 2 節を追加した（現在 563〜575 行）。
- **内容**: 次の 2 節。

（1）「既知の性能問題 — 学習ノイズシェイパー状態ファイル（2026-11-10 実測）」
- `%APPDATA%\ConvoPeq\noise_shaper_learn.xml`（278 KB / 180 要素）の読み込み・再構築に起動 +10.7 秒、`saveSettings` +10.2 秒を要する。
- 38 バイトにすると `saveSettings` は 66 ms、プロセス寿命は 5.9 秒。原因の関数は未特定（当時の記述）。
- `HeadlessAudioPathVerification` の 30 秒タイムアウトに余裕がない。タイムアウト延長で隠さない。

（2）「B-11 v5 の `retire_delay` は retire 遅延を測っていない（2026-11-10 実測）」
- `checkRetireDelayBound` は隣接イベント間隔のみを見る。2 件未満は無条件合格。
- 10 秒超の間隔は retire エッジと次の publish の間であり、retire 遅延ではない。

- **今後の扱い**: 指示に従い、追記の拡張や関連ドキュメントへの展開は行っていない。なお（1）の「原因の関数は未特定」は、本報告 §2 で `computeMaxElementDepth` と確定した。`AGENTS.md` の当該記述の更新は、承認があれば次回行う（今回は変更していない）。

## §5 未確認事項

- 実機負荷時の retire 遅延（A の設計でも実機測定まで不明）。
- bypass バーストで retire 16 件を観測できることは確認済み。ただしバースト回数と retire 件数の関係（間引きの有無）と、CI 環境での安定性は未確認である。
- 修正案 A の効果は推定であり、実装後の実測で確認が必要。
- `computeMaxElementDepth` 以外に、`juce::String` を添字で走査している箇所が残っている可能性（横断的な確認は未実施）。
- TSan／LSan、聴感確認、ASan 対象外の実機音声デバイス経路。

## §6 承認をお願いする事項

1. **B の関数計測**: §2.3〜§2.7 の内容（対象 5 ファイル、P1〜P5、復元手順）でプローブを入れてよいか。今回の外部測定で原因はほぼ確定しているため、プローブは「アプリ内での再現確認」と「修正後の回帰確認」に使う位置づけになる。
2. **B の修正案 A**（`computeMaxElementDepth` の走査をポインタ走査へ変更）の実装可否。計測の承認とは分けて判断いただきたい。
3. **A の測定設計**: D1 の定義、完了イベントの追加、`insufficient` の扱い（案 1 か案 2 か）、`N_req` の決め方（パイロット先行）でよいか。
