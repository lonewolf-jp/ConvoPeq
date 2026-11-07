# N1（PR #168）Bypass 切替の自動音声分析

- 実施: 2026-11-07 21:40
- 対象: PR #168（N1 Bypass Tail Crossfade）。比較は修正前 `main` の `ConvolverProcessor.Runtime.cpp` と本ブランチの実装。
- 目的: 主観聴感に依存せず、出力 WAV の波形分析で Bypass ON/OFF のクリック・段差・フェードを判定する。

## 1. テスト環境

| 項目 | 内容 |
|---|---|
| OS / ビルド | Windows 11 x64、MSVC Release、`CONVOPEQ_ENABLE_TEST_HOOKS=ON` |
| サンプルレート | 48,000 Hz |
| ブロックサイズ | 512 サンプル |
| 実行 | `AudioEngineHarness --n1-analysis <出力先> <post\|pre>` |
| 比較方法 | 修正後 = 本ブランチ。修正前 = 同じテストコードで `ConvolverProcessor.Runtime.cpp` だけを `main` の版に戻してビルド |

## 2. 入力・IR・操作条件

- IR: 1.0 秒の指数減衰（ピーク位置 100 サンプル、振幅 0.5、時定数 0.25 秒）。48 kHz / 2ch / 32-bit float WAV に固定して読み込む。
- 入力: 10 ms の 1 kHz サイン（Hann 窓、振幅 0.7）。IR 読み込み後の 20 ブロック（約 213 ms）を空けてから与える。
- 操作はすべてブロック境界に固定し、同じ入力・IR・タイミングで再実行できる。

| ケース | 切替（バーストからの経過） | 遷移数 |
|---|---|---|
| `on_160ms` | ON 160.0 ms | 1 |
| `on_202ms` | ON 202.7 ms | 1 |
| `onoff_202_405` | ON 202.7 ms → OFF 405.3 ms | 2 |
| `on_352ms` | ON 352.0 ms | 1 |
| `multi` | ON 160.0 ms → OFF 330.7 ms → ON 501.3 ms | 3 |
| `on_597ms` | ON 597.3 ms | 1 |

## 3. 計測項目と定義

| ID | 指標 | 定義 |
|---|---|---|
| T1 | 切替時の段差 | `\|x[s] - x[s-1]\|` ≤ 0.25 × `max(\|x[s-1]\|, \|ref[s]\|)` |
| T2 | 局所の最大段差 | 切替 ±256 サンプルの `max\|Δx\|`。修正前後の比較項目（post ≤ pre） |
| T3 | フェード長 | ミックス smoother のブロック境界値。ON は 1 ブロック後 0.466667、2 ブロック後 0。OFF は 0.533333、1.0 |
| T4 | フェードの単調性 | ON は非増加、OFF は非減少（許容 1e-3） |
| T5 | 非有限値 | 出力全体に NaN / Inf がない |
| T6 | 収束 | ON: 遷移後 2 窓（各 25 ms）の RMS 差 ≤ 10%。OFF: フェード後に wet が再開している（RMS > 1e-9） |

`ref` は同じ入力・IR で Bypass しない参照出力。

### 判定基準の根拠

- T1 の 0.25 は、既存の受け入れテスト `checkBypassTailContinuity` が使っている「切替前テールの 25%」をそのまま流用した。
- T3 の 960 サンプルは実装契約 `kBypassCrossfadeTimeSec = 0.020` × 48,000 Hz。許容はミックス値 0.0011（1 サンプル = 1/960 = 0.00104）で、約 ±1 サンプルに相当する。
- T4 は線形ランプの契約（単調）、T5 は非有限値の禁止。
- T6 の 10% は「バイパス定常は時間不変」という定義に対する運用閾値。実測の窓間差は 0.000 で、閾値には余裕がある。
- T2 は絶対閾値を設けず、修正前後の比較だけで判定する。

## 4. 結果

### 修正後（post）

| ケース | 遷移 | 段差 | 局所最大段差 | 1 ブロック後 mix | 2 ブロック後 mix | T1 | T3 | 総合 |
|---|---|---|---|---|---|---|---|---|
| on_160ms | ON | 2.0e-11 | 2.0e-11 | 0.466667 | 0.000000 | PASS | PASS | PASS |
| on_202ms | ON | 8.0e-12 | 8.0e-12 | 0.466667 | 0.000000 | PASS | PASS | PASS |
| onoff_202_405 | ON | 8.0e-12 | 8.0e-12 | 0.466667 | 0.000000 | PASS | PASS | PASS |
| onoff_202_405 | OFF | 1.0e-12 | 7.0e-12 | 0.533333 | 1.000000 | PASS | PASS | PASS |
| on_352ms | ON | 6.0e-12 | 6.0e-12 | 0.466667 | 0.000000 | PASS | PASS | PASS |
| multi | ON | 2.0e-11 | 2.0e-11 | 0.466667 | 0.000000 | PASS | PASS | PASS |
| multi | OFF | 1.1e-11 | 1.2e-11 | 0.533333 | 1.000000 | PASS | PASS | PASS |
| multi | ON | 6.0e-12 | 6.0e-12 | 0.466667 | 0.000000 | PASS | PASS | PASS |
| on_597ms | ON | 3.0e-12 | 3.0e-12 | 0.466667 | 0.000000 | PASS | PASS | PASS |

9 / 9 PASS。T4、T5、T6 も 9 / 9 PASS。

### 修正前（pre）

| ケース | 遷移 | 段差 | 局所最大段差 | 1 ブロック後 mix | 2 ブロック後 mix | T1 | T3 | 総合 |
|---|---|---|---|---|---|---|---|---|
| on_160ms | ON | 2.78e-09 | 2.78e-09 | 1.000000 | 1.000000 | PASS | FAIL | FAIL |
| on_202ms | ON | 1.81e-09 | 1.81e-09 | 1.000000 | 1.000000 | FAIL | FAIL | FAIL |
| onoff_202_405 | ON | 1.81e-09 | 1.81e-09 | 1.000000 | 1.000000 | FAIL | FAIL | FAIL |
| onoff_202_405 | OFF | 1.82e-10 | 1.82e-10 | 1.000000 | 1.000000 | PASS | FAIL | FAIL |
| on_352ms | ON | 1.52e-10 | 1.52e-10 | 1.000000 | 1.000000 | PASS | FAIL | FAIL |
| multi | ON | 2.78e-09 | 2.78e-09 | 1.000000 | 1.000000 | PASS | FAIL | FAIL |
| multi | OFF | 1.70e-09 | 2.78e-09 | 1.000000 | 1.000000 | PASS | FAIL | FAIL |
| multi | ON | 2.68e-10 | 2.68e-10 | 1.000000 | 1.000000 | PASS | FAIL | FAIL |
| on_597ms | ON | 1.52e-10 | 1.52e-10 | 1.000000 | 1.000000 | PASS | FAIL | FAIL |

0 / 9 PASS。T1 は 7 / 9、T3 は 0 / 9、T4 は 9 / 9、T5 は 9 / 9、T6 は 8 / 9。

### 比較

| 指標 | 修正前 | 修正後 | 差 |
|---|---|---|---|
| 切替時の段差（9 遷移） | 1.52e-10 〜 2.78e-09 | 1.0e-12 〜 2.0e-11 | 25〜226 倍小さい |
| 局所最大段差（post ≤ pre） | ― | ― | 9 / 9 で改善 |
| フェード長 | なし（mix が 1 のまま） | 960 サンプル（20.0 ms） | 契約どおり |
| T1 合格数 | 7 / 9 | 9 / 9 | +2 |
| T3 合格数 | 0 / 9 | 9 / 9 | +9 |
| 総合 | 0 / 9 | 9 / 9 | 全件改善 |

## 5. 制約・未確立

- 聴感評価は行っていない。本分析は波形の数値判定であり、主観的な「違和感」は対象外。
- テスト信号は合成 IR と 1 kHz バースト。実素材（楽曲・実 IR）での確認は別途。
- バイパス中は wet の畳み込みを止める既存設計のため、バイパス解除後の wet は「連続処理した参照」と同一レベルにならない。**OFF 後の絶対レベル一致は判定基準未確立**とし、受入基準から外した（N1 の対象外・既存設計）。フェード完了と wet の再開は確認している。
- T2 は絶対閾値を持たない比較項目のため、単独では PASS としていない。CSV の `T2_local` 列は参照局所段差との比較結果（情報）で、受入判定には使っていない。受入は「修正後の局所最大段差 ≤ 修正前」で判定し、9 / 9 で改善している。

## 6. 再現手順

```powershell
# 修正後
cmake --build build --config Release --target AudioEngineHarness -j 8
build\Release\AudioEngineHarness.exe --n1-analysis <出力先> post

# 修正前（同じテストコードで Runtime.cpp のみ main 版に戻す）
git checkout main -- src/convolver/ConvolverProcessor.Runtime.cpp
cmake --build build --config Release --target AudioEngineHarness -j 8
build\Release\AudioEngineHarness.exe --n1-analysis <出力先> pre
git checkout HEAD -- src/convolver/ConvolverProcessor.Runtime.cpp
cmake --build build --config Release --target AudioEngineHarness -j 8
```

必要な環境変数: `MKLROOT`、`IPPROOT`、`CONVOPEQ_SOURCE_ROOT`。

## 7. 成果物

- WAV: `post_<case>.wav`、`post_ref_<case>.wav`、`pre_<case>.wav`、`pre_ref_<case>.wav`（6 ケース）
- 計測: `metrics_post.csv`、`metrics_pre.csv`、`comparison.csv`
- グラフ: `plot_<case>_<transition>.png`（切替点と 20 ms フェードを表示）
- Dropbox: `/ConvoPeq-3items-20261107/N1-analysis/` に一式を配置

## 8. 判定

- 自動分析の受入基準（T1・T3・T4・T5・T6）は、修正後 9 / 9 PASS、修正前 0 / 9。
- 20 ms 専用フェードは、48 kHz・512 ブロックで 1 ブロック後 `0.466667`（ON）/`0.533333`（OFF）、2 ブロック後 `0`/`1.0` を実測し、契約と ±1 サンプルで一致した。
- 切替時の段差は修正前の 1/25〜1/226 に低下し、局所最大段差も 9 / 9 で改善した。
- 未確立は「OFF 後の wet 絶対レベル一致」の 1 点のみで、これは N1 の対象外（既存設計）。受入基準には含めていない。
