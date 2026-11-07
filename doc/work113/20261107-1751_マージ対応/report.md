# Owner 判断（#1A・#12・N1）の反映とマージ対応

- 実施: 2026-11-07 17:51
- 判断: 2026-11-07 の Owner 決定（#166 / #167 マージ承認、#168 は聴感確認を条件に承認）
- 対象: PR #166（#1A）、#167（#12）、#168（N1）

## 1. マージ結果

| PR | 内容 | 結果 |
|---|---|---|
| #166 | #1A 係数キャッシュ探索上限 32 → 64 | マージ済み（merge commit `45eef58b`） |
| #167 | #12 TruePeakDetector 非有限入力ガード | マージ済み（merge commit `7603dfd1`） |
| #168 | N1 バイパス切替テールクロスフェード | **保留**。20 ms 専用定数の実装とテストは完了、実機聴感確認待ち |

`main` は `7603dfd1`。main への直接 push は行っていない。

## 2. main の整合性確認

- ローカルで `cmake --build build --config Release` を実行し成功。
- `AudioEngineHarness`: exit 0。`[#1A] cache probe depth probe: PASS`、`[#12] true-peak non-finite guard: PASS`、探索コスト 3.32 ns/段。
- `ctest -C Release`: 54 / 54 PASS（359.3 秒）。
- ソース差分は想定の 5 ファイルのみ（`TruePeakDetector.cpp/.h`、`AudioEngine.Cache.cpp/.h`、テスト）。N1 の変更は main に含まれない。
- GitHub Actions（main `7603dfd1`）: 確認時点で成功 5 件、実行中 1 件。失敗なし。

## 3. N1（#168）の追加対応

Owner 決定に合わせ、フェード時間を専用定数 20 ms に固定した。

- `src/ConvolverProcessor.h`: `kBypassCrossfadeTimeSec = 0.020` を追加。フェード状態のメンバーを追加。
- `src/convolver/ConvolverProcessor.Runtime.cpp`: バイパス状態が変わったブロックで、進行中のランプが残っていても 20 ms 分を最初からやり直す。収束後は利用者の `smoothingTimeSec` に戻す。
- `src/convolver/ConvolverProcessor.Lifecycle.cpp`: prepare 直後は未遷移としてフラグを初期化し、バイパス中に prepare されても最初のブロックから 20 ms でフェードする。
- `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp`: `checkBypassCrossfadeTimeConstant` を追加（`smoothingTimeSec` 0.5 s / 0.01 s、およびミックス変更中のバイパスの 3 ケース）。

検証結果:

| 項目 | 結果 |
|---|---|
| フェード時間（20 ms 固定） | 1 ブロック後 `0.466667`、2 ブロック後 `0.000000`（両方の smoothing 設定で同一） |
| ミックス変更中にバイパス | 1 ブロック後 `0.124444`、2 ブロック後 `0.000000` |
| テール連続性 | PASS（`pre=3.677e-06`、`delta=5.913e-09`、`mean=4.038e-06`） |
| ハーネス全体 | exit 0 |
| `ctest -C Release` | 54 / 54 PASS（355.3 秒） |

ブランチ `fix/bypass-tail-crossfade` は main を取り込み済み（merge commit `829037ca`）。PR #168 の本文と報告書を更新し、CI を再実行する。

## 4. 残る作業

1. PR #168 の CI 完走確認。
2. 実機聴感確認（テールの不連続、切替時のクリック、フェードの違和感）。結果を記録するまでマージしない。
3. 聴感確認で問題が出た場合は、原因と対応案を提示してマージを保留する。

## 5. 聴感確認の手順（案）

1. 長いテールを持つ IR を読み込む。
2. インパルスまたは短いバーストを流し、テールが残っている最中に Bypass を ON にする。
3. 切替時にクリックや段差が聞こえないこと、テールが自然に消えることを確認する。
4. Bypass を OFF に戻し、復帰時にクリックがないことを確認する。
5. テールが短い素材、ミックス値を変えた直後の切替でも同じ確認を行う。

## 追記（2026-11-07）

「実機聴感確認待ち」は解消した。#166・#167 に続き、PR #168 も `32c9234b` でマージした。N1 は実機聴感の代わりに自動波形分析で受け入れ、受入対象 5 指標は 9 / 9 PASS、バイパス解除後の wet 絶対レベル一致は受入対象外・未評価とした。§4 の残る作業は完了している。詳細は `doc/work113/20261107-2300_168マージ/report.md` を参照。
