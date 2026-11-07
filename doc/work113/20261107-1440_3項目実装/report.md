# 3項目（#1A・#12・N1）修正実装の作業報告

- 実施: 2026-11-07 14:40
- 基準: `main` = `ae91385d`（最新 `ConvoPeq.md` は `7faf9fa9` のソースを反映）
- 指示: Owner 指示（#1A・#12・N1 の修正実装。PR 作成まで。マージは別承認）

## 1. 対象と結果

| 項目 | 内容 | ブランチ | PR | 状態 |
|---|---|---|---|---|
| #1A | 係数キャッシュ探索上限 32 → 64（可用性改善） | `fix/eq-cache-probe-depth` | #166 | 作成済み・未マージ |
| #12 | TruePeakDetector の非有限入力ガード | `fix/truepeak-nonfinite` | #167 | 作成済み・未マージ |
| N1 | バイパス切替のテールクロスフェード（Issue #158） | `fix/bypass-tail-crossfade` | #168 | 作成済み・未マージ |

3 項目はそれぞれ独立したブランチ・PR で、互いに相手のファイルを変更しない。N1 だけが `01-residual-register.md` の N1 行を更新する（他 2 本は触らない）。

## 2. 実装前確認の要点

- 情報源の `ConvoPeq.md` と最新ソースを突き合わせ、実装箇所・呼び出し経路・現行動作・RT 制約・既存テストを各報告書に記録した。
- 相違点は次の 1 点のみ。`ConvoPeq.md` の記載は探索上限 32 で最新ソースと一致し、他は差異なし。
- #1A: `AudioEngine::EQCacheManager::getOrCreate` の 2 ループ。RT 側は `pinByHash` の単発参照で探索しない。
- #12: `TruePeakDetector::processBlock`。RT 呼び出しのため確保・ロック不可。
- N1: `ConvolverProcessor::process` のバイパス dispatch。既存の `activeMixSmoother` と equal-power ミックスを再利用できる。

## 3. 検証結果

### 基準（変更前）

- `main` = `ae91385d` の CI（Push on main）は success。既存テストは緑であることを基準とした。

### 項目別

| 項目 | 対象テスト | ハーネス全体 | ctest |
|---|---|---|---|
| #1A | 40 段衝突の探索 PASS、探索コスト 3.15 ns/段 | exit 0 | 54 / 54 PASS（355.5 秒） |
| #12 | 非有限入力ガード PASS | exit 0 | 54 / 54 PASS（334.7 秒） |
| N1 | テール連続性 PASS（`pre=3.677e-06`, `delta=6.979e-09`, `mean=2.157e-06`） | exit 0 | 54 / 54 PASS（326.4 秒） |

### N1 の回帰テストが修正を検出することの確認

- 変更前と同じ即時切替に戻して同じテストを実行すると FAIL:
  `[N1] FAIL: discontinuity at bypass (pre=0.000004 first=0.000010 delta=0.000006)`
- 変更後は PASS で、境界の段差が約 3 桁小さい。テストが空振りでないことを確認した。

### 静的解析・Sanitizer

- ローカル cppcheck 2.22（`--project` + `--file-filter`、変更 3 ファイル）: 終了コード 0。`TruePeakDetector.cpp` は指摘なし。JUCE マクロ起因の `unknownMacro` 設定通知が `AudioEngine.h` に 1 件出るが、これは環境側の既知事象。
- CI: CodeQL（c-cpp）、`lint-audioengine`、`authority-static-checks`、ASan（MSVC）、TSan（Clang）を含む 15 チェック。意図的失敗プローブのみ dispatch 限定で skipping。

## 4. 変更ファイル一覧

- #1A: `src/audioengine/AudioEngine.h`、`src/audioengine/AudioEngine.Cache.cpp`、`src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp`、`doc/work113/20261107-1020_1A実装/report.md`
- #12: `src/TruePeakDetector.h`、`src/TruePeakDetector.cpp`、`src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp`、`doc/work113/20261107-1110_12実装/report.md`
- N1: `src/convolver/ConvolverProcessor.Runtime.cpp`、`src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp`、`doc/work113/20261107-1230_N1実装/report.md`、`doc/work113/20261106-2030_残存対応/01-residual-register.md`

## 5. 対象外の確認

- #1B の再変更なし。他の残存課題の実装なし。無関係なリファクタリングなし。
- main への直接変更なし。PR の自動マージなし。
- `ConvoPeq.md` の再生成・納品は行っていない（マージ後の別工程）。

## 6. 残存リスク・未検証

- #1A: 64 段を超える衝突は従来どおり fail-closed。32〜64 段の衝突頻度は未計測。
- #12: 非有限値を 0 に置換する方針は運用上の選択。prepare 容量を超えるブロックでは置換できない。
- N1: フェード長は `smoothingTimeSec` 依存。実機の聴感評価は未実施。IR 差し替えクロスフェードとの同時発生は未検証。テストの絶対振幅が小さい（IR 後段フェードのため）ので、判定は相対値で行っている。
- N1 のバイパス定常経路には約 `1.9e-5`（-94 dBFS 相当）の残留があり、これは変更前から存在する。本変更では導入していない。
- 既存コメントに簡体字（線・発・潔・浄の簡体字）が混じる箇所が `main` に存在する。`find-nise-kanji` で検出したが、今回の変更範囲外のため修正していない。

## 7. Owner 判断待ち

1. 3 本の PR（#166 / #167 / #168）のマージ可否と順序。
2. #1A の上限値を 64 で確定するか。
3. N1 のフェード時間を既存の `smoothingTimeSec`（既定 20 ms）で確定するか、専用時定数を設けるか。
4. N1 の聴感確認をマージ前に行うか。

## 8. 実行した確認

- `cmake --build build --config Release --target AudioEngineHarness`
- `build\Release\AudioEngineHarness.exe`（`CONVOPEQ_SOURCE_ROOT`、`CONVOPEQ_ENABLE_TEST_HOOKS=ON` のビルド）
- `ctest --test-dir build -C Release --output-on-failure`
- `cppcheck --project=... --file-filter=*<file> --enable=warning,performance,portability`
- `find-nise-kanji`（3 報告書と変更ソース）
- GitHub Actions（Windows MSVC / Release、ASan、TSan、CodeQL ほか）
