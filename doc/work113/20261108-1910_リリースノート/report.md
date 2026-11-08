# README のリリースノートを v0.6.12 に更新

- 実施: 2026-11-08 19:10〜
- 対象: `main` = `6f7779ea`
- 目的: `README.md` の `## New in v0.6.11` を `## New in v0.6.12` に変更し、1〜5 を commit `62a9513` から現在までの変更概要（5 項目まで）に差し替える。英語で記述する。

## 1. 対象範囲

- 起点: `62a9513`（2026-10-03 `docs(readme): update release notes to v0.6.11`）
- 終点: `6f7779ea`（2026-11-08 時点の `main`）
- 差分: 420 コミット、396 ファイル、+92,299 / −2,198 行
- 内訳（コミット種別）: docs 139、merge 95、test 53、fix 46、chore 33、ci 18、feat 15、ほか
- ソース変更: 96 ファイル、+18,485 / −739 行（`src/audioengine` 36、`src/tests` 36、`src/eqprocessor` 5、`src/core` 5、`src/convolver` 4 ほか）

## 2. 要約の作成方針

- リリースノートなので、コードと利用者に見える挙動の変更を優先し、文書・監査レポートの大量追加は項目に含めない。
- 5 項目に収めるため、関連する修正をテーマ単位でまとめた。
- 各記述は、該当コミットとコードを確認してから書いた（下記「裏取り」）。

## 3. 新しいリリースノート（英語）

1. Convolver bypass のテール保持: 専用 20 ms のクロスフェードで切り替え、途中で割り込んだフェードは現在のミックス値から 20 ms でやり直し、報告レイテンシとホスト PDC は不変。N1 の音声分析と前後比較を証跡として追加。
2. EQ 係数キャッシュの堅牢化: 無効バンドをキーで 0 に正規化し、ゲインの符号を保持。衝突探索を 64 スロットに拡張。復元された EQ 入力も通常編集と同じ範囲で検証。重複していた位相アンラップを共通化。
3. ISR/EBR のリーダと retire 寿命: 非 RT の EQ 状態リーダがスコープ付きエポックを取得し、公開 world のロード前にエポックへ入る。pin はインスタンス可視性ではなく登録スロットで解決。V11 のエンジン retire／リーダ再取得テストと STG-8/STG-10 の追加で契約を固定。
4. 破損サンプル境界の統一: `CorruptSampleGuard.h` の 2^53 上限をスカラーと AVX2 で共有。非有限入力を TruePeak の FIR 履歴に入れない。TruePeak の回帰テストを追加。
5. RT 計測・mixed-phase キャッシュ v3・ビルド整備: 固定長 RT 計測リングと非 RT 集計（モード 0/2/1、1 秒 drain）、v3 永続キャッシュの名前空間と状態／サンプルレート／失敗系の網羅、`app` 限定ビルド、clangd 用コンパイル DB、ヘッダ ODR チェック、preflight 検証、oneAPI キャッシュ付き Windows Release チェック、CI のオーディオテスト制御。

## 4. 裏取り

| 記述 | 確認先 |
|---|---|
| bypass 20 ms・現在値からの再開 | `ConvolverProcessor.h:226`（`kBypassCrossfadeTimeSec = 0.020`）、`ConvolverProcessor.Runtime.cpp:425-441`（`resetRT` + `applyImmediateValueRT(currentVal)`） |
| 報告レイテンシ／PDC 不変 | `ConvolverProcessor.Runtime.cpp:155,287` のコメント（PDC・報告基準は不変） |
| 破損判定 2^53 の共有 | `src/CorruptSampleGuard.h` の冒頭コメント（RES2-02、scalar/AVX2 の統一） |
| TruePeak の非有限入力 | `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp` の `checkTruePeakNonFiniteInput()`（#12） |
| キャッシュ探索 64 | `AudioEngine.h:2247`（`kMaxCollisionProbes = 64`） |
| 無効バンドの正規化・符号保持 | `EQProcessor.h` の `EQCoeffKey` 構築（無効時 0、`canonicalFloatBits`） |
| RT 計測モードと 1 秒集計 | `src/audioengine/RtMeasureRing.h`（モード 0/2/1、1000 ms ガード） |
| キャッシュ v3 の名前空間 | `src/MixedPhasePersistentCache.h:115`（`kCacheNamespaceV3`）と Phase 0/2 テスト |
| `app` 限定ビルド | `build.bat` の `TARGET_MODE`（PR #180） |
| ツール/CI 追加 | `scripts/gen_clangdb.py`、`tools/check_header_odr.py`、`scripts/preflight/*`、Windows Release ワークフロー |

## 5. 注意

- `ProjectMetadata.cmake` のバージョンは **0.6.11 のまま**（今回の指示は README の節のみ）。したがって本文の「ConvoPeq v0.6.11」表記は変更していない。0.6.12 としてリリースする際は、バージョン一元管理の更新が別途必要。
- 起点 `62a9513` は「v0.6.11 のリリースノートを書いた」コミットであり、そこから現在までは主に不具合修正・寿命強化・計測整備が中心である。

## 6. 変更ファイル

| 種別 | ファイル |
|---|---|
| 文書 | `README.md`（`## New in v0.6.11` → `## New in v0.6.12`、1〜5 を差し替え） |
| 報告 | 本ファイル |
