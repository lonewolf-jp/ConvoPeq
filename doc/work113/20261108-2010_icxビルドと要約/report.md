# Intel icx アプリ限定ビルドと 62a9513 以降の変更要約

- 実施: 2026-11-08 19:50〜
- 対象: `main` = `32de0a50`（バージョン 0.6.12）
- 目的: (1) `build.bat` で Intel icx の Release ビルドを行い、`ConvoPeq.exe` だけを生成する。(2) commit `62a9513` から現在までの主要な変更を最大 20 項目で要約する。

## 1. icx アプリ限定 Release ビルド

| 項目 | 結果 |
|---|---|
| コマンド | `build.bat Release icx app clean nopause` |
| コンパイラ | Intel icx（oneAPI 2026.0。`build.bat` が `setvars.bat intel64` を実行） |
| ターゲット数 | アプリのみ **139**（フルビルドは 678） |
| テストターゲット | ビルドなし（`test targets are skipped`） |
| Phase 8 | 実行なし（`Building Phase 8` の出現 0 回） |
| build identity gate | `[GATE-OK]`（`clean` で stamp を再作成） |
| 結果 | exit 0、`[SUCCESS] Executable created successfully.` |
| 成果物 | `build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe`（48,434,176 B、19:58:40） |
| FileVersion / ProductVersion | **0.6.12** / **0.6.12** |
| 所要 | 8 分 43 秒（19:49:57 → 19:58:40、クリーンから） |
| 新規のテスト実行ファイル | 0 件（`build-icx` 配下を mtime で確認） |

ビルドログ: `C:\Users\user\AppData\Local\Temp\opencode\icx0612\build-icx-app-0612.log`

## 2. 62a9513 以降の主要変更（20 項目）

- 範囲: `62a9513`（2026-10-03）→ `32de0a50`（2026-11-08）
- 差分: 420 コミット、396 ファイル、+92,299 / −2,198 行（うち `src/` は 96 ファイル、+18,485 / −739 行）

| # | 変更 | 内容 | 根拠 |
|---|---|---|---|
| 1 | バージョン 0.6.12 | `ProjectMetadata.cmake` のパッチ番号を 12 にし、`JUCE_APPLICATION_VERSION_STRING` から exe の版情報まで一貫させた。4 ルートドキュメントと CMake バナーも同期 | `27225a30` |
| 2 | `build.bat` のアプリ限定ビルド | `app` オプションを追加。`ConvoPeq` ターゲットのみ（139/678）をビルドし、Phase 8 を省略 | `18c4f571` |
| 3 | Convolver bypass のテール保持（N1） | bypass 遷移を専用 20 ms のクロスフェードに変更。割り込んだフェードは現在値から 20 ms で再開し、報告レイテンシとホスト PDC は不変。音声分析テストを追加 | `194af040`、`4108f315`、`N1BypassAudioAnalysis` |
| 4 | EQ 係数キャッシュの健全化 | 無効バンドをキーで 0 に正規化（#1B）、ゲインの符号を保持（#155）、衝突探索を 64 スロットへ拡張（#1A）、復元入力の非有限値を排除し位相アンラップを共通化（#156） | `7faf9fa9`、`1451a88e`、`db282548`、`f641a187` |
| 5 | 非 RT リーダの EBR スコープ | 非 RT の EQState リーダがスコープ付きエポックを保持（A-closure）。`ScopedEpochReader.h` を追加し、公開 world のロード前にエポックへ入るよう修正（R-13） | `230ba9f4`、`28cd3d1e` |
| 6 | DSPCore の pin ベース寿命保護 | pin 保持中は destroy を遅延（F1'/F1'-b）。pin 対象をインスタンス可視性ではなく登録スロットで識別（R-51）。エンジン retire とリーダ再取得の恒久テストを追加 | `34be42cd`、`V11EngineRetireProtectionTests`、`V11ReaderReacquireTests` |
| 7 | 破損サンプル境界の統一 | 2^53 の上限を `CorruptSampleGuard.h` の単一定義にし、スカラーと AVX2 のガードで共有（RES2-02） | `src/CorruptSampleGuard.h`、`CorruptSampleGuardTests` |
| 8 | TruePeak の非有限入力遮断 | NaN/Inf を検出器の FIR 履歴へ入れない（#12）。参照検出器との一致を回帰テストで固定 | `e5cd6900`、`checkTruePeakNonFiniteInput()` |
| 9 | RT 計測リングと集計 | 固定長の RT 計測リング（R-06/U1）、モード 0/2/1、共通サンプル領域、非 RT 集計（U2、1 秒 drain、`[RTM]` 出力）、呼び出し点の接続と DSP 本体／リング push の区間分離（D-02 案3） | `6f34cc1f`、`da472ec7`、`374d9323`、`9c89b1ea`、`45811d7f`、`RtMeasureRing.h` |
| 10 | mixed-phase 永続キャッシュ v3 | `MixedPhaseCacheV3` 名前空間と LRU 分類を導入し、状態・サンプルレート・失敗系・Phase 2 を専用テストで固定 | `PersistentCacheV3Phase0SR/State/Failure`、`PersistentCacheV3Phase2` |
| 11 | ISR ガバナンスのテスト拡張 | STG-8 の回復義務（+1,800 行）、STG-10/STG-11 の退避、V11 のエンジン retire（+3,502 行）とリーダ再取得（+583 行）を恒久テスト化 | `STG8RecoveryObligationTests`、`V11EngineRetireProtectionTests` ほか |
| 12 | 診断系の所有権と原子経路 | diag sink を engine と DSP ごとに所有（B-2）。シンクポインタを原子ヘルパ経由に統一。シャットダウン用テストシームを production から除外 | `66068da8`、`e8e83fc5`、`B2DiagSinkTests` |
| 13 | quarantine の CAS 判定 | assert を CAS の結果で判定し、deferred スロットをマーク（A2 → A3-a） | `A2A3aQuarantineTests` |
| 14 | Windows 必須コンパイルチェック | Windows/MSVC Release のコンパイル検証を CI に追加（R-47-a）。`ConvoPeq` と `AudioEngineHarness` を分割ビルド | `41fa8395`、`32932387` |
| 15 | CI の oneAPI と sanitizer | winget で oneAPI を導入し、MKL/IPP をキャッシュして導入時間を 10.7 分削減（R-101）。ASan ジョブを winget 導入で復旧（R-48） | `8561091d`、`45af6df3` |
| 16 | 文書のみ変更の CI スキップ | `changes` ジョブと job 条件スキップ、2 点リビジョン、非 ASCII パス対応（`-z`）、push の `before..sha` 判定、非必須ワークフローの `paths-ignore` | `91572463`、`e4e78c29`、`a1597bf8` |
| 17 | 実機計測の CI | VB-CABLE プローブを dispatch 専用で追加（R-50、判定基準と観測窓を調整）。H2R1 リリース計測ワークフローを追加（R-04） | `1df692a2`、`712847e4`、`7318e44f` |
| 18 | ビルド・検証ツール | clangd 用コンパイル DB 生成、ヘッダ ODR ゲート、3 層 preflight ガードと回帰テスト、MCP プローブ（秘匿情報のマスク付き） | `scripts/gen_clangdb.py`、`tools/check_header_odr.py`、`e289523b`、`scripts/mcp_probe.py` |
| 19 | ドキュメント整備 | 4 ルートドキュメントの実装整合を 3 回実施（README / ARCHITECTURE / SOUND_PROCESSING / BUILD_GUIDE）。`doc/work113` と `doc/residual-20261101` に計 250 ファイルの監査・設計資料を追加し、`ConvoPeq.md` を再生成 | `5109d613` ほか、`doc/` 250 追加ファイル |
| 20 | その他の修正群 | Windows C++ ビルドの復旧（R-46）、lint ベースラインの復旧、EQ ガード契約テストの修正、アトミックのドット呼び出し排除、STG-8 の flaky テスト安定化、F2 の ASan 偽陽性解消 | `7da877e8`、`2da5a18a`、`91d29903` |

## 3. 検証

- icx ビルド: `TARGET_MODE: app`、139 ターゲット、Phase 8 の実行 0 回、新規 `*Tests.exe` 0 件。
- 成果物: FileVersion / ProductVersion が 0.6.12、サイズ 48,434,176 B。
- 変更要約: すべて `git log` / `git diff 62a9513..HEAD` と該当ファイルで確認した。

## 4. 変更ファイル

| 種別 | ファイル |
|---|---|
| 報告 | 本ファイル（コード変更なし。ビルド成果物は `build-icx/` 配下で追跡対象外） |
