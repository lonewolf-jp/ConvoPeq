# 4 ドキュメントの実装整合・第 3 次（未検証領域の総当たり）

- 実施: 2026-11-08 18:35〜
- 対象: `main` = `c7ac4680`
- 目的: ルートの 4 ドキュメントを再度ソースコードと突き合わせ、1〜2 回目で触れていない領域の記述を修正する。英語で記述する。`README.md` の "New in v0.6.11" 節は対象外。

## 1. 方法

1. 「まだ検証していない主張」を先に列挙し、機械的に確認できるものから潰した。
   - 権限台帳の件数（`kFieldDescriptors` / `kRuntimeAuthorityInventory` / `kRuntimeReadAuthorityInventory`）
   - リング容量（`DiagEvent` 512、`XRunEvent` 64、`DSPHandleTable` 512、`PendingPublishRegistry` 64）
   - Convolver 分割フラグ 8 種の実在
   - MSVC / icx のコンパイラフラグ全文とリンカフラグ
   - CMake プリセット 3 + 2、VS Code タスク 77 件（68 ラベル）、起動構成 5 件
   - EQ の係数ハンドオフで使うシンボル（`currentStateBits`、`publishCurrentState`、`retireEQStateDeferred` 等）
   - オーバーサンプリング上限（96 kHz→8×、192 kHz→4×、384 kHz→2×、768 kHz→1×）
   - Dither の SR 帯域しきい値（46.05k / 72k / 144k / 264.6k / 529.2k）
   - アナライザの Hann 窓・平滑化 0.85・ピークホールド 1.0 秒・60/15/5 Hz
   - 起動時の AVX2 チェックが強制かどうか（`checkAVX2SupportAndWarn()` が MessageBox 後に中断）
2. 追加した 6 ファイル（`CorruptSampleGuard.h`、`ScopedEpochReader.h`、`IRLoadAdmission.h`、`IRTrimTestHooks.h`、`RtMeasureRing.h`、`IRRuntimeContract.h`）の役割説明を実ヘッダのコメントと照合した。
3. コード断片（`scaleBlockFallback`、`fastTanh`）を実装そのものと比較した。

## 2. README.md（6 件）

| 箇所 | 修正 |
|---|---|
| Dither 一覧 | 「Fixed 15th-order」は GUI 表記であり内部次数は 16 であることを明記 |
| Dither 共通性質 | 「すべて 64 バイト整列」→ 係数表・RNG リング等の実態に合わせ、補充は timer（メッセージ）スレッドと明記 |
| EQCoeffCache | 「refcounted」を削除（所有は `EQCacheManager` → `DSPHandleRuntime`、参照カウントなし） |
| 出力チェーン | 図の末尾に TruePeak/LUFS 計測 → ピークリミッタ（−0.5 dB / 1 dB ニー）→ ±1 dBFS ハードクランプ → 固定遅延を追加 |
| Notes | 「音声スレッドで乱数生成しない」→ Psychoacoustic のみリング、Fixed 4th/16th と Adaptive 9th は xoshiro256 を音声スレッドで生成 |

## 3. ARCHITECTURE.md（3 件）

| 箇所 | 修正 |
|---|---|
| §3.1 / 3.2 / 3.3 / 3.5 | 追加した 6 ファイルの役割を実ヘッダの記述に合わせて具体化（破損振幅の単一定義、非 RT 用 EBR リーダ、IR ロード受入契約、末尾フェード無効化フック、RT 計測リング、IR/ランタイム形状契約） |
| §6.4 | 学習器はコーディネータ 1 本 + 評価ワーカー最大 6 本。係数ハンドオフは `CoeffSet` 二重バッファ + 世代カウンタで、RCU ではない |
| §8 | 公開経路の重複文を統合し、プリセットの可搬範囲を明示（OS 倍率はデバイス設定側） |

## 4. SOUND_PROCESSING.md（9 件）

| 箇所 | 修正 |
|---|---|
| §2.1 | `scaleBlockFallback` のコード断片を実装そのもの（`vEnd` とスカラー末尾）に置換 |
| §3.3 | 「`killDenormal`」→ 実装どおり `fastAbs(acc) < kDenormThreshold` で 0 化し、非有限値は破損フラグを立てる |
| §10 | 図の入口を `processBlock(double)`（スタンドアロン構成）に修正 |
| §4.3 | `AudioBlock` のフィールド名を実体（`numSamples` / `sampleRateHz` / `adaptiveCoeffBankIndex`）に修正 |
| §6.4 | マスキング閾値の FFT は MKL DFTI ではなく Intel IPP（`MklFftEvaluator`） |
| §4.3 | 擬似コードの `irState` / `mklNUC` は存在しないため、`RCUReaderGuard` → `loadActiveEngine` → `StereoConvolver`（`nucConvolvers[ch]->Add/Get`）に修正 |
| §6 共通 | 64 バイト整列の範囲を実装どおりに修正（係数表・RNG リング・15tap の係数/誤差/RNG・lattice の係数/状態。4th のみ非整列） |
| §6.2 | 量子化は SSE2 ではなく **SSE4.1**（`_mm_round_sd`） |

## 5. BUILD_GUIDE_WINDOWS.md（1 件）

| 箇所 | 修正 |
|---|---|
| §6.2 | icx 手動ビルドから `--config Debug` を削除（icx は Release のみ。Debug 検証は MSVC） |

## 6. 検証

- 数値監査（KB・行数）: 不一致 0 件。
- 表の列数（ヘッダーとの一致）: malformed 0。
- コードフェンス: README 10 / ARCHITECTURE 20 / SOUND 16 / BUILD_GUIDE 42（すべて偶数）。
- `find-nise-kanji`: 4 ドキュメントとも偽漢字なし。
- `0.6.10` の残存: 0 件。
- 識別子監査（439 個）: 未発見は外部 API と `Class::member` 修飾による誤検出のみ。

## 7. 確認できなかった事項

- AMD CPU での icx バイナリ実行可否（環境がない）。
- オーバーサンプラ各段のループ単位ベクトル化の有無。
- Dither の RNG リングが実際に枯渇する頻度。

## 8. 変更ファイル

| 種別 | ファイル |
|---|---|
| 文書 | `README.md`、`ARCHITECTURE.md`、`SOUND_PROCESSING.md`、`BUILD_GUIDE_WINDOWS.md` |
| 報告 | 本ファイル |
