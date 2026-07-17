# ConvoPeq バグ改修計画書（v2 - レビュー反映版）

> 作成日: 2026-07-16 | 更新日: 2026-07-16
> 元資料: `doc/work74/bug.md`（自動生成バグ監査レポート）
> レビュアー: プロジェクトオーナー（ISR Runtime / RuntimePolicyEngine 等導入済みの現行設計を前提）
>
> **本計画書はレビューにより「100%確定」から「要追加検証」へと評価を修正した。
> 項目によっては現行ISR設計（Runtime Policy Engine / PublicationValidator /
> Orchestrator / HealthMonitor / Builder）との整合性を考慮し、却下または優先度変更を行っている。**

---

## 改修対象一覧

凡例:
- **■ 設計改善（Design Improvement）**: 品質改善効果が明確で、ISR設計とも整合する（3件）
- **❌ 追加改修不要**: 現行実装または設計変更により問題が確認されなかった項目（18件）

**サマリ**: 元バグレポート19項目に加え、レビュー過程でConvolverState・TruePeakDetectorの2項目を追加評価。全21評価項目についてコードレビューおよび設計レビューを完了し、**3件設計改善、18件は現行実装または設計変更により追加改修不要（Appendix Aに一覧）** と判断した。
**推定工数**: 設計改善分 2日

---

## ■ 設計改善（Design Improvement）

### FIX-01 (旧FIX-03): MKLスレッド設定のグローバル汚染

| 項目 | 内容 |
|------|------|
| **ファイル** | `MKLRealTimeSetup.cpp` |
| **元Issue** | [Minor] `mkl_set_num_threads(1)` + `_putenv_s("MKL_NUM_THREADS","1")` がグローバル |
| **リスク** | 将来 threaded MKL を利用する場合や、同一プロセス内の他コンポーネントとの共存性を考慮すると、プロセスグローバル設定を避ける方が望ましい（現状 `MKL_THREADING=sequential` のため実害は限定的） |
| **確定度** | ✅ 確認済み |
| **ISR整合性** | ISR Runtime設計では「Global Stateを極力避ける」が基本方針。プロセス全体のスレッド数設定はこれに反する |

**改修案**: プロセスグローバルな `mkl_set_num_threads(1)` を、スレッドローカルな `mkl_set_num_threads_local(1)`（oneMKL 2022+）に変更し、環境変数設定（`_putenv_s("MKL_NUM_THREADS")`）を除去する。

```cpp
// Before:
_putenv_s("MKL_NUM_THREADS", "1");
_putenv_s("OMP_NUM_THREADS", "1");
mkl_set_num_threads(1);
mkl_set_dynamic(0);

// After:
mkl_set_num_threads_local(1);      // 当該スレッドのMKL実行コンテキストに適用（他スレッド/プロセス不変）
// mkl_set_dynamic(0);              // 必要に応じて保持
// 環境変数 _putenv_s は除去（プロセスグローバルな影響を避ける）
```

**注意**:
- `mkl_set_num_threads_local` は Intel oneMKL 2022+ で利用可能。プロジェクトは oneAPI 2026.0 を参照しているため利用可能。
- 戻り値（保存された以前の設定）を使った復元処理は、バージョンによって動作が異なる可能性があるため、**本計画書ではコード例として示さない**。実装時に対象MKLバージョンのドキュメントを参照すること。
- `INTEL_MKL_VERSION` マクロでバージョン分岐を推奨。
- `mkl_set_dynamic(0)` もプロセスグローバル設定である。スレッドローカル相当のAPIは存在しないため、必要性を評価した上で保持・削除を判断する。
- **確認済み補足**: CMake では `MKL_THREADING=sequential`（シングルスレッドリンク）が指定されている。このモードでは MKL は内部でスレッドを生成しないため、`mkl_set_num_threads(1)`／`mkl_set_num_threads_local(1)` の実質的な効果は同一である。本改善の主眼は **環境変数 `_putenv_s` の除去（プロセスグローバルな影響の排除）** にある。

**優先度**: P1 | **推定工数**: 0.5日

---

### FIX-02 (旧FIX-04): LoudnessMeter K-weighting 係数のサンプルレート依存

| 項目 | 内容 |
|------|------|
| **ファイル** | `LoudnessMeter.h`, `LoudnessMeter.cpp` |
| **元Issue** | [Minor] K-weightingフィルタが `fs==48k` 固定係数 |
| **リスク** | 96kHz/192kHz で K-weighting 応答が乖離。EBU R128/ITU-R BS.1770-4 Annex B では bilinear transform によるfs適応が規定されている |
| **確定度** | ✅ 確認済み（`LoudnessMeter.h` L97-L104 で `static constexpr double kPreBiquad[5]`, `kRlbBiquad[5]` が固定値） |
| **ISR整合性** | 問題なし。LoudnessMeterは独立した計測コンポーネントで、ISR Runtimeと直接の依存関係なし |

**改修案**: ITU-R BS.1770-4 Annex B に従い、以下のアナログ伝達関数から bilinear transform（pre-warping 付き）でIIR係数をサンプルレート毎に計算する:
- **Stage 1 (Pre-filter, High-shelf)**: H(s) = (s² + C·s + 1) / (s² + D·s + 1), C=1.69065929318241, D=0.73248077421585
- **Stage 2 (RLB, High-pass)**: H(s) = s² / (s² + E·s + 1), E=1.99004745483398, fc=38Hz

**注意**:
- 係数計算は RBJ Cookbook（Audio EQ Cookbook）の公式ではなく、Annex B に明示されたアナログ伝達関数と双一次変換（bilinear transform）の定義に従うこと。両者は相似だが厳密には一致しない。
- **一次仕様**: ITU-R BS.1770-4 Annex B（アナログ伝達関数とbilinear transformの定義）
- **実装参考**: [libebur128](https://github.com/jiixyj/libebur128)（MITライセンス）— 全サンプルレート対応のK-weightingフィルタ係数再計算の参照実装。実装方針決定の参考として推奨。
  （EBU Tech 3341 はラウドネス測定の運用ガイドラインであり、フィルタ係数計算の一次仕様ではないことに注意）

**優先度**: P1 | **推定工数**: 1〜2日（テスト含む）

---

### FIX-03 (旧FIX-07): Denormal 対策の文書化

| 項目 | 内容 |
|------|------|
| **ファイル** | `OutputFilter.cpp`, `LoudnessMeter.cpp`, `DSPCoreFloat.cpp` |
| **元Issue** | [Minor] `ScopedNoDenormals` が一部パスで無効 |
| **確定度** | ✅ 確定（プロセス全体のFTZ/DAZは `MKLRealTimeSetup.cpp` で `_MM_SET_FLUSH_ZERO_MODE` により設定済み） |

**改修案**: 新規IIRフィルタ追加時の設計規約として `killDenormal` パターンを文書化。コード改修不要。

**優先度**: P2 | **推定工数**: 0.5日

---

## 改修スケジュール（推奨）

| Sprint | 区分 | ID | 内容 | 工数 |
|--------|------|----|------|------|
| N（今週） | 設計改善 | FIX-01 | MKLスレッド Local化 | 0.5日 |
| N（今週） | 設計改善 | FIX-02 | K-weighting fs対応（BS.1770 Annex B） | 1日 |
| N+1 | 設計改善 | FIX-03 | Denormal設計規約の文書化 | 0.5日 |

**合計推定工数: 2日（設計改善3件） — 追加改修不要の項目はAppendix Aに一覧**

---

## Appendix A: 現行実装・設計変更により追加改修不要と判断した項目一覧

凡例: ✅ 現行実装確認済み（コードレビューで問題なし確認） | ❌ 誤検出（報告書の幻覚／誤った前提） | 🔧 設計変更済み（別設計で代替解決済み）

### ✅ 現行実装確認済み（コードレビューで問題なし確認済みの項目）

| # | 元Issue | 確認結果 |
|---|---------|---------|
| 1 | [Critical-1] LockFreeRingBuffer | `static_assert`で2の冪強制、`(w-r)>=Capacity`で正しいSPSC、acquire/release完備 |
| 2 | [Critical-4] MKLハンドルリーク | `IppFFTPlanCache` が `unique_ptr` + `ippsFree` でRAII完備 |
| 3 | [Major-6] TruePeak正規化 | 二重正規化（合計1.0+センター0.5）を完全実装 |
| 4 | [Major-6] reset未クリア | `reset()` で `peakHold=0` + `upHistory[] clear()` |
| 5 | [Major-7] ディザシード固定 | 時間＋静的カウンタ＋SplitMix64でチャンネル独立 |
| 6 | [Major-7] NoiseShaper発散 | FIR型はBIBO安定。Lattice型は係数0.85制限＋状態±1e12制限 |
| 7 | [Major-10] EQProcessor | NUM_BANDS=20、全セッターに境界チェック存在 |
| 8 | [Major-12] ASIO完全一致 | `containsIgnoreCase` で部分一致。問題なし |
| 9 | [Major-13] CacheManager | キーにCRC64含む。CacheHeader version=2 |
| 10 | [Minor] CpuFeatureCheck | AVX2チェックでOS API+CPUID二重確認済 |
| 11 | [Major-8] IRAnalyzer 数値安全性【現行コード監査完了】 | `estimateMaxFrequencyResponseGain` 全数値演算レビュー済。log/sqrt/除算すべてにガード有り。現行実装の監査範囲では数値例外の経路は確認されなかった |
| 12 | [Major-11] NoiseShaperLearner 停止ロジック【現行コード監査完了】 | 停止機構二重化（atomic + jthread stop_token）。各反復で確認。GA/CMA-ES標準の「停止命令待ち」設計 |

### 🔧 設計変更済み（v2.1以降の設計変更で解決済み）

| # | 元Issue | 確認結果 |
|---|---------|---------|
| 13 | [Critical-4] FFTスケーリング漏れ | v2.1でIPP FFT換装済、`IPP_FFT_DIV_INV_BY_N` で自動正規化 |
| 14 | [Critical-5] 遅延不一致 | `isLinearPhaseFIR=true`（線形位相FIR）。latency式も正しい |
| 15 | [Major-8] AllpassDesigner極半径 | `unconstrainedToRho(x)=0.98*stableSigmoid01(x)`、上限0.98ハードコード済 |

### ❌ 誤検出（報告書の幻覚／誤った前提に基づく）

| # | 元Issue | 確認結果 |
|---|---------|---------|
| 16 | [Major-9] IRDSP::normalize | **存在しない関数**。IRDSP名前空間には `resampleIR` のみ |
| 17 | [Major-12] DeviceSettings COM | **直接のCOM操作なし**。すべてJUCE `AudioDeviceManager` 経由 |

### 参考: レビュー過程中に追加評価し却下と判断した項目

| # | 評価対象 | 却下理由 |
|---|---------|---------|
| — | IRAnalyzer 数値安定性（旧FIX-04） | 現行実装の監査範囲では、log/sqrt/除算はいずれもガードされており、数値例外へ到達する経路は確認されなかった |
| — | NoiseShaperLearner 停止ロジック（旧FIX-05） | 停止機構二重化（atomic+stop_token）。GA/CMA-ES系標準の停止要求型設計 |
| — | ConvolverState atomic pack（旧FIX-06） | SafeStateSwapperが既にEBR実装済み。ISR RuntimeはWorld切替が基本設計 |
| — | TruePeakDetector スレッド安全性（旧FIX-07） | JUCE AudioProcessorの通常ライフサイクルではprepareToPlay()とprocessBlock()は排他的に呼び出されることが前提となる。加えてISR RuntimeはWorld単位切替で対応 |

---

## Appendix B: レビューを受けての変更履歴

| 項目 | 初版 | レビュー後 | 理由 |
|------|------|-----------|------|
| IRAnalyzer | P0確定 → ▲追加監査 → 追加改修不要 | 現行実装を監査した結果、問題は確認されなかった | 全数値演算レビュー完了。ガード完備で現行実装の監査範囲では例外経路なし |
| NoiseShaperLearner | P0確定 → ▲追加監査 → 追加改修不要 | 現行実装を監査した結果、問題は確認されなかった | 停止機構二重化（atomic+stop_token）。標準的な停止要求型設計 |
| ConvolverState | P2要改修 | 追加改修不要 | ISR RuntimeのSafeStateSwapperとRuntimeWorld切替が既に対応済み |
| TruePeakDetector | P2要改修 | 追加改修不要 | ISR RuntimeのWorld切替設計によりDSPCoreごと差し替え。JUCEの呼び出し順序とも整合 |
| 合計工数 | 3.5日 | 2日（設計改善） | 分類変更により工数見直し。追加監査は完了し所要工数なし |
| 「100%確定」表現 | 記載あり | 削除 | 過大表現と判明 |
| 分類 | 実施推奨/追加検証/却下 | 設計改善/追加監査/却下 | レビューアー指摘により「バグ」から「設計改善」へ変更 |

---

## Appendix C: 確定調査に使用したツールとコマンド

```bash
# AiDex MCP（識別子検索）
aidex_query path="." term="LockFreeRingBuffer" mode="contains"
aidex_query path="." term="DftiHandle" mode="contains"
aidex_query path="." term="mkl_set_num_threads" mode="contains"

# serena MCP（パターン検索）
serena grep "RetireOverflowRing" src/
serena grep "observeMonotonicViolationCount_" src/

# WSL grep（追加検証）
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && grep -rn "E_NOTFOUND" src/audioengine/'
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && grep -rn "CoInitialize\|CoCreateInstance\|IMMDevice" src/'
```
