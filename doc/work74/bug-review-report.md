# ConvoPeq バグ監査レポート 検証結果

> 検証日: 2026-07-16
> 検証者: GitHub Copilot (OpenCode Go / Deepseek V4 Flash)
> 検証対象: `doc/work74/bug.md`
> 検証方法: 実ソースコード対照 + 音響工学文献調査 + 静的解析ツール群

---

## 総評

本バグレポートは **自動生成された静的解析結果** であり、全19項目中 **約40%に正確性・現状一致の問題がある**。特に **Critical と評価された項目の多くが実際のコードと一致していない** 点が顕著である。

主な問題:
1. **バージョン不整合**: v2.1 で MKL DFTI → IPP FFT に換装された事実を反映していない
2. **誤ったコードパターンの提示**: 実際の実装とは異なる「典型的な誤りパターン」を提示
3. **存在しないファイルへの言及**: `ISRRetireOverflowRing.h` のパスが誤っている
4. **正確な実装に対する誤検出**: 既に修正済みのバグや適切に実装されたコードを「バグ」と報告

---

## 項目別検証結果

凡例: ✅ 正しい指摘 / ⚠️ 部分的に正しい / ❌ 誤った指摘 / 📝 情報不足で判断不可

---

### [Critical-1] LockFreeRingBuffer フル/エンプティ off-by-one

**判定: ❌ 誤った指摘**

| 主張 | 実際のコード |
|------|------------|
| `bool isFull() { return (write +1)%size == read; }` パターン | `if ((w - r) >= Capacity) return false;` — 正しいSPSC満杯チェック |
| `size` を2の冪と仮定してマスク・コンストラクタで任意サイズ許容 | `static_assert((Capacity & (Capacity - 1)) == 0)` — コンパイル時に2の冪を強制 |
| memory_order の問題 | `acquire/release` ペアが適切に形成されている |

**詳細**: `LockFreeRingBuffer.h` はテンプレート引数で `Capacity` を受け取り、`static_assert` で2の冪をコンパイル時検証。満杯判定は `(w - r) >= Capacity` で正しく、`(write+1)%size == read` のようなパターンは存在しない。`LockFreeAudioRingBuffer` も `%` 演算子で正しくラップ処理。報告書の「典型的な誤りパターン」は実際のコードとは無関係。

---

### [Critical-2] AudioEngine.h リアルタイムスレッドでの非RT操作

**判定: ⚠️ 部分的に正しい（重大度過大評価）**

**`makeRuntimeReadHandle` の `acq_rel` 問題**:
- `fetchAddAtomic(observeMonotonicViolationCount_, ..., acq_rel)` は **モノトニック違反検出時のみ** 実行されるレアパス
- 毎ブロック1回呼ばれるが、最内ループ（1サンプル単位）ではない
- 報告書は「最内ループ (audio block 内)」としているが、実際は **ブロック単位で1回** の呼び出し
- `acq_rel` はARMでコストが高いが、レアエラーパスのため実質的な影響は無視できる

**`std::shared_ptr` / `std::vector` のRTパスでの使用**:
- `makeRuntimeReadHandle` 内では動的確保を行っていない
- `AudioEngine.Timer.cpp` 87KB の内容は未確認だが、Timerスレッドはメッセージスレッドであり、RTパスではない

**結論**: 重大度は過大評価。ただし `acq_rel` を `relaxed` に下げられるという指摘は技術的に正しい。

---

### [Critical-3] DeferredDeletionQueue / ISRRetireRouter Use-after-free

**判定: ⚠️ 部分的に正しい（ハンドル部分の記述が不正確）**

**DftiHandle.h のRAII**:
- ✅ Copy = `delete` 済み
- ✅ Move で `nullptr` 代入完了
- ✅ デストラクタで `DftiFreeDescriptor` 呼び出し
- ✅ `put()` で `reset()` → `&handle` の安全な取得
- **報告書の「コピー禁止が不徹底でムーブ時に二重free」は誤り**。Move代入でも先に既存ハンドルを解放してからムーブ元をnullptr化している。

**Retire 機構のEPRベース解放**:
- `ISRRetireOverflowRing` は `LockFreeRingBuffer<RetireOverflowEntry, 16384>` を使用
- バッファフル時は `tryPush` が `false` を返す（ドロップ、メモリリークではない）
- `RetireEnableQueueResult::QueuePressure` を返し、呼び出し元が対応
- `tryReclaim()` が500msクールダウン付きでフォールバック実行

**注意点**: EBR（Epoch-Based Reclamation）自体の正しさは確認済みだが、**overflow 時のエントリ損失** は理論上有り得る。報告書の「メモリリーク + 古いIRが再利用される」は若干ニュアンスが異なり、「エントリ損失によりリソース解放が遅延する」が正確。

---

### [Critical-4] MKLNonUniformConvolver FFTスケーリング漏れとMKLハンドルリーク

**判定: ❌ 大部分がv2.1で解決済み**

| 主張 | 実際のコード (v2.1) |
|------|-------------------|
| `DftiComputeBackward` 後に `1.0f/N` 未適用 | **IPP FFTに換装済み**。`IPP_FFT_DIV_INV_BY_N` フラグで自動正規化。コメント: "IP IFFT 時に 1/N 正規化自動適用 (旧 DFTI_BACKWARD_SCALE と等価)" |
| MKLハンドルリーク・例外時に DftiFreeDescriptor 未呼び出し | `IppFFTPlanCache` が `std::unique_ptr<IppFFTPlan>` でRAII管理。`fftSpecBuf = ippsMalloc_8u(...)` の解放もデストラクタで保証 |
| ゼロ除算 `numPartitions = 0` | `if (cfgs[li].len <= 0) continue;` のガードあり。0パーティションのレイヤーはスキップされる |
| ハンドル二重free | `DftiHandle.h` の move semantics は正しく実装。二重freeは発生しない |

**ただし**: `ConvolverProcessor.MixedPhase.cpp` と `ConvolverProcessor.ResampleAndFallback.cpp` では未だ `DftiComputeBackward` を使用している（非Audio Thread パス）。これらのパスでの正規化問題は完全には確認できていない。

---

### [Critical-5] CustomInputOversampler 遅延補償の不一致

**判定: ❌ 誤った指摘**

| 主張 | 実際のコード |
|------|------------|
| 最小位相FIRを使用 | `static constexpr bool isLinearPhaseFIR = true;` — **線形位相FIR** |
| 遅延が線形位相想定の `(taps-1)/2` | `Latency.cpp` の `static_assert` で `isLinearPhaseFIR && isSymmetricUpDown` を確認。遅延式は `taps-1`（up+down合計）で正しい |

**結論**: オーバーサンプラは線形位相FIRであり、遅延補償式もそれに合わせて設計されている。報告書の前提（最小位相FIR）が誤り。

---

### [Critical-5] DCBlocker float 精度問題

**判定: ⚠️ 部分的に正しい**

192kHz での DCBlocker 極が `0.99979` 付近になることは理論的に正しい。`float` での極の表現限界（≈24bit精度）では `1.0 - 0.99979 ≈ 2.1e-4` であり、float の epsilon ≈ 1.19e-7 に対して余裕があるため即座に発散するわけではないが、長時間の累積演算では問題になり得る。

コード上の `DSPCoreFloat` パスの有無は確認が必要。

---

### [Major-6] TruePeakDetector 補間フィルタ正規化漏れ

**判定: ❌ 誤った指摘**

**係数正規化**:
- `prepareStage()` で以下の正規化を実施:
  1. Kaiser窓 + sinc で係数生成
  2. `sum = Σcoeffs[i]` → `inv = 1.0/sum` → `coeffs[i] *= inv`（合計1.0に正規化）
  3. `rawCoeffs[centerTap] = 0.5` に設定
  4. 非Center係数の合計を計算 → `scale = 0.5 / nonCenterSum` → 非Center係数を再スケール
  5. `rawCoeffs[centerTap] = 0.5` を再設定

合計1.0 + センター0.5の二重正規化が完全に実装されている。`0.5 *` スケーリングは未完成のコードではなく、**halfbandフィルタの数学的性質**（DCゲイン0.5）による正常な設計。

**reset() のサンプルレート変更時動作**:
- `prepare()` の最後で `reset()` を呼び出し
- `reset()` は `peakHold = 0.0` と全 `upHistory` を `clear()` でクリア
- 報告書の「reset() が状態をクリアしない」は誤り

**注意**: `prepare()` はメッセージスレッドからのみ呼ばれることを前提としている。Audio Thread が `processBlock()` 実行中に `prepare()` が呼ばれると、Stage 構造が変更中に参照される可能性がある。ただしこれは設計上の前提（prepareは非RT）の問題であり、バグではない。

---

### [Major-7] PsychoacousticDither / NoiseShaper

**フィードバック発散**:
- **`PsychoacousticDither`**（エラーフィードバック型FIR NSF）: FIRフィルタは全ての極がz=0にあるため **BIBO安定**。報告書の「|H(z)| > 1 になる組み合わせを許容」はエラーフィードバック型NSFには適用されない（コメントでも明記）。誤検出。
- **`Fixed15TapNoiseShaper`**: エラーエンベロープ検出 (`errorEnvelope > kErrorStateThreshold`) で発散を検出し `needsReset` を発行。保護機構あり。
- **`LatticeNoiseShaper`**（IIR格子構造）: `clampCoeff()` で反射係数を `|k| < 0.85` に制限。`isStable()` 確認関数、`clampStateSIMD()` で状態値を ±1e12 に制限。**保護機構完備**。

**状態初期化漏れ**:
- `PsychoacousticDither` コンストラクタ: 時間 + 静的カウンタでユニークシード生成 → `SplitMix64` でチャンネルごとに独立シード → チャンネル個別の `VSLStream` 初期化。**チャンネル間で同一パターンにならない設計**。報告書の「左右チャンネルで同一パターン」は誤り。

**ビット深度変換の丸め**:
- `InputBitDepthTransform.h` ではビット深度変換（24bit→16bit等）のロジックは実装されていない。実際の量子化は `NoiseShaper` / `Dither` クラスで行われ、`scale` / `invScale` による適切な正規化が実装されている。`+0.5` 丸めの有無は各シェイパーの実装依存。

---

### [Major-8] AllpassDesigner / CMA-ES

**判定: ⚠️ 部分的に正しい（一部は確認不足）**

`CmaEsOptimizer.h` / `CmaEsOptimizerDynamic.cpp` の実際のコードは未確認。ただし:
- **極半径の制限**: `LatticeNoiseShaper` では `0.85` に制限。AllpassDesignerでも同様の制限がある可能性はあるが未確認。
- **位相アンラップ失敗**: 汎用的な問題。`2pi` ジャンプが目的関数を破壊することは理論的に正しい。
- **IRAnalyzerの `log(0)`**: 無音IRが入力されるエッジケースであり、理論的に起こり得る。

---

### [Major-9] IRConverter / IRDSP

**判定: ⚠️ 部分的に正しい（一部確認不足）**

- **FFTサイズ超過**: `nextPow2(irLen*2)` が MKL/IPP 制限を超える可能性は理論的に存在。ただし現状は `IppFFTPlanCache` が `ippsFFTGetSize_R_64f` でサイズ検証→失敗時 `nullptr` を返す安全設計。
- **リサンプリング品質**: `Blackman` 窓固定の sinc リサンプリングで -60dB エイリアス減衰は理論的に正しい。窓関数の選択が Blackman 固定であることは確認済み。
- **`peak==0` の除算ゼロ**: 無音IRの正常化時、理論的に起こり得る。実際のコードでは確認できず。

---

### [Major-10] EQProcessor

**判定: ⚠️ 部分的に正しい（一部確認不足）**

- **バンド数上限超過**: 報告書の「Releaseではバッファオーバーラン」は正しい可能性がある。`jassert` のみの防御では Release ビルドでチェックが無効になる。
- **高Qでの `tan(pi*f/fs)` 発散**: fs/2 に近い周波数で理論的に正しい。
- **係数更新のロックなし**: `prepareToPlay` と UI スレッド間の競合は一般的な問題。コードの確認が必要。

---

### [Major-11] NoiseShaperLearner

**判定: ⚠️ 部分的に正しい（一部確認不足）**

- `MklFftEvaluator` の `new` 問題と `CacheManager` のキーにサンプルレートが含まれない問題: コード確認が必要。
- 収束判定 `loss < 1e-12` 固定: double の精度限界に近く、理論的に収束しないケースがあり得る。

---

### [Major-12] DeviceSettings

**判定: ⚠️ 部分的に正しい**

- **COM Release漏れ**: `DeviceSettings.cpp` v0.2 では直接的なCOM呼び出しは見られない。JUCEの `AudioDeviceManager` 経由で間接的にCOMを使用している可能性が高い。JUCE側の管理下にあるためユーザーコードでのリークはJUCE次第。
- **ASIOブラックリスト文字列完全一致**: コード確認が必要だが、バージョン違いのドライバをブロックできない問題は理論的に正しい。

---

### [Major-13] CacheManager

**判定: ⚠️ 情報不足**

キャッシュキーにIR内容ハッシュを含まない問題、バージョニングなし問題は理論的に正しい指摘だが、実際のコードの確認が必要。

---

### [Major-14] ConvolverState 状態保存

**判定: ⚠️ 部分的に正しい**

`adaptiveCoeffBankIndex` と `adaptiveCoeffGeneration` の非同期更新による世代不一致は、EBR設計のプロジェクトでは発生し得る。`SnapshotTests` の `runtimeWorld authority projection contract` がこの問題を検出するためのテストであることも正しい。

---

## 優先度再評価

### P0（真のCritical - 早急な確認推奨）

1. **EQProcessor バンド数超過時のバッファオーバーラン** — `jassert` のみの防御
2. **DeviceSettings COM Release漏れ** — 間接的ながら潜在的なリソースリーク
3. **NoiseShaperLearner 収束判定問題** — `loss < 1e-12` 固定
4. **IRAnalyzer `log(0)` / IRDSP `peak==0` 除算** — エッジケースだがクラッシュの原因

### P1（確認推奨）

5. **AllpassDesigner 極半径制限の有無**（未確認）
6. **CacheManager キー設計**（未確認）
7. **MKLスレッド汚染** — `mkl_set_num_threads(1)` がグローバル設定
8. **CPU AVX-512 チェック** — OS無効化環境での `#UD`

### P2（情報提供）

9. **LockFreeRingBuffer off-by-one** — ❌ **現状は問題なし**
10. **MKLNonUniformConvolver スケーリング** — ❌ **v2.1でIPP FFTに換装済み**
11. **DftiHandle 二重free** — ❌ **RAII正しく実装済み**
12. **TruePeakDetector 正規化漏れ** — ❌ **正規化実装済み**
13. **CustomInputOversampler 遅延不一致** — ❌ **線形位相FIRで正しい**

---

## 報告書の主な問題点

### 1. バージョン不整合
報告書は v2.1 での IPP FFT 換装を反映していない。`MKLNonUniformConvolver.cpp` の先頭コメントに明確に「v2.1 変更点: Audio Thread 内 FFT を MKL DFTI → Intel IPP に換装」と記載されているが、報告書は旧 MKL DFTI コードを前提とした分析。

### 2. ファイルパスの誤り
`ISRRetireOverflowRing.h` は `src/` 直下ではなく `src/audioengine/` にある。報告書が参照した連結ソースではパス情報が欠落していた可能性が高い。

### 3. コードパターンの捏造
報告書は `bool isFull() { return (write +1)%size == read; }` を「典型的な誤りパターン」として提示しているが、実際のコードにはそのようなパターンは存在しない。自動生成モデルが「典型的な」バグパターンを幻覚（ハルシネーション）した可能性が高い。

### 4. メモリオーダーに関する過度に一般化された指摘
`acquire/release` に関する指摘は抽象的で具体的な違反箇所の特定に至っていない。実際のコードは SPSC HB 契約をコメントで明記し、適切な ordering を使用している。

### 5. 重大度評価の不均衡
Critical と評価された項目の多くが実際には問題なし。真に注意すべき項目（EQバッファオーバーラン、DeviceSettings COM、収束判定問題）は Major/Minor に分類されている。

---

## 結論

本バグ監査レポートは **自動生成特有の幻覚（ハルシネーション）を含み、約40%の項目が実際のコードと一致しない**。特に Critical とされた5項目中3項目は **現状のコードでは問題がない**。ただし、Major に分類された一部の項目（EQバッファオーバーラン、DeviceSettings、NoiseShaperLearner収束判定など）は **実際に確認・修正の価値がある**。

**推奨事項**:
1. 本レポートの指摘を盲信せず、実際のコードと対照して判断すること
2. P0再評価項目（上記）を優先的に確認すること
3. 自動生成コードレビューツールの出力は常にファクトチェックすること
4. 自動生成レポート利用時は、生成元のソースコードバージョンを明示的に指定すること
