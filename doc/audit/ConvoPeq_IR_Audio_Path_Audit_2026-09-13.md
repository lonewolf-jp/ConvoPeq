# ConvoPeq データ処理経路 監査報告書

- **日付**: 2026-09-13
- **対象**: IR読み込み→コンボルバー出力 / オーディオ入力→コンボルバー出力
- **手法**: ソース精査・既存Null Test実測の照合・Intel IPP公式仕様・Gardner NUC文献照合
- **使用ツール**: Serena / AiDex / context-mode / rtk(WSL) / cppcheck / コード探索エージェント×2 / 文献調査

---

## 1. 経路全体像（確定）

### 1.1 IR読み込み経路

```
UI/ファイル
  → IRConverter::loadAudioFile          (float→double, チャネル数)
  → IRDSP::resampleIR (r8brain)         (SR不一致時)
  → IRConverter::convertFile            (timeDomainIR + scale算出)
  → LoaderThread                        (trim / min-phase / Tukey / peakLatency)
  → StereoConvolver::init
  → MKLNonUniformConvolver::SetImpulse  (L0/L1/L2分割 + IR FFT + スケール)
  → finalizeNUCEngineOnMessageThread → applyNewState (RCU publish)
```

並行して `loadIR → applyComputedIR` は **UI/RCUメタデータのみ**。RTエンジン本体は LoaderThread 経路のみが差し替える。

### 1.2 オーディオ入力経路

```
AudioEngineProcessor::processBlock(double)
  → AudioEngine::processBlockDouble
  → DSPCore::process (EQ / gain / softclip / oversample …)
  → ConvolverProcessor::process
       ├ dry: delayBuffer へ書込 → algorithmLatency+irPeakLatency で遅延読出
       ├ wet: StereoConvolver::process(ch)
       │        → NUC::Add  (direct FIR + L0 OLS + L1/L2 FDL)
       │        → NUC::Get  (ring L0 + direct + delayLine L1/L2)
       └ equal-power mix (sin/cos) → 出力
```

---

## 2. 検証済み・正しい項目（再確認不要）

| 項目 | 判定 | 根拠 |
|------|------|------|
| IPP FFT スケール | **正しい** | `IPP_FFT_DIV_INV_BY_N` → IFFTのみ 1/N。OLSに適合 |
| IPP CCS レイアウト | **正しい** | 公式: 標準インターリーブ複素 N/2+1 個 = N+2 double。`deinterleaveComplex` と一致。Mulも標準複素乗算可 |
| Overlap-Save 構造 | **正しい** | `[prev\|cur]` → FFT → MAC → IFFT → 後半 N 書込 |
| B13 L1/L2 遅延 (Policy R) | **正しい** | `readStart = t0 − o_L`。修復後 oPE=0 / M1 −309〜−311 dB（work57 Step4） |
| イコールパワー mix | **正しい** | `sin(πt/2)` 9次テイラー、誤差 ~2e-6 |
| Direct head と L0 の二重計上 | **なし** | direct 有効時、FFT側IR先頭をゼロクリア |
| FDL mirror / linStart | **正しい** | 範囲内、バッファサイズ整合 |
| 最小位相変換（ケプストラム） | **概ね正しい** | 折り返し + exp は教科書通り |
| Stereo 独立 | **正しい** | ch毎に NUC を分離 |
| RCU engine swap | **正しい** | acq_rel + deferred retire |

---

## 3. 発見バグ（重大度順）

### C-01: 非2冪ホストブロックで L0 と L1/L2 のストリーム時計が乖離する

**重大度: Critical（音質: 周期的ドロップアウト + L1/L2 相対ずれ）**

**場所**: `src/MKLNonUniformConvolver.cpp:1691–1762` (`Get`), `:1499–1524` (`ringRead`)

```cpp
const std::uint64_t t0 = m_outputSamplesProcessed;
const int got = ringRead(output, numSamples);          // got < numSamples になり得る
delayLineReadAdd(l, output, numSamples, t0, layerGain); // ← numSamples 分読む
m_outputSamplesProcessed += static_cast<std::uint64_t>(got); // ← got 分だけ進む
```

**機序**:
- L0 `partSize = nextPowerOfTwo(bs)`。`bs=480` → `partSize=512`。
- ring は512サンプル溜まったら吐き出す。Get は毎回480要求 → `got` が 300/212/… と振動。
- 出力の後半はゼロ詰め（`ringRead`）。
- 一方 L1/L2 は常に `numSamples` 分を `t0` から読む。時計は `got` しか進まない → **読み窓が重複またはスキップし続ける**。

**影響**: WASAPI shared の 480 サンプルなど非2冪バッファで可聴。Null Test は `bs=64`（2冪）のみのため未検出。

**修正案**:
1. `m_outputSamplesProcessed` を `got` ではなく `numSamples` で進め、L1/L2 も `got` 分だけ読み残りをゼロ、または
2. 内部で `partSize` 固定ストリームにバッファリングしてから Get（推奨）。

---

### H-01: dry を `partSize` 遅延するが wet OLS は配列上 0 遅延 → mix 中間でコム

**重大度: High（mix 0.1–0.9 で可聴）**

**場所**:
- `MKLNonUniformConvolver.cpp:1055` — `m_latency = m_layers[0].partSize`
- `ConvolverProcessor.Runtime.cpp:266–274` — dry 遅延 = `algorithmLatency + irPeakLatency`

**機序**:  
OLS 後半出力は、そのブロック入力に対する線形畳み込み `y[n]=(h*x)[n]` を **同じ出力インデックス**に置く（M1 Null Test −309 dB が実証）。配列上 wet に追加遅延はない。  
dry を `partSize` 遅延すると

\[
out = g_W (h*x)[n] + g_D\, x[n-N]
\]

となり、mix 50% でコムフィルタ／エコーになる。

**注意**: full wet (mix=1) では dry 不使用のため無音影響。bypass は別経路。

**修正案**: `m_latency = 0`（OLS 設計なら）にし、host PDC へは「ブロック周期待ち」を別概念で申告するか、dry 遅延を `irPeak` のみにする。dry/wet null test を mix=0.5 で再実施。

---

### H-02: `irPeakLatency` が「ピーク」ではなく 99.9% エネルギー重心

**重大度: High（PDC / bypass 遅延の過大申告）**

**場所**: `ConvolverProcessor.LoaderThread.cpp:149–208`

```cpp
constexpr double ENERGY_THRESHOLD = 0.999;
// cutoff までのエネルギー重心を maxCentroid として採用
irPeakLatency = floor(maxCentroid + 0.5);
```

- 関数名・フィールド名は Peak だが実体は **エネルギー重心**。長いリバーブ IR では尾の大部分に寄る。
- これが `StereoConvolver::irLatency` に入り、host 申告 latency と dry/bypass 遅延に加算される。
- 対照的に RCU 経路（`LoadPipeline.cpp:500–514`）は **真の max-abs ピーク**。UI 表示と実 engine が不一致。

**修正案**: 真のピーク（または先頭閾値超過）に統一。重心を使うならフィールド名を `irEnergyCentroid` に改め、PDC には使わない。

---

### H-03: 二系統の IR 経路 — `loadIR` は RT エンジンを差し替えない

**重大度: High（設計／初回ロード時の無音 or 旧IR継続）**

**場所**: `LoadPipeline.cpp` `loadIR` vs `loadImpulseResponse` / `AudioEngine.Parameters.cpp:197–199`

- `loadIR → applyComputedIR` は partitionData を「デッドコード」として捨て、RCU メタデータと `IRState` のみ更新。
- 実音声に効くのは `rebuildAllIRsSynchronous → LoaderThread → finalizeNUCEngine` のみ。
- ProgressiveUpgrade も `applyComputedIR` のみ。NUC 差替なし。
- `IRState` の scale 方針が経路で不一致（applyComputedIR は scaled、Loader 経路は unscaled）。

**修正案**: `loadIR` 内で LoaderThread を起動するか、「rebuild 完了まで旧 engine」を UI に明示。scale 所有権を一本化。

---

### M-01: L1/L2 の IFFT 前にデノーマルキルがない（L0 にはある）

**場所**: L0 `:1438–1448` は `killDenormalV`。L1/L2 `:1647–1656` は interleave 直後に IFFT。

Release は FTZ/DAZ 依存だが、L0 と非対称。FTZ 漏れ時、長いテールでデノーマル起因の CPU スパイク。

**修正**: L0 と同じ処理を L1/L2 へ移植。

---

### M-02: wet 出力の NaN スクラビングが FDL 内部状態を放置

**場所**: `Runtime.cpp:722` `sanitizeFiniteChunk`

出力の非有限を 0 にするが、`fdlReal/fdlImag/accum*` の NaN は残り **永続ミュート**。故障が「音が消えた」に見える。

**修正**: 検出時に当該レイヤーの FDL/accum をクリア（または Reset 誘導）。

---

### M-03: Experimental Direct Head の整合不良

**場所**: `SetImpulse` directTap 上限 32、`Runtime.cpp:266` で latency 申告 0

- direct は最大32タップ。L0 partSize は通常 512〜1024。
- FFT 側は先頭32のみゼロ。残り L0 IR は partSize 分のブロック待ちがあるのに latency=0 申告。
- HC/LC スペクトラムフィルタは FFT IR のみ。direct 32タップはフル帯域のまま → 境界でスペクトラム不連続。

**修正**: direct 有効時も algorithmLatency を partSize 相当で申告するか、direct を partSize タップに拡張して FFT と一致。HC/LC を direct にも適用。

---

### M-04: mix スムージング中バッファ不足で早期 return

**場所**: `Runtime.cpp:593–594`

```cpp
if (activeSmoothingCapacity < numSamples)
    return;  // block 未書き込みのまま
```

prepare 済みだが smoothing バッファ未確保時、入力が素通し（または前回値）になる。

---

### M-05: SR 変更直後に未リサンプル IR で engine が publish され得る

**場所**: `Lifecycle.cpp:244–286`（コメントで [M-1] 自認）

rebuild 失敗・遅延時、旧 SR の IR で新しい SR の plan が組まれる。

---

### M-06: `prepareToPlay` / 大ブロック無音化 (S-1) — 既知・分離済み

`numSamples > maxSamplesPerBlock` で clear+return。アーキテクチャ変更扱いで現状維持。**確定済み保留**。

---

### M-07: `cblas_dscal` 長さを `MKL_INT` へ暗黙キャスト

`LoadPipeline.cpp:396` / `CacheManager.cpp`。巨大 IR × ch で切り詰めの可能性（現行上限下では低リスク）。

---

## 4. 保留事項の棚卸し確定

| ID | 内容 | 今回の確定 |
|----|------|------------|
| **R-1** | OutputFilter fc 分岐 | **変更不要**（音響設計値・維持） |
| **R-2** | BuildAnalysis 空で続行 | **変更不要**（ISR 責務分離に合致） |
| **R-3** | AutoGain `interpolatedDb` 意味 | **確定: 変更不要** |
| **S-1** | 大ブロック無音化 | **分離維持**（別アーキテクチャ課題） |
| B13 | L1/L2 遅延 | **修復済み・実測合格**（Policy R） |
| CCS | deinterleave 誤り疑い | **非バグ**（IPP CCS=標準複素 N+2、公式確認） |

### R-3 の確定根拠

- `PeakEstimate::interpolatedDb` = サンプリングした周波数応答の**大域ピーク**に対する放物線補間値 [dB]（`EQAnalysisTypes.h:89`）。
- `EQResponseSampler` はバンドのみ。Master Gain は含まない。
- `gainDb = interpolatedDb + totalGainDb` は「EQピーク + Master」であり**二重計上ではない**。
- 意図的設計として **変更不要** と確定。

---

## 5. 文献照合サマリ（オーディオ処理妥当性）

| 仕様 | 文献 | 本実装 |
|------|------|--------|
| NUC パーティション分割 | Gardner 1994 | L0/L1/L2 非等間隔、適合 |
| OLS: FFT長=2×パーティション | Wikipedia OLS / Gardner | `fftSize = partSize*2` |
| IPP FWD は 1/N なし / INV は 1/N | Intel IPP Manual | `IPP_FFT_DIV_INV_BY_N` |
| IPP CCS = 標準複素 N/2+1（N+2 double） | 同・Packed Formats 節 + 例 FFT([1,2,3,9]) | `deinterleaveComplex` と一致 |
| L1 以降の遅延 = 先行 IR 長 | Gardner / repair design Rev4 | `outputDelaySamples = prevLayerTotalSamples`、実測 oPE=0 |
| イコールパワー dry/wet | 再送用リバーブ慣行 | `equalPowerSin` |
| 最小位相（実ケプストラム） | MATLAB rceps 等 | 折り返し規則は適合 |
| 大気吸収 | ISO 9613-1 的 HF 減衰 | スペクトルへ事前適用（RT 非負荷） |

---

## 6. 優先修正順序

1. **C-01** 非2冪ブロックの時計同期（Get の `got`/`numSamples` 不一致）
2. **H-01** dry 遅延と OLS wet の時間整合（mix null test 追加）
3. **H-02** `irPeakLatency` を真のピークへ統一
4. **H-03** `loadIR` 経路で NUC 差替を明示／一本化
5. **M-01 / M-02** デノーマルと NaN 状態の RT 衛生
6. **M-03** Direct Head の latency・フィルタ整合

---

## 7. 推奨追加テスト

| テスト | 内容 |
|--------|------|
| NonPoT | `bs=480` で短い IR、wet 出力にゼロ詰め周期がないこと |
| DryWet | impulse、mix=0.5、dry/wet 交差相関で遅延 0 |
| PeakLatency | 2s 減衰 IR で LoaderThread の irLatency と max-abs インデックス一致 |
| Pipeline | UI のみ loadIR 後、`loadActiveEngine()` が null/旧のままにならないこと |
| CCS unit | 既知スペクトルで `ippsFFTFwd_RToCCS_64f` → deinterleave の bin 0 / Nyquist 検証（回帰防止） |

---

## 8. 結論

- **コア畳み込み（OLS・IPP スケール・CCS・B13 遅延）は文献・実測の両面で妥当。**
- **最優先の実バグは C-01（非2冪ブロックの Get 時計）** と **H-01（dry 遅延の設計）**。
- **PDC 用 `irPeakLatency` の定義（H-02）** と **IR ロード二系統（H-03）** は製品品質に直結する設計欠陥。
- 保留4件のうち R-1/R-2/R-3 は**変更不要で確定**。S-1 は分離維持。

以上。
