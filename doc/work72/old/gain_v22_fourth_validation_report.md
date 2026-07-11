# gain_revised.md v2.2 第4次検証レポート

> 検証日: 2026-07-11
> 対象: `gain_revised.md` v2.2（第3次検証修正後）
> 検証範囲: v2.2で新規追記・修正した部分を中心に、コードベース照合・数値再計算・論理的整合性をクロスチェック

---

## 0. 検証サマリ

| # | 問題 | 重大度 | 状態 |
|---|------|--------|------|
| 1 | `FADE_IN_SAMPLES`の誤認（コンボルバー用→DSPCore全体用） | 🔴 重大 | 要修正 |
| 2 | 非クロスフェード時の42msフェードイン（無音→1.0）が「スムージング」と誤記 | 🔴 重大 | 要修正 |
| 3 | §3.6.5非クロスフェード分析が`fadeInSamplesLeft`を見落とし | 🔴 重大 | 要修正 |
| 4 | Q=0.707の文献値が微小誤差（4.6%→4.32%, +0.40→+0.37dB） | 🟡 軽微 | 要修正 |
| 5 | `RuntimeBuilder.cpp`行番号の不正確さ（318-330→328-330） | 🟡 軽微 | 要修正 |

v2.1→v2.2で追加した他の部分（CR1リニアクロスフェード修正、CR2クロスフェード非発生リスク明記、CR3数値検証ラベル修正、MD1文献値比較表、MD2 procState/fadingState同一性、MD3 trim=0ガードスキップ、MD4 totalGainDb vs maxGainDb）は**すべて正確**であった。

---

## 1. 重大問題（3件）

### 1.1 `FADE_IN_SAMPLES`の誤認: 「コンボルバー用」→「DSPCore全体用」

**対象箇所**: §3.6.2 安全性の保証 項目4（line 449）

**文書の記載**:
> コンボルバーの`FADE_IN_SAMPLES=2048`（≈42ms）により、DSP内部でスムージングが行われる

**コードの実態**:
- `FADE_IN_SAMPLES = 2048`は`AudioEngine.h:969`で`DSPCore`の`static constexpr`として定義
- 適用箇所は`DSPCoreDouble.cpp:605-617`:

```cpp
int fadeLeft = ramp.fadeInSamplesLeft;
if (fadeLeft > 0)
{
    const int rampThisBlock = std::min(numSamples, fadeLeft);
    const double gainStep = 1.0 / static_cast<double>(FADE_IN_SAMPLES);
    const double startGain = static_cast<double>(FADE_IN_SAMPLES - fadeLeft) * gainStep;
    for (int ch = 0; ch < numChannels; ++ch)
        applyGainRamp(buffer.getWritePointer(ch), rampThisBlock, startGain, gainStep);
    ramp.fadeInSamplesLeft = fadeLeft - rampThisBlock;
}
```

- これは`dsp->process()`の最終段で、**出力バッファ全体**に`applyGainRamp`を適用するもの
- コンボルバー専用ではなく、EQ出力も含めたDSPCore全体の出力を0→1.0にランプする
- 設定箇所は`RebuildDispatch.cpp:910`で、**全rebuild**で無条件に`FADE_IN_SAMPLES`が設定される

**修正案**: 「コンボルバーの`FADE_IN_SAMPLES=2048`」→「DSPCore全体の出力フェードイン`FADE_IN_SAMPLES=2048`（`DSPCoreDouble.cpp:605-617`、`RebuildDispatch.cpp:910`で全rebuild時に設定）」

### 1.2 非クロスフェード時の42msフェードインが「スムージング」と誤記

**対象箇所**: §3.6.2 安全性の保証 項目4（line 449）

**文書の記載**:
> `processingOrder`のみの変更でクロスフェードがトリガーされない場合、EQの5msバイパスフェード（`BYPASS_FADE_TIME_SEC`）とコンボルバーの`FADE_IN_SAMPLES=2048`（≈42ms）により、DSP内部でスムージングが行われる。新旧DSPは同一の`ProcessingState`（同一ゲイン値）を使用するため、ゲインの不整合は発生しない

**コードの実態**:

非クロスフェード時の実際のフロー:

1. `CrossfadeAuthority::evaluate()`（`CrossfadeAuthority.cpp:8-48`）:
   - `irLoaded`同一、`structuralHash`同一、`oversamplingFactor`同一 → `needsCrossfade = false`

2. `DSPTransition::onPublishCompleted()`（`DSPTransition.h:108-112`）:
   ```cpp
   } else if (oldDSP != nullptr) {
       // Crossfade 不要: 即時 retire
       engine_.crossfadeRuntime_.complete();
       lifetime.retire(oldDSP);
   }
   ```
   - 旧DSPが**即時retire**される。`fading`ポインタはnullptrになる。

3. Audio Thread（`AudioBlock.cpp:359-443`）:
   ```cpp
   const bool canCrossfade = (fading != nullptr || useDryAsOld)
       && crossfadeRuntime_.getGain().isSmoothing() && ...;
   // canCrossfade == false (fading == nullptr)
   // → else分岐（line 434-443）:
   dsp->process(bufferToFill, analyzerFifo, ..., procState);
   ```
   - クロスフェードミックスループは実行されず、新DSPのみが処理

4. 新DSP内部（`DSPCoreDouble.cpp:605-617`）:
   - `fadeInSamplesLeft = 2048`（`RebuildDispatch.cpp:910`で設定済み）
   - `startGain = (2048 - 2048) / 2048 = 0.0` → **ゲイン0から開始**
   - 2048サンプル（≈42ms@48kHz）かけて0→1.0にリニアランプ
   - **出力が42ms間ほぼ無音（フェードイン）になる**

**問題の本質**:
- 文書は「DSP内部でスムージングが行われる」と記載し、安全な遷移を暗示している
- 実際は**42msのフェードイン（無音からの復帰）**が発生し、音切れ/音量低下のリスクがある
- 「ゲインの不整合は発生しない」は技術的に正しい（ゲイン値は同一）が、`fadeInSamplesLeft`が出力ゲインを0から上書きするため、**実質的に出力が無音化**する
- `BYPASS_FADE_TIME_SEC`（5ms）はEQバイパスオン/オフ遷移用であり、processingOrder変更時には無関係

**修正案**: 項目4を以下のように全面書き直し:
```
4. **クロスフェード非発生時の即時切替ケース**: `processingOrder`のみの変更で
   クロスフェードがトリガーされない場合、旧DSPは即時retireされ
   （`DSPTransition.h:108-112`）、新DSPのみが処理を担当する。新DSPには
   `fadeInSamplesLeft = FADE_IN_SAMPLES = 2048`（`RebuildDispatch.cpp:910`）が
   設定されており、出力バッファ全体がゲイン0→1.0へ42ms@48kHzでリニアランプされる
   （`DSPCoreDouble.cpp:605-617`）。これは**42msのフェードイン（無音からの復帰）**
   を引き起こし、音切れ/音量低下のリスクがある。新旧DSPの`ProcessingState`ゲイン値
   は同一だが、`fadeInSamplesLeft`が出力ゲインを上書きするため、実質的に出力が
   一時的に減衰する。この問題は本設計の対象外（既存のDSPライフサイクル管理に起因）
   だが、Phase 8での実機検証が必須である。
```

### 1.3 §3.6.5非クロスフェード分析が`fadeInSamplesLeft`を見落とし

**対象箇所**: §3.6.5（line 507）

**文書の記載**:
> **非クロスフェード時（即時切替）**: 新旧DSPは同一`ProcessingState`の同一ゲイン値を使用する。新しい処理順序で最新のゲイン値が適用されるため、ゲインの不整合は発生しない。

**問題**:
- `ProcessingState`のゲイン値が同一であることは正しい
- しかし、新DSPの`fadeInSamplesLeft = 2048`が`dsp->process()`内部で出力バッファに`applyGainRamp(0→1.0)`を適用する（`DSPCoreDouble.cpp:605-617`）
- したがって「最新のゲイン値が適用される」という記載は、`fadeInSamplesLeft`による出力上書きを見落としている
- 正確には「`ProcessingState`のゲイン値は同一だが、新DSPの`fadeInSamplesLeft`により出力が42ms間0→1.0にフェードインする」

**修正案**:
```
**非クロスフェード時（即時切替）**: 新旧DSPは同一`ProcessingState`の同一ゲイン値を
使用する。ただし、新DSPには`fadeInSamplesLeft = FADE_IN_SAMPLES`が設定されており
（`RebuildDispatch.cpp:910`）、`dsp->process()`内部で出力バッファに
`applyGainRamp(0→1.0)`が適用される（`DSPCoreDouble.cpp:605-617`）。したがって
`ProcessingState`のゲイン値には不整合がないものの、出力は42ms間フェードインする。
これは既存のDSPライフサイクル設計に起因する挙動であり、本ゲインステージング改修の
対象外だが、Phase 8での検証が必要である。
```

---

## 2. 軽微問題（2件）

### 2.1 Q=0.707の文献値が微小誤差

**対象箇所**: §3.3.1 文献値との比較表（lines 247-253）

**文書の記載**:
| Q | step overshoot(%) | overshoot(dB) |
|---|-------------------|---------------|
| 0.707 (Butterworth) | ≈ 4.6% | ≈ +0.40 |

**正確な計算値**:
- ζ = 1/(2×0.707) = 0.7072
- OS = exp(-π×0.7072/√(1-0.7072²)) = exp(-π×0.7072/0.7072) = exp(-π) = 0.0432 = **4.32%**
- dB = 20×log10(1.0432) = **+0.367 dB**

**修正案**: 4.6% → 4.32%、+0.40 → +0.37

> 注: Q=1.414（30.5%/+2.31dB）、Q=4.0（67.3%/+4.47dB）、Q=10.0（85.4%/+5.37dB）はすべて正確であった。

### 2.2 `RuntimeBuilder.cpp`行番号の不正確さ

**対象箇所**: §3.6.5（line 505）

**文書の記載**:
> RCU worldの`automation.*Gain`フィールドはworld構築時にatomicsからキャプチャされる（`RuntimeBuilder.cpp:318-330`）。

**コードの実態**:
- `RuntimeBuilder.cpp:328-330`がatomicsから`worldOwner->automation.*Gain`への直接代入:
  ```cpp
  worldOwner->automation.inputHeadroomGain = inputHeadroomGain;       // line 328
  worldOwner->automation.outputMakeupGain = outputMakeupGain;         // line 329
  worldOwner->automation.convolverInputTrimGain = convolverInputTrimGain; // line 330
  ```
- `RuntimeBuilder.cpp:293-295`は`sealedBuildInput`からの代入（別経路）
- "318-330"は範囲が広すぎる。正確には"328-330"

**修正案**: `RuntimeBuilder.cpp:318-330` → `RuntimeBuilder.cpp:328-330`

---

## 3. 検証結果: 正確だった部分

以下のv2.2新規追記・修正部分は、コードベース照合・数値再計算の結果、**すべて正確**であった:

| 項目 | 検証内容 | 結果 |
|------|----------|------|
| CR1: リニアクロスフェード修正 | `AudioEngine.h:3725`の`gNew`と`AudioBlock.cpp:422-424`の`gNew*gNew + dryScaledL*gOld`がリニアブレンド | ✅ 正確 |
| CR2: CrossfadeAuthority判定基準 | `CrossfadeAuthority.cpp:8-48`が`irLoaded`/`structuralHash`/`oversamplingFactor`のみ判定 | ✅ 正確 |
| CR3: 数値検証ラベル | `input+makeup=0`、`input+trim+makeup=0`、信号経路+15dBFSの数値 | ✅ 正確（Python再計算で確認） |
| MD1: Q Surge Margin計算 | gain=12, Q=4.0 → margin=10.18→clip 6.0; Q=10.0 → 25.46→clip 6.0 | ✅ 正確 |
| MD2: procState/fadingState同一性 | `AudioBlock.cpp:370-372`で`fadingState = procState`コピー、385-386で旧DSP、388-392で新DSP | ✅ 正確 |
| MD3: trim=0ガードスキップ | `DSPCoreDouble.cpp:483`の`if (state.convolverInputTrimGain != 1.0)` | ✅ 正確 |
| MD4: totalGainDb vs maxGainDb | `EQProcessor.h:281`の`float totalGainDb = 0.0f`と新規`maxGainDb`の区別 | ✅ 正確 |
| クロスフェードミックスループ | `AudioEngine.h:3694-3734`の`runLatencyAlignedCrossfadeMixLoop`、`out[i] = new*gNew + old*(1-gNew)` | ✅ 正確 |
| `captureAudioThreadParameterSnapshot` | `AudioEngine.h:3454-3509`、`world->automation.*Gain`から読み取り（line 3469-3471） | ✅ 正確 |
| `buildAudioThreadProcessingState` | `AudioEngine.h:3511-3549`、snapshot→ProcessingState変換（line 3532-3534） | ✅ 正確 |
| `BYPASS_FADE_TIME_SEC = 0.005` | `EQProcessor.h:524`の`static constexpr double BYPASS_FADE_TIME_SEC = 0.005` | ✅ 正確 |
| リスク表の非クロスフェード項目 | 即時切替となる旨の記載 | ✅ 正確（ただし安全性分析は問題1.2で指摘） |

---

## 4. 修正優先度

### 4.1 v2.3で修正すべき項目

| 優先度 | 問題 | 修正内容 |
|--------|------|----------|
| 🔴 高 | 問題1.1+1.2 | §3.6.2項目4を全面書き直し: `FADE_IN_SAMPLES`はDSPCore全体の出力フェードインであり、42ms間無音からの復帰が発生する旨を明記 |
| 🔴 高 | 問題1.3 | §3.6.5非クロスフェード分析を修正: `fadeInSamplesLeft`による出力上書きを明記 |
| 🟡 中 | 問題2.1 | 文献値表Q=0.707行: 4.6%→4.32%, +0.40→+0.37 |
| 🟡 低 | 問題2.2 | `RuntimeBuilder.cpp:318-330`→`328-330` |

### 4.2 設計への影響

問題1.1-1.3は**記述の正確性**の問題であり、設計式や実装方針の変更を必要としない。ただし:

- **承認基準の「純粋オーダー切替（IR同一）時の即時切替でノイズ/クリップが発生しない」は、フェードインによる音切れ（42ms）の観点で検証が必要**
- 42msフェードインが許容できない場合は、`CrossfadeAuthority::evaluate()`に`processingOrder`変化を検出するロジックを追加し、クロスフェードをトリガーする改修が別途必要（本設計の対象外だが、Phase 8テスト結果次第で検討）

---

## 5. 結論

v2.2の核心的な修正（リニアクロスフェード、クロスフェード非発生リスク、procState同一性、文献値比較表）は正確であった。ただし、非クロスフェード時の`FADE_IN_SAMPLES`による42msフェードイン現象を見落としており、安全性分析が不正確であった。この点をv2.3で修正する。

`FADE_IN_SAMPLES`による42msフェードインは既存のDSPライフサイクル設計に起因する挙動であり、本ゲインステージング改修とは独立した問題である。しかし、承認基準の「純粋オーダー切替時のノイズ/クリップなし」テストにおいて、このフェードインが音切れとして検出される可能性があるため、Phase 8での実機検証が必須である。
