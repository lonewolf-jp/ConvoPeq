# gain_revised.md v2.3 + Phase 8テスト計画 第5次検証レポート

> 検証日: 2026-07-11
> 対象: `gain_revised.md` v2.3（第4次検証修正後）+ `gain_phase8_test_plan.md`
> 検証範囲: v2.3修正部分のコード再照合、Phase 8テスト計画の正確性確認、全文一貫性チェック

---

## 0. 検証サマリ

| # | 問題 | 重大度 | 対象 | 状態 |
|---|------|--------|------|------|
| 1 | `bypassFadeGainDouble`の時間定数が5ms→2048サンプル(42ms)と誤記 | 🟡 中 | gain_revised.md §3.6.3 line 461 | 要修正 |
| 2 | UT-06の「Q Surge Marginが加算されない」が自己矛盾 | 🟡 中 | test_plan UT-06 | 要修正 |
| 3 | IT-04 Conv onlyモードでクランプ考慮漏れ（-4.5→-6.0にクランプ） | 🟡 中 | test_plan IT-04 | 要修正 |
| 4 | §3.6.5結論文がv2.3修正後も「不整合は発生せず」のままで誤解を招く | 🟢 軽微 | gain_revised.md §3.6.5 line 511 | 要修正 |

v2.3の核心修正（FADE_IN_SAMPLES再分類、42msフェードインリスク明記、文献値修正、行番号修正）は**すべて正確**であった。v2.3修正自体は新規エラーを導入していない。発見された4件はいずれもv2.3以前から存在する既存問題、またはテスト計画の記載ミスである。

---

## 1. 中程度問題（3件）

### 1.1 `bypassFadeGainDouble`の時間定数誤記

**対象**: `gain_revised.md` §3.6.3 line 461

**文書の記載**:
> コンボルバーのバイパスはDSPCoreレベルのフルバイパスブレンド（`bypassFadeGainDouble`、2048サンプル ≈ 42ms@48kHz）で処理される

**コードの実態**:
- `AudioEngine.h:721`: `bypassFadeGainDouble.reset(sampleRate, 0.005);` → **5ms**
- `bypassFadeGainDouble`は`LinearRamp`で、`reset(sampleRate, 0.005)`により5msのランプ時間が設定される
- 「2048サンプル ≈ 42ms@48kHz」は`FADE_IN_SAMPLES`の値であり、`bypassFadeGainDouble`とは無関係
- `bypassFadeGainDouble`（5ms、フルバイパスon/off用）と`FADE_IN_SAMPLES`（2048サンプル=42ms、新DSPフェードイン用）は別物

**修正案**:
```
- コンボルバーのバイパスはDSPCoreレベルのフルバイパスブレンド（`bypassFadeGainDouble`、5ms）で処理される
```

### 1.2 UT-06の自己矛盾

**対象**: `gain_phase8_test_plan.md` UT-06 (lines 90-100)

**文書の記載**:
```
検証:
  - computeMaxGainDb()の戻り値が ≈ +9.0dB（誤差±0.5dB）
  - Q Surge Marginが加算されない（Q=1.0 > 0.707だが、gain*0.15*(1.0/0.707) ≈ 1.91dBが加算される）
  - 実際の戻り値が ≈ 9.0 + 1.91 ≈ 10.9dBになること
期待結果: eqMax ≈ 10.9dB
```

**問題**:
- 1行目: 「戻り値が ≈ +9.0dB」と記載 → 実際は10.9dB
- 2行目: 「Q Surge Marginが加算されない」と記載 → 直後に「1.91dBが加算される」と矛盾
- 3行目: 「9.0 + 1.91 ≈ 10.9dB」と記載 → これが正しい
- 期待結果: 10.9dB → これが正しい

**正しい計算**:
- Q=1.0 > 0.707のため、Q Surge Margin **は加算される**
- margin = 9 × 0.15 × (1.0/0.707) = 9 × 0.15 × 1.414 = 1.91 dB
- eqMax = 9.0 + 1.91 = 10.91 ≈ 10.9 dB

**修正案**:
```
検証:
  - Q=1.0 > 0.707のためQ Surge Marginが加算される
  - margin = 9 × 0.15 × (1.0/0.707) ≈ 1.91 dB
  - computeMaxGainDb()の戻り値が ≈ 9.0 + 1.91 ≈ 10.9dBになること
期待結果: eqMax ≈ 10.9dB
```

### 1.3 IT-04 Conv onlyモードのクランプ考慮漏れ

**対象**: `gain_phase8_test_plan.md` IT-04 (line 173)

**文書の記載**:
```
2. Conv only: input = -max(0, 6-1.5) = -4.5, makeup = +4.5
```

**コードの実態** (`AudioEngine.Parameters.cpp:229-234`):
```cpp
const bool convBypassed = false;  // Conv active
const bool eqBypassed   = true;   // EQ bypassed
const bool convIsFirst = !convBypassed && (order == ConvolverThenEQ || eqBypassed);
// = true && (anything || true) = true
const float maxDb = -6.0f;  // Conv-first → upper limit -6 dB
float clampedDb = juce::jlimit(-12.0f, -6.0f, -4.5f);
// = -6.0f  (-4.5 > -6 ので上限クランプ)
```

- Conv onlyモードでは`eqBypassed = true`により、`convIsFirst = true`が常に成立
- input上限が-6 dBとなり、計算値-4.5 dBは-6.0 dBにクランプされる
- `recomputeAutoGainStaging()`はクランプ後の値を再読み込みしてmakeupを調整する設計（§3.5.1注記）
- したがって正しい値は: `input = -6.0 (clamped), makeup = +6.0`

**修正案**:
```
2. Conv only: input = -max(0, 6-1.5) = -4.5 → クランプ -6.0, makeup = +6.0
   ※ Conv onlyモードはeqBypassed=trueによりconvIsFirst=true、上限-6dBでクランプ
```

---

## 2. 軽微問題（1件）

### 2.1 §3.6.5結論文がv2.3修正後も誤解を招く

**対象**: `gain_revised.md` §3.6.5 line 511

**文書の記載**:
> したがって、いずれのケースでもゲイン値の不整合は発生せず、各DSPは最新のゲイン値で正しく処理される。

**問題**:
- v2.3でline 509に42msフェードイン（`fadeInSamplesLeft`による出力上書き）の記述を追加した
- しかしline 511の結論文はv2.2のままで「いずれのケースでも...正しく処理される」と完結している
- 「ゲイン値の不整合」という意味では技術的に正しいが、直前の段落で42msの出力減衰を説明した後に「正しく処理される」と結ぶのは読者に誤解を与える

**修正案**:
```
したがって、いずれのケースでもゲイン値自体の不整合は発生しない。ただし非クロスフェード時は
`fadeInSamplesLeft`による42msの出力フェードインが発生する（上記段落参照）。クロスフェード時は
新旧DSPが同一ゲイン値で処理し、リニアブレンドで滑らかに遷移する。
```

---

## 3. 検証結果: 正確だった部分

### 3.1 v2.3修正部分

| 項目 | 検証内容 | 結果 |
|------|----------|------|
| §3.6.2項目4: FADE_IN_SAMPLES再分類 | `AudioEngine.h:969`で`DSPCore`の`static constexpr`、`DSPCoreDouble.cpp:605-617`で出力バッファ全体に`applyGainRamp` | ✅ 正確 |
| §3.6.2項目4: 旧DSP即時retire | `DSPTransition.h:108-112`の`else if (oldDSP != nullptr)`分岐で`lifetime.retire(oldDSP)` | ✅ 正確 |
| §3.6.2項目4: `fadeInSamplesLeft`設定 | `RebuildDispatch.cpp:910`で`newDSP->ramps().fadeInSamplesLeft = DSPCore::FADE_IN_SAMPLES` | ✅ 正確 |
| §3.6.2項目4: ゲイン0開始 | `DSPCoreDouble.cpp:609-610`で`startGain = (FADE_IN_SAMPLES - fadeLeft) / FADE_IN_SAMPLES`、初回=0.0 | ✅ 正確 |
| §3.6.5: `fadeInSamplesLeft`出力上書き | `dsp->process()`内で`applyGainRamp`が出力に適用され、`ProcessingState`ゲインを上書き | ✅ 正確 |
| 文献値表Q=0.707: 4.32%/+0.37dB | Python再計算: `exp(-π) = 0.0432 = 4.32%`, `20*log10(1.0432) = 0.367dB` | ✅ 正確 |
| `RuntimeBuilder.cpp:328-330` | atomicsから`worldOwner->automation.*Gain`への直接代入がline 328-330 | ✅ 正確 |

### 3.2 Phase 8テスト計画

| 項目 | 検証内容 | 結果 |
|------|----------|------|
| UT-01〜UT-05 | 複素応答計算、AVX2一致、z=exp(+jω)、M/Sデコード、Tukey窓 | ✅ 正確 |
| UT-07: Q Surge Marginクリップ | Q=10, gain=12 → margin=25.46→clip 6.0, eqMax=18.0 | ✅ 正確 |
| UT-08: 全バンドバイパス | eqMax=0.0dB | ✅ 正確 |
| IT-01: Q Surge Margin Bound | eqMax=18, input=-15→clamp -12 | ✅ 正確 |
| IT-03: publicationSemanticHash | `RuntimeBuilder.cpp:410-412`の`bit_cast`ハッシュ | ✅ 正確 |
| IT-04: モード1,3,4 (PEQ only, Conv→PEQ, PEQ→Conv) | 計算値とクランプ範囲の整合 | ✅ 正確 |
| GC-01〜GC-06: ソースコード契約 | 行番号・構造体・ガード条件の参照 | ✅ 正確 |
| MT-01〜MT-10: 手動テスト | テスト手順・合格基準の妥当性 | ✅ 正確 |

---

## 4. 修正優先度

| 優先度 | 問題 | 修正内容 | 対象ファイル |
|--------|------|----------|-------------|
| 🟡 中 | 問題1.1 | `bypassFadeGainDouble`「2048サンプル ≈ 42ms」→「5ms」 | gain_revised.md |
| 🟡 中 | 問題1.2 | UT-06「加算されない」→「加算される」、1行目の期待値修正 | gain_phase8_test_plan.md |
| 🟡 中 | 問題1.3 | IT-04 Conv onlyにクランプ考慮を追加 | gain_phase8_test_plan.md |
| 🟢 軽微 | 問題2.1 | §3.6.5結論文に非クロスフェード時のフェードイン言及を追加 | gain_revised.md |

---

## 5. 結論

v2.3の修正自体は正確であり、新規エラーを導入していない。発見された4件は：
- 問題1.1: v2.2以前から存在する`bypassFadeGainDouble`時間定数の誤記（第1〜4次検証で見逃し）
- 問題1.2-1.3: Phase 8テスト計画の記載ミス（UT-06の自己矛盾、IT-04のクランプ考慮漏れ）
- 問題2.1: v2.3修正に伴う結論文の更新漏れ

いずれも設計式や実装方針への影響はなく、記述修正のみで対応可能。修正後にgain_revised.mdはv2.4、テスト計画はv1.1となる。
