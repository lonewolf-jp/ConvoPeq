# P1-2 二重補償 read-only 再監査（2026-09-22）

- **工程**: P1-1 commit 固定（`d6995b2b` = 実装 / `1e9e63e3` = 報告書）→ **P1-2 read-only 監査（本 doc）** → 採用判断（未着手・HOLD）
- **前提 BUILD-ID**: HEAD `1e9e63e3` / production 差分 = P1-1 の 1 行（`centerValue *= 2.0` を `#if` で gate）＋ CMake flag（**既定 OFF**・cache 実値 OFF）/ `shadow_candidate` 0 / snapshot `ConvoPeq.md` **header 2026-09-22 10:01:24 / 5,485,818 B / `--check` FRESH（NEWER_SRC_COUNT=0）**
- **snapshot 更新の記録**: P1-1 測定後の **test-only TU 修正**（`P1PolyphaseGainCharacterization.cpp` の決定論的 capture 修正・mtime 08:25:54）により 08:12:33 の snapshot が STALE 化したため再生成した（**production src は不変**・P1-1 の測定結果には影響しない）。P1-1 報告書が引用した 08:12:33 は「測定時点の identity」であり、本 doc が現在の identity を更新する。
- **本 doc は production change = 0**（read-only・コード読解と既存実測の照合のみ）。

**監査対象チェーン**:

```text
CustomInputOversampler (up/down)
  → DSPCoreFloat / DSPCoreDouble（同一 oversampler 実装を共有）
  → OutputStage（outputFilter HC/LC）
  → SoftClip（OS ドメイン or 局所 2x OS）
  → peakLimiter → hard clamp（±kOutputHeadroom）
  → final output
```

---

## 1. `0.75^N` を別箇所で補償していないか

- **SOURCE**: production（tests 除外）に対する `0.75` / `0.5625` / `0.421875` / `2.4988` / `6.0206` / `4.0/3.0` / `1.3333` の census。ヒットは `EQControlPanel.cpp:493`（UI alpha 0.75）・`PsychoacousticDither.h:219`（NS 係数表）・`SpectrumAnalyzerComponent.h:116`（smoothing）・`AudioEngine.Retire.cpp:26`（fade scale clamp 0.75〜1.50）・`RuntimeHealthMonitor.h:86`（reader slot 閾値 0.75）のみ。
- **OBSERVATION**: OS 往復利得（0.75^N）を打ち消す/**
  補償する計算は production に存在しない**。`oversamplingFactor` の参照も「OS 経路の有無判定（>1 / ==1）」「block size 検証」「latency 換算（`AudioEngine.h:4268` の `toBaseRateSamples`）」「preset 検証」に限られ、**利得計算には現れない**。
- **INFERENCE**: base の droop は **factor 依存（−2.4988 dB/stage）**であるため、単一の校正定数では補償不可能であり、そもそも「補償済み」という前提が構造的に成立しない。
- **CONTRACT IMPACT**: **既存 compensation なし** → 停止条件①非該当。

## 2. AutoGainPlanner が oversampler gain を前提にしていないか

- **SOURCE**: `AutoGainPlanner.h:51-55`（`PlannerInput`）・`AutoGainPlanner.cpp:52-95`・`RuntimeBuilder.cpp:303-337`（呼出）・`AudioEngine.h:436-441`（`AutoGainClampedData`）・`AudioEngine.Parameters.cpp:298-347`（`applyDefaultsForCurrentMode`）。
- **OBSERVATION**: planner の入力は `eqMaxGainDb` / `eqMaxQ` / `irFreqPeakGainDb` の 3 値のみで、`makeup = jlimit(0,12, −input − trim)`（:93-94）。**OS 倍率・往復利得の項は無い**。既定 staging（`applyDefaultsForCurrentMode`）は mode 依存（`0/0/0`・`0/+10/−6`・`−6/+12/0`）だが**factor 非依存**。`AutoGainClampedData` も eqBoost/convBoost/qMargin/rawMakeup のみ。
- **INFERENCE**: flag ON の level 上昇は自動 gain に吸収されない（自動 gain は EQ/IR の feed-forward のみ）。
- **CONTRACT IMPACT**: **AutoGain 依存なし** → 停止条件②非該当。

## 3. Output headroom / makeup が +2.4988 dB/stage と衝突しないか

- **SOURCE**: `DSPCoreDouble.cpp:477-481`（makeup 適用）・`:334`（headroom）・`:593/:655-673`（`kOutputHeadroom` と dither/NS）・`:715-749`（limiter/clamp）・`Parameters.cpp:225-267`（範囲 −12〜0 / 0〜12 dB）。
- **OBSERVATION**: `kOutputHeadroom = 0.8912509381337456` は固定 constexpr。makeup/headroom/trim は**素の scalar 乗算**で factor 非依存。P1-1 の chain 実測では ON は factor に対し level-flat（1 kHz: −1.0076/−1.0073/−1.0072 dB @ N=1/2/3）で、OFF/ON 差は +2.4988/+4.9975/+7.4963 dB（理論一致）。ON の 0 dBFS 入力では limiter 到達が 0 → **119/4096** に増加し、hard clamp は **0 件**のまま。
- **INFERENCE**: 「headroom/makeup が案E と衝突する」構造は存在しない（両者は factor 非依存で、案E は factor 依存項を消す方向に働く）。増えた limiter 到達は**level 変更の帰結**であり、安全鎖は破綻していない。
- **CONTRACT IMPACT**: **再調整は「構造上必須」ではない**（停止条件③非該当）。ただしユーザー向け既定値（makeup 12 dB 等）を案E 前提で見直すか否かは**採用判断（HOLD）**であり、本監査では変更しない。

## 4. SoftClip の drive / threshold に暗黙の compensating gain がないか

- **SOURCE**: `DSPCoreDouble.cpp:483-515`（`clipThreshold = 0.95 − 0.45·sat`・`clipKnee = 0.05 + 0.35·sat`・`clipAsymmetry = 0.10·sat`）、`DSPCoreFloat.cpp` の同一マッピング、`FastTanhApprox.h`（`SoftClipPadePolicy` clipThreshold 4.5）。
- **OBSERVATION**: 閾値・ニー・非対称はいずれも **`saturationAmount` のみの関数**で、OS 倍率・往復利得・`kOutputHeadroom` を参照しない。P1-1 実測では clip 作用の最大差（clipEngMax）が OFF 0.0018〜0.0687 → ON 0.0182〜0.1473 に増える（＝駆動が上がった結果であり、補償ゲインの混入ではない）。
- **INFERENCE**: SoftClip 側に「案E を打ち消す」「案E を前提とする」いずれの隠れ項も無い。
- **CONTRACT IMPACT**: **SoftClip threshold の再調整は不要**（停止条件④非該当）。ただし駆動増による音質変化の最終判断は聴感（HOLD）。

## 5. Limiter threshold は変更不要か

- **SOURCE**: `DSPCoreDouble.cpp:715-722`（`kPLThreshold = 0.8413951287507587`・`kPLKnee = 0.108748`）・`:724-749`（`±kOutputHeadroom` clamp）・`SimplePeakLimiter`（`prepare(sampleRate, 100.0)` = release のみ）。
- **OBSERVATION**: θ と knee は **固定 constexpr** で、OS 倍率・レベル・staging から独立。limiter は `kOutputHeadroom` から導出される（`kOutputHeadroom − 0.5 dB` / 1.0 dB knee）。P1-1 実測で ON の 0 dBFS は出力 peak が θ に一致（0.841395）し、hard clamp 0 件（＝limiter が先に働く設計どおり）。
- **INFERENCE**: θ の再調整を要求する構造的根拠は無い。
- **CONTRACT IMPACT**: **Limiter threshold 変更不要**（停止条件⑤非該当）。

## 6. Convolver trim と oversampler correction が独立しているか

- **SOURCE**: `DSPCoreDouble.cpp:446-453`（trim 適用）・`DSPCoreFloat.cpp:349-354`・`Parameters.cpp:270-297`（設定）・`AutoGainPlanner.cpp:83`（`trimDb = −max(0, convBoost − kMarginInterStage)`）。
- **OBSERVATION**: trim は **OS ドメイン内の素の scalar 乗算**で、値はユーザ設定または planner の dB 値。OS 倍率・往復利得を参照しない。P1-1 実測でも staging 0 dB 固定で OFF/ON 差が理論値に一致（＝trim と flag 効果が乗算的に独立）。
- **INFERENCE**: Convolver trim と oversampler correction は独立。
- **CONTRACT IMPACT**: **Convolver trim 変更不要**（停止条件⑥非該当）。

## 7. float / double の両経路で同一意味論か

- **SOURCE**: `AudioEngine.h:970-971`（`CustomInputOversampler oversampling; CustomInputOversampler softClipOS;` を **単一インスタンスとして一度だけ宣言**）・`DSPCoreDouble.cpp:361/505/513/541`・`DSPCoreFloat.cpp:263/405/413/419`（両 core が同一メンバを呼ぶ）・P0-G（Phase 0 実測: float 量子化入力での差 RT maxAbs 2.58e-08・SC 2.96e-08 ≤ 5e-7）。
- **OBSERVATION**: OS 補正は **単一実装（`CustomInputOversampler`）** にのみ存在し、float/double 両 core がそれを共有する。SoftClip のマッピングも両 core で同一の `sat` 由来式。float 経路も内部演算は double（Phase 0 実測）。
- **INFERENCE**: 意味論の分岐点は存在しない（コードが同一であるため）。
- **CONTRACT IMPACT**: **float/double divergence なし**（停止条件⑦非該当）。

## 8. bypass / wet / dry の gain relationship に意図しない差がないか

- **SOURCE**: `DSPCoreDouble.cpp:339-355`（dry 採取）・`:517-537` / `:544-559`（fade）・`AudioEngine.h:746-749`（`bypassFadeGainDouble/Float`）・`AutoGainPlanner.cpp:40-49`（両 bypass 時は plan 0）。
- **OBSERVATION**: dry は **pre-OS**（input headroom 後・up 前）で採取され、fade は `0.0 ↔ 1.0` の linear ramp のみ。**bypass 用の補償ゲイン定数は存在しない**。flag OFF では wet−dry 差に `0.75^N` が含まれるが、それを前提・相殺するコードは無い。flag ON では wet−dry 差が staging 分（headroom/makeup）のみに縮小する。
- **INFERENCE**: 変化は「意図した level 差の消失」であり、bypass 契約（fade は素の dry/wet ミックス）と矛盾しない。なお **headroom/makeup 由来の bypass 段差は案E と独立に残る**（既存挙動・本監査の対象外）。
- **CONTRACT IMPACT**: **bypass/wet gain contract の矛盾なし**（停止条件⑧非該当）。

---

## 判定

```text
P1-2 = PASS
```

**停止条件 8 項目 → すべて非該当**:

| 停止条件 | 実測・確認 |
|----------|-----------|
| 既存 compensation 発見 | なし（census・factor 依存 gain は OS 経路のみ） |
| AutoGain 依存発見 | なし（PlannerInput に OS 項なし・makeup は EQ/IR 由来） |
| Output headroom の再調整が必要 | 構造上不要（固定 constexpr・factor 非依存）。既定値の見直しは採用判断（HOLD） |
| SoftClip threshold の再調整が必要 | 不要（sat のみの関数・駆動増は level 変更の帰結） |
| Limiter threshold の再調整が必要 | 不要（固定 constexpr・OS/level 非依存） |
| Convolver trim が必要 | 不要（独立の scalar・planner 由来） |
| float/double divergence | なし（単一実装を両 core が共有・P0-G 実測 ≤2.96e-8） |
| bypass/wet gain contract contradiction | なし（dry pre-OS・fade 0↔1・補償定数なし） |

**これは案E採用の確定ではない。** 以下は引き続き **HOLD**:

```text
案Eの最終採用 / default ON / calibration 変更 / kOutputHeadroom 変更 / makeup 変更
SoftClip threshold 変更 / Limiter threshold 変更 / Convolver trim / F-3 / F-4 / O-20
flag 削除 / release-note 実装
```

## 付録: 限界

- 聴感評価（listening）は未実施。P1-2 は**コード読解 + P1-1 の実測**に基づく静的な再監査であり、新しい測定は行っていない。
- P1-1 の測定は 48 kHz / block 512 / identity EQ + conv bypass / staging 0 dB の固定条件。他の設定（EQ ブースト・IR 有無・staging 非 0）での非線形段の挙動差は未測定。
- 「既定値の見直しが必要か」という校正判断は本監査の対象外（採用判断工程の HOLD 項目）。
