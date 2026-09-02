# D162-1R-B Work Report — Minimal Attribution Instrumentation（最小帰属計装）

- Work item: D162-1R-B（D162-1R Phase B-1・診断専用計装。寿命・所有権・DSP 動作は不変）
- Date: 2026-09-02
- 判定: **PASS**（§14 判定基準・下記 §7）
- Evidence: evidence/D162-1RB_EVIDENCE_SUMMARY.md（本 report の数値根拠）/ D162-1RB_short_soak.log（20,577 行超）/
  D162-1RB_build_{rwdi,debug,asan}.log / D162-1RB_asan_smoke.log + D162-1RB_asan_stderr.txt

## 0. Executive Summary

1. **D162-1P の「未帰属 ~70 MB/DSP」は全額帰属完了**。retained DSP 1 個の実測 footprint は **142.48 MB**
   （Convolver 固定バッファ 100 + NUC pair 36.0 + IPP 1.73 + irData 2.93 + latency 0.125 + EQ 1.69）で、
   D162-1P 実測の private 増加 **141.7 MB/gen** と **Δ0.5%** で閉包した（AC-R3/R5 達成）。
2. **D162-1R-A 静的モデルの訂正 1 件**: 「latency 23.5 MB」は AudioEngine 保有の crossfade レイテンシ
   バッファ ×4（`AudioEngine.Processing.PrepareToPlay.cpp:177-196`）で **engine-level・1 回のみ**の確保。
   per-DSP の latency は `HistoryRuntimeState::fixedLatencyBufferL/R` = **131,104 B (0.125 MB)**。
   これにより R-A の「静的 169 MB vs 実測 141.7 MB (Δ12-15%)」の残差が解消（169 − 23.5 = 145.5 ≈ 142.5 実測）。
3. **H1 の帰属実証**: retained DSP（非 publish）は 11/11 gen で同一 pointer が
   construct → retained →（破壊されない）を辿り、全件が 142.48 MB を保持。destroy は 2 件のみ
   （placeholder + 置換された published）で D162-1P と同一構造。
4. **production 動作は不変**。変更は全て `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ガード内。
   非 DIAG ビルドには 0 差分。ASAN smoke で計装経路はエラー 0（shutdown 時の UAF は D162-1P 既知の
   事前存在課題・§6.4）。

## 1. 実装対象とログ契約（§1/§2/§4/§5/§6/§7 対応）

| ログ | 出力点 | 実サイズの源泉 |
| --- | --- | --- |
| `[CONV_FOOTPRINT] conv=%p delay dry smoothing oldDry wet fadeRamp TOTAL` | ConvolverProcessor::prepareToPlay 末尾（Lifecycle.cpp） | capacity メンバ × sizeof(double) × ch（6 種固定バッファ） |
| `[NUC_ALLOC] nuc=%p seq=%llu layer=%d kind=<15種> bytes=%zu` | SetImpulse 完了時（MKLNonUniformConvolver.cpp） | `Layer::allocSizes`（既存） |
| `[NUC_ALLOC] kind=ippSpec / ippWork` | 同上 | `ippsFFTGetSize_R_64f` 再問い合わせ（createPlan と同一引数・純サイズクエリ） |
| `[NUC_FOOTPRINT] nuc=%p persistent scratch ipp TOTAL` | 同上 | persistent = SoA IR + SoA FDL + tail + delayLine + ring + direct / scratch = AoS 中継 + fft + accum + inputAcc |
| `[DSP_ALLOC] dsp=%p gen=%llu kind=<6種> bytes=%zu` | enqueuePublicationIntentForRuntimeCommit（Commit.cpp） | 下記 capture 経由 |
| `[DSP_FOOTPRINT] dsp=%p gen=%llu phase=construct/retained convolver irData nuc ipp latency eq oversampler=UNMEASURED loudness=UNMEASURED truePeak=UNMEASURED other TOTAL` | construct: DSPCore::prepare 末尾 / retained: 上記 enqueue 点 | `DSPCore::diagCaptureFootprint()` |
| `[DSP_DESTROY_FOOTPRINT] dsp=%p gen=%llu trackedFootprint=(内訳)` | destroyDSPCoreNode 破壊前（Threading.cpp） | DSPCore 保存値（capture 時実測） |
| `[DSP_FOOTPRINT_RELEASED] dsp=%p remaining=0` | destroyDSPCoreNode 破壊後 | — |

- capture の主点を **enqueuePublicationIntentForRuntimeCommit** に置いた理由: NUC / irData は
  DSPCore::prepare の**後**の rebuildAllIRsSynchronous で構築される（D162-1P soak の順序実証）ため
  prepare 末尾では NUC=0。enqueue 点は (DSPCore*, generation) の唯一の合流点で、**全 generation
  （published 49 を含む）を 1:1 で記録する**。generation はここで stamp され、destroy 側の
  `[DSP_DESTROY_FOOTPRINT]` が保存値を参照する（D162-1R-A 案の TLS ownerCtx / 新規所有権テーブルは
  **不使用**・§3 準拠。既存オブジェクト identity: DSPCore → convolver（ConvolverProcessor）→
  StereoConvolver → NUC ×2 の構築関係をそのまま辿る）。
- IPP: FFTBackend.cpp は変更せず、NUC 側（スコープ内）から同一引数の `ippsFFTGetSize_R_64f` で
  spec/work を再問い合わせ（アロケーション無し）。実測値として ipp= に計上。
- `other` は実測分類外のみ（現行 0 固定）。**推定値の混入なし・残差の押し込みなし**（§6 準拠）。
  oversampler/loudness/truePeak は UNMEASURED として TOTAL から除外。

## 2. 変更ファイル（全て DIAG ガード内・挙動変更なし）

| ファイル | 変更 |
| --- | --- |
| src/DiagnosticsConfig.h | `diagFootprintLog(juce::String)` ヘルパ（DIAG ブロック内・JuceHeader include 追加） |
| src/ConvolverProcessor.h | DIAG 構造体 `DiagFixedBufferFootprint` + `diagFixedBufferFootprint()` + `diagActiveEngineFootprint()`（public・読取専用） |
| src/convolver/ConvolverProcessor.Lifecycle.cpp | prepareToPlay 末尾に `[CONV_FOOTPRINT]`（+ DiagnosticsConfig.h include） |
| src/MKLNonUniformConvolver.h | DIAG `diagFootprintBytes(persistent, scratch, ipp)`（public・jassert 無し）+ `m_diagIppSpec/WorkBytes` メンバ |
| src/MKLNonUniformConvolver.cpp | SetImpulse 完了時に `[NUC_ALLOC]`（layer×kind + IPP）+ `[NUC_FOOTPRINT]` |
| src/audioengine/AudioEngine.h | DSPCore に DIAG `DiagFootprint` 構造体 + `diagGeneration`/`diagFootprintCaptured` + `diagCaptureFootprint()` 宣言 |
| src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp | `diagCaptureFootprint()` 実装 + prepare 末尾に construct capture + `[DSP_FOOTPRINT] phase=construct` |
| src/audioengine/AudioEngine.Commit.cpp | enqueue 点で gen stamp + retained 再 capture + `[DSP_ALLOC]`×6 + `[DSP_FOOTPRINT] phase=retained` |
| src/audioengine/AudioEngine.Threading.cpp | destroyDSPCoreNode に `[DSP_DESTROY_FOOTPRINT]` / `[DSP_FOOTPRINT_RELEASED]`（+ include） |
| src/eqprocessor/EQProcessor.h | DIAG のみ `diagFootprintBytes()`（capacity 実測合算・読取専用） |

**スコープ逸脱の申告（§10）**: 上記のうち Commit.cpp / Threading.cpp / EQProcessor.h はユーザー指定
ファイルリストに明記されていない。いずれも (a) §6/§7 が要求する計測点そのもの（enqueue=唯一の
(dsp,gen) 合流点、destroy=§7 指定出力点、EQ=§6 の `eq=` 実測に必須）であり、(b) 変更は DIAG
ガード内のログ/counter のみで挙動・所有権・retire 経路に触れない。正味 3 ファイル分の追加だが、
§10 の精神（DIAG logging/counters のみ）には整合する。裁定が必要なら revert 可能（1 commit 相当）。

**禁止事項の遵守（§10）**: allocation サイズ/条件、free 条件、ownership、unique_ptr/RAII、retire、
epoch、publication、active engine exchange、queue、DSP processing、IR layout、AoS/SoA、crossfade
— いずれも未変更。`[LIVE_LEAK]` は作成せず（§8）。global LiveAllocRegistry / TLS ownerCtx は
**実装せず Phase B-2 に延期**（§9 準拠）。

## 3. B-1 ビルド（§12）

| 構成 | 結果 |
| --- | --- |
| RelWithDebInfo + DIAG=ON | EXIT=0（evidence/D162-1RB_build_rwdi.log） |
| Debug + DIAG=ON | EXIT=0（evidence/D162-1RB_build_debug.log） |
| RelWithDebInfo + DIAG=ON + ASAN | EXIT=0（evidence/D162-1RB_build_asan.log） |

## 4. B-2 短縮実行（§12/§13）

- 条件: D162-1P と同一（IR/SR/block/OS）で count のみ 60→12。RWDI+DIAG バイナリ。
- 結果: EXITCODE=0x00000000。11 generation 完走（gen=5..15、12 リロード要求の最終 1 件は exit
  window 外 — 計装正しさ確認用のため影響なし）。
- 構造の再現（§13 期待結果）: published / non-published / DSPCore live の関係は D162-1P と同一
  （DC live 1→10、SC live = DC live、NUC live = 2 × SC live、destroy = 2 件のみ）。

## 5. B-3 必須証拠突合（§12）

| 突合 | 結果 |
| --- | --- |
| enqueue gens == retained gens | [5..15] == [5..15] 完全一致 |
| construct↔retained pointer | 11/11 ポインタ一致（AC-R5） |
| `[DSP_FOOTPRINT]` nuc+ipp ↔ 2 × `[NUC_FOOTPRINT]` | 39,551,808 == 2 × 19,775,904 完全一致 |
| `[DSP_FOOTPRINT]` convolver ↔ `[CONV_FOOTPRINT]` TOTAL | 104,857,600 完全一致 |
| irData | 192,000 × 8B × 2ch = 3,072,000 完全一致 |
| `[DSP_DESTROY_FOOTPRINT]` ↔ retained 行 | pointer+gen+footprint 一致（gen=5 置換分） |
| `[IR_RELEASE]` / `[IR_LOAD]` / `[MEM_SNAP]` | 既存診断は従来通り動作（22/22/1085 行） |

## 6. 判定材料

### 6.1 PASS 条件（§14）の実測閉包

```text
142.48 MB (measured per retained DSP)
  ≈ 141.7 MB/gen (D162-1P measured private slope)
  Δ ≈ 0.5%
```

残差 ~0.8 MB は UNMEASURED 小カテゴリ（oversampler/loudness/truePeak・KB〜MB 程度）と
MEM_SNAP の MB 丸めで説明可能。**retained DSP ↔ footprint は pointer/generation 単位で 1:1 対応**。

### 6.2 D162-1P「未帰属 ~70 MB」の解消

11 gen すべてで、非 publish retained DSP が Convolver 100 MB + NUC pair 36 MB を 1:1 で保持。
D162-1R-A が静的に主張した「ConvolverProcessor 固定バッファ ≈ 100 MB = retained DSP の主項」が
**runtime 実測で確定**（§11 の要求どおり「retained object 内部の帰属」として実証）。

### 6.3 latency の再分類（§5 対応時の発見）

`[DSP_ALLOC] kind=latency` の実測は **131,104 B（0.125 MB）**。D162-1P/R-A が per-DSP と数えた
「latency ×4 ≈ 23.5 MB」は AudioEngine 保有（1 回のみ確保）。これにより静的モデルの per-DSP
期待値は 169 → 145.5 MB に訂正され、実測 142.48 MB と一致。

### 6.4 ASAN / NO-GO 検査（§14）

- 計装経路に ASAN エラー **0**（smoke 3 gen で全ログ正常出力）。
- shutdown 時に D162-1P 既知の teardown UAF と**同一 signature** を確認
  （`tryShutdownQuiescentReclaim` ← `EQCacheManager::CacheMap::~CacheMap`。
  行番号 4428 → 4454 は本計装が AudioEngine.h に追加した 26 行分のオフセットと正確に一致し、
  スタック・アドレスパターンも同一 → **本計装に起因しない事前存在課題**）。
- DSP lifecycle / audio thread allocation / retire・reclaim 挙動に変化なし。

## 7. 判定

**PASS** — §14 の PASS 条件（footprint 閉包 + pointer/generation 1:1 対応）を満たす。
HOLD 条件（~10% 以上の未説明残差）は不該当 → **B-2（LiveAllocRegistry）は不要**。
NO-GO 条件（lifecycle 変化・audio thread allocation 増・ASAN failure・対応崩れ・retire 変化）
はすべて不該当。

## 8. 次工程の推奨

1. **D162-2（retention 修正）の起票** — D162-1P 分岐基準「allocation で 123–141 MB/DSP が閉じる」を
   本計装で 142.48 MB 実測閉包が確認したため、起票条件を満たす。修正対象は H1（非 publish DSPCore
   の retire/destroy 漏れ）で確定済み。
2. 長時間 soak（60 gen・同一条件）での [DSP_FOOTPRINT] 閉包の再確認（本計装のまま実施可能）。
3. 別トラック: shutdown teardown UAF（D162-1P 発見分・本 report §6.4）は本件と独立して起票推奨。
4. 非 DIAG ビルドへの影響 0 を既存 regression（CTest 40/40）で最終確認（本 report は DIAG ビルド系
   のみ実施）。
