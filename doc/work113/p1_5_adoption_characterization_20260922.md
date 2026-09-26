# P1-5 HOLD 解除用 Characterization 報告書（2026-09-22）

## 1. Scope

- **目的**: P1-4 で HOLD とした D2（default ON）/ D4（calibration）を判断可能にするための characterization。
- **規約（遵守）**: production change = 0・CMake = 0・default 値 = 0・flag 操作 = 0・calibration = 0・**commit = 0**。測定コードは **test-only TU（`src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp`）の拡張のみ**（production instrumentation なし）。flag は最終的に **OFF** に復帰済み。
- **本報告の結論（先出し）**: **P1-5 = HOLD**（測定キャンペーン未完了）。D2 / D4 は **HOLD 継続**。理由は §13/§14 のとおり（ON build が in-session で確保できず、ON 側の比較データが取得できない）。

## 2. Fixed State / Source Identity

| 項目 | 値 |
|------|----|
| HEAD | `1e9e63e3`（P1-5 中は commit 0） |
| production diff | **0**（`src/audioengine`・`src/CustomInputOversampler.{cpp,h}`・`src/dsp`・`src/eqprocessor`・`src/convolver`・`CMakeLists.txt`・`build.bat` すべて差分なし） |
| flag（cache 実値） | `CONVOPEQ_CORRECT_POLYPHASE_GAIN:BOOL=OFF`（測定後に既定へ復帰） |
| Phase 0 battery（復帰後） | `PolyphaseGainFidelityTests` = **PASS=54 FAIL=0** |
| 測定コード | test-only TU 拡張（`Program` 信号・可変 capture 長・WAV 出力・IR ロード・`kP15FullMatrix` ガード）＝**未 commit** |
| 使用 IR | `tmp/p15_ir_g0.wav` / `_g3.wav` / `_g6.wav`（float32 stereo・1024 frames・spike gain 1.0 / 1.4125 / 1.9953） |
| 測定識別 | 各ログ先頭の `[P1CHAR] flag_macro=` を evidence の一次識別子とする |

## 3. Preset Replay（OFF 側のみ・ON 側は未取得）

条件: 1 kHz・−6 dBFS・sat=0.1・conv bypass・EQ identity。`gainDb` は chain 伝達利得（入力に対する出力基本波の比）。

| Case | Headroom | Makeup | Trim | OS | OFF gainDb（実測） | 内訳（staging + OS droop + kOutputHeadroom） |
|------|---------:|-------:|-----:|---:|--------------------|---------------------------------------------|
| P1 | −6 dB | +12 dB | 0 | Auto | **−2.5035 dB** | +6 − 7.4963 − 1.0 = −2.4963 |
| P2 | 0 | 0 | 0 | Auto | **−8.5035 dB** | 0 − 7.4963 − 1.0 = −8.4963 |
| P3 | −6 | +6 | 0 | Auto | **−8.5035 dB** | 0 − 7.4963 − 1.0 = −8.4963 |
| P4 | −6 | +12 | +6 | Auto | **−2.5035 dB** | trim は conv bypass 時**非適用**（下記注） |
| P5 | −6 | +12 | 0 | 2x | **+2.4937 dB** | +6 − 2.4988 − 1.0 = +2.5012 |
| P6 | −6 | +12 | 0 | 4x | **−0.0048 dB** | +6 − 4.9975 − 1.0 = +0.0025 |
| P7 | −6 | +12 | 0 | 8x | **−2.5035 dB** | +6 − 7.4963 − 1.0 = −2.4963 |

- **OBSERVATION**: P1（Auto）と P7（8x 明示）が**同一値**（−2.5035）＝ Auto は 48 kHz で 8x に解決される契約を実測で確認。staging 差（+6 dB 単位）が gainDb にそのまま現れる（P1 vs P2 = +6.000 dB ✓）。
- **OBSERVATION（副次・flag 独立）**: P4 の `convolverInputTrim +6 dB` が**出力に現れない**（P4 == P1）。実装では trim 適用が `!convBypassed` ガード内にある（`DSPCoreDouble.cpp:444-453`）ため、**convolver が bypass のとき trim は効かない**。本 adoption とは無関係の既存挙動として記録。
- **未取得**: **ON 側の同 7 行**（§13 の build 障害）。したがって「同一 preset value が ON/OFF でどう変わるか」の Δg は **本測定では確定していない**（P1-1/P1-3 の一般則から +2.4988·N dB と推定されるが、P1-5 の実測証拠としては未確立）。
- limiter 到達・hard clamp: OFF 側は全行 **0**（−6 dBFS では limiter 非接触 ✓）。

## 4. EQ + Oversampling（OFF 側・全 18 条件取得済み）

| 条件 | 実測 gainDb（OFF） | 解釈 |
|------|--------------------|------|
| os=1・1 kHz band +3 dB・−20 dBFS | **+1.9930** | 3 − 1.0 = +2.0（kOutputHeadroom）✓ |
| os=1・+6 / +9 dB | **+4.9930 / +7.9930** | ブースト段差が厳密に +3.000 dB ✓（決定性の control） |
| os=4・+3 dB | **−5.6273** | 期待 +3 − 4.9975 − 1.0 = −3.0 に対し **−2.63 dB 低い** |
| os=8・+3 dB | **−8.4083** | 期待 +3 − 7.4963 − 1.0 = −5.5 に対し **−2.91 dB 低い** |
| os=4/8・sc0 == sc1 | 完全一致（clipEngMax=0.000000） | クリップ非作用（低レベル）✓ |

- **OBSERVATION（本 adoption の外側・flag 独立）**: OS 倍率を上げると、**指定周波数（1 kHz）における EQ バンドの利得が縮小**する（+3 dB 指定が os1 で +2.0 dB、os4 で −5.6 dB、os8 で −8.4 dB）。これは「バンドが OS レート側で処理され、指定周波数が OS レート基準で解釈される」ことと整合する（os4 で 250 Hz、os8 で 125 Hz にピークが移る挙動と等価）。**OFF/ON で同一**（sc0 == sc1・後述の flag 効果とは独立）であり、案E の採用とは無関係な**既存挙動**として記録し、別途の調査対象とする（本 P1-5 の範囲外・判断しない）。
- **未取得**: ON 側の同 18 条件（Δg が (4/3)^N を維持するかの判定は未実施）。

## 5. IR + Oversampling（測定**不成立**）

- 実測値は同一条件で再現せず（例: 同一 IR g0・os=1・sc0 で、amp=−20 は **−14.5035 dB**、amp=−6 は **−13.0276 dB** — 線形域で 1.48 dB の不一致）、隣接行で値が「1 run 遅れ」のパターン（行 N の sc1 == 行 N+1 の sc0）を示した。
- 原因候補（OBSERVATION からの推定）: convolver の IR ロード/finalize と OS レート変更に伴う再準備の**整定不足**（`isIRFinalized()` が前回ロードの状態で true を返し得るため、待機判定が成立しない）。既存 harness の `irwet` 計測は `waitBacklogZero` + 2000 ms sleep を挟んでおり、本 TU にはそれが無い。
- 併せて、エンジン側は IR を target length（実測 `irLen` = processingRate × 1 s: os1=48000 / os4=192000 / os8=384000 samples）へ加工するログ（`[IR_TAIL_GEOM]`・`[L0_WRITE]`）を出しており、**IR の実効スカラー利得はロード構成に依存**する（テスト IR の nominal 0/+3/+6 dB ラベルは絶対値としては成立しない）。
- **判定**: 本項目は **測定不成立（invalid）**。Step 3 の目的（conv path における二重補償の再確認）は **未達**。必要な修正（整定待ちの追加・`loadImpulseResponse` 戻り値の確認・必要なら 2 連続測定の一致確認）を §13 に記録。

## 6. Staging × Nonlinear（OFF 側・全 12 条件取得済み）

- `staging{−12,−6,0,+6} × os{2,4,8} × amp{−20,−6,0}` の OFF 側 12 条件を取得（`tmp/p15char_off_partial.log`）。
- 代表値: staging 0・os8・−20 dBFS で **gainDb_sc0 = −8.5035**（= 0 − 7.4963 − 1.0 ✓）、staging −6・os8・−6 dBFS で **−14.5035** ✓ — P1-3 §6.3 の値と一致（再現 ✓）。
- **未取得**: ON 側（非線形域の Δg・clipEngMax の staging 依存は未確定）。

## 7. Saturation Sweep（OFF 側・部分取得）

- 全 42 条件のうち **18 条件**（sat ∈ {0, 0.1, 0.25} 相当）を OFF 側で取得（partial log）。残り（sat 0.5 / 0.75 / 1.0）と **ON 側全条件**は未取得。
- したがって「flag ON による clipEngMax の増加量の sat 依存」は **定量化未了**（P1-1/P1-3 では sat=1.0 のみ: clipEngMax が OFF 0.0018〜0.0687 → ON 0.0182〜0.1473）。

## 8. Limiter Engagement Sweep（未実施）

- `amp −20..0 step 2 × os{1,4,8}`（33 条件）は**未実施**（時間予算・build 障害のため）。
- 参考（P1-1/P1-3 の既知値）: os≥2・0 dBFS で limiter 到達 **0 → 119/4096**。到達開始レベルの境界（sweep）は未取得。

## 9. Listening Evaluation（AB/ABX 素材: **生成済みだが無効**）

- 6 カテゴリ（dry / identity EQ / EQ +6 dB / IR / SoftClip / limiter 近傍）× OFF/ON の 12 WAV を生成（各 327,724 B = 81,920 frames × 2ch × 16 bit = **1.7 s**、Program 信号: 1 kHz −6 dBFS 0.6 s → クリック 0.5 ms → 決定論的ノイズ −12 dBFS 1.0 s）。
- **重要な限界**: これらの WAV は**すべて OFF build（flag_macro=0）が生成**した。ON build が確保できなかったため、`_on.wav` も **flag OFF の出力**であり、**AB/ABX 素材としては無効**（OFF vs OFF の比較になる）。
- したがって Step 7 の素材提供（および聴感評価の evidence 化）は **未達**。

## 10. A/B/C Evidence Classification

| 分類 | 内容 |
|------|------|
| **A — invariant（P1-5 で再確認）** | latency（P1-1/P1-3 で OFF==ON）・hard clamp 0（本測定の全取得行で 0）・係数/topology・factor 非依存の閾値（P1-2/P1-3） |
| **B — deterministic behavioral change（P1-1/P1-3 で確立・P1-5 では未追加）** | output level（+2.4988·N dB）・SoftClip drive（clipEngMax ×5〜10 @sat1.0）・limiter 到達（0→119/4096 @os≥2・0dBFS） |
| **C — subjective / insufficient（本 P1-5 で不足が確定）** | preset compatibility の ON 側実測・EQ/IR 併用の ON 側・staging × 非線形の ON 側・saturation sweep 完了・limiter sweep・listening（素材自体が無効） |

## 11. D2 Impact（default ON）

**HOLD 継続。** P1-5 の主眼である「同一 preset value が ON/OFF でどう変わるか」は **ON 側データが無いため未確立**。したがって D2 を DECISION-READY に上げる材料は本工程では得られていない（P1-4 の評価を維持）。

## 12. D4 Impact（calibration）

**HOLD 継続。** §6〜§8 の sweep（staging × 非線形 / saturation / limiter）は **ON 側が未取得**で、校正判断に必要な「非線形域での Δg と clipEngMax の sat 依存」「limiter 到達開始レベルの移動量」は確定していない。

## 13. Unresolved Items

1. **ON build が in-session で確保できず**（`AudioEngineHarness` のコンパイルで **`fatal error C1060`（コンパイラのヒープ枯渇）**。初回は OFF build 直後の並行実行、再試行でも同エラーが再発。コードの誤りではない）。この結果、P1-5 の ON 側測定はすべて未取得。
2. **IR + OS 測定の不成立**（§5）: 整定待ちの実装が必要（`waitBacklogZero` + 固定 sleep + 2 連続測定の一致確認、`loadImpulseResponse` の戻り値確認）。
3. **測定レート**: 1 config-pair あたり実測 **約 37 s**（world publish 待ち + 800 ms settle + 16 ブロック処理 + DSP 再構築）。全行列（142 pair × 2 build）は 1 build あたり 90 分超となり、in-session では非現実的 → **matrix の絞り込み**が必要（`kP15FullMatrix=false` で preset + listening のみに縮小済み）。
4. **EQ × OS の周波数マッピング**（§4）: 指定周波数での EQ 利得が OS 倍率で縮小する（flag 独立・既存挙動）。adoption の範囲外だが、製品影響の観点で別途調査が必要。
5. **Convolver input trim の bypass 時非適用**（§3 P4）: 既存挙動（flag 独立）。
6. 聴感素材の再生成（ON build 確保後）と listening 評価。

## 14. PASS / HOLD

```text
P1-5 = HOLD
  - 理由: ON 側の characterization が取得不能（ON build の C1060）＋ IR 測定の不成立 ＋ 測定レート制約
  - 停止条件該当: 「characterization 用の test harness だけでは測定できない」項目が残存（IR 整定・ON build のリソース）
  - production / CMake / default / flag / calibration / commit: すべて 0（§2）・flag は OFF に復帰済み・refgate 54/54
D2 Default ON   = HOLD（P1-4 の状態を維持）
D4 Calibration  = HOLD（P1-4 の状態を維持）
D1 / D3 / D5    = P1-4 の状態を維持（本工程では変更しない）
```

**次工程（提案）**: ① ON build を単独・低並列（`cmake --build ... -- -j 2`）で実行して確保 → ② 絞り込み matrix（preset 7 pair + staging 4 pair + sat 6 pair + limiter 11 点 + listening 6 pair ≈ 34 pair + 11 run ≈ 25 分/build）で OFF/ON を取得 → ③ IR 測定は整定修正後に再試行 → ④ 聴感素材を ON build で再生成。

## 15. Appendix — Measurement Identity

| 項目 | 値 |
|------|----|
| ログ（有効） | `tmp/p15char_off.log`（19 p15 行・`flag_macro=0`）= preset 7 行 + listening 12 行 |
| ログ（部分・OFF 側のみ） | `tmp/p15char_off_partial.log`（73 p15 行・`flag_macro=0`）= preset/EQ/IR/staging 完了 + sat 途中 |
| ログ（**無効**・破棄） | `tmp/p15char_on.log`（`flag_macro=0` のため ON 測定と見なさない） |
| WAV（**無効**） | `tmp/p15_listen_*.wav` 12 ファイル（すべて OFF build 出力） |
| test IR | `tmp/p15_ir_g0.wav` / `_g3.wav` / `_g6.wav` |
| 復帰確認 | `tmp/p15b_refgate.txt` = PASS=54 FAIL=0・cache `CONVOPEQ_CORRECT_POLYPHASE_GAIN:BOOL=OFF` |
| snapshot | `ConvoPeq.md`（P1-3 後に再生成したものを使用。本工程では production 変更なしのため内容は production 不変） |
