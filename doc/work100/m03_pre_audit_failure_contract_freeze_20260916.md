# M-03 Pre-Audit / Failure Contract Audit (read-only)

- **日付**: 2026-09-16
- **対象**: 2026-09-13 監査系 M-03「Experimental Direct Head の整合不良」
- **根拠文書**: doc/audit/ConvoPeq_IR_Audio_Path_Audit_2026-09-13.md §M-03 /
  ConvoPeq_Bug_Verification_2026-09-13.md §6（判定: 確認）/ BugList_and_FixPlan §6 Phase 3
- **前例**: work97（M-04）・work98（M-02）・work99（M-01）
- **方式**: read-only（production 変更 0 / commit 0）。baseline origin/main `53847087`
- **基準**: ConvoPeq.md 2026-09-16 01:00:43（M-01 適用後・FRESH・`--check` NEWER_SRC_COUNT=0 確認済）
- **中核監査条件（ユーザー指示）**: 「direct head の期待位置」と「実際の wet 到着位置」を
  **推測で同一視しない**。§2 の静的導出は「候補モデル」であり、確定事実として §3 の申告鎖に
  接続しない。仲裁は T-M03-1 実測（§7）に委ねる（決定点 D1）。

---

## 1. Source Trace — Direct Head 実コード全景

### 1.1 構築（NonRT・SetImpulse）

```
kMaxDirectTaps = 32（cpp:703）
directPart = nextPoT(max(blockSize,64))            （:704）
m_directTapCount = min(irLen, min(directPart, 32)) 有効時（:705）
m_directIRRev[i] = impulse[tapCount-1-i] * scale     時間反転・生の IR（:730-731）
memset(impulseForFft, 0, tapCount)                   FFT 側は先頭 tapCount を零化（:744-745）
→ 以後 partition/irFreq 構築 → applySpectrumFilter（:1104-1105・**irFreqReal/Imag のみ**）
```

- **HC/LC 不適用は静的確定**: direct taps（m_directIRRev）は applySpectrumFilter 対象外・
  スケールのみ。FFT 残り taps（32 以降）は HC（>18k/22kHzshelf）・LC・tail ゲイン適用後。
- 零化と direct 抽出は同一 tapCount → **二重計上なし**（和集合=完全 IR）。

### 1.2 実行（RT・processDirectBlock :1346-1409 / Get :1787-1798）

- Add 冒頭で `processDirectBlock(input, numSamples)` → 32tap FIR を現ブロック +
  `m_directHistory`(≤31) から即計算 → `m_directOutBuf`、`m_directPendingSamples = numSamples`。
- 内部 hygiene 既存有: `isFiniteAndAboveThresholdMask(y, 1e-20)`（:1396-1397、非有限/tiny→0）
  — **M-01/M-02 の防塞とは独立の direct 専用ガード**（意味論は policy 1e-20 + NaN/Inf 一并）。
- Get: `ringRead(L0) → + direct(min(numSamples, pending)) → + L1/L2 delayLineReadAdd(t0−o_L)` の
  同一ストリーム index へ加算。**direct は o_L 変換を介さず ringRead と同一座標**（§2）。
- numSamples > m_directMaxBlock（=max(blockSize,1)）時: pending=0 で direct のみ黙秘 return（:1351-1356）。
  RT 側の正常経路では callLen（=preferredCallSize=prepare の bs）≤ directMaxBlock で成立（不変条件）。

### 1.3 申告鎖（PDC）

```
updateLatencyCache()（StateAndUI:745-750, cachedLatency snapshot）
  ← getLatencyBreakdown()（StateAndUI:686-730）:
      algorithmLatencySamples = directHeadActive ? 0 : max(0, conv->latency)  ※conv->latency=l0Part（NUC:1102）
      irPeakLatencySamples    = irLatency（H-02 実測ピーク）
      totalLatencySamples     = 両者和
AudioEngine::getTotalLatencySamples()（+ OS/softclip 合成・base rate 換算）
AudioEngineProcessor::prepareToPlay → setLatencySamples(...)（:38）→ ホスト PDC
```

- ON 時 algo=0 は **PDC・retarget（Runtime:129/167/309/328）・refreshLatency の全読出点で共通**。
- H-01 旗（`kDryDelayUsesAlgorithmLatency=false` 現行本番値・h:641）: dry 読出遅延は
  ON/OFF 共通で **irPeak のみ**。retarget 判定・PDC は報告基準（partSize+peak / peak）のまま。
  → direct 有効時の「algo=0」は dry 側挙動を変えない（変換恒等・H-01 §h641 コメント実证済）。
- `LatencySnapshot.hasParallelDryPath = directHeadActive`（h:486/:750）は**消費者 0**（書込専用・棚卸し §5-3）。

### 1.4 到達可能性・有効化経路

- 既定 false（ConvolverProcessor.h:82/:1068）。UI「Exp Direct Head」トグル（ControlPanel:441/1035）→
  `setExperimentalDirectHeadEnabled`（Runtime:1104-1110）→ 変更不要時 rebuild（トグル文言
  「Rebuilds the convolver when changed」）→ LoaderThread/finalize で新 engine に `enableDirectHead` 伝播
  （LoadPipeline:673・Lifecycle:300・LoaderThread:223・init 引数 h:837/:899）。
- **ValueTree 永続化**（StateAndUI:242/:368）: ON 設定はセッション横断で再現。experimental 命名どおり
  opt-in 機能。**既定構成では非到達**（重大度 §6）。

## 2. 時間座標 — 4 系統の静的モデル（候補。T-M03-1 が仲裁）

```
                    [taps 0..31]        [taps 32..l0Len]      [L1/L2 tail]      dry
direct OFF           FFT（零化なし）      FFT                  delayLine(t0−o_L)  ring read（irPeak）
direct ON  期待      direct FIR 即（cb k 内）  FFT（零化後）    同上              同上
```

**候補モデル（静的導出・未実測）**: 分割畳み込み（各 IR パート長 ≤ partSize、OLS の
[prev|cur]→後半 valid = 現ブロック領域）は、ブロック k の出力をコールバック k で完全生成しうる
→ 追加レイテンシの実効は **ブロック量子化 ≤ l0Part−bs（bs==l0Part で ≒0）** の範囲。
このモデルが成立する場合:
- BugList §M-03「残り L0 IR は partSize 分のブロック待ちなのに latency=0 申告」の
  **「partSize 待ち」前提が成り立たない**（ON の algo=0 申告は正しい／逆に OFF の
  partSize 申告側が過大）→ M-03 の是正対象が反転しうる。
- 成立しない場合（実測で wet ≒ partSize 遅延）→ BugList 記述どおり（ON 申告 0 は過小申告）。
**いずれにせよ §1.3 の実測 quantum と §3(b) の申告値の突合がなければ修正対象は確定できない**
（監査条件: 期待位置＝到着位置の同一視禁止）。bypass 経路は direct 非使用（dry-only・irPeak）で
ON/OFF 無差（Runtime:155-159 + H-01 旗 false）。

## 3. BugList 三点検証の確定状態

| 主張 | 静的状態 | 残る争点 |
|---|---|---|
| (a) direct は最大 32 tap（L0 partSize 512-1024） | **TRUE・確定**（§1.1） | 是正として tap 拡張（→partSize 相当）は D2/D1 依存の選択肢であって欠陥ではない |
| (b) latency=0 申告 | 申告コード上 TRUE（§1.3）。**実遅延との不整合は未確定** | §2 の候補モデル反転を含む。**T-M03-1 実測まで修正方向を確定しない** |
| (c) HC/LC 不適用 | **TRUE・確定**（§1.1） | 可聴有意性（32 tap ≒0.7ms 窓の >18kHz ステップ）と是正方式（D2） |

## 4. 隣接・既存防塞との責務分離（非接触境界）

- **M-01**: L1/L2 pre-IFFT hygiene（guard 追加済）。direct 経路は guard 対象外（guard 前段の
  DSP ではない）で無関係。接触禁止。
- **M-02**: direct 加算は Get 内 → wetOut に含まれ、出力 scrub（Runtime:787）が直接覆う。
  direct 内の非有限は :1396 ガードが一次吸収。**分離維持・M-02 の scrub/counter に接続しない**。
- **SR-01/02/03・H-01/H-02・B13（o_L/delayLine・I4 clock）**: 一切変更禁止。
  direct は ring/tail のストリーム座標に同一 index で加算されるだけで o_L 変換対象外（§1.2）—
  この加算位置自体は B13 契約と無衝突（T-M03-4 で回帰確認）。
- Publish/Retire/Crossfade/Epoch・Coordinator・MAX_*・engine edge sanitize: 接触 0 のまま凍結。
- 新規閾値・新規 atomic・RT ログ新設禁止（M-02/M-01 同一規律）。

## 5. 棚卸し（監査中に確定した付随事実）

1. `hasParallelDryPath`（LatencySnapshot/h:486）: 書込のみ・消費者 0 — 死蔵 or 予約。是正不要・記録のみ。
2. `m_directMaxBlock` ガード（:1351）は pending=0 の黙秘 return（非決定的無音化ではない）—
   正常経路では不変条件で不達。将来の経路追加時に監査対象となりうる（記録のみ）。
3. direct tap の scale 適用は `impulse*scale`（:731）— headroom scale の二重適用有無は
   Rebuild 側の scale 管理（storedScale）に依存。M-03 対象外（異常報告なし）。

## 6. 重大度の再判定（監査所見）

- 既定 OFF の **opt-in experimental**（§1.4）。ON 時のみ顕在化うる。
- 確定欠陥は (c) HC/LC 不適用のみ（可聴有意性は小: >18kHz の 32tap 窓境界）。
  (b) は実測前のため**欠陥方向（過大/過小申告のいずれか、または非該当）未確定**。
- 暫定判定: **Medium → Low-Medium（opt-in・latent。うち (c) は Low 確定）**。
  T-M03-1 の結果により (b) の実質有無が確定する（GO 判断の主材料）。

## 7. テスト契約（先行凍結・実測が仲裁）

| ID | 内容 |
|---|---|
| T-M03-1 | **期待/到着の分離実測**: taps 10（direct 域）と 600（FFT L0 域）に双 peak を持つ IR で、OFF/ON 各構成の wet 初回到達 index（peak 位置）を計測。(i) OFF 実測遅延 vs 申告 l0Part の突合、(ii) ON 実測 direct 域/FFT 域の相互間隔が理論値（peak tap 差 590）と一致すること、(iii) 申告 PDC と実測の |Δ| を閾値内（数値ブロック量子化 ≤ l0Part）に収めることを**契約値として凍結**。結果を §2 の候補モデルの採否判定に使う（実装前ゲート） |
| T-M03-2 | HC/LC 連続性: ON 時の direct 域（taps<32） impulse 応答と FFT 域応答の >18kHz 帯域ゲイン比が spec 通りか（(c) 実測基線。D2 採択時の回帰台） |
| T-M03-3 | 非退行: H-01 全テスト（T-H01-1..6）、H-02、SR-01/02/03、M-01、M-02、M-04 が ON/OFF 両状態・両構成で既存 assert 不変（T-H01-6 は taps<32 を検証しない=本件非反証である点を §8 に記録済。new T-M03-1 が初検証） |
| T-M03-4 | bypass/retarget 無差確認: ON/OFF で bypass 出力・retarget 発火 that 不変（bypass は direct 非使用） |
| 回帰 | Release/Debug ctest 40/40 + 安定性再走 + 静的解析3種（M-01/M-02 と同一ゲート構成） |

## 8. 出口判定

```text
Source Trace          完了（構築/実行/申告/有効化の 4 鎖 §1）
Failure Reproduction  静的には確定不能。(b) の方向は候補モデル反転含（§2）。(c) のみ確定。
                      T-M03-1（harness 実測）が最初のゲート（期待位置≠到着位置の同一視禁止）
Time Coordinates      4 系統 + L1/L2 マトリクス §2。bypass 無差 §2 末。direct 加算座標 §1.2
Responsibility Split  M-01/M-02/B13/SR/H 系との非接触境界 §4 凍結
Severity Reassessment Medium → Low-Medium（opt-in・(c) のみ Low 確定）§6
Contract Freeze       §4（禁止）+ §7（テスト）。実装案（tap 拡張/PDC 再定義/HC-LC 適用/
                      機能縮小）は分岐前に決定点として凍結:
  D1  T-M03-1 実測採否 = 修正対象・方向（(b) の正反転ありうる）の仲裁者
  D2  HC/LC 適用範囲（direct taps へ NonRT 適用 / 受容 / direct 対象外化）
  D3  PDC 申告 authority（Q1 結果次第で OFF 側過大申告が別 item 化の可否=スコープ判断）
  D4  機能の扱い（experimental 継続 / 既定値 / 実験的注記の十分性）
Implementation Gate   **PENDING** — 上記決定点のうち D1 は「実装フェーズ最初の計測タスク」と
                      して先行実施を承認いただく形でないと確定不能（read-only 制約）。
                      cond-A: T-M03-1 の計測-only ステップを GO の第一成果物とすること
                      cond-B: §4 非接触境界の混入禁止（M-01/M-02/M-04/SR/H/B13/Publish 系）
                      cond-C: 重大度 Medium→Low-Medium（うち (c) Low）訂正の承認
```

---

## 9. T-M03-1 計測結果と (b) クローズ（2026-09-16・D1=GO 執行）

計測条件: 48000 Hz / block 512 / IR 4800 samples（`writeH01TempIr`・16bit・2ch）。`irLen` は
ロード後に 48000（1.0 s ターゲット長）となり、peak 600 は **L0 域**（l0Len 域内）に属する。
計測コード: 既存 `AudioEngineHarness` target 内の `measureM03DirectHeadTiming()`（計測専用・
assert なし・production 変更 0）。計測値は Release/Debug で完全一致（両構成 exit 0）。

| 構成 | 期待 peak | **実到着 index** | **delta** | 申告 algo / irPeak / total | directActive |
|---|---|---|---|---|---|
| OFF / direct 域 (peak@10) | 10 | **10** | **0** | 512 / 10 / 522 | 0 |
| ON / direct 域 (peak@10) | 10 | **10** | **0** | 0 / 10 / 10 | 1 |
| OFF / FFT-L0 域 (peak@600) | 600 | **600** | **0** | 512 / 600 / 1112 | 0 |
| ON / FFT-L0 域 (peak@600) | 600 | **600** | **0** | 0 / 600 / 600 | 1 |

### 判定: Case A（本件測定条件において不整合は反証）

```
M-03-(b): CLOSED — Case A measurement disproves the suspected mismatch.

ON / FFT-L0:
  expected peak = 600
  actual peak   = 600
  delta         = 0

Declared:
  algorithmLatency = 0
  irPeakLatency    = 600
  totalLatency     = 600

Therefore:
  reported algorithmLatency=0 is consistent with measured effective latency.
```

- **一般化の禁止（記録条件）**: 「latency=0 が常に正しい」ではなく、
  **§9 の測定条件（bs=512・l0Part=512・peak が L0 域・L0 のみ活性）において
  「ON の受理機構が追加遅延を導入しない」ことが実測で示され、申告と整合した**、が正しい記録。
  他条件（L1/L2 活性・beat 非整数比・非 2 冪ブロック等）への外挿は本測定の対象外。
- §2 の候補モデル（分割畳み込みはブロック k の出力をコールバック k で生成し、実効追加遅延は
  ブロック量子化以下）は **実測で支持**（delta=0・量子化すら観測されず）。「partSize ブロック待ち」
  前提は本条件で成立しない。
- (c) HC/LC は静的確定のまま **OPEN**（T-M03-2 で測定）。

### D3（OFF 側申告の過大）— 分離・裁定待ち候補として固定

```
OFF:  declared algorithmLatency = l0Part = 512
      measured FFT/L0 delta     = 0
```

- 観測事実として記録するが、**M-03 の実装変更対象にしない**。
  PDC 定義変更・`algorithmLatency` の意味変更・H-01 再変更は**本 work で禁止**（§4 維持）。
- 状態: `M-03-(b) CLOSED / M-03-(c) OPEN / OFF-side PDC = SEPARATE CANDIDATE (HOLD・実装なし)`。
  H-01 が `algo` を dry 遅延基準から外している（旗 false）ため、OFF の 512 は主に **PDC 申告値**
  として現れる。可聴影響の有無・host 側の扱いは D3 裁定時の再 Pre-Audit 対象。

### ゲート更新

```text
D1 T-M03-1 = DONE（Case A CONFIRMED）／(b) = CLOSED
D2 HC/LC scope = GO → T-M03-2
D3 OFF-side 512 over-report = SEPARATE CANDIDATE / HOLD（実装なし・起票なし）
D4 experimental policy = HOLD
M-03 Implementation Gate = PENDING（(c) の T-M03-2 結果待ち）
production 変更 0 / commit 0
```

---

## 10. T-M03-2 計測結果と (c) 判定（2026-09-16・D2=GO 執行）

計測専用（PASS/FAIL なし・値のみ）。production 変更 0 / commit 0。

### 10.1 測定構成

| 項目 | 値 |
|---|---|
| サンプルレート / ブロック | 48000 Hz / 512 |
| IR | `writeH01TempIr`（48k・2ch・16bit・4800 frame・`peakPos` 単一 impulse = 1.0） |
| 入力 | block 0 の sample 0 に impulse 1.0（wet のみ観測・`setMix(1.0)`） |
| 走査窓 | 24 block = 12,288 samples（peak 600 + マージン） |
| 直交因子 | direct(ON/OFF) × HC(Sharp/Soft) × LC(Natural/Soft) × peak 域(tap=10 / FFT-L0=600) = 16 構成 |
| 指標 | `arrival` = 出力 peak の global sample index ／ `amp` = |peak| ／ `spread` = |v| ≥ 1%·|peak| の sample 数 |
| 実行 | Release / Debug 各 1 回。**両者の M03 出力は完全一致**（同一 25 行・diff 0）。両者 exit 0 |

### 10.2 測定値（16 構成・Release / Debug 同一）

| # | direct | HC | LC | peak 域 | arrival | amp | spread |
|---|---|---|---|---|---|---|---|
| 1 | OFF | Sharp | Natural | tap(10) | 10 | 0.494371 | 5 |
| 2 | ON | Sharp | Natural | tap(10) | 10 | 0.500002 | 1 |
| 3 | OFF | Sharp | Natural | FFT(600) | 600 | 0.494371 | 5 |
| 4 | ON | Sharp | Natural | FFT(600) | 600 | 0.494371 | 5 |
| 5 | OFF | Sharp | Soft | tap(10) | 10 | 0.494371 | 5 |
| 6 | ON | Sharp | Soft | tap(10) | 10 | 0.500002 | 1 |
| 7 | OFF | Sharp | Soft | FFT(600) | 600 | 0.494371 | 5 |
| 8 | ON | Sharp | Soft | FFT(600) | 600 | 0.494371 | 5 |
| 9 | OFF | Soft | Natural | tap(10) | 10 | 0.426011 | 17 |
| 10 | ON | Soft | Natural | tap(10) | 10 | 0.500002 | 1 |
| 11 | OFF | Soft | Natural | FFT(600) | 600 | 0.426011 | 17 |
| 12 | ON | Soft | Natural | FFT(600) | 600 | 0.426011 | 17 |
| 13 | OFF | Soft | Soft | tap(10) | 10 | 0.426011 | 17 |
| 14 | ON | Soft | Soft | tap(10) | 10 | 0.500002 | 1 |
| 15 | OFF | Soft | Soft | FFT(600) | 600 | 0.426011 | 17 |
| 16 | ON | Soft | Soft | FFT(600) | 600 | 0.426011 | 17 |

導出（表からの機械的帰結。解釈は §10.3）:

- **arrival ≡ peak 域 index が 16/16 で成立**（10→10, 600→600）。HC/LC/direct の別を問わず
  追加遅延は観測されず、§9 Case A と整合。
- **tap 域（peak@10）**: OFF の `amp` は HC に依存（Sharp 0.494371 / Soft 0.426011・差 0.068360 =
  相対 13.8%）、`spread` も 5 / 17 と変化する。**ON は HC/LC に依らず `amp=0.500002`・`spread=1` で一定**。
- **FFT-L0 域（peak@600）**: ON と OFF が **全 8 対で完全一致**（HD/LC 別に 0.494371/5 または 0.426011/17）。
- **LC（Natural vs Soft）**: 8 対すべてで `amp`・`spread` に差 0（分解能以下）。

### 10.3 判定: **Case H2**（direct 域が HC 適用後の FFT 域と明確に異なる）

| 候補 | 判定 | 根拠 |
|---|---|---|
| H1（差は無視できる） | **反証** | tap 域で OFF/ON の `amp` 差 13.8%（HC=Soft）、`spread` が 1 vs 5/17。指標分解能を大きく超える |
| **H2（direct 域が HC/LC 適用と不一致）** | **確定** | tap 域 ON は `amp=0.500002`・`spread=1`（= 単一サンプル delta・生 IR 値）で HC/LC に依らず不変。同じ入力に対する FFT 域（OFF）は HC で減衰・時間拡散する。差は HC 係数に単調（Soft > Sharp） |
| H3（境界の不連続/二重処理） | **本測定では判定不能** | 時刻方向の不連続は 16/16 で不検出（arrival 完全一致）。加えて FFT 域 ON==OFF は「taps 0..31 の FFT 経路除外が taps ≥ 32 に対して透過」を示す。ただし単一 impulse IR は tap 31/32 境界に同時エネルギーを持たないため、**フィルタ状態段差（HC 適用/非適用の切替）そのものは本 IR で励起できていない** |

**責務分離の明示（誤読防止）**:

- **HC** → H2 として**実測で確定**（direct 域が HC 非適用）。
- **LC** → direct が `applySpectrumFilter` を丸ごと迂回する**単一呼び出し**であることは §3 の静的確定。
  LC 非適用はその同一経路から従うが、**本測定の指標（peak 振幅 + 1%·peak 拡散）は low-cut の効果を
  解像しない**（8 対すべて差 0）。「LC を実測で確認した」とは記録しない。
- **H3** → 未測定。判定には tap 31/32 境界に同時エネルギーを持つ broadband IR が必要（§10.4）。

### 10.4 限界（本測定が主張しないこと）

1. **単一 impulse IR** のため、direct 域と FFT 域の**同時励起**が起きていない。H2 は「direct 域の応答が
   HC 非適用である」ことの確定であり、「実 IR で境界段差が可聴/有意」ことの確定ではない。
2. HC の可聴有意性は §6 の暫定評価（>18kHz・32tap 窓境界）のまま。本測定は**係数の差**を示したが
   **聴感上の有意性は測っていない**。
3. `irLen=48000` は WAV 実長（4800 frame）ではなく **target IR 長**である。`computeTargetIRLength` は
   `originalLength` を無視し `sampleRate × targetIRLengthSec`（既定 1.0 s → 48000）を返し、loader は
   48000 にゼロ詰めして末尾 2%（min 256）をフェードする（`LoaderThread.cpp:585-602`）。peak@10 / 600 は
   いずれも実データ域（< 4544）にあり、この整形の影響を受けない。
4. 測定値の `amp` 絶対値（≈0.5）は入力 impulse 1.0 に対する wet 経路のゲイン（16bit 量子化 + 経路
   スケール）を含む。本判定は同一構成内の**相対比較**のみに依拠しており、絶対値の解釈は行わない。

証跡: `evidence/M03_HARNESS_RELEASE.log` / `evidence/M03_HARNESS_DEBUG.log`（M03 出力 25 行・両者一致・
両者 exit 0。harness 全体も `checkM01/M02/M04` PASS = T-M03-3 非退行を同時確認）。

### 10.5 ゲート更新

```text
D1 T-M03-1 = DONE（Case A CONFIRMED）／(b) = CLOSED
D2 T-M03-2 = DONE（Case H2 CONFIRMED: direct 域は HC 非適用）
              LC = 同一経路から従うが本指標では未解像 / H3 = 未測定（broadband IR 待ち）
D3 OFF-side 512 over-report = SEPARATE CANDIDATE / HOLD（実装なし・起票なし）
D4 experimental policy = HOLD
M-03 Implementation Gate = **PENDING（D2 契約裁定待ち）**
  → 裁定対象は (c) の扱いのみ: direct taps へ HC/LC を適用する / 現状を受容する / direct を対象外として明記する
  → PDC 定義変更・algorithmLatency 意味変更・direct tap 拡張・OFF 修正・H-01 再変更は引き続き禁止
production 変更 0 / commit 0
```

---

## 11. D2 契約裁定 — (c) CLOSED AS INTENTIONAL CONTRACT（2026-09-16・D2 裁定）

裁定結果: **(c) は「実装バグ」ではなく experimental direct head の意図的な非適用範囲**として契約化する。
direct taps への HC/LC 追加実装は**行わない**。production 変更 0 / commit 0 を維持。

### 11.1 凍結契約

```text
M-03-(c) = CLOSED AS INTENTIONAL CONTRACT

Experimental Direct Head:
  taps [0, 32) are raw/direct IR taps.
  HC/LC spectral filtering is intentionally not applied.

HC:
  experimentally confirmed by T-M03-2.

LC:
  implementation path confirms exclusion;
  T-M03-2 did not resolve LC effect independently.

No production implementation change is required.
```

補足（測定由来の限定・§10.3/§10.4 と同一）:

```text
HC: direct head 対象外 — 実測確認済み（T-M03-2 tap 域 ON 不変 / OFF は HC 依存）
LC: direct head 対象外 — 現行実装経路から確定。
    本 T-M03-2 の測定値による LC 効果そのものの実測確認ではない。
H3（tap 31/32 のフィルタ状態段差）: 未測定のまま。本裁定は「段差が無い」ことを主張しない。
    （境界に同時エネルギーを持つ broadband IR での測定は将来の任意項目であり、CLOSED の条件ではない）
```

### 11.2 裁定理由

1. **HC 非適用は実測で確定**: ON tap=10 は HC Sharp/Soft・LC Natural/Soft の全条件で `0.500002 / spread=1`
   のまま不変。OFF は HC により `0.494371/5`（Sharp）〜 `0.426011/17`（Soft）へ変化（T-M03-2 §10.2）。
2. **FFT 域の既存経路は正常に HC/LC を受けている**: peak@600 は ON/OFF が全 8 対で完全一致。
   direct head を「追加実装」しても FFT 側を変更する必要が無く、変更すれば逆に既存正常経路へ
   副作用を持ち込む。
3. **direct head は明示的な experimental / opt-in 経路**（§1.4）: 「IR 全体と同一の spectral processing を
   保証する経路」ではなく **低レイテンシーの raw direct-head approximation** として仕様化する方が
   現行コード構造と整合する。
4. **RT 経路を変更する必要がない**: Practical Stable ISR Bridge Runtime の
   「RT は Read → Execute → Output」「build / validate / delete / policy decision をしない」境界を維持。
   本裁定はコード・authority をいずれも増やさない（Publish/Retire 等の authority 不変）。

### 11.3 本裁定で変更しないもの（禁止継続・§4 維持）

```text
× direct tap 数拡張
× direct FIR への HC/LC 実装
× FFT filter authority の変更
× PDC 再定義
× algorithmLatency の意味変更
× OFF 512 修正
× H-01 再変更
```

§11 は契約（ドキュメント）の裁定であり、`src/` の production コードは 1 行も変更していない。

### 11.4 D3 の分離（維持・本裁定に混ぜない）

```text
M-03-D3 / OFF-side PDC over-report candidate
  OFF:  declared algorithmLatency = 512
        measured FFT/L0 arrival delta = 0
```

- 別問題として SEPARATE CANDIDATE / HOLD を継続。**M-03 の修正対象にも本裁定にも含めない**。
- D2 の CLOSED は D3 の扱いに影響しない（D3 は PDC 申告 authority の論点であり、(c) とは独立）。

### 11.5 ゲート更新

```text
D1 = CLOSED（Case A CONFIRMED／(b) CLOSED）
D2 = CLOSED（(c) CLOSED AS INTENTIONAL CONTRACT・§11）
D3 = HOLD / SEPARATE CANDIDATE（実装なし・起票なし）
D4 = OPEN
M-03 Implementation Gate = PENDING(D4 only)
production = 0
commit = 0
```

### 11.6 次工程（未実施）

```text
M03-C2: D4 契約裁定のみ
  - experimental 継続
  - default OFF 維持
  - direct head は raw approximation
  - HC/LC 非適用を UI / 開発者向け仕様に明記
  - spectral-equivalent convolution を保証しない
  → ここまで完了で M-03 は実装なしに CLOSED 可能（H2 は仕様適合であり、
    実装バグと断定する必要がないため）
```

本 §11 の時点では D4 = OPEN（→ §12 / M03-C2 で裁定済み）。

---

## 12. D4 Contract Adjudication — Experimental Direct Head Policy（2026-09-16・M03-C2 執行）

裁定: D4 = **CLOSED**。Experimental Direct Head は experimental のまま継続し、既定 OFF を維持する。
direct taps `[0, 32)` は raw direct-head approximation であり、HC/LC spectral filtering の非適用は
**意図された仕様**として凍結する（§11 の (c) 裁定と同一の設計判断）。

### 12.1 Frozen Policy

```text
Experimental Direct Head = retained.
Default state = OFF.

When explicitly enabled:
  taps [0, 32) are processed by the raw/direct FIR head.
  HC/LC spectral filtering is intentionally not applied to these taps.
  The direct head is a low-latency raw approximation, not a
  spectral-equivalent replacement for the full convolution path.
```

### 12.2 User-Facing / Developer-Facing Meaning

```text
The experimental direct head is opt-in.
Enabling it does not imply spectral equivalence with the normal
FFT convolution path.

HC/LC filtering remains authoritative for the FFT convolution path.
The direct head intentionally bypasses that spectral-filtering stage.
```

### 12.3 Scope Boundary

```text
This decision does not modify:
  - PDC definition
  - algorithmLatency semantics
  - H-01
  - direct tap count
  - FFT filter authority
  - M-01/M-02/M-04
  - SR-01/02/03
  - Publish/Crossfade/Retire/Epoch
  - RT synchronization/atomic policy

No production implementation change is required.
```

### 12.4 D4 Result

```text
D4 = CLOSED

M-03-(a) = CLOSED
M-03-(b) = CLOSED
M-03-(c) = CLOSED AS INTENTIONAL CONTRACT

M-03 Implementation Gate = READY FOR FINAL GATE REVIEW
```

### 12.5 UI / tooltip の扱い — 本ステップでは未変更

**「HC/LC 非適用を UI に明記する」は、本 M03-C2 で UI 文言を変更する意味ではない。**
本ステップは契約文書側の明確化のみであり、`src/` の UI コードは変更していない。
UI / tooltip の実変更の要否は **Final Gate の scope review で別途判断**する。

現行 UI 文言（ソース確認済み・変更なしの記録）:

```text
src/ConvolverControlPanel.cpp:441
  experimentalDirectHeadToggle.setButtonText("Exp Direct Head");
src/ConvolverControlPanel.cpp:442
  experimentalDirectHeadToggle.setTooltip(
      "Experimental zero-latency direct head path. Rebuilds the convolver when changed.");
```

- 現行 tooltip は **HC/LC 非適用に言及していない**（「zero-latency」「Rebuilds the convolver」
  のみ）。本裁定で凍結した「raw approximation / spectral-equivalent でない」はコード上の
  コメント・本契約文書・開発者向け仕様にのみ記載され、UI には未反映である。
- したがって UI 上の現状は**契約と矛盾しないが、契約を開示していない**状態。
  開示するかは Final Gate scope review の論点として起票せず保留する（本ステップで
  tooltip 変更・文言追加・新規 UI 要素のいずれも行わない）。

### 12.6 Default OFF のソース確認（12.1 の根拠）

```text
src/ConvolverProcessor.h:82    bool experimentalDirectHeadEnabled = false;   // snapshot 既定
src/ConvolverProcessor.h:1068  bool experimentalDirectHeadEnabled = false;   // pendingOverride 既定
src/ConvolverProcessor.StateAndUI.cpp:242   v.setProperty("experimentalDirectHeadEnabled", ...)  // 保存
src/ConvolverProcessor.StateAndUI.cpp:368   if (v.hasProperty("experimentalDirectHeadEnabled")) ... // 復元
src/ConvolverControlPanel.cpp:1035  toggle → setExperimentalDirectHeadEnabled(state)  // 明示操作時のみ ON
```

- 既定値は両構造体で `false`（= OFF）。ON は UI トグル / state 復元の**明示操作時のみ**。
- state へ保存・復元されるため、ON はセッションを跨いで保持されうる（＝opt-in の永続化。
  既定値自体は OFF のまま）。
- §1.4 の「明示的 experimental / opt-in 経路」と一致。

### 12.7 D3（HOLD 維持・本裁定に混ぜない）

```text
D3 OFF-side PDC over-report
= SEPARATE CANDIDATE
= HOLD
= no implementation
= no issue creation yet
```

- T-M03-1 の観測（`OFF declared algorithmLatency = 512` / `OFF measured FFT/L0 delta = 0`）は
  **M-03-(b) の再オープン理由ではない**。(b) は Case A で CLOSED のまま（§9）。
- D4 の CLOSED は D3 の扱いに影響しない（D3 は PDC 申告 authority の論点であり、
  (c) / D4 の experimental policy とは独立）。

### 12.8 本ステップの検証（M03-C2・文書のみ変更）

```text
production src changes = 0
test harness changes   = existing +204 only
HEAD                   = 53847087
commit                 = 0
git diff --check       = clean
D1 = CLOSED
D2 = CLOSED
D3 = HOLD / SEPARATE
D4 = CLOSED
```

未実施（本ステップで意図的に行わない）: ConvoPeq.md 再生成 / full CTest / clang-tidy / cppcheck /
commit / push。次工程は **M-03 Implementation Gate / Final Gate の read-only 判定**。

---

## 13. M-03 Implementation Gate — read-only 判定（2026-09-16）

判定方式: read-only。コード変更・実装・commit を一切行わず、現行ツリーと契約文書のみで判定する。
基準ソース: `ConvoPeq.md`（Generated 2026-09-16 20:13:19）+ 現行 `src/`。

### 13.1 A. Scope integrity

| 確認項目 | 実測 | 判定 |
|---|---|---|
| production source changes | **0**（`src/` の変更は `src/tests/.../ConvolverStateRoundTripTests.cpp` のみ = test harness） | OK |
| CMake changes | **0**（`CMakeLists.txt` は git 未変更・mtime 2026-09-09） | OK |
| new test target | **0**（`add_test()` 40 件・不変。M-03 は既存 CTest target 内で計測） | OK |

非接触境界（M-03 責務外）の確認: `git diff --name-only` の tracked 変更は
`ConvoPeq.md`（生成物・既存差分）/ `README.md`（既存差分）/ `src/tests/.../ConvolverStateRoundTripTests.cpp`
の 3 件のみで、以下はいずれも**無変更**。

```text
H-01 / H-02 / M-01 / M-02 / M-04
SR-01 / SR-02 / SR-03
Publish / Crossfade / Retire / Epoch
PDC / Coordinator
RT synchronization / atomic policy
```

**注記（scope 外の既存差分・M-03 由来ではない）**: `build.bat` に **staged 済みの未 commit 差分**が存在する
（`NINJA_FLAGS` の `-j 1` → `-j 2`、icx/icpx のメモリ回避策）。これは §4 の PRESERVE 3 ファイル
（`.gitignore` / `build.bat` / `doc/work68/...`）に属する**先行作業の持ち越し**であり、
M-03 の変更でも CMake 変更でもない（CMakeLists.txt とは別物）。scope integrity の判定には影響しないが、
Final Gate の diff boundary 判定で PRESERVE として扱う必要があるため記録する。

### 13.2 B. D4 contract ↔ source 整合

| 契約項 | ソース上の根拠 | 判定 |
|---|---|---|
| `Exp Direct Head`（UI ボタン文言） | `ConvolverControlPanel.cpp:441` `setButtonText("Exp Direct Head")` | 一致 |
| tooltip（現行・未変更） | `ConvolverControlPanel.cpp:442` `"Experimental zero-latency direct head path. Rebuilds the convolver when changed."` | 一致 |
| **default OFF** | `ConvolverProcessor.h:82` / `:1068` ともに `= false` | 一致 |
| direct taps = `[0, 32)` | `MKLNonUniformConvolver.cpp:703` `constexpr int kMaxDirectTaps = 32;` ／ `:705` `m_directTapCount = (enableDirectHead ? min(irLen, min(directPart, kMaxDirectTaps)) : 0)` | 一致 |
| raw/direct IR taps | `:742` `memcpy(impulseForFft, impulse, irLen)` → `:745` `memset(impulseForFft, 0, m_directTapCount * sizeof(double))`（direct 分は FFT 入力から除外）→ `:963` `irSrc = impulseForFft + cfgs[li].offset` | 一致 |
| HC/LC spectral filtering not applied | `applySpectrumFilter` は **production 内で定義 `:336` と単一呼び出し `:1105` のみ**。`:1105` は spectrum 構築段（NonRT・FFT レイヤ用）。`processDirectBlock` 本体 `:1346-1409` に HC/LC 参照は **0 件** | 一致 |
| experimental / opt-in | `setExperimentalDirectHeadEnabled`（`:368-369`）＋ state 保存/復元（`StateAndUI.cpp:242/368`）。ON は UI トグル `:1035` の明示操作時のみ | 一致 |

補足（T-M03-2 の測定値との対応）: `:745` の zeroing により direct 域（taps 0..31）は FFT 経路から
除外されるため、peak@600 で **ON == OFF が完全一致**した §10.2 の結果は静的に裏付けられる。

**UI / tooltip は本判定で変更していない**（D4 は契約裁定の工程であり、UI 文言変更は implementation
scope ではない）。§12.5 の整理を維持する（現行 UI は契約と矛盾しないが、契約内容を完全には開示して
いない状態）。

### 13.3 C. D3 の非再オープン

```text
D3 = HOLD / SEPARATE CANDIDATE
```

- T-M03-1 の観測

```text
OFF: declared algorithmLatency = 512
     measured FFT/L0 arrival delta = 0
```

  は **D3 および M-03-(b) を再オープンする理由にしない**。
- (b) は §9 の Case A で CLOSED のまま。D3 は**別候補として保持**し、**M-03 implementation scope に
  含めない**（実装なし・起票なし）。
- D3 と D4 は独立（D3 = PDC 申告 authority の論点 / D4 = experimental policy）。D4 の CLOSED は
  D3 の扱いを変えない。

### 13.4 D. Contract boundary（誤読防止）

§11/§12 の契約を「今後の implementation requirement」と読み替えないこと。以下はいずれも
**要件ではなく禁止・非対象**。

```text
Direct FIR に HC/LC を追加しない
direct tap 数を増やさない
FFT filter authority を変更しない
PDC の定義を変更しない
algorithmLatency の意味を変更しない
H-01 dry alignment を変更しない
```

### 13.5 Gate 判定

```text
D1 = CLOSED
D2 = CLOSED
D3 = HOLD / SEPARATE CANDIDATE
D4 = CLOSED

M-03-(c) = CLOSED AS INTENTIONAL CONTRACT

M-03 Implementation Gate = PASS
Production implementation required = NO
Implementation scope = NONE
Final Gate eligibility = READY
```

**PASS の限定範囲（重要）**: 本 PASS が意味するのは **Gate の準備性と scope closure** のみである。

- D3 は **PASS に格上げしない**（HOLD / SEPARATE CANDIDATE のまま）。
- M-03 全体について「**すべての latency 問題が解決済み**」とは主張しない。本 work で確定したのは
  (a) 32-tap 境界・(b) Case A・(c) 意図的契約の 3 点であり、§10.4 に記録した未測定項
  （H3 の境界段差、実 IR での可聴有意性）は**未測定のまま**である。
- 本判定は read-only であり、production 実装 0 を前提とする。実装が 0 である以上、
  「実装後の検証」に相当するものは存在しない。

### 13.6 本判定で行っていないこと

```text
× ConvoPeq.md 再生成
× full Release CTest
× full Debug CTest
× clang-tidy
× cppcheck
× commit
× push
× UI / tooltip 変更
```

理由: M03-C2 は**契約裁定のみ**であり Production implementation が 0。実装後検証一式を先行実施する
必要がない。

### 13.7 ConvoPeq.md 基準の精度（記録上の限定）

- 基準 `ConvoPeq.md` は **Generated 2026-09-16 20:13:19**。freshness check の結果は
  `NEWER_SRC_COUNT = 1`、newer = `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp`
  （mtime 2026-09-16 21:28、M-03 計測コード・test のみ・未 commit）。
- したがって **production `src/` については 20:13:19 の snapshot が現行と一致**する（本 §13 の
  B 判定はこの基準で有効）。差分は **test harness 1 ファイルのみ**で、production コードには及ばない。
- ただし §13.2 の `:703`/`:705`/`:745`/`:963`/`:1105`/`:1346` 等の行番号は**現行ツリー**のものである。
  ConvoPeq.md 再生成後は一致するが、本判定時点では未再生成（§13.6）。
- 監査で旧 stamp を引用する場合は、production に関しては有効・test harness に関しては**未反映**と
  明示すること。

---

## 14. M-03 Final Gate（2026-09-16）

### FG-1 前提固定

```text
M-03 Implementation Gate = PASS

D1 = CLOSED
D2 = CLOSED
D3 = HOLD / SEPARATE CANDIDATE
D4 = CLOSED

M-03-(a) = CLOSED
M-03-(b) = CLOSED
M-03-(c) = CLOSED AS INTENTIONAL CONTRACT

Production implementation required = NO
Implementation scope = NONE
```

### FG-2 Commit boundary

| 区分 | 対象 | 扱い |
|---|---|---|
| M-03 documentation | `doc/work100/m03_pre_audit_failure_contract_freeze_20260916.md` | **candidate**（untracked） |
| M-03 test-only measurement | `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp` | **candidate**（+204 のみ） |
| 生成物 | `ConvoPeq.md` | **candidate**（FG-3 で FRESH 確認済み） |
| PRESERVE | `.gitignore` / `build.bat` / `doc/work68/automatic_sound_test_plan_v7.4_validation_20260911.md` | staged 状態を**変更しない** |
| **EXCLUDE** | `README.md` | **M-03 由来ではない**（下記） |

**README.md の由来判別（M-03 由来ではない）**: 差分は **E-G3-1（icx Debug 非サポート方針・2026-09-13）** に
関する記述で、(i) icx Debug/RWDI 行を「no build support / does not compile（`/QxCORE-AVX2` は Release
限定 = work92 C-8、AVX2 intrinsic が 16+ の非テストソースで unguarded）」へ変更、(ii) 実行例から
`build.bat Debug icx` をコメントアウト、(iii) 出力先表で icx Debug を「N/A — not supported」へ変更、
(iv) 方針注記（E-G3-1 option a）を追加。M-03 の調査対象（(a) 32-tap 境界 / (b) latency 申告 / (c) HC/LC）
とは無関係。**M-03 commit boundary から除外する。**

### FG-3 `ConvoPeq.md` freshness

- **再生成前**の分離確認（「最新ソースを参照した」と「commit 対象が最新を反映しているか」の分離）:

```text
baseline Generated : 2026-09-16 20:13:19
NEWER_SRC_COUNT    : 1
newer              : src\tests\AudioEngineHarness\ConvolverStateRoundTripTests.cpp  (mtime 2026-09-16 21:28:11)

→ production source newer than snapshot = 0   （唯一の差分は test harness）
→ ただし当該 1 ファイルは snapshot に未反映 = 20:13:19 版は commit 対象にできない
```

- **再生成**（`python output_sourcecode_markdown.py`、22:18:13）→ 生成後の確認:

```text
baseline Generated : 2026-09-16 22:18:13  (ConvoPeq.md)
NEWER_SRC_COUNT    : 0
STATUS             : FRESH — snapshot は現行ソースを反映しています

ConvoPeq.md freshness = FRESH
production source newer than snapshot = 0
```

- 再生成物の内容確認（すべて OK）: `kMaxDirectTaps = 32` / `m_directTapCount = (enableDirectHead ...)` /
  `memset(impulseForFft... m_directTapCount)` / `applySpectrumFilter` 定義 / `processDirectBlock` 定義 /
  `measureM03DirectHeadTiming` / `measureM03HcLcBoundary` / M-01 guard / `nonFiniteBlockCounter` /
  `setButtonText("Exp Direct Head")` / tooltip 全文。サイズ 4,752,545 bytes。
- §13.7 の「行番号は現行ツリーのもの」という限定は、再生成により**解消**（stamp 一致）。

### FG-4 文書の自己整合性（§9〜§13）

17/17 項目 OK。

```text
§9  : Case A / (b) = CLOSED / D1 = DONE
§10 : Case H2 / HC = 実測確認済み / LC = 未解像 / H3 = 未測定
§11 : (c) = CLOSED AS INTENTIONAL CONTRACT
§12 : D4 = CLOSED / experimental retained / default OFF / raw approximation / no HC/LC
§13 : Implementation Gate = PASS / scope = NONE / D3 = HOLD
```

**限定の保持を確認**: 「LC は T-M03-2 で独立実測確認していない」（`T-M03-2 did not resolve LC effect
independently`）と「H3 の tap 31/32 境界段差は未測定」はいずれも**削除されていない**。

### FG-5 UI scope

```text
UI source change = 0     （src/ConvolverControlPanel.{cpp,h} 無変更）
tooltip change   = 0
```

現行 UI は不変:

```text
Exp Direct Head
Experimental zero-latency direct head path. Rebuilds the convolver when changed.
```

`UI disclosure enhancement = separate future scope`（機構的には §12.5 の整理を維持。本 Final Gate でも
tooltip を変更していない）。

### FG-6 Full verification

**目的**: M-01 適用後の baseline に M-03-C2 の test/documentation 作業が回帰を持ち込んでいないことの確認
（M-03 自体に production 変更が無いため、新しい M-03 failure reproduction は追加しない）。

| 項目 | 結果 |
|---|---|
| Release build | `ninja -f build-Release.ninja` **exit 0**・FAILED 0 件 |
| Debug build | `ninja -f build-Debug.ninja` **exit 0**・FAILED 0 件 |
| Release CTest | **100% 40/40**（run1 run2 とも・`--output-on-failure` で失敗 0） |
| Debug CTest | **100% 40/40**（run1 run2 とも） |
| stability | Release run2 / Debug run2 とも 40/40（同一結果） |
| clang-tidy（M-03 変更ファイル） | **警告・エラー 0 件** |
| clang-tidy（M-01 baseline ファイル） | `bugprone-branch-clone` @ `MKLNonUniformConvolver.cpp:658` **1 件のみ** = M-01 記録 baseline と同一（回帰なし） |
| cppcheck（M-03 変更ファイル） | **exit 0**（指摘 0） |
| cppcheck（M-01 baseline ファイル） | exit 2・既存 style/`dangerousTypeCast` 指摘のみ（後述・M-03 非接触） |
| RT architectural scan | `.github/scripts/check-src-atomic-dotcall.ps1` **PASS**（exit 0） |
| T-M03-1/T-M03-2 再計測（再ビルド後 harness） | Release/Debug 各 **exit 0**・計測値は再ビルド前と**完全一致**（各 25 行・計 50 行が同一） |

`RuntimeHealthMonitorTierTests` を含む 40 テストすべてが Release/Debug 各 2 回 Passed。harness 内の
`checkM01DenormalHygiene` / `checkM02NonFiniteTelemetry` / `checkM04OversizedContainment` も PASS
（= T-M03-3 非退行）。

#### FG-6 補記 1: cppcheck の M-01 baseline ファイル指摘（M-03 非回帰）

`src/MKLNonUniformConvolver.cpp` は **M-03 で 1 行も変更していない**（production diff = 0）ため、
ここで出る指摘は M-03 の回帰では**あり得ない**。実際の指摘は
`dangerousTypeCast`（`:1480`/`:1481`/`:1692`/`:1693` の `_mm_prefetch((const char*)(...))` = AVX2 プリ
フェッチの既存 C キャスト）ほか `functionStatic` / `cstyleCast` / `constVariableReference` /
`useStlAlgorithm` / `truncLongCastAssignment` で、いずれも M-01 以前からの既存コードに属する。
`--error-exitcode=2` は style 指摘でも exit 2 を返すため、exit 2 自体は「M-03 の失敗」ではない。

（M-01 の記録では cppcheck 対象が `MKLNonUniformConvolver.h` ヘッダ中心だったため、これら .cpp レベルの
style 指摘は当時のログに現れていない。対象範囲の差であって状態の差ではない。）

#### FG-6 補記 2: ビルド環境依存（**M-03 スコープ外・既存の脆弱性**）

本 Final Gate のビルド検証にあたり、**M-03 とは無関係な既存のビルド fragility** が2件顕在化した。
M-03 は `CMakeLists.txt` を変更していない（CMake diff = 0）ため、いずれも M-03 起因ではない。

**(1) oneAPI 環境が未適用だと MKL include が解決しない**

- `src/DiagnosticsConfig.h:48` は `#if defined(JUCE_DSP_USE_INTEL_MKL)` — **値ではなく defined 判定**。
  一方 `JUCE/modules/juce_dsp/juce_dsp.h:156-157` が
  `#ifndef JUCE_DSP_USE_INTEL_MKL` → `#define JUCE_DSP_USE_INTEL_MKL 0` と**無条件に定義**する。
  したがって `juce_dsp.h` を先に取り込む TU では本 `#if` が**常に真**となり `#include <mkl.h>` が走る。
- `RuntimeHealthMonitorTierTests`（`CMakeLists.txt:348-368`）は他の兄弟ターゲットと異なり
  `target_compile_definitions(... JUCE_DSP_USE_INTEL_MKL=1)` / `$ENV{MKLROOT}/include` /
  `target_link_libraries(... MKL::MKL)` のいずれも持たない（例: `TerminalTelemetryContractTests:341-342`、
  `RetrySchedulerTests:522`、`PublicationAdmissionTests:166-169` との差）。
- 結果として `mkl.h` は `%INCLUDE%` 経由でしか解決できず、oneAPI `setvars.bat` 未適用の環境では
  `fatal error C1083: 'mkl.h'` で失敗する。

**(2) clean ビルドでは `JuceHeader.h` の生成順序が保証されない**

- `ConvoPeq_artefacts\JuceLibraryCode\JuceHeader.h` は juceaide の CUSTOM_COMMAND で生成され、
  `cmake_object_order_depends_target_ConvoPeq_RELEASE: phony || ConvoPeq_artefacts\JuceLibraryCode\JuceHeader.h`
  として `ConvoPeq` にのみ順序依存が付く。
- `RuntimeHealthMonitorTierTests` の order-only 依存は **`|| .`（空）** で `JuceHeader.h` を含まない
  （`CMakeLists.txt:345-347` のコメントが「intentionally NO add_dependencies」と明記）。このため clean
  ビルドでは同ターゲットが `JuceHeader.h` 生成前にコンパイルされ
  `fatal error C1083: 'JuceHeader.h'` で失敗する。**incremental ビルドでは既存ファイルがあるため成功する。**

**(3) build identity gate（`src/tools/build_identity_gate.py`）**

- `build.bat` は configure 後の stamp と現行環境の `compiler_path` を突合し、不一致なら fail-closed する
  （COHERENCE-4）。stamp は絶対パス
  `.../MSVC/14.51.36231/bin/Hostx64/x64/cl.exe` を記録する一方、`build.bat` が渡す
  `-DCMAKE_CXX_COMPILER=cl` により CMakeCache 側は `cl` になる。
- ゲートが示す回復手順は `build.bat Release clean`（明示 clean）。ただし clean すると (2) により
  `JuceHeader.h` が再び失われ、再び失敗する。
- **本 Final Gate での実際の手順**: oneAPI 環境変数（`MKLROOT`/`IPPROOT`/`INCLUDE`/`LIB`）を設定 →
  `vcvarsall.bat x64` → `ConvoPeq` ターゲットを先にビルドして `JuceHeader.h` を生成 → 以降は
  **`ninja -f build-Release.ninja` / `ninja -f build-Debug.ninja` を直接実行**（`cmake --build` は
  configure を再実行して stamp を乱すため使用しない）。この手順で Release/Debug とも exit 0・FAILED 0。

**位置付け**: (1)(2) はリポジトリ側の既存結合（CMakeLists のターゲット定義差と、意図的に張られていない
依存）に起因し、(3) はそれを踏まえたビルド手順の制約である。いずれも **M-03 の修正対象ではなく、
本 work で修正しない**。別 work 候補として記録のみ行う（起票・実装はしない）。

### FG-7 Diff boundary

```text
M-03 candidate:
  doc/work100/m03_pre_audit_failure_contract_freeze_20260916.md   (untracked)
  src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp   (+204 / 1 file changed)
  ConvoPeq.md                                                     (regenerated 22:18:13 / FRESH)

PRESERVE:
  .gitignore
  build.bat
  doc/work68/automatic_sound_test_plan_v7.4_validation_20260911.md

EXCLUDE:
  README.md                                     (E-G3-1 icx 方針 / M-03 由来ではない)
  evidence/**, その他既存差分                 (M-03 由来ではない)
```

機械的確認:

```text
M-03 source production diff = 0   （src/convolver, src/MKLNonUniformConvolver.cpp,
                                   src/ConvolverProcessor.h, src/ConvolverControlPanel.cpp,
                                   src/audioengine いずれも無変更）
M-03 CMake diff             = 0   （CMakeLists.txt 無変更・add_test() 40 件で不変）
M-03 new test target        = 0
git diff --check            = clean
HEAD                        = 53847087
commit                      = 0
```

### 最終判定

```text
M-03 Final Gate = PASS

D1 = CLOSED
D2 = CLOSED
D3 = HOLD / SEPARATE CANDIDATE
D4 = CLOSED

M-03-(a) = CLOSED
M-03-(b) = CLOSED
M-03-(c) = CLOSED AS INTENTIONAL CONTRACT

Production changes = 0
Implementation scope = NONE

UI/tooltip changes = 0
D3 implementation = 0

ConvoPeq.md = FRESH
Release CTest = PASS
Debug CTest = PASS
Static analysis = PASS
Diff boundary = PASS

Commit eligibility = READY
Push eligibility = READY
```

**限定（誤読防止）**:

- 本 PASS は **Gate の準備性・scope closure・回帰なしの確認**を意味する。D3 は **HOLD / SEPARATE
  CANDIDATE のまま**（PASS に格上げしない）。
- §10.4 の未測定項（**H3 の tap 31/32 境界段差**、実 IR での可聴有意性）は**未測定のまま**。
- 「Static analysis = PASS」は、**M-03 変更ファイルに新規指摘が無く、M-01 baseline ファイルの指摘が
  記録済み baseline と同一**であることを指す（FG-6 補記 1 参照）。cppcheck の既存 style 指摘を
  「解消済み」とは主張しない。
- ビルドは FG-6 補記 2 の手順（oneAPI 環境 + `JuceHeader.h` 事前生成 + `ninja` 直接実行）で実施した。
  この環境依存性は **M-03 スコープ外の既存事項**であり、本 PASS はそれを解消したことを意味しない。

### 本ステップで行っていないこと

```text
× commit
× push
```

次工程: **M-03 Commit Gate → commit → push → post-push integrity → CLOSED**。
