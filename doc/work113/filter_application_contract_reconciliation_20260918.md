# WORK113-9 — Filter Application Contract / Canonical Path Reconciliation Audit

- **作成日**: 2026-09-18
- **種別**: Read-only / contract audit（production 変更 0 / 実装 0 / test-only 変更 0 / audit artifact のみ）
- **前工程**: `doc/work113/filter_application_architecture_audit_20260918.md`（113-8）
- **保留**: 113-7 block sweep / WORK114（resample 比 7.6241）

---

## A. Source baseline

| 項目 | 値 |
|---|---|
| `ConvoPeq.md` | 5,276,742 bytes / 2026-09-18 22:33:38 |
| HEAD | `0654e7b5` |
| production 変更 | **0** |
| test-only 変更 | **0** |
| audit artifact | 本ファイルのみ |

---

## B. FilterSpec semantic ownership（113-8 の訂正を含む）

113-8 では「同じ atomic が 2 経路へ」と記述したが、正確には **2 段の状態オブジェクト**を経由する。
ただし **同期は AudioEngine の setter が行う**（fan-out）。

```text
UI: ConvolverControlPanel (590/605/620: setConvHCFilterMode, 645/660: setConvLCFilterMode)
StateIO: "convHCFilterMode" / "convLCFilterMode" / "eqLPFFilterMode"
        ↓
AudioEngine::setConvHCFilterMode(mode)   [AudioEngine.Parameters.cpp:665-679]
   ├─ (1) publishAtomic(convHCFilterMode, mode)            ← ① OutputFilter 用
   └─ (2) uiConvolverProcessor.setNUCFilterModes(
              consumeAtomic(convHCFilterMode),
              consumeAtomic(convLCFilterMode))              [同 :671-673]   ← ② NUC 用
AudioEngine::setConvLCFilterMode(mode)  [同 :681-696] （同じ fan-out: :687-689）
        ↓
ConvolverProcessor::setNUCFilterModes()  [StateAndUI.cpp:864-887]
   → pendingOverride.nucHCMode / nucLCMode （別状態・別 property "nucHCMode"/"nucLCMode"）
   → postCoalescedChangeNotification() → rebuild 経路
        ↓
buildSnapshot.nucHCMode / nucLCMode
   → FilterSpec.hcMode / lcMode  [LoaderThread.cpp:213-214, LoadPipeline.cpp:658-659, Lifecycle.cpp:303-304]
   → MKLNonUniformConvolver::applySpectrumFilter()   ★ IR の周波数領域
```

- **確認 1**: `FilterSpec.hcMode/lcMode` の正式な意味 = 「ConvolverProcessor が持つ NUC 用 HC/LC モード」
  （`nucHCMode/nucLCMode`。property 名も `nucHCMode`/`nucLCMode` で `convHCFilterMode` とは別）。
- **確認 2**: UI の mode 名（Sharp/Natural/Soft、LC は Natural/Soft）と enum は一致（`ConvolverControlPanel` が engine setter を呼ぶ）。
- **確認 3**: StateIO の保存値は **2 系統**（`convHCFilterMode`/`convLCFilterMode` と `nucHCMode`/`nucLCMode`）。
- **確認 4**: `nucHCMode/nucLCMode` = NUC の IR 周波数領域フィルタ用モード。rebuild の structural hash に含まれる。
- **確認 5**: `state.convHCMode/convLCMode`（OutputFilter 用）と `nucHCMode/nucLCMode`（NUC 用）は
  **「同じ atomic から来る」のではなく、AudioEngine の setter が両方へ配る**。
  **ユーザー操作 1 回で両経路が変化する**（semantic は同一の意図を二重に保持）。
- **確認 6**: コメント・enum・UI 表示に**食い違いは検出されなかった**（enum は共有 `HCMode`/`LCMode`）。
  ただし **`FilterSpec` 側コメントは「NUC が SoA に周波数ゲインを直接適用（Audio Thread 追加コストゼロ）」と
  最適化として記述**しており、`OutputFilter` 側コメントは「convIsLast=true = ① コンボルバー最終段。
  LC→HC0→HC1」と記述。**どちらが canonical かを明示した契約文は見つからない**（§J Level C 未確認）。

---

## C. NUC transfer function（`applySpectrumFilter`）

`MKLNonUniformConvolver.cpp:361-468`。ゲイン配列（実数、位相なし）を `irFreqReal/Imag` へ乗算。

| 項目 | NUC `applySpectrumFilter` |
|---|---|
| LC topology | ゲイン配列の**低域側を 0/raised-cosine**（biquad ではない） |
| LC cutoff | `lcFcEnd=8 / lcFcStart=18`（Natural）, `6/15`（Soft） |
| LC order | 実装は **bin 単位のゲイン**（フィルタ次数の概念なし） |
| LC 実効 | **N=4096, fs=384000 では `kEnd=round(8·4096/384000)=0`, `kStart=round(18·4096/384000)=0`** → `k<=kEnd` で gain=0、`k<kStart` は発生せず → **DC bin のみ 0**（18 Hz HPF ではない） |
| HC topology | **raised-cosine ゲイン**（`0.5(1+cos(πx))`）、位相 0 |
| HC cutoff | `kStart=round(22000·N/fs)`, `kEnd=min(N/2, round(nyquist·N/fs))` = **22 kHz → Nyquist の遷移** |
| HC order | 次数なし（振幅のみの線形位相テーパ） |
| phase | **ゼロ位相**（real gain を re/im 双方に乗算） |
| Nyquist behavior | **gain = 0**（`0.5(1+cos π) = 0`） |
| DC behavior | **gain = 0**（LC の `k<=kEnd`） |
| linear / time invariant | 周波数領域で**静止**（IR に焼き込み）。ただし後段 OLS と組み合わせると kernel 長が P を超え block-rate 歪み（7F-0） |
| block dependence | ゲイン自体は block 非依存。**結果は OLS 境界で block 依存** |

**「LC = DC bin only zero」は全 mode で成立するか** → **実用的 fs では成立**：
- Natural（8/18 Hz）: kEnd=kStart=0 → DC のみ。
- Soft（6/15 Hz）: kEnd=round(6·4096/384000)=0, kStart=round(15·4096/384000)=0 → DC のみ。
- fs ≤ 48 kHz でも partSize が小さいため bin 幅が広く、同様に bin 0 に落ちる（例: N=1024, fs=48000 → bin=46.9 Hz → 8 Hz, 18 Hz とも bin 0）。
→ **NUC の LC は「18 Hz HPF」ではなく事実上「DC 除去」**。これは OutputFilter の LC（2次 Butterworth HPF 18 Hz）と**別の演算**。

---

## D. OutputFilter transfer function（`process(convIsLast=true)`）

`OutputFilter.cpp:199-...`、係数は `prepare()`（`:78-122`）で生成。

| 項目 | OutputFilter `convIsLast=true` |
|---|---|
| LC topology | **2次 biquad HPF**（`makeHPF`） |
| LC cutoff | Natural=18 Hz(Q=0.70711) / Soft=15 Hz(Q=0.5) |
| LC order | **2次** |
| HC topology | **2段 biquad LPF カスケード**（HC0→HC1） |
| HC cutoff | `fc_hc = (fs<=48k) ? 19000 : 22000` |
| HC order | Sharp=Butterworth4次(Q=0.5412,1.30656) / **Natural=LR4(Q=0.70711×2)** / Soft=2次(Q=0.5)+identity |
| phase | biquad の位相（最小位相、IIR） |
| Nyquist behavior | LPF のロールオフ（LR4 → 高域減衰） |
| DC behavior | HPF により DC 除去（18 Hz の −3 dB 付近から減衰） |
| linear / time invariant | LTI（状態を持つ IIR） |
| block dependence | 無（サンプル連続、状態は チャンネル毎に保持） |

※ `convIsLast=false` 分岐は **HPF(20 Hz 固定) → LP0 → LP1（`fc_lp = (fs<=48k)?19000:24000`）** で、
**LC/HC は適用しない**（`lpMode`/`hpfCoeff` のみ）。

---

## E. `convIsLast` truth table

`convIsLast = convActive && (!eqActive || order == ProcessingOrder::EQThenConvolver)`（DSPCoreDouble:458-459 / DSPCoreFloat:358-359）

| convActive | eqActive | order | convIsLast | OutputFilter が適用するもの |
|---|---|---|---|---|
| 0 | 0 | — | `process()` 自体が呼ばれない（`:456` のガード） | なし |
| 1 | 0 | — | **true** | **LC → HC0 → HC1** |
| 0 | 1 | — | **false** | HPF20 → LP0 → LP1 |
| 1 | 1 | ConvolverThenEQ | **false** | HPF20 → LP0 → LP1（**LC/HC は NUC 側のみ**） |
| 1 | 1 | EQThenConvolver | **true** | **LC → HC0 → HC1**（+ NUC 側にも同設定 → **二重**） |

### 実直列順序（order = ConvolverThenEQ、既定プリセット）

```text
入力 → [NUC: IR に LC/HC 焼き込み済み] → conv 出力
      → [EQ]
      → OutputFilter(convIsLast=false): HPF20 → LP0 → LP1
      → outputMakeupGain → 出力
```

### 実直列順序（order = EQThenConvolver）

```text
入力 → [EQ]
      → [NUC: IR に LC/HC 焼き込み済み] → conv 出力
      → OutputFilter(convIsLast=true): LC → HC0 → HC1   ← 同じ LC/HC が 2 回
      → outputMakeupGain → 出力
```

**ユーザー期待の観点**：UI の「Convolver HC/LC」は 1 つの操作で両方を動かすため、
ユーザーは「1 つのフィルタ」を期待している。`EQThenConvolver` では **同一意図の 2 つの非等価なフィルタが直列**、
`ConvolverThenEQ` では **NUC 側のみ**が働く（OutputFilter は EQ 用の別フィルタ）。**この非対称は契約として明示されていない。**

---

## F. bypass semantics

| 対象 | 状態 | 挙動 |
|---|---|---|
| NUC `applySpectrumFilter` | `filterSpec == nullptr` | **呼ばれない**（IR フィルタなし）。ただし同時に `tailMode=1`（デフォルト）・`tailEnabled=true` になる（113-7C で確認）→ **`nullptr` ≠ 「フィルタだけ無効」** |
| NUC | `HCMode/LCMode` に Disabled 値 | **存在しない**（Sharp/Natural/Soft のみ。LC は Natural/Soft） |
| OutputFilter | 同上 | **存在しない**。`process()` が呼ばれる限り HC/LC（または EQ 用 HPF/LP）は常に適用 |
| 全体 | `convBypassed` | `convActive=false` → OutputFilter 分岐が ② に切替（フィルタは残る） |
| 全体 | `eqBypassed` | `eqActive=false` → `convIsLast=true` になり OutputFilter が ① を使用 |

→ **「HC/LC を無効化する」正規の手段は存在しない**。`nullptr` は NUC 側だけを止めるが副作用を持つ。

---

## G. mode transition / reset

```text
UI mode change → setConvHC/LCFilterMode
   ├─ OutputFilter:  snapshot を毎 block 取得 → 次 block から新係数。biquad 状態は保持（reset なし）
   └─ NUC:           setNUCFilterModes → postCoalescedChangeNotification → rebuild（crossfade）
```

| 項目 | 結果 |
|---|---|
| `OutputFilter::reset()` 呼び出し | **`AudioEngine.Processing.DSPCoreLifecycle.cpp:390` のみ**（prepare系） |
| prepare 時 | reset される（`prepare()` 末尾 `:121` も reset を呼ぶ） |
| mode change 時 | **reset されない**（係数のみ差替え、状態継続）→ 過渡（クリック）の可能性 |
| crossfade/rebuild による暗黙 reset | NUC 側は rebuild（crossfade）で IR 差替え。OutputFilter は reset されない |

→ **2 経路で更新セマンティクスが異なる**（OutputFilter=サンプル連続・即時、NUC=rebuild・crossfade）。

---

## H. stereo / state lifetime

### OutputFilter
- `lcState[2]`, `hcState[2][2]`, `lpState[2][2]`, `hpfState[2]` → **ch0/ch1 完全独立**（`:236-241` で packing、`:296-...` で個別 store）。
- `process` は `chCount = min(channels, 2)`。mono 時は L のみ（R 状態は保持）。
- `reset()` は 2ch 分を対称にゼロ（`:127-...`）。
- 順序：`prepare()`（係数生成→reset）→ 各 `process()`。

### NUC
- `nucConvolvers[0]` / `[1]` の **2 インスタンス**。IR/FDL/FilterSpec を各々保持。
- `SetImpulse` は両者に**同一 filterSpec** で呼ばれる（`ConvolverProcessor.h:856/861`）。
- `StereoConvolver::reset()` が両者を reset。

→ **stereo 実装が sideband の原因という仮説は閉じられる**（状態は独立・対称）。

---

## I. R0 / R1 / R2 正式定義

```text
R0 = x * h_raw                                  （raw convolution, フィルタなし）
R1 = x * (h_raw · H_intended)                    （意図した filtered LTI）
R2 = PartitionOLS(x, h_raw · H_partition)        （現行 partition-OLS, = NUC）
```

- `R2 == NUC`（corr 0.9999999999985, lag −2048）
- `R1 != R2`（corr 0.883, 振幅比 0.5738）／`R1` は clean（137.5 Hz = 1.22e-6）
- `R1 ≈ intended OutputFilter response`（H_total(50 Hz)=0.9917 → intended 0.2479 ≈ R1 0.2509）
- **表現**：R2 は「数学的に誤り」ではなく、**現行の partitioned-convolution operator が R1 と異なる**。

---

## J. canonical evidence Level A / B / C

| Level | 内容 | 判定 |
|---|---|---|
| **A — Code evidence** | `OutputFilter::process(convIsLast=true)` が **存在し実配線されている**（AudioEngine メンバ + DSPCore から呼出） | **CONFIRMED** |
| **B — Semantic evidence** | HC/LC mode が「① コンボルバー最終段」フィルタとして定義（コメント＋`convIsLast` 分岐） | **CONFIRMED** |
| **C — Contract evidence** | 「HC/LC は OutputFilter で一度だけ適用する」と明示した契約文 | **NOT FOUND**（`FilterSpec` 側は最適化として記述。どちらが権威かの明文なし） |

→ canonical は **`OutputFilter` を canonical candidate とする**までに留める（Level C 未確認）。

---

## K. 修正案 A/B/C — 契約適合性比較（winner は決めない）

| 軸 | A: IR 全体を時間領域フィルタ→partition | B: partition spectral filtering + FFT/OLS 再設計 | C: convolution 後に OutputFilter |
|---|---|---|---|
| 1. intended transfer function | ○（設計どおり 1 回） | ○（条件充足時） | ○（既存 OutputFilter と一致） |
| 2. OLS validity | ○（partition 化前の IR 長で決まる） | △（`L_p ≤ P` を保証する設計が必要） | ○（フィルタは OLS 外） |
| 3. latency | △（フィルタ分の群遅延が IR 先頭に） | ○（変更小） | ○（既存 latency） |
| 4. CPU | △（IR 前処理が重く、numPartsIR 増） | ○ | ○ |
| 5. memory | △（フィルタ後 IR 長に比例） | ×（FFT 長拡大で増） | ○ |
| 6. existing API compatibility | ○（FilterSpec を前処理へ） | ○（NUC 内部） | ○（既存 OutputFilter をそのまま） |
| 7. double application risk | 低（NUC から撤去すれば） | 低 | **要解消**（現行は二重） |
| 8. mode transition | rebuild 必要 | rebuild 必要 | **サンプル連続（reset なしの過渡に注意）** |
| 9. stereo | NUC が ch 毎に実施 | 同 | OutputFilter が ch 毎（既存） |
| 10. tail path (L1/L2) | フィルタ後 IR を partition 化するため一貫 | 同 | L1/L2 にも 1 回（tail に別処理がある点に注意） |
| 11. EQ ordering | 既存 order と独立 | 同 | **`convIsLast` 依存**（EQ 最終段では ① が使われない問題） |
| 12. FilterSpec semantic compatibility | `FilterSpec` を「IR 前処理」へ再定義 | 現状のまま | `FilterSpec` の HC/LC を撤去し OutputFilter に一本化 |

**B の数学条件（再掲）**：
```text
L_p = partition にフィルタを適用した kernel 長 ≤ P + L_f − 1
2P-OLS valid-half が厳密: L_p ≤ P ⇔ L_f ≤ 1
FFT 長拡大: N ≥ 2·max(P, P + L_f − 1) = 2(P + L_f − 1)
ただし h に減衰しない DC 起因 floor がある場合、有限 N では解消しない。
→ 「4P で必ず解決」とは言えない。
```

---

## L. Implementation Gate

| Gate | 内容 | 判定 |
|---|---|---|
| 1 | FilterSpec semantic = confirmed | **PARTIAL**（NUC 用モードと判明。ただし canonical の明文契約なし） |
| 2 | HC/LC canonical application point = confirmed | **PARTIAL**（Level A/B ○, Level C ×） |
| 3 | `applySpectrumFilter` removal safety = confirmed | **NOT CONFIRMED**（下記） |
| 4 | convIsLast routing = fully understood | **CONFIRMED**（truth table 完成） |
| 5 | bypass / mode transition = understood | **PARTIAL**（bypass 手段が存在しないこと・reset が prepare のみは確定。過渡の実測は未実施） |
| 6 | R1 / R2 mathematical distinction = documented | **CONFIRMED** |
| 7 | implementation candidate(s) = contract-compatible | **PARTIAL**（A/C は有望、B は条件付き） |

### Gate 3 が NOT CONFIRMED である理由（重要）

`applySpectrumFilter` を単純に撤去すると、
**`convIsLast == false`（既定プリセットの `ConvolverThenEQ` + EQ 有効）では HC/LC がどこでも適用されなくなる**
（OutputFilter の ② 分岐は HPF20 + LP であり LC/HC ではない）。
→ 撤去は「HC/LC の消失」を招くため、**OutputFilter の適用条件/順序の再設計と同時でなければ安全でない**。

---

## M. Final verdict

```text
CONTRACT_PARTIALLY_CONFIRMED
```

**確定したこと**
1. `FilterSpec.hcMode/lcMode` は **ConvolverProcessor の `nucHCMode/nucLCMode`**（`convHCFilterMode` とは別状態）であり、
   AudioEngine の setter が **両経路へ fan-out** する（ユーザー操作 1 回で NUC と OutputFilter が同時に変化）。
2. NUC の LC は**全 mode で実質「DC bin のみ 0」**であり、OutputFilter の 2次 HPF 18 Hz とは**別演算**。
   NUC の HC は raised-cosine 振幅テーパ（Nyquist で 0）で、LR4 LPF とは**別演算**。
3. `convIsLast` の truth table は確定。**`EQThenConvolver` では同一意図の 2 フィルタが直列**、
   `ConvolverThenEQ` では **NUC 側のみ**。この非対称は契約に明示されていない。
4. `OutputFilter::reset()` は **prepare 系のみ**。mode change では reset されない（過渡リスク）。
5. stereo 状態は両経路とも独立・対称（**stereo 起因説は棄却**）。
6. `R2 == NUC`、`R1 != R2`、`R1 ≈ intended`。R2 は「誤り」ではなく**別 operator**。

**未確定（実装を禁止する理由）**
- Level C（「HC/LC は OutputFilter で 1 回」という明文契約）が存在しない。
- **Gate 3**：`applySpectrumFilter` の撤去は、`convIsLast=false` 構成で HC/LC を消失させるため**単独では安全でない**。
- bypass の実測（過渡）と mode 変更時の挙動の実測が未実施。

**次の作業（順序どおり）**
`113-10 Implementation Design`（実装はまだしない）。その中で
(a) HC/LC の権威をどちらに置くかの契約を明文化し、
(b) `convIsLast` の両分岐で HC/LC が必ず 1 回だけ適用される構成（例: OutputFilter を常時 ① 相当にし、
    EQ 最終段でも LC/HC を通す。または A 案で IR へ 1 回だけ焼き込む）を設計する。
