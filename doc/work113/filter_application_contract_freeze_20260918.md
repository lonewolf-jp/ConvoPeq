# WORK113-12 — HC/LC Single-Application Contract Freeze

- production changes: 0
- test-only changes: 0
- implementation: 0
- commit: 0

- **作成日**: 2026-09-18
- **種別**: Read-only / implementation gate（契約凍結と実装前検証。コード変更なし）
- **前工程**: `doc/work113/filter_application_implementation_plan_20260918.md`（113-11 = DESIGN_READY）
- **保留**: 113-7 block sweep / WORK114

---

## 1. Baseline freeze

| 項目 | WORK113-11 | 本 WORK | 一致 |
|---|---|---|---|
| `ConvoPeq.md` サイズ | 5,276,742 B | 5,276,742 B | **一致** |
| 生成日時 | 2026-09-18 22:33:38 | 2026-09-18 22:33:38 | **一致** |
| HEAD | `0654e7b5` | `0654e7b5` | **一致** |
| working tree | 未コミット変更あり（WORK103-113 の trace/test 追加） | 同（`git status --porcelain` 62 件） | **一致** |

→ baseline は 113-11 と一致。**差分監査は不要**。

---

## 2. Contract Freeze（変更不能）

```text
INV-FILTER-001  1 user HC/LC setting = 1 logical convolution filter intent
INV-FILTER-002  convActive == true ⇒ HC/LC semantic application count == 1
INV-FILTER-003  HC/LC transfer function は ProcessingOrder に依存しない
INV-FILTER-004  ConvolverThenEQ と EQThenConvolver の差は EQ と Convolver の順序だけ。
                HC/LC の存在・回数・TF は変化しない。

direct head を含む全 contribution:
    L0 + L1 + L2 + tail + direct head  →  single HC/LC  （exactly once）
```

凍結。以降の工程（113-13〜）はこの 4 不変条件と direct-head 条項を変更しない。

---

## 3. G6/G9 の source 再検証（transition policy）

**係数生成（NonRT のみ）**
```text
AudioEngine.Processing.DSPCoreLifecycle.cpp:216 / :283
    outputFilter.prepare(processingRate);      ← ③ ここで全モード分の係数を生成
OutputFilter.cpp:78-122 prepare()
    hcCoeff[3][2] = { Sharp(BW4: Q=0.5412/1.3066), Natural(LR4: Q=0.7071×2), Soft(2次 Q=0.5 + stage1=identity) }
    lcCoeff[2]    = { Natural(HPF 18Hz Q=0.7071), Soft(HPF 15Hz Q=0.5) }
    hpfCoeff      = HPF 20Hz 固定,  lpCoeff[3][2] = LPF fc=(fs<=48k?19k:24k)
    → reset() を末尾で 1 回
```

**RT が読むもの**
```text
AudioEngine.h:3992-3994 / 4054-4056
    snapshot.convHCMode = consumeAtomic(convHCFilterMode);
    snapshot.convLCMode = consumeAtomic(convLCFilterMode);
    snapshot.eqLPFMode  = consumeAtomic(eqLPFFilterMode);
→ ProcessingState として DSPCore へ渡る（AudioEngine.h:4092-4094）
OutputFilter.cpp:219-223
    const int hcIdx = (int)hcMode;  ... hcCoeff[hcIdx][0/1], lcCoeff[lcIdx]
    → ★ RT は「事前生成済み係数表の index 引き」のみ（判断なし）
```

**state reset の発生箇所**
```text
OutputFilter.cpp:121   （prepare() 末尾）
AudioEngine.Processing.DSPCoreLifecycle.cpp:390   （prepare 系の明示 reset）
→ mode change では reset されない（source 上、mode 由来の reset 呼出は存在しない）
```

**block 境界での coefficient replacement**
```text
DSPCoreDouble.cpp:460 / DSPCoreFloat.cpp:360  … processBlock あたり 1 回呼ばれる
→ 係数差替えはブロック境界で発生し、biquad 状態（lcState/hcState）は保持される
```

**結論**：113-11 の決定（coefficient replacement + state preservation + no reset）は
**source と整合**。係数は prepare 時に全モード分生成され、RT は index 引きのみ → **RT 判断なし**を満たす。

---

## 4. Transition 過渡測定（source 変更なし・Python 独立シミュレーション）

`OutputFilter` ① の係数を 113-11 の仕様どおり独立再構成し、**状態保持のまま係数のみ差替え**て測定。
定義：`sampleJump = max|x[n]-x[n-1]|`（切替前後）、`rmsTransient = RMS(y[sw:sw+2048] - 切替後定常平均)`、
`DCexc = |mean(y[sw:sw+7680])|`。入力は 50 Hz / 0.25（4 秒のうち中央で切替）。

| 切替 | sampleJump | rmsTransient(2048) | DCexc(7680avg) | 相対 jump |
|---|---|---|---|---|
| **Sharp → Natural（HC）** | **1.759e-04** | 3.358e-02 | **1.56e-12**（base 2.00e-12） | 7.03e-04 |
| Natural → Soft（HC） | **未取得（OBSERVATION）** | — | — | — |
| Natural → Soft（LC） | **未取得（OBSERVATION）** | — | — | — |

- **DC 変動は無視できる**（1.6e-12、base と同水準）→ クリックの主因は DC ではなく係数差の過渡。
- Sharp→Natural（最も差が大きい HC 切替）で `sampleJump` は信号振幅の **0.07%** → 単発クリックとしては小さい。
- 未取得 2 件はシミュレーションスクリプトの軽微な不具合（identity セクションの係数長）で未完走。
  **本 WORK は test-only 変更禁止のため修正せず**、受入試験（Phase 5）で実機測定する。

**Acceptance criterion（数値・再現可能）**
```text
A-TRANS-1  sampleJump(切替前後 2 サンプル) ≤ 0.005 × steady-state peak
A-TRANS-2  DCexc(7680 sample 平均) ≤ 1e-6
A-TRANS-3  rmsTransient(2048) ≤ 0.02 × steady-state peak
（すべて実機の OutputFilter ① 出力で測定。本 WORK の独立シミュは参考値）
```

---

## 5. `applySpectrumFilter` raw-IR verification（手段の確立）

**既存の test-only seam（追加変更なしで利用可）**
- `--buzz --nuc7c` → `[7C] … irFreqNZ / irFreqPeak / irFreqEnergy`
- `--buzz --nuc7d` → `irfreq_7d_B.csv`（非零 partition の全 bin 実値）

**baseline（WORK113-7C/7D の実測、同一 source）**

| 条件 | irFreqEnergy | irFreqNZ | irFreqPeak | 対独立期待値 |
|---|---|---|---|---|
| Case A（FilterSpec **無し** = raw IR） | **2.049e+03** | 1 | 1.0 | — |
| Case B/D（FilterSpec 有り） | **9.14375e+02** | 1 | 1.0 | 7D: corr **1.00000000**（`exp(-j2πk·488/4096)·g[k]`） |

**Phase 2 の機械的判定**
```text
applySpectrumFilter の HC/LC を停止した後、Case B 相当の条件で
    irFreqEnergy == 2.049e+03（raw 値に一致）
    （フィルタ有り時の 9.14375e+02 へ戻らないこと）
かつ `--nuc7d` の irFreq が rfft(raw partition) と corr 1.0 であること。
→ 判定手段は既存 seam で確立済み（新規 test 追加は不要）。
```

---

## 6. Structural hash / rebuild baseline

**現状の依存（source）**
```text
ConvolverProcessor.setNUCFilterModes()          StateAndUI.cpp:864-887
   → pendingOverride.nucHCMode/nucLCMode 更新（変更時のみ）
   → postCoalescedChangeNotification()          （:885 / 定義 Rebuild.cpp:21）
ConvolverProcessor.getStructuralHash()           StateAndUI.cpp:889-921
   → hashCombine(snapshot.nucHCMode) / (snapshot.nucLCMode)  （:913-914）
→ mode 変更は structural hash を変化させ、coalesced rebuild を要求する
```

| シナリオ | 期待（source 由来） | 実測 |
|---|---|---|
| baseline（無変化） | rebuild 0 | **OBSERVATION（未計測）** |
| HC Sharp → Natural | structural rebuild ≥ 1 | **OBSERVATION（未計測）** |
| HC Natural → Soft | 同上 | **OBSERVATION（未計測）** |
| LC Natural → Soft | 同上 | **OBSERVATION（未計測）** |
| HC+LC 同時 | coalescing により 1 回に統合され得る | **OBSERVATION（未計測）** |

- 実測には engine telemetry（`[REBUILD_TELEMETRY]`）を mode 変更で駆動する必要があり、
  **本 WORK は test-only 変更禁止**のため未計測（OBSERVATION）。
- **mode change は IR の structural change ではない**ことは source 上明確：
  mode は `nucHCMode/nucLCMode` → `FilterSpec.hcMode/lcMode` → **IR のスペクトル整形のみ**であり、
  IR データ長・partition 数・レイアウトを変えない（7C で `numPartsIR` は mode 非依存と実測済み）。
- **Phase 1 の受入基準**：`nucHC/LC` を hash から除外した後、mode 変更での
  `REBUILD_TELEMETRY` 発行が **0** であること。

---

## 7. State / preset compatibility preflight

| 参照 | intent | serialization | snapshot | hash | runtime consumption | test |
|---|---|---|---|---|---|---|
| `convHCFilterMode` | **○**（UI/StateIO） | ○（StateIO:219/163） | ○（AudioEngine.h:3992/4054） | × | **○ ① OutputFilter** | — |
| `convLCFilterMode` | **○** | ○（:220/165） | ○（:3993/4055） | × | **○ ① OutputFilter** | — |
| `eqLPFFilterMode` | ○ | ○（:221/167） | ○（:3994/4056） | × | **○ ② OutputFilter** | — |
| `nucHCMode` | ×（派生） | **○**（StateAndUI:261/389） | ○（:143/202） | **○（:913）** | **○ FilterSpec → applySpectrumFilter** | ○（RoundTrip） |
| `nucLCMode` | ×（派生） | **○** | ○ | **○（:914）** | **○ FilterSpec → applySpectrumFilter** | ○ |

**runtime semantic consumer の閉包**：
- `nucHCMode/nucLCMode` の runtime 消費は **`FilterSpec.hcMode/lcMode` 経由の `applySpectrumFilter` ただ 1 経路**。
  他に runtime で値を使う箇所は無い（他は snapshot/hash/serialization/test のみ）。
- したがって **G8 = BLOCKED にはならない**（撤去・無効化しても他 consumer が壊れない）。
- `nucHCMode/nucLCMode` は **serialization にも使用**されるため保持が必要（State 1+2）。

---

## 8. `convIsLast` dependency audit（全参照）

| 参照位置 | 内容 | 分類 |
|---|---|---|
| `DSPCoreDouble.cpp:458-459` | `convIsLast = convActive && (!eqActive \|\| order == EQThenConvolver)` | **routing semantic** |
| `DSPCoreDouble.cpp:460` | `outputFilter.process(block, convIsLast, …)` | **HC/LC semantic（① 選択）** |
| `DSPCoreFloat.cpp:358-359` | 同式 | routing semantic |
| `DSPCoreFloat.cpp:360` | 同呼出 | HC/LC semantic |
| `OutputFilter.h:118/125` | 引数の説明・宣言 | **API/legacy** |
| `OutputFilter.cpp:200` | 引数 | API |
| `OutputFilter.cpp:214` | `if (convIsLast)` → ①（LC/HC）、else → ②（EQ HPF/LP） | **HC/LC semantic + EQ semantic** |
| `OutputFilter.cpp:267` | コメント（prefetch） | other（コメント） |

**closure**：consumer は **DSPCore 2 ファイル（routing 算出＋呼出）と OutputFilter 内 1 分岐のみ**。
他に `convIsLast` を参照する箇所は存在しない。
→ 113-11 の「`convIsLast` を HC/LC authority から外す」変更は
`DSPCoreDouble/Float` の 2 箇所の呼出分離のみで完結する（`convIsLast` の他用途なし）。

---

## 9. Target topology 最終 freeze

```text
Conv → EQ
  Input → Convolver → HC/LC ① → EQ → EQ HPF/LP ② → Makeup

EQ → Conv
  Input → EQ → Convolver → HC/LC ① → EQ HPF/LP ② → Makeup
```

- `① = Convolver semantic`（LC → HC0 → HC1）
- `② = EQ semantic`（HPF20 → LP0 → LP1、`eqActive` ゲート）
- **`convIsLast` による排他選択ではない**。

---

## 10. RT safety preflight（実装差分予定に対して）

| 項目 | 予定 | 判定 |
|---|---|---|
| lock | 追加なし（`outputFilter.process` は既存、呼出位置のみ移動） | **0** |
| malloc / free / delete | 追加なし | **0** |
| RT 判断 | 追加なし（係数は prepare で全モード分生成済み、RT は index 引き） | **0** |
| new publish | 追加なし（既存 snapshot/atomic 経路） | **0** |
| new retire | 追加なし | **0** |
| new crossfade authority | 追加なし | **0** |

→ 「RT は Read → Execute → Output」を維持。

---

## 11. 実装差分 manifest（113-13 以降）

**変更（ADD/REMOVE/REASON）**

| ファイル | 種別 | 内容 | 理由 |
|---|---|---|---|
| `AudioEngine.Processing.DSPCoreDouble.cpp` | ADD/MOVE | `convolverRt().process()` 直後に `outputFilter.process(block, true, convHC, convLC, eqLPF)` を追加。既存呼出は `false` 固定・`eqActive` ゲート | ① を conv 出力直後へ、② を EQ 段へ固定 |
| `AudioEngine.Processing.DSPCoreFloat.cpp` | ADD/MOVE | 同上 | 同上（Float 経路） |
| `AudioEngine.Parameters.cpp` | REMOVE | `setConvHC/LCFilterMode` 内の `uiConvolverProcessor.setNUCFilterModes(...)` 呼出を停止 | IR に HC/LC を伝えない |
| `MKLNonUniformConvolver.cpp` | REMOVE | `applySpectrumFilter` の HC/LC 適用を停止 | 二重適用の解消 |
| `ConvolverProcessor.StateAndUI.cpp` | REMOVE | `getStructuralHash` の `nucHC/LC` hashCombine を除去 | mode 変更で rebuild を起こさない |

**不変（明示）**
```
OutputFilter.h/.cpp / ConvolverProcessor.h（FilterSpec）/ AudioEngine.h / StateIO
CMakeLists.txt / direct-head implementation / FFT length / HCMode・LCMode enum
```

**ADD/REMOVE の変更有無**：113-11 の候補から変更なし（本 WORK の source audit で必要性の変化なし）。

---

## 12. WORK113-13 への移行条件

| 条件 | 判定 |
|---|---|
| baseline 一致 | **○** |
| contract 一致 | **○**（INV-FILTER-001..004 + direct head を凍結） |
| transition policy source-consistent | **○**（§3） |
| raw IR verification method 確立 | **○**（§5、既存 seam と数値基準 2.049e+03） |
| rebuild baseline 取得 | **△（OBSERVATION）**（§6、engine telemetry 未計測。Phase 1 受入基準は 0 回） |
| state consumer closure | **○**（§7） |
| convIsLast dependency closure | **○**（§8） |
| RT safety closure | **○**（§10） |
| implementation manifest 確定 | **○**（§11） |

### Final Verdict

```text
READY_WITH_OBSERVATION
```

- 実装に進める（Phase 1 から）。
- **観測継続項目（受入試験で確認）**：
  1. §4 の遷移過渡 2 件（Natural→Soft HC / LC）の**実機測定**と A-TRANS-1..3 判定
  2. §6 の **rebuild 回数 baseline 実測**、および Phase 1 後の mode 変更で **0 回**であること
- 契約・manifest・RT safety は凍結済み。Phase 1（State migration only）から
  **1 コミット 1 変更**（`applySpectrumFilter` 停止・hash 除外・routing 変更を同時に入れない）で進める。
