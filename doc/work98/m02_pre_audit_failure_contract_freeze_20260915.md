# M-02 Pre-Audit / Failure Contract Audit (read-only)

- **日付**: 2026-09-15
- **対象**: 2026-09-13 監査系 M-02「wet 出力の NaN スクラビングが FDL 内部状態を放置」
- **根拠文書**: doc/audit/ConvoPeq_IR_Audio_Path_Audit_2026-09-13.md §M-02 /
  ConvoPeq_Bug_Verification_2026-09-13.md §6 / ConvoPeq_BugList_and_FixPlan_2026-09-13.md §4・§6（Step 6）
- **前例フォーマット**: doc/work97（M-04）— BugList 記述の実ソース突合と重大度再判定を必須とする
- **baseline**: origin/main `ef231ee8`（M-04 entry gate 適用後・ConvoPeq.md 2026-09-15 19:13:04 / FRESH 確認済）
- **方式**: read-only Source Trace（production 変更 0・実装案は §3/§7 の決定点として保留）

---

## 1. Source Trace — 実コード現状（BugList 記述の修正）

### 1.1 スクラブの所在と適用域

- 定義: `ConvolverProcessor.Runtime.cpp:59` `sanitizeFiniteChunk`（ビット判定 `isFiniteAndAbsBelowNoLibm(x, 1.0e300)`。
  非有限 **および** |x| ≥ 1e300 を 0 にする）。
- 唯一の呼び出し: `Runtime.cpp:771`（BugList 記述の :722 は M-04 適用後の行数シフト。現在 :771）。
  `conv->process(ch, input, wetOut, chunkSamples)` 直後、**wetOut（= wetBuf チャンク）のみ**・mix 前。
- 適用外（実コード確認）: `dst`（= block チャネルポインタ、in-place 最終出力）、`dryBuf`、
  smoothing 経路の remainder 転写（`copy(remDst, remDry)`）。
- 別実体: `AudioEngine.Processing.DSPCoreIO.cpp:39` にも同名がある（エンジン境界用・§1.4 参照）。

### 1.2 汚染循環チェーン（NUC 内部）

```
input → inputAccBuf → [prev|cur] FwdFFT → fdlReal/fdlImag[slot]（+ mirror）
      → accum（直近 numPartsIR スロット窓）→ accumBuf → killDenormal(L0/immediate のみ:1492-1502)
      → IFFT → L0: ringWrite → ring / L1,L2: tailOutputBuf → delayLineWrite
      → Get = ringRead + delayLineReadAdd → wetOut（:771 でスクラブ）→ mix
```

- 汚染 fdl スロットは毎ブロック 1 ずつ上書き（mirror 含め周期 numParts ブロック）。
  accum 窓（numPartsIR）から出るか上書きされるかで消失 ⇒ **受動的自己失効**。
  単発汚染の出力汚染持続 ≦ 当該レイヤーの IR 窓（L2 最悪: カバレッジ ≤ l2Len samples、
  上限は MAX_IR_LATENCY=2^21 由来、通常 2 s IR ≦ 数秒）。
- accumReal/accumImag/accumBuf はブロック毎 memset（:1463-1464/:1663-1664/:1701）— 残留しない。
- ring/tail/delayLine は書換え消費で有限窓で流出。direct tap は m_directHistLen ≤ 31（:703-706）。

### 1.3 「永続ミュート」の訂正（本監査中核の結論その1）

BugList/検証資料の「FDL 残留で**永続**ミュート」は不正確:

1. 汚染源が一度限りなら、出力は **f(dL)（レイヤー IR 窓長）で自己失効復旧**する（§1.2）。
2. 「永続」が成立するのは (i) 入力の**連続汚染**（自己発振ループ等）、(ii) IR 自体が非有限、
   のいずれか。いずれも §1.4 の上流ガードにより **shipped チェーンでは通常到達不能**。
3. 実在する欠陥は: **(a) 無観測**（スクラブが replacement を捨てて silent 動作。telemetry 不在）、
   **(b) recovery が受動的失効のみ**（能動クリア/再構築の誘導なし）、
   **(c) エンジン外駆動時 dry 経路が完全に無防備**（:1.1 適用外。§2-混入）。
   ※ **訂正ポインタ（実装過程・§8-12）**: §1.2 の窓推定は部分事実。実測失効窓は構成により
   ≒ 100 blocks（約 10×IR 窓級・4800-sample IR 構成で burst 終端から ~100 blocks）。
   「有限窓で失効（永続ミュートなし）」の結論は実測で有効。テストは有界回復判定ループ採用。

### 1.4 NaN 侵入ベクター網羅（棚卸し・全項目確定済み）

| ID | ベクター | 現状ガード | 残存 |
|---|---|---|---|
| V1 | ホスト入力 NaN/Inf | エンヂョ境界二重 sanitize: `processInputFloat/Double` 前置（DSPCoreIO:231-232/288-289）+ **DC 後**（:256-257/311-312、work92 C-3）。OS 経路: 補間 FIR 線形 + `UltraHighRateDCBlocker` 内部状態ガード（h:185-186、|state|≥1e15→0、係数検証 :81-105） | shipped チェーンでは **遮断済** |
| V2 | IR 非有限 | LoadPipeline スケーリング後全サンプル `isfinite→0`（:425-430）、minimum-phase 変換は spectrum ±50 クランプ＋全サンプル検証（ResampleAndFallback:441-466、失敗時 `{}` フォールバック）、peak スキャン非有限 skip（LoaderThread:627）| init(SetImpulse) 到達前に保証。ただし**再構築経路（Rebuild 経由 init）の呼び出し鎖が同一検証を必ず通る前提**（テスト T-M02-3 で固定） |
| V3 | HC/LC スペクトラムフィルタ | `applySpectrumFilter`（cpp:336-）ゲインは pow/√ の域内有限（denom 0 分岐は kEnd=N/2 構造上 k>kEnd でしか到達せず到達不能）、有限 irFreq に乗算 | 遮断済（形式上是限） |
| V4 | 内部オーバーフロー（有限大入力 × 有限大 IR 積 > 1.8e308 → ±Inf→NaN） | **ガードなし**（V1 のしきい 1e300 は巨大有限を通す。FFT × numPartsIR 窓加算）| 数学的可能・要病的条件（入力量子 ≥1e200 級 or IR 振幅 ≥1e200 級。V4b: IR 振幅由来は再読込以外に処置なし=スコープ外） |
| V5 | エンジン外直接駆動（harness / 将来の直接 entry） | なし（engine edge を通らない） | テストのみが該当。dry 露出は §2-(a) に記録 |

**到達可能性結論**: M-02 の症状（wet 側ミュート/非有限残留）は shipped チェーンでは V4 の病的条件または
V1/V2 ガードの**退行時**にのみ顕在化する latent hardening 課題。M-04（work97 §6）と同型の格下げ対象。

### 1.5 スレッド契約とクリアコスト（上側から見た選択肢空間）

- NUC ヘッダ契約（MKLNonUniformConvolver.h:34-36）: `SetImpulse/Add/Get` = 呼び出し側規定、
  **`Reset()` = Message Thread または releaseResources()**。RT スレッドからの直接 `Reset()` は
  **現行契約違反**（RT 実行体として memset のみ・malloc 無しなので物理的には可能だが、契約変更を要する）。
- `Reset()` 本体（cpp:1892-1943）は fdlSoa・ring・tail・delayLine・direct を全消去（無積割、ログ無し）。
  worst-case fdlSoa = numParts×2×complexSize×2 arrays。幾何上限（L2・mult=8・bs=512）で
  数十 MB 级の memset = **RT callback 予算として不適**（M-04 gate の block.clear()（≦2MB ×2ch）より
  桁違い）。→ 能動クリアは NonRT 再構築誘導が契約上唯一の整合形（§3-c）。
- 既存同型パターン: `latencyResetPendingGen` / `mixSmootherResetPendingGen`（NonRT→RT 世代通知）。
  M-02 で必要なのは逆方向（RT 検出→NonRT 駆動）の pending であり、SR-03/M-04 telemetry と
  同一原子語彙（fetchAdd acq_rel / consume acquire）で表現可能。
- 再構築経路の再構築元: `StereoConvolver` は irData を保持（h:872-880・init で所有）→ 保持 IR からの
  re-init は H-03 deferred rebuild と同一の NonRT コスト。交換は publish/retire + epoch で保護済。

### 1.6 reporter 駆動の棚卸し（確定）

`ConvolverProcessor::timerCallback`（Lifecycle:144-）の production 駆動は、
**リポジトリ全体 + git 履歴（`git log -S startTimer -- src/convolver/ src/ConvolverProcessor.h` = 0 hit）で
startTimer 呼び出しが一度も存在しない**ことを確認した。すなわち SR-03 clamp reporter（:154）と
M-04 G-3 reporter（:162-169）は**休眠経路**（実働は test pump のみ）。本 work の telemetry も同
pattern 踏襲を契約値とし、配線自体は §2-(d) の別 work 記録に分離する（本 work で暗黙に直さない）。

---

## 2. 潜在する隣接欠陥（本 work スコープ外・記録のみ）

- **(a) dry 最終出力の無スクラブ**: :771 は wet のみ。out-of-engine 駆動（V5）で入力 NaN が
  `dst = wet*G + dry*G` の dry 項を汚染し、スクラブされず出力へ。shipped チェーンは engine edge で無害化。
  → 別 work「convolver self-edge input/output sanitize」候補（M-02 実装に混入禁止）
- **(b) M-01**: L1/L2 pre-IFFT デノーマルガード欠落（:1492-1502 は immediate=L0 のみ、分散経路
  :1701-1708 に killDenormal なし）。**Step 6 同日別 item として計画済**（BugList §5）。混入禁止。
- **(c) OOM-prepare 例外逸脱**: work97 §2 と同一（分割管理）。
- **(d) timerCallback 休眠配線**: §1.6。reporter 実働性を要求するなら別 work で startTimer 配線 +
  RCU guard 監査（#021 の GlobalGuard は済、駆動元が不在）。

---

## 3. 回収セマンティクス比較（「検出時に何をすべきか」）

| 案 | 内容 | 判定 |
|---|---|---|
| (a) telemetry-only（現状 + 観測） | スクラブ置換数を counter に加算。状態クリアは §1.2 の受動失効に委ねる | **最小契約**。recovery 窓 = レイヤー IR 窓長（契約値として明記）。V4 病的入力が続く間は静音継続（現状と同じだが可視化） |
| (b) RT 内 active clear | 検出時 RT から fdl/accum/ring を memset | **否**。NUC スレッド契約（h:36）違反 + 数十 MB memset で RT 予算外。契約変更なしに正当化不能 |
| (c) RT 検出 → pending（acq_rel counter）→ NonRT re-init（保持 irData から再構築・publish/retire 交換） | H-03 deferred rebuild / SR-03 reporter と同一部品群の再利用 | **有力**。能動 recovery を保証。コスト: NonRT rebuild（engine 再構築）+ 交換までの stale窓（epoch 保護下・既存 H-03 と同等）。ただし「検出 1 回で rebuild 発火は重すぎる/連続検出でループ」の閾値設計が必要 → 実装 GO 時に凍結 |
| (d) レイヤー局所 clear | 汚染レイヤーのみ特定消去 | **否**。検出点が Get 加算後で層帰属不能（帰属不能を §1.3 で実証済） |

契約凍結値: **C-1（観測）は必須**。C-2 は **(a) か (c) のいずれか**を実装 GO 判定に委ねる（決定点 D-1）。
本 Pre-Audit は実装案を確定しない（ユーザー方針）。

---

## 4. M-02 契約凍結（実装 GO の場合の変更スコープ）

```
C-1 検出・観測（Runtime.cpp）:
      sanitizeFiniteChunk を置換カウント返し（int）へ。呼び出し側で
      nonFiniteBlockCounter()（仮称・SR-03 latencyClampCounter / M-04 oversizedBlockCounter と
      同一文法: static atomic + fetchAdd acq_rel(RT) / consume acquire(timerCallback・NonRT)）
      に加算。reporter は timerCallback に 1 行ログ（ヒステリシス lastReportedXxx_ 方式）。
      RT 側ログ禁止。検出位置は :771 のまま（スクラブ仕様は不変: 非有限 ∪ |x|≥1e300）。
C-2 回収（決定点 D-1 = (a) or (c)）:
      (a): 何もしない（受動失効）。recovery 上限を契約値として §1.2 式で明記。
      (c): RT は検出時に pendingNanRecoveryGen（acq_rel fetchAdd）を publish（音の処理は
           現状通りスクラブ継続）。NonRT（timerCallback / LoaderThread 相当）が世代差分で
           保持 irData から re-init→exchangeActiveEngine 誘発。連続検出ループ回避のため
           「1 recovery あたり最小間隔 N ms」閾値を凍結（実装 GO 時に N 承認）。
C-3 配置契約: C-1 は entry gate（M-04・G-1）と dry mix の**間**（wetOut 検査点として唯一）。
      bypass 経路（:304-307）は wet を生成しないため対象外（bypass の dry 露出は §2-(a)）。
禁止（変更 0）: M-04 gate・SR-03 clamp・H-01/H-02・SR-01/SR-02、Publish/Retire/Crossfade/Epoch、
      Coordinator、NFC 判定閾値 1e300、MAX_BLOCK_SIZE、killDenormal（M-01 側）、dry 出力仕様、
      エンジン edge sanitize（V1 側）、新規 CTest target
隣接欠陥 §2(a)(b)(c)(d) は本 work に混入禁止。
```

RT safety 監査（凍結案に対する事前評価）: C-1 = 既存ループの return 値化（ALU 増分のみ・RT 合法）。
C-2(c) の RT 側 = atomic fetchAdd 1 回（SR-03 telemetry と同一語彙）—「RT は判断主体ではない」原則に
対し、RT は検出（比較）と予約（加算）のみを行い、再構築判断は NonRT。M-04 clamp と同格の
「境界強制 + 無状態予約」に限定。recovery 動作本体は NonRT-only で NUC 契約（h:36）と整合。

---

## 5. テスト契約（先行凍結）

| ID | 内容 |
|---|---|
| T-M02-1 | 病的入力注入（direct drive・V5 相当: wet を非有限化できる入力。例: IR 正規化後の大振幅 impulse 経路で検出自体を強制）→ (i) 出力全有限（静音混 OK）(ii) counter +n（置換数整合）(iii) 他 40/40 suite・M-04/SR-03 counter 無干渉 |
| T-M02-2 | recovery（D-1=(a)時）: 汚染注入停止後、レイヤー窓長（契約式）以内で対照系出力と**ビット一致**復旧。(c) 時: re-init 要求後 exchange 完了から正常出力（交換窓の扱いを H-03 precedent 記述と照合） |
| T-M02-3 | IR 経路回帰防止: 非有限混入 IR → LoadPipeline 検証で全サンプル有限化 or `{}` fallback（**既存検証の呼び出し鎖**が rebuild 経由 init でも成立する回廊を 1 本テストで固定） |
| T-M02-4 | reporter ヒステリシス: pumpReport 2 回 → ログ 1 件（M-04 T-M04-5 と同一形） |
| T-M02-5 | V1 エッジ不変: engine 側 sanitize 経路（work92 C-3）の回帰テスト既存あれば相互参照、無ければ DSPCoreIO 側は触れない契約確認のみ |

---

## 6. 重大度の再判定（監査所見）

- BugList v1.3 の Medium（検証: 確認）は **Low（latent hardening）** へ訂正を求める:
  主侵入ベクター（V1/V2/V3）は shipped チェーンで遮断・ガード済。残存 V4/V5 は病的条件/テスト専用。
  「永続ミュート」は §1.3 の受動失効モデルに差し替え（1-line revert 準備度マトリクス §4 も
  C-1+C-2(c) 採用時は「低」該当性消失 — マトリクスは本監査記録で更新）。
- 但し症状が起きた場合の聴感（完全静音）と可視性の無さ（無観測）は、M-04 と同じく
  「費用 10 数行・再発時調査不能の解消」比で導入価値を認める。

## 7. 出口判定

```text
Source Trace          完了（BugList「永続ミュート」記述の是正 §1.3・適用域 §1.1・循環 §1.2）
Failure Reproduction  shipped チェーンでは構造的不可能（V1-V3 ガード §1.4。harness 直接駆動で再現可
                      → T-M02-1 で固定）
RT Safety Boundary    C-1/clamp 前例と同格。active clear (b) は契約・予算両面で不可 §1.5
Recovery Semantics    (a) 受動失効 / (c) NonRT re-init 誘導 — 決定点として凍結 §3
Contract Freeze       §4+§5（実装案は未確定・D-1 判定待ち）
Implementation Gate   **PENDING（GO 可能・条件付）** — 条件:
  cond-1  BugList M-02 の重大度・「永続」記述・準備度マトリクス（1-line revert）訂正の承認
  cond-2  D-1 の選定（(a) telemetry-only 最小版 か (c) NonRT recovery 付）。
          推奨: 第一弾 (a)（変更最小・M-04 と同一 telemetry 語彙）、(c) は閾値設計を別 commit 化
  cond-3  §2 の混入禁止（(a) self-edge sanitize / M-01 / timerCallback 配線 / OOM-prepare）
  cond-4  (c) 選択時は NUC スレッド契約 h:34-36 の文言更新を contract change として明記
```

**保留の論点**: D-1 のみ（実装 GO 承認時にユーザー裁定。本 work は契約凍結まで）。
棚卸しした全調査項目（V1-V5、循環持続、クリアコスト、thread 契約、reporter 駆動）は本文で確定済み。

以上。

---

## 8. 実装確定事項（2026-09-15 Implementation — D-1=(a) telemetry-only 承認済み GO に基づく）

実装 GO（Implementation Gate: cond-1 APPROVED / D-1=(a) APPROVED / cond-3 APPROVED / cond-4 N/A）による
契約値の確定記録。**§4-C-1 が実装対象全部**（C-2(c) の Reset/re-init/exchangeActiveEngine/
pendingNanRecoveryGen/閾値/cool-down は今 commit に混入禁止＝承認済み変更禁止リスト）。

1. **単位凍結（混同禁止条項の実装形）**:
   - `sanitizeFiniteChunk(double*, int)` の返却値 = **置換サンプル数**（検出判定専用）。
   - `nonFiniteBlockCounter` の増分 = **1 / scrub 発火 chunk**（`replacements > 0` のとき +1）。
     置換サンプル数を counter に足さない（名前「BlockCounter」と単位一致）。
   - reporter ログ文言: `ConvolverProcessor: M-02 non-finite wet scrub fired (total: N blocks)`
     — 単位 = chunk 数であることを文言 (blocks) で明示。
2. **配置確定（§4 C-3 の現在行番号）**: 検出・増分は `Runtime.cpp:781`（mix 前・wetOut スクラブ点、
   chunk ループ内）で M-04 gate（:257）と dry mix の間に位置。bypass 経路（:304-307）は対象外不変。
3. **raw atomic 追加 0（承認条件遵守）**: 増分=`convo::fetchAddAtomic(acq_rel)`、
   reporter=`convo::consumeAtomic(acquire)`＋`lastReportedNonFiniteCount_` ヒステリシス。
   SR-03/M-04 と同一語彙（wrapper 必須条項）。RT 側ログ・確保・待機 0。
4. **reporter は休眠のまま（承認 §5 のとおり）**: `timerCallback` の production 駆動配線
   （startTimer）は追加しない（§2-(d) 別 work）。test pump のみで駆動・検証。
5. **HealthMonitor / policy / World / Crossfade 接続 0**（承認 §3 禁止条項遵守）。
6. **テストのトリガー設計とビルド差の解消（T-M02-1 相当の両ビルド成立）**:
   Debug では L0 pre-IFFT `killDenormalV`（DspNumericPolicy.h:274）の ordered 比較マスクが
   NaN を零化するため L0 単独では発火しない。L1（分散経路・killDenormal 無し＝ §2-(b) M-01 の
   既知欠陥）経由で tailOutputBuf→delayLine→Get に非有限が到達し、Release/Debug 双方で
   scrub 発火を確認できる。4800 sample IR + 512 block で l1Part=4096＝burst 8 blocks が
   L1 part block ちょうど。recovery 比較窓（burst 後 ≧40 blocks）は §1.2 持続式の上位。
   mix=1.0 で dryG=0 とし「最終出力=scrub 適用 wet」を主張成立させている（dry 経路は §2-(a)
   スコープ外に厳密なまま）。
7. **T-M02-3 の実装形（§5 からの調整）**: WAV 経由ロードは整数サンプル由来で非有限が注入不能な
   ため、「検証済ロード鎖 + clean warmup 16 blocks で counter delta 0」＝観測可能な
   ロード回帰防止として実装。非有限 peak 除外の単体検証は本 TU 既存テスト（IRPeak T4b/T4d・
   LoaderThread :627 系）がカバー済み（§1.4 V2 の根拠と同一）。
8. **T-M02-5（§5 の T-M02-5 振替）**: エンジン edge（V1）は無変更で diff boundary 側（§9 相当）で
   担保し、テスト側は「recovery 後 clean 8 blocks で誤発火 0」を contract 値として実装。

9. **実装過程の発見（テストハーネス側の潜伏バグ・production 无関係）**:
   既存 checkM04 の `makeInput` と本 test の `makeClean` は
   `getWritePointer(0)[p]`（p=37+97·idx / 11+61·idx）を **%512 なしで buf(512) へ書き**、
   idx が大きいブロックでは **range 外書き込み（heap corruption）** をoccasionally発生させていた
   （Release AudioEngineHarness の非決定 SEGFAULT / T-M02-2 非決定 divergence の原因として確定。
   Dr.Memory 相当の再現: warmup j≒9（p≧512）以降にクラッシュ位置が移動）。
   - 対照系と被験系が同一入力で OOB するため T-M04 の bit-identical 判定自体は偶然成立していた
     （決定論的に同一破損）が、**解放後ヒープ状態に依存する非決定クラッシュ**の実在を認める。
   - 本 commit で `makeInput`/`makeClean` とも `%512` 化（テスト専用・production 差分 0）。
     注入パターンの意味（決定的 1 サンプルインパルス）は不変。
   - 教訓: `juce::AudioBuffer::getWritePointer` は境界検証しない。テストで計算 index を書く場合
     `%getNumSamples()` を契約化する（別 work「test harness index lint」候補）。
10. **T-M02-1 の主張範囲の訂正（§5 からの調整）**: burst 中の per-block 出力 finiteness は
    主張しない。mix ramp 収束前の dry リング経由非有限（§2-(a)）と混在するため、finiteness/
    bit-identical 主張は warmup 尾部（ramp 収束後）と drain 尾部（失効完了後）に限定。
    counter 発火は drain 窓内で確認（Release=L0/L1 両経路、Debug=L1 分散経路のみ）。
11. **±Inf を注入対象から除外**: Debug の L0 `killDenormalV` は ordered 比較のため Inf を
    零化せず通過させる（NaN のみ零化）→ Inf 混入時は汚染持続がレイヤー横断で非決定化し、
    契約実証に不要な変数を増やす。NaN 単独で両ビルド発火が成立するため Inf は使わない。
12. **失効窓の実測訂正（§1.2 の見積り差し替え）** — Pre-Audit 式 `(numPartsIR-1+numParts)·partSize` は
    **部分事実**だった。実測（4800-sample IR / 512 block / 48k / mix=1.0 / 8-block NaN burst、
    L0 単独構成）: scrub は burst 終了後も **≒100 blocks（約 1.07 s、最終発火 block 147／
    burst 終端 47）** 継続し、その後完全失効 → 対照系とビット一致（cnt=216 で凍結、
    drain 512 blocks 内で安定）。永続ミュートは観測されず **D-1=(a) の前提（有限窓で失効）は成立**
    するが、窓は §1.2 の数倍〜一桁長い。従って:
    - T-M02-2 の recovery 主張は「drain cap 512 blocks 内に連続 16 blocks ビット一致」の
      **有界回復判定ループ**として実装（固定窓 tail は不可）。
    - §1.2/§3(a) の「持続 ≦ 層 IR 窓」は **≦ 約 10×IR 窓級（構成依存）** に読み替え。
      機構の厳密な全経路（ring 4096 + fdl 窓 + L0 part 境界相互作用）は本 work の
      telemetry-only 範囲外 — 将来 recovery work（C-2(c) 採用時）の再解析対象として記録。
13. **テストの片付け契約**: checkM02NonFiniteTelemetry は exit 経路全てで
    `control/gated.releaseResources()` を呼ぶ（LoaderThread 滞留由来の非決定性排除）。
14. **burst 中 dst finiteness は主張不能と確定（§5 T-M02-1 の調整）**: burst 入力 NaN は
    dry リング（delayBuffer 生入力写像）へも供給され、`mixSteadySmall` の
    `dry[i]*dryG` は **dryG=0.0 でも 0*NaN=NaN** を生成（無分岐実測）→ in-place dst は
    burst block で非有限になりうる。これは §2-(a) dry self-edge の機構そのものであり、
    shipped チェーン（V1 エッジ sanitize）では非到達。C-1 の wet スクラブ有限性証明は
    **drain 窓（入力=dry 洁净）の per-block allFinite** で担保（発火継続 ≒100 blocks 全区間で
    成立 = scrub 決定的無音の実証）。T-M02-1 断言 = 「burst 後に counter 発火・drain 内で
    出力有限」、T-M02-2 =「cap 512 内に連続 16 blocks ビット一致」で凍結。

以上を以て M-02 実装は §4 C-1 範囲（sanitize 返却値化 / counter / RT 増分 / NonRT reporter /
テスト T-M02-1..5 / 既存 CTest target 内）に閉じている。

## 9. 検証記録（2026-09-15 Implementation 完了時）

| 項目 | 結果 |
|---|---|
| ctest Release（フル） | **40/40 PASS**（REL_CTEST_EXIT=0） |
| ctest Debug（フル） | **40/40 PASS**（DBG_CTEST_EXIT=0） |
| 安定性再走（harness のみ ×3 ×両構成） | **全 100% PASS**（非決定 SEGFAULT の再現 0 — §8-9 の OOB 修正で確定解消） |
| T-M02-1..5 | checkM02NonFiniteTelemetry: PASS（load-clean baseline / burst→fired / drain finiteness + 有界 aging recovery / reporter hysteresis / no false fire） |
| clang-tidy（変更 TU） | error/warning 0（registry チェック内・抑制 58 は既存非 user code / filter） |
| cppcheck（変更 3 ファイル） | 指摘 0 |
| RT safety scan（check-src-atomic-dotcall） | **PASS**（raw atomic dot-call 0 — wrapper 必須条件遵守） |
| ConvoPeq.md | 再生成 2026-09-15 23:22:48 / `--check` FRESH・NEWER_SRC_COUNT=0 |
| 実測失効窓 | burst 終端+≒100 blocks まで発火継続（最終発火 block 147）→ 以後ビット一致・cnt 凍結（§8-12） |

evidence: `evidence/M02_FINAL_VERIFICATION.txt`（台帳）/ `evidence/M02_BUILD_TEST_LOG.txt`（チェーン生ログ）/
`evidence/M02_STATIC_ANALYSIS_LOG.txt`（静的解析）。

出口: commit/push は Final Gate 承認後（本 work は報告までで停止）。
