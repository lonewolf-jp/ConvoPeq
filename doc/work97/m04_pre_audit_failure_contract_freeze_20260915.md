# M-04 Pre-Audit / Failure Contract Audit (read-only)

- **日付**: 2026-09-15
- **基準**: commit `1a9b5d61`（SR-02 CLOSED 後）の実ソース = origin/main 正準。`ConvoPeq.md` 生成版 2026-09-15 15:22:01（FRESH）と突合済
- **Type**: read-only（production / tests / CMake 変更 0、commit 0）
- **監査対象**: BugList_and_FixPlan_2026-09-13 §3.1（M-04 Medium-High 格上げ案「smoothing 容量不足時 stale block」）

---

## 1. Source Trace — 実コード現状（BugList 記述の修正）

### 1.1 指摘された return の所在

```613:616:src/convolver/ConvolverProcessor.Runtime.cpp
    if (isSmoothing)
    {
        if (activeSmoothingCapacity < numSamples)
            return;
```

この return は **dry 信号生成・delay ring 更新之后** にある:

| 順序 | 行 | 状態変更 | oversized(>524,288) block での挙動 |
|---|---|---|---|
| 1 | :232 | — | isPrepared=false なら無変更 return（別経路・本件と無関係） |
| 2 | :405-412 | delay ring へ numSamples 書き込み | ring=2^22>1<<20(ABSOLUTE_MAX) のため **安全**（mask wrap） |
| 3 | :415-563 | crossfade 時 oldDryBuf/fadeRamp 書き込み | `fadeRampValid = delayFadeRampCapacity >= numSamples` の**ガードあり**（:423、M-05 遺制）→ invalid 時 crossfade 部をスキップ |
| 4 | :571-589 | ring→dryBuf に **numSamples 無ガード copy** | dryBuf 容量=MAX_BLOCK_SIZE(524,288)。**numSamples>容量なら OOB ヒープ書込** |
| 5 | :591 | `delayWritePos += numSamples` | ring は continue（writePos は mask 済） |
| 6 | :594-604 | !needsConvolution 時 dry→block copy | dry は host block へ（安全） |
| 7 | :613-616 | ← **この return は順序 4 を通過した後にのみ到達** | — |
| 8 | :681-760 | NUC Add/Get（callLen チャンク済み・wetBuf へ numSamples 累積書込）+ smoothing 時 `wetGains/dryGains[processed]` 全ブロック長テーブル参照 | 到達前に順序 4 で corrupt |

### 1.2 到達可能性の確定（監査中核の結論）

- `smoothingBufferCapacity` は prepareToPlay の `allocateIfNeeded` で **必ず MAX_BLOCK_SIZE に設定**（Lifecycle.cpp:316-330。失敗時は `makeAlignedArray` が **bad_alloc を throw** → isPrepared=true（Lifecycle:417）まで到達しない。null ポインタ＋capacity 満タンの「嘘容量」は生成されない）
- `releaseResources` は capacity=0 にするが、同時に isPrepared=false（:477）→ RT は :232 guard で止まり :615 に到達しない
- したがって **:615 の return が実行される唯一の条件は `numSamples > 524,288`（prepared 状態）であり、その条件では順序 4（dryBuf copy）が先に着弾する**
- **BugList v1.3 の記述「block を書き換えず return → 前回 block の stale 残存・周期的ノイズ」は現行コードに対して不正確**（stale return は shadowed され、実害は OOB 書き込み＝ヒープ破損＝クラッシュ/未定義動作）。当時の記述は旧コード構造由来

### 1.3 numSamples の上限チェーン（合法 / 異常分離）

```
host prepare samplesPerBlock
  → maxSamplesPerBlock = PrepareBlockSizingPolicy::apply() = max(256, spb)   ← 上限クランプなし (AudioEngine.h:1128-1131)
  → G1: numSamples > maxSamplesPerBlock → buffer.clear(); return (DSPCoreDouble.cpp:317)
  → OS 経路 G2: num×OS > maxInternalBlockSize = inputMaxBlock×8 → clear+return (:326/:363)
  → convolver process(processBlock)  numProc = num×OS ≤ inputMaxBlock×8
convolver バッファ容量 MAX_BLOCK_SIZE = 524,288 = SAFE_MAX_BLOCK_SIZE(65,536) × 8
```

- **supported envelope（host prepare ≤ 65,536 かつ OS ≤ 8）では :615 も順序 4 OOB も構造的不可能** — `numProc ≤ 524,288 = capacity` が恒成立（等号は通過、`<` 不発）
- 破れ条件: **host が 65,536 を超える samplesPerBlock で prepare する構成**（JUCE/ASIO 契約上は不正ではないが、実 DAW では観測例のない異常域。v8.3 で SAFE_MAX floor 廃止時に **ceiling も不存在**が確定）
- 判定: **「合法な runtime condition」ではない（異常クラス）**。ただしコードはどの層でも上限を強制しておらず、破綻時の帰結がヒープ破損である点はハードニング対象

## 2. 潜在する隣接欠陥（本 work スコープ外・記録のみ）

- **OOM-prepare 例外逸脱**: `ConvolverProcessor::prepareToPlay` のバッファ確保域（Lifecycle:316-330）は try/catch なし。`makeAlignedArray` の bad_alloc が **ホストの prepareToPlay 呼び出し側に伝播**（JUCE オーディオ設定スレッドで terminate リスク）。NUC re-init 部（:292-297）のみ catch 済。→ 別 work「OOM prepare containment」候補
- DSPCore 側（DSPCoreLifecycle:152-165）も同構造

## 3. フォールバック仕様比較（「容量不足時に何を出力すべきか」）

| 案 | 内容 | 判定 |
|---|---|---|
| (a) **entry gate: block.clear() + return（状態変更前）** | `numSamples > MAX_BLOCK_SIZE` を :245 直後（ring 書き込み・retarget・smoother 消費・crossfade 進行の**すべて前**）で門扉化。SR-03/エンジン G1・G2 の clear+return 前例と同型 | **採用**。決定論的無音・ring/dry/wet/NUC ストリーム不変 → 次正常 block で完全無傷復旧 |
| (b) dry-only passthrough | ring を進めて dry を転写 | **否**。NUC 未給餌のギャップが wet に残り（恢复後も穴）、ring 進行で dry 遅延尾も一時的欠損。状態変更を gate に入れてしまう |
| (c) smaller-chunk 分割 | gain テーブルを callLen 窓で再生成しチャンク合算 | **否**（本件では不要）。wet ループは元々 callLen チャンク化済みだが、dryBuf/oldDry/wetBuf の numSamples 書込も併せ再構成が必要 = RT 大重构を envelope 外のために行う費用対効果不正 |
| (d) 現状の :615 を残し順序 4 だけガード追加 | 個別防塞 | **否**。stale return のまま（wet/desync + 部分出力混在）で仕様が二階建てになる |
| (e) engine prepare で inputMaxBlock を 65,536 に clamp | G1 超大 block 全ブロック無音化 | **否**。convolver 単位の隔離より blast radius が広い（EQ/passthrough が死に、plugin 全体無音）。engine 準備契約変更は非対称に重い |

## 4. M-04 契約凍結（実装 GO の場合の変更スコープ）

```
G-1  process() entry gate（Runtime.cpp、isPrepared guard 直後・:245-246 間）:
       if (numSamples > MAX_BLOCK_SIZE)
       {
           block.clear();
           oversizedBlockCounter() の relaxed fetch_add (SR-03 latencyClampCounter と同一文法・static atomic)
           return;
       }
     ※ bypass 経路を含む全分岐を門前払い（envelope 外での統一的決定的無音）。
       ※ 順序 2 の ring 書き込みより前 = 无任何状态推进。
G-2  :615 既存チェックは defense-in-depth として維持（コメントに gate 依存を明記。挙動不変）
G-3  NonRT レポータ: prepareToPlay 部（Lifecycle:154 の SR-03 consume-log パターン踏襲）で
     oversized カウンタ差分を 1 行ログ（RT ログ禁止維持）
禁止（変更 0）: ring/retarget/crossfade/NUC/wet ループ本体、Publish/Retire/Crossfade/Epoch、
     Coordinator、Host PDC、H-01 dryAlign、H-02 peak、SR-03 clamp、SR-01 clamp、
     MAX_IR_LATENCY、DELAY_BUFFER_SIZE、MAX_BLOCK_SIZE 自体の値、新規 CTest target
隣接欠陥 §2 は本 work に含めない（分離 work 化）
```

RT safety 監査（凍結案に対する事前評価）: gate = 比較 1 + `FloatVectorOperations::clear`（memset・RT 合法）+ relaxed fetch_add + return。lock/alloc/free/ログ/待機/ownership 遷移 追加 0。SR-03「bounded latencyDelay publish gate (clamp + telemetry)」前例と同格の safety clamp であり「RT は判断主体ではない」原則との整合は**クランプ（境界強制）であって意味的判断ではない**点に限定。

## 5. テスト契約（先行凍結）

| ID | 内容 |
|---|---|
| T-M04-1 | prepared+IR loaded+mix ramp 中に numSamples=524,289 の block を process() → (i) クラッシュ/ASAN ヒープエラーなし (ii) 出力全ゼロ (iii) oversized カウンタ ≥1 |
| T-M04-2 | **無状態性**: oversized 1 回を挟んだ実例と、挟まない対照実例（同一 IR・同一シーケンス）で、其后の正常 block 出力が**ビット一致**（gate が ring/smoother/crossfade を一切進行させない証明） |
| T-M04-3 | bypass=true で oversized → 同一 gate により無音（現行の「delay passthrough 継続」からの仕様変更を凍結値として明記・ログ検証） |
| T-M04-4 | envelope 内回帰: 65,536×8=524,288 ちょうどは gate 不発（`>` 境界確認）。既存 40/40・H-01/H-02/SR-03/SR-01(B)/SR-02 -suite 不変 |
| T-M04-5 | G-3 NonRT レポーティング: prepare 時にカウンタ差分ログ（注入可テストでないため構造検査＋ログ文字列照合） |

## 6. 重大度の再判定（監査所見）

- BugList v1.3 の **Medium-High 格上げ理由は現行コードで成立しない**（stale 残存経路が dead、条件域が supported envelope 外）。実効: **Low-Medium（latent hardening — envelope 外の入力をヒープ破損から決定的無音へ降格させる門扉）**。格下げの是認をユーザーに求める。
- 但し OOB が成立した場合は症状は爆発的（heap corruption）であり、entry gate の費用（5 行・RT 合法）との比で導入価値は高い。

## 7. 出口判定

```text
Source Trace          完了（BugList 記述が stale である証拠を確定 §1.2）
Failure Reproduction  異常クラスと確定（supported envelope では構造的不可能 §1.3。
                      harness 再現は oversized 直接呼び出しで可能 → T-M04-1 で固定）
RT Safety Boundary    gate 案は clamp+telemetry 前例（SR-03）と同格 §4
Fallback Semantics    (a) entry-gate deterministic silence を採用案として凍結 §3
Contract Freeze       §4+§5
Implementation Gate   **GO（縮小スコープ）** — ただし以下を条件:
  cond-1  BugList の M-04 記述・重大度訂正を本監査記録で差し替え承認（Medium-High stale → Low-Medium latent/OOB）
  cond-2  §2 の OOM-prepare 例外逸脱は本 work に混入禁止（分離 work 起票）
  cond-3  gate は bypass 経路も含む全経路（仕様の单一性）
```

**HOLD 相当の論点なし**（実装なし・契約のみ）。GO 承認時は §4 G-1..G-3 + §5 T-M04-1..5 のみを実装対象とする。

---

## 8. 実装過程の訂正（2026-09-15 Implementation 時 — §1.2/§1.3 の是正・承認 GO の前提を差し替えるもの）

実装前の最終読解で **§1.2〜1.3 の「dryBuf OOB が :615 より先に着弾する」主張は誤り**であることを発見した。
監査記録の訂正として本节を追加する（原文は歴史的記録として保持、以下が有効）:

1. **既存の capacity guard**: `process()` には :319-323 / :326-330（行番号は訂正時）に
   `if (numSamples > activeDryCapacity/activeWetCapacity) { block.clear(); return; }`
   が**既に存在**し、oversized 入力の dry/wet/smoothing/oldDry/fadeRamp への OOB は発生しない
   （prepared 時に全容量は MAX_BLOCK_SIZE へ統一されるため :615 も含め到達順は guard 側が先）。
2. **実在する欠陥は 3 点**（G-1 の是正対象として有効性は変わらない）:
   - (i) **guard が retarget 節（:286-311）の後**にあり、oversized block を破棄する際にも
     latencySmoother / crossfadeGain / oldDelay が進行済みになり得る（crossfade が発火したまま
     対応サンプルが処理されない不整合）。dry 容量 guard 前に状態変更がある点こそが本質。
   - (ii) **bypass 分岐（:280-284）は guard を通らない**（ring 2^22 で OOB はないが、
     containment 契約が分岐ごとに不一致 — cond-3 の単一 entry contract と両立不能）。
   - (iii) 現行 guard は**無観測**（telemetry なしの暗黙 clear。G-3 の counter/log 不在）。
3. §3 採用案 (a)「entry gate（全状態変更前・bypass 含む単一点）」は訂正後も正しい処方。
   G-1 は :319/:326 を置換せず**前段に単一点として追加**（defense-in-depth 維持、G-2 反映済）。
4. 重大度の再判定（§6）は不変: supported envelope 外限定の latent hardening = **Low-Medium**。
   「stale 残存バグ」でも「OOB 破損」でもなく「**破棄前の状態進行 + containment 非一様 + 無観測**」が正確。
5. 監査側の独立取得ができていない `ConvoPeq.md` 2026-09-15 15:22:01 版との突合は、実ファイル
   （origin/main `1a9b5d61`）への直接 grep で完遂しており、本节の事実は実物ソースに依存する。
