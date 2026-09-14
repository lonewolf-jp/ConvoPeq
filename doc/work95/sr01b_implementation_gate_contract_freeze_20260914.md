# SR-01(B) Implementation Gate / Contract Freeze

- **日付**: 2026-09-14
- **基準**: source snapshot @ commit `5262957b`（H-01 CLOSED 後・ConvoPeq.md Generated 2026-09-14 17:06:17 と同一内容の作業ツリー）
- **Type**: read-only Implementation Gate（production code 変更 0）
- **監査根拠**: doc/audit/ConvoPeq_SampleRate_Support_Audit_2026-09-13.md §1・ConvoPeq_SampleRate_Reverification_2026-09-13.md §2・ConvoPeq_BugList_and_FixPlan_2026-09-13.md §2.1/§5 Step 4
- **ツール**: rg/grep・sed（WSL 経由）・git grep・直接読取（headroom proxy 自動圧縮併用）

---

## 1. 現行コードの事実（2026-09-13 監査からの更新点を含む）

### 1.1 定数（src/ConvolverProcessor.h）

| 定数 | 値 | 位置 | 意味 |
|---|---|---|---|
| `IR_LENGTH_MIN_SEC` | 0.5f | :222 | UI/秒パラメータ下限 |
| `IR_LENGTH_MAX_SEC` | 3.0f | :223 | **推奨**上限（UI デフォルト最大） |
| `MAX_IR_LATENCY` | 2,097,152 = 2^21 | :250 | IR サンプル数のメモリハード上限 |
| `MAX_BLOCK_SIZE` | 524,288 | :254 | — |
| `MAX_TOTAL_DELAY` | 2^21 + 524,288 = 2,621,440 | :255 | — |
| `DELAY_BUFFER_SIZE` | 4,194,304 = 2^22 | :257 | **本 work で変更禁止**（既に MAX_TOTAL_DELAY を内包） |

`MAX_IR_LATENCY/sr ≥ 3.0` となるのは `sr ≤ 699,050 Hz` まで。768 kHz では
`2,097,152 / 768,000 = 2.730666…s` がハード上限（SR-01 の原始症状 = 3.0s 要求に対し 0.269s の尾が無言切断）。

### 1.2 targetLength 生成・clamp・trim・allocation chain（確定）

```
[UI] irLengthSlider (max 静的3.0 / updateIRInfo で max(3.0, cur))
  └ engine.setConvolverTargetIRLength → AudioEngine.Parameters.cpp:575-580
     └ ConvolverProcessor::setTargetIRLength  (Runtime.cpp:883-897)
          maxAllowed = getMaximumAllowedIRLengthSec(currentSampleRate)  ★SR依存 hardMax 適用済
          jlimit(0.5, maxAllowed, req) → pendingOverride.targetIRLengthSec
[auto-detect] applyAutoDetectedIRLength (Runtime.cpp:900-923)   ★同一 hardMax clamp 済
[state復元]   setState (StateAndUI.cpp:301-356)
   手動/autoDetected とも jlimit(…, getMaximumAllowedIRLengthSec(currentSampleRate), …) ★適用済 (:329-331)
[snapshot]    copySnapshotToPendingUnlocked (StateAndUI.cpp:146-199)
   :162-167 targetIRLengthSec/autoDetectedIRLengthSec を
   jlimit(IR_LENGTH_MIN_SEC, IR_LENGTH_MAX_SEC=3.0, …) ★★静的3.0のみ — ギャップ残存（§3 C-1）
[loader]      LoaderThread step (LoaderThread.cpp:505-607)
   silence-trim → resample to processing SR (:527-541, loadedSR:=sampleRate)
   → DC blocker → Tukey
   → targetLength = computeTargetIRLength(loadedSR, len)  (:585 / StateAndUI.cpp:938-953)
        target = (int)(sr × pendingOverride.targetIRLengthSec)
        min(target, MAX_IR_LATENCY)  ★二重安全網 — 発動时无音（ログなし、§3 C-3）
   → stepTrimmed = 先頭 copySamples + 末尾 fade-out (:586-602)
[build]       buildConvolverFromTrimmed (:149-195)
   irL/irR = makeAlignedArray<double>(targetLength) ×2ch (:163-164)  ← mono は ch1=ch0 複製 (:167)
   H-02 measureIrPeak → StereoConvolver::init → NUC SetImpulse（層割付は targetLength 依存）
[publish]     queueFinalizeOnMessageThread / initializeConvolverSynchronously
   → applyNewState → publishAtomic(irLength, targetLength) (:828) / 失敗時 success=false・旧 engine 保持 (:231-233)
```

### 1.3 既に存在する SR 依存機構（09-13 監査後の進展）

- `getMaximumAllowedIRLengthSecForSampleRate(sr) = (float)(MAX_IR_LATENCY / sr)`（StateAndUI.cpp:921-927、sr≤0 → 3.0）
- `getMaximumAllowedIRLengthSec()`（:929-936、currentSampleRate acquire コンシューム、HB コメント済）
- preview: `hardMaxSec = MAX_IR_LATENCY/processingSR`、`exceedsHardLimit`（:451-452, :533-534）
- preview 完了時 `exceedsHardLimit` → **「IR Too Long」ダイアログで reject**（ControlPanel.cpp:1159-1173）

**注意（freeze 判断に直結）**: 本関数は `min(3.0, ·)` を**取らない**。48 kHz では hardMax = 43.69s
であり、3.0s 超 IR の "Load as-is"（preview :1184-1202 → applyAutoDetectedIRLength(detected)）は
**現行仕様として正常に成立**する（NUC/遅延リングは 2^21 サンプル上限で SR 非依存に確保されるため、
メモリ増はゼロ）。よって凍結式に 3.0 との min を入れると 48k 系で**新規リグレッション**になる — 採用しない。

### 1.4 NUC 層割付（MKLNonUniformConvolver.cpp:738-758）— 変更不要の確認

`l0Part=nextPow2(max(bs,64))`、`l1Part=l0Part×mult`、`l2Part=l1Part×mult`、`kL0MaxParts=32`、`kL1MaxParts=64`。
層の合計サンプルは `irLen ≤ MAX_IR_LATENCY` に律動し、**上限自体は sr に依存しない**（2^21 サンプルは
48k でも 768k でも同じ確保量）。したがって SR-01(B) は allocation 系一切に触れない。

---

## 2. SR-01 問題の再証明（現行 HEAD での残余範囲）

| 経路 | 768k × 3s 要求の現挙動 | 無言か |
|---|---|---|
| UI プレビュー（ファイル > hardMax） | reject ダイアログ（既装） | NO（正常） |
| UI スライダー 3.0 選択 | setTargetIRLength が 2.73 へ暗クランプ→ slider 表示は次 update で追随 | ほぼ無言（§2-B） |
| **BuildSnapshot 適用（rebuild/device SR 変更）** | copySnapshotToPendingUnlocked が 3.0 を素通し → computeTargetIRLength が 2^21 で無音切断・fade 付与 | **YES — 残余本体** |
| 起動時復元ロード（StateAndUI.cpp:409-410 loadIR） | setter 側 clamp で 2.73 相当に収まるが、ファイル 2.304M>target で無音 trim | **YES（切断事実）** |
| プログラム API setTargetIRLength(>hardMax) | clamp 済（§1.2） | NO |

数値証明: 3s @768k = 2,304,000 samples ×2ch ×8B = **36.86 MB** 要求に対し上限 2^21（正確には §6 T-SR01-4 の targetLength 2,097,151）= **33.55 MB**
（差 206,849 samples = 0.269s ≈ 1.65 MB/ch・両 ch 計 3.31 MB）。Option A（2^22 化 + DELAY 2^23）は監査実測 **+134 MB** — 不採択（維持）。

---

## 3. Option B 契約凍結

### 凍結式（唯一の権威）

```
hardMaxSec(sr)  := MAX_IR_LATENCY / sr          （関数 getMaximumAllowedIRLengthSecForSampleRate と一致・変更なし）
                   sr <= 0 のとき IR_LENGTH_MAX_SEC (3.0)
maxSliderSec(sr):= std::min(std::max(3.0, cur), hardMaxSec(sr))   （UI 表記上限・cur=現在値）
targetLength(sr, sec) := jmax(1, min((int)(sr × min(sec, hardMaxSec(sr))), MAX_IR_LATENCY))
```

### 変更は次の 3 点に限定（これ以外 production code 変更禁止）

- **C-1** `copySnapshotToPendingUnlocked`（StateAndUI.cpp:162-167）:
  上限を静的 `IR_LENGTH_MAX_SEC` → `getMaximumAllowedIRLengthSec(currentSampleRate)` に置換
  （setter/state 経路と同一権威に統一。noexcept 維持: consumeAtomic 既存）。
- **C-2** ControlPanel slider 上限の動的化（:87 初期・:1309-1311 updateIRInfo）:
  表記 max を `maxSliderSec(sr)` 準拠（768k で 2.73 表示・48k 以下は現行と同一挙動）。
- **C-3** `computeTargetIRLength`（StateAndUI.cpp:938-953）: cap が実際に発動した場合のみ
  loader スレッドで `juce::Logger::writeToLog("[SR-01] IR trimmed to MAX_IR_LATENCY …")`（無音→可視化）。
  戻り値・呼び出し契約は不変。

### 境界条件（凍結）

| 条件 | 契約 |
|---|---|
| SR テーブル | 44.1/48/88.2/96/176.4/192/352.8/384 kHz: hardMax ≥ 3.0 → **全経路挙動不変**。705.6k: 2.972s。768k: 2.731s（=2.7306667f。int 変換で targetLength = **2,097,151** — 2^21−1。テストは正確一致で凍結） |
| 3s request @768k | slider 上限 2.73（選択不可）、preview reject（既装）、snapshot 復元は 2.73 へ clamp + C-3 ログ（発動するのはファイル実長が target を超える trim のみ） |
| 上限超過挙動 | **trim（fade-out 付き）+ reject（UI/preview）+ warning（log）** の三段。setTargetIRLength 等の clamp は黙認せず C-2/C-3 で可視化 |
| channel count | 2ch 正規化（mono→複製、>2ch→先頭2）現行不変（LoaderThread.cpp:166-169） |
| allocation failure | NUC init false → success=false + 現行エラーメッセージ、旧 engine 保持・publish 経路単一性維持（LoaderThread.cpp:231-233）— 変更なし |
| 既存 IR ロード互換 | hardMax 未満のファイルはビット一致動作。48k "Load as-is" >3s は現行どおり許容（min(3.0,·) を clamp 式に入れない決定の根拠） |
| 永続状態互換 | Parameters.cpp:52-53 の妥当性窓 [0.5, 3.0] は**変更しない**（データ互換）。SR 固有 clamp は適用時（C-1）に実施。48k で保存した 3.0 → 768k 復元時 2.73。768k で保存した 2.73 → 48k 復元時は 2.73 のまま（引き上げない） |

### 禁止事項（work93/94 原則継承）

`DELAY_BUFFER_SIZE` / `MAX_IR_LATENCY` / RuntimeWorld ownership / Publish authority /
CrossfadeAuthority / Retire / Epoch / ISR Coordinator / Host PDC / H-01 `dryAlignDelay` /
H-02 `measureIrPeakLatencySamples` / SR-03 clamp — **すべて不変**。
検査スクリプトを旧実装に合わせるために production code を戻すことも禁止。

---

## 4. RT / ISR impact audit

- **process() 変更 0 行**（Runtime.cpp の C 系変更なし。dryAlignDelay/retarget 判定は無関係）
- 波及対象はすべて NonRT: message thread（C-1/C-2）、loader thread（C-3 ログ）
- RT lock / allocation / delete / decision 追加なし。atomic wrapper 経由のみ（既存 `consumeAtomic(currentSampleRate)` の acquire 正規パターン踏襲、HB コメント既存）
- Publish/Crossfade/Retire 単一経路: 変更なし（irLength publish は load 完了時のみ・targetLength 値域の縮小に伴う新分岐なし）
- refreshLatency / uiTotalLatencySamples / breakdown（algo+peak）: 不変（H01-C1 契約と整合）

## 5. Capacity proof

上限定数一切不変のため**新規確保ゼロ**。worst-case（cap に到達する 2^21 IR）は現行でも 48k "as-is"
ロードで到達可能経路が既にあり、NUC 層設計はサンプル数律動（SR 非依存確保）— 768k 特有の増分なし。
delayBuffer 67.1 MB（2×2^22×8B）静的・unchanged。Option A 比 +0 MB。

---

## 6. テスト契約（実装より先に凍結）— 追加ファイル: src/tests/AudioEngineHarness（H-02/SR-03/H-01 と同ファイル群で可）

| ID | 内容 |
|---|---|
| T-SR01-1 | `getMaximumAllowedIRLengthSecForSampleRate` 表計算一致（44.1k→47.55…, 48k→43.69…, 96k→21.85…, 192k→10.92…, 384k→5.46…, 705.6k→2.972…, 768k→2.7306667f, sr≤0→3.0）— 現関数の語義凍結（回帰防止） |
| T-SR01-2 | @768k: `setTargetIRLength(3.0)` / `applyAutoDetectedIRLength(3.0)` → pending == hardMax。@48k: 3.0 → 3.0 維持（不変領域） |
| T-SR01-3 | **C-1 回帰**: BuildSnapshot{target=3.0} を currentSR=768k で copySnapshotToPendingUnlocked → pending == hardMax（3.0 でない）。currentSR=48k → 3.0 のまま |
| T-SR01-4 (**必須 768k-3s**) | processing 768k で 3.0s 合成 IR ロード（H-01 駆動系踏襲・pump 方式）→ publication 成功、`irLength == 2,097,151`、trim 警告ログ発火、breakdown total=algo+peak 契約維持、process 出力 wet==dry==… H-01 アサーション非抵触 |
| T-SR01-5 | 44.1/48k 回帰: 3.0s ロードで targetLength == 144,000/147,000、trim ログ**なし**、UI 経路 reject なし |
| T-SR01-6 | 永続互換: 48k 保存 3.0 → 768k 復元で 2.73（ログ）、768k 保存 2.73 → 48k 復元で 2.73 のまま。Parameters 窓 [0.5,3.0] 受理不変 |
| T-SR01-7 | allocation failure 経路: （注入可なら）init false → 旧 engine 保持・publish 0 回・エラー文言一致 |
| T-SR01-8 | channel 変換: mono IR ロードで ch0==ch1（irL/irR 複製契約） |

## 7. 判定

```text
SR-01(B) Pre-Audit                PASS（§1-§2、残余ギャップは C-1 に局在と確定）
Capacity proof                    PASS（確保増ゼロ・上限不変）
Contract Freeze                   PASS（§3 凍結式・境界条件）
Test contract                     FROZEN（T-SR01-1..8、768k-3s 必須）
SR-01(B) Implementation Gate      GO
```

**付帯事実（本 freeze までに判明した 09-13 計画からの進展）**: setter/state/preview の SR 依存
hardMax 適用は既装済だったため、Step 4 の実作業は「applySnapshot 同期（C-1）＋UI 上限動的化（C-2）
＋trim ログ（C-3）」の 3 点に縮退。BUGLIST §2.1 の方案 B 記述「min(3.0, MAX_IR_LATENCY/sr)」は
48k "Load as-is" 経路との互換衝突により §3 の式へ修正して凍結する（監査文書の原文はそのまま残す）。
