# ConvoPeq 残件 改修計画書（2026-09-20 時点）

- **版**: v1.5（レビュー①（`doc/work113/renew_plan.md` 添付・38 観点）反映・Phase 0 測定仕様改訂版）
- **対象**: `doc/work113/residual_tasks_20260920.md` 全残件（§1〜§5）
- **方針**: 段階リリース重視・既存動作への影響を最小化・ロールバック容易性確保
- **前提**: production `src/` の未 commit 差分は 0 件、HEAD = `8f127bfe`（docs 1 件）+ `c4a08171`（B-2 test 1 件）が未 push
- **本書は read-only 監査と方針案の提示**。実装着手は本書承認後
- **基準ソース**: `ConvoPeq.md`（Generated 2026-09-19 22:08:26）および HEAD の production source
- **§2 B-1 修正方針**: 案 E（polyphase gain convention 対称化・既存 half-band FIR 維持）を「**有力仮説**」として採用（oracle C 案はユーザー監査で撤回）
- **Phase 0 着手 GO・Phase 1 以降 HOLD**: 案 E は Phase 0 の characterization（§2.6 P0 GATE）で全 PASS しなければ Phase 1 実装 GO とはしない
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** を authoritative source とする（`kCorrectPolyphaseGain` / `kUseV2Oversampler` は採用しない）

---

## v1.4 → v1.5 改訂サマリ（レビュー① 7 必須項目 + 採用指摘の反映）

レビュー①の必須修正 7 項目と、採用した追加指摘を統合。**技術的中核（案 E・原因特定）は不変**。主な変更は Phase 0 の再構造化。

| 区分 | 修正内容 | 根拠 |
|------|----------|------|
| **R1（必須1）** | **P0 gate を Baseline / Candidate / Differential の 3 層に再構築**。旧 P0-A「round-trip = 1.0」は production 無変更の Phase 0 では論理不成立（現行は 0.75 で FAIL する）ため廃止。Baseline は **record-only**、Candidate E にのみ PASS 条件を付す | §2.6 参照 |
| **R2（必須2）** | **案 E の shadow characterization を新設**。production `src/` 変更 0 のまま、test-only reference implementation（test ディレクトリ内に係数生成〜up/down を複製し `centerValue *= 2.0` を適用）で Candidate E を測定 | §2.6 Phase 0-2 |
| **R3（必須3）** | DC gain 1.0 の根拠を「測定結果」ではなく **DESIGN-CONTRACT-A（設計契約）として Phase 0-0 で明示**。根拠: interpolation-by-2 の標準 convention（zero-insert 後は両 polyphase 位相が同一 DC gain を持つ必要）+ `isSymmetricUpDown` 契約 + JUCE `dsp::Oversampling` との比較 | §2.5 Phase 0-0 |
| **R4（必須4）** | 周波数軸を `Fs_in` 一本から **stage-rate 基準へ再定義**。旧「output-side stopband = 0.55〜0.95 Fs_in」は**入力 Nyquist（0.5 Fs_in）超の tone を入力できない**ため不正。stopband/image/alias は stage-local 軸（Fs_stage）で測定 | §2.6 Phase 0-3 / §2.7 |
| **R5（必須5）** | P0 判定に**数値 tolerance を導入**（passband ±0.1 dB、stopband/image 差分 ≤0.1 dB、latency ±1 sample 等）。P0-C は「ripple = passband sweep の max−min」とスポット値に分離 | §2.6 GATE 表 |
| **R6（必須6）** | P0-G を再定義: `CustomInputOversampler` は **double 専用 API**（`AudioBlock<double>`・DSPCoreFloat は float→double 変換後に OS を通す — `DSPCoreFloat.cpp:253-263` 確認済み）のため、「float host 経路 vs double host 経路の end-to-end equivalence」とする | §2.6 Phase 0-6 |
| **R7（必須7）** | **§7.1 の残骸 `--buzz-osdirect=2` / `round-trip=` を削除**（v1.4 改訂サマリ P0-1 と自己矛盾していた）。検証は `PublishPipelineIntegrationTests` 実行 + `[OS_DIRECT] ... roundTripGain=` 観察に統一 | §7.1 |
| **R8** | 案 E の表現を厳密化: 「FIR 係数列の周波数応答（normalized shape）は不変。ただし **polyphase output の絶対振幅は変化**する。phase / group delay は理論上不変」 | §2.4 観点 3 |
| **R9** | compile-time flag の authoritative source を **CMake option** に統一（in-source `#ifndef` は safety net のみ）。Phase 2 = `-D` で ON、Phase 3-A で option 既定 ON。旧計画の「Phase 2 で build option / Phase 3 で source #define」という authoritative source の移動を廃止 | §2.5 |
| **R10** | rollback を **behavioral rollback（flag OFF + rebuild）** と **source-level rollback（`#if` ブロック削除）** に分離。「1 行 revert」単独表記を廃止 | §8 |
| **R11** | Phase 3 を **3-A（flag 常時 ON・calibration 不変）→ 3-B（新 production gain 実測）→ 3-C（calibration-only commit）→ 3-D（rigcheck 窓-only commit）** に分割。oversampler 修正と `kOutputHeadroom`（別定数: `DSPCoreDouble.cpp:593` = 0.8912509381337456）の修正を同一 commit にしない | §2.5 |
| **R12** | F-2 修正に **state propagation 手順を明示**（staging toggle → configureProbeFlatEQ → … → `waitBacklogZero` → 測定）。呼出順変更直後の即測定を禁止 | §3.1.4 |
| **R13** | F-1 移行に **semantic equivalence gate を追加**: validator の検査順序（`RuntimePublicationValidator.cpp:14-41`: SemanticConsistency → Topology → Resources → checkNoConflictingTransitions・first-fail-wins・実測確認済み）を踏まえ、移行ケースごとに「意図した failureReason に到達するか（前段 check で先に fail しないか）」を検証 | §4.2 Phase B' |
| **R14** | F-3 を「test-only」から **「production behavior の潜在欠陥（本セッション修正対象外）」に再分類**。F-4 の `ir` モードも同様（harness の挙動だが測定基準の意味変更要） | §3.2 / §9 |
| **R15** | O-1 案 A の呼称を「最小 commit」→「**B-1 regression detection に必要な最小実行可能資産**」に修正。BassBuzzMeasurement.cpp と PublishPipelineIntegrationTests.cpp の双方に触れる可能性を明記 | §1.1 |
| **R16** | parity 表記の反転を修正: 全 production taps で `centerParity=1 / convParity=0` のため、up 出力は **even 位置 = conv 位相（現行 1.0）/ odd 位置 = center 位相（現行 0.5）**。旧文の even/odd ラベルは逆 | §2.2.2 / §2.2.4 |
| **R17** | TruePeakDetector の確認結果を反映: `TruePeakDetector.cpp:284-318` の `interpolateStage` は **even/odd とも `cCoeff × sample + conv dot` で ×2 なし**。本計画の reference には**使わない**（別設計目的・計測専用経路）。Phase 0 の参考記録 or 別 work item | §2.7 教訓 9 |
| **R18** | §6 見出しの残骸「（v1.2）」を「（v1.5）」に修正 | §6 |
| **R19** | 本検証（2026-09-20 ソース監査 + NumPy 再現 + cppcheck）のエビデンスを §12 に記録 | §12 |

**レビュー①で指摘されたが本計画で変更しない点**:
- 案 E の技術的内容自体（`centerValue *= 2.0` 1 行）: レビュー①も「有力仮説として維持してよい」と判定 → 維持
- 段階リリース 5 Phase の骨格・compile-time gate の採用: 維持（authoritative source のみ R9 で CMake へ統一）
- §1〜§5 の各 GO/HOLD 判定: レビュー①も概ね妥当と評価 → 維持

---

## 0. 凡例と全体戦略

| 区分 | 着手優先度 | 影響範囲 | ロールバック |
|------|------------|----------|--------------|
| **P0 即時**（破壊的・不可逆） | なし | — | — |
| **P1 高**（影響大・段階リリース） | H-02 関連 + B-1 | 全オーディオ経路 | flag OFF rebuild |
| **P2 中**（harness / テスト cleanup） | F-1〜F-4 | test-only（F-3 は production 欠陥の記録） | ファイル revert |
| **P3 低**（運用 / 環境 / 既知制限） | O-1, B-3, D-1, D-2, R-1, R-2 | 環境 or CLI | 設定 revert |
| **別課題**（既存記録・本セッション対象外） | B-3, D-1, D-2, R-1, R-2 | — | — |

**全体戦略**:
1. **§1（O-1〜O-7）の意思決定** → コミット方針確定（ユーザー判断待ち）
2. **§2 B-1** → 大規模 DSP 修正。**Phase 0（Baseline + Shadow Candidate + Differential characterization）→ Phase 1（flag 導入）→ Phase 2（flag ON 限定検証）→ Phase 3（全段 ON + calibration 分割 commit）→ Phase 4（flag 削除）**
3. **§3 F-2** → harness cleanup（test-only）。B-1 とは独立して先に着手可能。F-3/F-4 は記録のみ
4. **§4 F-1** → テスト移行（custom main ハーネスへ）。`CrossfadeAuthority` 4 件を最優先
5. **§5 別課題** → 既存記録のまま保留。必要時に別 work item 化

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-7）

### 1.1 O-1: B-1 計測用 test-only 計装（+642 行）

| 項目 | 内容 |
|------|------|
| ファイル | `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp`（+634/−2）、`PublishPipelineIntegrationTests.cpp`（+8/−0） |
| 内容 | `eqdiag` / `eqdiagser` / `eqos<digit>` / `irwet<digit>` 診断モード、`gainpath` 行、`[EQ_DIRECT]` / `[OF_DIRECT]` / `[OS_DIRECT]` 直接駆動測定 |
| 影響 | test-only。production `src/` 変更なし |
| 判定 | 既存 `eq` 判定窓 `[0.486,0.496]` は不変（`eqIdentityMode` は `rigCheckMode == "eq"` のみで真 — `BassBuzzMeasurement.cpp:2042` 確認済み） |
| 出力キー | `[OS_DIRECT] ... roundTripGain=`（`round-trip=` ではない） |
| エントリ | `runEqDirectDriveAttribution()`（BassBuzzMeasurement.cpp:1529）経由で `PublishPipelineIntegrationTests` main から**無条件呼出し**（+8 diff 確認済み） |

**3 案**:

| 案 | 内容 | メリット | デメリット |
|----|------|----------|------------|
| **A（推奨）** | **「B-1 regression detection に必要な最小実行可能資産」を commit**（`[OS_DIRECT]` 本体 + main から `runOversamplerDirect()` を直接呼ぶエントリ配線） | 過去回帰検出を最小コストで残せる | `eqdiag` / `irwet` 診断モードは手元検証用に残る。**2 ファイル（BassBuzzMeasurement.cpp + PublishPipelineIntegrationTests.cpp）双方に触れる** |
| **B** | 全 +642 行を 1 commit | 将来 B-1 再発の即時検出資産として完全保存 | 行数が大きい。レビュー負荷大 |
| **C** | すべて退役（破棄） | 後方互換性問題なし | 再発時の解析遅延 |

**推奨案**: **A**。v1.4 の「最小 commit」は不正確 — `[OS_DIRECT]` / `[EQ_DIRECT]` / `[OF_DIRECT]` は同一エントリ関数 `runEqDirectDriveAttribution()` に束ねられており、`git add -p` だけでの分離は不可能。案 A の実体は:
1. `runOversamplerDirect()`（BassBuzzMeasurement.cpp:1293）とその呼び出し必要分を commit
2. `PublishPipelineIntegrationTests` main 側で `runEqDirectDriveAttribution()` の代わりに（または追加で）`runOversamplerDirect()` を呼ぶ配線を commit
の 2 か所。`eqdiag` 系の harness 拡張はテスト目的が局所的（OS 段分離仮説）なので手元検証後に退役検討。

**CLI 注記**: `--buzz-osdirect=2` 等の ConvoPeq.exe CLI フラグは**存在しない**。検証は `PublishPipelineIntegrationTests` 実行（`cmake --build build --config Release --target PublishPipelineIntegrationTests && build\Release\PublishPipelineIntegrationTests.exe`）で行い、stderr の `[OS_DIRECT] ... roundTripGain=` を観察する。

### 1.2 O-2: 台帳更新（`doc/work113/residual_tasks_20260919.md`）

| 項目 | 内容 |
|------|------|
| ファイル | `doc/work113/residual_tasks_20260919.md`（B-1 帰属・F-2〜F-6 追記分。+71/−1 実測） |
| 影響 | 監査記録のみ。production 影響なし |

**方針**: **O-1 と同梱 commit** または **独立 commit** のいずれかをユーザー判断。推奨は O-1 とは独立（commit 粒度の独立性確保）。

### 1.3 O-3: `ConvoPeq.md`（生成物）

| 項目 | 内容 |
|------|------|
| 状態 | `Generated: 2026-09-20 07:00:33`、test-only 計装追加後 stale |
| 方針 | **commit しない**（既存方針どおり・監査用一時ファイル・ユーザー運用） |

### 1.4 O-4: `Testing/Temporary/CTestCostData.txt`

| 項目 | 内容 |
|------|------|
| 状態 | ` D`（worktree で削除状態・tracked・numstat 0/1 実測） |
| 方針 | **現状維持**。戻す場合は `git checkout -- Testing/Temporary/CTestCostData.txt` の 1 行（ユーザー判断） |

### 1.5 O-5: `.opencode/opencode.json`

| 項目 | 内容 |
|------|------|
| 状態 | 未追跡・作成者・目的不明 |
| 方針 | **触らない**（削除も commit もしない） |

### 1.6 O-6: push（`c4a08171` + `8f127bfe`）

| 項目 | 内容 |
|------|------|
| 内容 | B-2 test 1 件 + docs 1 件 |
| 方針 | **未実施のままユーザー承認待ち** |

### 1.7 O-7: `AGENTS.md`（MIMO Desktop パイプライン運用メモ・未 commit）

| 項目 | 内容 |
|------|------|
| 状態 | ` M`（`+5/−3`・numstat 実測・MIMO Desktop パイプライン運用メモ） |
| 内容 | headroom proxy・context-mode・rtk 常時運用、3 層パイプラインの役割分担指針 |
| 影響 | 環境運用のみ。production DSP / テスト成果物への影響なし |
| 方針 | **触らない**（削除も commit もせず、現状維持。別 work item でドキュメント更新するか、ユーザー判断） |

### 1.8 §1 全体の推奨手順

```
[Step 1.1] ユーザー判断: O-1（A/B/C 案選択） + O-2（同梱 or 独立）
[Step 1.2] 該当ファイルを commit（メッセージ: "test: B-1 attribution diagnostic instrumentation ([OS_DIRECT])" 等）
[Step 1.3] O-3 は触らない・O-4/O-5/O-7 も触らない
[Step 1.4] O-6 push は独立運用操作としてユーザー承認後に実行
```

---

## 2. §2 B-1: CustomInputOversampler の up/down round-trip 欠陥（最大規模）

### 2.1 確定事項（再掲・報告書 §2.1・2026-09-20 ソース監査で再確認）

```
主因        CustomInputOversampler の up/down round-trip 利得欠陥
1 stage     round-trip 0.750000（up 1.000000 / down 0.750000・peak 基準）
多段        0.75^N（ratio 2/4/8 = 0.75 / 0.5625 / 0.421875、preset 非依存）
SoftClip局所OS 0.75（prepareSingleStage(31, 90.0) を production 引数で実測）
数学的導出  0.5×0.5 + 0.5×1.0 = 0.75
engine fit  0.98379 × 0.75^max(log2 effOS, 1) で 4 条件 ≤0.05%（doc の経験式・コード定数ではない）
契約不整合  isSymmetricUpDown / Latency の static_assert 前提と矛盾
production 変更 0 / commit 0
```

数値再現（2026-09-20 本監査・NumPy/SciPy で `prepareStage` → `interpolateStage` → `decimateStage` を逐語移植）:
- 1 段（taps=31/90dB）DC round-trip = **0.750000**
- 3 段（IIRLike 511/127/31・LinearPhase 1023/255/63）DC round-trip = **0.421875 = 0.75³**
- 案 E（center ×2 追加）では両構成とも **1.000000**

### 2.2 根本原因の特定（コード追跡・行番号は 2026-09-20 実測）

#### 2.2.1 `prepareStage`（`src/CustomInputOversampler.cpp:287-390`）

```cpp
stage.centerTap = (stage.taps - 1) / 2;        // 全 production taps で奇数
stage.centerParity = stage.centerTap & 1;      // = 1
stage.convParity = 1 - stage.centerParity;     // = 0
// ...
rawCoeffs[stage.centerTap] = 0.5;              // ← center 一個の重み = 0.5（335/348 行）
// ...
scale = 0.5 / nonCenterSum;                    // ← 非 center 合計 = 0.5 に正規化（341 行）
// ...
stage.convCount = (stage.taps - stage.convParity + 1) / 2;
stage.convCoeffs[r] = rawCoeffs[convParity + 2r];  // convParity タップ抜き出し（362-363 行）
```

**FIR 合計**: center (0.5) + 非 center (0.5) = **1.0**（normalize 済み）
**convParity タップ合計**: **0.5**（half-band FIR の偶奇分離。同 parity の非 center は 319-323 行でゼロ化済みのため構造的に保証）

#### 2.2.2 `interpolateStage`（`src/CustomInputOversampler.cpp:492-568`）

```cpp
double centerValue = stage.centerCoeff * history[idx - stage.centerDelayInput];
// ...
double convValue = Σ stage.convCoeffsReversed[r] * xWindow[r];  // = 0.5
// ...
convValue *= 2.0;                                // ← ★ conv 位相のみ ×2 補正（557 行）

output[outBase + stage.convParity]   = convValue;    // even 位置 = 1.0 × input（convParity=0）
output[outBase + stage.centerParity] = centerValue;  // odd 位置  = 0.5 × input（centerParity=1）
```

**up の出力（1 入力 → 2 サンプル・全 production taps で parity 固定）**:
- even 位置（conv パス）= 0.5 × 2 = **1.0**
- odd 位置（center パス）= **0.5**
- 平均 = **0.75** / 合計 = 1.5

（v1.4 以前の「偶数=center/奇数=conv」表記は parity 反転 — v1.5 で修正。平均 0.75 の結論は不変）

#### 2.2.3 `decimateStage`（`src/CustomInputOversampler.cpp:570-723`）

```cpp
double acc = stage.centerCoeff * centerSample;     // 0.5 × history[base - centerTap]（658 行）
// ...
acc += Σ coeffs[r] * history[base - convParity - 2r];  // stride-2 FIR = 0.5 × history[even]
// ...
output[n] = acc;                                  // ← ★ ×2 補正なし（717 行）
```

**down の出力（2 入力 → 1 サンプル）**: center 寄与 0.5 + conv 寄与 0.5 = **1.0**

#### 2.2.4 round-trip 整合式

```
up:   even=1.0 (conv パス ×2) + odd=0.5 (center パス) → 平均 0.75
down: center=0.5 (odd 側読み) + conv=0.5 (even 側読み) = 1.0
round-trip = 0.75 × 1.0 = 0.75   （報告書「down 0.75」は peak 基準の downGain 表記・数学的に同一）
```

**多段（ratio 8 = 3 stages）**: round-trip_3 = 0.75³ = 0.421875 = −7.5 dB
**SoftClip 局所 OS** も `prepareSingleStage(31, 90.0)` で同一経路を通るため 0.75。

### 2.3 修正案の比較（確定）

#### 2.3.1 案 A/B/C/D の再評価

| 案 | 内容 | 評価 |
|----|------|------|
| A | `decimateStage` に ×2 追加 | **不採用**：round-trip = 1.125 で過剰補正 |
| B | `interpolateStage` の ×2 削除 | **不採用**：round-trip = 0.375 で過少 |
| C | `prepareStage` を center=1.0 / non-center=0.0 に変更 | **撤回（ユーザー監査で誤り確定）**：`h[n] = δ[n-center]` の pure delay 化、`convCoeffs = 0` → zero-stuffing 構造、anti-alias/anti-imaging 消失 |
| D | A+B 統合 | **不採用**：offset 残・`isSymmetricUpDown=true` の意味的整合に難 |

#### 2.3.2 案 E（有力仮説として採用）

**案 E: polyphase gain convention 対称化** — 既存 half-band FIR を維持し、`interpolateStage` の **両 polyphase に ×2 を適用**:

```cpp
// src/CustomInputOversampler.cpp interpolateStage() 内（567 行付近）
// 現行: center phase × 1, conv phase × 2
// 修正後: center phase × 2, conv phase × 2（対称化）

double convValue = ...           // = 0.5 × input（half-band FIR convParity タップ合計）
double centerValue = ...;       // = 0.5 × input（center タップ）
convValue *= 2.0;              // 既存（557 行）
centerValue *= 2.0;             // ★ 新規追加（1 行）

output[outBase + stage.convParity]   = convValue;    // even 位置 = 1.0 × input
output[outBase + stage.centerParity] = centerValue;  // odd 位置  = 1.0 × input
```

**案 E 採用時の計算**:
- up: even=1.0（conv 0.5 × 2）、odd=1.0（center 0.5 × 2）、平均=1.0、合計=2.0
- down: 既存通り center=0.5、conv=0.5、合計=1.0（**変更なし**）
- **constant-signal / DC gain round-trip = 1.0 × 1.0 = 1.0**

**厳密な表現（v1.5・レビュー① §4 反映）**:
- FIR **係数列そのものの周波数応答（normalized shape）は不変**。ただし **polyphase output のスケーリング変更により interpolation branch の絶対振幅は変化する**
- phase / group delay は理論上不変（係数位置不変・純粋な振幅スケーリング）
- 一般信号での perfect reconstruction (PR)（H_up(z) × H_down(z) = z^(-L) の全帯域保証）は案 E だけでは**自動的に証明されない**。「perfect reconstruction」表現は Phase 0 で数値確認されるまで使用しない
- 2026-09-20 NumPy 再現の補助知見: 入力帯域内の f ≤ 0.25 Fs_in では candidate/baseline 比 ≈ (4/3)^N（一様スケール）だが、**0.45 Fs_in 近傍では帯域端 alias 混合により比率が非一様になる**（実測例: 1 段で 0.410 → 0.660）。→ 帯域端の再構成挙動は変わる。Phase 0 の実測（production ブロック連続処理）が唯一の確定手段

**gain 変化**:
- 1 stage: 0.75 → 1.0 = **+2.4988 dB**（≈ +2.5 dB）
- 2 stages: 0.5625 → 1.0 = **+4.998 dB**（≈ +5.0 dB）
- 3 stages: 0.421875 → 1.0 = **+7.4988 dB**（≈ +7.5 dB）

**案 E の本質**: FIR の形状（center=0.5 + non-center 合計=0.5、parity 間引き済み）は維持。**polyphase のゲイン分配**のみを `×1/×2` から `×2/×2` に修正する。

#### 2.3.3 案 E vs 案 C の対比（ユーザー監査の根拠）

| 観点 | 案 C（撤回） | **案 E（採用）** |
|------|-------------|-----------------|
| FIR 形状変更 | **大変更**：half-band FIR 構造を破壊（anti-aliasing 消失） | **無変更**：既存 FIR 維持 |
| 結果 FIR | `h[n] = δ[n-center]`（pure delay） | 既存 half-band FIR（sinc × Kaiser × parity 間引き） |
| anti-aliasing / anti-imaging | **消失** | **維持** |
| `convCoeffs = 0` | 0 になる | 0.5 のまま維持 |
| `interpolateStage` 変更 | なし | **1 行追加**（`centerValue *= 2.0`） |
| `decimateStage` 変更 | なし | **なし** |
| round-trip DC gain | 1.0 | **1.0** |
| エイリアシング再評価 | 必要 | **不要**（係数不変） |
| 修正行数 | 約 5 行（prepareStage normalize） | **1 行**（interpolateStage 1 行） |
| rollback | 5 行 revert | **flag OFF rebuild（第一手段）/ `#if` ブロック削除（source 級）** |

### 2.4 8 観点への影響（v1.5・案 E）

| 観点 | 影響 | 対応 |
|------|------|------|
| 1. decimateStage 側の補正 | **不要** | 案 E は `decimateStage` に変更なし |
| 2. up/down の gain convention 対称化 | 中（gain convention のみ変更） | `interpolateStage` に `centerValue *= 2.0` を追加（1 行） |
| 3. `isSymmetricUpDown` の意味 | **FIR 係数列は不変** → normalized shape 不変・絶対振幅は変化・phase/group delay は理論上不変 | Phase 0 で絶対振幅と shape を分離実測 |
| 4. `AudioEngine.Processing.Latency.cpp` | **1 段あたり往復 `(taps−1)` サンプル（stage レート・up+down 合算）**、base レート換算は `× (baseRate / stageRate)`。コード実装どおり（`groupDelaySamplesAtStageRate = taps[stage] - 1`） | Phase 0 判定基準 F で impulse 実測 |
| 5. 既存 calibration / rigcheck（窓 `[0.486,0.496]`） | 再校正必要（後段・HOLD） | gain +2.5 dB/段。Phase 3-C/D で分割 commit |
| 6. SoftClip 分岐（`effOS==1`）と主 OS 分岐（`effOS>1`）の双方 | 同一修正で両者対応 | `prepareSingleStage(31, 90.0)` も同経路（Phase 0 判定基準 I） |
| 7. Float host / Double host 経路 | **oversampler 自体は double 専用**。Float host は float→double 変換後に同一 double OS を通す | float-host vs double-host の end-to-end equivalence として検証（Phase 0-6） |
| 8. 既存 regression への波及 | 全 regression 期待値要再校正（HOLD） | work57 の NUC、outputStage gain、`kOutputHeadroom` 等。Phase 3 まで保留 |

### 2.5 段階リリース計画（v1.5・authoritative source = CMake option）

```
[Phase 0] Characterization（read-only・production src/ 変更 0）
  - 構造は §2.6（Phase 0-0 〜 0-6 + GATE）
  - Baseline（現行・record-only）+ Shadow Candidate E（test-only reference）+ Differential
  - 全 PASS → Phase 1 eligibility

[Phase 1] Feature Flag 導入（compile-time・破壊なし）
  - CMake option を authoritative source にする:
    ```cmake
    # CMakeLists.txt（CustomInputOversampler.cpp をコンパイルする全ターゲットへ適用）
    option(CONVOPEQ_CORRECT_POLYPHASE_GAIN
           "Correct polyphase gain convention (B-1 案E)" OFF)

    target_compile_definitions(<対象ターゲット> PRIVATE
        CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${CONVOPEQ_CORRECT_POLYPHASE_GAIN}>)
    ```
  - in-source は safety net のみ（authoritative source は CMake）:
    ```cpp
    // src/CustomInputOversampler.cpp 冒頭（#include 後）
    #ifndef CONVOPEQ_CORRECT_POLYPHASE_GAIN
    #define CONVOPEQ_CORRECT_POLYPHASE_GAIN 0   // safety net・通常は CMake option が供給
    #endif

    // interpolateStage() 内
    #if CONVOPEQ_CORRECT_POLYPHASE_GAIN
        centerValue *= 2.0;
    #endif
    ```
  - 既定 OFF（既存挙動維持）。runtime flag は **不採用**（RT path の毎 sample/毎 block 分岐を避ける）
  - Phase 2 で `cmake -DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` により rebuild（authoritative source が Phase を通じて移動しない — レビュー① §19 採用）

[Phase 2] Flag ON で限定検証
  - `-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` で rebuild
  - 内部 dogfood: ratio 1/2/4/8 × preset IIRLike/LinearPhase 全段検証
  - Phase 0 の Shadow Candidate 測定値と**直接比較**（同一測定系・全周波数点で一致すること。帯域端の非一様差分を含め ±0.02 dB 以内を目安）
  - rigcheck 判定窓・経験式 0.75^N の再判定（必要なら Phase 3-C/D で対応）

[Phase 3] 全段 ON（破壊的変更・commit 分割）
  - **Phase 3-A**: CMake option の既定値を ON に変更（**calibration は一切変更しない** 1 commit）
  - **Phase 3-B**: 新 production gain を実測（characterization report のみ・commit なし）
  - **Phase 3-C**: calibration 値のみの commit（`kOutputHeadroom` 等が必要な場合のみ。oversampler 修正と混在させない）
  - **Phase 3-D**: rigcheck 判定窓 `[0.486, 0.496]` 更新のみの commit
  - 全テスト + 新 rigcheck 窓で PASS を確認

[Phase 4] flag 削除（次メジャーリリース）
  - `CONVOPEQ_CORRECT_POLYPHASE_GAIN` option / macro / `#if` 分岐を削除し修正行を恒久化
```

**重要（v1.5）**:
- `0.98379 × 0.75^N` は `residual_tasks_20260919.md:79` の**経験式**であり、production コードに `0.98379` / `0.75^N` / `effOS` 定数は**存在しない**（全域 grep で確認済み）。`kOutputHeadroom = 0.8912509381337456`（−1.0 dBFS・`DSPCoreDouble.cpp:593` / `NoiseShaperLearner.cpp:18`）は別定数。Phase 3 の「0.75^N 削除」は **rigcheck 判定窓・監査記録・経験式記述の更新**と読み替える（HOLD）
- architectural boundary: 案 E は `interpolateStage()` 内の純粋な数値変更であり、Publish / RuntimeWorld / Retire / Crossfade / ISR bridge / lifetime には触れない（Practical Stable ISR Bridge Runtime の原則と整合）

### 2.6 Phase 0 characterization（v1.5 新構造・production src/ 変更 0）

**レビュー① §5/§6/§33/§34 の最大修正点**: 旧 P0-A「round-trip = 1.0」は現行 production（0.75）を PASS 条件にしており論理不成立。Phase 0 を以下の 4 段構造に再構築する。

```
Phase 0-0  source/contract audit
    DESIGN-CONTRACT-A を確定する（測定ではない・文書作業）
    ─────────────────────────────────────────────────
    DESIGN-CONTRACT-A: Oversampler は interpolation convention として
    両 polyphase 位相が同一 DC gain（= 2 倍密化の gain convention）を持ち、
    up/down round-trip の DC/passband gain が unity であることが要求される。
    根拠:
      (1) interpolation-by-2 の標準 convention（zero-insert で密度 2 倍 →
          polyphase 全体を ×2 convention に合わせる）
      (2) `CustomInputOversampler.h:21-22` の isLinearPhaseFIR / isSymmetricUpDown
          宣言と `AudioEngine.Processing.Latency.cpp:6-8` の static_assert が
          「対称」を前提とした契約を課している
      (3) down 側 half-band decimator は既に DC gain 1.0（up だけ 0.75 で非対称）
      (4) JUCE `dsp::Oversampling` の polyphase 実装パターンとの比較（文献調査）
    結論: baseline（0.75）は contract（1.0）に対する不整合 = defect の根拠資料となる

Phase 0-1  P0-BL: Baseline characterization（record-only・PASS/FAIL なし）
    production 無変更のまま現行実装を測定し記録する
    期待（sanity gate・一致しなければ測定系の誤り）:
      DC round-trip = 0.75 / 0.5625 / 0.421875（ratio 2/4/8・preset 非依存・±1e-6）

Phase 0-2  P0-CAND: Shadow Candidate E characterization
    production `src/` 変更 0 のまま、test ディレクトリ内の
    test-only reference implementation（prepareStage/interpolate/decimate を複製し
    `centerValue *= 2.0` を適用）で Candidate E を測定する
    期待: DC round-trip = 1.0 ± 1e-6（全構成）
    注: 2026-09-20 の NumPy 再現で本値は既に予備確認済み（§2.1）。
        C++ test-only 実装で production と同一コード経路（juce::FloatVectorOperations
        等）を通して正式化する

Phase 0-3  frequency transfer characterization（stage-rate 軸）
    ── 周波数軸の定義（v1.5・R4）──
    [A] Full-chain round trip（入力帯域内・Fs_in 基準）:
        50 Hz / 1 kHz / 10 kHz / 0.25 Fs_in / 0.45 Fs_in / 0.49 Fs_in
        ※ 入力は 0.5 Fs_in 超の tone を持てない。旧「0.55〜0.95 Fs_in input sweep」
          は不正（alias した成分としてしか観測されない）→ 廃止
    [B] Stage-local interpolation（各 stage・Fs_stage 基準）:
        0.01 / 0.10 / 0.25 / 0.45 / 0.49 / 0.50 / 0.51 / 0.55 Fs_stage
        （ratio 8 なら stage 0/1/2 の 3 本をそれぞれ測定）
    [C] Stage-local decimation（高レート側・Fs_stage 基準）:
        0.50 〜 0.95 Fs_stage の sweep（ここで初めて alias rejection を
        意味のある形で測定できる）
    ── P0-C passband ripple の定義（レビュー① §12 反映）──
        ripple は 2 点測定ではなく、passband = [0, 0.45 Fs_in] の sweep における
        |H(f)|dB の max − min として定義。0.25/0.45 Fs_in のスポット値は別枠で記録

Phase 0-4  block/reset characterization（P0-H 具体化）
    同一 input stream を以下の block 分割で処理し output waveform を比較:
      A: 4096 × 1
      B: 1024 × 4
      C: 256 × 16
      D: 513 + 777 + 1000 + … の非一様分割
    加えて reset() 後の初回 block と continuous processing の差を確認
    （CustomInputOversampler は upHistory/downHistory を保持し、reset()/
      clearAllStages() がこれらを clear する — 実測実装確認済み）

Phase 0-5  SoftClip local OS characterization
    prepareSingleStage(31, 90.0) について以下を分離:
      I-a: round-trip DC gain / sine amplitude（candidate = 1.0 が gate）
      I-b: SoftClip 動作点: 案 E は SoftClip へ入る信号レベルを +2.5 dB 相当
           変え得るため「oversampler が unity」 と「SoftClip の動作点が意図位置」
           は別問題として扱う。upPeak / downPeak / 閾値への到達率を
           baseline / candidate 双方で記録（閾値再校正は Phase 3-C の归属）

Phase 0-6  float-host / double-host end-to-end equivalence
    （レビュー① §16 反映。oversampler 自体は double 専用）
      float host: float → double 変換 → double DSP / oversampler → float output
      double host: double → double DSP / oversampler → double output
    の end-to-end equivalence を測定（tolerance は GATE 表参照）
```

#### 2.6.1 B-1-P0 GATE（v1.5・絶対値 + 差分の二重 gate）

Baseline は record-only。PASS/FAIL は **Candidate vs CONTRACT**（絶対値）と **Candidate vs Baseline の差分帰属**（differential）の 2 系統で判定する。

| ID | 判定対象 | 判定条件（数値 tolerance 付き） | 合否 |
|----|----------|--------------------------------|------|
| **G-0** | Phase 0-0 | DESIGN-CONTRACT-A が文書化・承認済み | □ PASS / □ FAIL |
| **G-BL** | Baseline sanity | 現行 DC round-trip = 0.75^N（±1e-6・測定系 sanity。逸脱時は Phase 0 を中断し測定系を見直す） | □ PASS / □ FAIL |
| **P0-A** | Candidate DC | Shadow Candidate E の DC round-trip = **1.0 ± 1e-6**（ratio 2/4/8 × preset 全構成） | □ PASS / □ FAIL |
| **P0-B** | Low-freq passband（絶対） | Candidate: 50 Hz / 1 kHz で unity ± **0.1 dB** | □ PASS / □ FAIL |
| **P0-C** | Passband | Candidate: [0, 0.45 Fs_in] sweep の **ripple ≤ 0.2 dB** + 0.25 / 0.45 Fs_in スポット値を記録（絶対 unity ±0.1 dB は f ≤ 0.25 Fs_in に適用） | □ PASS / □ FAIL |
| **P0-C'** | Differential（帰属） | Candidate/Baseline 比が f ≤ 0.25 Fs_in で **(4/3)^N ± 0.05 dB** に一致（center-phase ×2 のみで説明可能）。0.45 / 0.49 Fs_in は差分値を記録し、Phase 2 flag-ON 測定が shadow 値と **±0.02 dB** で一致すること | □ PASS / □ FAIL |
| **P0-D** | Stopband（stage-local） | Stage-local decimation 0.50〜0.95 Fs_stage sweep: Candidate の attenuation が Baseline と **各測定点で差 ≤ 0.1 dB** | □ PASS / □ FAIL |
| **P0-E** | Image / alias（stage-local） | Stage-local interpolation の image band（0.5 Fs_stage 近傍〜）: rejection の Candidate/Baseline 差 ≤ **0.1 dB** | □ PASS / □ FAIL |
| **P0-F** | Latency | **測定方法を明示**: impulse 応答の peak 位置が `Σ (taps[s]−1) × (baseRate/stageRate)` と一致（許容 **±1 sample**）。group delay centroid を cross-check（±0.05 sample 目安）。static_assert は設計前提の宣言であり latency contract の PASS とは**別物**であることを明記 | □ PASS / □ FAIL |
| **P0-G** | Float / Double host | Float host 経路（float→double 変換後）と Double host 経路の end-to-end 出力が **float 量子化誤差内（目安 ≤ −120 dBFS 相対）** で振幅・位相 equivalent | □ PASS / □ FAIL |
| **P0-H** | Block / reset boundary | §2.6 Phase 0-4 の 4 分割（A〜D）出力が **double 経路で bitwise 同等（または相対誤差 ≤ 1e-12）**・reset() 後初回 block と continuous の差が規定内 | □ PASS / □ FAIL |
| **P0-I** | SoftClip local OS | I-a: `prepareSingleStage(31, 90.0)` の round-trip（taps=31, center=15, centerParity=1, convParity=0 確認済み）で Candidate DC = 1.0 ± 1e-6。I-b: SoftClip 動作点影響（+2.5 dB 相当）を upPeak/downPeak/閾値到達率で記録 | □ PASS / □ FAIL |

**全 PASS → Phase 1 eligibility**。いずれか FAIL → 案 E を見直し、案 D 統合や tap 再設計に方針変更。

### 2.7 Phase 0 測定対象（v1.5・stage-rate 分離）

```
[A] Full-chain round trip（Fs_in 基準・入力帯域内のみ）:
      50 Hz / 1 kHz / 10 kHz / 0.25 Fs_in / 0.45 Fs_in / 0.49 Fs_in

[B] Stage-local interpolation（Fs_stage 基準・stage ごと）:
      0.01 / 0.10 / 0.25 / 0.45 / 0.49 / 0.50 / 0.51 / 0.55 Fs_stage
      （ratio 8 は stage 0/1/2 の 3 本を独立測定）

[C] Stage-local decimation（Fs_stage 基準・高レート側から）:
      0.50 〜 0.95 Fs_stage の sweep
```

Input Nyquist（0.5 Fs_in）・stage rate（2^k × Fs_in）・image band の各軸を**分離**することで、案 E の「FIR 係数列不変 → normalized shape 不変」仮説を厳密に検証する。

### 2.7 教訓・監査記録（v1.5）

1. **oracle の「教科書の答え」は現行実装と一致しない場合がある**: C 案は half-band FIR の教科書条件としては正しいが、既に half-band FIR で構築された実装に適用すると pure delay に退化する
2. **「gain convention 不整合」と「FIR 構造不整合」を分離する**: 修正は polyphase 配分の対称化に限定し、FIR 構造は維持する
3. **engine fit の `0.75^N` は即時削除不可（HOLD）**: 経験式であり production 定数ではない。Phase 0/3 で再判定
4. **Phase 0 を必ず先行**: Baseline / Candidate / Contract / Differential の 4 構造で「defect か意図された convention か」を確定する。**測定だけでは design intent は証明できない**（契約・既存テスト・calibration・基準実装との照合が必要）
5. **「案 E = 仮説」**: 「正しいことが証明された修正」ではなく「gain imbalance を最小変更で修正する有力仮説」
6. **「perfect reconstruction」の表現に注意**: DC gain 1.0 のみ現状確定。帯域端（0.45 Fs_in 近傍）の再構成挙動は変わる（NumPy 予備測定で確認）
7. **「anti-aliasing 維持」はコード変更のみから断定不可**: Phase 0-D/E の stage-local 実測で確認
8. **feature flag は compile-time gate（authoritative source = CMake option）**: RT path の runtime 分岐を避け、flag の真実の情報源をビルド設定に固定する
9. **姉妹実装の gain convention 差異（v1.5・コード確認済み）**: `src/TruePeakDetector.cpp:284-318` の `interpolateStage` は **even/odd とも `cCoeff × sample + conv dot` で ×2 補正なし**。TruePeakDetector は True Peak 計測専用経路（main OS を経由しない）のため影響範囲外。**ただし TruePeakDetector の convention を案 E の正解とみなすことは避ける**（別設計目的）。Phase 0 の参考測定として round-trip DC gain を実測・記録するか、別 work item として残置

### 2.8 B-1-P0 GATE の運用規則（v1.5・production 0 変更 gate）

```
B-1-P0 GATE
────────────────────────────────────
production src/:
    modified = 0
    staged   = 0

measurement/harness:
    test-only additions permitted
    （Phase 0-2 の test-only reference implementation を含む）

commit:
    forbidden

calibration:
    forbidden

threshold update:
    forbidden

engine-fit 0.75^N（経験式・監査記録）:
    unchanged
────────────────────────────────────
```

Phase 0 完了時の PASS/FAIL は §2.6 GATE 表（G-0 / G-BL / P0-A〜I）に記録する。

---

## 3. §3 harness / production 潜在欠陥（F-2/F-3/F-4）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC が意図と一致しない

#### 3.1.1 根本原因（2026-09-20 実測）

```cpp
// src/audioengine/AudioEngine.h:1416-1432（1425 行）
void setAutoGainStagingEnabled(bool enabled) noexcept
{
    const bool current = convo::consumeAtomic(autoGainStagingEnabled, ...);
    if (current == enabled) return;          // ← 既に同じ値なら AGC に触れない
    // ...
    getEQProcessor().setAGCEnabled(!enabled);  // ★ ON→OFF 遷移時のみ AGC が true に再設定される
    submitRebuildIntent(...);
}
```

- 既定値: `autoGainStagingEnabled { true }`（AudioEngine.h:2626 実測）
- `setEQAGCEnabled`（1306）と `getEQProcessor()`（1292）は同一オブジェクト `uiEqEditor`（EQEditProcessor）を経由する → `configureProbeFlatEQ` 内 `setEQAGCEnabled(false)`（BassBuzzMeasurement.cpp:1081）が staging toggle で上書きされる
- `eq` モードの呼出順（BassBuzzMeasurement.cpp:1799/1803）: `configureProbeFlatEQ` → `setAutoGainStagingEnabled(false)` → **AGC が true に再設定される**
- `eqdiag` 系（1833/1834）は逆順で最終 AGC=OFF — 修正案 A はこのパターンに揃える

#### 3.1.2 修正案

| 案 | 内容 | 評価 |
|----|------|------|
| **A（推奨）** | 呼出順を逆に: `setAutoGainStagingEnabled(false)` → `configureProbeFlatEQ(e)` | 最小変更・eqdiag パターンと整合 |
| B | `setAutoGainStagingEnabled` から `setAGCEnabled(!enabled)` を削除 | 設計変更（Bug#4 の前提が崩れる） |
| C | `configureProbeFlatEQ` の最後に staging OFF を追加（順序固定） | コード複雑化 |

**推奨**: **A**。

#### 3.1.3 検証手順（v1.5・R12・state propagation を明示）

```
1. e.setAutoGainStagingEnabled(false)   // staging OFF（既定 true → 遷移が発生）
2. configureProbeFlatEQ(e)              // 最終的に AGC OFF を固定
3. e.setEQFilterStructure(EQProcessor::FilterStructure::Parallel)
4. e.setEqBypassRequested(false)
5. waitBacklogZero(e, 30000)            // ★ rebuild intent の完了待ち（staging toggle が
                                        //   submitRebuildIntent を発行するため必須）
6. sleepPump(2000)
7. gainpath 行で staging=0 eqAGC=0 を確認
8. rigcheck 測定 → ratio 再測定（窓 [0.486,0.496] 再確認）
```

- **窓 `[0.486,0.496]` は数学的に必ず不変とは言わない**: F-2 は EQ AGC の実効状態を変える修正。AGC ON 状態での既存窓整合は実測で再確認
- **B-1 の原因切り分けとは独立**: AGC を強制 OFF にした診断モードでも ratio 0.4912 が不変（監査記録済み）のため、F-2 修正は B-1 の根本原因に影響しない
- 窓境界が ±ε を超えて変動する場合は判定窓再校正を Phase 1 の前段で実施

### 3.2 F-3: EQ dry/wet 混合の潜在欠陥（v1.5・R14 再分類）

#### 3.2.1 確認事項（2026-09-20 実測）

`src/eqprocessor/EQProcessor.Processing.cpp:978-1015`（ブレンド本体。970 起点は直前の gain ramp を含む）:

```cpp
if (bypassTransitionActive)
{
    const bool canBlendDry = (dryCopyBase != nullptr);
    for (sampleIndex...)
    {
        if (canBlendDry)
            wetPtr[n] = wetPtr[n] * wetGainState + dryValue * dryGain;  // ← 正常
        else
            wetPtr[n] = wetPtr[n] * wetGainState;  // ← dry 補償なし（wet-only 減衰）
    }
}
```

`dryCopyBase` は `bypassTransitionActive && dryBypassBuffer 容量十分` のときのみ充填される（570-582 行）。

#### 3.2.2 影響評価

- **定常状態の B-1（−5.11 dB）の原因ではない**（定常では `bypassTransitionActive=false`）
- 遷移時のみの潜在欠陥
- **分類上の注意（v1.5）**: 本件は **production behavior の欠陥**（`EQProcessor.Processing.cpp` 内）であり「test-only」ではない。本セッションでは修正しない（修正対象 = false・実装対象 = なし）

#### 3.2.3 修正案（将来 work item）

| 案 | 内容 | 評価 |
|----|------|------|
| A | dry バッファ確保失敗時は遷移を遅延（最大 N ms）してリトライ | 複雑・RT 影響 |
| B | dry バッファ確保失敗時は無音フェード（mute crossfade） | 聴感変化小 |
| C | ドライコピー失敗時は事前コピー（process 開始前に確保済みに） | prepareToPlay で保証 |
| **D（推奨）** | **記録のみ**。B-1 と分離して別 work item | 既存動作変更なし・リスク最小 |

### 3.3 F-4: `--buzz-rigcheck=ir` の convolver 未有効化

#### 3.3.1 根本原因（2026-09-20 実測）

`BassBuzzMeasurement.cpp:1744-1762`:
```cpp
e.setEqBypassRequested(true);
e.setConvolverBypassRequested(true);   // ← 1745: 一律 ON
// ...
if (rigCheckMode == "ir")
{
    e.getConvolverProcessor().loadImpulseResponse(irFile, false);
    // ← setConvolverBypassRequested(false) が無い → 出力は dry コピー
}
```

`irwet<digit>` モード（1763-1792）は 1778 行目で `setConvolverBypassRequested(false)` を実行済み。

#### 3.3.2 修正案

| 案 | 内容 | 評価 |
|----|------|------|
| A | `ir` モードに `setConvolverBypassRequested(false)` を追加 | 既存窓 `[0.880,0.897]` の意味が変わる（dry → wet） |
| **B（推奨）** | `irwet<digit>` で wet 対照をカバー済みとし、`ir` は記録のみ（既存 dry 測定基準を維持） | 測定基準の意味変更なし |

**推奨**: **B**。`ir` モードの wet 有効化は将来別 work item。

---

## 4. §4 F-1: PublicationValidatorIsolationTests 移行・退役

### 4.1 現状（2026-09-20 実測）

- `src/tests/PublicationValidatorIsolationTests.cpp`（**519 行**・TEST_F 34 + TEST 4 = 38 ケース）
- **CMake 未登録**（ルート CMakeLists.txt `add_executable` 39 ターゲットのいずれにも該当なし。evidence 配下の 3 CMakeLists も含め全 4 ファイルに未登録）
- **gtest 使用は本ファイルのみ**（`grep -rln 'gtest/gtest.h' src` = 1 ファイル）
- `validator_.checkNoConflictingTransitions(...)` を 135/246/255/265/273 行ほか **9 箇所**で呼ぶが、現行 header（`RuntimePublicationValidator.h:101`）では `private`
- `FRIEND_TEST` 0 件 → **コンパイル不可**
- `tools/build-debug.bat:29` の `--target PublicationValidatorIsolationTests` は stale（**2026-06-03 commit `85ac377c` で CMake から削除済み・git -S で日付確認**）

### 4.2 移行方針（v1.5・MIGRATE CASES THEN RETIRE）

#### Phase 0: Case Classification（機械的移植の前に実施）

34 ケースを 4 軸で分類:

| 分類軸 | 判定 |
|--------|------|
| **A. 公開 API coverage** | 公開 API（`validatePublication` / `validateSemanticConsistency` / `validateTopology` / `validateResources`）で再現可能か |
| **B. semantic equivalence** | 「同じ invalid state を作れば同じ failureReason に到達するか」（R13・レビュー① §27） |
| **C. error contract** | **error category（`ValidationFailureReason` enum）を第一優先契約**。error string は参考情報（ログ用）で契約としない |
| **D. 優先度** | `CrossfadeAuthority` 4 件（直接カバレッジ維持）/ validator 25 件 / `CheckTransition_*` 系 9 件 |

**実測分類（v1.4 で 25/9 に修正・2026-09-20 再実測で一致）**:
- `ValidatePublication_*`（7）/ `ValidateSemanticConsistency_*`（1）/ `ValidateTopology_*`（6）/ `ValidateResources_*`（11）= **25 件**
- `CheckTransition_*`（8）/ `CheckNoConflictingTransitions_*`（1）= **9 件**

#### Phase A: `CrossfadeAuthorityRegressionTest` 4 ケース移植（最優先・API 不変）

| 移行先 | 内容 |
|--------|------|
| 既存カスタムハーネス（`AudioEngineHarness` 系・custom main・gtest 非依存） | `Decision{needsCrossfade, fadeTimeSec}` / `evaluate(old, new, policy)` の 4 ケース（DeterministicDecision / PolicyChangeChangesDecision / SameStructuralHashNoCrossfade / OversamplingChangeTriggersCrossfade） |

#### Phase B: validator 25 ケース移植（Phase 0 分類後）

| ケース種別 | 移行方法 |
|------------|----------|
| `Validate*` 系 25 件 | 公開メソッド直接使用。`world` を構築してテスト。**error contract は `ValidationFailureReason` enum** |
| `Check*` 系 9 件 | `checkNoConflictingTransitions` が private のため **`validatePublication` 経由**で `result.failureReason` enum 検証に書き換え |

#### Phase B': semantic equivalence gate（v1.5・R13 新設）

private API → public API 移行では「旧テスト PASS」だけでは不十分。**ケースごとに**次を確認する:

```
旧 private-path での期待 failure
   ==  新 public-path での result.failureReason
```

**根拠（2026-09-20 実測）**: `validatePublication` は `RuntimePublicationValidator.cpp:14-41` で
`SemanticConsistency → Topology → Resources → checkNoConflictingTransitions` の順に検査し **first-fail で return** する。したがって:
- 旧テストが private check（例: `checkNoConflictingTransitions`）の failure を狙っていても、public API では **前段の Topology/Resources が先に fail** して別の failureReason が返る可能性がある
- 移行手順: 各ケースの world を「前段 check をすべて PASS させる状態」に構築し直すか、期待 failureReason を前段到達形に書き換える。ケース別の対応表（intended check / 前段状態 / 到達 failureReason）を移行時に残す

#### Phase C: テストファイル退役

| 項目 | 内容 |
|------|------|
| `PublicationValidatorIsolationTests.cpp` 削除 | 移行完了後 |
| `tools/build-debug.bat:29` 修正 | stale 行を除去（`--target PublicationValidatorIsolationTests` を別ターゲットへ） |
| 関連 CMake / presets | 既に未登録のため変更なし |

### 4.3 実装パターン（既存 custom main() への統合）

```cpp
// 例: RuntimePublicationValidator 移行先（既存カスタムハーネス）
// error category を契約とする（string ではない）
// EXPECT_* 相当は custom harness の assert マクロに置換
bool TestValidatePublication_SemanticConsistency_Success()
{
    RuntimePublicationValidator validator;
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 100;
    // ... (元のテストの world 構築をコピー)

    const auto result = validator.validatePublication(world);
    CHECK(result.isValid);
    CHECK(result.failureReason == ValidationFailureReason::None);
    // errorMessage 文字列は contract 化しない（実装詳細の安定性のため）
    return true;
}
```

カスタム `main()` であれば `<gtest/gtest.h>` 依存は不要。

### 4.4 影響度

- test-only 修正（production `src/` 変更なし）
- validator 動作の直接カバレッジは減少するが、`RuntimePublicationBridge` 経由（`AudioEngine.h:3666-3670` の `validator_->validatePublication(world)`）の間接カバレッジは維持
- `CrossfadeAuthority` 4 ケースの移植で直接カバレッジ維持
- **error category 契約**により error string の揺れに対するテスト安定性が向上
- Phase B' の semantic equivalence gate により、移行時の「failureReason 空振り」リスクを排除

---

## 5. §5 別課題（既存記録・本セッション対象外）

| ID | 内容 | 方針 |
|----|------|------|
| **B-3** | timestamp-based capture（非一様 callback rate を `out.size()/2.0s` が完全補正しない） | 保留。flipIndex 誤差 <0.5% 実測で現行 transition probe 目的には十分（`computeTransitionMetrics(out, effectiveCaptureRate, ...)` — BassBuzzMeasurement.cpp:2346 実測） |
| **D-1** | build identity gate の M1/M2 欠陥（E-G3-3 既知・未修正） | 保留。stamp 再作成で回避継続 |
| **D-2** | headroom ランタイム統合（4 ランタイム混在） | 保留。既存 fork litellm パッチ運用継続 |
| **R-1** | `--buzz-flip-eqgain=` 受理して破棄（1613 行: `opt.flipKind = 4; opt.flipValue = 0;`・値すら parse しない） | 保留。silent-ignore 系の別パターンとして記録 |
| **R-2** | `parseHcIdx`/`parseLcIdx`（1558/1564 行）と `stod`/`stoi`/`stof` 計 9 箇所の不正入力で未捕捉例外（try/catch なし） | 保留。fail-closed 化は別 work item |

### 5.1 §5 の修正方針（将来 work item）

#### B-3: timestamp-based capture

```cpp
// 案: 受信時のホストタイムスタンプを記録し、平均実効レートを再計算
const auto captureTimestamp = std::chrono::high_resolution_clock::now();
session.captureTimestamps.push_back(captureTimestamp);
```

実装コスト: 中・gain は限定的（flipIndex 誤差 0.5% が問題になる場面が少ない）

#### D-1: build identity gate M1/M2

- M1: ベアシェル/icx 起動時の cache 素字化 → `find_package` 段階でキャッシュハッシュ検証
- M2: commit 毎の `source_revision` 変更 → COHERENCE-4 fail-closed → stamp 自動化

実装コスト: 高（外部システム統合）

#### D-2: headroom runtime 統一

```bash
uv tool install --force headroom-ai --with fastapi --with uvicorn
# Startup の .lnk / bat / vbs 3 起動体を 1 つの .lnk に統合
```

実装コスト: 中・リスク: 既存 proxy 起動失敗の可能性

#### R-1: silent-ignore 修正

```cpp
else if (a.rfind("--buzz-flip-eqgain=", 0) == 0) {
    std::fprintf(stderr, "[BUZZ] INFO: --buzz-flip-eqgain=%s accepted but ignored (design-fixed -3dB step)\n",
                 a.substr(20).c_str());
    opt.flipKind = 4; opt.flipValue = 0;
}
```

実装コスト: 極小

#### R-2: stod/stoi 例外の try/catch

```cpp
inline int parseIntOrFail(const std::string& flag, const std::string& v) {
    try { return std::stoi(v); }
    catch (const std::exception&) {
        std::fprintf(stderr, "[BUZZ] FAIL: %s expects integer (got '%s')\n", flag.c_str(), v.c_str());
        std::exit(2);
    }
}
```

実装コスト: 極小・全 `std::stoi` / `std::stod` / `std::stof` 呼び出し（9 箇所）を統一

### 5.2 §5 の着手優先度（将来）

1. **R-2**（極小・fail-closed 化・即時効果）
2. **R-1**（極小・silent ignore 解消）
3. **B-3**（中・timestamp 化）
4. **D-2**（中・環境統一）
5. **D-1**（高・build identity gate 修正）

---

## 6. 推奨する実装順序（v1.5）

```
[Step 1] §3 F-2（harness cleanup・test-only）
  - F-2: 呼出順修正（1 行差替・§3.1.4 の検証手順に従う）
  - F-3/F-4: 記録のみ
  - 検証: rigcheck=eq で gainpath staging=0 eqAGC=0 を確認

[Step 2] §4 F-1（テスト移行）
  - CrossfadeAuthority 4 ケース → カスタム main ハーネス
  - Phase 0 分類 → Phase B 移行 + Phase B' semantic equivalence gate
  - PublicationValidatorIsolationTests.cpp 退役

[Step 3] §1 O-1〜O-7（commit 意思決定）
  - ユーザー判断待ち
  - 案 A 採用時はエントリ配線の最小変更を含めて commit

[Step 4] §2 B-1（DSP 修正・破壊的変更・案 E）
  - Phase 0-0: source/contract audit（DESIGN-CONTRACT-A 確定）
  - Phase 0-1: Baseline characterization（record-only・期待 0.75^N）
  - Phase 0-2: Shadow Candidate E（test-only reference・production 0 変更）
  - Phase 0-3〜0-6: 周波数（stage-rate 軸）/ block/reset / SoftClip / float-double characterization
  - B-1-P0 GATE（§2.6）全 PASS → Phase 1 eligibility
  - Phase 1: CMake option 導入（compile-time・既定 OFF・既存テスト全 PASS）
  - Phase 2: flag ON で限定検証（Phase 0 shadow 値と直接比較）
  - Phase 3-A/3-B/3-C/3-D: flag 常時 ON → 実測 → calibration-only commit → 窓-only commit
  - Phase 4: flag 削除（次メジャーリリース）

[Step 5] §5 別課題（将来 work item 化）
  - R-2（最優先・極小）→ R-1（次・極小）→ B-3, D-2, D-1
```

**Step 4 の Phase 0 が完了するまで、Phase 1 以降の着手は不可**。Phase 0 で契約・差分・周波数特性のいずれかが不成立の場合、案 E を見直し、案 D 統合や tap 再設計に方針変更する。

---

## 7. 検証計画

### 7.1 単体検証（v1.5・R7 修正済み）

| 検証項目 | コマンド | 期待 |
|----------|----------|------|
| §3 F-2 修正 | `build\Release\PublishPipelineIntegrationTests.exe --buzz-rigcheck=eq --buzz-dur=2.0` | `[BUZZ] RIGCHECK(eq) gainpath: staging=0 eqAGC=0 ...` |
| §4 F-1 移行 | `ctest --test-dir build --output-on-failure` | 移行先テストが PASS・PublicationValidatorIsolationTests 削除済 |
| §2 B-1 Phase 0/1 | `cmake --build build --config Release --target PublishPipelineIntegrationTests && build\Release\PublishPipelineIntegrationTests.exe` | `[OS_DIRECT] ... roundTripGain=`（flag OFF 時 ≈0.75 系・**flag ON 時 ≈ 1.0**）。DC/多周波の厳密判定は Phase 0 test-only reference implementation で実施 |
| §2 B-1 Shadow Candidate E | Phase 0-2 の test-only reference implementation 実行 | DC round-trip = 1.0 ± 1e-6（全構成） |

**注**: `--buzz-osdirect=2` 等の ConvoPeq.exe CLI フラグは存在しない（v1.4 改訂サマリ P0-1 のとおり §7.1 旧行は削除済み）。`[OS_DIRECT]` は `PublishPipelineIntegrationTests` main の無条件呼出し経由で出力される。

### 7.2 統合検証

| 検証項目 | 内容 | 期待 |
|----------|------|------|
| §2 B-1 Phase 3 | `eq` 判定窓の再校正 | 新窓での PASS（既存窓 0.4912 → 新範囲） |
| §4 F-1 Phase C | `tools/build-debug.bat` 実行 | `--target ...` エラーなし |

### 7.3 ビルド検証

```bash
# MSVC ビルド
build.bat
# icx ビルド
build-icx.bat
# AudioEngineHarness Release インクリメンタル
cmake --build build --config Release --target AudioEngineHarness
```

### 7.4 静的解析

```bash
# cppcheck（C++指定・機能欠陥0 を目標）
cppcheck --language=c++ --std=c++20 --enable=warning,performance,portability \
    src/CustomInputOversampler.cpp src/audioengine/AudioEngine.Processing.Latency.cpp

# clang-tidy（対象 TU）
clang-tidy -p compile_commands.json src/CustomInputOversampler.cpp
```

---

## 8. ロールバック計画（v1.5・R10 修正済み）

| Step | ロールバック方法 |
|------|------------------|
| §1 commit | `git revert <sha>` |
| §2 B-1（behavioral rollback） | **CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` で rebuild**（第一手段・production source は触らない） |
| §2 B-1（source-level rollback） | `interpolateStage` 内の `#if CONVOPEQ_CORRECT_POLYPHASE_GAIN … #endif` ブロックと in-source safety net を削除（rebuild 必須） |
| §3 F-2 | 1 行 revert（呼出順を元に戻す） |
| §4 F-1 | 移行先テストが残る場合、ファイルを git から復元 |
| §5 別課題 | 実装しないため不要 |

※ flag 名は **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`**（CMake option）で全文統一（`kUseV2Oversampler` は採用しない）。behavioral rollback と source rollback を明確に分ける（レビュー① §20）。

---

## 9. 影響度まとめ

| 区分 | 影響範囲 | 影響度 | 段階リリース |
|------|----------|--------|--------------|
| §1 commit | リポジトリ履歴のみ | 極小 | 任意 |
| §2 B-1（案 E） | **全オーディオ経路**（最大 +7.5 dB 想定） | **極大**（1 行追加で済む） | 必須（Phase 0→1→2→3-A/B/C/D→4） |
| §3 F-2 | test-only | 小 | 不要 |
| §3 F-3 | **production behavior の潜在欠陥（記録のみ・本セッション修正なし）** | なし（本セッション） | 不要 |
| §3 F-4 | test-only（記録のみ） | なし | 不要 |
| §4 F-1 | test-only | 中 | 不要 |
| §5 別課題 | 環境 or CLI | 極小〜中 | 不要 |

---

## 10. 監査ログ・参照

- 前スナップショット: `doc/work113/residual_tasks_20260919.md`
- 本セッション報告書: `doc/work113/residual_tasks_20260920.md`
- レビュー記録: `doc/work113/renew_plan.md`（v1.4 本文 + レビュー① 38 観点）
- 関連監査記録:
  - `doc/work104/bass_buzz_measurement_20260917.md`
  - `doc/work105/ir_runtime_contract_and_remeasure_20260917.md`
  - `doc/audit/ConvoPeq_BugList_and_FixPlan_2026-09-13.md`
  - `doc/audit/ConvoPeq_Bug_Verification_2026-09-13.md`
  - `doc/audit/ConvoPeq_SampleRate_Reverification_2026-09-13.md`
- 関連ソース（行番号は 2026-09-20 実測）:
  - `src/CustomInputOversampler.h`（isLinearPhaseFIR/isSymmetricUpDown: 21-22）
  - `src/CustomInputOversampler.cpp`（prepareStage 287-390 / prepareSingleStage 392 / interpolateStage 492-568 / decimateStage 570-723 / convValue ×2 は 557）
  - `src/audioengine/AudioEngine.h:1416-1432`（F-2 根本）・2626（staging 既定 true）・1292/1306/1314（uiEqEditor 経由）・3666-3670（Bridge → validator_）
  - `src/audioengine/AudioEngine.Processing.Latency.cpp:6-8`（static_assert）・22-24（taps 表）
  - `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp:188/261`（prepareSingleStage(31, 90.0)）
  - `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp:253-263`（float→double 変換）・405/413（softClipOS up/down）
  - `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:505/513`（softClipOS up/down）・593（kOutputHeadroom）
  - `src/eqprocessor/EQProcessor.Processing.cpp:570-582 / 978-1015`（F-3）
  - `src/audioengine/RuntimePublicationValidator.cpp:14-41`（検査順序・first-fail-wins）
  - `src/audioengine/RuntimePublicationValidator.h:101`（F-1 private）
  - `src/TruePeakDetector.cpp:284-318`（姉妹実装・両位相 ×2 なし）
  - `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp`（診断モード 1700-1930 / runOversamplerDirect 1293 / runEqDirectDriveAttribution 1529 / 窓 2042-2045）
  - `src/tests/PublicationValidatorIsolationTests.cpp`（F-1）

---

## 11. ユーザー判断待ち項目

| 項目 | 選択肢 | 推奨 |
|------|--------|------|
| §1 O-1 案選択 | A: 最小実行可能資産（[OS_DIRECT]+配線）/ B: 全 +642 行 / C: 退役 | **A** |
| §1 O-2 commit 方針 | O-1 と同梱 / 独立 | **独立** |
| §1 O-6 push | 即時 / 別タイミング / ユーザー手動 | **ユーザー手動** |
| §1 O-7 AGENTS.md | 触らない / 別 work item で更新 | **触らない** |
| §2 B-1 修正案 | **E: interpolateStage に `centerValue *= 2.0` 追加（既存 half-band FIR 維持）** | **E（ユーザー監査確定）** |
| §3 F-2 修正 | A: 呼出順逆 / B: setAGCEnabled 削除 / C: 順序固定 | **A** |
| §4 F-1 移行 | Phase A→B→B'→C 全実施 / Phase A のみ / 記録のみ | **Phase A→B→B'→C 全実施** |
| §2 Phase 3 の commit 分割 | 3-A〜3-D 分割 / 従来の単一 Phase 3 | **3-A〜3-D 分割（R11）** |

### §2 B-1 採用案の前提条件（v1.5）

**案 E（polyphase gain convention 対称化）は「有力仮説」であり、Phase 0 characterization（v1.5 GATE 表）が全 PASS しなければ Phase 1 実装 GO とはしない**:

1. **DESIGN-CONTRACT-A**（unity-gain requirement）が文書化・承認される（Phase 0-0）
2. **Shadow Candidate E の DC gain round-trip = 1.0 ± 1e-6**（test-only reference implementation・production 0 変更）
3. **anti-aliasing / anti-imaging が維持**される（stage-local 実測・Phase 0-D/E）
4. **latency が不変**（impulse 実測・Phase 0-F）
5. **SoftClip 局所 OS（`prepareSingleStage(31, 90.0)`）が正常動作**（Phase 0-I）
6. **Differential（Candidate/Baseline 差分）が center-phase ×2 のみに帰属**（P0-C'）

Phase 0 で前提が崩れた場合は、案 E を見直し、案 D 統合（offset 補正）や tap 再設計を検討。

### v1.5 最終判定表

| 項目 | 判定 | 備考 |
|------|------|------|
| §1 O-1〜O-7 | **GO 候補** | commit/push の意思決定として分離 |
| §3 F-2 | **GO 候補** | 呼出順修正（1 行差替・検証手順 §3.1.4） |
| §3 F-3 | **記録継続** | production behavior の潜在欠陥（test-only ではない）・定常 B-1 の原因ではない |
| §3 F-4 | **記録継続** | 既存 dry 測定基準維持・`irwet<digit>` で wet 対照 |
| §4 F-1 | **GO 候補** | case classification 先行 → 公開 API 経由移植 + Phase B' gate |
| §5 | **保留継続** | 将来 work item 化 |
| **B-1 原因分析** | **GO** | コード整合性は再確認済み |
| **B-1 Phase 0** | **GO（測定仕様 v1.5 適用が条件）** | 旧 P0-A = 1.0 は廃止・Baseline/Candidate/Differential 3 層構造 |
| B-1 Phase 1 | **HOLD** | Phase 0 全 PASS が前提 |
| B-1 Phase 2/3/4 | **HOLD** | Phase 0 → Phase 1 → Phase 2 の順次判定 |
| **B-1 案 E** | **仮説として採用** | 強い仮説だが Phase 0 で検証前は確定ではない |
| B-1 「perfect reconstruction」 | **表現修正** | DC gain 1.0 は言えるが PR 全帯域保証は未証明 |
| B-1 calibration 変更 | **HOLD** | Phase 3-C/D まで保留・分割 commit |
| `0.75^N` engine-fit 削除 | **HOLD** | 経験式（コード定数なし）・監査記録/窓の更新は Phase 3-C/D |
| compile-time flag | **CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN`** | authoritative source = CMake・in-source `#ifndef` は safety net |
| TruePeakDetector 姉妹実装 | **記録継続** | 両位相 ×2 なし・案 E の正解としない・参考測定 or 別 work item |

---

## 12. 本計画書の検証エビデンス（2026-09-20 ソース監査）

本計画書 v1.5 は、以下の検証を 2026-09-20 に実施した結果に基づく。

1. **行番号監査**: 計画書の引用箇所を production source と照合（§10 の参照一覧はすべて実測行番号）。内容の不一致は parity 表記の反転 1 件（v1.5 R16 で修正）のみ
2. **数値再現**: `prepareStage` → `interpolateStage` → `decimateStage` を NumPy/SciPy で逐語移植
   - 現行: 1 段 DC = 0.750000 / 3 段 = 0.421875（=0.75³）— §2.1 の確定事項と厳密一致
   - 案 E shadow: 全構成 DC = 1.000000 — Phase 0-2 の期待値の予備確認
   - 帯域端 0.45 Fs_in で candidate/baseline 比が非一様（帯域端 alias 混合）— Phase 0 の実測必須性を裏付け
3. **静的解析**: cppcheck（--language=c++ --std=c++20 --enable=warning,performance,portability）を `CustomInputOversampler.cpp` に実行 → **指摘 0 件**
4. **git 実測**: ahead 2（`c4a08171` + `8f127bfe`）、numstat 全項目が計画書前提と一致
5. **レビュー① の照合**: 38 観点のうち、7 必須項目をすべて採用（v1.5 R1〜R7）、追加指摘 R8〜R15 を採用、技術的中核（案 E・原因特定・段階リリース骨格）はレビュー①も妥当と評価 → 維持

---

**本書は方針案です（v1.5・レビュー① 反映版・Phase 0 測定仕様改訂済み）。実装着手は本書承認後**。
**§2 B-1 の DSP 修正方針：案 E（polyphase gain convention 対称化）を「有力仮説」として採用**。
**Phase 0 は「現行 baseline と Shadow Candidate E を同一測定系で比較し、設計契約・周波数特性・遅延・境界挙動のすべてを検証する characterization フェーズ」である（現行 production の PASS/FAIL で案 E を判定するフェーズではない）**。
**compile-time flag は CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN`（authoritative source = CMake）・段階リリース Phase 0 → 1 → 2 → 3-A/B/C/D → 4**。
**Phase 0 全 PASS → Phase 1 eligibility。Phase 1 以降は Phase 0 結果次第で GO 判定**。
**案 E は「正しいことが証明された修正」ではなく「現行実装の gain imbalance を最小変更で修正する有力仮説」として扱う**。
**「perfect reconstruction」表現は Phase 0 で PR が数値確認されるまで使用しない**。
**承認状態: §1/§3 F-2/§4 F-1 = GO候補 / §3 F-3・F-4/§5 = 保留継続 / §2 = Phase 0 着手承認（測定仕様 v1.5 適用が条件）・Phase 1 以降は HOLD**。
