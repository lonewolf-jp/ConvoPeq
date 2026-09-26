doc\work113\remediation_plan_20260920.md
# ConvoPeq 残件 改修計画書（2026-09-20 時点）

- **版**: v1.4（ユーザー監査 3 ラウンド反映・最終確定版）
- **対象**: `doc/work113/residual_tasks_20260920.md` 全残件（§1〜§5）
- **方針**: 段階リリース重視・既存動作への影響を最小化・ロールバック容易性確保
- **前提**: production `src/` の未 commit 差分は 0 件、HEAD = `8f127bfe`（docs 1 件）+ `c4a08171`（B-2 test 1 件）が未 push
- **本書は read-only 監査と方針案の提示**。実装着手は本書承認後
- **基準ソース**: `ConvoPeq.md`（Generated 2026-09-19 22:08:26）および HEAD の production source
- **§2 B-1 修正方針**: 案 E（polyphase gain convention 対称化・既存 half-band FIR 維持）を「**有力仮説**」として採用（oracle C 案はユーザー監査で撤回）
- **Phase 0 着手 GO・Phase 1 以降 HOLD**: 案 E は Phase 0 の characterization（§2.5.1 判定基準 A〜I）で全 PASS しなければ Phase 1 実装 GO とはしない
- **compile-time flag**: `CONVOPEQ_CORRECT_POLYPHASE_GAIN`（compile-time gate で全文統一。`kCorrectPolyphaseGain` / `kUseV2Oversampler` は採用しない）

---

## v1.3 → v1.4 改訂サマリ（ユーザー監査 3 ラウンド反映）

ユーザー監査 3 ラウンド（`doc/work113/renew_plan.md` 添付版）の結果を統合し、計画書の技術的中核は維持しつつ記述上の実務不整合を修正した。

| 区分 | 修正内容 |
|------|----------|
| **P0-1** | §7.1 / §2 検証コマンドを `PublishPipelineIntegrationTests` 経由に書き換え。出力キーは `roundTripGain=`（`round-trip=` ではない）。`--buzz-osdirect=2` フラグは**存在しない**ため削除 |
| **P0-2** | §1.1 案 A に「エントリ配線の最小変更を含む」ことを明記（`[OS_DIRECT]` 単独 commit は現状そのままでは不可） |
| **P0-3** | 全文 **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** で統一（`kCorrectPolyphaseGain` / `kUseV2Oversampler` は削除） |
| **P1-1** | §2.4 / §2.5.1 F: latency を **`(taps−1)` サンプル（stage レート）** に修正（旧 `(taps-1)×2 per stage` は誤り） |
| **P1-2** | §4.2 Phase B テスト分類 **28/6 → 25/9** に修正（実測一致） |
| **P1-3** | §2.5「engine fit の 0.75^N 項の削除」を **「rigcheck 判定窓・監査記録・経験式記述の更新」** に読み替え（production に該当定数なし） |
| **P2-1** | 行番号補正：`prepareStage` 287-390、テスト 519 行、F-3 ブレンド本体は 978 起点、static_assert 6-8 行 |
| **P3-1** | `AGENTS.md`（MIMO Desktop パイプライン運用メモ・`+5/−3`・未 commit）を §1 O-7 として「触らない」方針で明記 |
| **P3-2** | `TruePeakDetector::interpolateStage`（姉妹実装）が **両位相とも ×2 なし**の gain convention を持つことを記録（Phase 0 の参考測定 or 別 work item） |

---

## 0. 凡例と全体戦略

| 区分 | 着手優先度 | 影響範囲 | ロールバック |
|------|------------|----------|--------------|
| **P0 即時**（破壊的・不可逆） | なし | — | — |
| **P1 高**（影響大・段階リリース） | H-02 関連 + B-1 | 全オーディオ経路 | フラグ revert |
| **P2 中**（harness / テスト cleanup） | F-1〜F-4 | test-only | ファイル revert |
| **P3 低**（運用 / 環境 / 既知制限） | O-1, B-3, D-1, D-2, R-1, R-2 | 環境 or CLI | 設定 revert |
| **別課題**（既存記録・本セッション対象外） | B-3, D-1, D-2, R-1, R-2 | — | — |

**全体戦略**:
1. **§1（O-1〜O-6）の意思決定** → コミット方針確定（ユーザー判断待ち）
2. **§2 B-1** → 大規模 DSP 修正。フィーチャーフラグ・段階リリース・既存 calibration 再校正を 3 段階で実施
3. **§3 F-2/F-3/F-4** → harness cleanup（test-only 修正）。B-1 とは独立して先に着手可能
4. **§4 F-1** → テスト移行（custom main ハーネスへ）。`CrossfadeAuthority` 4 件を最優先
5. **§5 別課題** → 既存記録のまま保留。必要時に別 work item 化

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-6）

### 1.1 O-1: B-1 計測用 test-only 計装（+642 行）

| 項目 | 内容 |
|------|------|
| ファイル | `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp`（+634/−2）、`PublishPipelineIntegrationTests.cpp`（+8/−0） |
| 内容 | `eqdiag` / `eqdiagser` / `eqos<digit>` / `irwet<digit>` 診断モード、`gainpath` 行、`[EQ_DIRECT]` / `[OF_DIRECT]` / `[OS_DIRECT]` 直接駆動測定 |
| 影響 | test-only。production `src/` 変更なし |
| 判定 | 既存 `eq` 判定窓 `[0.486,0.496]` は不変（`eq` は `ratio=0.4912` で PASS 継続） |
| 出力キー | `[OS_DIRECT] ... roundTripGain=`（`round-trip=` ではない） |
| エントリ | `runEqDirectDriveAttribution()` 経由で `PublishPipelineIntegrationTests` の main から無条件呼出し |

**3 案**:

| 案 | 内容 | メリット | デメリット |
|----|------|----------|------------|
| **A（推奨）** | `[OS_DIRECT]` のみ commit（最小資産）+ **エントリ配線の最小変更** | 過去回帰検出を最小コストで残せる | `eqdiag` / `irwet` 診断モードは手元検証用に残る。**main 側で `runOversamplerDirect()` を直接呼べる配線追加が必要** |
| **B** | 全 +642 行を 1 commit | 将来 B-1 再発の即時検出資産として完全保存 | 行数が大きい。レビュー負荷大 |
| **C** | すべて退役（破棄） | 後方互換性問題なし | 再発時の解析遅延 |

**推奨案**: **A**（最小資産コミット）。`[OS_DIRECT]` は CustomInputOversampler の `processUp`/`processDown`/`prepareSingleStage` の round-trip を直接検証するため、B-1 修正後の回帰検出に直結する。

**v1.4 追記（重要）**: `[OS_DIRECT]` / `[EQ_DIRECT]` / `[OF_DIRECT]` は **同一エントリ関数 `runEqDirectDriveAttribution()` に束ねられている**。`git add -p` だけで `[OS_DIRECT]` のみを分離 commit することは **現状そのままでは不可能**。案 A を採用する場合は **main 側の呼出し配線替え**（例: `runOversamplerDirect()` を直接呼ぶエントリ追加）を含める必要がある。`eqdiag` 系の harness 拡張はテスト目的が局所的（OS 段分離仮説）なので手元検証後に退役検討。

**CLI 注記**: `--buzz-osdirect=2` 等の ConvoPeq.exe CLI フラグは存在しない。検証は `PublishPipelineIntegrationTests` 実行（`cmake --build build --config Release --target PublishPipelineIntegrationTests && build\Release\PublishPipelineIntegrationTests.exe`）で行い、stderr の `[OS_DIRECT] ... roundTripGain=` を観察する。

### 1.2 O-2: 台帳更新（`doc/work113/residual_tasks_20260919.md`）

| 項目 | 内容 |
|------|------|
| ファイル | `doc/work113/residual_tasks_20260919.md`（B-1 帰属・F-2〜F-6 追記分） |
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
| 状態 | ` D`（worktree で削除状態・tracked） |
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

### 1.7 O-7（v1.4 追加）: `AGENTS.md`（MIMO Desktop パイプライン運用メモ・未 commit）

| 項目 | 内容 |
|------|------|
| 状態 | ` M`（`+5/−3`・MIMO Desktop パイプライン運用メモ） |
| 内容 | headroom proxy・context-mode・rtk 常時運用、ConvoPeq の 3 層パイプライン（ソースで防ぐ → 自動圧縮 → CLI 出力圧縮）の指針 |
| 影響 | 環境運用のみ。production DSP / テスト成果物への影響なし |
| 方針 | **触らない**（削除も commit もせず、現状維持。別 work item でドキュメント更新するか、ユーザー判断） |

### 1.8 §1 全体の推奨手順

```
[Step 1.1] ユーザー判断: O-1（A/B/C 案選択） + O-2（同梱 or 独立）
[Step 1.2] 該当ファイルを commit（メッセージ: "test: B-1 attribution diagnostic instrumentation ([OS_DIRECT])" 等）
[Step 1.3] O-3 は触らない・O-4/O-5 も触らない
[Step 1.4] O-6 push は独立運用操作としてユーザー承認後に実行
```

---

## 2. §2 B-1: CustomInputOversampler の up/down round-trip 欠陥（最大規模）

### 2.1 確定事項（再掲・報告書 §2.1）

```
主因        CustomInputOversampler の up/down round-trip 利得欠陥
1 stage     round-trip 0.750000（up 1.000000 / down 0.750000）
多段        0.75^N（ratio 2/4/8 = 0.75 / 0.5625 / 0.421875、preset 非依存）
SoftClip局所OS 0.75（prepareSingleStage(31, 90.0) を production 引数で実測）
数学的導出  0.5×0.5 + 0.5×1.0 = 0.75
engine fit  0.98379 × 0.75^max(log2 effOS, 1) で 4 条件 ≤0.05%
契約不整合  isSymmetricUpDown / Latency の static_assert 前提と矛盾
production 変更 0 / commit 0
```

### 2.2 根本原因の特定（コード追跡）

#### 2.2.1 `prepareStage`（`src/CustomInputOversampler.cpp:287-390`）

```cpp
stage.centerTap = (stage.taps - 1) / 2;
stage.centerParity = stage.centerTap & 1;
stage.convParity = 1 - stage.centerParity;
// ...
rawCoeffs[stage.centerTap] = 0.5;              // ← center 一個の重み = 0.5
// ...
nonCenterSum = Σ rawCoeffs[i] (i ≠ centerTap);
// ...
scale = 0.5 / nonCenterSum;                    // ← 非 center 合計 = 0.5 に正規化
// ...
stage.convCount = (stage.taps - stage.convParity + 1) / 2;
stage.convCoeffs[r] = rawCoeffs[convParity + 2r];  // convParity タップ抜き出し
```

**FIR 合計**: center (0.5) + 非 center (0.5) = **1.0**（normalize 済み）
**convParity タップ合計**: **0.5**（half-band FIR の偶奇分離）

#### 2.2.2 `interpolateStage`（`src/CustomInputOversampler.cpp:492-568`）

```cpp
double centerValue = stage.centerCoeff * history[idx - stage.centerDelayInput];
// ...
double convValue = Σ stage.convCoeffsReversed[r] * xWindow[r];  // = 0.5
// ...
convValue *= 2.0;                                // ← ★ conv 位相のみ ×2 補正

output[outBase + stage.centerParity] = centerValue;  // 偶数サンプル = 0.5 × input
output[outBase + stage.convParity]   = convValue;    // 奇数サンプル = 1.0 × input
```

**up の出力（1 入力 → 2 サンプル）**:
- 偶数（center パス）= **0.5**
- 奇数（conv パス）= **0.5 × 2 = 1.0**
- 平均 = **0.75** / 合計 = 1.5

#### 2.2.3 `decimateStage`（`src/CustomInputOversampler.cpp:570-723`）

```cpp
double acc = stage.centerCoeff * centerSample;     // 0.5 × history[center]
// ...
acc += Σ coeffs[r] * history[base - convParity - 2r];  // stride-2 FIR = 0.5 × history[odd]
// ...
output[n] = acc;                                  // ← ★ ×2 補正なし
```

**down の出力（2 入力 → 1 サンプル）**:
- center 寄与 = **0.5**
- conv 寄与 = **0.5**
- 合計 = **1.0**

#### 2.2.4 round-trip 整合式

```
up:   even=0.5 (center パス) + odd=1.0 (conv パス ×2) → 平均 0.75
down: center=0.5 + conv=0.5 = 1.0
round-trip = 0.75 × 1.0 = 0.75   （報告書「down 0.75」と表記揺れあるが数学的に同一）
```

**多段（ratio 8 = 3 stages）**:
```
up_3 = up^3 = 0.75^3 = 0.421875
down_3 = 1.0
round-trip_3 = 0.421875 = -7.5 dB
```

**SoftClip 局所 OS** も `prepareSingleStage(31, 90.0)` で同一経路を通るため 0.75。

### 2.3 修正案の比較（v1.2・ユーザー監査反映）

#### 2.3.1 v1.1 で挙げた案 A/B/C/D の再評価

| 案 | 内容 | v1.1 評価 | **v1.2 評価** |
|----|------|-----------|---------------|
| A | `decimateStage` に ×2 追加 | +2.5 dB | **不採用**：round-trip = 1.125 で過剰補正 |
| B | `interpolateStage` の ×2 削除 | -8.5 dB | **不採用**：round-trip = 0.375 で過少 |
| C | `prepareStage` を center=1.0 / non-center=0.0 に変更 | perfect | **撤回（ユーザー監査で誤り確定）** |
| D | A+B 統合 | offset 残 | **不採用** |

#### 2.3.2 v1.2 新規案 E（ユーザー監査で確定）

**案 E: polyphase gain convention 対称化** — 既存 half-band FIR を維持し、`interpolateStage` の **両 polyphase に ×2 を適用**:

```cpp
// src/CustomInputOversampler.cpp interpolateStage() 内
// 現行: center phase × 1, conv phase × 2
// 修正後: center phase × 2, conv phase × 2（対称化）

double convValue = ...           // = 0.5 × input（half-band FIR convParity タップ合計）
double centerValue = ...;       // = 0.5 × input（center タップ）
convValue *= 2.0;              // 既存
centerValue *= 2.0;             // ★ 新規追加（v1.2）

output[outBase + stage.centerParity] = centerValue;  // 偶数サンプル = 1.0 × input
output[outBase + stage.convParity]   = convValue;    // 奇数サンプル = 1.0 × input
```

**案 E 採用時の計算**:
- up: even=1.0（center 0.5 × 2）、odd=1.0（conv 0.5 × 2）、平均=1.0、合計=2.0
- down: 既存通り center=0.5、conv=0.5、合計=1.0（**変更なし**）
- **constant-signal / DC gain round-trip = 1.0 × 1.0 = 1.0**

**注意（v1.3 で表現修正）**:
- 「**round-trip = 1.0**」は **constant-signal / DC gain についてのみ成立**する
- 一般信号での perfect reconstruction (PR) は **H_up(z) × H_down(z) = z^(-L) となる全帯域での保証** を意味するが、これは案 E の修正だけでは**自動的に証明されない**
- 案 E は **FIR coefficient shape を変更せず**、polyphase のゲイン分配のみ対称化する。FIR shape は維持されるため、**周波数応答の形状（passband ripple / stopband attenuation / image rejection / alias rejection / group delay / phase）は不変と予想**されるが、Phase 0 で実測確認が必要
- v1.2 で使用していた「**perfect reconstruction**」表現は **Phase 0 で PR が数値確認されるまで使用しない**（constant-signal / DC gain のみが現状の確定事項）

**gain 変化**:
- 1 stage: 0.75 → 1.0 = **+2.4988 dB**（≈ +2.5 dB）
- 2 stages: 0.5625 → 1.0 = **+4.998 dB**（≈ +5.0 dB）
- 3 stages: 0.421875 → 1.0 = **+7.50 dB**

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
| エイリアシング再評価 | 必要 | **不要** |
| 修正行数 | 約 5 行（prepareStage normalize） | **1 行**（interpolateStage 1 行） |
| rollback | 5 行 revert | **1 行 revert** |

**案 C 撤回理由（ユーザー監査原文）**:
> C は FIR の中心タップを 1 にして他を 0 にするもので、`h[n] = δ[n-center]` という単なる pure delay になる。convCoeffs = 0 となり、現在の構造では `convValue = 0` → `output[convParity] = 0`、`output[centerParity] = input` という、実質的な zero-stuffing / delay 構造。これは anti-alias / anti-imaging filtering を失う。したがって「C は half-band PR の教科書条件」という oracle の説明は、この `CustomInputOversampler` の実装に対しては採用できない。

### 2.4 8 観点への影響（v1.4・案 E）

| 観点 | 影響 | 対応 |
|------|------|------|
| 1. decimateStage 側の補正 | **不要** | 案 E は `decimateStage` に変更なし |
| 2. up/down の gain convention 対称化 | 中（gain convention のみ変更） | 案 E：`interpolateStage` に `centerValue *= 2.0` を追加（1 行） |
| 3. `isSymmetricUpDown` の意味を実装に合わせる | **FIR tap symmetry は維持**（FIR coefficients は不変） | polyphase gain は対称化されるが、`isSymmetricUpDown=true` の元となる FIR 構造自体は不変 |
| 4. AudioEngine.Processing.Latency.cpp の対称 FIR 前提 | **latency contract は別途数値検証**（Phase 0 判定基準 F） | FIR tap 不変 → **1 段あたり往復 (taps−1) サンプル（stage レート）**、base レート換算は `× (baseRate / stageRate)` の見込み。Phase 0 で実測 |
| 5. 既存 calibration / rigcheck（窓 `[0.486,0.496]`） | 再校正必要（後段・HOLD） | gain +2.5 dB/段 → `[0.486,0.496]` を新 gain 値へ。ただし Phase 0 で再測定 |
| 6. SoftClip 分岐（`effOS==1`）と主 OS 分岐（`effOS>1`）の双方 | 同一修正で両者対応 | `prepareSingleStage(31, 90.0)` も同経路を通る（Phase 0 判定基準 I） |
| 7. Float / Double 両経路 | 同一経路 | DSPCoreFloat / DSPCoreDouble 両方で gain 検証（Phase 0 判定基準 G） |
| 8. 既存 regression への波及 | 全 regression 期待値要再校正（HOLD） | work57 の NUC、outputStage gain、`kOutputHeadroom` 等。Phase 3 まで保留 |

**v1.4 で明確化した点**:
- 「FIR tap symmetry は維持」: 案 E は FIR の `prepareStage` 出力を変更しない
- 「latency contract は別途数値検証」: `isSymmetricUpDown=true` 宣言と latency `static_assert` は FIR shape のみを参照するため FIR 不変なら保持されるが、Phase 0 判定基準 F で実測
- 「anti-aliasing 維持」→「FIR coefficient shape unchanged・relative frequency-response shape 不変と予想」（Phase 0 判定基準 B/C/D/E で実測確認）
- 「5/8 の項目」は Phase 0 完了まで **HOLD** 扱い（calibration / regression 期待値）

### 2.5 段階リリース計画（v1.4・Phase 0 gate 独立・compile-time flag）

```
[Phase 0] Characterization（read-only・production src/ 変更 0）
  - 詳細は §2.8 B-1-P0 GATE 参照
  - 判定基準 A〜I（§2.5.1）すべて PASS → Phase 1 eligibility

[Phase 1] Feature Flag 導入（compile-time・破壊なし）
  - interpolateStage に `centerValue *= 2.0` を追加（案 E）
  - フィーチャーフラグは **compile-time gate**:
    ```cpp
    // src/CustomInputOversampler.h
    #ifndef CONVOPEQ_CORRECT_POLYPHASE_GAIN
    #define CONVOPEQ_CORRECT_POLYPHASE_GAIN 0  // 既定 false（既存挙動維持）
    #endif

    // src/CustomInputOversampler.cpp interpolateStage() 内
    #if CONVOPEQ_CORRECT_POLYPHASE_GAIN
        centerValue *= 2.0;
    #endif
    ```
  - 既定値 false。Phase 2 で build option `-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=1` を指定して rebuild
  - ※ runtime flag は **採用しない**（RT path で毎 sample/毎 block 分岐を避けるため）
  - flag 名は compile-time macro **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** で統一（理由: FIR redesign ではなく polyphase gain convention の修正のみであることを正確に表現）

[Phase 2] Flag ON で限定検証
  - `-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=1` で rebuild
  - 内部 dogfood で ratio 1/2/4/8 × preset IIRLike/LinearPhase 全段検証
  - Phase 0 の測定結果と直接比較
  - 「engine fit の 0.75^N」の必要性を再判定（必要なら calibration 補正 / 不要なら定数削除）

[Phase 3] 全段 ON（破壊的変更）
  - コンパイル時 flag を #define で常に 1 に
  - kOutputHeadroom 等 calibration 値を再校正
  - rigcheck 判定窓 [0.486, 0.496] を新 gain 値に更新
  - 全テスト + 新 rigcheck 窓で PASS を確認

[Phase 4] フラグ削除（次メジャーリリース）
  - CONVOPEQ_CORRECT_POLYPHASE_GAIN 分岐と修正行を恒久化
```

**重要（v1.4）**:
- `0.98379 × 0.75^N` は `residual_tasks_20260919.md:79` の**経験式**であり、production コードに `0.98379` / `0.75^N` / `effOS` 定数は**存在しない**（全域 grep で確認）。`kOutputHeadroom = 0.8912509381337456`（−1.0 dBFS）は別定数。Phase 3 の「0.75^N 削除」は **rigcheck 判定窓 `[0.486,0.496]` と監査記録・経験式記述の更新** と読み替えるのが正確（HOLD）
- Feature flag は **compile-time gate** 採用（runtime flag は RT path 負荷のため不採用）
- flag 名は compile-time macro **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** で全文統一（`kUseV2Oversampler` は採用しない）

#### 2.5.1 Phase 0 判定基準（v1.4・A〜I 厳密化）

| ID | カテゴリ | 判定条件 | 合否 |
|----|----------|----------|------|
| **A** | **Constant / DC** | 定常 DC 入力での round-trip gain = 1.0（=20 log10 1.0 = 0 dB） | □ PASS / □ FAIL |
| **B** | **Low-frequency passband** | 1 kHz 以下の振幅保存（規定 tolerance 内、ex ±0.1 dB） | □ PASS / □ FAIL |
| **C** | **Passband** | 10 kHz、0.25 Fs_in、0.45 Fs_in での amplitude / phase 応答が期待値範囲（案 E 採用前と一致） | □ PASS / □ FAIL |
| **D** | **Stopband** | 0.55 Fs_in 〜 0.95 Fs_in での stopband attenuation が案 E 採用前と同等（relative attenuation preserved） | □ PASS / □ FAIL |
| **E** | **Image / alias** | Fs_in/2 〜 Fs_out/2 での image / alias rejection が案 E 採用前と同等（案 E は FIR 不変なので不変と予想、Phase 0 で実測） | □ PASS / □ FAIL |
| **F** | **Latency** | `AudioEngine.Processing.Latency.cpp` の **`groupDelaySamplesAtStageRate = taps[stage] - 1`（up+down 合算・stage レート）** 計算と実測が一致（`isSymmetricUpDown=true` の contract 維持）。base レート換算は `× (baseRate / stageRate)` | □ PASS / □ FAIL |
| **G** | **Float / Double** | DSPCoreFloat / DSPCoreDouble 経路で振幅・位相が equivalent within tolerance | □ PASS / □ FAIL |
| **H** | **Block / reset boundary** | block size 変化 / 初回 block / repeated process で semantic 結果同一（`reset()` / `clearAllStages()` の境界挙動） | □ PASS / □ FAIL |
| **I** | **SoftClip local OS** | `prepareSingleStage(31, 90.0)` 経路で round-trip / 閾値 / 入出力整合（taps=31, center=15, centerParity=1, convParity=0） | □ PASS / □ FAIL |

**全 PASS → Phase 1 eligibility**。いずれか FAIL → 案 E を見直し、案 D 統合や tap 再設計に方針変更。

#### 2.5.2 Phase 0 測定対象（v1.4・Nyquist / image band を分離）

```
Input passband:
  - 50 Hz
  - 1 kHz
  - 10 kHz
  - 0.25 Fs_in
  - 0.45 Fs_in

Input Nyquist:
  - 0.49 Fs_in

Image / transition band:
  - Fs_in/2 〜 Fs_out/2

Output-side stopband:
  - 0.55 Fs_in 〜 0.95 Fs_in
```

各 ratio について input/output Nyquist・image band・passband・stopband を **分離して測定**することで、案 E の「FIR shape 不変」という仮説を厳密に検証可能。

### 2.6 ユーザー監査結果（v1.4）

#### 2.6.1 oracle レビュー実施履歴

| 派遣 | 目的 | 結果 |
|------|------|------|
| **ora-1**（`ses_f4293306affebaY6j4rxkhOc76`） | 包括的 8 観点レビュー | **terminal evidence 取得失敗**（status uncertain）。2 回リトライ後 error |
| **ora-2**（`ses_f427e237bffegbt7leHnjIDD6F`） | 焦点絞込：A/B/C/D 推奨案の即答 | ✅ **完了：C 案推奨 → ユーザー監査で撤回** |

#### 2.6.2 ユーザー監査による C 案撤回（v1.4 確定）

`ConvoPeq.md`（Generated 2026-09-19 22:08:26）を基準ソースとして確認した結果、oracle の C 案推奨は **現行 `CustomInputOversampler` の実装と整合しない**ことが確定。

**C 案の問題点（ユーザー監査原文）**:
> 案 C は `rawCoeffs[centerTap] = 1.0` かつ非中心をゼロにする案。これは `h[n] = δ[n-center]` という単なる pure delay になる。つまり：
> - `convCoeffs = 0` になる
> - `convValue = 0`、`output[convParity] = 0`、`output[centerParity] = input`
> - 実質的な zero-stuffing / delay 構造
> - **anti-alias / anti-imaging filtering を失う**
>
> 「C は half-band PR の教科書条件」という oracle の説明は、この `CustomInputOversampler` の実装に対しては採用できない。

**現行コードの FIR 構造（287-390 行で確認）**:
```cpp
// 1. sinc × Kaiser 窓で FIR 生成
for (n = 0; n < stage.taps; ++n) {
    rawCoeffs[n] = sinc * window;
}
// 2. center と同 parity の非 center タップをゼロ化（parity 間引き）
for (n = 0; n < stage.taps; ++n) {
    if (n != centerTap && ((n & 1) == centerParity))
        rawCoeffs[n] = 0.0;
}
// 3. sum で正規化
// 4. center = 0.5 に強制、non-center 合計 = 0.5 に再スケール
```

→ 現行 FIR は既に half-band 構造（center=0.5、non-center 合計=0.5、parity 間引き済み）。
C 案はこれを pure delay に置き換えてしまうため、anti-aliasing を失う。

#### 2.6.3 ユーザー監査の最終結論：案 E 採用

ユーザー監査原文:
> むしろ最初に検証すべき修正は「center phase の ×2」。現在 `convValue *= 2.0` だけが存在する。一方、標準的な interpolation-by-2 の gain convention では、zero insertion によってサンプル密度が 2 倍になるため、FIR polyphase 全体をその convention に合わせるなら、**両 phase が同じ DC gain になる必要がある**。

> **FIR の形状を破壊せず、既存の half-band FIR をそのまま利用**する方向。第一候補は `centerValue *= 2.0` を追加し、`convValue *= 2.0` を維持すること。

#### 2.6.4 案 E 採用時の影響評価

| 項目 | 内容 |
|------|------|
| **音量変化** | 段あたり **+2.5 dB**（=20×log10(1/0.75)）。最大 +7.5 dB（ratio 8 で 3 段） |
| **engine fit** | **`0.75^N` の扱いは Phase 0/3 で再判定**。immediate 削除不可（empirical compensation として機能している可能性、HOLD） |
| **SoftClip 局所 OS** | `prepareSingleStage(31, 90.0)` の閾値再校正（案 E で up=1.0, down=1.0 → SoftClip 入力の DC 振幅が変化）。Phase 0 判定基準 I で実測 |
| **`static_assert`** | `AudioEngine.Processing.Latency.cpp:6-8` の `static_assert(isSymmetricUpDown=true)` は **FIR tap のみ参照**するため FIR 不変なら保持される（予想）。Phase 0 判定基準 F で実測 |
| **フィーチャーフラグ** | compile-time macro **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`**（`#define CONVOPEQ_CORRECT_POLYPHASE_GAIN`） |
| **rollback** | compile-time flag の再ビルドまたは 1 行 revert（revert 範囲は `centerValue *= 2.0;` の 1 行） |
| **修正行数** | **1 行追加**（`interpolateStage` に `centerValue *= 2.0;`） |
| **anti-aliasing** | **FIR coefficient shape unchanged → relative frequency-response shape 不変と予想**。Phase 0 判定基準 D/E で実測確認 |
| **latency** | **FIR tap 不変 → group delay 不変と予想**。Phase 0 判定基準 F で実測 |

#### 2.6.5 Phase 0 着手前の検証チェックリスト（v1.4）

- [ ] 既存テスト（`prepareStage` を呼ぶテスト）の期待値が half-band FIR と整合するか
- [ ] `CustomInputOversampler` の利用箇所（DSPCoreFloat / DSPCoreDouble / softClipOS）が half-band FIR と整合するか
- [ ] `prepareSingleStage(31, 90.0)` の挙動確認（taps=31, center=15, centerParity=1, convParity=0）
- [ ] 現行コードの polyphase gain convention の理論的根拠を文献調査（JUCE `dsp::Oversampling` の実装パターンとの比較）
- [ ] **B-1-P0 GATE 設定完了**（§2.8 参照）

#### 2.6.6 Phase 1 着手前チェックリスト（v1.4・案 E 採用判断後）

- [ ] Phase 0 の characterization 結果で §2.5.1 判定基準 A〜I が全 PASS
- [ ] 案 E 採用で **constant-signal / DC gain round-trip = 1.0** になることが数値確認済み
  ※ **「perfect reconstruction」は Phase 0 で PR が数値確認されるまで使用しない**
- [ ] **latency contract** が不変であることが数値確認済み（Phase 0 判定基準 F）
- [ ] **`isSymmetricUpDown=true` 意味的整合** が数値確認済み（FIR tap 不変 + group delay 不変）
- [ ] "anti-aliasing 維持" は Phase 0 判定基準 D/E で実測確認（コード変更のみから確定不可）
- [ ] `kOutputHeadroom` 等 calibration 値への影響予測が ±2.5 dB/段 で説明可能

### 2.7 再監査で得た教訓（v1.4・表現強化）

1. **oracle の「教科書の答え」は現行実装と一致しない場合がある**: C 案は half-band FIR の教科書条件としては正しいが、本 `CustomInputOversampler` のように **既に half-band FIR で構築されている実装**に対して適用すると pure delay に退化してしまう
2. **「gain convention 不整合」と「FIR 構造不整合」を分離して考える**: 修正は gain convention（polyphase 配分）の対称化に限定し、FIR 構造は維持する方が安全
3. **engine fit の `0.75^N` は即時削除不可（HOLD）**: empirical compensation として機能している可能性がある。Phase 0 の characterization で再判定
4. **Phase 0 を必ず先行**: 実装前に DC / impulse / 各周波数 / Nyquist 近傍 / group delay の read-only measurement で「defect か意図された convention か」を確定する
5. **「案 E = 仮説」**: 案 E は **「正しいことが証明された修正」ではなく「現行実装の gain imbalance を最小変更で修正する有力仮説」** として扱う。Phase 0 で確定後、Phase 1 実装 GO
6. **「perfect reconstruction」の表現に注意**: DC gain = 1.0 は言えるが、一般信号での PR（H_up(z) × H_down(z) = z^(-L) 全帯域保証）は案 E だけでは自動的には成立しない。Phase 0 判定基準 A〜I で多面的に確認する
7. **「anti-aliasing 維持」はコード変更のみから断定不可**: FIR shape 不変でも、実機での周波数応答は Phase 0 判定基準 D/E で実測確認する
8. **feature flag は compile-time gate**: DSP の RT path で runtime flag 分岐を避けるため、compile-time gate（`#define CONVOPEQ_CORRECT_POLYPHASE_GAIN`）を採用
9. **姉妹実装の gain convention 差異**: `src/TruePeakDetector.{h,cpp}` の `interpolateStage` は **`CustomInputOversampler` と別の gain convention**（両 polyphase とも ×2 補正なし）。`TruePeakDetector` は True Peak 計測専用経路で main OS を経由しないため影響範囲外だが、Phase 0 の参考測定として round-trip DC gain を実測・記録する（案 E とは別 work item として残置も可）

### 2.8 B-1-P0 GATE（v1.4 確定・production 0 変更 gate）

Phase 0 を production source に対する **変更ゼロの gate として独立定義**し、後続の calibration 改修が混入しないようにする。

```
B-1-P0 GATE
────────────────────────────────────
production src/:
    modified = 0
    staged   = 0

measurement/harness:
    test-only additions permitted

commit:
    forbidden

calibration:
    forbidden

threshold update:
    forbidden

engine-fit 0.75^N:
    unchanged
────────────────────────────────────
```

**Phase 0 終了時の PASS/FAIL 記録**:

| ID | 判定基準 | 結果 |
|----|----------|------|
| P0-A | Constant/DC（round-trip gain = 1.0） | □ PASS / □ FAIL |
| P0-B | Low-frequency passband amplitude preservation | □ PASS / □ FAIL |
| P0-C | Passband ripple（0.25 / 0.45 Fs_in） | □ PASS / □ FAIL |
| P0-D | Stopband attenuation（0.55〜0.95 Fs_in） | □ PASS / □ FAIL |
| P0-E | Image / alias rejection | □ PASS / □ FAIL |
| P0-F | Latency contract | □ PASS / □ FAIL |
| P0-G | Float / Double equivalent | □ PASS / □ FAIL |
| P0-H | Block / reset boundary | □ PASS / □ FAIL |
| P0-I | SoftClip local OS | □ PASS / □ FAIL |

**P0 全 PASS → Phase 1 eligibility**。いずれか FAIL → 案 E を見直し、案 D 統合や tap 再設計に方針変更。

---

## 3. §3 harness 欠陥（F-2/F-3/F-4）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC が意図と一致しない

#### 3.1.1 根本原因

```cpp
// src/audioengine/AudioEngine.h:1416-1432
void setAutoGainStagingEnabled(bool enabled) noexcept
{
    // ...
    // ★ Bug#4: Auto Gain 有効時は EQ AGC を無効化（二重ゲイン補正防止）
    getEQProcessor().setAGCEnabled(!enabled);  // ← ★ ON → AGC OFF、OFF → AGC ON
    submitRebuildIntent(...);
}
```

`BassBuzzMeasurement.cpp:1799-1803` の呼出順:
```cpp
configureProbeFlatEQ(e);                  // 1. setEQAGCEnabled(false)
e.setAutoGainStagingEnabled(false);       // 2. setAGCEnabled(!false)=setAGCEnabled(true) ← 上書き
e.setEQFilterStructure(Parallel);
e.setEqBypassRequested(false);
```

#### 3.1.2 修正案

| 案 | 内容 | 評価 |
|----|------|------|
| **A（推奨）** | 呼出順を逆にする: `setAutoGainStagingEnabled(false)` → `configureProbeFlatEQ(e)` | 最小変更。eqdiag モード（1833 行目）と同じパターンで整合 |
| **B** | `setAutoGainStagingEnabled` から `setAGCEnabled(!enabled)` を削除 | 設計変更（Bug#4 の前提が崩れる）|
| **C** | `configureProbeFlatEQ` の最後に `setAutoGainStagingEnabled(false)` を追加（順序固定） | コード複雑化 |

**推奨**: **A**。`--buzz-rigcheck=eq` のブロック全体を `eqdiag` パターンに揃える。

#### 3.1.3 検証（v1.4・表現修正）

- 修正後: `[BUZZ] RIGCHECK(eq) gainpath: staging=0 eqAGC=0 ...` を確認
- **窓 `[0.486,0.496]` は数学的に必ず不変とは言わない**（F-2 は EQ AGC の実効状態を変える修正。AGC ON 状態の既存窓に対する整合性は実測で再確認する）
- **B-1 の原因切り分けとは独立**: AGC を強制 OFF にした診断モードでも ratio 0.4912 が不変だったため、F-2 修正は B-1 の根本原因に影響しない
- 修正後に `gainpath=staging=0 / eqAGC=0` を確認し、既存 window `[0.486,0.496]` を再測定する
- 窓の境界が ±ε を超えて変動する場合は、判定窓の再校正を Phase 1 の前段で実施

### 3.2 F-3: EQ dry/wet 混合の潜在欠陥

#### 3.2.1 確認事項

`src/eqprocessor/EQProcessor.Processing.cpp:978-1015`（ブレンド本体。970 起点は gain ramp を含む）:
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

`dryCopyBase` は `bypassTransitionActive && dryBypassBuffer 容量十分` のときのみ充填される（570-580 行目）。

#### 3.2.2 影響評価

- **定常状態の B-1（−5.11 dB）の原因ではない**（定常では `bypassTransitionActive=false` でブレンド自体スキップ）
- 遷移時のみの潜在欠陥
- ユーザーが bypass を ON/OFF するとき wet が wet-only 減衰する可能性（dry 項なし）

#### 3.2.3 修正案

| 案 | 内容 | 評価 |
|----|------|------|
| **A** | dry バッファ確保失敗時は遷移を遅延（最大 N ms）してリトライ | 複雑・RT 影響 |
| **B** | dry バッファ確保失敗時は無音フェード（mute crossfade） | 聴感変化小 |
| **C** | ドライコピー失敗時は事前コピー（process 開始前に確保済みに） | prepareToPlay で保証 |
| **D（推奨）** | **記録のみ**。本セッションでは B-1 と分離して別 work item | 既存動作変更なし・リスク最小 |

**推奨**: **D**。B-1 と独立しており、定常 B-1 の原因ではないため、本セッションでは記録のみ。

### 3.3 F-4: `--buzz-rigcheck=ir` の convolver 未有効化

#### 3.3.1 根本原因

`BassBuzzMeasurement.cpp:1744-1762`:
```cpp
e.setEqBypassRequested(true);
e.setConvolverBypassRequested(true);   // ← 1745: 一律 ON
if (!waitBacklogZero(e, 30000)) ...;
sleepPump(1500);
if (rigCheckMode == "ir")
{
    e.getConvolverProcessor().loadImpulseResponse(irFile, false);
    // ← setConvolverBypassRequested(false) が無い！
}
```

`ir` モードでは IR をロードしても convBypassed が true のまま → **出力は dry コピー**。

#### 3.3.2 修正案

| 案 | 内容 | 評価 |
|----|------|------|
| **A** | `ir` モードに `setConvolverBypassRequested(false)` を追加 | 既存窓 `[0.880,0.897]` の意味が変わる（dry → wet） |
| **B（推奨）** | irewet モードのみ残し、`ir` は記録のみ（既存 dry 測定基準を維持） | 既存窓 `[0.880,0.897]` は dry のまま（変更なし）。wet 対照は `irwet<digit>` を使用 |

**推奨**: **B**。`irwet<digit>` モード（1778 行目で既に実装）で wet 対照をカバー済み。`ir` モードの wet 有効化は将来別 work item。

#### 3.3.3 検証

- 修正不要（記録のみ）
- `irwet<digit>` モードで既存 wet 経路の gain 検証が可能

---

## 4. §4 F-1: PublicationValidatorIsolationTests 移行・退役

### 4.1 現状

- `src/tests/PublicationValidatorIsolationTests.cpp`（519 行・TEST_F 34 + TEST 4 = 38 ケース）
- **CMake 未登録**（`add_executable` 39 ターゲットに該当なし）
- **gtest 使用は本ファイルのみ**（リポジトリ内 gtest 依存 0）
- `validator_.checkNoConflictingTransitions(...)` を **9 箇所**で呼ぶが、現行 header では `private`
- `FRIEND_TEST` 0 件 → **コンパイル不可**
- `tools/build-debug.bat:29` の `--target PublicationValidatorIsolationTests` は stale（2026-06-03 削除済み）

### 4.2 移行方針（v1.4・MIGRATE CASES THEN RETIRE・case classification 先行）

#### Phase 0: Case Classification（v1.3 で追加・機械的移植の前に実施）

**34 ケースを機械的に移植する前に**、各ケースを以下 4 軸で分類する:

| 分類軸 | 判定 |
|--------|------|
| **A. 公開 API coverage** | 既存 public API（`validatePublication` / `validateSemanticConsistency` / `validateTopology` / `validateResources`）で再現可能か？ |
| **B. semantic equivalence** | 元の private API 経由と、公開 API 経由の結果が意味的に等価か？ |
| **C. error contract** | error string を **文字列そのもの** で契約にするのか、**error category**（`ValidationFailureReason` enum）で契約にするのか |
| **D. 優先度** | `CrossfadeAuthority` 4 件（直接カバレッジ維持）/ validator 25 件 / `CheckTransition_*` 9 件 |

**error contract 方針（v1.3 で明示）**:
- **error category**（`ValidationFailureReason` enum）を **第一優先の契約**とする
- error string は参考情報（ログ・デバッグ用）であり契約としない
- 移行後のテストは `result.failureReason == ValidationFailureReason::Xxx` を assert
- string 完全一致は contract 化しない（将来の実装詳細の安定性のため）

#### Phase A: `CrossfadeAuthorityRegressionTest` 4 ケース移植（最優先・API 不変）

| 移行先 | 内容 |
|--------|------|
| 既存カスタムハーネス（`AudioEngineHarness` 系） | `Decision{needsCrossfade, fadeTimeSec}` / `evaluate(old, new, policy)` の 4 ケース |
| 検証 | production 側 API 不変・機械的移植可（Phase 0 の分類不要） |

#### Phase B: validator 34 ケース移植（Phase 0 分類後）

| ケース種別 | 移行方法 |
|------------|----------|
| `ValidatePublication_*`（7） / `ValidateSemanticConsistency_*`（1） / `ValidateTopology_*`（6） / `ValidateResources_*`（11）= **25 件** | 公開メソッド直接使用。`world` を構築してテスト。**error contract は `ValidationFailureReason` enum** |
| `CheckTransition_*`（8） / `CheckNoConflictingTransitions_*`（1）= **9 件** | `checkNoConflictingTransitions` が private のため **`validatePublication` 経由**で `result.failureReason` enum 検証に書き換え |

#### Phase C: テストファイル退役

| 項目 | 内容 |
|------|------|
| `PublicationValidatorIsolationTests.cpp` 削除 | 移行完了後 |
| `tools/build-debug.bat:29` 修正 | stale 行を除去（`--target PublicationValidatorIsolationTests` を別ターゲットへ） |
| 関連 CMake / presets | 既に未登録のため変更なし |

### 4.3 実装パターン（既存 custom main() への統合）

```cpp
// 例: RuntimePublicationValidator 移行先（既存カスタムハーネス）
// v1.3: error category を契約とする（string ではない）
TEST(ValidatorMigration, ValidatePublication_SemanticConsistency_Success)
{
    RuntimePublicationValidator validator;
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 100;
    // ... (元のテストの world 構築をコピー)
    
    const auto result = validator.validatePublication(world);
    EXPECT_TRUE(result.isValid);
    EXPECT_EQ(result.failureReason, ValidationFailureReason::None);
    // errorMessage 文字列は contract 化しない（実装詳細の安定性のため）
}
```

カスタム `main()` であれば、`<gtest/gtest.h>` 依存は不要。

### 4.4 影響度

- test-only 修正（production `src/` 変更なし）
- validator 動作の直接カバレッジが減少するが、`RuntimePublicationBridge` 経由（`AudioEngine.h:3670`）の間接カバレッジは維持
- `CrossfadeAuthority` 4 ケースがカスタムハーネスへ移植されることで、直接カバレッジは維持される
- **error category 契約**により、error string の揺れに対するテストの安定性が向上

---

## 5. §5 別課題（既存記録・本セッション対象外）

| ID | 内容 | 方針 |
|----|------|------|
| **B-3** | timestamp-based capture（`out.size()/2.0s` は非一様 callback rate を完全補正しない） | 保留。flipIndex 誤差 <0.5% 実測で現行 transition probe 目的には十分 |
| **D-1** | build identity gate の M1/M2 欠陥（E-G3-3 既知・未修正） | 保留。stamp 再作成で回避継続 |
| **D-2** | headroom ランタイム統合（4 ランタイム混在） | 保留。既存 fork litellm パッチ運用継続 |
| **R-1** | `--buzz-flip-eqgain=` 受理して破棄（設計上ステップ固定） | 保留。silent-ignore 系の別パターンとして記録 |
| **R-2** | `parseHcIdx` / `parseLcIdx` と `stod` / `stoi` / `stof` の不正入力で未捕捉例外 | 保留。fail-closed 化は別 work item |

### 5.1 §5 の修正方針（将来 work item）

各項目についての方針案を提示するに留め、本セッションでは実装しない。

#### B-3: timestamp-based capture

```cpp
// 案: 受信時のホストタイムスタンプを記録し、平均実効レートを再計算
const auto captureTimestamp = std::chrono::high_resolution_clock::now();
session.captureTimestamps.push_back(captureTimestamp);
// 解析時: captureTimestamps の差分から局所実効レートを推定
```

実装コスト: 中・gain は限定的（flipIndex 誤差 0.5% が問題になる場面が少ない）

#### D-1: build identity gate M1/M2

- M1: ベアシェル/icx 起動時の cache 素字化 → `find_package` 段階でキャッシュハッシュ検証
- M2: commit 毎の `source_revision` 変更 → COHERENCE-4 fail-closed → stamp 自動化（commit hook で再 stamp）

実装コスト: 高（外部システム統合）

#### D-2: headroom runtime 統一

```bash
# 案: 4 ランタイム → 1 ランタイムに統合
uv tool install --force headroom-ai --with fastapi --with uvicorn
# Startup の .lnk / bat / vbs 3 起動体を 1 つの .lnk に統合
```

実装コスト: 中（テスト運用）・リスク: 既存 proxy 起動失敗の可能性

#### R-1: silent-ignore 修正

```cpp
// 案: --buzz-flip-eqgain= 受理時に "[BUZZ] INFO: --buzz-flip-eqgain=N is design-fixed (total -3dB step)" を出力
else if (a.rfind("--buzz-flip-eqgain=", 0) == 0) {
    std::fprintf(stderr, "[BUZZ] INFO: --buzz-flip-eqgain=%s accepted but ignored (design-fixed -3dB step)\n",
                 a.substr(20).c_str());
    opt.flipKind = 4; opt.flipValue = 0;
}
```

実装コスト: 極小

#### R-2: stod/stoi 例外の try/catch

```cpp
// 案: 共通ヘルパー parseIntOrFail / parseFloatOrFail で try/catch
inline int parseIntOrFail(const std::string& flag, const std::string& v) {
    try { return std::stoi(v); }
    catch (const std::exception&) {
        std::fprintf(stderr, "[BUZZ] FAIL: %s expects integer (got '%s')\n", flag.c_str(), v.c_str());
        std::exit(2);
    }
}
```

実装コスト: 極小・全 `std::stoi` / `std::stod` / `std::stof` 呼び出しを統一

### 5.2 §5 の着手優先度（将来）

1. **R-2**（極小・fail-closed 化・即時効果）
2. **R-1**（極小・silent ignore 解消）
3. **B-3**（中・timestamp 化）
4. **D-2**（中・環境統一）
5. **D-1**（高・build identity gate 修正）

---

## 6. 推奨する実装順序（v1.2）

```
[Step 1] §3 F-2/F-3/F-4（harness cleanup・test-only）
  - F-2: 呼出順修正（1 行差替）
  - F-3/F-4: 記録のみ
  - 検証: rigcheck=eq で AGC=0 を確認

[Step 2] §4 F-1（テスト移行）
  - CrossfadeAuthority 4 ケース → カスタム main ハーネス
  - validator 34 ケース → 公開メソッド経由に書き換え
  - PublicationValidatorIsolationTests.cpp 退役

[Step 3] §1 O-1〜O-6（commit 意思決定）
  - ユーザー判断待ち
  - 該当ファイルを commit

[Step 4] §2 B-1（DSP 修正・破壊的変更・案 E 採用）
  - Phase 0: Characterization（read-only・production 変更 0）
    - DC / impulse / 各周波数 / Nyquist / 0.5 Fs_processing / group delay / round-trip / up-only / down-only 測定
    - 「0.75 = defect か意図された convention か」確定
  - Phase 1: Feature Flag 導入（破壊なし）
    - interpolateStage に centerValue *= 2.0 を追加（案 E）
    - フィーチャーフラグ `CONVOPEQ_CORRECT_POLYPHASE_GAIN`（compile-time macro・既定 0）
    - 既存テスト全 PASS を確認
  - Phase 2: Flag ON で限定検証
    - 内部 dogfood で ratio 1/2/4/8 全段検証
    - Phase 0 の測定と比較（round-trip = 1.0 になることを確認）
    - engine fit の 0.75^N の必要性を再判定
  - Phase 3: 全段 ON（破壊的変更）
    - フィーチャーフラグを true に固定
    - kOutputHeadroom 等 calibration 値を再校正
    - rigcheck 判定窓 [0.486, 0.496] を新 gain 値に更新
    - 全テスト + 新 rigcheck 窓で PASS を確認
  - Phase 4: フラグ削除（次メジャーリリース）

[Step 5] §5 別課題（将来 work item 化）
  - R-2（最優先・極小）
  - R-1（次・極小）
  - B-3, D-2, D-1
```

**Step 4 の Phase 0 が完了するまで、Phase 1 以降の着手は不可**。Phase 0 で「0.75 = defect」が確定しない場合、案 E を見直し、案 D 統合や tap 再設計に方針変更する。

---

## 7. 検証計画

### 7.1 単体検証

| 検証項目 | コマンド | 期待 |
|----------|----------|------|
| §3 F-2 修正 | `ConvoPeq.exe --buzz-rigcheck=eq --buzz-dur=2.0` | `[BUZZ] RIGCHECK(eq) gainpath: staging=0 eqAGC=0 ...` |
| §4 F-1 移行 | `ctest --test-dir build --output-on-failure` | 移行先テストが PASS、PublicationValidatorIsolationTests 削除済 |
| §2 B-1（Phase 1） | `ConvoPeq.exe --buzz-osdirect=2 --buzz-dur=1.0` | `[OS_DIRECT] round-trip=1.000000`（フィーチャーフラグ OFF 時 0.75） |

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

## 8. ロールバック計画

| Step | ロールバック方法 |
|------|------------------|
| §1 commit | `git revert <sha>` |
| §2 B-1 Phase 1〜3 | compile-time flag `CONVOPEQ_CORRECT_POLYPHASE_GAIN=0` で rebuild または 1 行 revert（`centerValue *= 2.0;` の `#if` 内側を削除） |
| §3 F-2 | 1 行 revert（呼出順を元に戻す） |
| §4 F-1 | 移行先テストが残る場合、ファイルを git から復元 |
| §5 別課題 | 実装しないため不要 |

※ flag 名は **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** で全文統一（`kUseV2Oversampler` は採用しない）。compile-time gate のため rollback 時は rebuild または `#if` 内側 1 行 revert。

---

## 9. 影響度まとめ

| 区分 | 影響範囲 | 影響度 | 段階リリース |
|------|----------|--------|--------------|
| §1 commit | リポジトリ履歴のみ | 極小 | 任意 |
| §2 B-1（案 E） | **全オーディオ経路**（最大 +7.5 dB 想定） | **極大**（1 行追加で済む） | 必須（Phase 0→1→2→3→4） |
| §3 F-2 | test-only | 小 | 不要 |
| §3 F-3 | test-only（記録のみ） | なし | 不要 |
| §3 F-4 | test-only（記録のみ） | なし | 不要 |
| §4 F-1 | test-only | 中 | 不要 |
| §5 別課題 | 環境 or CLI | 極小〜中 | 不要 |

---

## 10. 監査ログ・参照

- 前スナップショット: `doc/work113/residual_tasks_20260919.md`
- 本セッション報告書: `doc/work113/residual_tasks_20260920.md`
- 関連監査記録:
  - `doc/work104/bass_buzz_measurement_20260917.md`
  - `doc/work105/ir_runtime_contract_and_remeasure_20260917.md`
  - `doc/audit/ConvoPeq_BugList_and_FixPlan_2026-09-13.md`
  - `doc/audit/ConvoPeq_Bug_Verification_2026-09-13.md`
  - `doc/audit/ConvoPeq_SampleRate_Reverification_2026-09-13.md`
- 関連ソース:
  - `src/CustomInputOversampler.h/.cpp`（B-1）
  - `src/audioengine/AudioEngine.h:1416-1432`（F-2 根本）
  - `src/audioengine/AudioEngine.Processing.Latency.cpp:6-8`（static_assert）
  - `src/eqprocessor/EQProcessor.Processing.cpp:978-1015`（F-3 ブレンド本体）
  - `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp:1700-1850`（F-2/F-3/F-4）
  - `src/tests/PublicationValidatorIsolationTests.cpp`（F-1）
  - `src/audioengine/RuntimePublicationValidator.h:101`（F-1 private）

---

## 11. ユーザー判断待ち項目

| 項目 | 選択肢 | 推奨 |
|------|--------|------|
| §1 O-1 案選択 | A: 最小（[OS_DIRECT]のみ）/ B: 全 +642 行 / C: 退役 | **A** |
| §1 O-2 commit 方針 | O-1 と同梱 / 独立 | **独立** |
| §1 O-6 push | 即時 / 別タイミング / ユーザー手動 | **ユーザー手動** |
| §2 B-1 修正案 | **E: interpolateStage に `centerValue *= 2.0` 追加（既存 half-band FIR 維持）** | **E（ユーザー監査確定）** |
| §3 F-2 修正 | A: 呼出順逆 / B: setAGCEnabled 削除 / C: 順序固定 | **A** |
| §4 F-1 移行 | Phase A→B→C 全実施 / Phase A のみ / 記録のみ | **Phase A→B→C 全実施** |

### §2 B-1 採用案の前提条件（v1.4）

v1.3 で採用した **案 E（polyphase gain convention 対称化）** は「**有力仮説**」であり、以下の Phase 0 characterization が全 PASS しなければ Phase 1 実装 GO とはしない：

1. **constant-signal / DC gain round-trip = 1.0** が達成される（数値確認）
2. **anti-aliasing / anti-imaging が維持**される（周波数応答確認、Phase 0 判定基準 D/E）
3. **latency が不変**（group delay 確認、Phase 0 判定基準 F）
4. **SoftClip 局所 OS（`prepareSingleStage(31, 90.0)`）が正常動作**（Phase 0 判定基準 I）

※ **「perfect reconstruction」表現は Phase 0 で PR が数値確認されるまで使用しない**（constant-signal / DC gain のみが現状の確定事項）

Phase 0 で前提が崩れた場合は、案 E を見直し、案 D 統合（offset 補正）や tap 再設計を検討。

### 不採用案（A/B/C/D）

| 案 | 結論 | 理由 |
|----|------|------|
| A: `decimateStage` に ×2 | 不採用 | round-trip = 1.125 で過剰 |
| B: `interpolateStage` の ×2 削除 | 不採用 | round-trip = 0.375 で過少 |
| C: half-band FIR 再設計（center=1.0） | **撤回** | pure delay 化、anti-aliasing 消失 |
| D: 統合（offset 補正） | 不採用 | `isSymmetricUpDown=true` の意味的整合に難 |

### v1.4 最終判定表

| 項目 | 判定 | 備考 |
|------|------|------|
| §1 O-1〜O-6 | **GO 候補** | commit/push の意思決定として分離 |
| §3 F-2 | **GO 候補** | 呼出順修正（1 行差替） |
| §3 F-3 | **記録継続** | 遷移時のみの潜在欠陥、定常 B-1 の原因ではない |
| §3 F-4 | **記録継続** | 既存 dry 測定基準維持、`irwet<digit>` で wet 対照 |
| §4 F-1 | **GO 候補** | case classification 先行 → 公開 API 経由移植 |
| §5 | **保留継続** | 将来 work item 化 |
| **B-1 Phase 0** | **GO** | **B-1-P0 GATE（§2.8）準拠** |
| B-1 Phase 1 | **HOLD** | Phase 0 全 PASS が前提 |
| **B-1 案 E** | **仮説として採用** | 強い仮説だが Phase 0 で検証前は確定ではない |
| B-1 「perfect reconstruction」 | **表現修正** | DC gain 1.0 は言えるが PR 全帯域保証は未証明 |
| B-1 calibration 変更 | **HOLD** | Phase 3 まで保留 |
| `0.75^N` engine-fit 削除 | **HOLD** | empirical compensation の可能性、Phase 0/3 で再判定 |
| compile-time flag 名 | **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** | `kUseV2Oversampler` は採用しない（修正範囲を正確に表現） |

---

**本書は方針案です（v1.4・ユーザー監査 3 ラウンド反映版・最終確定版）。実装着手は本書承認後**。
**§2 B-1 の DSP 修正方針：案 E（polyphase gain convention 対称化）を「有力仮説」として採用**。
**Phase 0 characterization 必須（B-1-P0 GATE・判定基準 A〜I）・compile-time flag `CONVOPEQ_CORRECT_POLYPHASE_GAIN`・段階リリース 5 Phase（Phase 0 → 1 → 2 → 3 → 4）**。
**Phase 0 全 PASS → Phase 1 eligibility。Phase 1 以降は Phase 0 結果次第で GO 判定**。
**案 E は「正しいことが証明された修正」ではなく「現行実装の gain imbalance を最小変更で修正する有力仮説」として扱う**。
**「perfect reconstruction」表現は Phase 0 で PR が数値確認されるまで使用しない**。
**承認状態: §1/§3 F-2/§4 F-1 = GO候補 / §3 F-3・F-4/§5 = 保留継続 / §2 = Phase 0 着手承認・Phase 1 以降は HOLD**。
**承認状態: §1 / §3 / §4 / §5 = GO候補 / §2 = Phase 0 完了後に GO 判定**。


レビュー①
結論から言います。

**v1.4 は、v1.3 までの主要な誤りをかなり適切に修正しており、B-1 を「案 E の仮説 → Phase 0 → 条件付き実装」という順序にした点は妥当です。**
また、現行 `ConvoPeq.md` を照合すると、`CustomInputOversampler` が `centerCoeff = 0.5`、非 center 合計も 0.5 に構成され、`interpolateStage()` では現状 `convValue` のみ `×2`、`centerValue` は `×1` のままであることは確認できます。 

ただし、**「最終確定版」として Phase 0 GO を出すには、まだ重大な修正が必要です。**

最大の問題は次です。

> **現在の production code が 0.75 なのに、production 無変更の Phase 0 で「round-trip = 1.0」を PASS 条件にしている。**

これは論理的に成立しません。
さらに、Phase 0 の周波数試験には **Nyquist / stopband / image の正規化軸が混在している**箇所があります。

したがって私の判定は、

> **B-1 Phase 0 の準備着手は GO 可能。ただし「P0-GATE の測定仕様を修正してから実測開始」が条件。Phase 1 は当然 HOLD のまま。**

です。

---

# 1. 総合判定

| 項目           | 判定      | コメント                      |
| ------------ | ------- | ------------------------- |
| §1 O-1〜O-7   | 🟢 概ね妥当 | ただし O-1 A は実装手順をさらに明確化すべき |
| §3 F-2       | 🟢 GO候補 | 呼出順変更という局所修正は合理的          |
| §3 F-3       | 🟢 記録継続 | 本件から分離する判断は妥当             |
| §3 F-4       | 🟢 記録継続 | `ir` と `irwet` の意味整理として妥当 |
| §4 F-1       | 🟢 GO候補 | 25/9 分類＋enum契約化は妥当        |
| §5           | 🟢 保留   | 現セッションから分離する方針は妥当         |
| **B-1 原因仮説** | 🟢 強い   | 現行コードとの整合性は高い             |
| **案 E**      | 🟡 有力仮説 | まだ「正しい修正」とは証明されていない       |
| **Phase 0**  | 🟡 要修正  | 測定設計に重大な論理的不整合あり          |
| Phase 1      | 🔴 HOLD | 現状維持が正しい                  |
| Phase 2      | 🔴 HOLD | Phase 0修正後                |
| Phase 3      | 🔴 HOLD | calibration を後置する判断は正しい   |
| Phase 4      | 🔴 HOLD | さらに先                      |

特に重要なのは、**案 E 自体を否定する必要はない**ということです。

問題は、

> 「案 E を検証するための Phase 0」

になっている部分が、現状では完全にはそうなっていないことです。

---

# 2. 現行 `ConvoPeq.md` との照合

今回の基準ソースとして `ConvoPeq.md` を再確認しました。

現行 `CustomInputOversampler` は、

```cpp
static constexpr bool isLinearPhaseFIR = true;
static constexpr bool isSymmetricUpDown = true;
```

となっています。

また `prepareStage()` では、

```cpp
rawCoeffs[stage.centerTap] = 0.5;
...
scale = 0.5 / nonCenterSum;
```

となっており、center と non-center の総量をそれぞれ 0.5 にしています。

一方、`interpolateStage()` は、

```cpp
convValue *= 2.0;
...
output[outBase + stage.convParity] = convValue;
output[outBase + stage.centerParity] = centerValue;
```

で、**conv phase のみ ×2** です。

したがって、計画書の「現状の gain imbalance」という問題設定は、少なくともコード構造とは整合しています。

また down 側は `processDown()` から各 stage の `decimateStage()` を逆順に通る構造です。

さらに latency 実装では、

```cpp
const double groupDelaySamplesAtStageRate =
    static_cast<double>(taps[stage] - 1); // up + down
```

と実装されており、v1.4 が旧 `(taps-1)×2` 説明を修正した方向は正しいです。

---

# 3. 案 E の数学的評価

ここはかなり良いです。

現状、DC に対して、

* center phase = 0.5
* conv phase = 0.5 × 2 = 1.0

なので interpolation の DC 平均値は

$$
(0.5+1.0)/2=0.75
$$

です。

案 E では、

```cpp
centerValue *= 2.0;
```

を追加するため、

* center = 1.0
* conv = 1.0

になります。

したがって interpolation の DC gain は 1.0。

down 側の FIR 総和は 1.0 なので、steady-state DC の round-trip は

$$
1.0\times1.0=1.0
$$

となる、という計算は妥当です。

### ただし重要

これは、

> **DC / constant-signal gain の修正**

を証明するだけです。

これは計画書自身が「perfect reconstruction」と区別している点も正しいです。

---

# 4. 案 E について一つだけさらに厳密化すべき点

計画書には、

> FIR coefficient shape unchanged → relative frequency-response shape 不変と予想

とあります。

これは方向としては妥当ですが、表現をさらに厳密にするなら、

> **「FIR coefficient 自体は不変なので、係数列そのものの周波数応答は不変。ただし polyphase output のスケーリング変更によって interpolation branch の絶対振幅は変化する」**

とした方がよいです。

つまり、

* normalized shape → 基本的に同じ
* absolute gain → 変わる
* phase → 理論上変わらない
* group delay → 理論上変わらない

です。

これは Phase 0 で測定するべき項目です。

---

# 5. 最大の問題：P0-A が read-only Phase 0 と矛盾

計画書は明確に、

```text
production src:
    modified = 0
```

としています。

そして P0-A は、

> Constant/DC round-trip gain = 1.0

です。

しかし Phase 0 は現行 production code を変更しません。

現行コードは上記のとおり `center ×1 / conv ×2` なので、現在の steady-state DC gain は 0.75 です。

したがって、

```text
Phase 0
  ↓
現行 production code を測る
  ↓
0.75
  ↓
P0-A: 1.0 ?
  ↓
FAIL
```

になります。

これは**案 E が悪いのではなく、Phase 0 の gate 定義が間違っています。**

---

# 6. Phase 0 は「Baseline」と「Candidate」を分離すべき

ここが今回の最大修正点です。

Phase 0 は次の2種類に分けるべきです。

### P0-BL: 現行 baseline characterization

production 変更なしで、

```text
Current implementation
```

を測定する。

期待値は例えば、

```text
DC roundTrip ≈ 0.75
```

です。

これは PASS/FAIL ではありません。

**Baseline capture** です。

---

### P0-CAND: 案 E shadow characterization

production を変更せず、

```text
current algorithm
+
centerValue *= 2.0
```

だけを test harness / reference implementation 上で再現する。

そして、

```text
Candidate E
```

を測定します。

これなら、

```text
production src = unchanged
```

を維持したまま、

```text
baseline = 0.75
candidate E = 1.0
```

を比較できます。

---

# 7. さらに重要：「0.75 が defect」と測定だけでは証明できない

計画書には、

> 「0.75 = defect か意図された convention か確定」

という表現があります。

ここも厳密には修正が必要です。

**測定だけでは design intent は証明できません。**

例えば、

```text
測定結果 = 0.75
```

から言えるのは、

> 現在の実装は 0.75 の DC gain を持つ。

までです。

それが defect かどうかは、

* API contract
* DSP design contract
* 既存テスト
* calibration contract
* JUCE等の基準実装
* 実装者の明示的設計意図

などと照合する必要があります。

したがって Phase 0 は、

```text
A. 現状値を測定
B. 設計契約を確認
C. 案Eをshadow測定
D. baseline/candidate/referenceを比較
```

の4段構造にした方がよいです。

---

# 8. P0-C/D/E の周波数軸に重大な問題

ここはかなり重要です。

現在の計画書は、

```text
0.55 Fs_in ～ 0.95 Fs_in
```

を output-side stopband としています。

しかし、**`Fs_in` が元入力サンプルレートなら、0.55〜0.95 Fs_in は元入力の Nyquist (`0.5 Fs_in`) より上です。**

元の discrete-time input に、

```text
0.75 Fs_in
```

の独立した tone を入れることはできません。

それは既に alias した周波数として観測されます。

したがって、

> `0.55 Fs_in〜0.95 Fs_in` を単純に input frequency sweep として測る

設計は不正確です。

---

# 9. stopband / image / alias は stage-rate で定義すべき

ratio 2 の一段について考えると、

```text
Fs_in
 ↓ interpolation ×2
Fs_stage = 2 Fs_in
```

です。

このとき interpolation filter の image rejection を調べるなら、stage output の周波数軸で、

```text
0.5 Fs_in
```

付近から image band を見る必要があります。

つまり、

```text
元入力基準
Fs_in
```

と、

```text
その stage の sampling rate
Fs_stage
```

を混同しないことが重要です。

---

# 10. ratio 4 / 8 はさらに stage ごとに分ける必要がある

ratio 8 なら、

```text
Stage 0:
Fs0 → 2Fs0

Stage 1:
2Fs0 → 4Fs0

Stage 2:
4Fs0 → 8Fs0
```

です。

したがって P0-E は、

```text
global Fs_in
```

だけではなく、

```text
stage 0 normalized frequency
stage 1 normalized frequency
stage 2 normalized frequency
```

を分けて測るべきです。

これは案 E の「polyphase gain correction」を検証するには重要です。

---

# 11. 推奨する P0 周波数試験

私は次の構造を推奨します。

### A. Full-chain round trip

```text
50 Hz
1 kHz
10 kHz
0.25 Fs_in
0.45 Fs_in
0.49 Fs_in
```

これは元入力帯域内。

---

### B. Stage-local interpolation

各 stage について、

```text
0.01 Fs_stage
0.10 Fs_stage
0.25 Fs_stage
0.45 Fs_stage
0.49 Fs_stage
0.50 Fs_stage
0.51 Fs_stage
...
```

を測る。

---

### C. Stage-local decimation

高レート側から、

```text
0.50 Fs_stage
〜
0.95 Fs_stage
```

を sweep。

ここで初めて、

> alias rejection

を意味のある形で測れます。

---

# 12. P0-C「ripple」も定義不足

現在、

> 0.25 / 0.45 Fs_in

とあります。

しかしこれは2点測定であって、厳密には **ripple** ではありません。

ripple は通常、

```text
frequency interval
    ↓
|H(f)| の max-min
```

として定義します。

例えば、

```text
passband = [0, 0.45 Fs]
rippleDb = max(magnitudeDb) - min(magnitudeDb)
```

のようにする必要があります。

したがって P0-C は、

> 0.25/0.45 Fs のスポットチェック

と、

> passband ripple

を分離した方がよいです。

---

# 13. P0-D「同等」も閾値がない

現在、

> 案 E 採用前と同等

となっています。

これは比較としては良いですが、PASS/FAIL には不足します。

例えば、

```text
candidate attenuation
baseline attenuation

difference <= 0.1 dB
```

など、比較 tolerance を定義する必要があります。

同様に、

* phase
* image rejection
* alias rejection
* latency
* float/double
* block boundary

も同じです。

---

# 14. P0-H はかなり重要なので、block partition invariance を明示すべき

現在、

> block size 変化 / 初回 block / repeated process

とあります。

これを具体化してください。

例えば同一 input stream を、

```text
A: 4096

B: 1024 × 4

C: 256 × 16

D: 513 + 777 + 1000 + ...
```

で処理して、

```text
output waveform
```

を比較する。

さらに、

```text
reset()
```

後の最初のブロックと、

```text
continuous processing
```

の差を確認します。

`CustomInputOversampler` は `upHistory` / `downHistory` を保持し、`reset()` と `clearAllStages()` がこれらを clear する実装なので、この試験は意味があります。 

---

# 15. P0-I SoftClip は「round-trip」だけでは足りない

これは計画書が既にかなり良いところまで来ていますが、もう一段必要です。

`softClipOS` は `CustomInputOversampler` として保持されています。

したがって、

```text
prepareSingleStage(31, 90.0)
```

について、

1. DC gain
2. sine amplitude
3. SoftClip threshold
4. oversampled signal peak
5. downsampled output
6. bypassとの比較

を分離してください。

特に案 E は **SoftClip に入る信号レベルそのものを +2.5 dB 相当変える可能性がある**ため、

> 「oversampler 自体が unity になった」

と

> 「SoftClip の動作点が意図された位置にある」

は別の問題です。

---

# 16. P0-G「Float / Double」は表現を修正した方がよい

現行 `CustomInputOversampler` の API は、

```cpp
AudioBlock<double>
```

です。

さらに float input 側でも `float → double` 変換を行ってから DSP に入る経路があります。

したがって、

> DSPCoreFloat / DSPCoreDouble がそれぞれ別 precision の CustomInputOversampler を動かす

という意味なら、現行実装の記述としては正確ではありません。

P0-G は、

```text
float host input
    ↓
float→double conversion
    ↓
double DSP / oversampler
    ↓
float output
```

と、

```text
double host input
    ↓
double DSP / oversampler
    ↓
double output
```

の**end-to-end equivalence**として定義する方が正確です。

少なくとも「oversampler arithmetic itself: float vs double」と書かない方がよいです。

---

# 17. latency 判定は v1.4 の修正方向で正しい

ここは評価できます。

現行コードは、

```cpp
groupDelaySamplesAtStageRate = taps[stage] - 1;
```

と明示しています。

したがって、

> `(taps−1) サンプル（stage rate、up+down 合算）`

という v1.4 の修正は現行ソースと整合します。

ただし Phase 0 の「実測一致」は、

```text
impulse peak
```

だけではなく、

```text
expected group delay
vs
measured peak / centroid
```

のどちらを採用するか明示した方がよいです。

---

# 18. static_assert についても一段厳密化

現行コードは、

```cpp
static_assert(
    CustomInputOversampler::isLinearPhaseFIR
    && CustomInputOversampler::isSymmetricUpDown
);
```

で latency formula を守っています。

案 E は係数列を変えないので、

```text
isLinearPhaseFIR
isSymmetricUpDown
```

の compile-time property 自体は変わりません。

したがってここは問題ありません。

ただし、

> static_assert PASS = latency contract PASS

ではありません。

static_assert は**設計前提の宣言チェック**です。

Phase 0-F は別途、

```text
impulse response
expected delay
```

を測定する必要があります。

計画書も概ねそうしていますが、この区別を明示するとより堅牢です。

---

# 19. compile-time flag は合理的。ただし実装場所を変更推奨

この設計：

```cpp
#ifndef CONVOPEQ_CORRECT_POLYPHASE_GAIN
#define CONVOPEQ_CORRECT_POLYPHASE_GAIN 0
#endif
```

は技術的には成立します。

ただし、私は **header 内 default macro より CMake build option を authoritative source にする**ことを推奨します。

例えば、

```cmake
option(CONVOPEQ_CORRECT_POLYPHASE_GAIN
       "Enable corrected polyphase gain convention"
       OFF)

target_compile_definitions(
    ConvoPeq PRIVATE
    CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${CONVOPEQ_CORRECT_POLYPHASE_GAIN}>
)
```

のようにする方が、

```text
Release OFF
Release ON
CI OFF
CI ON
```

を明確にできます。

現状の計画では Phase 2 で build option を使い、Phase 3 で source の `#define` を 1 にするとしているので、**flag の authoritative source が途中で変わる**のが気になります。

これは統一した方がよいです。

---

# 20. rollback の「1行 revert」は少し不正確

計画書では、

> 1行 revert

としています。

しかし実際には、

```cpp
#if CONVOPEQ_CORRECT_POLYPHASE_GAIN
    centerValue *= 2.0;
#endif
```

という macro plumbing が存在します。

rollback には少なくとも、

```text
build define = 0
```

または

```text
source revert
```

があります。

したがって、

> 「production behavior の rollback は compile-time flag OFF + rebuild」

を第一手段にするのが正確です。

`centerValue *= 2.0` の1行 revert は**source-level rollback**です。

この2つを分けて書くとよいです。

---

# 21. Phase 3 の calibration はさらに分離すべき

ここは非常に重要です。

現在、

```text
Phase 3
  ↓
gain correction
  ↓
kOutputHeadroom recalibration
  ↓
rigcheck window recalibration
```

となっています。

しかし、

> DSP oversampler correction

と

> output calibration

を同一 commit にすると、問題が発生した際に帰属が難しくなります。

推奨は、

```text
Phase 3-A
  flag permanently ON
  calibration unchanged

Phase 3-B
  measure new production gain

Phase 3-C
  calibration-only commit

Phase 3-D
  rigcheck window-only commit
```

です。

特に現行 `kOutputHeadroom` は別の production constant として存在し、0.891250938... が出力処理で使用されています。

したがって、

> oversampler correction ≠ output headroom correction

を commit 単位でも維持した方が安全です。

---

# 22. `0.75^N` の扱いは v1.4 の方針で正しい

ここは変更不要です。

`0.75^N` は production code の構造的定数ではなく、計画書上の empirical fit として扱い、

```text
Phase 0
  ↓
Phase 2
  ↓
production response
  ↓
Phase 3
  ↓
必要なら削除
```

とするのが妥当です。

いきなり削除しない判断は正しいです。

---

# 23. O-1 `[OS_DIRECT]` は v1.4 の修正が正しい

ここも評価できます。

計画書自身が、

> `[OS_DIRECT]` は `runEqDirectDriveAttribution()` に束ねられている

ため、単純な `git add -p` では分離できない、と認識しています。

これは重要です。

さらに `[OS_DIRECT]` の出力が `PublishPipelineIntegrationTests` の main 経由で出るという実装構造も、監査資料で確認されています。

したがって、

```text
O-1 A
  ↓
main entry を整理
  ↓
runOversamplerDirect()
  ↓
OS_DIRECT
```

という最小配線を別変更として明示するのは正しいです。

---

# 24. ただし O-1 A は「最小 commit」と「最小コード変更」を分けるべき

現在、

> A = 最小資産コミット

となっています。

実際には、

```text
BassBuzzMeasurement.cpp
PublishPipelineIntegrationTests.cpp
```

双方に変更が必要になる可能性があります。

したがって、

> 「最小 commit」

ではなく、

> **「B-1 regression detection に必要な最小実行可能資産」**

と定義した方が正確です。

---

# 25. F-2 は GO でよいが、検証条件を1つ追加

F-2 の呼出順変更：

```cpp
setAutoGainStagingEnabled(false);
configureProbeFlatEQ(e);
```

は合理的です。

ただし `setAutoGainStagingEnabled()` が

```cpp
submitRebuildIntent(...)
```

を行うことが計画書自身にも記載されています。

したがって、

```text
setAutoGainStagingEnabled(false)
↓
configureProbeFlatEQ
↓
setEQFilterStructure
↓
setEqBypassRequested
↓
wait backlog zero
↓
measure
```

までを一連の test procedure としてください。

単に呼出順を変更して即測定すると、state propagation の timing を混ぜる可能性があります。

---

# 26. F-1 は v1.4 の 25/9 分類でよい

ここは改善されています。

監査資料では、

* ValidatePublication = 7
* SemanticConsistency = 1
* Topology = 6
* Resources = 11

で 25。

そして、

* CheckTransition = 8
* CheckNoConflictingTransitions = 1

で 9。

合計 34。

この分類は監査結果とも整合しています。

また `ValidationFailureReason` を string ではなく structured contract とする方針も妥当です。

---

# 27. F-1 にはもう一つ gate を追加した方がよい

private API から public API への移行では、

```text
旧テストが PASS
```

だけでは不十分です。

必ず、

```text
old private-path expected failure
       ==
new public-path failureReason
```

をケースごとに確認する必要があります。

特に `validatePublication()` が複数の validation を順番に実行する場合、

```text
元テストが private check A の failure を狙っていた
↓
public API では前段の check B が先に fail
```

という問題が起こり得ます。

したがって Phase 0 classification の B は、

> **「同じ invalid state を作れば同じ failureReason に到達するか」**

まで含めるべきです。

---

# 28. F-3 / F-4 の「test-only」という分類には注意

これは軽微ですが、記述上は、

> test-only

と、

> production behavior の潜在欠陥

を分けた方がよいです。

F-3 は `EQProcessor.Processing.cpp` の production behavior に存在する問題です。

ただし今回修正しない。

したがって、

```text
実装対象: test-only = false
今回の修正対象: false
```

です。

「test-only bug」と書くと、production code の欠陥ではないように読めます。

---

# 29. Practical Stable ISR Bridge Runtime との整合

添付された `Practical Stable ISR Bridge Runtime` の原則では、

* RT は判断しない
* RT は allocation/delete しない
* Publish/Retire/Crossfade の authority を増やさない
* Validation を bypass しない

という構造が明示されています。 

今回の B-1 案 E は、

```cpp
interpolateStage()
```

という既存 DSP execution path の純粋な数値変更なので、

* Publish
* RuntimeWorld
* Retire
* Crossfade
* lifetime
* ISR bridge

には触れません。

したがって、**architectural boundary を破る改修ではありません。**

これは良い点です。

---

# 30. TruePeakDetector との差異も重要

現行 `TruePeakDetector` は別実装で、`interpolateStage()` は、

```cpp
even = center + conv
odd  = center + conv
```

という構造で、`CustomInputOversampler` と gain convention が異なります。

したがって計画書が、

> 姉妹実装との差異を記録する

としているのは妥当です。

ただし、

> TruePeakDetector の convention を CustomInputOversampler の正解とする

ことは避けるべきです。

これは別設計目的です。

---

# 31. v1.4 に残っている明確な自己矛盾

これはかなり重要です。

v1.4 の改訂サマリでは、

> `--buzz-osdirect=2` は存在しないため削除

と明記されています。

ところが **§7.1 の実際の検証表にはまだ**

```text
ConvoPeq.exe --buzz-osdirect=2 --buzz-dur=1.0
```

と残っています。

さらに期待値も、

```text
round-trip=1.000000
```

となっています。

これは v1.4 の P0-1 修正内容と直接矛盾しています。

したがって、**v1.4 は文書としてはまだ「最終確定版」ではありません。**

ここは単純な記述残りですが、実装時には実害があります。

---

# 32. §6 の「v1.2」も残骸

§6 の見出しが、

> `推奨する実装順序（v1.2）`

になっています。

本文は v1.4 の方針です。

これは機能欠陥ではありませんが、監査文書としては消すべきです。

---

# 33. P0 の最終仕様はこう変更することを推奨

現在の、

```text
P0-A Constant/DC
    round-trip gain = 1.0
```

を、

```text
P0-BL-A Baseline DC characterization
    current production DC gain is measured and recorded

P0-E-A Candidate E DC characterization
    shadow/reference E implementation DC gain ≈ 1.0

P0-E-B Differential
    candidate / baseline transfer difference is exactly attributable
    to the intended center-phase gain correction
```

のようにします。

そして別途、

```text
DESIGN-CONTRACT-A
    Oversampler DC/passband unity-gain is required by contract
```

を設定します。

これで初めて、

```text
Baseline = 0.75
Contract = 1.0
Candidate E = 1.0
```

という三者が揃います。

---

# 34. 推奨する Phase 0 の完全な構造

私なら次のようにします。

```text
Phase 0-0
  source/contract audit
      ↓
  unity-gain requirement confirmed?
      ↓
  YES
      ↓
Phase 0-1
  baseline characterization
      ↓
  current = 0.75
      ↓
Phase 0-2
  shadow candidate E
      ↓
  candidate = 1.0
      ↓
Phase 0-3
  transfer characterization
      ↓
  passband
  stopband
  image
  alias
  phase
  delay
      ↓
Phase 0-4
  block/reset characterization
      ↓
Phase 0-5
  SoftClip local OS
      ↓
Phase 0-6
  Float-input / Double-input equivalence
      ↓
Phase 0 gate
```

これなら **production source = 0変更** を維持できます。

---

# 35. Phase 0 の PASS 条件を「絶対値」と「差分」に分ける

これも重要です。

例えば P0-C は、

### Absolute requirement

```text
candidate passband gain
within ±0.1 dB of unity
```

### Differential requirement

```text
candidate vs baseline normalized response
matches expected gain-only transformation
```

の2つに分ける。

これなら案 E の目的に直接対応します。

---

# 36. 最終的な修正要求

v1.4 を実装承認可能な計画書にするには、私は最低限次の **7項目**を修正します。

### 必須 1

**P0-A を baseline PASS 条件から外す。**

現行 production = 0.75 を測定する baseline にする。

### 必須 2

**案 E の shadow/reference characterization を追加する。**

production source 0変更のまま candidate E を測れること。

### 必須 3

**DC gain 1.0 の根拠を「測定結果」ではなく design contract として明示する。**

### 必須 4

**周波数軸を `Fs_in` 一本から stage-rate 基準へ整理する。**

特に `0.55 Fs_in〜0.95 Fs_in` の stopband 定義を修正。

### 必須 5

**P0-C〜I に数値 tolerance を付ける。**

「同等」だけでは gate にならない。

### 必須 6

**float/double を「oversampler precision」ではなく実際の float-input / double-input 経路として定義する。**

現行 oversampler API は double です。

### 必須 7

**§7.1 の古い `--buzz-osdirect=2` / `round-trip=` を削除する。**

これは v1.4 の改訂サマリと自己矛盾しています。

---

# 37. その上での最終判定

私は現時点で、

### B-1 原因分析

**GO**

です。

現行 `ConvoPeq.md` の `prepareStage()` と `interpolateStage()` は、計画書が示す gain convention imbalance と整合しています。 

### 案 E

**有力仮説として維持してよい**

です。

C 案の撤回も妥当です。

### Phase 0

**GO だが、現在の P0-A〜I の定義のまま実測開始するのは不可**

です。

### Phase 1

**HOLD 継続**

で正しいです。

### F-2

**GO候補**

です。

### F-1

**GO候補**

です。

---

# 38. 最も重要な一文

今回の v1.4 で直すべき核心はこれです。

> **Phase 0 は「案 E が正しいか」を現行 production code の PASS/FAIL で判定するフェーズではなく、「現行 baseline と案 E の shadow candidate を同一測定系で比較し、設計契約・周波数特性・遅延・境界挙動のすべてを検証する characterization フェーズ」にする。**

これに変更すれば、

```text
production = 0変更
↓
current baseline = 0.75
↓
candidate E = 1.0
↓
frequency/phase/alias/delay comparison
↓
contract verification
↓
Phase 1 eligibility
```

という非常に明確な gate になります。

**したがって、v1.4 は「方針そのものはほぼ妥当、ただし Phase 0 の測定仕様と文書残骸を修正してから正式承認」という判定です。**
特に B-1 の Phase 0 をこのまま `P0-A = 1.0` で走らせるのは避けるべきです。
