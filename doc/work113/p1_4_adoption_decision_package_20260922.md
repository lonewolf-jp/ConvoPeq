# P1-4 Adoption Decision Package（2026-09-22）

## 1. Scope / Decision Boundary

- **工程**: P1-1-A/B/C/D = PASS → P1-2 = PASS → P1-3 = PASS → **P1-4（本 doc・read-only）**
- **本 doc の性格**: **decision package の作成のみ**。production implementation 工程ではない。
- **P1-4 中に実施しないこと（変更禁止）**: production source 変更 / CMake 変更 / default 値変更 / flag 削除・既定値変更 / calibration 変更 / preset migration / release-note 実装 / **commit**（すべて 0 を維持）。
- **判断の独立性**: D1〜D5 を**独立**に扱う。**一つの総合 PASS/FAIL にまとめない。**
- **語彙**: `DECISION-READY` = 材料が揃い判断可能（最終判断はユーザー）／`HOLD` = 材料不足で判断不可。本 doc は判断を代行しない。

## 2. Source Identity

| 項目 | 値 |
|------|----|
| HEAD | `1e9e63e3`（`d6995b2b` = P1-1 実装 / `eeeba325` = P1-0 audit / `de95f335` = C2 evidence） |
| flag | `CONVOPEQ_CORRECT_POLYPHASE_GAIN` = **OFF**（`build/CMakeCache.txt` 実値） |
| production diff | **0**（`src/audioengine`・`src/CustomInputOversampler.{cpp,h}`・`src/dsp`・`src/eqprocessor`・`src/convolver`・`CMakeLists.txt`・`build.bat` に差分なし） |
| staged | 空 |
| working tree（未 commit） | `?? doc/work113/p1_2_double_compensation_review_20260922.md` / `?? doc/work113/p1_3_pre_adoption_impact_review_20260922.md` / `M src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp`（test-only・P1-3 追加測定） |
| authoritative snapshot | `ConvoPeq.md` **Generated 2026-09-22 10:28:33** / 5,488,742 B / `--check` **FRESH（NEWER_SRC_COUNT=0）** |
| 現行 source の確認 | `src/CustomInputOversampler.cpp:557-560` = `convValue *= 2.0;` → `#if CONVOPEQ_CORRECT_POLYPHASE_GAIN` → `centerValue *= 2.0;` → `#endif` → denorm clear（案E の対象は center phase のみ） |

## 3. Evidence Summary

### 3.1 Evidence supports（採用判断の材料として確立している事項）

| # | Evidence | 出所 |
|---|----------|------|
| E1 | round-trip DC: OFF = **0.75^N**（厳密） / ON = **1.0**（厳密） | Phase 0 P0-A（±1e-6・IIR3/LP3 × r=2/4/8） |
| E2 | 理論補正量 = **(4/3)^N**（= +2.4988·N dB） | B-1 の代数（P0-C' の devTerm） |
| E3 | 実測 Δg: N1 **+2.4988** / N2 **+4.9975** / N3 **+7.4963** dB（linear 域・±0.0003 dB） | P1-1（chain 実測・50 Hz〜10 kHz） |
| E4 | OS=1 control: Δg = **0.0000 dB**（polyphase が経路に無い構成） | P1-3（−20/−6/0 dBFS） |
| E5 | staging −6 dB: Δg = **+7.4963 dB**（staging と乗算独立） | P1-3 §6.3 |
| E6 | latency: reported / impulse argmax が **OFF == ON**（4 経路） | P1-1 |
| E7 | hard clamp: **24 条件すべて 0 件** | P1-3 §6.2 |
| E8 | downstream compensation census: **factor 依存 compensation = 0**、`oversamplingFactor` は gain 計算に現れない | P1-2 §1/§2 |
| E9 | float/double: 単一 OS 実装を両 core が共有（P0-G 実測 ≤2.96e-8） | P1-2 §7 |
| E10 | OFF build は Phase 0 battery **54/54 を 2 回再現**、ON build は **19/54**（base 期待値が candidate 挙動になる意図的差分＝正の対照） | P1-1 §P1-1-B/§P1-1-C/§P1-1-D |

### 3.2 Evidence does not establish（本パッケージでは確定しない事項）

| # | 未確立事項 | 理由 |
|---|-----------|------|
| N1 | **0.75^N が設計意図だったか**（意図の有無） | 測定では判定不能。本パッケージでは設計文書・ADR 上の根拠を特定していない（「バグである」と断定しない） |
| N2 | +2.4988·N dB の level 変化の**聴感上の許容性** | listening evaluation 未実施 |
| N3 | **EQ boost + OS / IR boost + OS** の組合せ挙動 | 測定は identity EQ + conv bypass 条件のみ |
| N4 | **既存プリセットの replay** での level 再現 | preset replay 未実施（保存値は不変だが出力は変わる） |
| N5 | staging 非0 × 非線形の**複数点**での挙動 | P1-3 は 1 条件（−6 dB・os8・−6 dBFS）のみ |
| N6 | saturation sweep / limiter engagement sweep | 端の条件のみ（sat=1.0 中心・−20/−6/0 の 3 点） |
| N7 | up ドメイン image（−9.5424 dB = 1/3 振幅）の**可聴影響** | linear 経路では down 後 −184 dB 以下（P0-E E-2）。SoftClip 併用時の寄与は推論に留まる |
| N8 | **project safety** | Mimosa 完全監査は未実施（§10 参照） |

## 4. D1 — Adoption of Option E

**判定対象**: 案E（center phase にも ×2 を適用する polyphase gain convention 補正）を採用するか。

| 観点 | Evidence supports | Evidence does not establish |
|------|-------------------|------------------------------|
| 数学契約 | E1・E2・E3（0.75^N → 1.0、(4/3)^N の一致）・E4（作用範囲は polyphase 段のみ） | N1（設計意図の有無） |
| 回帰の有無 | E6（latency 不変）・E7（hard clamp 0）・E9（float/double 一致） | N7（image の可聴影響） |
| 補償の衝突 | E8（二重補償なし）・E5（staging と乗算独立） | — |
| 検証基盤 | E10（OFF 54/54 の再現・ON の正の対照） | N3〜N6（C 群の測定不足） |

- **含意**: **DSP convention の修正としては材料が揃っている**（採用可否の最終判断はユーザー）。
- **明示的な非含意**: D1 の判断は **D2（default ON）を自動承認しない**。D1 と D2 は独立である（§5）。
- **Status**: `DECISION-READY`（数学契約・回帰・補償の三点で材料充足。ただし N1 は未確立であり「バグだった」とは主張しない）

## 5. D2 — Default ON/OFF

**判定対象**: 採用する場合に flag を default ON にするか。以下の 3 層を**分離**して扱う。

### A. DSP correctness

| 事実 | 値 |
|------|----|
| polyphase gain convention | OFF = 0.75^N → ON = 1.0（E1/E3） |
| latency・係数・topology | 不変（E6・P1-1 §P1-1-A） |

→ **A は「修正の正しさ」の層**であり、それ自体は default ON を要求しない。

### B. Existing product behavior（実測済みの製品挙動変化）

| 項目 | OFF | ON | 出所 |
|------|-----|----|------|
| default output level（staging 0 dB・1 kHz） | −2.4988·N −1.0 dB | −1.0 dB | P1-1（−8.5035 vs −1.0072 @ N=3） |
| OS 倍率別の出力変化 | — | 2x: +2.4988 / 4x: +4.9975 / 8x: **+7.4963 dB** | P1-1・P1-3 |
| SoftClip drive | clipEngMax 0.0018〜0.0687 | **0.0182〜0.1473**（5〜10 倍） | P1-1・P1-3 |
| limiter 到達（os≥2・0 dBFS） | 0 | **119/4096** | P1-1・P1-3 |
| hard clamp | 0 | 0 | E7 |

→ **B は「既定設定での可聴結果が変わる」層**。変更量は OS 倍率に依存し、既定は Auto（≤96 kHz で 8x → 最大 **+7.4963 dB**）。

### C. User-facing compatibility

| 項目 | 状態 |
|------|------|
| 保存済みパラメータ値（makeup/headroom/trim/OS factor） | **不変**（`DeviceSettings` は独立属性として保存・読込。P1-3 §4） |
| 既存プリセットの**結果としての出力レベル** | **変わり得る**（N2/N4 未確立） |
| 既定値・UI 文言 | 不変（"unity" の保証は UI に存在しない） |

- **結論（含意）**: 「数学的に正しい」ことは **default ON の十分条件ではない**。default ON を判断するには最低でも **N2（聴感）・N4（preset replay）・N3（EQ/IR 併用）** の材料が必要。
- **Status**: `HOLD`（材料不足。A は充足、B は実測済みだが許容性未評価、C は未測定）

## 6. D3 — Flag Retention/Removal

3 案の **documented consequence** を並べる（**winner は選ばない**）。

### Option F1 — flag 維持 + default OFF

| 軸 | consequence |
|----|-------------|
| rollback 性 | 最高（compile-time 1 行で base に戻る） |
| binary / configuration complexity | flag 分岐が恒久的に残る（`#if` 1 箇所・CMake option 1 個・genex 3 行） |
| regression isolation | 高い（Phase 0 battery が OFF で 54/54 を保証） |
| preset compatibility | 現行挙動を維持（既定 OFF のまま） |
| testability | 両意味論を同一ソースで検証可能（正の対照が得られる） |
| maintenance burden | 2 つの意味論が共存し続ける（二重の期待値管理） |
| release engineering | 既定挙動が変わらないため release note は「実験的 flag の追加」で済む |

### Option F2 — flag 維持 + default ON

| 軸 | consequence |
|----|-------------|
| rollback 性 | 高（flag で OFF に戻せる） |
| complexity | F1 と同じ分岐を維持 |
| regression isolation | ON が既定になるため、Phase 0 battery（base 期待値）は**恒久的に FAIL** する → 期待値の切替（evidence の分離）が必要 |
| preset compatibility | 既定で出力が +2.5〜+7.5 dB 変化（§5 B/C） |
| testability | 同上（2 意味論の共存） |
| maintenance burden | 「正しいのは ON」を既定にしつつ OFF 経路を保守し続ける |
| release engineering | 既定挙動の変更を含む release note が必要 |

### Option F3 — 恒久採用 + flag 削除

| 軸 | consequence |
|----|-------------|
| rollback 性 | 低（revert commit 以外に戻す手段がない） |
| complexity | 最小（`#if`/option/genex を削除） |
| regression isolation | Phase 0 battery の base 期待値を**恒久に失う**（expect 更新が必要・比較基準が消える） |
| preset compatibility | F2 と同等の出力変化 |
| testability | 単一意味論（candidate と production の区別が消え、正の対照が得られなくなる） |
| maintenance burden | 最小 |
| release engineering | 挙動変更 + flag 追加なしの一発変更として告知 |

- **Status**: `DECISION-READY`（判断材料は 7 軸で整理済み）。ただし **F2 は D2 の決定に、F3 は D1 の（採用方向の）決定に従属**する。本 doc では選択しない。

## 7. D4 — Calibration / Default Values

- **現時点で変更しない値**（確認済み・不変）:

```text
inputHeadroomDb   = -6 dB
outputMakeupDb    = +12 dB
convolverTrim     =  0 dB
kOutputHeadroom   = 0.8912509381337456（-1.0 dBFS・固定 constexpr）
```

- **判断**: `Calibration decision = HOLD`。P1-3 の **B 群（再校正候補）** と **C 群（実測不足）** の境界を維持する。
- **HOLD を解除するために必要な characterization（不足している順）**:

| # | 条件 | 目的 |
|---|------|------|
| 1 | EQ boost + OS | 実 EQ カーブ/合計 gain/AGC との組合せで flag 効果が乗算的かを確認 |
| 2 | IR boost + OS | Convolver 有効・trim 非0 との組合せ |
| 3 | saturation sweep | `saturationAmount` 0.1→1.0 の知覚差（現在は端のみ） |
| 4 | limiter engagement sweep | 入力レベル細分による到達率カーブ（現在は 3 点） |
| 5 | existing preset replay | 保存済み設定での出力変化の再現（N4） |
| 6 | staging 非0 × nonlinear 複数点 | 乗算独立性の非線形域での確認（現在 1 条件） |
| 7 | listening evaluation | 許容性の判断材料（N2） |

## 8. D5 — Release / Preset Compatibility

- **明示すべき事実（分離）**:

```text
parameter values themselves : unchanged（makeup/headroom/trim/OS factor の保存値は不変）
resulting audio level       : may change（同一プリセットでも出力レベルが変わり得る）
```

- **release note 設計上の 2 つの別問題**:
  1. **parameter migration**（数値の移行）: **不要**。`DeviceSettings` は独立属性として保存・読込し（P1-3 §4）、値の再解釈も行われない。
  2. **behavioral / output-level change**（挙動・出力レベルの変更）: **必要**。案E 採用時の default 設定では OS 倍率に応じて **+2.4988 / +4.9975 / +7.4963 dB**（既定 Auto=8x で最大）の出力変化が起こり、limiter 到達量と SoftClip 駆動も変わる（§5 B）。
- **既存プリセットの扱い**: 数値は書き換えずに、**同じプリセットが異なる出力レベルになる**ことを release note に明記する必要がある（N4 の replay 測定は未実施のため、具体的な記述は測定後に確定）。
- **Status**: `DECISION-READY`（方針の分離は確立）／具体的な release note 文面は **別工程**（本 doc では書かない）。

## 9. Open Evidence / Required Follow-up

| # | 未確立事項（§3.2） | 次工程で必要な作業 |
|---|-------------------|--------------------|
| 1 | N2（聴感） | listening evaluation（採用判断の前提として最優先） |
| 2 | N4（preset replay） | 代表プリセット（makeup/headroom/trim/OS factor の組合せ）での出力レベル再現 |
| 3 | N3（EQ/IR 併用） | §7 の characterization 1・2 |
| 4 | N5・N6（sweep） | §7 の characterization 3・4・6 |
| 5 | N1（設計意図） | 設計文書・ADR・履歴の探索（本パッケージの範囲外） |
| 6 | N7（image 可聴影響） | 推論のまま保持（linear 経路では −184 dB 以下。SoftClip 併用時は未測定） |
| 7 | N8（project safety） | **Mimosa 完全監査の再実行**（未実施。§10 の分離を維持） |

## 10. Decision Status

### 10.1 判断ステータス（独立・総合 PASS にしない）

```text
D1 Option E adoption        = DECISION-READY（数学契約・回帰・補償の材料充足。最終判断はユーザー）
D2 Default ON               = HOLD（A 充足 / B 実測済みだが許容性未評価 / C 未測定）
D3 Flag policy (F1/F2/F3)   = DECISION-READY（7 軸の consequence を整理。F2 は D2、F3 は D1 に従属）
D4 Calibration              = HOLD（§7 の 7 条件が必要）
D5 Release handling         = DECISION-READY（方針分離は確立。文面は別工程）
```

### 10.2 P1-4 中に実施していないこと（禁止事項の遵守）

```text
default ON 変更 / flag 削除 / calibration 変更 / preset migration
release-note 実装 / production source 変更 / commit — すべて 0
HEAD = 1e9e63e3 のまま・staged 空・production diff 0（§2）
```

### 10.3 停止条件の確認（すべて非該当）

| 停止条件 | 確認 |
|----------|------|
| P1-3 の証拠と source snapshot の矛盾 | なし（snapshot FRESH 10:28:33・現行 `:557-560` が案E の対象と一致） |
| 案E以外の production gain compensation 発見 | なし（P1-2 §1/§2 の census を再確認） |
| preset compatibility の追加問題 | なし（値は不変・レベルは変化：既知の論点として §8 に記録） |
| default 値変更が必要だと証明される | なし（§7 の 7 条件はすべて未測定であり「変更が必要」の証明ではない） |
| calibration を決めるには characterization 不足が重大 | **該当**（該当するが、D4 を HOLD とすることで本工程の停止条件には至らない） |
| Mimosa で production safety に関係する新規問題 | 今回の指摘は解析スクリプト（`tmp/`・git 管理外）の argv パス検証であり、production safety に関係しない（§11.4） |
| production source modification が必要になる | なし（本 doc 作成のみ） |

### 10.4 Mimosa 境界（混同しない）

```text
script argv hardening PASS  ≠  project safety PASS
```

- 本セッションで実施した hardening（`tmp/p1_analyze.py`・`tmp/p1_3_analyze.py`：argv 入力の廃止・`resolve()` 正規化・許可ディレクトリ（`tmp/`）内限定・通常ファイル検証）は **P1-4 の evidence として記録**するが、**プロジェクト全体の安全性は未検証**である（Mimosa 完全監査は未実施）。
- 本 adoption は **RT / RuntimeWorld / Publish / Retire の責務分離と不変条件を変更しない**（polyphase 段の局所的な gain convention の変更のみ）。RT 側への判断・所有権・動的確保・delete の持ち込みは発生しない。

## 11. Appendix — Measurement / Snapshot Identity

### 11.1 BUILD-ID

| 項目 | OFF build | ON build |
|------|-----------|----------|
| snapshot_identity | `ConvoPeq.md` 2026-09-22 06:20:11 → 10:28:33（FRESH） | 同左（08:12:33 時点で FRESH） |
| git_head | `eeeba32`（P1-1 時点） | `eeeba32` |
| working_tree | production diff = `src/CustomInputOversampler.cpp`（意図した 1 行） | 同左 |
| production_flag | **OFF (Phase 1: default OFF)** | **ON (Phase 1: 明示指定 — characterization 用)** |
| shadow_candidate | 0 | 0 |
| build_config | Debug;Release;RelWithDebInfo / cl / Ninja | 同左 |

### 11.2 測定ログ（本パッケージの引用元）

| ログ | 内容 |
|------|------|
| `tmp/p1char_off.log` / `tmp/p1char_on.log` | P1-1：level（3 factor × 3 freq × 2 level + 0 dBFS record）・latency 4 経路・SoftClip/Limiter・flag 間 sample 差（1024 点） |
| `tmp/p13char_off.log` / `tmp/p13char_on.log` | P1-3：os{1,2,4,8} × ampDb{−20,−6,0} × sc{0,1}（1 kHz）+ staging −6 dB ケース |
| `tmp/p13_off2_refgate.txt` | OFF 復帰後の Phase 0 battery = **PASS=54 FAIL=0** |
| `tmp/p1_seq_off1_refgate.txt` / `tmp/p1_seq_on_refgate.txt` / `tmp/p1_seq_off2_refgate.txt` | OFF 54/54（×2）・ON 19/54（意図的差分＝正の対照） |

### 11.3 判定スクリプト（P1-4 時点の状態）

| スクリプト | 状態 |
|-----------|------|
| `tmp/p1_analyze.py` | ハードニング済（固定リテラル・`resolve()`・`tmp/` 内限定・通常ファイル検証）。ハードニング後も解析結果は不変 |
| `tmp/p1_3_analyze.py` | 同上（P1-3 の `RESULT: ALL PASS` を同一数値で再現） |

### 11.4 Mimosa 記録

- 指摘: `tmp/p1_3_analyze.py` の argv 由来パス未検証（2 行）。
- 対応: argv 入力を廃止し固定リテラル化 + `resolve()` 正規化 + 許可ディレクトリ（`tmp/`）内限定 + 通常ファイル検証。2 スクリプトで逸脱パス（`tmp/` 外・絶対パス・`..` 含み）3 種がすべて拒否されることを確認。
- 境界: 上記は **tooling の hardening** であり、プロジェクト安全性の総合判定ではない。
