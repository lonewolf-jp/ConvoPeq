# Phase 0 characterization 報告書（2026-09-22）

- **工程**: C1 commit `85aa13b9` → C0 commit `88c6f000` → **ConvoPeq.md 再生成 + FRESH 確認** → **Phase 0 characterization**
- **snapshot baseline**: `ConvoPeq.md 2026-09-22 00:41:57 / 5,431,515 B / FRESH (NEWER_SRC_COUNT=0)` — 生成後に FRESH 確認済み。その後の test 追加により本実行時点では STALE（4 件 = Phase 0 test 追加ファイル）— Phase 0 freeze 後に再生成するのが O-12 フロー
- **実行環境**: HEAD = `88c6f00`（親 `85aa13b9`）・production `src/*.cpp|h`（tests 除外）差分 **0 件**・`production_flag: undefined/off`・`shadow_candidate: 0`・build_config `Debug;Release;RelWithDebInfo / cl / Ninja`（BUILD 6 要素ブロックは log 先頭）
- **判定**: **全 gate PASS（54 判定・FAIL=0・exit 0）** — 実ログ `tmp/phase0_characterization_20260922.txt`
- **★ 性格（R12-8 継承）**: C1 PASS ≠ B-1 PASS ≠ 案E採用。本 battery は **Shadow fidelity gate + P0 gate 実測**であり、production の数学的正しさの最終 gate は Phase 1 の flag ON ビルド実測（HOLD 継続）

---

## 1. Phase 0 gate 判定表（全 PASS）

| Gate | 判定 | 実測値 |
|------|------|--------|
| **P0-F latency ×4 経路** | **PASS** | argmax ∈ [floor(D), floor(D)+1]・base==cand・bitwise（S1 15/511 255/IIR3 290.25/LP3 582.25） |
| **0-1 G-BL + 0-2 P0-A DC ×6 config（±1e-6）** | **PASS** | dcBase = 0.75^N 厳密 / dcCand = 1.0 厳密（IIR3/LP3 × r=2/4/8） |
| **P0-B/C/C' ×4 経路** | **PASS** | unity@50Hz/1kHz ±0.1・ripple[0.005,0.30] ≤0.05・P0-C' diff ≤0.05（実測: u50=−0.0000〜0.0000・ripple 0.0000〜0.0010・maxDev 0.0000〜0.0002） |
| **P0-D stopband floor ×6 design** | **PASS** | stopMax ≤ −(A−10) dB（511/140: −137.1 ≤ −130・127/110: −107.0 ≤ −100・31/90: −87.0 ≤ −80・1023/160: −157.1 ≤ −150・255/140: −137.0 ≤ −130・63/120: −117.0 ≤ −110） |
| **P0-E E-1 base D1** | **PASS** | −9.542425 ±0.01 @f̂≤0.30（prod 31/127/511 ×4 f̂） |
| **P0-E E-1c cand D1** | **PASS** | ≤ −9.442 @f̂≤0.35（shadow 3 designs・**min margin = 36.8 dB**） |
| **P0-E E-2 D2 base==cand** | **PASS** | ≤0.01 dB @f̂ {0.05,0.10,0.20}（S1） |
| **P0-I I-a OS-only DC** | **PASS** | base 0.75 / cand 1.0 ±1e-6（A=0.5 入力・利得比 gate） |
| **P0-I I-b/c safety + per-phase attribution ×15 stimuli** | **PASS** | I-0 bitwise・c1〜c4 全成立・NaN/Inf=0・\|y\| ≤ 1.0 |
| **P0-G float/double ×2 pairs** | **PASS** | RT maxAbs=2.58e-08 / rms=9.01e-09・SC maxAbs=2.96e-08 / rms=9.16e-09 ≤ 5e-7/5e-8 |

**Phase 0 = ALL PASS（54/54・exit 0）**。

## 2. 記録指標（gate 不使用・R12-12/R15-1/E-1b）

```text
E-1b cand D1 窓不変性: rect / Hann+trim8 併記（手法感度帯 — gate 不使用）
D2 base (f̂ 0.05/0.10/0.20) = −223.31 / −217.69 / −184.26 dB
P0-I events (15 stimuli 合計): base=119,768 cand=121,267 ΔS=+1,499 R_s=1.013
  → 増分はすべて R17-1 c1〜c4 契約の範囲内（conv 位相 bitwise 一致 + center 位相
     base ⊆ candidate + envelope |x_b| ≥ θ_evt/2 + 点wise 出力恒等式）
max|y| = 0.900000（= θ+κ = 0.5+0.4・この動作点での理論上限 — 1.0 未満を確認）
clamp(|y| ≥ 1−1e-12) = 0 件（θ+κ=0.9 < 1.0 のため 0 件が正当）
tanhClamp(|arg| ≥ 4.5) = 18,200 件（Padé クランプ飽和・有意な記録）
P0-G 相対誤差 max = 1.84e-05（|ref| > 1e-6 のみ・record 専用）
```

## 3. Phase 0 の実装範囲と性格

| 項目 | 実測レベル |
|------|-----------|
| REF-FIDELITY（0-1b） | **production↔shadow bitwise**（C1 31/31 に含む・同一 target 同一 compile option） |
| P0-A/0-1 G-BL | production 実測（warm-start impulse + DC round-trip） |
| P0-F | production 実測（impulse argmax） |
| P0-B/C/C' | production impulse → DTFT（cand）/ prod↔shadow 差分 |
| P0-D | shadow 係数（REF-FIDELITY により production と bitwise 等価が保証済み）→ DTFT |
| P0-E E-1 | **production up 出力の単点 DFT**（31/127/511 単段） |
| P0-E E-1c | shadow cand up 出力（REF-FIDELITY により production と bitwise 等価） |
| P0-I | production softClipOS wiring + harness 点wise 参照 F（R12-4 harness 責務） |
| P0-G | harness 参照実装 ×2 pairs（float 経路は double 演算・同一 10395 Padé — 入力量子化差のみ） |

## 4. HOLD 継続の確認

以下は Phase 0 PASS 後も**引き続き HOLD**（R15-4 条件D + ユーザー GO 前は着手禁止）:

- production `centerValue *= 2.0`（案 E）
- `CONVOPEQ_CORRECT_POLYPHASE_GAIN` の CMake 定義
- default ON / calibration / F-3 / F-4 / O-20

## 5. 次の境界

```text
Phase 0 characterization 全 PASS ← 完了
    ↓
Phase 0 freeze（本報告書で確定）
    ↓
ユーザー GO gate（条件D R15-4 適用）
    ↓
Step 2.5 条件D 最終判断プロセス（影響評価・二重補償再 census・関係者レビュー）
    ↓
Phase 1 eligibility
```

**案 E の採用判断は Phase 0 全 PASS では確定しない** — 条件D（R15-4）+ ユーザー最終 GO が必要。

---

## 6. Phase 0 freeze 記録（2026-09-22 確定）

**freeze 宣言**: 本節をもって Phase 0 の **判定・実測値・BUILD-ID** を凍結する。凍結後の「Phase 0 結果の改変」（gate 追加・期待値変更・再測定値による上書き）は禁止。変更が必要な場合は freeze 解除として扱い、**新 BUILD-ID での全 gate 再測定**を要する（R17-2 凍結窓の契約）。

### 6.1 BUILD-ID 6 要素（freeze 確定値）

| # | 要素 | freeze 確定値 |
|---|------|--------------|
| 1 | snapshot identity（測定時） | `ConvoPeq.md` header **2026-09-22 00:41:57 / 5,431,515 B** — 測定時は test-only 追加 4 件により STALE（production セクションは不変） |
| 2 | snapshot identity（freeze 後 O-12 再生成） | `ConvoPeq.md` header **2026-09-22 06:20:11 / 5,464,930 B**（mtime 06:21:03）・`--check` = **FRESH / NEWER_SRC_COUNT=0** |
| 3 | git HEAD | `88c6f00`（親 `85aa13b9` = C1 commit） |
| 4 | working tree | production `src/**`（`CustomInputOversampler.{cpp,h}` / `src/audioengine/**`）・`CMakeLists.txt` の HEAD 差分 **0** |
| 5 | production_flag | **undefined/off**（`CONVOPEQ_CORRECT_POLYPHASE_GAIN` は CMake に不在・R15-2） |
| 6 | shadow_candidate / build config | shadow_candidate **0**（`CONVOPEQ_POLYPHASE_REF_CANDIDATE` 既定）・MSVC 1951 / C++20 / AVX2（`__FMA__` 無し）/ Ninja Multi-Config（Debug;Release;RelWithDebInfo） |

### 6.2 attestation（SHA-256・freeze 対象）

| 対象 | SHA-256 |
|------|---------|
| `tmp/phase0_characterization_20260922.txt`（実ログ） | `284394400bdf63cf0887c0b896f909318f0d7b6c24899586f5755d108b3338b5` |
| `build/Release/PolyphaseGainFidelityTests.exe` | `da93033cf1deaf12a30f4a0197eac701711ba52662400c61b99a609971268b19` |
| `src/tests/AudioEngineHarness/PolyphaseGainFidelityTests.cpp` | `9ca451fb4791b7cd4f6302d56bb8bd17de4bd2b3ac55666764eb76d65701b9fd` |
| `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` | `944294e27e7078f605d98402a10db2072eb166cfb0b8895429938626d92b8706` |
| `src/CustomInputOversampler.cpp`（production・不変条件の attestation） | `abca338da828d605dbc4c334adee433d4fe3f8309b7d7ba75d03a56fd3e49834` |
| `CMakeLists.txt` | `528a68d98a4706407a2e311ae95b86d330d6c472ce401dcd7add2618c4f9ed17` |

**EOL 正規化の注記**: 上表の SHA-256 は **working file のバイト列**に対する値。tracked ファイルのうち CRLF を持つもの（本件では `PolyphaseGainFidelityTests.cpp`）は Git の EOL 正規化（LF）を受けるため、commit blob のバイト列は一致しない（実測: worktree 64,133 B → C2 blob 62,758 B。正規化後のバイト列が working file と完全一致することは確認済み）。`tmp/` の実ログと `.exe` は C0（`88c6f000`）以降 **untracked**（git 管理外）であり、attestation は working file に対してのみ意味を持つ。commit 後の同一性検証は `git show <rev>:<path>` に対して行う（`PolyphaseGainCandidateRef.h` / `CustomInputOversampler.cpp` / `CMakeLists.txt` は worktree == HEAD blob を確認済み）。

### 6.3 実測同一性の確認

- exe mtime **06:12:03** > test source mtime **06:11:40** → exe は freeze 対象ソース（stopMax 初期値 `-1.0e30` 修正・P0-I I-a 利得比 gate 修正を含む）から生成済み。
- 実ログ mtime **06:13**・`PASS=54 FAIL=0`・exit 0（`[FAIL]` 行 0 件）。
- 測定時 snapshot の STALE は **test-only 追加 4 件**が原因で、production セクションは不変（6.1 #4 の差分 0 が担保）。freeze 後に O-12 フロー（再生成 → FRESH 確認）を完了（6.1 #2）。

### 6.4 freeze 対象と不変条件

- **凍結対象**: 上記 6.2 の 6 ファイル・`tmp/phase0_characterization_20260922.txt`・本報告書。
- **不変条件**: production `src/**` の HEAD 差分 0 / `CMakeLists.txt` は HEAD と同一 / flag token 不在（production_flag undefined/off）/ shadow_candidate 0。いずれかが崩れた場合は freeze 無効（全 gate 再測定）。
- **working tree の注記（R18-3 継承）**: `BassBuzzMeasurement.cpp`（+636 行）・`PublishPipelineIntegrationTests.cpp`（+8 行）・`src/tools/check_layout_offsets.py`（削除）は**外部変更**であり Phase 0 の生成物ではない（本 freeze の対象外・attestation 対象外）。C2 evidence commit では Phase 0 対象（`PolyphaseGainFidelityTests.cpp` + 本報告書 + 条件D エビデンス）のみを stage する。

### 6.5 freeze 後の予定

- **C2** = freeze 済み Phase 0 evidence の commit（R17-2: C2 evidence after freeze）。C2 は C1 の test foundation とは別 commit とし、production 変更を含めない。
- 次工程の判断材料 = `step25_condition_d_evidence_20260922.md`（D-1〜D-6・read-only）。
