# C1 実装報告書 — PolyphaseGainFidelityTests（REF-FIDELITY 基盤）

- **日付**: 2026-09-22（work113・ZCode 環境）
- **対象計画**: `doc/work113/remediation_plan_20260922_v3.2_revised.md`（v3.1 + R18-1〜R18-3）
- **実施範囲**: v3.1 レビュー（総合判定「妥当」）で **GO とされた C1（Shadow Reference 完全実装 + REF-FIDELITY 基盤）** と **Step 0.0 維持確認** の実装・検証。**commit は行っていない**（レビュー §10「staged 一覧を人間が確認してから commit」に従い、staged 一覧を提示した段階で停止）。
- **HOLD の遵守**: production `src/*.cpp|h`（tests 除外）の変更 **0 件**・`CONVOPEQ_CORRECT_POLYPHASE_GAIN` の CMake 定義 **0 件**（本セッションで再確認）・default ON / calibration / production patch / 案 E 採用決定は未着手。

---

## 1. 実装成果物

| ファイル | 区分 | 内容 | 検証 |
|----------|------|------|------|
| `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` | 上書き（102 行骨格 → 完全実装） | `prepareStageModel`（R16-1 の 10 手順を演算順序ごと独立実装）・`dotProductAvx2Ref` / `loadStride2Ref` / `dotProductDecimateAvx2Ref`（production SIMD 帰還構造と同一順序の独立実装）・`interpolateStageRef` / `decimateStageRef`（production :492-723 と同一観測契約・履歴 copy/guard/denorm/isBadSample 位置同期）・`PolyphaseGainShadow`（両方向 multi-stage・reset 契約 atomic 3 / clearAllStages atomic 1 相当・hardFallback/corruption auto-clear）・candidate は runtime 切替（`CONVOPEQ_POLYPHASE_REF_CANDIDATE` macro 既定） | 31 検定 PASS |
| `src/tests/AudioEngineHarness/PolyphaseGainFidelityTests.cpp` | 新規 | R17-4 (A)〜(E) 5 検証 + REF-FIDELITY 本体（シード固定 PRNG 3 ブロック × 2 preset × ratio {2,4,8} の up/down bitwise + reset contract）+ scalar↔scalar 3-tap 経路 + candidate 2 系統一致性 assert + `[BUILD]` 6 要素出力。FAIL = fail-closed（exit 1） | 31/31 PASS・exit 0 |
| `CMakeLists.txt` | test-only 変更 | 40 番目 `add_executable(PolyphaseGainFidelityTests …)` + AudioEngineHarness と同一の compile option（/arch:AVX2 等・R17-3 同一 compiler configuration）+ add_test。flag token は定義せず | configure/ビルド成功 |
| `src/tools/build_identity_gate.py` | test-only 変更 | `--emit-build-id [--shadow-candidate 0/1]` モード追加: 6 要素 `[BUILD]` ブロック出力（snapshot identity / git HEAD / working tree / production_flag 未定義確認 / shadow_candidate / build configuration） | 動作確認済み |
| `doc/work113/remediation_plan_20260922_v3.2_revised.md` | 新規（計画書 v3.2） | R18-1（C1 実装範囲契約）/ R18-2（同一 target・別 TU 精緻化）/ R18-3（外部 working-tree 変更記録）+ 検証記録 | 本報告書と一体 |

## 2. 検証結果（build + run・静的解析）

### 2.1 ビルド・実行

```text
configure : vcvars64 (VS 18/Enterprise) → cmake -S . -B build -G "Ninja Multi-Config"
            -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl → Generating done
build     : cmake --build build --config Release --target PolyphaseGainFidelityTests
            → build\Release\PolyphaseGainFidelityTests.exe 生成
run       : PASS=31 FAIL=0・exit 0（fail-closed 契約を満たす）
log       : tmp/c1_ref_fidelity_run_20260922.txt
            （[BUILD] 6 要素ブロック = runner 出力 + exe 出力の両方・全 31 判定）
```

### 2.2 test 一覧（31 判定・全部 PASS）

| # | test | 検証内容 | 結果 |
|---|------|----------|------|
| 1 | R16-1 `kPiRef == juce::MathConstants<double>::pi`（bitwise） | 定数パス契約 | PASS |
| 2 | R16-1 `kDenormThresholdRef == numeric_policy::kDenormThresholdAudioState`（bitwise） | 同上 | PASS |
| 3 | R16-3 candidate macro 2 系統一致性（macro 既定 ↔ runtime {false,true}） | DC 予測系の一致（base 0.75³ bitwise / cand 1.0 bitwise / centerPhaseGain 1.0↔2.0） | PASS |
| 4 | R17-4(A) coefficient invariant ×7 design（511/140・127/110・31/90・1023/160・255/140・63/120・3/90） | Stage 全フィールド（taps/centerTap/parity/convCount/convCoeffs/reversed/centerCoeff/centerDelayInput/historyUpKeep/historyDownKeep/history sizes）+ FIRsum=1.0 / center=0.5 bitwise / convSum=0.5 | PASS |
| 5〜10 | R17-4(B) zero-input × 6 config（IIR3/LP3 × r=2/4/8） | 全出力 0 + prod↔shadow bitwise | PASS ×6 |
| 11〜16 | R17-4(C) constant-input DC × 6 config | base DC = 0.75^N ±1e-9 / shadow DC bitwise 一致 / shadow cand DC = 1.0 ±1e-9 | PASS ×6 |
| 17 | R17-4(D) impulse warm-start S1(31/90) | `h_rt = 2·(conv⋆conv) + 0.25·δ[15]`（expectPeak = 0.679897021291）+ Σ = 0.75 ±1e-9 / cand Σ = 1.0 / peak@15 / bitwise | PASS |
| 18 | R17-4(D) impulse warm-start IIR3 r=8 | Σ = 0.421875 ±1e-9 / peak ∈ [290,291] / bitwise | PASS |
| 19〜24 | R17-4(E) random block partition invariance × 6 config | partition {1,2,3,5,7,11,15,31,63,127,256,512,1024}+残・mixed {n/2,n/4,n/8,n/8}: one-shot == partitioned（up/dn 累積 bitwise）| PASS ×6 |
| 25〜30 | REF-FIDELITY 本体 × 6 config | 3 PRNG ブロックの up+dn bitwise + reset contract（atomic 3 相当の reset 後の同一刺激で再び bitwise 一致） | PASS ×6 |
| 31 | scalar↔scalar REF-FIDELITY（3-tap synthetic・convCount=2） | production/shadow 両側 scalar 経路の bitwise 一致（R17-3 の scalar 契約） | PASS |

### 2.3 実装中に修正した自バグ（失敗→修正→再検証）

```text
(a) 初回実行で r=4/8 zero-input が "prod round-trip failed" — up バッファを
    2×block で確保していたため 4×/8× の up 出力で超過 → 8×block に修正。
(b) S1 impulse の h_rt[15] 期待値を誤って 0.25 と記載 — R5-9 構造恒等式では
    h_rt[15] = 2·(conv⋆conv)[15] + 0.25 ≈ 0.679897021291（0.25 は δ 項のみ）。
    expectPeak を shadow の独立係数から構築する閉形式に修正 → bitwise 一致。
(c) configure 再生成時に CMakeLists.txt:576 の
    "PriorityIntegrationTests → juce::juce_core not found" を 1 回観測 —
    外部 working-tree/build-dir クリーンアップとの競合による過渡状態。
    vcvars 正環境での re-configure で解消（再発なし）。
```

### 2.3 静的解析

- **cppcheck 2.21.0**: `--enable=warning,performance,portability --std=c++17 --suppress=missingIncludeSystem` → test TU で新規 error なし。`uninitMemberVarNoCtor` 2 件は constexpr aggregate（`static constexpr Config kConfigs[]`）の aggregate initializer 済みの誤検知。header 直渡し時の `syntaxError` は C-mode 解析の既知制約。
- **clang-tidy 23.1.1**（tmp/clangdb・`clang-analyzer-*`/`bugprone-*`/`performance-*`/`misc-*`）: `stages_[3]` 境界の analyzer 警告に対し `prepare()` の clamp + `processDown()` の fail-closed guard を追加 → **警告 0 件**。guard 加筆後の再実行で PASS=31 FAIL=0 を維持。

## 3. R17-1/R17-2 契約との対応

| 契約 | 本実装での位置付け |
|------|---------------------|
| **R17-1（P0-I 数値契約）** | `tmp/v31_p0i_attribution_check.py`（42 判定 ALL PASS）は **契約の事前数学検証 PASS であり、P0-I 実装済み PASS ではない**（レビュー §2 の区別を厳守）。SoftClip local OS の実測（0-5）は Phase 0 characterization で G-0/C0 承認後に実施する |
| **R17-2（commit 境界）** | C0/C1 は独立 commit・測定中 commit 全凍結。本実装では **C1 commit を行っていない** — staged 一覧の人間確認（§4）が前提 |
| **R17-3（REF-FIDELITY 前提）** | 同一 target（`PolyphaseGainFidelityTests`）内で production TU（`CustomInputOversampler.cpp`）と Shadow TU（`PolyphaseGainCandidateRef.h`）を同一 compile option で compile → **同一 BUILD-ID 契約を満たす**。bitwise mismatch 時は triage 5 分類で原因確定まで進めない |
| **R17-4（追加 5 検証）** | T4〜T9 として全部実装・全部 PASS |
| **R15-2 / R12-5（HOLD 維持）** | production src 変更 0・CMakeLists flag token 0 件・二重補償 0 件（本セッションで再確認） |

## 4. 想定外事態と対応（R18-3）

本セッション実行中に、**C1 作業者が意図して変更したものではない外部 working-tree 変化として観測された**（開始時には実在を確認済み・因果の特定は本報告だけでは第三者検証できないため、この表現に限定する）:

| 変更 | 内容 | 対応 |
|------|------|------|
| (i) `output_sourcecode_markdown.py` 削除（` D`） | O-18 の未 commit 修正 +11/−4 が消滅（stash なし・ConvoPeq.md には 0 hit） | **復元せずユーザー判断に委ねる**（並行セッションのクリーンアップと思われる） |
| (ii) `scripts/fix_lint.py` / `fix_tables.py` / `fix_tables_v2.py` 削除（` D`） | 同上 | 同上 |
| (iii) untracked root 解析スクリプト群（`analyze_mcp.py` 等 6 件）消失 | 一時的な解析产物 | 同上 |
| (iv) `build/CMakeCache.txt`・`build-Release.ninja`・`build.ninja` が一時消失 → 復活 | 並行作業中の configure 競合 | **vcvars 正環境での re-configure で解消**（configure 自体は成功済み） |

**ConvoPeq.md 本体（16:16:35 生成・5,359,772 B）は変更されていない**ため authoritative snapshot の ID 自体は無傷。ただし **O-18 の +11/−4 未 commit 修正はディスクから失われた** — 復元はユーザー判断（他環境の working tree に残っている可能性を確認のこと）。

## 5. 実 index の staged 一覧（commit 前の人間確認対象・R18-1 (b)）

**★ 2026-09-22 追記（外部監査 §11 の指摘対応）**: 本節の旧版は `git status --porcelain` の `??` 表記をそのまま引用しており「staged ではない」という誤解を招いた。**staged の実態は `git diff --cached`（index vs HEAD）で確認するのが正しい**。以下は 2026-09-22 に再実行した実 index の内容である。

```text
$ git status --short（参考・XY = index/working-tree）
 M .gitignore 等（C1 以外の既存 working-tree 変更は staged していない）
M  CMakeLists.txt                     ← staged
AM doc/work113/...（両 doc・最終 add 後は M）
M  output_sourcecode_markdown.py      ← staged

$ git diff --cached --name-status（実 index の commit 候補・7 ファイル）
  M  CMakeLists.txt
  A  doc/work113/implementation_report_20260922_c1_ref_fidelity.md
  A  doc/work113/remediation_plan_20260922_v3.2_revised.md
  M  output_sourcecode_markdown.py
  A  src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h
  A  src/tests/AudioEngineHarness/PolyphaseGainFidelityTests.cpp
  M  src/tools/build_identity_gate.py

$ git diff --cached -- CMakeLists.txt | grep -c CONVOPEQ_CORRECT_POLYPHASE_GAIN
  0          ← flag token 0 件（R18-1 契約を満たす）

$ git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'
  （空）      ← production 変更 0（R15-2 / R12-5 不変条件）
```

**commit 前の人間確認事項**:
1. `git diff --cached --name-status` が上記 7 ファイルのみであること（production `src/*.cpp|h`（tests 除外）を含まない）。
2. `git diff --cached -- CMakeLists.txt` に **flag token の定義行が含まれない**こと（コメント内 token も含めて 0 — 実測 0 件）。
3. C0（evidence/ 追跡外化）と C1 を混ぜない（独立 commit）。

## 5b. C1-PRECOMMIT GATE（外部監査 §19 の 5 項目・実測結果）

| # | 項目 | 実行コマンド | 実測結果 | 判定 |
|---|------|--------------|----------|------|
| 1 | git index | `git diff --cached --name-status` | C1 の 7 ファイルのみ（上記） | **PASS** |
| 2 | production source | `git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'` | 出力なし（0 件） | **PASS** |
| 3 | CMake | `git diff --cached -- CMakeLists.txt \| grep -c CONVOPEQ_CORRECT_POLYPHASE_GAIN` | 0 件（test target 登録のみ・+61 行） | **PASS** |
| 4 | BUILD-ID / executable | log 先頭 `[BUILD]` 6 要素ブロック | `snapshot_identity : ConvoPeq.md 2026-09-21 16:16:35 / 5359772 B / STALE(C1 変更 4 件が baseline より新 — O-12 再生成フローどおり)`・`production_flag : undefined/off`・`shadow_candidate : 0`・`build_config : Debug;Release;RelWithDebInfo / cl / Ninja` | **PASS** |
| 5 | Fidelity | 実 exe 再実行 | **PASS=31 FAIL=0・exit 0**（31 `[PASS]` 行・0 `[FAIL]` 行を実 log で確認） | **PASS** |

**C1-PRECOMMIT GATE: 5/5 PASS。** 本 gate が全 PASS のまま working tree に C1 以外の変化が入った場合は gate 再実行が必要。

## 6. 判定と次工程

- **C1 の実装と検証: 完了 + commit 済み**（HEAD `85aa13b9`・31 検定 ALL PASS・静的解析警告 0・production 変更 0・PRECOMMIT GATE 5/5 PASS）。
- **REF-FIDELITY の性格**: 本 exe は **Shadow fidelity gate**（production 正しさの gate ではない・R12-8）— 「production == shadow」は「production が数学的に正しい」を意味せず、T4〜T8 の独立した数学的制約が共通バグリスクを低減する位置づけ。**C1 PASS ≠ B-1 PASS ≠ 案E採用**。P0-A〜P0-I の characterization は Phase 0 で実施（P0-I は事前数学検証 PASS のみ・実測は 0-5）。
- **承認状態（2026-09-22）**: G-0 / T-1 / T-2 / T-3 / C0 **全承認済み**（ユーザー確定・v3.2 計画書 §3 に記録）。
- **次の着手境界**: C0 独立 commit（evidence/ 追跡外化・承認済み・実行は次の指示時）→ ConvoPeq.md 再生成 → FRESH 確認 → Phase 0 characterization（0-1 〜 0-6）。production `centerValue *= 2.0`・CMake flag・default ON・calibration・案E 最終採用・F-3/F-4 は全 HOLD 継続。
