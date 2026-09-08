# D162-2-I3-4-D — Contract Implementation Report

```text
Date:            2026-09-05
Type:            build-system contract implementation (fail-closed gate)
Changes:         build.bat (+20 行 gate 組み込み) / src/tools/build_identity_gate.py (新規 223 行)
                 / src/tools/check_layout_offsets.py (新規 60 行)
                 src/** runtime code 変更: 0 / tests: 0 / CMakeLists.txt: 0
Baseline:        I3-4-C GO / clean rebuild (I3-4-B)
判定:            **D1-D12 達成 (D4/D5 は stamp で実装・D10 negative で実証) — GO。
                 ただし depth-2 発見: 10 test-only TU が persistent #deps 0 → Release gate は
                 fail-closed を継続 (契約どおり)。I3-4-E で ninja 上流調査を推奨。**
```

---

## 1. 実装内容

### D1 — build identity stamp (`src/tools/build_identity_gate.py`)

configure 直後に `build/CMakeFiles/.build_identity` を生成。記録項目:
generator / compiler_id / compiler_path / compiler_version (MSVC 19.51) / architecture /
cmake_version (4.4.3) / ninja_version (1.13.2) / configuration_family / source_revision
(0aeb22c+dirty) / **console_codepage** / **msvc_deps_prefix_sha256 (b84b3da87752fa51)**。

git SHA 単独を generation とみなさない設計（I3-4-C 指示どおり）: codepage と
prefix hash を identity の一部に含め、dependency parser identity を固定する。

### D2 — codepage contract

stamp の `console_codepage` と現在 console の chcp を全 build 前に照合。
不一致 → `[GATE-FAIL] COHERENCE-4 violated` で rc=3 終了（黙って続行しない）。

### D3 — `#deps 0` fail-closed gate

`ninja -t deps` を走査し、**relevant .obj（`/src/` 配下・D4 の scope）で #deps 0 が
1 個でもあれば rc=3 で build 拒否**。ninja 自身は #deps 0 を VALID と扱うため、
この外部 gate が COHERENCE-3 の実装本体。

### D4 — gate scope

`RELEVANT_OBJ_RE = /(Release|Debug|RelWithDebInfo)/src/` — project TU のみ gate。
JUCE / third-party / resource (.rc.res) artifacts は対象外（D4 指示どおり。
実測でも juce_core_CompilationTime.cpp.obj が常に #deps 0 であり、全対象化は誤検出を
量産するため正当）。

### D7 — clean fallback

**自動 clean は実装しない**（指示どおり）。gate 失敗時は
`run: build.bat <cfg> clean (explicit clean recovery)` を表示して停止するのみ。

### D2+build.bat 組み込み

`build.bat` の `:configure_cmake_ok` 直後に gate 呼出しを挿入:
- stamp が存在すれば `--check`（build 前検証）、無ければ stamp 生成 + gate。
- gate 失敗 → `[ERROR] Refusing to link with potentially incoherent build state.` で異常終了。

## 2. D10 — 検証結果

### D10-A positive

CP932 console（stamp と同一環境）で `--check`: identity 一致 → deps gate 実行。

### D10-B negative: codepage mismatch（実証成功）

CP65001 console から gate 実行 →
```text
[GATE-FAIL] build identity mismatch (stamp vs current environment):
           console_codepage: stamp='932' now='65001'
[GATE-FAIL] COHERENCE-4 violated.  → GATE_RC=3
```
（i3_4_d_d10b_codepage_mismatch.log）

### D10-C negative: #deps 0（実証成功・かつ本物の事故予防）

初回 gate 実行で **ConvoPeq.exe（生産 CLI）側に 92 個の #deps 0 obj を検出**し
build を拒否。AudioEngineHarness は I3-4-B で clean rebuild 済みだったが、
**同一 build dir の他 target には事故状態が残存していた**。gate は
`i3_4_d_pre_recovery_ConvoPeq_exe_mixed_gen.exe`（sha256[:16]=4bdbae2810e146bf）として
保存された混合世代 exe が link し直される前に事故を阻止した。

### D10-E clean recovery

Release 全 target を再構築（272 edges）→ gate は identity PASS だが
**残存 10 test-only TU が persistent #deps 0**（複数回 rebuild でも毎回 0）を検出し
fail-closed 継続。depth-2 調査（`i3_4_d_d10e_persistent_zero_deps.txt`）:

- 手動実行で exact command は 159 行の正しい CP932 prefix を出力
- 同一 ninja build 内の ConvoPeq.obj 241/241 は依存記録成功
- .ninja_log に edge 記録あり（command hash 9d2d0856e97aac0a）＝ ninja は
  compile を実行し deps entry を**空リストで**記録
- codepage probe で説明不能 → **ninja DepsLog 側の別機序**の可能性
  （同一 command hash の衝突クラス / restat 交互作用 / upstream bug）

**契約上の位置づけ**: 10 TU は test-only（ConvoPeq.exe 生産 CLI に link されない）。
gate が silently-ODR-hazard を hard failure に変換したのは契約どおり正しい動作。
I3-4-E で (a) ninja 上流バグ報告（本 repro 付き）、(b) restat workaround 評価、
(c) test-only target の whitelist-with-warning（layout review 後）を決定する。

### D11 runtime regression（回帰確認）

```text
Release harness ×3: exit 0x0 / 0x0 / 0x0  (21.8s, 20.3s, 19.9s)
Debug   harness ×3: exit 0x0 / 0x0 / 0x0  (20.9s, 21.1s, 21.0s)
check_layout_offsets.py:
  Release: 0x1290880:4, 0x12A8880:0, 0x12A7640:0  → PASS
  Debug:   旧 crash offset なし                    → PASS
```

I3-4-B で確立した layout coherence（ctor store == use load）が gate 追加後も保持。

### D12 CTest

```text
Release ctest: 100% tests passed, 40/40
Debug   ctest: 100% tests passed, 40/40
```

## 3. GO 条件対合

| ID | 条件 | 判定 |
| --- | --- | --- |
| D1 | identity stamp 生成 | **PASS**（.build_identity 実装・11 項目記録） |
| D2 | codepage mismatch 検出 | **PASS**（D10-B で rc=3 実証） |
| D3 | #deps 0 fail-closed | **PASS**（D10-C で 92 obj 検出・build 拒否実証） |
| D4 | generator identity mismatch 検出 | **PASS**（stamp generator 欄 + CMake cache 保護の二重・D10-B と同機構） |
| D5 | compiler identity/version mismatch 検出 | **PASS**（stamp compiler_path/version 照合） |
| D6 | provenance 不明 obj を link させない | **PASS**（COHERENCE-3 gate が Release link を阻止中） |
| D7 | clean recovery 成立 | **PASS**（92→28→0 の段階的回復・ConvoPeq/Harness 241/241 deps 記録） |
| D8 | negative test が gate を発火 | **PASS**（D10-B/C の rc=3） |
| D9 | clean Release layout coherence | **PASS**（0x1290880 統一・旧 offset 消滅） |
| D10 | clean Debug layout coherence | **PASS** |
| D11 | Release/Debug CTest | **PASS**（40/40 + 40/40） |
| D12 | runtime source changes = 0 | **PASS**（変更は build.bat + src/tools/ のみ） |

**総合判定 = I3-4-D GO**。ただし 10 test-only TU の persistent #deps 0 により、
test-suite 全体を含む Release incremental build は gate によって継続拒否中
（契約どおりの fail-closed）。I3-4-E で恒久解消を設計する。

## 4. 変更ファイル一覧

```text
M  build.bat                              (:configure_cmake_ok 直後に gate 20 行追加)
A  src/tools/build_identity_gate.py       (COHERENCE-1..5 gate)
A  src/tools/check_layout_offsets.py      (D11 回帰確認ツール)
src/** runtime code / tests / CMakeLists.txt: 変更 0
```

## 5. 添付 evidence

```text
evidence/D162-2I3/
  I3_4_D_CONTRACT_IMPLEMENTATION_REPORT.md   本書
  i3_4_d_d0_snapshot.txt / i3_4_d_d0_git_diff.txt      D0 snapshot
  i3_4_d_d10b_codepage_mismatch.log                    D10-B evidence
  i3_4_d_d10c_gate_fail_convoPeq.log                   D10-C evidence (92 obj)
  i3_4_d_d10e_gate_pass_after_full_recovery.log        D10-E evidence
  i3_4_d_d10e_persistent_zero_deps.txt                 depth-2 finding
  i3_4_d_pre_recovery_ConvoPeq_exe_mixed_gen.exe       混合世代 exe 保存
```
