# D162-2-I3-4-B — Build/Dependency Integrity Audit + Clean-Rebuild Control Experiment

```text
Date:            2026-09-05
Type:            read-only build provenance audit + build-artifact clean rebuild
                 (production source 0 / test source 0 / CMake source 0)
Baseline:        ConvoPeq.md 2026-09-05 12:02:31 / I3-4-A GO
Build system:    CMake 4.4 + Ninja Multi-Config, deps = msvc (/showIncludes), cl.exe 14.51.36231
Targets:         crash vehicle Release exe (10:06:53) vs clean-rebuild Release exe (19:02)
判定:            **B1-B8 全 PASS → I3-4-B GO。crash 根因 = stale .obj 混在 (ODR/layout 不一致) で確定**
```

---

## 0. 結論（要旨）

1. crash 車両 exe（10:06:53 link）は **旧 layout を焼き付けた 08-29 obj**（88/143 個が
   `.ninja_deps` に依存情報なし＝ #deps 0）と **新 layout obj** を混在 link していた。
2. dependency tracking（deps=msvc + /showIncludes）自体は**有効だった**。しかし
   08-29 obj は **依存情報が記録されないまま存在** し、ninja は mtime 判定で
   「直接入力（.cpp）が変わっていない → 再構築不要」と判断し続けた。
   header は direct input ではないため、deps 情報が無い限り header 変更は検知不能。
3. **full clean rebuild（143/143 obj 再生成）後は、ctor 側の束縛 store と use 側の load が
   同一 offset（0x1290880）に統一**され、Release harness は **3/3 exit 0x0（crash 消滅）**。
4. 以上により I3-4-A の「stale-obj 混在が十分原因」が **B5 falsification 条件を満たして確定**。

## 1. B0 — Baseline 固定

```text
git HEAD                0aeb22c (worktree dirty ~95 files; layout 影響 edit は 09-05 10:04 まで)
ConvoPeq.md             2026-09-05 12:02:31
generator               Ninja Multi-Config / cl.exe 14.51.36231 / deps=msvc
crash vehicle exe       Release/AudioEngineHarness.exe 10:06:53 (sha256[:16] ca2dc59a1f004bde)
RWDI coherent exe       13:22:33 (f12883193cf658e5)
Debug exe               09:57:51 (2128ab7feed6f92f)
layout (coherent PDB)   sizeof(RuntimeIntentCoordinator)=1,600,960 @ engine+0x1121AC0
                        worldAuthority_ @ engine+0x12A8880 (= bridge end)
link 時刻以降の obj     10:06:53 link 後に生成された 13:23-13:24 obj batch が同一 build dir に
                        存在（exe には未反映）— I3-4-A epoch 表で実証済み
```

3 exe は evidence に保存済み（`i3_4_b_*_exe_*.exe`）。

## 2. B1 — stale-object provenance

```text
ISRRuntimePublicationCoordinator.h (08-31 23:38 mtime / 09-01 362dccd で 372 行改変)
   ↓ include（direct/transitive, .ninja_deps VALID obj では 747 deps として記録される）
   ↓ AudioEngine.CtorDtor.cpp / Threading.cpp / Timer.cpp / ProcessIntent.cpp /
     Init.cpp / RuntimeWorldAuthority 使用 TU / tests (T1-T4, Soak, DeferredFlow 等)
   ↓ .obj
   ↓ 08-29 21:36 batch（ProcessIntent / Init / test 系 88 個）… deps 記録なし (#deps 0)
   ↓ Release exe link 10:06:53（08-29 obj + 09-02 obj 2 個 + 09-05 obj の混在）
```

- **#deps 0 の意味**: その obj がコンパイルされたとき、ninja は /showIncludes 出力から
  依存を 1 件も記録できなかった。以後 header が何回変わっても **ninja は header 変更を
  検知できない**（header は direct input ではなく、deps 経由でのみ追跡される）。
- crash-critical 2 obj は共に zero-dep:
  `ISRRuntimePublicationCoordinator_ProcessIntent.cpp.obj`（use 側 W 読出し元）
  `AudioEngine.Init.cpp.obj`。
- 一方 13:23 batch（CtorDtor/Threading/Timer/Harness）は #deps 740-753 が正常記録 →
  新 layout で再構築されていた。

## 3. B2 — dependency tracking の実証

```text
rules.ninja: rule CXX_COMPILER__*_unscanned_Release { deps = msvc; command = cl.exe … /showIncludes … }
build.ninja / impl-Release.ninja: regen 09-02 00:11:18（build dir 内の glob 変更検知のみ自動再走）
PCH / unity build: なし / JUCE generated headers: JuceLibraryCode 配下（通常の include 依存）
custom script: build.bat は cmake --build の thin wrapper（RC cleanup のみ・obj には無関係）
```

**問題は mechanism ではなく state**。tracking は正しく設定されていたが、
08-29 obj の deps state が空だったため mechanism が機能しなかった。

## 4. B3 — incremental build が stale obj を再利用することの再現

```text
ninja -f impl-Release.ninja -n AudioEngineHarness.exe (dry-run, 事故前 .ninja_deps のまま)
→ plan: CMake glob recheck のみ・**ProcessIntent/CtorDtor ともに再構築対象外**
→ "header changed → should rebuild → did NOT rebuild" を dry-run で実証
   （#deps 0 の obj は header が新しくても direct input の .cpp が同じなら skip）
```

## 5. B4 — Full clean rebuild（対照実験）

実施内容（生産 source 無変更）:

1. crash 車両 3 exe を evidence へ保存（sha256 記録）
2. build dir の build state を初期化（事故記録 §8 参照 — CMakeCache の generator 崩壊
   を含む。結果的に **完全な fresh configure + 全 obj 再生成** となり B4 の要件を満たす）
3. 再 configure: `Ninja Multi-Config` + `cl.exe 14.51.36231`（明示指定）
4. `cmake --build --config Release --target AudioEngineHarness` → **[146/146] Link OK**
   - obj 143/143 が 09-05 19:00:48-19:02:52 に全再生成（epoch 表: i3_4_b_obj_epoch_table_clean.txt）

## 6. B5 — clean exe の layout coherence（決定証拠）

```text
crash 車両 exe (10:06:53):    ctor store → [engine+0x12A8880]
                              use  load  → [engine+0x12A7640]   ← 不一致（0x1240 差）
RWDI coherent (13:22:33):     両者 → 0x12A8880
clean rebuild (19:02):        ctor store → [engine+0x1290880]   (0x1F5C7CA: mov [r14+0x1290880],r12)
                              use  load  → [engine+0x1290880]   (0x1F624A5/0x1F624B1: mov rcx,[r13+0x1290880])
                              **両者完全一致**
```

- 旧 W disp32 (0x12A7640): crash 車両 3 件 → **clean exe 0 件**（消滅）
- 0xB0 memset の「越境」も消滅（clean exe では coordinator extent 内）

## 7. B6 — crash 消滅の実証

```text
clean Release harness free run ×3:
  run1: exit 0x0, 21.8s
  run2: exit 0x0, 19.9s
  run3: exit 0x0, 20.0s
  → 旧 crash（0.05s / 0xC0000005 / RVA 0x1F7F89D）は再現しない
```

## 8. B7 — Debug coherence

```text
Debug build（同一 cl.exe・同一 cache）: [147/147] link OK → harness exit 0x0 (24.7s)
→ Debug/RWDI/Release の 3 config が同一 source・同一 layout generation で coherent
```

## 9. GO 条件対合

| ID | 条件 | 判定 |
| --- | --- | --- |
| B1 | stale/new obj 世代混在の再現 | **PASS**（I3-4-A epoch 表 + 本監査 §1/§2） |
| B2 | dependency chain 特定 | **PASS**（header→TU→obj、deps=msvc は有効・state が空） |
| B3 | incremental が stale obj を再利用する理由 | **PASS**（#deps 0 → header 変更検知不能・dry-run 実証） |
| B4 | clean rebuild で全 affected TU 再コンパイル | **PASS**（143/143 obj 19:00-19:02） |
| B5 | clean exe で ctor/use offset 一致 | **PASS**（0x1290880 統一・旧 disp32 消滅） |
| B6 | clean Release で旧 crash 非再現 | **PASS**（3/3 exit 0x0） |
| B7 | Debug/RWDI/Release coherence | **PASS**（3 config 同一 generation） |
| B8 | production/test/CMake source 変更 0 | **PASS**（git diff で実証・build dir は artifact のみ） |

**総合判定 = I3-4-B GO → D162-2-I3-4-C（Build-system permanent prevention contract）へ進行**

## 10. 補足: build dir 操作の事故記録（透明性のため）

監査中に 2 件の tooling ミスが発生した。いずれも **evidence への影響はゼロ**
（crash 車両 exe は事前保存・解析済みの証拠は全て evidence/ に書き出し済み）:

1. Git Bash での `cmake --regenerate-during-build` 実行時にバックスラッシュ引数が
   崩壊し、CMake が **既定 generator (Visual Studio 18 2026) で build-diag の cache を
   上書き** → Ninja の impl-*.ninja / rules.ninja / 全 obj dir が削除された。
2. 復旧: cache 削除 → `Ninja Multi-Config` + `cl.exe` 明示指定で reconfigure →
   結果的に B4 の full clean rebuild として成立（上記 §5）。
   1 回目の build 試行は `clang++` が選ばれたため失敗（LLVM が PATH にあった）→
   cl.exe 明示指定で解決。
3. 教訓: **`cmake --regenerate-during-build` を interactive shell から生で呼んではならない**
   （MSYS path mangling・CMAKE_GENERATOR 継承）。以後は `cmake -G` 明示 + bat 経由のみ。

## 11. 添付

```text
evidence/D162-2I3/
  I3_4_B_BUILD_INTEGRITY_AUDIT.md        本書
  i3_4_b_b0_baseline.txt                 B0 baseline snapshot
  i3_4_b_obj_epoch_table_clean.txt       clean rebuild 後 obj epoch 表 (143)
  i3_4_b_crash_vehicle_Release_exe_20260905_100653.exe   crash 車両保存
  i3_4_b_rwdi_coherent_exe_20260905_132233.exe           RWDI 対照群保存
  i3_4_b_debug_exe_20260905_095751.exe                   Debug 保存
  (I3-4-A 分: i3_4_a_obj_epoch_table.txt / i3_4_a_pdb_type_layout.txt)
```
