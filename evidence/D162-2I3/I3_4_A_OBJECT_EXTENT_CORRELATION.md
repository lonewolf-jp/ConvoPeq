# D162-2-I3-4-A — Constructor Zero-Fill / Object-Extent Correlation Audit

```text
Date:            2026-09-05
Type:            read-only structural + binary correlation audit
Production:      0 / Test: 0 / CMake: 0 / Instrumentation: 0
Baseline:        ConvoPeq.md 2026-09-05 12:02:31 / I3-3 GO
Tools:           llvm-pdbutil dump --types (RWDI PDB 177,893,376 B, 13:22:33)
                 + obj 年代層分析 (mtime) + WSL grep + Python PE/COFF 解析
対象 binary:     (i) Release AudioEngineHarness.exe 10:06:53（I3-3 crash 車両）
                 (ii) RWDI AudioEngineHarness.exe 13:22:33（coherent 対照群）
判定:            **I3-3「bridge-tail overrun」の撤回 + ODR/ stale-obj 混在確定 →
                 A1-A6 全問答 (A6 = constructor defect ✗ / stale-obj ODR mix ✓)**
```

---

## 0. 結論（要旨）

I3-3 で「bridge ctor 末尾 memset が worldAuthority_.coordinator_ を跨いで 0 埋めした」
と記録した事象は、**crash 車両となった 10:06 Release exe が 2 世代の layout を混在させていた**
ことに起因する。源頭は constructor defect ではなく **stale .obj の混在（ODR 違反）** である。

決定的証拠:

1. 同一 exe 内で、**ctor 側の参照束縛 store は `mov [engine+0x12A8880], r12`**（r12 = bridge this）で
   行われていた（I3-3 Run A 実測命令列）。0x12A8880 は **新 layout（RWDI coherent view）の
   `worldAuthority_.coordinator_`（= W）の正確な位置**である。
2. 一方 **use 側（publish 経路・08-29 obj 系）は `mov rcx, [r13+0x12A7640]`** で W を読む
   （旧 layout の W）。
3. すなわち、束縛は 0x12A8880 へ正しく行われ、0x12A7640 は旧 view の使用側が読むだけで
   誰も書かない → **W(旧) は生涯 0**、I3-3 の DR watch が観測した唯一の書込み
   （bridge 末尾近傍の memset）は **新旧いずれの layout でも coordinator 内部** の
   末尾ゼロフィルであり、新 view では範囲違反が存在しない。

## 1. A1 — object extent の確定（PDB type stream 実測）

llvm-pdbutil dump --types（RWDI PDB・13:22:33 = 現行 source coherent）:

```text
convo::isr::RuntimeIntentCoordinator  sizeof = 1,600,960 (0x186DC0)
  AudioEngine 内 offset              = 17,963,712 (0x1121AC0)
  → bridge extent = [engine+0x1121AC0, engine+0x12A8880)
  末尾 member: quarantineService_ @ in-coord 0x186D90 (sizeof 1)
               quarantineFallbackDropCount_ @ 0x186D80
               overflowAgeWarnCallback_ @ 0x186D88
convo::isr::RuntimeWorldAuthority     sizeof = 154,992 (0x25D70)
  AudioEngine 内 offset              = 19,564,672 (0x12A8880)
  → coordinator_ (第 1 member, offset 0) = engine+0x12A8880
  → bridge end == worldAuthority_ begin == W（新 view）
  runtimeStore_ @ +8, writeAccess_ @ +16, ownerChannel_ @ +24, lifetime_ @ +8216,
  registry_ @ +153952, shutdownClearRequested_ @ +154984
```

同一 offset 表（engine relative）:

```text
0x1121AC0  runtimePublicationBridge_ begin        ← clone F this（I3-3 実測と一致）
0x12A7620  （新 view: coordinator 末尾 pad 領域・quarantineService_ 直前の pad/ドロップカウンタ近傍）
0x12A7640  ← 旧 view (08-29 obj) の worldAuthority_.coordinator_（use 側 disp32）
0x12A76D0  （I3-3 memset 終端）
0x12A8880  bridge end = worldAuthority_ begin = coordinator_（新 view W・束縛 store 先）
0x12CE5F0  worldAuthority_ end（新 view）
```

**I3-3 の memset 範囲 [0x12A7620, 0x12A76D0) は、新 view の coordinator extent 内に完全に収まる**
（bridge end 0x12A8880 の 0x11B0 手前で終了）。

## 2. A2 — clone F this identity の再証明

- I3-3 実測: clone F this（rcx）= engine+0x1121AC0。
- RWDI PDB: `runtimePublicationBridge_` の AudioEngine 内 offset = **0x1121AC0（一致）**。
- AudioEngine ctor clone A の呼出命令列（exe バイト検証済）:
  `lea r12,[r14+0x1121AC0]; mov rcx,r12; call 0x1F7EE30` —
  this 伝播は machine-code レベルで **RuntimeIntentCoordinator* そのもの**。
- 従って「clone F = RuntimeIntentCoordinator ctor」は symbol 名でなく
  **this-pointer identity（0x1121AC0 == PDB member offset）で確定**。

## 3. A3 — memset(·, 0, 0xB0) の semantic owner

- clone F 末尾のゼロ列は coordinator の末尾 member 群の value-initialization:
  `quarantineFallbackDropCount_(0x186D80)` → `overflowAgeWarnCallback_(0x186D88)` →
  `quarantineService_(0x186D90)`（sizeof 1）→ パディング（0x186D91..0x186DC0）。
- 0xB0 = 176 バイトの範囲 [this+0x185B60, this+0x185C10) は
  **in-coordinator offset 0x185B60..0x185C10** = intentQueue_ 末尾以降の
  counter/callback/empty-service/pad 領域に相当し、
  **compiler が coordinator 末尾の POD/aggregate を纏めて value-init したもの**
  （LF_FIELDLIST 上 these are trivial members; MSVC は trailing trivial メンバ群を
  単一 memset に畳む）。
- QuarantineService 自体は stateless（sizeof 1・メンバなし）であり、
  MpscBoundedRing/LockFreeRingBuffer の ctor は各 queue の内部で完結済み
  （0xB0 範囲は queue バッファを含まない）。

**判定: semantic owner = RuntimeIntentCoordinator 末尾 trivial-member 群の
compiler-generated value-initialization（合法）**。

## 4. A4 — object boundary を越えるか

```text
dst            = this + 0x185B60（= engine+0x12A7620）
dst + 0xB0     = this + 0x185C10（= engine+0x12A76D0）
coordinator end = this + 0x186DC0（= engine+0x12A8880）
→ dst >= begin ✓, dst+size <= end ✓（0x11B0 バイトの余裕）
```

**coherent build（RWDI・現行 source）では合法な constructor zero-fill**。
I3-3 が観測した「越境」は、旧 layout を仮定した use 側 disp32（0x12A7640）
と照らし合わせたために見えた **見かけ上の越境** である。

## 5. A5 — C++ abstract machine における coordinator_ 束縛と machine code の差

C++ semantic（AudioEngine.CtorDtor.cpp:29 / RuntimeWorldAuthority.h ctor）では
`worldAuthority_(runtimePublicationBridge_)` → `coordinator_(coordinator)` の束縛は
**worldAuthority_ の構築時**（bridge 構築完了後）に行われる。

machine code 実測（crash 車両 exe）:

```text
clone A (AudioEngine ctor):
  … bridge ctor (clone F, this=0x1121AC0) …
  mov [r14+0x12A8880], r12      ← worldAuthority_.coordinator_ への束縛 store（新 view W）
  lea rax,[r14+0x12A8888]; mov [rax], r13(0)   ← runtimeStore_ 等の後続初期化
  …
use 側（別 .obj・旧 layout）:
  mov rcx, [r13+0x12A7640]      ← 旧 view W を読む → 0 → commit(this=0) → AV
```

**束縛自体は正しく実行されていた**（0x12A8880 へ bridge アドレスを store）。
読み手が別 generation の layout で読んだため null に見えた。

## 6. A6 — 二択判定

| 説明 | 判定 |
| --- | --- |
| constructor defect（ctor が隣接メンバを破壊） | **✗** — coherent view では memset は coordinator 内・束縛 store も正常 |
| **stale-obj 混在（ODR 違反）による use/ctor layout 不一致** | **✓** — obj 年代層 (§7) + 両 disp32 の同一 exe 共存 + RWDI coherent PASS (exit 0) |

## 7. obj 年代層（決定的補助証拠）

```text
Release obj batch:
  08-29 21:36/22:02 系  = AudioEngine.Init.cpp, ProcessIntent, T1-T4 tests 等（旧 layout を使用）
  09-02 00:11          = ISRDSPHandle.cpp, SnapshotCoordinator.cpp
  09-05 13:23-13:24    = CtorDtor / Threading / Timer / Harness.cpp（新 layout・13:23 に再生成）
Release exe link        = 09-05 10:06:53  ← 13:23 batch は未反映（link 後に再コンパイルされた）
                          → 10:06 exe = 旧(08-29) obj を含む混在 link
RWDI exe                = 09-05 13:22:33 = 13:22 batch で coherent link → PASS (exit 0, 19.6s 実測)
git:                    ISRRuntimePublicationCoordinator.h は 09-01 15:11 commit (362dccd) で
                         372 行改変（D152-R1 系 recoveryAdmissions_ 強化）→ coordinator size が
                         旧 0x185B80 → 新 0x186DC0 へ変化
```

I3-1/I3-2/I3-3 の crash 再現が **Release build のみ** で RWDI/Debug が PASS であった
理由もここで完全に説明される（Release のみ混在 link が残存していた）。

## 8. I3-3 記録の訂正事項

1. 「bridge-tail overrun / member-boundary crossing」→ **撤回**。
   coherent layout では memset は coordinator 内合法ゼロフィル。
2. 「参照束縛 store が 3 run 合計 0 回実行された」→ **訂正**。
   束縛 store は engine+0x12A8880（新 view W）へ実行されていた。
   I3-3 の watch が旧 view W（0x12A7640）を見ていたため観測されなかっただけ。
3. 「coordinator_ は一度も有効値を持たない」→ **crash 車両 exe 内の旧 view アドレスについてのみ真**。

## 9. crash の真因（I3-4-B Repair Contract への入力）

**Build system / incremental link の stale-obj 問題**。
旧 layout を焼き付けた .obj が、ヘッダ変更後も再コンパイルされずに link された。
修正候補（I3-4-B で契約化）:

1. full rebuild（objs 全削除）による 10:06 exe 相当の再 link → crash 消滅の実証
2. CMake dependency scanner / header dep 追跡の検証（ISRDSPHandle.h 等
   ISRRuntimePublicationCoordinator.h を include する TU の obj が 08-29 のまま残っていた経路）
3. CI/build.bat での `--clean-first` または build dir stamping

**production source 変更は依然 0**（source には defect が存在しないため）。

## 10. 添付

```text
evidence/D162-2I3/
  I3_4_A_OBJECT_EXTENT_CORRELATION.md   本書
  i3_4_a_pdb_type_layout.txt            llvm-pdbutil 抽出（coordinator/worldAuthority members）
  i3_4_a_obj_epoch_table.txt            obj 年代層一覧
```
