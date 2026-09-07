# D172-3 — MEM_SNAP RuntimeWorld Resolver Implementation Report

- Date: 2026-09-08
- Task: D172-2 GO contract に基づく MEM_SNAP resolver 実装 + 回帰検証
- Scope: production 1 箇所変更 + contract comments / 3 config build + CTest / diagnostic runtime validation
- Evidence: `evidence/D172/D172_3_MEM_SNAP_RESOLVER_IMPLEMENTATION.md` + logs

## 判定

> ## **D172-3 PASS — AC-1〜12 全項目 PASS**

| AC | 条件 | 判定 |
| --- | --- | --- |
| AC-1 | MEM_SNAP の DSP source が RuntimeWorld resolver に変更済み | PASS |
| AC-2 | `activeRuntimeDSPSlot` writer / retire / destroy path 変更 0 | PASS |
| AC-3 | 新規 lifetime authority 0 | PASS |
| AC-4 | RuntimeReadHandle lifetime scope 不変 | PASS |
| AC-5 | TRK semantics 変更をコメントで明記 | PASS |
| AC-6 | world 未公開時 TRK=0 | PASS |
| AC-7 | Debug CTest 40/40 | PASS |
| AC-8 | Release CTest 40/40 | PASS |
| AC-9 | diagnostic build MEM_SNAP 継続 | PASS |
| AC-10 | rebuild → retire → destroy 後 stale slot dereference なし | PASS |
| AC-11 | dormant `logRuntimeTransitionEvent` 再導入禁止契約 | PASS |
| AC-12 | Case 3 window を修正対象に拡大していない | PASS |

## 実装内容

1. **AudioEngine.Timer.cpp MEM_SNAP block（1 箇所）**: `getActiveRuntimeDSP()` → `resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle)`。既存 handle（callback 冒頭取得・callback 全体スコープ）と既存 resolver のみ使用 — 新規 helper / atomic / queue / registry なし。slot writer W1-W4・retire/destroy path は完全に触れていない。
2. **TRK 意味論コメント**: 「TRK source = RuntimeWorld current DSP / legacy slot は観測 authority として使用しない / world 未公開時 TRK=0」— 「slot を lifetime-safe 化した」という誤解を生む表現は不使用（正: MEM_SNAP が slot を読むことをやめた）。
3. **R3 契約コメント**（AudioEngine.h）: dormant `logRuntimeTransitionEvent` に「復活時は world resolution 経由に統一 — slot dereference 新規追加禁止」。

## 検証結果

| config | build | CTest | freshness gate |
| --- | --- | --- | --- |
| Debug | RC=0 | **40/40** | exe 00:15 > source 23:55 ✔ |
| Release | RC=0 | **40/40** | exe 00:29 > 23:55 ✔ |
| RWDI (diagnostic) | RC=0 | **40/40 ×2 連続** | exe 01:11 > 23:55 ✔ |

- **D172-3.0 baseline**: 方式 α の決定的実例は未取得（CLI flow では gen1 publish が先行し placeholder 分岐が不発 → slot null・TRK=0.0 6,058/6,058）。D172-1 source-level proof は変更なく、指示どおり implementation を継続。
- **D172-3.5 runtime validation（fresh diagnostic exe）**: IR reload storm + `--cli-rebuild` + IR 内容 swap（irA→irB）で構造 rebuild（gen4→gen6）→ 旧 DSP 物理破壊（`[DSP_FOOTPRINT_RELEASED] dsp=…3D8E0080`）→ **破壊直後以降の MEM_SNAP が新 world current DSP の TRK=1.2MB（OS=0.0 EQ=0.2 AL=0.2 LT=0.3）を全 236 サンプル出力・旧 address dereference 痕跡なし**・exit 0x0。旧コードでは slot null のため構造的に不可能な値（TRK 非ゼロ）であり、resolver への切替を行動学的に実証。

## 過程で確認した教訓（既知パターンの再確認）

1. **build-diag は Ninja Multi-Config** — `--config` 指定なしの `cmake --build` は Debug を対象にする。初回 RWDI CTest（AudioEngineHarness SEGFAULT 1 回）は**変更前 stale exe での実行**であり無効。freshness gate（exe mtime > source mtime）適用後に 40/40 ×2。D169-2-7 の freshness gate 教訓の再現。
2. mspdbsrv 強制終了起因の PDB 破損（C2471 / LNK1285）→ PDB 削除 + 再 build で解消。
3. TRK は旧コードでも D162-1P 記録どおり「常時 0.0（計測未配線）」— 本修正により初めて意味ある統計（world current DSP の実 member 値）を出力するようになった。

## 次工程

- **ConvoPeq.md 再生成**（`output_sourcecode_markdown.py`）— source 更新を派生 snapshot に反映（次監査監視 window で実施）
- D172 series closure: D172-1（STOP 証明）→ D172-2（契約 GO）→ D172-3（実装 PASS）で MEM_SNAP lifetime hazard track 完了
- doc-only maintenance（inventory STALE 化反映 + buildErrorCount_ trigger 登録）は別 window
