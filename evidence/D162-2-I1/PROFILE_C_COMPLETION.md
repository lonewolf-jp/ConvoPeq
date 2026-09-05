# D162-2-I1-C — Profile C Completion Record（Repeated IR/rebuild・6/6 PASS）

```text
Date:     2026-09-04
Type:     Profile C 完遂（production 変更 0 / test 0 / build 0）
Baseline: ConvoPeq.md Generated 2026-09-04 23:14:29（R2 A′ 修復込み）
条件:     --cli-run --cli-ir D162-1P_active.wav --cli-rebuild --cli-exit-ms 15000
          （--cli-rebuild = IR load 後 500ms に structural rebuild intent を発行・MainWindow.cpp:1029-1036）
          run 間隔 2s
判定:     **Profile C = 6/6 PASS**・STOP 条件該当 0 件
```

---

## 0. 実行上の訂正記録（1 回目 run の無効化）

初回 C-1〜C-6 は `--cli-rebuild` フラグを付けずに実行したため **rebuild 0 回**
（IR load のみ・`[D125_IRFLAG_PROMOTE_SNAPSHOT] promoted=0`）となり、C の目的
（IR load → rebuild の因果系列確認）を満たさなかった。→ 6 log を
`evidence/D162-2-I1/invalid-no-rebuild-flag/` に保存（control data として有用:
IR load 単独でも shutdown は clean・residual 0）し、`--cli-rebuild` 付きで再実行。

## 1. Profile C 集計表（6 runs・--cli-rebuild 付き）

| 項目 | 実測 |
| --- | --- |
| runs | **6/6** |
| exit | **6/6 = 0x00000000** |
| dump | **0**（CrashDumps 10 不変） |
| zone | **6/6 = clean** |
| IR load / rebuild | **6/6 で IR load 成功 + rebuild 1 回**（BUILD_PHASE 1・[PUBLISH] 1・rebuild gen = 6/5/6/6/5/4） |
| generation | 構築 = 破壊 1:1（2/2 × 6・placeholder + rebuild DSP） |
| residual | **0 × 6/6**（remaining=0 全件） |
| EBR | pend 最終 **0**（E-3 = 0 で保証） / ovf **0** × 6/6 |
| stale-map | MISS 2 / HIT 1（midrun 正常 destroy・forbidden HIT **0**） × 6/6 |
| registered direct destroy | **0** × 6/6 |
| signatures | A / B / C = **0 / 0 / 0** × 6/6 |
| E-4 accounting | deferred 活動 0（0 = 0+0+0+0 × 6/6・rebuild が即 publish され churn なし） |
| CLEAR_SHUTDOWN_DISPOSITION | **0 × 6/6**（slot が shutdown 時点で空 = 契約上正常・FAIL ではない） |
| S3 disposition | 不発（slot 空）— ただし A′ 呼出自体は毎 run 実行（no-op 帰還） |
| memory | DC live=2 / Priv 548-549MB（6/6 同型） |
| XRUN | 0/0/1/1/2/1 = 計 5 件・Callback ≤1.23ms・Pressure=0・**shutdown window 0** × 6 |

## 2. 必須確認項目

### C-1: S3 必要性の判別

- **CLEAR_SHUTDOWN_DISPOSITION = 0 × 6/6**: rebuild が即 publish 完了し（churn なし）、
  shutdown 時点で deferred slot が空だったため。**契約上正常**（指示どおり FAIL ではない）。
- A′ 無条件呼出自体は 6/6 run で到達・no-op 帰還（冪等性の追加実証）。
- S3 retired=1 の実測は Profile B の 6/6 で確立済み。

### C-2: E-4 会計

- CREATE 0 = CONSUME 0 + OVERWRITE 0 + CLEAR 0 + DISCARD 0 × 6/6 — **完全収支**。
  （B profile との差: churn がないため CREATE 自体が発生しない。invariant は維持。）

### C-3: rebuild 世代連続性（old/new DSP lifecycle chain・C-1 実例）

```text
:402  BUILD_PHASE generation=6（build 84.4ms + rebuildIR 413.2ms）
:420  [PUBLISH] seq=6 gen=6 worldId=6                            ← gen6 publish 成功
:479  D117_RETIRE_BY_HANDLE dsp=A4FAD080 lookup=HIT（old gen5 DSP retire・epoch 12）
:3210 D117_DESTROY dsp=A4FAD080                                   ← old DSP EBR digest ✓
:3254 D117_RETIRE dsp=9E6CC080 retired=1（V-D active-final = gen6 DSP）
:3255 D117_RETIRE enqueue=0 epoch=15（EBR Success）
:3257 D117_RETIRE dsp=A4FAD080 retired=0（fading-final stale resolve → no-op ✓）
:3272 RETIRE_BY_HANDLE handle=252/254 lookup=MISS ×2（map erase 済み確認 ✓）
      D117_DESTROY dsp=9E6CC080 → remaining=0（dtor body D5/D8 内 digest）
```

- **publish → old DSP retire → EBR → destroy の chain が全 run で途切れず成立** ✓
- gen6 published DSP は V-D active-final で閉包（remaining=0）✓
- rebuild gen の振れ（6/5/6/6/5/4）は init timing 由来の採番差であり連続性に影響なし。

## 3. STOP 条件評価（12 項目 × 6 run）

exit≠0 / dump / T0-T2 / residual≠0 / EBR ovf / final pend≠0 / E-3≠0 / Signature / direct destroy /
forbidden HIT / gen mismatch / dup destroy / remaining≠0 / shutdown-window XRUN — **全 0 件**。

## 4. 判定

**D162-2-I1-C = Profile C 6/6 PASS。** A′ shutdown disposition は反復 rebuild profile でも
lifecycle 経路を壊さず（S3 不発 = slot 空の正当な no-op・E-4 会計成立・世代 1:1）。
次工程: **Profile D（device cycle ×6）**。device type は D-1 の
`[CLI_AUDIO_DEV_TYPES]` 実測で確定する。
