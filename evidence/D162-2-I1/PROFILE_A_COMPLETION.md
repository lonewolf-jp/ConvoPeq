# D162-2-I1-A — Profile A Completion Record（A-1〜A-8・8/8 PASS）

```text
Date:     2026-09-04
Type:     Profile A 残り 7 run（A-2〜A-8）+ 集計。production 変更 0 / test 0 / build 0
Baseline: ConvoPeq.md Generated 2026-09-04 20:28:50・binary = G4-1 と同一 RWDI（Sep 4 21:16）
条件:     A-1 pilot と完全同一 CLI（--cli-run --cli-log-file <log> --cli-exit-ms 15000・IR なし）
          run 間隔 2s・exit capture = PowerShell $p.ExitCode
判定:     **Profile A = 8/8 PASS**（STOP 条件該当 0 件）
```

---

## 1. Profile A 集計表（8 runs）

| 項目 | 実測 |
| --- | --- |
| runs | **8/8**（A-1 pilot + A-2〜A-8） |
| exit | **8/8 = 0x00000000**（PowerShell process object capture） |
| dump | **0**（CrashDumps 巡查: run 前後で新規 ConvoPeq dump なし・総数 10 で不変） |
| zone | **8/8 = clean**（SHUTDOWN_BEGIN / reset completed / LOGGER_DETACH-SHUTDOWN_END の 3 行全出現） |
| lifecycle | 8/8 で **V-D が placeholder（active-final）を `retired=1 → EBR Success(epoch 10) → D117_DESTROY → FOOTPRINT_RELEASED remaining=0`** で閉包（plain profile は V-D 主役経路・S3 は deferred 活動 0 のため正当に不発） |
| generation | 構築 1 = 破壊 1（gen 単位 1:1）・duplicate destroy **0** |
| EBR | pend 最終 **0** / ovf **0**（8/8） |
| stale-map | **MISS 1 / HIT 0** × 8/8（V-D の authority erase → dtor retireByHandle MISS = no-op 正常） |
| registered direct destroy | **0**（`[D162-2B_DESTROY]` 0 件 × 8/8） |
| E-3 / INV-D162-8 | **0 件** × 8/8 |
| signatures | A / B / C = **0 / 0 / 0** × 8/8 |
| residual | **0**（remaining≠0 0 件 × 8/8） |
| E-4 accounting | deferred 活動 0（plain）→ 収支 0=0 × 8/8 |
| memory | 最終 MEM_SNAP: DC live=1・Priv ~395MB 台・NUC live=0（8/8 同型） |
| XRUN | 計 42 件 / 8 runs（4〜9 件/run）— **全て startup transient**（run 冒頭 ~12s に集中）・Callback max **1.72ms**（A-1・Expected 5.33ms 対し軽微）・Pressure=**0 全件**・**shutdown window 0 件**（8/8） |

### per-run 明細

| run | exit | zone | V-D closure | destroy/release | stale | E-3 | Sig | XRUN (cbMax / shutdown窓) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-1 | 0x0 | clean | retired=1→destroy rem=0 | 1/1 | MISS1/HIT0 | 0 | 0/0/0 | 9 / 1.72ms / 0 |
| A-2 | 0x0 | clean | 同型 | 1/1 | MISS1/HIT0 | 0 | 0/0/0 | 4 / 1.37ms / 0 |
| A-3 | 0x0 | clean | 同型 | 1/1 | MISS1/HIT0 | 0 | 0/0/0 | 5 / 1.14ms / 0 |
| A-4 | 0x0 | clean | 同型 | 1/1 | MISS1/HIT0 | 0 | 0/0/0 | 8 / 0.89ms / 0 |
| A-5 | 0x0 | clean | 同型 | 1/1 | MISS1/HIT0 | 0 | 0/0/0 | 4 / 0.84ms / 0 |
| A-6 | 0x0 | clean | 同型 | 1/1 | MISS1/HIT0 | 0 | 0/0/0 | 4 / 0.78ms / 0 |
| A-7 | 0x0 | clean | 同型 | 1/1 | MISS1/HIT0 | 0 | 0/0/0 | 6 / 1.10ms / 0 |
| A-8 | 0x0 | clean | 同型 | 1/1 | MISS1/HIT0 | 0 | 0/0/0 | 6 / 0.82ms / 0 |

## 2. STOP 条件評価（run 時点で逐次評価・該当 0）

```text
exit != 0                → 0 件
新規 ConvoPeq dump       → 0 件
T0/T1/T2                 → 0 件（8/8 clean）
E-3 != 0                 → 0 件
Signature A/B/C          → 0 件
EBR overflow             → 0 件
final pending != 0       → 0 件
residual != 0            → 0 件
generation mismatch / duplicate destroy → 0 件
registered DSP direct destroy → 0 件
forbidden stale-map HIT  → 0 件
```

XRUN: 全 run startup transient・Pressure=0・shutdown window 0 件 → **件数を理由にした STOP は不発**（I0 契約どおり）。

## 3. D120 historical との比較（Profile A 部分）

```text
D120 historical (frozen binary):  plain 6s  = 2/5 crash
I1 current  (G4 config + 修正測定): plain 15s = **0/8 crash**
```

plain profile における teardown race の非再現を 8 連続で記録（D116 の無効測定ではなく
PowerShell 正規 capture による初めての有効な plain profile 記録）。

## 4. 次工程

**Profile A 完了（8/8 PASS）→ B-1 開始の GO/STOP 判定をユーザーが実施。**
B（IR + rebuild + shutdown ×6）は D120 historical 3/3 crash condition の再現試験であり、
I1 の核心 profile である。
