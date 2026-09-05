# D162-2-I1-B — Profile B Completion Record（修復後 B-1 再試験 + B-2〜B-6 = 6/6 PASS）

```text
Date:     2026-09-04
Type:     Profile B 完遂（production 変更 0 / test 0 / build 0）
Baseline: ConvoPeq.md Generated 2026-09-04 23:14:29（R2 A′ 修復込み）
条件:     B-1 と完全同一（--cli-run --cli-ir D162-1P_active.wav --cli-intent-burst-count 3
          --cli-intent-burst-interval-ms 2000 --cli-exit-ms 15000・run 間隔 2s）
判定:     **Profile B = 6/6 PASS**（B-1 post-fix + B-2〜B-6）・STOP 条件該当 0 件
```

---

## 1. Profile B 集計表（6 runs = B-1 post-fix + B-2〜B-6）

| 項目 | 実測 |
| --- | --- |
| runs | **6/6** |
| exit | **6/6 = 0x00000000** |
| dump | **0**（CrashDumps 10 不変） |
| zone | **6/6 = clean**（SHUTDOWN_BEGIN / reset completed / LOGGER_DETACH-END 全出現） |
| lifecycle | 6/6 で全構築 DSP が authority 経由で destroy → remaining=0 |
| EBR | pend 最終 **0** / ovf **0** × 6/6 |
| stale-map | MISS 2 / HIT 1（midrun 正常 destroy・禁止 HIT **0**） × 6/6 |
| registered direct destroy | **0** × 6/6 |
| E-3 / INV-D162-8 | **0** × 6/6 |
| signatures | A / B / C = **0 / 0 / 0** × 6/6 |
| residual | **0 × 6/6**（remaining≠0 0 件・remaining=0 全件） |
| XRUN | 1/1/0/1/2/1 件 = **計 7 件**・全て Callback ≤1.75ms・Pressure=0・**shutdown window 0** × 6（既知 jitter クラス） |

## 2. 必須記録 4 項目（指示 §B-2〜B-6）

### 2.1 E-4 完全収支（CREATE = CONSUME + OVERWRITE + CLEAR + DISCARD）

| run | CREATE | CONSUME | OVERWRITE | CLEAR | DISCARD | 収支 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B-1 (post-fix) | 68 | 67 | 0 | 1 | 0 | **68 = 68 ✓** |
| B-2 | 70 | 69 | 0 | 1 | 0 | **70 = 70 ✓** |
| B-3 | 66 | 65 | 0 | 1 | 0 | **66 = 66 ✓** |
| B-4 | 70 | 69 | 0 | 1 | 0 | **70 = 70 ✓** |
| B-5 | 70 | 69 | 0 | 1 | 0 | **70 = 70 ✓** |
| B-6 | 68 | 67 | 0 | 1 | 0 | **68 = 68 ✓** |

**CLEAR_SHUTDOWN_DISPOSITION: 6/6 run で = 1** — shutdown terminal disposition が
**毎回実際に働いた**（0 件の run は出現せず）。

### 2.2 residual

**6/6 = 0**（remaining≠0 0 件・destroy = release 全件・E-4 会計完全収支）。
B-1 pre-fix の residual=1 は**修復により 6 run 連続で 0** を維持。

### 2.3 S3 authority closure（6/6 完全閉包）

| run | S3 target | CLEAR_SHUT → retire(shutdown-clear) | retired | EBR | destroy | remaining |
| --- | --- | --- | --- | --- | --- | --- |
| B-1 post-fix | gen8 0000025EBA320080 | ✓ | **1** | Success (epoch 17) | ✓ | **0** |
| B-2 | 0000027D69300080 | ✓ | **1** | Success | ✓ | **0** |
| B-3 | 000002C75BD8F080 | ✓ | **1** | Success | ✓ | **0** |
| B-4 | 000001BA69A47080 | ✓ | **1** | Success | ✓ | **0** |
| B-5 | 000002794FA4C080 | ✓ | **1** | Success | ✓ | **0** |
| B-6 | 000001F1AB808080 | ✓ | **1** | Success | ✓ | **0** |

**S3 standalone `retired=1` が 6/6 run で再現**（G1-G4 の未観測事項が解消・
R0 Candidate A′ により S3 が shutdown terminal point に確実に到達）。

### 2.4 V-D-b stale resolve（regression guard・6/6 観測）

| run | fading-final resolve 対象 | retired |
| --- | --- | --- |
| B-1 post-fix | 0000025E9C7D7080（midrun 破壊済み dangling） | **0 = no-op 安全** |
| B-2 | 0000027D4E83A080（同型） | 0 |
| B-3 | 000002C739FDD080（同型） | 0 |
| B-4 | 000001BA583C1080（同型） | 0 |
| B-5 | 0000027935016080（同型） | 0 |
| B-6 | 000001F190DCE080（同型） | 0 |

**6/6 run で fading handle の stale resolve が発生し、V-D-b（authority retire）が全て
no-op で安全に帰還。** V-D-a（direct destroy）だった場合、6 run 全てで double-free に
なり得た経路 — **G0 Candidate A′ 選択の有効性が連続実証された**。

## 3. per-run 明細

| run | exit | zone | E-4 収支 | CLEAR_SHUT | S3 closure | destroy/release | gen 1:1 | stale | E-3 | Sig | XRUN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-1 post-fix | 0x0 | clean | ✓ | 1 | retired=1 | 4/4 | ✓ | M2/H1 | 0 | 0/0/0 | 1 |
| B-2 | 0x0 | clean | ✓ | 1 | retired=1 | 4/4 | ✓ | M2/H1 | 0 | 0/0/0 | 2 |
| B-3 | 0x0 | clean | ✓ | 1 | retired=1 | 4/4 | ✓ | M2/H1 | 0 | 0/0/0 | 0 |
| B-4 | 0x0 | clean | ✓ | 1 | retired=1 | 3/3 | ✓ | M2/H1 | 0 | 0/0/0 | 1 |
| B-5 | 0x0 | clean | ✓ | 1 | retired=1 | 4/4 | ✓ | M2/H1 | 0 | 0/0/0 | 2 |
| B-6 | 0x0 | clean | ✓ | 1 | retired=1 | 4/4 | ✓ | M2/H1 | 0 | 0/0/0 | 1 |

- rebuild builds: 3/run（burst 3 intents）・placeholder + 3 = 構築 4 個体/run（B-4 のみ 3 個体 = burst timing variance）→ destroyed = constructed（1:1・gap なし・重複 0）
- V-D active-final: retired=1 × 6/6（各 run 1 個体）

## 4. 判定

**D162-2-I1-B = Profile B 6/6 PASS。** R0/R1/R2 の修復（A′）により:

1. **B-1 pre-fix の residual=1 は解消**（6 run 連続 residual 0）
2. **E-4 会計が短時間 shutdown でも完全収支**（CLEAR_SHUTDOWN_DISPOSITION 6/6）
3. **S3 standalone `retired=1` が 6/6 再現**（未観測事項の解消）
4. **V-D-b の stale resolve 防御が 6/6 動作**（V-D-a なら 6 回の double-free 潜在）

**次工程**: Profile C（Repeated IR/rebuild ×6）→ Profile D（device cycle ×6）→ I1 final。
（crossfade 非完了 root cause 調査は引き続き DEFER。）
