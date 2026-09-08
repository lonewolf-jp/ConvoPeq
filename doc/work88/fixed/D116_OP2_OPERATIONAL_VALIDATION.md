# D116-OP-2 — Current-HEAD Operational Validation（Work Report）

```text
D116-OP-2 — Current-HEAD Operational Validation

Type: read-only / operational validation（production source 変更 0）
Date: 2026-09-01
Baseline: ConvoPeq.md Generated 2026-09-01 21:47:45（--check 実測 FRESH / NEWER_SRC_COUNT=0 / CHECK_EXIT=0）
Production source: 0 / Test source: 0 / CMake: 0 / Build: 0（既存 exe 使用）/ CTest: 0 / stress: 0
CR-α reopen: 0 / D159 frozen area 接触: 0 / Episode layer: 0 / E_max・O_max・E×O≤32: 0
BuildErrorPolicy 変更: 0 / RetryScheduler 変更: 0
```

## 総合判定

> ## **D116-OP-2 = OP-2-A で TRIGGERED → 停止（指示どおり原因調査・修正には進まない）**
>
> **OP-2-A（長時間 memory soak）: memory が publication/rebuild 反復に比例して線形増加
> （plateau なし）** — 判定表「Memory 線形増加 → **即 TRIGGERED**」に該当。
> 現行 HEAD（HW-1 retirePublishedDSP 配線済み Release exe）でも D116-6 修正前と同規模の
> メモリ滞留が再現した。OP-2-B / OP-2-C / OP-2-D は **OP-2-A PASS 条件付きのため未実施**。
> 本報告は観測の記録と **TRIGGERED work item の起票**のみを行う（原因調査は次 work item で実施）。

---

## 1. 実行環境

```text
machine : Windows 11 26200 x64（本検証環境・実オーディオデバイス経由の CLI モード）
binary  : build/ConvoPeq_artefacts/Release/ConvoPeq.exe（47,785,984 B・2026-09-01 21:33 生成
          = CR-α-3R Release full build・現行 HEAD 相当・CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF）
git     : working tree = CR-α closed 状態（BuildErrorPolicy.h +81/RebuildDispatch +67/-17/test +88/-2）
```

## 2. exact command line（OP-2-A）

```text
# 事前: cp evidence/D116_irA.wav evidence/D116_OP2_active.wav
powershell.exe -NoProfile -ExecutionPolicy Bypass -File evidence/D116_memory_sampler_poll.ps1 ^
  "C:\VSC_Project\ConvoPeq\evidence\D116_OP2_memory_soak.csv" 480000   &
python evidence/D116_ir_swapper.py "C:\VSC_Project\ConvoPeq\evidence\D116_OP2_active.wav" 420 6 &

cmd //c "build\ConvoPeq_artefacts\Release\ConvoPeq.exe --cli-run ^
  --cli-log-file evidence\D116_OP2_soak.log ^
  --cli-ir evidence\D116_OP2_active.wav ^
  --cli-ir-reload-count 60 --cli-ir-reload-interval-ms 6000 ^
  --cli-intent-burst-count 60 --cli-intent-burst-interval-ms 6000 ^
  --cli-exit-ms 420000"
```

## 3. 実行時間・4. publication/rebuild 回数

```text
実行時間        : 426 秒（--cli-exit-ms 420000 で自動終了・APP_EXIT=0・crash/hang なし）
reload         : 60/60 実行（[CLI] IR reload iteration=1..60 実測）
rebuild generations: 59 件処理（"rebuildThreadLoop: generation=4..62 build=…" 実測 59 gen）
publications   : 10 件（[PUBLISH] seq=6,9,12,…,33 gen=6..33 worldId 同番・3 gen に 1 回 publish）
publishDurationUs: 645–3189 µs（RT コールバック外）
※ D116-6 修正前実績（同条件フラグ）: 59 publications — 本実行では 3 gen に 1 回しか publish
   していないが、memory 増加は generation 処理数に追従（§6 参照）
```

## 5. memory 時系列（D116_OP2_memory_soak.csv・2s 間隔 209 サンプル）

| 時点 | Private Memory |
|---|---|
| 4.1 s（起動直後） | 82.1 MB |
| ~84 s | 2,097 MB（初期割当 + 準備完了後） |
| 165 s | 3,607 MB |
| 245 s | 5,415 MB |
| 326 s | 7,073 MB |
| 422–426 s（終端） | 7,829 → 7,740 MB（shutdown 解放開始） |

```text
傾き（全区間）   : +1,088.2 MB/min（線形・単調）
活性区間傾き     : 約 +1,130〜1,340 MB/min（80s 窓ごと）
per-generation  : (7,740 − 548) MB ÷ 59 gen ≈ 122 MB / rebuild generation
对照: 修正前 D116-6 = +1,453.9 MB/min・59 publications（D116_longrun_memory.csv 実測再計算）
     → 現行 HEAD でも同規模（オーダー一致）の線形滞留が再現
```

## 6. retire / reclaim counters について（正直な記録）

```text
CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF の Release バイナリのため、
pendingRetireCount / quarantineResident / reclaimCount / NUC liveCount の時系列は
ログに出力されない（BUILD_PHASE 行 = 0 件を実測 — D116 と同一の制約）。
代替担保: 外部 sampler（Private/WorkingSet 時系列）+ [PUBLISH] / generation ログ集計。
internal counter の直接取得は診断ビルドを要する → 起票 work item の初手として推奨。
shutdown 時観測（下記 §10 の観測行）: pendingPub=0 pendingRetire=0 crossfade=0
routerPendingRetire=1 maxDeferredAgeMs=578 deferred=0 quarantine=0
oldestAgeMs=361325（= 361 秒滞留していた router pending-retire エントリ 1 件・"observation only"）
```

## 7. failure injection 結果（OP-2-B）

**未実施** — OP-2-B は「OP-2-A が PASS した場合のみ実施」と指示に明記。OP-2-A = TRIGGERED のため
実施せず。異常 IR（evidence/D116_OP2_corrupt.wav・irB の 600 byte truncate 版）は作成済みのまま保存。

## 8. deferred recovery 結果（OP-2-C）

**未実施**（OP-2-A TRIGGERED による停止・上記と同じ理由）。markTransientFailure → redrive 鎖の
実機通過確認は次回の operational validation に持ち越し。

## 9. restart cycle 結果（OP-2-D）

**未実施**（同上・停止順序遵守）。修正前 D116-7 実績（6/6 cycle・823-824MB flat・残留なし）は
本実行とは負荷形態が異なる（reload なし）ため、比較材料としてのみ引用。

## 10. TRIGGER 判定

| 判定表の条件 | 本実測 | 判定 |
|---|---|---|
| Memory plateau / reclaim 正常 | **線形増加 +1,088 MB/min・plateau なし** | — |
| Failure injection 正常 | 未実施（条件付き） | — |
| Deferred backlog drain | 未実施（条件付き） | — |
| Restart cycle 正常 | 未実施（条件付き） | — |
| UAF / crash / hang | なし（exit 0・Responding=True 継続） | — |
| **Memory 線形増加** | **該当** | **即 TRIGGERED** |
| recovery 無限 loop / duplicate consumption | 観測対象外（OP-2-B/C 未実施） | — |
| diagnostic chain 断絶 | shutdown drain 観測行は出力された（chain は生存） | TRIGGERED ではない |
| failure が発生しない | crash/異常テレメトリなし（10 publications 全 gen/worldId 連番正常） | negative observation として記録 |

**重要な観測事実（原因断定はしない — 起票 work item の入力として記録のみ）**:
1. memory 増加は publication 数（10）より **generation 処理数（59）に比例**（≈122 MB/gen）。
   publish されなかった generation（supersede/obsolete 経路）でも滞留が発生していることを示唆。
2. shutdown 時に routerPendingRetire=1 が残存（oldestAgeMs=361325 = 走行期間全体）—
   drain timeout → safe tryReclaim 経路で終了（"observation only" マーク付き）。
3. 現行 HEAD（HW-1 配線済み）でも修正前同規模の滞留 → HW-1 が publish 経路の滞留を解消しても、
   別の滞留経路（publish されない generation の構築物）が存続している可能性。
   ※ 以上は相関の記録であり、原因確定は次 work item の監査で行う。

## TRIGGERED work item 起票

```text
work item ID   : （ユーザー付番待ち — 仮称）D162 候補「rebuild generation 系メモリ滞留の
                 原因監査（診断ビルドによる counter 直接観測）」
性格           : TRIGGERED（D116-OP-2 OP-2-A 実測に基づく新規 work item — CR-α / D159 frozen /
                 Phase-II とは無関係の独立項目）
入力 evidence  : evidence/D116_OP2_memory_soak.csv（209 サンプル）/ D116_OP2_soak.log /
                 D116_OP2_swapper.log / D116_OP2_sampler.log / 本報告
推奨初手       : 診断ビルド（CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON）で pendingRetireCount /
                 quarantineResident / reclaimCount / liveCount の時系列を取得し、
                 generation 単位の確保/解放を直接観測（D117 推奨事項の踏襲）
境界           : 原因調査は本 work item で実施 — 本報告では修正・原因断定を行わない
```

## 生成物

```text
evidence/D116_OP2_memory_soak.csv   : メモリ時系列（2s 間隔・timestamp_ms,pid,private_mb,workingset_mb,responding）
evidence/D116_OP2_soak.log          : アプリ診断ログ全文（4,942 行）
evidence/D116_OP2_swapper.log       : IR swapper 周回記録（65 swaps・A/B/C 循環）
evidence/D116_OP2_sampler.log       : sampler 実行記録
evidence/D116_OP2_active.wav        : soak 中の active IR（swapper が A/B/C を原子置換）
evidence/D116_OP2_corrupt.wav       : OP-2-B 用に作成した異常 IR（600 byte truncate・未使用・保存）
doc/work88/D116_OP2_OPERATIONAL_VALIDATION.md : 本報告
```

## Next

```text
D116-OP-2 = TRIGGERED（OP-2-A）→ 停止
  ├─ OP-2-B / OP-2-C / OP-2-D: 未実施（PASS 条件付き工程のため）
  ├─ 新規 work item 起票: 上記仮称 D162 候補（ユーザー付番・指示待ち）
  └─ 原因調査・修正は起票 work item で実施（本工程では行わない）
```
