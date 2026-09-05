# D162-2-I1 — I1-0 Environment Check + A-1 Pilot Run Record

```text
Date:     2026-09-04
Type:     I1-0 = environment check（read-only）/ A-1 = Profile A pilot ×1（production 変更 0 / test 0 / build 0）
Baseline: ConvoPeq.md Generated 2026-09-04 20:28:50（H0/I0 と同一・開始時再確認済み）
Binary:   build-diag RelWithDebInfo ConvoPeq.exe（Sep 4 21:16 build = G4-1 と同一。ソース変更なしのため再ビルド不要）
Output:   evidence/D162-2-I1/（I0 契約どおり新規分離）
判定:     I1-0 PASS / A-1 PASS（全契約項目合格）→ **ユーザーの 3 択判定（継続 / STOP / 測定契約修正）待ち**
```

---

## 1. I1-0 Environment Check（全 ✓）

| 項目 | 結果 |
| --- | --- |
| Baseline | `Generated: 2026-09-04 20:28:50` 再確認 ✓ |
| working tree | git diff = G1/G2 既存 6 files（+216/−71）のみ — **I1 由来の production/test/CMake 変更 0** ✓ |
| binary | `build-diag/.../RelWithDebInfo/ConvoPeq.exe` mtime **Sep 4 21:16 = G4-1 build と同一** ✓ |
| CrashDumps 開始時状態 | `DUMP_BASELINE.txt` に記録（最新 = ConvoPeq.exe.12068.dmp 21:17 = G4 CTest era の HeadlessAudioPathVerification 旧 build-icx class・rpcs3 は外部アプリ）。I1 帰属規則: mtime > I1 開始の新規 ConvoPeq dump のみ I1 起因 |
| 出力分離 | `evidence/D162-2-I1/`（log・script・evidence json 群・本書） |
| exit capture | PowerShell `Start-Process -PassThru -Wait` + `$p.ExitCode`（旧 `cmd /c ... & echo %ERRORLEVEL%` 不使用） |
| device type（Profile D 準備） | `--cli-device-type` は実行時に `getAvailableDeviceTypes()` で解決し `[CLI_AUDIO_DEV_TYPES] available=...` を出力（MainWindow.cpp:430-437）。**存在しない type を仮定しない** — D 開始前に最初の 1 run の log から実在 type 名を取得する方式を採用 |

## 2. A-1 Pilot 実測（Profile A: plain startup → audio 15s → clean shutdown）

**実行**: `--cli-run --cli-log-file A1_pilot.log --cli-exit-ms 15000`（IR なし・plain）
**結果: `EXITCODE=0x00000000`** / 新規 crash dump **0** / log 3,401 行

### 2.1 Shutdown trace zone

```text
:3295 [CLI] Auto-exit flush: shutting down
:3297 [D123] SHUTDOWN_BEGIN
:3399 [D123] SHUTDOWN: mainWindow.reset() completed
:3401 [D123] LOGGER_DETACH / SHUTDOWN_END
```
→ **zone = clean**（T0/T1/T2 不該当）

### 2.2 測定契約項目

| 項目 | 実測 | 合格 |
| --- | --- | --- |
| EXITCODE | **0x00000000** | ✓ |
| 新規 dump | 0 | ✓ |
| zone | clean | ✓ |
| lifecycle closure | 構築 1 個体（bootstrap placeholder gen=4）= V-D で destroy → `DSP_FOOTPRINT_RELEASED remaining=0`。1:1 閉包 | ✓ |
| generation 1:1 | destroyed gen: {4} のみ・重複 0・gap なし | ✓ |
| E-4 accounting | deferred 活動 0（plain run のため CREATE 0 = exits 0・自明的収支） | ✓ |
| EBR | pend 最終 0 / ovf 0（最終 MEM_SNAP: `Ret: pend=0 … ovf=0`） | ✓ |
| registered direct destroy | **0**（`[D162-2B_DESTROY]` 0 件） | ✓ |
| stale map | dtor `retireByHandle` **MISS 1 / HIT 0**（V-D が map erase 済み → no-op 正常） | ✓ |
| E-3 / INV-D162-8 | 0 件 | ✓ |
| Signature A / B / C | 0 / 0 / 0 | ✓ |
| residual | 0（remaining≠0 0 件） | ✓ |
| XRUN | **9 件 — 全て起動後 12 秒内（22:08:52-22:09:02）の startup transient**・Callback max 1.72ms / Interval ~8.3ms vs Expected 5.33ms（既知 jitter signature・全 baseline と同クラス）/ Pressure=0 全件 / **shutdown window 0 件**（最終 XRUN 22:09:02 < SHUTDOWN_BEGIN 22:09:05.056） | ✓（件数ではなく契約条件で判定） |
| memory | 最終 MEM_SNAP: DC live=1 / Priv=395MB / NUC live=0 / residual 0 | ✓ |

### 2.3 Profile A 特有の観測（契約への補足・修正不要）

1. **V-D が plain run の主体**: IR なしのため rebuild は発生せず、構築 DSP は bootstrap
   placeholder 1 個体のみ。これを V-D（`active-final`）が `retired=1 → EBR Success(epoch 10) →
   DESTROY → remaining=0` で閉じた — **plain shutdown こそ V-D の本来の主役経路**であることを
   実証（60-gen では G シリーズと同型）。
2. **S3 は不発**（deferred 活動 0 件のため `CLEAR_SHUTDOWN_DISPOSITION` 0）— plain profile
   では deferred slot が最初から空であり正当。S3 の観測は IR/burst 系（B/C）に委ねる。
3. **XRUN 9 件の rate について**: G4 60-gen（420s）と同数だが本 run は 15s のため単位時間
   rate は高い。全て run 冒頭 12s に集中 = **startup/device warmup transient** 分類。なお
   D116 時代の restart run で anomalies=0 だったのは `build\` Release（非 DIAG binary）に
   XRUN ログが存在しなかったため（診断差分であり挙動差分ではない）。
4. **evidence json 副産物**: `retire_timeline.json` / `shutdown_trace.json` /
   `world_lifecycle_audit.json` 等 8 ファイルが app 側出力として `evidence/D162-2-I1/evidence/`
   に自動生成（I1 の shutdown 契約観測と整合）。

## 3. 判定

**I1-0 PASS / A-1 PASS。** STOP 条件該当なし。測定契約の修正も不要
（plain profile で V-D 単独 closure・zone 分類・3 channel 測定が全て機能した）。

→ **ユーザーの 3 択判定（①継続 A-2〜A-8 → B → C → D / ②STOP / ③測定契約修正）を待つ。**
継続の場合の残 run: A ×7 → B ×6 → C ×6 → D ×6（Profile D は最初の 1 run の
`[CLI_AUDIO_DEV_TYPES]` 実測値で type 名を確定してから実施）。
