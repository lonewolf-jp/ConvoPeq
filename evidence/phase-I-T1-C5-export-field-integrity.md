# Phase I-T1-C5 — Export Field Integrity Verification

> **Verdict: PASS**
> 目的: C2〜C4 で確定した measurement state が Export 層で欠落・重複・再計算・意味変更されず、field contract として 1:1 に保存されることを証明する。
> コード変更なし。C6 以降へは進まない。
> 一次根拠: 現行 `src/`（`ConvoPeq.md` 2026-08-21 スナップショットと照合、差分は R3 切断箇所のみ）。

## 1. C5-1 — Export API / DTO の全列挙

### 1.1 Export 関数と caller

| 項目 | 内容 |
|---|---|
| Export 関数 | `AudioEngine::emitEvidenceTickNonRt(bool force)`（`AudioEngine.Commit.cpp:666`） |
| T1 export block | 同関数内 `Commit.cpp:709-770` — `world_retirement_telemetry.json`（tmp + atomic rename） |
| caller 全列挙 | `Timer.cpp:466/788/1558/1574/1679`（timerCallback 系・MessageThread）、`ReleaseResources.cpp:350/517/610`、`Commit.cpp:171/443/635` — **全て NonRT**（audio callback caller 0件） |

### 1.2 関連 API / DTO

- getter: `acquireObserved()` / `releaseObserved()` / `observedOutstandingEstimate()` / `observedOutstandingMax()` / `windowTag()` / `lastClosedSnapshot()`
- reference 側: `referenceAcquireCount()` / `referenceReleaseCount()` / `referenceMax()` / `referenceOutstanding()`
- DTO: `MeasurementSnapshot`（`Telemetry.h:41-56`、trivially copyable 14 members、`std::atomic<MeasurementSnapshot>` で immutable publish）
- writer: sampler のみ（`beginWindow/sampleWindow/closeWindow`、C3/C4 で一意性確定済み）
- `addReleaseObserved()` / `worldReclaimCount()` / `lastSampledWorldReclaimCount_`: **export block には出現しない**（rg `worldReclaimCount` in Commit.cpp/ISREvidenceExporter.* = 0 hits）

### 1.3 field 数の確定（推測ではなくソースから）

シリアライズ済み JSON field 総数 = **24**（top-level 6 + measurementWindow 13 + referenceObserver 5）。

このうち **15-field measurement contract** は:

- top-level 6（acquireObserved / releaseObserved / observedOutstandingEstimate / observedOutstandingMax / windowTag / windowTagName）
- window 計測 9（windowId / startAcquire / startRelease / endAcquire / endRelease / finalEstimate / windowMax / windowStartTimestampUs / windowEndTimestampUs）

残り 9 は診断補助: sampling quality 4（sampleCount / maxSamplingGapUs / missedTickCount / counterWrapped・D89.2/D91 基準 9）+ referenceObserver 比較 5（D98・E_w は「診断のみ」とコード明記）。**全 24 field を以下で mapping し、省略なし。**

## 2. C5-2 — field mapping table（全 24 行・★=15-field contract）

| # | Export field | authoritative source | transformation | writer | reader/exporter | RT/NonRT |
|---|---|---|---|---|---|---|
| ★1 | `acquireObserved` | `acquireObserved_` | none（getter 直読） | `onAcquireObserved()`（commit path のみ） | `emitEvidenceTickNonRt` | writer NonRT / exporter NonRT |
| ★2 | `releaseObserved` | `releaseObserved_` | none（getter 直読） | sampler `addReleaseObserved(delta)` のみ | 同上 | 同上 |
| ★3 | `observedOutstandingEstimate` | A/R 两 counter | signedWide(A)-signedWide(R)（D82 同式） | 計算値（読み取り時） | 同上 | NonRT |
| ★4 | `observedOutstandingMax` | `observedOutstandingMax_` | none | sampler `updateObservedOutstandingMax` のみ | 同上 | NonRT |
| ★5 | `windowTag` | `windowTag_` | int cast | sampler `setWindowTag` のみ | 同上 | NonRT |
| ★6 | `windowTagName` | 同上 tag | `windowTagName(tag)` 文字列化 | — | 同上 | NonRT |
| ★7 | `measurementWindow.windowId` | snapshot.windowId | none | sampler `beginWindow`（++・monotonic） | 同上 | NonRT |
| ★8 | `.startAcquire` | snapshot.startAcquire | none | sampler `beginWindow` baseline | 同上 | NonRT |
| ★9 | `.startRelease` | snapshot.startRelease | none | 同上 | 同上 | NonRT |
| ★10 | `.endAcquire` | snapshot.endAcquire | none | sampler `closeWindow` | 同上 | NonRT |
| ★11 | `.endRelease` | snapshot.endRelease | none | 同上 | 同上 | NonRT |
| ★12 | `.finalEstimate` | snapshot.finalEstimate | none | sampler `closeWindow`（A1-R1） | 同上 | NonRT |
| ★13 | `.windowMax` | snapshot.windowMax | none（bounded sampled max・D91 基準 8） | sampler CAS max | 同上 | NonRT |
| ★14 | `.windowStartTimestampUs` | snapshot 同名 | none | sampler `beginWindow` | 同上 | NonRT |
| ★15 | `.windowEndTimestampUs` | snapshot 同名 | none | sampler `closeWindow` | 同上 | NonRT |
| 16 | `.sampleCount` | snapshot 同名 | none | begin=1 / fetchAdd per tick | 同上 | NonRT |
| 17 | `.maxSamplingGapUs` | snapshot 同名 | none | sampler `updateMaxSamplingGap` | 同上 | NonRT |
| 18 | `.missedTickCount` | snapshot 同名 | none | sampler（gap > 2×tick で ++） | 同上 | NonRT |
| 19 | `.counterWrapped` | snapshot 同名 | `(a1<a0)\|\|(r1<r0)` 診断 | sampler `closeWindow` | 同上 | NonRT |
| 20 | `referenceObserver.referenceAcquireCount` | `referenceAcquireCount_` | none | `onAcquire()`（commit path） | 同上 | NonRT |
| 21 | `.referenceReleaseCount` | `referenceReleaseCount_` | none | `onRelease()`（drain path） | 同上 | NonRT |
| 22 | `.referenceOutstanding` | 上記 2 差分 | int64 減算 | 計算値 | 同上 | NonRT |
| 23 | `.referenceMax` | `referenceMax_` | none（T_w） | observer running max | 同上 | NonRT |
| 24 | `.Ew` | refMax - maxObserved | 減算（**診断のみ**と明記） | 計算値 | 同上 | NonRT |

source は全て一意。exporter が別 counter を参照・measurement state を再計算して置き換える箇所はなし（#3/#22/#24 は同一 authority からの表示用派生で、命名も derived と明示）。

## 3. C5-3 — field conservation

核心経路の保存検証:

```
destruction → worldReclaimCount(+N) → sampler delta(N) → releaseObserved_(+N) → export "releaseObserved"（getter 直読）
```

- **Export が `releaseObserved` を destruction count から再計算していない**: export block 内 `worldReclaimCount` 出現 0件（rg 実測）。getter 直読のみ。
- **`lastSampledWorldReclaimCount_` を current measurement と誤認していない**: 同カウンタは export されない（sampler 内部 cursor のみ・C3 §1）。JSON に存在しない。
- `observedOutstandingEstimate` は export 時点の live 値（sampler tick 値と同一式 D82）。sampler の windowMax/finalEstimate とは別フィールドで意味が混在しない。
- snapshot は `closeWindow` 内で全 member を構築後 **単一 `publishAtomic(snapshot_, snap)` で immutable publish** → export 側 `consumeAtomic` 1回読み。field 間世代不整合は構造上不可能。

各 field について `production measurement state == exported representation` が成立。

## 4. C5-4 — window / baseline integrity（Export 側から独立確認）

| 懸念 | 実装 | 判定 |
|---|---|---|
| 前 window observed の混入 | `beginWindow` が baseline（a0/r0）・windowMax（firstEstimate）・stats を全て再 publish | なし ✅ |
| baseline が current に置換される | startAcquire/startRelease は begin 時固定、closeWindow まで不変 | なし ✅ |
| max の不当継承 | windowMax は firstEstimate で初期化（継承しない・D91 監視項目 1）、finalEstimate を含めて閉じる（監視項目 2） | 正しい ✅ |
| tag と measurement 値の世代不一致 | snapshot は tag を含まない。top-level tag = live 分類、measurementWindow = last Closed（D91 基準 10 immutable read）— windowId で対比する設計 | 設計どおり ✅ |
| request 損失 | End→Idle CAS 失敗時は次 tick beginWindow（監視項目 3） | 保証 ✅ |
| counterWrapped | a1<a0/r1<r0 の診断フラグのみ（trigger にしない・D91 基準 9） | 診断限定 ✅ |

## 5. C5-5 — Export の RT isolation

- `emitEvidenceTickNonRt` の caller 11箇所は全て timerCallback / ReleaseResources / commit path = **NonRT のみ**（§1.1）。audio callback からの到達経路 0件。
- 関数内は `std::ofstream` / `std::filesystem` を使用（RT 禁止操作）→ 名称と実装が一致した NonRT 専用 boundary。
- 読み取り対象も全て NonRT writer の counter / immutable snapshot（C4 §1-2 と整合）。

## 6. C5-6 — duplicate / omission census

| 判定基準 | 結果 |
|---|---|
| writer 複数 authority | なし（各 counter の writer は C2-C4 で一意確認済み） |
| exporter が複数 source を選択 | なし（field ごとに 1 source） |
| DTO に存在しない field | なし（snapshot 14 member 中 13 を export、`valid` は null/object gate として機能 — 意図的） |
| source にあるが export されない | `lastSampledWorldReclaimCount_` / `worldReclaimCount` / `measurementState_` は内部状態として非 export（設計どおり・誤認防止） |
| 同一 source の二重表現 | なし（releaseObserved 等は 1回ずつ。derived 値 #3/#22/#24 は別名で明示） |
| 意味変更する再計算 | なし（derived は D82 同式 or 明示的減算、comment で「診断のみ」） |

## 7. C5-7 — PASS criteria

| Gate | 条件 | 判定 |
|---|---|---|
| C5-1 | Export API / DTO / caller 全列挙 | ✅ §1 |
| C5-2 | field mapping 一意（24 field 全行・15 contract 確定） | ✅ §2 |
| C5-3 | conservation 成立 | ✅ §3 |
| C5-4 | window/baseline/tag integrity | ✅ §4 |
| C5-5 | Export NonRT-only | ✅ §5 |
| C5-6 | omission / duplication / alternate authority = 0 | ✅ §6 |
| C5-7 | コード変更 = 0 | ✅ |

## 8. Tool Coverage

| 系統 | ツール | 結果 |
|---|---|---|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | export block 全行読取・caller census・worldReclaimCount 非出現確認・field census |
| MCP#1 | serena | `find_symbol(MeasurementSnapshot)` DTO 定義確定（Telemetry.h:41-56, trivially copyable） |
| MCP#2 | ccc | search 実行（未 init のため rg/semble 補完） |
| CLI#1 | graphify 0.9.48 | exe 存在確認 |
| CLI#2 | semble 0.5.5 | `world_retirement_telemetry json export fields` 検索実行 |
| MCP#3 | AiDex | C2-C4 で `releaseObserved` 17 hits 等確定済み（本 Gate の reader census と一致） |
| 文献 | crossbeam-epoch / rigtorp / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9件 200 OK（Vyukov SSL失効→rigtorp代替明記） |

## 9. 記録事項

- `Ew`（#24）は reference running max と lifetime accumulated max の cross-generation 比較であり、コード自身が「診断のみ（M の安全側根拠にしない）」と明記 — conservation 違反ではなく diagnostic-only として確認。
- テストハーネス delta-transfer 欠落は **C6 test harness gap として保留継続**。

## 10. 判定

```
T1-C5 = PASS
```

**次 Gate（未着手・停止中）:** T1-C6 Test coverage/gap → C7/C8 measurement readiness。A_max candidate は C1〜C6 完了後、R は未決定。

---

*Evidence generated: Phase I-T1-C5 — no code change. C6 以降へは進まない。*
