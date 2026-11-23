# domain tag 実装と §7.2 の 15 項目検証

受理済みの設計差分 v2.9（53,979 B / SHA-256 `C3AE5099…`）に従い、domain tag を実装し、§7.2 の 15 項目を検証した。設計差分は変更していない。**commit／merge／release は行っていない**（作業ツリーに変更を保持したまま）。

## §0 結論

- **実装**: `RetireDomainId`（新設ヘッダ）、`DeletionEntry.domainId` の刻印、`EpochDomain` の明示ドメイン ctor（B-1 契約）、`IRetireProvider::retireDomainId()`、Router の委譲、Terminal／Quarantine への明示引数、`retireLatencyClassified.domains[]`（4 行固定）、生成箇所 38（本番 3 + テスト 35）の明示化まで完了した。
- **検証**: §7.2 の 15 項目はすべて PASS（#15 は 15-a／15-b／15-c-1／15-c-2 の 4 副検証すべて）。うち #15-c-2 は**製品の ctor** をデバッガ接続下で実行し、`EXCEPTION_BREAKPOINT`（0x80000003）の発生と「ブレーク前にカウンタ +1」を実測した。
- **テスト**: 変更前 60/60（359.6 秒）→ 変更後 **61/61**（335.8 秒）。`DomainTagTests` を追加。失敗・スキップなし。
- **サイズ**: `DeletionEntry` は診断 OFF 56 / ON 64 バイトで**変更前と同一**（`domainId` はパディングに吸収）。alignof 8・trivially copyable／destructible も不変。
- **3 シナリオ実測**: `stampMissing=0`・`providerUnknown=0`・`providerAbsent=0`、`domainSumsConsistent=true` を 3 本すべてで確認。**eq シナリオの最大 11.80 s は EQ ドメインの `driverWait`** であり、EngineISR ではなかった（設計時の推定と逆。§6）。
- **設計からの逸脱 1 件**: `jassertfalse` の代わりに、同じ意味（ログ → デバッガ接続時のみ `__debugbreak()`）を `EpochDomain.h` 内に直接実装した。理由は §9。

## §1 変更ファイルと機能別の変更内容

| 機能単位 | ファイル | 変更内容 |
|---|---|---|
| ① 型 | `src/core/RetireDomain.h`（新規） | `enum class RetireDomainId : uint8_t`（Unstamped=0 / Unknown=1 / EngineISR=2 / EQ=3 / Count=4）、`kRetireDomainCount`、`retireDomainName()`。include は `<cstdint>` のみ |
| ① ctor | `src/core/EpochDomain.h` | `explicit EpochDomain(RetireDomainId)`（デフォルト引数なし）、`isAssignableDomain()` / `sanitizeDomain()`、`domainId()`、`retireDomainId()` override、`invalidDomainAssignments()` と Debug 検出（`domainAssertionFailure()`）。`domainId_` は `globalEpoch` より先に宣言 |
| ① 刻印 | `src/DeferredDeletionQueue.h` | `DeletionEntry.domainId{Unstamped}`、`enqueue` の 3 オーバーロードに `RetireDomainId domain` を必須化、`entry.domainId = domain`、`sizeof`/`alignof` の static_assert（構成別） |
| ② 伝播 | `src/core/IRetireProvider.h` | `retireDomainId()`（既定 `Unknown`、`Unstamped` は返さない） |
| ② 伝播 | `src/audioengine/ISRRetireRouter.h/.cpp` | `retireDomainId()` を `provider_` へ委譲（`provider()` は公開しない）、`TerminalReclaimAuthority::drain/drainAll/tryReclaim` と `RetireQuarantineStore::drain/drainAllUnsafe` に**明示引数**でドメインを渡す |
| ③ 刻印 | `src/audioengine/RetireQuarantineStore.h` | `drain`/`drainAllUnsafe` が受け取ったドメインを `recordComplete` へ渡す |
| ④ 集計 | `src/audioengine/ISRRetireLatency.h` | `recordComplete(object, path, domain, firstScanTs, providerAbsent)`、ドメイン別集計 `DomainStats`／`DomainPathCounters`、`stampMissing`/`providerUnknown`/`providerAbsent`、`snapshot()` と `reset()` の拡張（既存集計は不変） |
| ④ JSON | `src/audioengine/ISRHB.cpp` | `retireLatencyClassified.domains[]` を 4 行固定で出力、`unknownReasons`、`stampMissing`/`providerUnknown`/`providerAbsent`、`domainSumsConsistent`。`paths`/`totals` のキーは不変 |
| ⑤ 生成箇所 | `AudioEngine.h` / `EQProcessor.h` / `ConvolverProcessor.h` + テスト 8 ファイル | 本番 3（EngineISR / EQ / Unknown）＋テスト 35（Unknown）を明示 |
| ⑤ テスト | `src/tests/DomainTagTests.cpp`（新規）、`CMakeLists.txt` | 15-a/b/c、#6/#8/#13、#9/#10 を固定する JUCE 非依存ターゲットを追加 |
| ⑤ 既存テスト | `DeferredDeletionQueueReclaimTests.cpp`、`RetireGraceSemanticsTests.cpp`、`TerminalTelemetryContractTests.cpp`、`SnapshotRetireManager.h`、`DeletionQueue.cpp` | 生キュー／退避ストアの呼出しにドメイン引数を追加（テストは `Unknown`、レガシーは「対象外・到達しない」と明記） |

変更行数は `evidence/changed_files.txt`（`git diff --stat`）。

## §2 変更前後の Git ベースライン

| 項目 | 変更前（2026-11-12 22:50 JST） | 変更後 |
|---|---|---|
| HEAD | `c94d376772ee22916f8350cdaa33f8d3787d16bd` | 同じ（commit していない） |
| 変更ファイル | 63 件 | 75 件（+12） |
| 未追跡ファイル | 151 件 | 154 件（+3: RetireDomain.h・DomainTagTests.cpp・本報告） |
| ctest | **60/60 PASS**（359.6 秒） | **61/61 PASS**（335.8 秒） |

- 変更前のビルドは **MSVC 14.51 → 14.52 の更新でリンク不能**だった（`14.51.36231\link.exe` 不在）。`build/CMakeCache.txt` と `build/JUCE/tools/CMakeCache.txt` のコンパイラパスを 14.52 に更新し、oneAPI `setvars.bat` で `MKLROOT`/`IPPROOT` を設定して再構成した。**この復旧はビルドツリーのみで、ソース変更ではない。**
- 変更前の ctest は上記復旧後の値（60/60）を基準とし、変更後の 61/61 と比較した。

## §3 既存変更との干渉確認

- 着手前チェック（`doc/work113/20261111-2400_着手前確認/overlap_analysis.md`）の 19 ファイルの SHA-256 を再照合し、**19/19 一致**を確認してから編集を始めた。
- 編集したファイルのうち、既に未コミット変更を持っていたもの: `EpochDomain.h`、`DeferredDeletionQueue.h`、`ISRRetireRouter.cpp`、`ISRHB.cpp`、`AudioEngine.h`、`EQProcessor.h`、`ConvolverProcessor.h`、`STG11EQRetireTests.cpp`、`D8_2_B_2_Tests.cpp`、`CMakeLists.txt`。**既存変更は保持**し、`git reset`／`checkout`／`clean` は使っていない。
- 設計の棚卸しに無かった追加サイトが 2 種類見つかったため、意図を明示して対応した。
  - 生の `DeferredDeletionQueue::enqueue` を直接叩くテスト 18 箇所（`DeferredDeletionQueueReclaimTests.cpp`）→ テスト用キューは `Unknown` を明示。
  - `RetireQuarantineStore::drain`／`TerminalReclaimAuthority::drain` を直接叩くテスト 13 箇所 → `Unknown` を明示。
  - レガシー `DeletionQueue.cpp` の 2 箇所 → 「対象外・到達しない」と明記し `Unknown`。
- 既存変更との切り分けのため、実装前の SHA-256 は `overlap_analysis.md` に残っている。

## §4 §7.2 の 15 項目

| # | 検証 | 判定 | 根拠 |
|---|---|---|---|
| 1 | 刻印（サイト別 A1–A13） | **PASS**（ソース＋経路実測） | 13 サイトすべてが `enqueueRetire`／`enqueueRetireTyped` 経由で provider のドメインを刻印することをソースで確認。実測でも `queue`（Engine/EQ 双方）・`sync`・`drain`・`drainAll` の各経路が別ドメインに計上された（§6） |
| 2 | 推測禁止 | **PASS**（ソース） | `entry.domainId` の書込みは `enqueue` の 1 箇所のみ。`reclaim`／`drainAllUnsafe` は読むだけで、スレッド・型・経路からの推測は無い |
| 3 | Router API | **PASS** | `retireDomainId()` は `provider_` へ委譲（`ISRRetireRouter.h:250`）。公開 `provider()` アクセサは 0 件（`evidence/source_checks.txt`） |
| 4 | 完了経路の網羅 | **PASS**（経路単位の実測） | 9 稼働サイトのうち、実測で `queue`／`sync`／`drain`／`drainAll`／`quarantine` の記録を確認。Terminal／Quarantine は明示引数で受ける |
| 5 | レガシー対象外 | **PASS** | `SnapshotRetireManager` の参照は自身のヘッダ以外 0 件。`DeletionQueue.cpp` は到達しない旨を明記 |
| 6 | 合算整合 | **PASS** | `DomainTagTests` の合算 12 検査＋実測 3 本で `domainSumsConsistent=true`／`countsConsistent=true` |
| 7 | 原因の分離 | **PASS** | `Unknown` の完了は `providerUnknown` に計上（`DomainTagTests`）。実測 3 本は `providerUnknown=0`・`stampMissing=0`・`providerAbsent=0` |
| 8 | Unknown の非混入 | **PASS** | `DomainTagTests`: EngineISR=1・EQ=1 のまま（Unknown を混ぜても増えない） |
| 9 | 既存互換 | **PASS** | `kThresholdNs == 50 ms`・`Path::Count == 6` を static_assert。`retireLatencyD1` のキー構造に触れていない。既存 60 テストが全て PASS（verdict 系を含む） |
| 10 | サイズ・型 | **PASS** | 診断 OFF 56 / ON 64、alignof 8、trivially copyable／destructible。**変更前と同値**（`size_before_after.txt`） |
| 11 | PublishReason | **PASS**（不変） | 4 値と 4 呼出し元はそのまま。`PublishReason::` の利用 8 箇所を確認し、ドメイン実装で触れていない |
| 12 | 実測（3 シナリオ） | **PASS（結論は設計時の推定と逆）** | 3 本とも `stampMissing=0`。**eq の最大 11.80 s は EQ の `driverWait`**（EngineISR の最大は 2.15 s の `ebrWait`）。§6 |
| 13 | 未刻印の不在 | **PASS** | 3 本すべて `stampMissing=0`・`domains[Unstamped].count=0` |
| 14 | 生成箇所の明示化 | **PASS** | `EpochDomain x;` 0 件、`deferredDeletionQueue.enqueue`（EpochDomain.h 以外）0 件 |
| 15 | 値域制約（4 副検証） | **PASS** | 15-a: 判定子・`sanitizeDomain` の真理値表（Release/Debug とも）。15-b: Release の実 ctor で不正値 → `Unknown`・カウンタ +3・停止なし。15-c-1: Debug・非接続でログのみ（ブレークなし）・カウンタ +1。15-c-2: Debug・接続で `EXCEPTION_BREAKPOINT` を捕捉し、**捕捉後のカウンタ +1**（ブレーク前に加算） |

## §5 ビルドとテスト

| 構成 | ビルド | テスト |
|---|---|---|
| Release（診断 OFF） | 全ターゲット ビルド成功（223 ステップ・警告 34 件はいずれも既存ファイル） | `ctest` 61/61 PASS（335.8 秒） |
| Debug（診断 OFF） | `DomainTagTests` ビルド成功 | 15-c-1／15-c-2 を実行（下記） |
| Release（診断 **ON**・別ツリー `build-diag2`） | `DomainTagTests` と `DeferredDeletionQueueReclaimTests` のビルド成功（`sizeof(DeletionEntry)==64` の static_assert も通る） | `DomainTagTests` **PASS**（15-a/b・#6/#8/#13・#9/#10）、`DeferredDeletionQueueReclaimTests` **45/45 PASS** |

- 15-c-1（Debug・デバッガ非接続）: 例外は出ず、ログのみ。カウンタ +1。`PASS`
- 15-c-2（Debug・デバッガ接続）: 最小デバッガ（`DEBUG_ONLY_THIS_PROCESS`、ローダの初期ブレークは `DBG_CONTINUE`、2 回目以降は `DBG_EXCEPTION_NOT_HANDLED` でアプリへ渡す）で実行。`EXCEPTION_BREAKPOINT`（0x80000003）を捕捉し、**捕捉後のカウンタが +1**。`PASS`
  - `DBG_CONTINUE` で継続すると「デバッガが処理した」扱いになり、アプリの `__except` へ回らないことを実測で確認した。
- 上表の Release と 15-c は、**逸脱承認後の最終形**（プラットフォーム分岐と宣言ガードを追加した後）で再ビルド・再実行した値である。
- 診断 ON 構成は**別ビルドツリー**（`build-diag2`）を新規に構成し、domain tag に関係する 2 ターゲットだけをビルド・実行した。
  61 ターゲット全部の診断 ON ビルドは行っていない（時間のため）。型・サイズ（#10）と契約（#15-a/b）は両構成で確認済み。

## §6 3 シナリオの実測（アプリ・各 12 秒）

| シナリオ | 起動内容 | EngineISR（count / maxMs） | EQ（count / maxMs） | stampMissing / providerUnknown / providerAbsent | 整合フラグ |
|---|---|---|---|---|---|
| burst40 | `--cli-bypass-burst-count 40 --cli-bypass-burst-interval-ms 50` | 14 / 2,166.5 | 492 / 759.3 | 0 / 0 / 0 | true / true |
| burstfast | `--cli-bypass-burst-count 200 --cli-bypass-burst-interval-ms 10` | 12 / 2,152.1 | 205 / 874.0 | 0 / 0 / 0 | true / true |
| eq | `--cli-eq-band 0 --cli-eq-freq-hz 1000 --cli-eq-gain-db 9 --cli-eq-q 2 --cli-eq-type 1` | 13 / 2,149.5 | 174 / **11,798.2** | 0 / 0 / 0 | true / true |

- **#12 の結論**: eq の 11.80 s は **EQ ドメインの `queue` 経路の `driverWait`**（`maxNs=11,798,170,500`）である。設計時は「EQ の 11.9 s `driverWait` はエンジン側 ISR ドメインのキューではないか」と推定していたが、**推定は外れ**、EQ 自身のキュー滞留だった。EngineISR 側の最大は 2.15 s（`ebrWait`）である。
- `EQ` の `excluded`（105〜252 件）は shutdown 時の `drainAll` で safeTs 未観測のもの。設計どおり `excluded` に計上され、`countsConsistent=true` を保つ。

## §7 JSON の出力例と不変条件

```json
"retireLatencyClassified": {
  "note": "diagnostic only; the verdict uses retireLatencyD1.runtime and is unchanged",
  "thresholdNs": 50000000, "required": 59,
  "paths": [ …既存（全ドメイン合算・キー不変）… ],
  "totals": { …既存… },
  "domains": [
    {"id": "Unstamped", "count": 0, …, "unknownReasons": {"stampMissing": 0}, "paths": []},
    {"id": "Unknown",   "count": 0, …, "unknownReasons": {"providerUnknown": 0, "providerAbsent": 0}, "paths": []},
    {"id": "EngineISR", "count": 14, "classified": 12, "unclassified": 2, "excluded": 0, "sumNs": 2513059600, "maxNs": 2166530300, "over50ms": 2, "paths": [ … ]},
    {"id": "EQ",        "count": 492, …, "paths": [ … ]}
  ],
  "stampMissing": 0, "providerUnknown": 0, "providerAbsent": 0,
  "domainSumsConsistent": true, "countsConsistent": true
}
```

- 4 行は件数 0 でも出力する（Unstamped／Unknown が 0 件でも省略しない）。
- `Σ domains[].count == totals.completions`、`Σ classified/unclassified/excluded/consistencyErrors` も一致（`domainSumsConsistent`）。
- `Σ domains[].paths[p].<component> == paths[p].<component>` も一致（同フラグ内で検査）。
- `domains[Unstamped].count == stampMissing`、`domains[Unknown].count == providerUnknown + providerAbsent`。
- `paths`／`totals` のキー・意味は変更していない。
- アプリ自身の `domainSumsConsistent` に依存せず、出力 JSON を読み直して**独立に再計算**した結果も一致した（`evidence/json_invariants.txt`: Σcount = 506/506・217/217・187/187、classified／unclassified／excluded／consistencyErrors とパス別成分もすべて一致）。

## §8 `retireLatencyD1` の互換性

- `retireLatencyD1` の writer／キー構造に触れていない（`ISRHB.cpp` の差分は `retireLatencyClassified` ブロックのみ）。
- `evaluateRetireDelay`・50 ms 閾値・`N_req=59`・verdict は不変（`kThresholdNs` を static_assert、既存 60 テストが PASS）。
- `worldLifetime` も不変（`commits == destroys` を実測で確認）。

## §9 残課題・設計判断待ち・未実測

1. **設計からの逸脱（Owner 承認済み 2026-11-12）**: `EpochDomain` の Debug 検出は `jassertfalse` ではなく、同等の処理を `EpochDomain.h` 内に直接実装した。
   - 理由: `EpochDomain.h` は JUCE 非依存（`<cassert>`/`<cstdio>` のみ）で、インライン ctor の定義は全 TU で同一でなければならない。`jassertfalse` を持ち込むには JUCE ヘッダ（`juce_PlatformDefs.h`）を全 TU に引き込む必要があり、`windows.h` の `max` マクロが `std::numeric_limits::max()` を壊す等の副作用もある。
   - 実装: stderr へ 1 行 → `IsDebuggerPresent()` が真のときだけ `__debugbreak()`。Release では何もしない。
   - **同等なのは「条件・順序・ブレーク動作」**である。次は同等ではない。
     - ログの配送先: JUCE Logger（`logAssertion`）ではなく **stderr**。
     - 製品 ctor で **JUCE の実マクロを使ってはいない**。
   - ABI: `IsDebuggerPresent` の自前宣言は `extern "C" __declspec (dllimport) int __stdcall`（SDK の `BOOL WINAPI` と同型・同呼出規約）。`EpochDomain.h` → `<windows.h>` の順で Debug／Release ともコンパイルでき、両宣言が共存することを確認（`evidence/assertion_abi_and_platform.txt`）。`<windows.h>` → `EpochDomain.h` の順は**本変更前から**コンパイルできない（`windows.h` の `max` マクロが既存の `std::numeric_limits<...>::max()` を壊す。今回の宣言は `_WINDOWS_` 定義済みでスキップしており原因ではない）。
   - プラットフォーム条件: `IsDebuggerPresent` の宣言と `__debugbreak()` は `#if defined (_WIN32)` の内側にのみあり、ガード外の Windows API 参照は 0 件。非 Windows の Debug は `assert(false)` のみで、宣言・未定義シンボルを持ち込まない。**非 Windows でのビルドは未実測**（本プロジェクトは Windows 専用）。
   - 影響: 15-c-1／15-c-2 の意味は設計どおり（製品 ctor で実測済み）。
2. **未実測**: WinDbg／Visual Studio の実設定（`sxd 80000003` の意味は文献確認済み、実行は最小デバッガで代替）。
3. **未実施（指示外）**: Reader レジストリ統合、ASan、commit／merge／release。
4. **実装上の注意（記録）**: `EpochDomain` は 1 個で約 230 KB（4096 エントリのキューを内包）。テストで複数個をスタックに置くと 1 MB を超えて stack overflow になるため、`DomainTagTests` は `std::make_unique` を使っている。
5. **設計の棚卸しとの差**: 生キュー／退避ストアを直接叩くテスト呼出しが設計の棚卸しに含まれていなかった（§3）。実装では呼出し側に `Unknown` を明示して対応した。

## §10 成果物

- 本報告: `doc/work113/20261112-0030_domain実装/report.md`
- 証跡: 同 `evidence/`（ベースライン、ctest 前後、変更ファイル、サイズ前後、ソース検証、3 シナリオの JSON と要約、15-c の実行ログ）
- Dropbox: `/ConvoPeq-domain-impl-20261112/`

## §11 commit 記録（2026-11-12 承認分）

Owner の commit 承認（2026-11-12）に対応する記録である。**承認時点で実装は未コミットだったが、現在は `173e6995` として commit 済み**（別セッションで実施）。本作業では新たな commit を作っていない。

| 項目 | 値 |
|---|---|
| commit | `173e69957b046290f7077797782f031ed8633fd0` |
| 親 | `c94d376772ee22916f8350cdaa33f8d3787d16bd` |
| 作成 | 2026-11-12 01:48:35 +0900（lonewolf-jp） |
| subject | `feat(isr): stamp retire entries with their domain and aggregate latency per domain` |
| 規模 | **49 ファイル**／+5,455 −215 |
| push | **済み**（origin/main と同一。commit 直後の記録にあった「未 push」は現在は該当しない） |

承認条件の確認:

| 条件 | 結果 |
|---|---|
| 1. 75 件・未追跡 154 件を一括 commit せず選別 | **満たす**。commit は 49 ファイルのみ。`.github/scripts/*`（6 件）・`ConvoPeq.md`・`doc/**` は含まない |
| 2. 無関係な変更・既存作業・生成物の混入がないこと | **満たす（結果検証）**。バイナリ 0 件、`doc/`・`.github/`・生成物 0 件。49 ファイルは domain tag 本体とその前提（段階①／段階③／Gate1／reader 計測・分類・テスト）に限られる |
| 3. 設計差分 v2.9 の内容・SHA-256 の維持 | **満たす**。53,979 B / `C3AE50991D9AB0A50BF09C9D8BDBD0B8919AC4F04732F567D93AE76656848427`（不変・git 未追跡のまま） |
| 4. hash と変更ファイル一覧の記録・報告への対応付け | 本節と `evidence/commit_record.txt`（49 ファイルの追加/削除行数つき） |

- 全 49 ファイルの一覧は `evidence/commit_record.txt` の表にある。内訳は次の 2 群である。

| 群 | 件数 | 内容 |
|---|---|---|
| domain tag 本体（今回の実装・検証で触った範囲） | **28** | `RetireDomain.h`・`EpochDomain.h`・`IRetireProvider.h`・`DeferredDeletionQueue.h`・`ISRRetireRouter.h/.cpp`・`ISRRetireLatency.h`・`ISRHB.cpp`・`RetireQuarantineStore.h`・`AudioEngine.h`・`EQProcessor.h`・`ConvolverProcessor.h`・`DeletionQueue.cpp`・`SnapshotRetireManager.h`・テスト 11 本・`DomainTagTests.cpp`・`CMakeLists.txt`・`AGENTS.md` |
| それ以外（今回より前に承認済みで、当時未コミットだった作業） | **21** | `AudioEngine.{CtorDtor,EQResponse,RebuildDispatch,Timer}.cpp`・`ISRDebugRuntime.*`・`ISREvidenceExporter.*`・`ISRHB.h`・`ISRRuntimePublicationCoordinator.cpp`・`RetireLatencyTeardownReporter.h`・`WorldLifetimeRecorder.h`・`IReaderEpochProvider.h`・`RCUReader.h`・`ReaderMeasureRegistry.h`・`ScopedEpochReader.h`・`SnapshotCoordinator.h/.cpp`・`EQProcessor.{Coefficients,Core,Processing}.cpp` |

- **ファイル単位の依存確認（2026-11-12 実測）**: 21 ファイルはいずれも `domainId` / `RetireDomain` を**1 箇所も参照していない**
  （`rg` で 0 件）。`recordComplete` の呼出しも 0 件で、`EpochDomain` の生成箇所も無い（参照はコメントと `#include` のみ）。
  したがって **21 ファイルは domain tag の API 変更に追随するために必要なファイルではない**。段階①・段階③・Gate1・reader 計測・
  案2（publish 理由）という、別途承認済みで当時未コミットだった作業が同じ commit に同梱されたものである。
- **したがって、49 ファイル構成を「当初の承認条件（今回の実装・検証に直接関連する変更に限定）に完全適合」と評価しない。**
  Owner の判断（2026-11-12）は「`173e6995` を現状のまま受け入れ、履歴の書き換え・分割・再 commit は行わない」であり、
  push 済み履歴を書き換える利益がリスクに見合わないためである。28 ファイルの実装本体と 21 ファイルの過去承認済み作業は区別して記録する。
- AGENTS.md の push 表記訂正は、commit 済みの実装とは分離した**未コミットのドキュメント変更**として扱う。
- commit に含めなかったもの: `.github/scripts/isr-*.ps1`（6 件）、`ConvoPeq.md`（実装後に再生成していない）、`doc/**`（設計・報告・証跡は履歴に含めない運用）、その他当時の無関係な変更。
- 現在の作業ツリーには本作業と無関係な変更 34 件・未追跡 222 件がある（別作業）。本記録はそれらに触れていない。
