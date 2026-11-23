# Gate 1（Reader safety）修正 — 実装・検証・判定

Owner 承認（2026-11-11）に基づく Gate 1 修正の最終報告である。**段階①（reclaim ドライバ）には着手していない**（Owner の指示どおり Gate ② の判定を待つ）。commit／merge／release も未実施（HEAD `c94d3767`、全変更は未コミット）。

## §0 結論

1. `EQProcessor::process()` は RCU 入場の成否を確認し、**失敗時は退役対象（`EQState` と `BandNode`）を一切参照しない**構造になった。
2. その過程で **`BandNode` も同じ保護欠如だった**ことが判明し、併せて修正した（§2.2）。これは当初スコープ（`currentStateBits` のみ）を広げる必要があった箇所である。
3. `RCUReader::enter()` は Owner 指摘の同時入場競合を排除するため、**所有権 CAS を depth 更新より先**に行う形へ再設計した（§3.2）。
4. **Gate ② は合格**と判定する（7 ケースすべてに再現可能な証拠がある。§5）。
5. 段階①（reclaim ドライバ）は未着手。判定結果を見たうえでの再承認をお願いしたい。

## §1 承認範囲と、それに対する結果

| Owner 承認項目 | 結果 |
|---|---|
| RCU 入場成否の確認 | 実装済み（`guard.rootEnterSucceeded()`） |
| 入場失敗時のフォールバック | **`nullptr` 経路を採用**（`fallbackState_` は新設せず。理由は §3.1） |
| fallbackState_ の退役対象からの分離 | 方式変更により**条件自体が不要**（退役対象を参照しない） |
| 通常経路の変更 | 成功時は従来どおり `loadCurrentState(acquire)` 1 回。BandNode も従来どおり |
| RCU の memory-order 変更 | **していない**（acquire/release/acq_rel の指定は不変。変更したのは制御順序） |
| reclaim 頻度の変更 | **していない**（ドライバ未実装） |
| テスト用 friend 1 行 | 追加（`EQProcessorGate1TestAccess` のみ） |
| ケース 2〜5・7 の実装 | 実装済み・全 PASS（§5） |
| ケース5 fixture | 実装済み（a〜e の 5 段。§4.4） |
| 段階① ドライバ | **未着手**（Owner の最新指示は「未承認・未着手」） |

## §2 欠陥は 3 つあった

### 2.1 状態ポインタ（Gate 1 本体・Owner 指摘）

`process()` は `RCUReaderGuard` の入場成否を確認せず `loadCurrentState(acquire)` を実行していた。入場に失敗した Reader は epoch を公開しないため、reclaimer から見ると「reader はいない」。そこへ reclaim が走ると、読んだ旧ポインタは解放済みになり得る（UAF）。

保護された経路（入場成功）は happens-before で安全であることは前回報告した（`S_B` = `exchangeCurrentState` → `S_E` = `publishEpoch`(release) → `L_E` = `currentEpoch()`(acquire) → `L_B` = `loadCurrentState`(acquire) の連鎖）。欠陥は「保護されない経路」の存在である。

### 2.2 `BandNode` も同じ保護欠如だった（今回の発見）

`process()` は `loadCurrentState()` のほかに、`loadBandNode(i)` で得た `BandNode*` を `node->active` / `node->coeffs` / `node->mode` として**無条件に deref** していた。`BandNode` も setter ごとに `retireBandNodeDeferred()` で遅延解放される退役対象であり、寿命の保証は EBR だけである。

したがって「保護に失敗した Reader が退役対象に**一切**触れない」という Gate 1 の完了条件は、状態ポインタだけの修正では満たせない。**この 1 か所を追加で塞いだ**（§3.1）。

> 参考: `process()` 内で EBR 依存の共有オブジェクトを辿るのは `currentStateBits`（`EQState`）と `bandNodeBits`（`BandNode`）の 2 系統だけで、他は自前バッファと atomic のみである（`grep` で確認）。

### 2.3 `RCUReader` の入場状態管理（Owner 指摘）

- 元の `enter()` は `nestingDepth` を先に増やし、`previousDepth > 0` ならネストとして即 return していた。別スレッドが同じインスタンスに入ると「ネスト」と誤認され、所有者 CAS に到達しない。
- `rootEnterSucceeded_` は「thread confined」と明記された非 atomic フラグであり、他スレッドから書くと所有者側とデータ競合する。
- 1〜3 行の対症療法（nested 分岐で所有者確認）は、この 2 点のどちらも解決しないため**採用しなかった**（Owner の差し戻しどおり）。

## §3 実装

### 3.1 `process()` — 入場失敗時は退役対象を参照しない

```cpp
convo::RCUReaderGuard guard(rcuReader);
const bool protectedRead = guard.rootEnterSucceeded();
if (! protectedRead)
    unprotectedReadCount_.fetch_add (1u, std::memory_order_relaxed);

const auto* stateSnapshot = protectedRead
    ? loadCurrentState(std::memory_order_acquire)
    : nullptr;
```

```cpp
// BandNode も EBR 保護下でのみ寿命が保証されるため、取得自体を行わない
if (protectedRead)
{
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        auto* node = loadBandNode(i, std::memory_order_acquire);
        ...
    }
}
```

- 入場失敗時は `stateSnapshot == nullptr` かつ `numActiveBands == 0` となり、**出力は素通し**（バンド処理も飽和処理も行わない）。取得しないので deref も起こらない。
- `fallbackState_` を新設しなかった理由:
  - 退役対象と分離するには「このオブジェクトは退役させない」という規約を新たに守り続ける必要がある。参照しなければ契約違反が起こり得ない。
  - `process()` は `stateSnapshot == nullptr` を「状態未公開」として既に扱える（`saturation` は 0.2、`agcEnabled` は false。参照は 3 か所、いずれも null チェック付き）。
  - EQProcessor の寿命に紐づく新メンバが不要。
- Owner 条件 A（寿命・退役対象外・RT 読み取り専用）と B（ニュートラル設定の正しさ）は、この方式では**条件自体が発生しない**。C（`rootEnterSucceeded()` の意味を変えない）は既存 API を読むだけで満たす。

### 3.2 `RCUReader::enter()` の線形化と `RCUReaderGuard`

| 手順 | 内容 |
|---|---|
| 1 | `ownerThreadToken` を acquire load。**他スレッド所有なら状態を一切変更せず `false`**（depth も owner も触らない） |
| 2 | 自分が所有者なら depth を +1 して外側の成否を返す（同一スレッドのネスト） |
| 3 | owner が 0 なら **CAS で所有権を先に確保**（ここが線形化点）。負けたら状態を変更せず `false` |
| 4 | 確保後に depth 0→1、slot 取得、`enterReader`、`rootEnterSucceeded_ = true` |
| 5 | slot 取得失敗時は depth と owner を**両方ロールバック**して `false` |

- `rootEnterSucceeded_` は所有者スレッドだけが書く（thread confined を維持）。
- `RCUReaderGuard` は `ok_` に**呼出し単位**の成否を保持し、`rootEnterSucceeded()` は共有フラグではなく `ok_` を返す。
- 入場に失敗した guard は `exit()` を**呼ばない**（他スレッドの depth を壊さない）。
- memory order の指定は一切変更していない。

### 3.3 診断カウンタと RT 制約の担保

`unprotectedReadCount()`（`std::atomic<std::uint32_t>`）は入場失敗の回数を数える。Audio Thread から `fetch_add(1, relaxed)` するだけであり、atomic であることを lock-free の根拠にしないため、**コンパイル時保証**を置いた。

```cpp
static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
              "Gate1: unprotectedReadCount_ must be lock-free for RT use");
```

lock-free でない処理系ではビルドが失敗する（実行時に確かめない）。追加コストは失敗時に 1 命令のみで、確保・ログ・待機は行わない。

### 3.4 テスト専用 friend

`friend struct EQProcessorGate1TestAccess;` の 1 行のみ。製品 API の追加・既存メンバの公開は行っていない。用途は `loadCurrentState()` の参照、`flushPendingEpochAdvance()` の呼出し、`m_epochDomain->closeReaderRegistration()` による決定的な入場失敗の注入である。

## §4 検証

### 4.1 ビルドと回帰

- ビルド: `build_exit=0`（ConvoPeq ターゲット／テストターゲットとも）。
- ctest: **60/60 合格・333.03 秒**（最終ツリー。`is_always_lock_free` の static_assert を含む）。途中経過も 337.80／332.68／357.30 秒でいずれも 60/60。
- `CTest` の対象は 60 件で、追加したケース5 fixture は既存の standalone target `STG11EQRetireTests` 内で実行される（CMake の追加なし）。

### 4.2 アプリ実行（3 シナリオ）

| シナリオ | req | comp | unmatched | retire_delay | maxMs |
|---|---|---|---|---|---|
| burst40 | 1849 | 1849 | 0 | fail（latency_exceeds_threshold） | 4,421 |
| burstfast | 627 | 627 | 0 | fail | 11,655 |
| eq | 187 | 187 | 0 | fail | 13,989 |

- D1 の `unmatched=0`・`pending=0` は維持。`world: commits == destroys`、`overflow=0`。
- `retire_delay` が fail のままなのは**想定どおり**である。EQ ドメインの reclaim は `prepareToPlay` / `releaseResources` / 破棄時にしか走らないため、最大待ちは数秒になる。これを短くするのが段階①（reclaim ドライバ）であり、本修正はその前提条件を整えただけである。

### 4.3 ガード意味論ハーネス（実 EpochDomain ＋ 偽 provider）

`IReaderEpochProvider` を偽装して失敗を決定的に注入する。証跡は `evidence/gate1_guard_result.txt`（**54 PASS / 0 FAIL**）。

| 検証 | 結果 |
|---|---|
| 正常入場／退出後（active 0・minReaderEpoch 公開） | PASS |
| ケース2 他スレッド入場 → `false`、保持者の active slot 不変 | PASS |
| ケース2 所有者解放後の再入場（同一スレッド／別スレッド） | PASS |
| ケース3 実 EpochDomain の 64 slot 枯渇 → `false`・active 0・再試行も `false` | PASS |
| ケース3 偽 provider 枯渇 3 回 → slot 回復後に即成功（**owner/depth ロールバックの実証**） | PASS |
| ケース3 枯渇時に register が毎回呼ばれる（キャッシュ汚染なし） | PASS |
| ケース4 registration closed（新規は `false`・既存は保護維持） | PASS |
| ケース7 ネスト（内側・退出後・最終退出・再入場×3） | PASS |
| ケース7 外側が失敗なら内側も失敗（保護済みと誤認しない） | PASS |
| 追加 owner=0 からの同時入場 8 スレッド（成功 1・失敗 7・depth 残留なし） | PASS |
| 追加 同一スレッド反復 200 回（depth リークなし） | PASS |

> 実装メモ: `EpochDomain` は 218 KB あり、1 関数フレームに複数置くと 1 MB のスタックを超える。ハーネスは `/F 33554432` でリンクしている（テスト側の都合で、製品コードとは無関係）。

### 4.4 ケース5 fixture（実 EQProcessor ＋ 実 reclaim ＋ 競合）

`src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` の `checkGate1ProcessIsolation()`。

| 段 | やること | 実測 |
|---|---|---|
| (a) 正常入場 | `prepareToPlay` 済み EQProcessor で 3 ブロック処理 | `unprotectedReadCount()==0`、出力は有限・非無音、`currentState` 非 null |
| (b) 入場失敗（reclaim 済み） | coordinator を接続し setter で交換 → 退役 → `flushPendingEpochAdvance()` で publish + reclaim（`eqOwnedPendingRetire()==0`・`eqRetireDropCount()==0`・currentState が退役ポインタと異なる）。`closeReaderRegistration()` で入場を決定的に失敗させ 3 ブロック処理 | 失敗経路の出力は**バンド無効の素通し参照と完全一致**（`dRef=0`）。バンド有効参照とは `dBand=2.105`（振幅 3.000）だけ異なる |
| (c) 毒入れ | reclaim 後に同型オブジェクト 8 個を確保し saturation 1.0 / gain +24 dB を書き込む（解放ブロック再利用時に誤読が必ず観測される） | 結果は不変（`dRef=0`） |
| (d) reclaim 反復 | 入場失敗のまま setter → flush → process を 5 回反復 | `unprotectedReadCount` が +15、出力は常に有限 |
| (e) 競合 | 音声側は入場失敗のまま連続 `process()`、非 RT スレッドが setter → `flushPendingEpochAdvance()`（publish + reclaim）を 300 回 | 774〜1,000 ブロック処理・`unprotectedReadCount` = ブロック数と一致・drop 0・非有限値なし・クラッシュなし |

- 入場失敗と reclaim の**順序を制御**した検証（b）と、**同時進行**の検証（e）を分けている（Owner 指示の「分離」）。
- （b）は「入場失敗 → 退役対象を読まない」、（e）は「reclaim と入場失敗が同時に起きても触れない」の実証である。

### 4.5 ミューテーション試験（テストの検出力の実証）

`process()` のガードを一時的に外す（状態は無条件 `loadCurrentState`、バンドは `if (true)`）と、同一テストが `dRef=2.100` で **FAIL** した（`evidence/gate1_process_fixture_mutant_fail.txt`）。すなわち本テストは「保護なしで退役対象を読んだ」ことを検出できる。復元後は **PASS**（`evidence/gate1_process_fixture_pass.txt`）。

> 「`nullptr` が返ったこと」だけでは合格にしない、という Owner 条件に対しては、この (b)+(c)+ミューテーションの 3 点で応えている。

## §5 Gate ② 判定

| # | ケース | 証拠 | 判定 |
|---|---|---|---|
| 1 | 正常入場時の従来処理との同値性 | fixture (a) と ctest 60/60（既存 EQ 系回帰） | 合格 |
| 2 | owner CAS 失敗時のフォールバック | ハーネス「case2 …」7 項目、`enter()` 線形化 | 合格 |
| 3 | slot 枯渇時のフォールバック | ハーネス「exhaustion …」5 項目＋「fake exhaustion …」9 項目 | 合格 |
| 4 | registration closed 時のフォールバック | ハーネス「case4 …」5 項目 | 合格 |
| 5 | reclaim と入場失敗の同時進行 | fixture (b)(c)(d)(e)＋ミューテーション | 合格 |
| 6 | `unprotectedReadCount_` の寿命・不変性 | atomic＋`is_always_lock_free` static_assert＋全テストで 0／期待値一致 | 合格 |
| 7 | ネストした Reader の入退場 | ハーネス「case7 …」7 項目＋「fake: …」4 項目 | 合格 |

**Gate ② は合格**。ただし判定は「上記の証拠で確認できる範囲」であり、製品全体の並行性証明ではない。特に `process()` と setter の同時実行は設計上 EBR 前提であり、本修正はその前提が破れた場合の安全側の扱いを定めたものである。

## §6 変更ファイルと差分

| ファイル | 変更 |
|---|---|
| `src/core/RCUReader.h` | `enter()` を `bool` 返却へ・所有権先行 CAS による線形化・他スレッド入場は無変更で `false`・slot 失敗時に owner/depth をロールバック・`RCUReaderGuard` が呼出し単位の成否を持ち失敗時は `exit()` を呼ばない |
| `src/eqprocessor/EQProcessor.Processing.cpp` | 入場成否で状態参照と BandNode 取得を分岐（失敗時はどちらも参照しない） |
| `src/eqprocessor/EQProcessor.h` | `unprotectedReadCount()`（診断）・`is_always_lock_free` の static_assert・friend 1 行 |
| `src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` | ケース5 fixture（a〜e） |
| `AGENTS.md` | Gate 1 の設計・検証・ミューテーション試験の要点（約 20 行） |

差分は 2 系統を用意した。

- `patches/` — HEAD `c94d3767` 基準（01 process / 02 counter+friend / 03 RCUReader / 05 fixture）
- `patches/gate1_only/` — 変更前スナップショット `tmp/rollback/gate1_before/` 基準（Gate 1 の差分だけを見たい場合）

## §7 経緯（差し戻し 3 回の要約）

| 回 | Owner 指摘 | 対応 |
|---|---|---|
| 1 | `fallbackState_` 案の採用と、寿命・内容・Guard 状態の条件 | `nullptr` 経路に倒す設計へ変更（条件自体が不要になる）＋friend 承認 |
| 2 | 失敗した guard に対する明示 `exit()` がハーネスの欠陥、owner=0 からの同時入場競合が未解決 | ハーネスの `exit()` を除去。`enter()` を所有権先行 CAS に再設計 |
| 3 | （本報告）RCUReader 再設計の結果とケース5 | 54 PASS／0 FAIL、fixture a〜e、ミューテーション試験で検出力を実証、ctest 60/60 |

途中で判明した「ケース2 FAIL」の原因は、ハーネス側の不正な `exit()` と、`nested` 判定が所有者判定より先にあったことの 2 つであり、**いずれも製品コードの slot cache（thread_local）の問題ではなかった**。slot cache（`preferredThreadId` / `acquireThreadSlot`）は変更していない。

## §8 未着手・制約

- **段階①（reclaim ドライバ）は未着手**。設計 v3（`doc/work113/20261111-0600_reclaim設計v3/reclaim_driver_design_v3.md`）は提出済みで、実装は Gate ② 合格を条件に再承認を待つ。
- 段階③（`firstScanTs` と 3 分類）以降、Reader レジストリ統合は未承認・未着手。
- `ENABLE_ASAN` ビルドでの実行は未実施。代わりにミューテーション試験（ガードを外すと必ず FAIL）で検出力を担保している。ASan での再確認が必要なら別途実施する。
- commit／merge／release は未実施。

## §9 成果物と提出先

- 本報告: `doc/work113/20261111-0800_Gate1修正/report.md`
- 差分: `patches/` 4 本、`patches/gate1_only/` 3 本
- 証跡: `evidence/`（ガードハーネス cpp/exe/結果、fixture PASS/ミューテーション FAIL のログ、変更前マニフェスト、変更後 SHA-256、ctest ログ）
- 結合ソース: `ConvoPeq.md`（7,083,911 B）と 2 MB 未満の分割 4 本（`ConvoPeq_part1..4.md`・各 1.9 MB 以下）
- Dropbox: `/ConvoPeq-gate1-fix-20261111/`（本フォルダ一式）、`/ConvoPeq-source-20261111-gate1/`（`ConvoPeq.md` と分割 4 本）
