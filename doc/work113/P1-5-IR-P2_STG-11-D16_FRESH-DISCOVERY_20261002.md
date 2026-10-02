# STG-11-D16 Fresh Discovery — Lifecycle / Authority / Queue / RT Boundary Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D16_FRESH-DISCOVERY_20261002.md`
- Work item: **STG-11-D16** — D15-1 修正後の最新状態に対する新規 defect 探索
- Date: 2026-10-02
- Authority: リポジトリルート `ConvoPeq.md`（監査 AI が参照する `ConvoPeq(20261002-142258).md` と同一ファイル。
  ファイル名の違いを理由に別世代として扱わない。Owner 確認済みの同一最新統合ソース）
  - baseline commit: `a3b2d5b0f5eab553e1c5a76db69c9e412966865c`（D15-1 repair）
  - `Generated: 2026-10-02 23:19:21` / 5,941,824 B
  - SHA-256: `3A51AB3414DBA90005B99540376A93D6AB76AE1A96A754CE9E7050E2E0593045`
  - `--check`: `NEWER_SRC_COUNT = 0` / **FRESH**（調査開始時に確認）
  - 過去資料・記憶・旧行番号は根拠として使用していない。結論はすべて現行ソースから導出。
- 本 work item は **read-only Fresh Discovery**。**production / test / build ファイルの変更は一切行っていない。**
- 本書は **未 commit**（Owner の判断待ち）

---

## 0. 判定

```
STG-11-D16 — NO-GO（concrete defect 0 件、observation 1 件）
```

D15-1 までの修正を前提に、RuntimeWorld / Snapshot lifecycle、Publish admission と
failure rollback、Registry / OwnerChannel / Intent の二重表現、sequence / generation /
epoch identity、terminal ownership、shutdown 中の race、deferred lifetime、
queue overflow、Epoch / RCU、非同期 callback、RT boundary、atomic 規約、validator bypass、
authority duplication、silent ownership loss を現行ソースで確認した。
Owner の 7 条件をすべて満たす新規 defect は存在しない。

---

## 1. 追跡結果

### 1.1 sequence / generation / epoch identity

`reserveRuntimePublicationIdentity`（`AudioEngine.h:3763-3772`）は generation・worldId・
publicationSequence をいずれも atomic fetch-add 系で採番し、再利用は無い。
`publicationSequence` は 0 始まりの counter に +1 するため初回から非ゼロであり、
facade の seqId==0 拒否（`:4903` / `:4915-4917`）と整合する。
identity 衝突の経路は無い。

### 1.2 D15-1 mutex の NonRT-only 確認（Owner 指定の必達事項）

`ownerChannelProducerMutex_` の参照は次のみに限定される。

- 宣言（`AudioEngine.h:2928`）
- facade 内の 2 箇所の `lock_guard`（`:4946` enqueue、`:4983` take-back）
- コメント（`:4869` / `:4939`、テスト内の mirror を除く）

facade 入口には `ASSERT_NON_RT_THREAD()` があり、呼び出し元は
Message / Rebuild / CoordinatorLoop の NonRT 3 スレッドのみである
（audio-thread trySubmit は排除済み）。consumer `take()`（`RuntimePublishExecutor.h:31`）
は無変更・lock-free である。RT への mutex 波及は無い。

### 1.3 RT boundary 全面走査

`DSPCoreIO.cpp` / `DSPCoreFloat.cpp` / `DSPCoreDouble.cpp`（`OutputFilter` を含む）に
mutex / lock / allocation / delete / logging（RT-safe ring への言及コメントのみ）は無い。
`RT Thread = Read / Execute / Output` の境界は維持されている。

### 1.4 非同期 callback の lifetime

- `adaptiveAutosaveCallback`: MainWindow が登録（`MainWindow.cpp:292`、`safeThis` 使用）し、
  dtor で空にクリアする（`:1131-1132`）。learner worker からの呼出しは engine shutdown 時に
  join される。UAF 経路は無い。
- `m_healthMonitor` の 3 callback: member 間であり、発火元は自スレッド（join 済み）である。
  lifetime 違反は無い。

### 1.5 receipt timeout / failure rollback / deferred lifetime

`commitRuntimePublication`（`:5006-5040`）の timeout ≠ failure 扱い、
`submitPublishRequest` の disposition 分岐、deferred slot の terminal 処理は
いずれも設計どおりであり、新規の穴は無い（D19 の failure boundary 結論と一致）。

### 1.6 D7〜D15-1 の再発確認（該当なし）

---

## 2. Observation O-1: `currentBuildSnapshot_` の非 lock 読出し（defect 計上なし）

`currentBuildSnapshot_` は mutex 保護の建付けである。
書込みは lock 下（`AudioEngine.Commit.cpp:816-819`）、getter も lock 下
（`AudioEngine.h:4742-4746` `getCurrentBuildSnapshotForRecovery`）である。

しかし `AudioEngine.Timer.cpp:1165` は `&currentBuildSnapshot_` を直接 builder に渡し、
`RuntimeBuilder::buildRuntimePublishWorld` は lock なしに field を読む
（`:239-254` dspProjection、`:351-355` resource/timing）。
書込み側（RebuildThread 経由の commit enqueue）と読出し側（message thread の Timer）が
重なると data race になる。

**defect として計上しない。** 理由は次のとおり。

1. struct は plain int / double / bool のみであり、x86-64 の aligned access では
   tearing は実質起きない。現実の帰結は新旧 field の混合（mixed-generation snapshot）である。
2. 混合された各 field は個別に有効であり、validator の検査対象（dither 等）は
   集合内に収まる。無効な world が publish される構造ではない。
3. 次回 publish で自己修復する。stuck・leak・crash のいずれにも至らない。
4. ISR invariant の違反が無い（RT safety・authority・ownership のいずれも無傷）。

将来 TSan 稼働環境では検出される class の問題であり、修正するなら Timer 側を
lock 下の getter 経由に変える 1 行であるが、現行基準では「設計上気になる」に留まるため
計上しない。対応の要否は Owner 判断に委ねる。

---

## 3. 本 D16 の作業記録（read-only 遵守）

```text
変更ファイル: 0 件（production / test / build / CMakeLists.txt / ConvoPeq.md すべて無変更）
追加ファイル: 本書 1 件（未 commit。Owner の判断待ち）

実施した操作:
  - 読み取り（git show / rg / Read）: 多数
  - ビルド / テスト: なし（ソース未変更のため不要）
  - reset / clean / rebase / amend / squash / force push: いずれも不使用

main checkout (C:\VSC_Project\ConvoPeq): 未改変（HEAD = bda43034）
作業 worktree: HEAD = a3b2d5b0（= origin/main）、clean、未追跡 1 件（本書のみ）
```
