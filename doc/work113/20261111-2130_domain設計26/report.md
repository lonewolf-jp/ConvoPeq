# domain tag 設計 v2.6 — Release 契約の確定と版表記の修正

Owner レビュー（2026-11-11、v2.5 に対する 2 点）に応えた v2.6 を作成した。方針は v2.5 から変えていない。**製品コードは未変更**で、実装着手は未承認のままである。

## §0 結論

- **指摘 1（ctor の Release 契約）**: **B-1「全ビルドで不正値を採用しない」**を採用した。Release でも不正値はドメインとして採用されず `Unknown` に落ち、`invalidDomainAssignments()`（atomic カウンタ）で発生を観測できる。Debug では加えて `jassertfalse` で停止する。`std::abort()` による B-2 は採らない（理由は §1.2）。
- **指摘 2（§9 の表記）**: 「本設計差分 v2.4 の確定」を「本設計差分 v2.6 の確定（v2.5 の条件 2 点を反映済み）」に修正した。
- 契約が **Debug と Release で同じ結果になる**ことを、JUCE に依存しない最小ハーネスで実測した。`clang++ 23.1.5` の `-O0` と `-O2 -DNDEBUG` の両方で PASS、警告なし（§3）。
- 検証項目は 15 のまま。#15 を「判定子だけ」から「**ctor の実動作（両構成）**」まで広げた。Owner 承認範囲の「14 項目」は v2.4 時点の番号である。

## §1 指摘 1 — ctor の Release 契約

### 1.1 案の比較

| 案 | 内容 | Release での挙動 | テスト | 判定 |
|---|---|---|---|---|
| A | Debug 専用の事前条件検査（`jassert`）。Release は呼出し側の責任 | 何も起きない | 判定子のみ | 不採用（不正値がそのまま採用される） |
| **B-1** | `sanitizeDomain()` で不正値を `Unknown` に落とし、**採用しない**。カウンタで観測。Debug は assertion 追加 | 不正値は `Unknown` になる | ctor の実動作を両構成で検証できる | **採用** |
| B-2 | 不正値で `std::abort()` | プロセスが落ちる | 死亡テストが必要 | 不採用 |

### 1.2 B-2 を採らない理由

`EpochDomain` の ctor は `AudioEngine` と `EQProcessor` の初期化時に走る。Release で `abort` すると起動不能になり、現場で原因を切り分けにくい。加えて in-process のテストでは観測できず、死亡テストを別に用意する必要がある。「不正値を受け入れない（安全側の `Unknown` に落とす）＋観測可能」のほうが、この製品の実運用に合う。

### 1.3 確定した契約（設計差分 §3.2）

| 項目 | 内容 |
|---|---|
| 全ビルド | `isAssignableDomain()` が false の値は**採用しない**。`domainId()` は必ず `Unknown` |
| 誤帰属の防止 | 不正値が `EngineISR` / `EQ` になることはない |
| 観測 | `invalidDomainAssignments()`（`std::atomic<std::uint32_t>`）が発生回数を数える。両ビルドで読める |
| Debug の追加検出 | `jassertfalse`。`jassert` は Release で無効なので**契約の本体ではない** |
| 出力 | カウンタは診断用。JSON には出さない（§8 の非目標に追加） |
| 宣言順 | `domainId_` を `globalEpoch` より先に宣言する（初期化子の順序警告を出さない） |

v2.5 までの「`jassert` だけで拒否する」という記述は、Release では何も起きない点で不正確だった。v2.6 ではこの点も本文に明記した。

## §2 指摘 2 — §9 の版表記

| 箇所 | 修正前 | 修正後 |
|---|---|---|
| §9 の実施順序 #1 | 本設計差分 v2.4 の確定 | 本設計差分 v2.6 の確定（提出済み・レビュー待ち／v2.5 の条件 2 点を反映済み） |

あわせて、履歴的な「本版」表記（v2.2／v2.3／v2.4 を指す 3 箇所）を版番号に置き換え、現行版が v2.6 だけを指すようにした。

## §3 検証 — 契約が両構成で同じになること

設計差分 §3.2 のコードをそのまま写した最小ハーネス（`evidence/contract_check.cpp`、JUCE 非依存）で、次の 16 項目を確認した。

- 判定子: `Unknown` / `EngineISR` / `EQ` が true、`Unstamped` / `Count` / 5 / 255 が false。
- ctor: 許可値 3 つは採用され、カウンタは増えない。
- ctor: 不正値 4 つ（`Unstamped` / `Count` / 5 / 255）は `Unknown` に落ち、カウンタが 4 増える。
- 不正値が `EngineISR` / `EQ` にならないこと。

| 構成 | ビルド | 実行 |
|---|---|---|
| Debug | `clang++ -std=c++17 -O0 -Wall -Wextra` → 警告なし | `PASS`（counter=4, failures=0） |
| Release | `clang++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra` → 警告なし | `PASS`（counter=4, failures=0） |

ソース側の現状も再確認した。`RetireDomainId` 0 件、`isAssignableDomain` / `sanitizeDomain` 0 件、既定 ctor は `src/core/EpochDomain.h:27` に現存、生成箇所は本番 3（宣言 2 + `make_unique` 1）とテスト 35（宣言 32 + `make_unique` 3）。**domain tag の実装は未着手**で、v2.5 の棚卸しから変わっていない。

## §4 成果物と状態

| 項目 | 内容 |
|---|---|
| 設計差分 | `doc/work113/20261111-1512_domain設計/design_diff.md`（v2.6・45,247 B） |
| 取得経路 | `doc/work113/20261111-1512_domain設計/MANIFEST.md`（v2.6 に更新） |
| 証跡 | 本報告の `evidence/`（契約ハーネス・ビルド実行ログ・ソース現状） |
| 製品コード | 未変更（設計のみ） |

未承認のまま残る項目は、domain tag 実装、§7.2 の 15 項目による検証、Reader レジストリ統合、ASan、commit／merge／release。案 2（publish 理由タグ）は v2.6 でも現状維持である。
