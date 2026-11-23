# 文書整合性修正および ConvoPeq.md 再生成 報告

- 実施日時: 2026-11-14 15:16〜15:28 JST
- 種別: 文書整備（ソースコード変更なし・コミットなし）
- 生成元: `6e099932bcb4f122a72526f1c9786eecd7fc08b6`

---

## 1. 作業開始前の前提確認

| 項目 | 期待値 | 実測 |
| --- | --- | --- |
| branch | `main` | `main` |
| HEAD | `6e099932` | `6e099932` |
| `origin/main` | `6e099932` | `6e099932` |
| staged | 0 件 | 0 件 |
| stash | 0 件 | 0 件 |
| 未コミット変更 | 172 件 | 172 件（modified 38 / untracked 134） |
| `AGENTS.md` / `ConvoPeq.md` | unstaged | どちらも ` M` |

前提はすべて一致したため、そのまま作業を実施した。

---

## 2. 作業 A: AGENTS.md の 3 点修正

### A-1. AC1 ミラーの状態

修正前（847〜849 行付近）:

```
  - **既知の未修正**: `testAC1_drainBreakdownMatchesPredicate` の `allDrained` ラムダは 14 条件のまま
    （C9-A 後は 13）。シナリオ上 `retireBacklog` が常に 0 なので PASS するが、ミラーが乖離している。
    独立課題として起票済み: `doc/work113/20261113-2350_AC1修正課題/issue.md`（修正は未承認）。
```

修正後:

```
  - **AC1 ミラー（解消済み）**: `testAC1_drainBreakdownMatchesPredicate` の `allDrained` ラムダは
    C9-A 後も 14 条件のままで、`retireBacklog` が常に 0 のシナリオでは PASS するため乖離を検出できなかった。
    PR #207 の `74e9aab0` で `b.retireBacklog == 0` を削除して述語と同じ 13 条件に揃え、
    `setRetireAcceptedCount(7)` を与えても述語とミラーが一致するシナリオを追加した
    （起票: `doc/work113/20261113-2350_AC1修正課題/issue.md`）。
```

根拠（実測）: `74e9aab0` の差分で `b.retireBacklog == 0` が削除され、`setRetireAcceptedCount(7)` を
使うシナリオ (4) が追加されている。現行 `src/tests/ISRSemanticValidationTests.cpp` でも両方を確認した。
履歴（14 条件だったこと・起票先）は残した。

### A-2. `origin/main` の参照

修正前（656 行付近）:

```
## domain tag: 実装済み・commit 済み `173e6995`（origin/main と同一）・設計 v2.9 受理済み（2026-11-12）
```

修正後:

```
## domain tag: 実装済み・commit 済み `173e6995`（当時の origin/main と同一。現在の main は `6e099932`）・設計 v2.9 受理済み（2026-11-12）
```

根拠: 現在の `main` / `origin/main` はどちらも `6e099932`（最新は `Merge pull request #207`）。
当時の記述を消さず、「当時」と「現在」を併記した。

### A-3. domain tag の承認状態

修正前（652 行付近）:

```
  `driverWait` が 100 ms を大きく超えない）。帰属確定には domain tag が必要（未承認）。
```

修正後:

```
  `driverWait` が 100 ms を大きく超えない）。帰属確定には domain tag が必要（`173e6995` で実装済み）。
```

根拠: domain tag は `173e6995` に含まれて commit 済み。実装済みであることだけを書き、
別途必要な承認・検証の有無とは混同していない。

### A-4. 差分の範囲

`AGENTS.md` の差分は **+154 / −8**（作業前は +151 / −7）。増分は A-1 の +2 行、A-3 の +1/−1 行で、
A-2 は 1 行の置換のため行数は変わらない。3 点以外の記述は変更していない。

---

## 3. 作業 B: ConvoPeq.md の再生成

### B-1. 生成元の固定

本体 worktree は未コミット 172 件を含むため、**一時 worktree** を `6e099932` で作成して生成した。

```powershell
git worktree add --detach C:\VSC_Project\regen6e099932\ConvoPeq 6e099932
```

- 一時 worktree の `HEAD` は `6e099932`、`git status` は空（クリーン）を確認。
- ディレクトリ名を `ConvoPeq` にしたのは、生成物のタイトルとディレクトリツリーが
  `root_path.name` を使うためである（`ConvoPeq-regen-...` で一度生成したところタイトルが
  その名前になり、破棄して作り直した）。

### B-2. 生成処理

`AGENTS.md` の記載どおりリポジトリ直下の `output_sourcecode_markdown.py` を一時 worktree 内で実行した。

```
[15:27:23] 完了しました！ 出力先: ConvoPeq.md
スナップショット保存: ...\doc\ConvoPeqMD\20261114152723\ConvoPeq.md
```

- 生成対象は `TARGET_ITEMS = {'src', 'build.bat', 'CMakeLists.txt'}`（スクリプト既定）。
- 生成物のヘッダは `# Project Extract & Source Code: ConvoPeq`、`> Generated: 2026-11-14 15:27:23`。
- 一時 worktree のパス名（`regen6e099932`）は生成物に **0 件**（混入なし）。

### B-3. 分割版

既存の慣習（`doc/ConvoPeqMD/<timestamp>/` に `ConvoPeq.md` と `ConvoPeq_partNN.md`）に合わせ、
各パーツの先頭に `<!-- ConvoPeq.md part N/M (split at 2000000-byte limit) -->` を置き、
2,000,000 バイト以下になるよう行境界で分割した。

| ファイル | サイズ (bytes) | 2MB 以下 |
| --- | ---: | --- |
| `ConvoPeq.md`（完全版） | 7,067,912 | — （分割前の原本） |
| `ConvoPeq_part01.md` | 1,999,902 | はい |
| `ConvoPeq_part02.md` | 1,999,970 | はい |
| `ConvoPeq_part03.md` | 1,999,939 | はい |
| `ConvoPeq_part04.md` | 1,068,353 | はい |

ヘッダ行を除いて連結すると完全版と **バイト一致**する（7,067,912 バイト）。
分割漏れ・重複・順序の逆転はこの一致で排除される。

### B-4. マーカー検証（完全版・分割版とも）

| # | マーカー | 完全版 | 分割版（4 分割の合計） |
| --- | --- | ---: | ---: |
| 1 | `compareExchangeWeakAtomic`（PR #207 の Atomic Access） | 6 | 6 |
| 2 | `述語と同じ 13 条件`（#209 の c9 計算） | 1 | 1 |
| 3 | `setRetireAcceptedCount(7)`（AC1 の非 0 シナリオ） | 1 | 1 |
| 4 | `b.retireBacklog == 0`（AC1 修正前の条件） | **0** | **0** |

文字列の有無だけでなく前後を確認した。

- #1 は `AtomicAccess.h` の weak CAS ヘルパー定義（`std::enable_if_t<std::is_convertible_v<U, T>>` 付き）。
- #2 は `[DRAIN_DIAG]` の `const bool c9BridgeDrained =` を 13 条件で計算している箇所と、
  その直前の `CI-SHUTDOWN-006` コメント。
- #3 は AC1 のシナリオ (4)（新しい `RuntimeIntentCoordinator` に 7 を与え、述語とミラーが
  一致することを確認するブロック）。
- #4 は `allDrained` ラムダから削除済み。`b.retireBacklog` 自体は getter 値の一致確認
  （`b0.retireBacklog != c->getRetireAcceptedCount()`）に残っており、これは述語ではない。

未コミット変更の混入がないことも確認した（いずれも 0 件）:
`probeStopLearning` / `InputInspection` / `ConvolverStopProbe` / `LearnerStateAuthorityTests` /
`CacheManagerHeaderValidationTests`。

### B-5. 本体 worktree への反映

| 対象 | 結果 |
| --- | --- |
| `ConvoPeq.md`（本体） | 7,233,999 → **7,067,912** バイト。一時 worktree の生成物と SHA-256 一致（`d51edc1abf4eea037dfa63013395725a60117ea7e0850936b1e4c7105363ec16`） |
| 分割版 | `doc/ConvoPeqMD/20261114152723/` に完全版と 4 分割を配置（`.gitignore` 対象のため Git 状態は不変） |
| それ以外のファイル | 未変更 |

---

## 4. 最終検証

### Git 状態（作業前後で同一）

| 項目 | 値 |
| --- | --- |
| branch | `main` |
| HEAD | `6e099932bcb4f122a72526f1c9786eecd7fc08b6` |
| `origin/main` | `6e099932`（不変） |
| staged | 0 件 |
| stash | 0 件 |
| 追加コミット | なし（`git log -1` は `6e099932 Merge pull request #207`） |
| push / merge | 未実施 |

### 既存変更の保護

| 確認 | 結果 |
| --- | --- |
| modified の集合 | 38 件で**作業前と完全一致**（36 のコード・スクリプト＋`AGENTS.md`＋`ConvoPeq.md`） |
| untracked の件数 | 134 件で不変 |
| ソースコードの変更 | なし（今回書き込んだのは `ConvoPeq.md` と `doc/ConvoPeqMD/20261114152723/` のみ） |
| 一時 worktree | `git worktree remove --force` で撤去済み（残るのは本体と `ci-fix/173e6995` の 2 つ） |

### 文書検証

| 項目 | 結果 |
| --- | --- |
| AGENTS.md の 3 点 | 現在の事実と一致（A-1〜A-3） |
| ConvoPeq.md の生成元 | `6e099932`（クリーンな一時 worktree で生成） |
| マーカー 4 種 | すべて期待どおり（§B-4） |
| 分割ファイル数 | 4（最大 1,999,970 バイト） |
| 完全版と分割版の一致 | ヘッダ除去後の連結がバイト一致 |

---

## 5. 変更したファイル

| ファイル | 変更 |
| --- | --- |
| `AGENTS.md` | 3 点修正（A-1〜A-3）。+154 / −8 |
| `ConvoPeq.md` | `6e099932` から再生成。7,233,999 → 7,067,912 バイト |
| `doc/ConvoPeqMD/20261114152723/ConvoPeq.md` | 新規（生成スナップショット。`.gitignore` 対象） |
| `doc/ConvoPeqMD/20261114152723/ConvoPeq_part01.md`〜`part04.md` | 新規（2MB 以下の分割版。`.gitignore` 対象） |
| `doc/work113/20261114-1528_文書整合修正/report.md` | 本報告 |

---

## 6. 実施していないこと

- commit / push / merge: 未実施
- `git add` / ステージング: 未実施
- `git reset` / `git checkout` / `git clean` / `git stash`: 未実施
- ソースコードの変更: 未実施
- 既存の未コミット変更（36 ファイル・134 untracked）への変更: 未実施

---

## 7. 判断が必要な事項

なし。分割の出力先と命名は既存の `doc/ConvoPeqMD/<timestamp>/ConvoPeq_partNN.md` の慣習
（61 ディレクトリ分の実績）に合わせた。AGENTS.md の規定はサイズ上限のみを定めており、
出力先は明記されていないため、この慣習を根拠にした。
