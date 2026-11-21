# PRI-01・PRI-02 と C-01 の照合（2026-11-21 22:33 JST）

マージ後の `main`（`250f533e`）を基準に、PRI-01・PRI-02 が PR #229 の C-01 と同じ問題か、別の問題かを実コードで判定した。結論は「PRI-01 は C-01 と同一で修正済み、PRI-02 は C-48 と同一で非バグ」である。C-22 を PRI-01・PRI-02 へ集約する整理は、登録簿の対応表と合わない。

## 1. 結論

| 項目 | 旧 ID | 実体 | マージ後の状態 |
| --- | --- | --- | --- |
| PRI-01 | C-01 | `RCUReader::exit()` が所有者解放後に成否フラグを書く | 修正済み（`712ecb55`、main に取り込み済み） |
| PRI-02 | C-48 | `AudioSegmentBuffer::copyLatest` の読み順 | 非バグ（書き順と一致している） |
| C-22 | MED-16 | atomic のメモリ順序・静止契約の検証不足 | 判定不能のまま。PRI 番号は付いていない |
| C-24 | MED-18 | 学習 XML の再解析で起動と終了が遅い | 保留（要再測定） |
| C-39 | LOW-08 | 非 2 冪ブロックで遅延境界のブロックが欠ける | 条件付き（設計の前提外） |

この表は 2026-11-21 22:33 時点の状態である。以後の変更は §10 に追記した。

PRI-01 と C-01 は同一である。C-22（MED-16）は PRI 番号を持たないため、「C-22 を PRI-01・PRI-02 へ集約する」と、修正済みの C-01 と非バグの C-48 に合流してしまう。C-22 は判定不能のまま独立して残す。

## 2. 基準コミットの確認

- `main` の HEAD は `250f533e53f67690dceeefc60d9bfa8c9d54c6af`（PR #229 のマージコミット。親は `a9598d34` と `7531a2a8`）。
- マージコミットのツリーは `7531a2a8` のツリーと同一で、`git diff 7531a2a8 250f533e` は空である。
- `a9598d34` は PR #229 の base、つまりマージ前の `main` である。`main` の HEAD ではない。
- この環境の作業ツリーは次のとおりで、`a9598d34` をチェックアウトしているものは無い。
  - `C:\VSC_Project\ConvoPeq`: ブランチ `main`、HEAD `4d476d5c`（`a9598d34` より 103 コミット古い。未コミット 194 件）
  - `C:\VSC_Project\fix55`: ブランチ `fix/c01-c55-20261121`、HEAD `7531a2a8`
  - `C:\VSC_Project\main-audit`: 今回作った照合用 worktree、HEAD `250f533e`
- したがって「main を `a9598d34` へ更新した」という記述は、少なくともこの環境の状態とは一致しない。どこかの端末でその記述が正しいなら、その端末のローカル `main` はマージ前のままである。`git fetch` は追跡参照を更新するだけでローカルの `main` は動かないため、`git fetch origin` の後に `git merge --ff-only origin/main`（または `git pull --ff-only`）で `250f533e` へ進める。作業ツリーに変更がある場合は、独立した worktree で確認する。
- 以降の判定は、すべて `250f533e` のソースで行った。

## 3. PRI-01（C-01）の照合

### 3.1 登録簿が指す問題

登録簿の §9.1 は、`RCUReader::exit()` が `ownerThreadToken` を 0 に公開した後に `rootEnterSucceeded_` を false にし、その間に別スレッドが所有権 CAS に勝って true を書くと、旧所有者の false が新しい true を上書きする、と特定している。bool が非アトミックであることも同時に指摘している。参照元は PRI-01 である。

### 3.2 マージ後の実装

`src/core/RCUReader.h`（`250f533e`）で、修正は 3 点に入っている。

| 修正 | 箇所 | 内容 |
| --- | --- | --- |
| 呼出しローカルの reader | `src/audioengine/RuntimePublicationOrchestrator.cpp:48-59` | 複数スレッドから呼ばれる `trySubmitImpl` が、メンバ reader ではなく呼出しごとの `submitReader` を使う |
| フラグの atomic 化 | `src/core/RCUReader.h:241` | `std::atomic<bool> rootEnterSucceeded_`。読みは acquire、書きは release |
| 解放より先にフラグを落とす | `src/core/RCUReader.h:171-186` | `rootEnterSucceeded_` の false 公開を、`ownerThreadToken` の解放より前に移動 |

`a9598d34..250f533e` で `src/core/RCUReader.h` を変えたコミットは `712ecb55` の 1 件だけである。差分は、フラグの全アクセスを `consumeAtomic`／`publishAtomic` へ置き換え、末尾のリセットを削除して解放前の 1 回にまとめたものになっている。

`exit()` の順序は、深さの減算（139）、計測区間の終了（160-164）、所有者の照合（165）、フラグの false 公開（174）、`activeThreadId` の交換と `exitReader`（177-184）、`ownerThreadToken` の解放（186）である。`enter()` は所有権 CAS（58）を深さの更新（72）より先に行い、成功時にフラグを true で公開する（78）。したがって、登録簿が指した「解放後の false が新しい true を上書きする」窓は無くなっている。

### 3.3 残る経路の評価

所有者でないスレッドが `exit()` を呼ぶと、所有者の照合（165）より前に深さを減らしているため、深さがずれたまま所有権が残る。ただし、この経路は現行の製品コードからは到達しない。

- 製品コードの `RCUReaderGuard` はすべて関数ローカルの自動変数で、スコープを出ると同じスレッドで破棄される（例: `AudioEngine.Processing.AudioBlock.cpp:190`、`AudioEngine.Processing.BlockDouble.cpp:158`、`EQProcessor.Processing.cpp:527`）。
- `RuntimeReadHandle` も関数ローカルで作られ、同じ関数の中で破棄される。
- guard や handle を別スレッドへ渡す `std::move` は無い。ムーブは同一スレッド内の受け渡しだけで（`AudioEngine.h:2607`、`3766`）、スレッドを跨がない。
- `ObservedRuntime` はムーブ可能だが、スレッド親和性の検査は Debug の `get()` と `operator bool` にあり、製品経路で別スレッドへ渡していない。

よって、これは現時点の不具合ではなく、防御経路の非対称（`enter()` は所有権が先、`exit()` は深さが先）として残る。強化するなら、所有者の照合を深さの減算より前へ移すか、所有者不一致時に CAS で所有権を解放する。今回はコードを変えていない。

## 4. PRI-02（C-48）の照合

`src/AudioSegmentBuffer.h`（`250f533e`）で、書き込みは `writePosition` を release（104 または 112）した後に `totalSamples` を release（118）する。読み出しは `writePosition` を acquire（128）、続いて `totalSamples` を acquire（129）する。順序は一致しており、登録簿の「訂正（非バグ）」と一致する。コードの変更は要らない。

## 5. C-22（MED-16）の扱い

登録簿では C-22 の出典は MED-16 で、PRI 番号は付いていない。内容は「atomic のメモリ順序と静止契約の検証が足りない」で、判定は「判定不能（具体箇所がない）」である。PRI 番号を持つのは C-01（PRI-01）、C-02（PRI-03）、C-03（PRI-04）、C-04（PRI-06）、C-05（PRI-11）、C-06（PRI-12）、C-07（PRI-13）、C-08（PRI-14）、C-30〜C-32（PRI-07〜09）、C-48（PRI-02）、C-49（PRI-05）、C-50（PRI-10）である。

今回、reader 経路の atomic 操作 21 箇所（`RCUReader.h` の 45、51、53、58、67、68、72、78、86、88、90、126、139、145、148、165、167、174、177、183、186 行）について、順序の対応を確認した。

| 操作 | 順序 | 対応する相手 | 判定 |
| --- | --- | --- | --- |
| 所有権の CAS（58、90） | acq_rel / acquire | 他スレッドの CAS と解放 | 線形化点として妥当 |
| `ownerThreadToken` の読み（45、165） | acquire | 解放（186）の release | 妥当 |
| `rootEnterSucceeded_` の書き（78、86、145、167、174） | release | 読み（53、68、126）の acquire | 妥当 |
| `nestingDepth` の加減（51、67、72、88、139） | acq_rel | 同一スレッドの入れ子と `isInside()` | 妥当 |
| フラグ解除（145、148） | release | 次の `enter()` | 妥当 |
| `activeThreadId` の交換（177） | acq_rel | `acquireThreadSlot()` の acquire（202） | 妥当 |
| `preferredThreadId`（183） | release | 202-207 の acquire | 妥当 |

弱すぎる順序や、対応の欠けた公開は見つからなかった。したがって C-22 は「判定不能」のまま維持する。C-22 を PRI-01・PRI-02 へ寄せると、修正済みの C-01 と非バグの C-48 に合流するため、残課題としての追跡にならない。

## 6. C-24 と C-39

- C-24（MED-18）: 学習 XML の再解析による起動・終了の遅延。原因コードの特定と修正前後の同一条件実測が済んでいないため、保留を維持する。
- C-39（LOW-08）: 非 2 冪ブロックは設計の前提外で、2 の冪では境界を跨がない。現状維持（条件付き）を維持する。

## 7. Issue の状態

- #225（`ReaderMeasureRegistry` のタグ公開順）: C-29 として修正済み。`src/core/ReaderMeasureRegistry.h:142-159` が「初期化中(2) → タグ書込 → ready(1)」の順で公開し、読み側は `used` を acquire で読む（133）。契約テスト 2 項目（空タグで ready にならない、公開後にタグが見える）も通っている。修正箇所と検証根拠を Issue に記録し、クローズした。
- #215（MSVC の AVX2/FMA 経路）: C-27 として条件を `#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))` に変更済み（`src/CustomInputOversampler.cpp:212`、`664`）。ただし、これで Issue の受入条件が満たされたわけではない。MSVC の逆アセンブル確認、AVX2/FMA 経路の命令生成確認、性能測定が未実施のため、オープンを維持し、残る受入条件を Issue に明記した。

## 8. この照合での変更

コードの変更は無い。本報告書だけを追加した。新たな不具合も見つかっていない。

## 9. 参照

- 登録簿: `doc/work113/20261121-1201_詳細検証/report.md`（§9.1、§9.9、§9.10、対応表の出典列）
- 修正報告書: `doc/work113/20261121-1306_C55改修/report.md`（C-01、C-29、C-27）
- 判定対象: `250f533e`（`src/core/RCUReader.h`、`src/core/ReaderMeasureRegistry.h`、`src/AudioSegmentBuffer.h`、`src/audioengine/RuntimePublicationOrchestrator.cpp`、`src/CustomInputOversampler.cpp`）

## 10. 追記（2026-11-22 07:30 JST）

本文は 2026-11-21 22:33 時点の記録である。その後の実測と Issue の処理をここに追記する。本文は書き換えていない。

- C-24（MED-18）: 計画（PR #232）で固定した条件により再測定した（PR #234）。判定は「維持（原因コード無し）」である。学習 XML の増分として判定に使えるのは設定保存の 6.6 ms だけで、起動・終了・設定読込の差はドリフト以下だった。設定読込の絶対値 799.1 ms は固定ゲート 400 ms を超えるが、原因は未分類で、独立した項目 #236 として追跡する。
- C-27（#215）: #231 の受入条件を個別に検証し、PR #233 に記録した。MSVC の `/arch:AVX2` で `dotProductDecimateAvx2` の ymm 67・vfmadd 9 を確認し、スカラー経路に対して 1.68〜3.57 倍の性能を測った。命令生成に関する条件は満たしている。ただし性能はカーネル単体を 1 回測っただけで、#231 が求める方法（4 タップ、N ≥ 20、中央値と IQR、1 M サンプルあたり）には届いていない。このため #231 を再開し、追加の測定を行う。#215 は受入条件を満たしたためクローズしたままとする。
- C-22（MED-16）: 判定不能のまま独立して維持する。変更はない。
- C-39（LOW-08）: 条件付きのまま変更しない。

追記の根拠は `doc/work113/20261122-0720_最終総括/report.md` にまとめた。
