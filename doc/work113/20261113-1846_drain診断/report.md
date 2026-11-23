# drain タイムアウト診断ログの実装と計測

- 作業日時: 2026-11-13 18:46 JST（本機の時計。直前の報告フォルダ `20261113-1940_drain診断計画` を作成した後に本機の時計が戻っているため、フォルダ名の並びは前後する）
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 種別: 承認済み範囲（診断ログ 2 行 + getter 1 個）の実装と計測。**原因修正は行っていない。**
- 承認条件: 1 shutdown あたり最大 2 行、9 条件は `isFullyDrained()` の実評価と一致、近似値禁止、ログ追加以外の動作変更禁止。

---

## 1. 調査対象の識別情報

| 項目 | 値 |
| --- | --- |
| 実行ファイル | `build/ConvoPeq_artefacts/Release/ConvoPeq.exe`（48,031,744 B、SHA-256 先頭 16 桁 `3eac0b32a581a42a`、2026-11-13 19:10:18） |
| 計測時のバイナリ | 上記の 1 つ前のビルド（18:32:15）。差分はコメント内の行番号参照 1 箇所だけで、機械語は同一。再ビルド後の追試 1 回でも同じ値を確認（§4.1） |
| 実行引数 | `--cli-run --cli-exit-ms 2000 --verify-runtime-evidence-live` |
| 作業ディレクトリ | `%TEMP%\opencode\drain_diag_run\run1`〜`run3`（リポジトリの `evidence/` を汚さないため） |
| ログ | `<exe フォルダ>/ConvoPeq.log`（`juce::FileLogger`、追記） |
| 証跡 | 各 run の `evidence/`（`shutdown_trace.json` ほか 27 ファイル） |
| 対象コード | `AudioEngine::isFullyDrained()`（`AudioEngine.Threading.cpp:243-303`）、`AudioEngine::waitForDrain()`（同 `:305-337`）、`AudioEngine::releaseResources()`（`AudioEngine.Processing.ReleaseResources.cpp`） |

実行環境: Windows 11 x64、MSVC 14.52.37025、CMake 4.4、Ninja Multi-Config、Release。

## 2. 実装（承認された 3 箇所）

差分は**追加のみ**（削除 0 行）。`ReleaseResources.cpp` の差分ハンクは 5 つあるが、うち 3 つは
本作業より前から作業ツリーにあった停止計測用の変更（`ConvolverStopProbe.h` の include と
`probeStopLearning` 2 箇所）である。今回追加したのは残り 2 ハンクと、ヘッダの getter 1 個だけ。

| # | 場所 | 内容 | 追加行 |
| --- | --- | --- | --- |
| 1 | `AudioEngine.h:1652-1660` | `pendingReclaimHandleCount()`（`pendingReclaimHandlesMutex_` で保護して `size()` を返すのみ、本体は `:1655-1659`） | 9 |
| 2 | `ReleaseResources.cpp:586-605` | clear→retire 直後の診断 1 行（`diagLog` は `:598`） | 20 |
| 3 | `ReleaseResources.cpp:691-745` | タイムアウト時の診断 1 行（`diagLog` は `:728`） | 55 |

変更していないこと（確認済み）:

- `waitForDrain(2000, 2)` は `ReleaseResources.cpp:673` にそのまま。タイムアウト値・ポーリング間隔は不変。
- `collectDrainAudit()` の呼出し位置・回数は不変（タイムアウト分岐の既存 1 回を再利用）。
- drain / reclaim / epoch 前進 / retire の呼出しは追加していない。処理順序・同期・ロック範囲も不変。
- RT 経路は触れていない（`releaseResources()` は非 RT）。

### 2.1 ログ書式

```
[DRAIN_DIAG] worldClear cleared=1 retireResult=void publishedAudit=4 retiredAudit=3 activeAudit=1 terminalAfterRetire=0
[DRAIN_DIAG] timedOut=1 domain=EngineISR failed=c3,c9 c1_deferredPublish=0 c2_pendingReclaim=0 c3_retireDepth=1 c4_lifetimeIntent=0 c5_overflowRing=0 c6_dspQuarantine=0 c7_retireQuarantine=0 c8_terminalReclaim=0 c9_bridgeDrained=0 readers=0 reclaimAttempts=87378 reclaimSuccess=6 fallbackDepth=0 published=4 retired=3 activeWorld=1
```

`worldClear` の各値が示すもの（混同しないよう区別した）:

| 値 | 意味 |
| --- | --- |
| `cleared` | `clearPublishedRuntimeSnapshotsNonRt()` が引き渡し対象 World を返したか（1/0）。**retire の成否ではない** |
| `retireResult=void` | 定数。`retirePublishedRuntimeWorldNonRt()` は `void` を返すため、retire 成功/失敗はこの経路では判定できない |
| `publishedAudit` / `retiredAudit` | `WorldLifecycleAudit` の**起動からの累計** publish / retire 件数 |
| `activeAudit` | 同 audit の**現在** active World 件数 |
| `terminalAfterRetire` | clear→retire 直後に `TerminalReclaimAuthority` が保持している件数。直後の `drainAllQuarantineStore()` より**前**の値 |

## 3. 9 条件の対応（近似なし）

| # | `isFullyDrained()` の実条件 | ログの取得元 | 一致の根拠 |
| --- | --- | --- | --- |
| 1 | `!hasDeferredCommit`（`runtimeOrchestrator_->hasDeferredRequest()`） | `audit.deferredPublish` | `collectDrainAudit()` が同じ式で `?1:0` を作っている |
| 2 | `pendingReclaimHandles_.empty()` | `c2_pendingReclaim`（新 getter、同一 mutex） | 同じ mutex・同じ `size()` |
| 3 | `m_retireRouter->pendingRetireCount() == 0` | `c3_retireDepth`（直接呼出し） | **`audit.routerPendingRetire` は `fallbackQueueDepth_` を含む合算なので使っていない** |
| 4 | `worldAuthority_.lifetime().pendingIntentCount() == 0` | `audit.pendingRetire` | 同じ式 |
| 5 | overflow ring の `residentCount() == 0` | `audit.overflowRingResident` | 同じ式（null 時 0 も同一） |
| 6 | `dspQuarantineManager_.residentCount() == 0` | `audit.quarantineResident` | 同じ式 |
| 7 | `m_retireRouter->quarantineResidentCount() == 0` | `c7_retireQuarantine`（直接） | audit に無いため直接 |
| 8 | `m_retireRouter->terminalReclaimResidentCount() == 0` | `c8_terminalReclaim`（直接） | 同上 |
| 9 | `runtimePublicationBridge_.isFullyDrained()` | `c9_bridgeDrained`（直接） | 同じ関数 |

`failed=` は 9 条件のうち**満たしていない条件の番号**を列挙したもの。空なら `none`。

## 4. 計測結果（3 回）

| 実行 | exit | 総経過 | `DRAIN_DIAG` 行数 | タイムアウト行 | 偽の条件 | `c3_retireDepth` | `c9_bridgeDrained` | その他 7 条件 | `reclaimAttempts` / `Success` |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| 1 | 0 | 6.58 s | 2 | 1 | `c3,c9` | 1 | 0 | すべて 0 | 87,378 / 6 |
| 2 | 0 | 5.99 s | 2 | 1 | `c3,c9` | 1 | 0 | すべて 0 | 88,826 / 6 |
| 3 | 0 | 6.01 s | 2 | 1 | `c3,c9` | 1 | 0 | すべて 0 | 87,520 / 6 |

3 回とも同じ値で再現した。1 shutdown あたり 2 行（`worldClear` 1 行 + タイムアウト 1 行）で、
ポーリングごとの出力は無い。

### 4.1 再ビルド後の追試（1 回）

計測後にコメント内の行番号参照を 1 箇所だけ直したため、アプリを再ビルドして 1 回追試した。
機械語は変わらない。結果は `exit=0`、6.69 s、`DRAIN_DIAG` 2 行、タイムアウト 1 行で、
`failed=c3,c9`・`c3_retireDepth=1`・`c9_bridgeDrained=0`・その他 7 条件すべて 0 と、
3 回の計測と同一だった（`reclaimAttempts` だけは経過時間依存で 113,740）。

証跡との一致:

| 値 | ログ | 証跡 |
| --- | --- | --- |
| 残存 retire | `c3_retireDepth=1`（タイムアウト時点） | `shutdown_trace.json` の `sh3_pendingRetire=2`（**フォールバック後**の値） |
| Reader | `readers=0` | `reader_registry_final.json` の `activeAtStage: []` |
| 完了 phase | — | `phase=10` / `phaseName=ShutdownComplete` |
| World | `published=4 retired=3 activeWorld=1` | `world_retirement_telemetry.json` の `acquireObserved=4 / releaseObserved=3 / outstanding=1` |

## 5. 偽になった条件と、その意味

### 5.1 `c3`（EngineISR ドメインの `pendingRetireCount() != 0`）

- タイムアウトした瞬間に **1 件**が EngineISR ドメインの退役キューに残っていた。
- ドメインは `m_epochDomain`（`AudioEngine.h:5456` で `RetireDomainId::EngineISR`）で確定。
  `pendingRetireCount()` はこの provider へ委譲されるため、残存は EngineISR のもの。EQ ドメインではない。
- **1 件でも `isFullyDrained()` は偽になる**ため、c3 だけでタイムアウトは説明できる。
- この 1 件が 2 秒間回収されなかった理由（epoch が安全にならない・先頭で詰まっている等）は
  **未確定**。回収試行は 8.7 万回、成功 6 件で、先頭が進まない状態だった。

### 5.2 `c9`（`runtimePublicationBridge_.isFullyDrained() == false`）

- 9 条件目の内側は **14 条件**（`swapPending_` の早期 return ＋ 13 個の `&&`、
  `ISRRuntimePublicationCoordinator.cpp:521-572`）。当初「11 個」と書いたのは数え違いで、
  A/C 計画の §0 で訂正した。今回のログは外側の真偽だけを記録しており、**どの内訳が偽かは
  未確認**（内訳の個別出力は承認範囲外）。
- 既存の監査行 `[ISR][Shutdown] Drain incomplete: pendingPub=0 …` から、
  `publicationBacklogCount_ == 0` は満たしている。残る候補は
  `swapPending_`、4 本の transport キュー、`publicationIntentResidencyCount_`、
  `pendingIntentCount_`、`reclaimInFlightCount_`、`quarantineIntent/QuarantineRingResidency`、
  `recoveryAdmissionPending_`、`liveLogicalRecoveryObligationCount()`。
- **推定（未確定）**: c3 の 1 件に対応する reclaim が in-flight のまま、あるいは同じ滞留を
  別カウンタで見ている可能性がある。ただし c9 が独立に偽である可能性を否定できない。

### 5.3 c3 が 1 → 2 に増える点

タイムアウト時点は 1、フォールバック後の監査行と `sh3_pendingRetire` は 2。増加の出所は
**追加調査で確定した**: 2 秒待ちの後に走る `m_coordinator.finalizeShutdown()`
（`SnapshotCoordinator::retireCurrentAndTarget`）が `publishEpoch()` で epoch を 1 進めてから
`GlobalSnapshot` を 1 件 enqueue するためである。フォールバック区間の
`drainDeferredRetireQueues(true)` は enqueue しない（`doc/work113/20261113-2031_c3c9設計検討/`）。

## 6. clear→retire 経路

| 項目 | 実測 | 読み方 |
| --- | --- | --- |
| `cleared` | 1 | clear は World を 1 件返した（引き渡し対象が存在した） |
| `retireResult` | `void` | retire の成否は**判定できない**（戻り値が無い） |
| `terminalAfterRetire` | 0 | 直後に Terminal へは退避していない（deferred queue 側と推定） |
| `publishedAudit` / `retiredAudit` / `activeAudit` | 4 / 3 / 1 | 起動から 4 件 publish、3 件 retire、1 件が未 retire |

**前回報告の訂正**: 前回は `world_lifecycle_audit.json` の `publishedCount=1 / retiredCount=0` を
根拠に「clear→retire が効いていない可能性」を挙げた。今回、この JSON は
**起動直後の定期ダンプ（60 秒間隔）のままで古い**と分かった。終了時点のライブ値は 4/3/1 で、
`world_retirement_telemetry.json`（4/3/1）とも一致する。したがって retire 経路は 3 件について
機能しており、未 retire は 1 件だけである。前回の候補 2（経路が通っていない）は**否定**される。

## 7. タイムアウト・フォールバックへの影響

| 確認項目 | 結果 |
| --- | --- |
| タイムアウトの発生 | 3/3 で発生（`drain timeout reached, performing safe tryReclaim (drainAll skipped)`） |
| フォールバックの実行 | 3/3 で実行（同区間で reclaim 試行が 8.7 万回まで増加、`reclaimSuccess=6`） |
| 終了コード | 3/3 で 0、クラッシュ・ハングなし |
| 2,000 ms の維持 | 変更なし（`waitForDrain(2000, 2)`） |
| ログ量 | 1 shutdown あたり 2 行（上限内）。ポーリングごとの出力なし |
| 総経過時間 | 変更前 6.26〜6.57 s（既存計測）→ 変更後 5.99〜6.69 s（4 回）。増加なし（ノイズ範囲） |

## 8. ビルド・テスト

| 項目 | 結果 |
| --- | --- |
| 構成 | `cmake -S . -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCONVOPEQ_PGO_INSTRUMENT=OFF -DCONVOPEQ_PGO_USE=OFF` → 成功 |
| ビルド | `cmake --build build --config Release` → 784 ターゲット、エラー 0。再実行で `ninja: no work to do` |
| バイナリ確認 | exe 内に `DRAIN_DIAG` 2 件・`worldClear`・`c9_bridgeDrained`・`terminalAfterRetire` の文字列が存在 |
| ctest | `ctest --test-dir build -C Release` → **61/61 成功**（379.65 s） |
| 追加ビルド | コメント修正後に `cmake --build build --config Release --target ConvoPeq` → 成功（警告 2 件は `MklFftEvaluator.h` の既存 C4390） |

ビルド環境の注意（今回踏んだ点）: `build.bat` は MSVC 環境を自前で初期化しない。また本機では
PATH 上で GNU Octave の `mingw64\bin` が Windows SDK より先に来るため、CMake が `link.exe` の
代わりに Octave の `ld.exe` を検出し、`juceaide` のサブビルドが失敗する。今回は PATH から
Octave 系を外し、`vcvars64.bat` の後に `MKLROOT` / `IPPROOT` を設定して構成した。あわせて
`build/JUCE/tools/` のサブキャッシュを削除した（`build.bat` と同じ手順）。この環境作業は
リポジトリのファイルを変更していない。

今回の実測知見（タイムアウト時に偽になる条件、`world_lifecycle_audit.json` が起動直後の
ダンプのままで古いこと、Octave の mingw64 が PATH にあると CMake 構成が壊れること）は
`AGENTS.md` の末尾に 2 節として追記した。

## 9. 未実施・未確認

| # | 内容 | 理由 |
| --- | --- | --- |
| 1 | c9 の内訳 11 条件の個別値 | 承認された 9 条件の外。個別出力は別承認が必要 |
| 2 | 残存 1 件の「中身」（型・サイズ・epoch） | キュー走査 API が無い。別承認が必要 |
| 3 | 1 件が 2 秒間回収されない理由の確定 | 上記 1・2 が前提 |
| 4 | ロールバックの実ビルド | 診断を残すか削除するかは Owner 判断待ちのため、削除ビルドは未実施 |
| 5 | Debug 構成での確認 | 今回は Release のみ（既存の計測と同じ構成） |

## 10. 残留リスク

- 診断ログは**読み取りのみ**だが、`pendingReclaimHandleCount()` が短時間 mutex を取る。
  非 RT 経路の終了時 1 回であり、実測でも終了時間に影響は出ていない。
- タイムアウトは 3/3 で再現しており、**フォールバックが常態**という前回の評価は変わらない。
  機能的には安全に終了するが、退役キューの 1 件と未 retire の World 1 件はプロセス終了まで
  解放されない（両者が同一物かは未確定）。
- c9 の内訳が未確定のため、「c3 を直せば drain が成立する」とは言えない。c9 が独立に偽なら、
  c3 を解消してもタイムアウトは残る。

## 11. ロールバック

追加した 3 箇所（`AudioEngine.h:1652-1660` の getter、`ReleaseResources.cpp:586-605`、
同 `:691-745`）を削除するだけで元に戻る。制御フローの書き換えが無いため、他への波及はない。

## 12. 判定

- 承認された範囲の実装と計測は**完了**。1 shutdown あたり 2 行、9 条件は実評価と一致、近似なし。
- タイムアウト時に**偽になる条件は c3 と c9 の 2 つ**で、3 回とも同一だった。c3 だけで
  `isFullyDrained()` は偽になる。
- `sh3_pendingRetire` の残存 2 件が EngineISR ドメインであることはコード上で確定済みで、
  今回の実測とも矛盾しない。
- ただし「その 1 件が 2 秒で回収されない理由」と「c9 の内訳」は未確定であり、
  **原因修正の設計には追加の計測が必要**。次節に案を示す。

## 13. 次の計測案（未実施・承認待ち）

原因修正（コード変更）は今回の承認範囲外なので、まず確定に必要な計測だけを案として示す。

| 案 | 内容 | 変更規模 | 分かること | 承認 |
| --- | --- | --- | --- | --- |
| A | c9 の内訳 14 条件を、同じタイムアウト分岐で 1 行に出す（`swapPending_`・4 キュー・residency 系・`reclaimInFlightCount_`・`recoveryAdmissionPending_`・live obligation） | ログ 1 行の追加（既存 getter のみで足りるか要確認） | c9 が独立の原因か、c3 と同じ滞留の別表現か | 必要 |
| B | 退役キュー先頭 1 件の属性（epoch・`DeletionEntryType`・`domainId`）を読む | キューに読み取り専用の先頭参照 API を 1 つ追加 | 回収されない理由が epoch なのか型なのか | 必要（API 追加） |
| C | タイムアウト直前の `minReaderEpoch` と先頭 entry の epoch の比較を 1 行で出す | ログ 1 行（既存値の比較のみ） | `isOlder` が成立していないことの直接確認 | 必要 |

A と C はログ追加だけで済む見込みで、B は API 追加を伴う。いずれも実施前に変更箇所と副作用を
提示する。原因修正（epoch 前進の駆動、reclaim 条件、c9 内訳の解消）は、A〜C の結果を見てから
別途設計する。タイムアウト値 2,000 ms は変更しない。
