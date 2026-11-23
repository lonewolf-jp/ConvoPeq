# shutdown 設計 v2-a 改訂案（実装なし）

- 作業日時: 2026-11-13 16:30 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 種別: **設計改訂案の提出のみ**。ソース変更・ビルド・テストは行っていない。
- 前提: Owner 判断（2026-11-13）により v2-a の実装 GO は未付与。本書は未確定事項 7 点への回答である。

---

## 0. 改訂の対象と参照

| 資料 | 位置づけ |
| --- | --- |
| `doc/work113/20261113-0200_設計改訂/report.md` | v2 本体（状態モデル・A〜F・不変条件） |
| `doc/work113/20261113-0300_設計v2修正/report.md` | v2-a の実装単位 P1〜P8 と未解決 U-1〜U-8 |
| `doc/work113/20261113-0500_保持経路と予算/report.md` | U-5（保持経路）と停止予算（U-1・U-2） |
| `doc/work113/20261110-1418_プローブ計測/report.md` | 受入れ基準（`retireLatencyD1`）の原典 |

Owner 指定の 7 項目との対応:

| # | Owner 指定 | 本書の節 |
| --- | --- | --- |
| 1 | U-1（`bound` を 5 秒のままか、実測で決めるか） | §1 |
| 2 | U-2（`stopLearning()` の所要と停止順序） | §2 |
| 3 | U-5（audio-only teardown の副作用、保持／破棄の明確化） | §3 |
| 4 | U-6（stuck テレメトリの項目と記録方法） | §4 |
| 5 | P8（停止しなかった Worker の保持機構とライフタイム） | §5 |
| 6 | P1〜P8 の順序・完了条件・統合受入れ | §6 |
| 7 | 統計的受入れ基準の確定 | §7 |

U-3 との整合は別報告 `20261113-1630_U-3整合` に分離した。

---

## 1. U-1: `bound`（`B_pre`）の決め方

### 1.1 結論

**実測して決める。5 秒を「停止保証」として採用しない。**
ただし実装と値の確定を分離できるため、P3 の実装は `kPreflightWaitMaxMs`（名前付き定数）で行い、
値は計測後に差し替える。計測前に確定が必要なのは「待機の意味」だけである。

### 1.2 意味の確定（実装前に必要）

`B_pre` は **「待つ時間の上限」** であり、Worker が停止することの保証ではない。

- 超過した場合は停止できなかったものとして S3（保持）へ落ちる。
- したがって `B_pre` を大きくしても安全側に倒れるだけで、危険側には倒れない。
- 文書・ログ・報告で「5 秒で停止する」と書かない（既存の I-B4 を維持）。

### 1.3 計測計画（3 段階）

| # | 対象 | 方法 | 追加で必要な承認 |
| --- | --- | --- | --- |
| M-1 | W2（`coordinatorLoop_`）の停止時間 | 既存 `[DIAG] releaseResources` の前後に時刻を付ける（ログ行の追加のみ） | なし（ログ追加） |
| M-2 | W5・W6 の**走行中**停止時間 | 走行中を再現する計測専用 CLI スイッチ（`loadImpulseResponse` 強制／`enableProgressiveUpgrade` 一時有効） | **計測専用変更の承認** |
| M-3 | `stopLearning()` の join（U-2） | M-1 と同じ span 計測 | なし（ログ追加） |
| M-4 | W1（`rebuildThread`）の停止時間 | `stopRebuildThread` の前後（M-1 と同じ手法） | なし（ログ追加） |

M-1・M-3・M-4 は**ログに時刻を足すだけ**で取得できる。M-2 だけが「走行中を人工的に作る」ため承認が要る。

### 1.4 値の確定規則（計測後に適用）

```
B_pre = clamp( ceil(1.5 × p99(停止時間の全観測) ), 500 ms, 5000 ms )
```

- 下限 500 ms: 短すぎると正常時にも保持へ落ちる。
- 上限 5000 ms: 既存 `kGracefulDrainMaxMs` と同じ桁を超えない。
- p99 が取れない（観測 < 100 件）場合は `max` を使う。
- 計測前に実装する場合の初期値は 5000 ms（暫定であることをコメントに明記）。

### 1.5 確定条件

1. M-1・M-3・M-4 の観測が各 30 件以上ある。
2. M-2 が承認され、W5・W6 の走行中停止が各 10 件以上ある。
3. 上記から p99 を出し、§1.4 の式で値を決め、根拠（生データ）を報告に添える。

1 と 3 が揃えば W1・W2・W5・W6 の 4 つについては確定できる。M-2 が承認されない場合、
W5・W6 は「計測不能」として 5000 ms のまま据え置き、その旨を明記する。

---

## 2. U-2: `stopLearning()` の所要と停止順序

### 2.1 位置（既に確定）

`stopLearning()` は terminal pass（`ReleaseResources.cpp:146-150`）と reconfigure pass（`:818-822`）の
2 箇所で呼ばれる。保持経路（S3）では `releaseResources` が入口で return するため**呼ばれない**。
したがって U-2 は **通常経路 C の項**である。

### 2.2 順序への影響

現行の C は「learner join → coordinator stopThread → rebuild join → drain」の順で、
learner の join が長引くと **drain に入る前に** C が伸びる。drain の 5 秒上限はこの区間を守らない。

### 2.3 提案

1. M-3 で `stopLearning()` の join 時間を測る（30 件以上）。
2. p99 が 500 ms 未満なら現行順序を維持する（変更しない）。
3. p99 が 500 ms 以上なら、learner の停止を drain の**後**へ移す案を別途設計する
   （学習の保存と退役の順序が変わるため、独断では行わない）。
4. いずれの場合も「C は有界ではない」という I-B3 を維持する。

---

## 3. U-5: audio-only teardown の副作用（確定させる契約）

`20261113-0500` §3 の設計を土台に、今回**契約として固定**する。

### 3.1 実行する／しない（`~MainWindow` の 10 ステップ）

| step | 処理 | 保持経路 |
| --- | --- | --- |
| 0 | CLI ロガーを外す | 実行しない（保持中もログを残す） |
| 0b | `cliAutomationCallbacksEnabled = false` | 実行する |
| 1 | `removeChangeListener(this)` | 実行する |
| 2 | `setAdaptiveAutosaveCallback({})` | 実行する |
| 3 | `setCliProcessingTelemetryEnabled(false)` | 実行する |
| 4 | `audioProcessorPlayer.setProcessor(nullptr)` | 実行する |
| 5 | `stopTimer()`（500 ms） | 実行する |
| 6 | `DeviceSettings::saveSettings(...)` | **実行しない** |
| 7 | `removeAudioCallback` | 実行する |
| 8 | `closeAudioDevice()` | 実行する（無界。既存挙動） |
| 9 | `audioEngineProcessor.reset()` | **実行しない**（保持） |
| 10 | UI の reset | **実行しない**（保持） |

### 3.2 保持するもの／破棄するもの

- 破棄する: オーディオコールバック、デバイス、MainWindow のタイマ、ChangeListener、自動保存コールバック、CLI telemetry。
- 保持する: `AudioEngine`（値メンバ）、`audioEngineProcessor`、`audioDeviceManager`、`audioProcessorPlayer`、
  UI パネル、`settingsWindow`、`cliLogFileLogger`、停止しなかった Worker。

### 3.3 `saveSettings` を呼ばない理由の訂正（重要）

`20261113-0500` §3.5 は理由の 1 つに「`saveSettings` が +10.2 秒かかる」を挙げているが、
**その遅延は BUG-2 の修正（`computeMaxElementDepth` の O(n) 化）で解消済み**であり、現在の実測は約 31 ms である。
したがって理由は次の 1 点に絞る。

> 保持経路では engine が「解放済み」ではない中途状態にあり、そこから設定を永続化すると
> 次回起動時に不整合な構成（閉じたデバイス等）を読み込む可能性がある。

I-U5-1（保持経路で `saveSettings` を呼ばない）は維持する。理由だけを差し替える。

### 3.4 副作用（明示）

stuck 時はデバイス設定・IR 設定が永続化されない。次回起動は既定値または前回の保存値で立ち上がる。
これは受容済みの副作用であり、保持経路の目的（停止失敗時にプロセスを残す）と引き換えである。

---

## 4. U-6: stuck テレメトリの項目と記録方法（新規設計）

### 4.1 記録する項目

保持経路に入った時点で 1 行、key=value 形式で記録する。

| key | 型 | 内容 |
| --- | --- | --- |
| `retained` | int | 保持した `MainWindow` の件数（累積） |
| `retained_id` | uint64 | 保持した個体の識別子（`MainWindow` 生成時に採番） |
| `stuck_workers` | string | 停止できなかった Worker の集合（`W1,W3` 等） |
| `wait_ms` | int | 各 Worker の待機実測（`W1=812,W2=2001,...`） |
| `bound_ms` | int | 適用した `B_pre` |
| `reason` | string | `preflight_timeout` / `close_device` など |
| `sync_work_active` | bool | 保持時点で同期区間が有効だったか |
| `terminal_state` | string | `Closed` / `Closing` |
| `phase` | string | `ShutdownPhase` の値 |

### 4.2 記録方法

- 出力先は既存の `juce::Logger`（ファイルロガー）1 行。形式は既存の `[AUDIT]` / `[DIAG]` 接頭辞に合わせ
  `[SHUTDOWN] retained ...` とする。
- 集計用に `AudioEngine::shutdownTelemetry()`（値戻し・noexcept）を追加し、
  既存の evidence exporter から読めるようにするかは別判断（P8 の範囲外にしない）。
- 記録は Message Thread のみ。RT 経路からは書かない。

### 4.3 記録しないもの

- Worker のスタックや生ポインタのアドレス（診断に不要で、ログに残す価値が無い）。
- 保持した UI の内容。

---

## 5. P8: 保持機構と保持オブジェクトのライフタイム

### 5.1 機構

```
// MainApplication.cpp（関数ローカルの静的ポインタ。破棄しない）
static std::vector<MainWindow*>* g_retainedWindows = new std::vector<MainWindow*>();
```

- 保持は `mainWindow.release()` の後、上記へ `push_back` する。
- `unique_ptr` は空になるため、`MainApplication::shutdown()` の `mainWindow.reset()` で破棄されない。
- **静的オブジェクトにしない**（静的 `std::vector` は終了時に破棄され、`~MainWindow` が走る）。

### 5.2 ライフタイム

| 対象 | 寿命 | 破棄 |
| --- | --- | --- |
| 保持した `MainWindow` | プロセス終了まで | 破棄しない（意図的リーク） |
| `g_retainedWindows` 本体 | プロセス終了まで | 破棄しない |
| 停止しなかった Worker | プロセス終了まで | 破棄しない（H-3） |

### 5.3 成立条件（既存 H-1〜H-4 の再掲）

1. `MainWindow` はヒープに残し、デストラクタを実行しない。
2. `AudioEngine`（値メンバ）も破棄しない（`~AudioEngine` の join を走らせない）。
3. joinable な `std::thread` のデストラクタを実行しない（`std::terminate` を避ける）。
4. 保持ポインタを静的破棄しない。

### 5.4 受容する副作用

- Debug のリーク検出器が `MainWindow` 1 件を報告する（S-1）。意図的な保持として許容する。
- `AudioEngine` の 100 ms タイマは止まらない（`juce::Timer` を private 継承しており外部から停止できない）。
  保持中も `driveRetireReclaimNonRt` 等が動く。RT 経路は停止済みなので音は出ない。

---

## 6. P1〜P8 の順序・完了条件・統合受入れ

| # | 単位 | 完了条件 | 検証 |
| --- | --- | --- | --- |
| P1 | 状態機械（`terminalStopRequested_`、`isStopAcceptingWork()`） | S0→S1 の遷移が MT のみで起き、述語が既存 40 箇所の判定時期を変えないこと | 単体テスト（述語の真理値表）、既存テスト 61 件 |
| P2 | W1 状態（`rebuildThreadLoopExited`） | ループ末尾で true、起動点で false。`join` 前に true を観測できること | 単体テスト＋W1 停止の実測（M-4） |
| P3 | preflight（`requestWorkerStop` / `waitForWorkersToExit`） | 要求が `stopThread` / `join` を呼ばないこと（I-A3）、待機が単一 deadline を共有すること | 単体テスト＋停止時間の実測（M-1・M-2） |
| P4 | producer 拒否（`loadIR` / `loadImpulseResponse` / `startProgressiveUpgrade` / `prepareToPlay`） | S1 以降に開始しないこと（I-A1） | 単体テスト（各入口で拒否） |
| P5 | 非破壊分岐（`releaseResources` の terminal / reconfigure 入口） | `workersStuck_` が true の間は何も破棄しないこと（I-B1） | 単体テスト＋保持経路の実走 |
| P6 | callback 抑止（C2〜C7 の 6 箇所） | epoch 不一致または StopAccepting で本体を実行しないこと（I-C1）。抑止しても解放は行う（I-C2） | 単体テスト |
| P7 | 同期区間（`SyncWorkToken` と 4 区間） | 同期区間内でメッセージループを回さないこと（I-E1）、`bound` を適用しないこと（I-E2） | 単体テスト |
| P8 | 保持（`StuckProcessHolder` と audio-only teardown） | 保持後に `~MainWindow` / `~AudioEngine` が走らないこと（I-F2）、`saveSettings` を呼ばないこと（I-U5-1） | 保持経路の実走＋ログ検証 |

順序は P1 → P2 → (P3, P4, P5, P6, P7) → P8。P3 と P5 は独立に検証できる。

**統合受入れ（全単位完了後）**

1. 既存 `ctest` 61 件が全件成功する。
2. 通常終了で保持経路に入らないこと（`retained` が 0 のまま終了する）。
3. 通常終了の起動・終了時間が現行と同等であること（起動 1.8〜2.0 秒、teardown 2.2〜2.5 秒）。
4. §7 の統計基準を満たすこと。
5. 保持経路を人工的に発火させた試験（別途承認）で、プロセスが終了まで到達し、
   `std::terminate` も `~MainWindow` も発生しないこと。

---

## 7. `retireLatencyD1` の統計的受入れ基準（確定案）

原典は `20261110-1418` §3.4〜§3.5。今回、運用上の扱いを次で確定させる。

### 7.1 判定式

```
V = { 要求 r | 完了イベントがちょうど 1 件、completeTs >= requestTs }
N_valid = |V|
pass 条件: N_valid >= 59 かつ max(L_i) <= 50 ms かつ orphans == 0 かつ 未完了要求 == 0
```

- `drainAll`（終了時の強制解放）は別集計とし `N_valid` に含めない。
- `insufficient`（件数不足・未完了・orphan あり）は **pass に読み替えない**。
- 50 ms は変更しない。

### 7.2 測定回数と環境

- 1 回の実行で 59 件を満たさない場合、満たすまで実行を繰り返す（合算しない）。
- **負荷条件を変えた 2 群**で、それぞれ `N_valid >= 59` を満たすことを要件とする。
  1. 通常（IR を読み込まない）
  2. IR リロード併用（`--cli-ir` 相当）
- 環境変動が大きいことは既知である（同一関数で 9.9〜13.0 秒の幅を観測した記録がある）。
  そのため 1 回の `max` で判定し、平均では判定しない。

### 7.3 独立性の留保

59 件は「超過率 5% 未満を 95% 信頼で示す」ための下限であり、観測が独立同分布であることを前提にする。
実際の退役は負荷・epoch・スケジューリングで相関するため、**実効の信頼度は 95% より低い**。
この留保を報告に必ず併記する。

---

## 8. 未確定のまま残る事項

| # | 事項 | 状態 |
| --- | --- | --- |
| R-1 | `B_pre` の値 | 計測待ち（§1）。実装は定数で先行可能 |
| R-2 | W5・W6 の走行中停止時間 | 計測専用スイッチの承認待ち（M-2） |
| R-3 | `stopLearning()` の p99 と、それに応じた順序変更の要否 | 計測待ち（§2） |
| R-4 | 保持経路の人工発火試験 | 計画の承認待ち（U-3 側 §5 と共通） |
| R-5 | `closeAudioDevice()` の無界性 | 既存挙動として受容（U-13） |

## 9. 停止条件の確認

- production source / test / CMake を本書のために変更していない。
- build / test / debugger / replay を本書のために実行していない。
- git の変更操作を行っていない。HEAD は `173e6995`。
- U-3 の実装には着手していない（整合整理のみ、別報告）。
- 本書は設計改訂案であり、実装 GO を意味しない。
