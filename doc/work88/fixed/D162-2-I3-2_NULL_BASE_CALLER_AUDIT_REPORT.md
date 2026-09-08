# D162-2-I3-2 Release Null-Base Admission CAS Caller Audit Report

```text
Date:     2026-09-05
Type:     read-only structural/root-cause audit（production/test/CMake 変更 0）
Evidence: evidence/D162-2I2/I3_2_NULL_BASE_CALLER_AUDIT.md + i3_2_cvparse.py
判定:     **NO-GO（repair contract には producer 未確定）— ただし faulting statement と
          rdx identity は確定した。次手 = data-breakpoint による writer 特定**
```

---

## 1. faulting instruction → C++ statement（確定）

- crash 命令（RVA 0x1F7F89D・`mov byte [rdx+0x65], 2`、rdx=0）は
  **`RuntimeIntentCoordinator::commit`（ISRRuntimePublicationCoordinator.cpp:75-121）の
  source line 109** にマップされた。
  - 根拠: Release obj（BIGOBJ）の COMDAT section 746 に付属する .debug$S line fragment
    （offset→line: 0→82, 7→83, 26→93, 65→94, 72→95, 79→97, 100→101, 161→104,
      **173→109**, 177→110, 209→117, 227→119, 231→120, 236→121）。
  - 検証: 書き込み値 **2 == CoordinatorState::Publishing**（h:119-127 の enum）、隣接
    byte write **1 == swapPending_=true**（:110）、3 qword write **[r9+0x198/0x1A0/0x1A8]
    = line 119 の PublicationSemantic{sequenceId, epoch, mappedGeneration}**、
    同 base に **Faulted(6) write**（line 83/104 の兄弟 path）— すべて整合。

- **faulting C++ statement**:
  `convo::publishAtomic(state_, CoordinatorState::Publishing, std::memory_order_release);`
  （RuntimeIntentCoordinator::commit 内・ISRRuntimePublicationCoordinator.cpp:109）

## 2. rdx の object identity（確定）

- **rdx = `this`（RuntimeIntentCoordinator\*）= NULL**。
  - `mov rdx,rcx`（arg1 = this）以降、crash まで rdx 再代入なし（全命令走査）。
  - 書き込み先 this+0x65 = `state_`、this+0x64 = `swapPending_`（coordinator member pair）。
- **Case B（MSVC STL lock-pool base null）は棄択** — lock-pool spinlock は別関数
  （0x...F410・無関係）。Case A の形で確定: **commit が null coordinator で呼ばれた**。
- Case C（破壊済み object へのアクセス）/ Case D（zeroed object）は「producer 未確定」
  として残存（§5）。

## 3. caller chain（確定分）

```text
CoordinatorLoop worker thread
  → runCoordinatorPhase (Threading.cpp:293)
    → runtimePublicationBridge_.processIntent(*this, lifetimeMgr)  (:298)
      → PublishIntentHandler::handle
        → PublishExecutor{}.executePublish(ctx.engine.worldAuthority(), intent, ctx)
                                                   (ProcessIntent.cpp:149)
          → authority.publish(owner, metadata, &committed)      (RuntimePublishExecutor.h:60)
            → coordinator_.commit(Granted, boundary, newWorld, version, seq, epoch, gen, prevWorld)
                                                     (RuntimeWorldAuthority.h:299)
              → [this = NULL] → line 109 で AV（WRITE @ 0x65）
```

- `coordinator_` は **RuntimeWorldAuthority の reference member**（:333）で、
  AudioEngine ctor（CtorDtor.cpp:29 `worldAuthority_(runtimePublicationBridge_)`）により
  member subobject `runtimePublicationBridge_`（h:4964）に束縛される。
- **reference member が正当に null になる経路は存在しない** → 世界Authority オブジェクトの
  coordinator_ フィールド（8 バイト）が**構築後に 0 で上書きされた**ことが必然帰結。
- crash thread は CoordinatorLoop worker（新規 tid・main frame なし・ucrtbase thread-start frame あり）。
  **bootstrap commit（main thread・loop 起動前）は成功済み** → coordinator_ フィールドは
  「bootstrap 成功後〜 worker 最初の publish まで」の窓で 0 化された（時間窓確定）。

## 4. recoveryAdmissions_ 全 call-site inventory（29 site）

全アクセスを列挙した結果、**recoveryAdmissions_ 経由の OOB（i ≥ 32）は source 上存在しない**
（全ループ `i < kCapacity`・tryInsert capacity guard・resolve は検索結果使用 — 詳細表は
evidence 版 §3）。**crash 命令は recoveryAdmissions_ ではなく commit の `state_` publish** —
**I3-1 記録の「RecoveryAdmissionTable 系コード」表記を本監査で訂正**。

## 5. layout 上の隣接性（writer 候補の絞り込み）

AudioEngine member 宣言順（h:4964-4966）:

```text
runtimePublicationBridge_  ← RuntimeIntentCoordinator
    （tail: recoveryAdmissions_(32×320=10KB) → intentQueue_(MpscBoundedRing<Intent,4096>)
      → nextIntentId_ → quarantineFallbackQueue_(1024×Intent) → quarantineFallbackDropCount_
      → overflowAgeWarnCallback_ → quarantineService_）
worldAuthority_            ← RuntimeWorldAuthority { coordinator_ (reference, 8B) ← 第 1 member }
```

- **bridge のクラス終端の直後に worldAuthority_.coordinator_（8 バイト）が配置される**
  （Alignment 8・中間 padding は最小）。
- bridge tail member（quarantineService_ 等の内部 buffer）からの **8〜16 バイトの
  超過書き込み / 0 埋めが正確に coordinator_ を潰す位置関係**。
- 書かれた値が **0（NULL）** である点 — garbage ではなく綺麗な 0 の場合、writer は
  「0 を書くコード」（atomic store(0)・zero-init object の copy・memset 有限長等）。

## 6. Release-only 条件（仮説分離）

| 仮説 | 状態 |
| --- | --- |
| 1 UB 顕在化 | 部分維持 — writer 自体が UB の可能性（確定 evidence は writer 特定後） |
| 2 data race（timing で window 顕在化） | 有望 — 発生 window は確定済み（§3 最終行） |
| 3 destruction ordering | 低 — engine 構築直後（0.07s）で破棄未発生 |
| 4 config/build 差 | 低 — member layout は config 非依存・RWDI は同一最適化系で PASS |
| 5 stale object | 低 — engine 1 個体・再構築なし |

## 7. I2 independence

I2（PrepareToPlay.cpp）は commit 呼出経路と無関係。Release harness crash は G2 世代から
存在（0xc0000374・旧 suite）。**I2-independent 確定。I2 closure は変更禁止のまま遵守。**

## 8. root cause 確定度と GO/NO-GO

```text
確定: faulting C++ statement（line 109）・rdx = null coordinator `this`・
      AV 種別（WRITE @ state_）・caller（WriteAccess::publish・CoordinatorLoop worker）・
      null の所在（worldAuthority_.coordinator_ reference field = 0）・発生時間窓
未確定: coordinator_ フィールドを 0 化した writer（code / buffer）
```

**判定 = NO-GO（I3-3 repair contract へは進めない）** — 指示の GO 条件「why null」
（= どの code が coordinator_ フィールドを 0 化したか）が未確定のため。
「〜っぽい」の段階（bridge tail overflow 仮説が最有力だが未証明）での修正は禁止どおり実施しない。

### 次の手順（I3-2 の延長・要 instrumentation 権限）

1. x64dbg hardware breakpoint（DR0）を `&worldAuthority_.coordinator_` 実アドレスに設定し
   WRITE を 1 回捕獲 → writer の RIP + call stack が直接得られる
   （engine instance address は commit entry bp での rcx から 1 ステップで特定可能）。
2. または page-heap（gflags /admin）+ WER full dump。
3. writer 特定後 → I3-3 repair contract（境界修復 or lifetime 修正）。

## 9. 修正禁止事項（遵守確認）

null guard / assert 追加 / release idempotent 化 / admission reopen / I2 closure 変更 /
production への `_exit`・`abandon` 導入 — **いずれも未実施**。本監査の生成物は
解析ツール（i3_2_cvparse.py）と本記録のみ。
