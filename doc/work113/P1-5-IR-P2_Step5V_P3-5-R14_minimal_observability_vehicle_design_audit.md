# P1-5-IR-P2 — Step 5-V / P3-5-R14: Minimal Observability Vehicle Design Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R14）
- **種別**: read-only design audit。実装・build・run なし。
- **目的**: U6-post／U4／build-failure cluster／commit/admission の境界を
  最小の観測点数で切れるか確定する（R14-A/B/C）。
- **結論**: **R14-C**（§10）。take 観測に production accessor が要る。
  実装はせず、R7–R10 形の設計監査へ戻す。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
F vehicle＋R1 diagnostic 保持／R10 accessor 実装済み（未実行利用）
production／CMake／JUCE 0 diff（P1 TU・AudioEngine.h の承認済み分を除く）
```

## 2. 観測軸の限定（A／B／C・維持）

- Axis A（queue→consume）：take 境界の観測が核心（§3）。
- Axis B（build outcome）：take 確定後の staged 条件付き（§5）。
- Axis C（commit／admission）：commit-attempt 確定後の staged 条件付き（§6）。
- R10 accessor・T6／T9 は変更しない（active 側は確定済みのため）。

## 3. 単一観測点の判定（take boundary）

理想形 `queuedCount Δ／takenCount Δ` のうち後者が既存に存在しない：

```text
queued： debugRebuildDispatchQueuedCount（:764・live・public差分可）
taken：  対応 counter なし。
  DrainedCommand 系は dead field（writer なし・R13 確定）。
  rebuildBacklog_／hasPendingTask／pendingTask は private かつ getter なし。
  Timer 監視は read のみ（Timer.cpp:596-643）。
  R1 vehicle log の prefix inventory（P1CHAR／IR_*／L0_WRITE のみ）は
  worker 側既存診断の vehicle 到達ゼロを実証する。
```

よって **production 変更なしでは take 観測は不可能**と確定する。
（C++ access 制御は絶対であり、test-only friend も production header 変更である。）

## 4. 候補比較（A–E）

| Candidate | 内容 | production変更 | 判定 |
| --- | --- | --- | --- |
| A 既存dispatch counter | request／queued／duplicate／reject 差分 | 0 | baseline（pre-queue まで。take 以降は不可） |
| B test-only friend／access | private 到達 | 要（friend／getter は production 変更） | 純 test-only としては IMPOSSIBLE。C-track へ |
| C worker既存診断再利用 | diagLog／D129／[CONV_STATUS]／Timer | 0（効果なし） | vehicle-silent のため不足。Drained 復活は禁止 |
| D 新規production counter | take 点等 | あり | 後回し（R14-C の設計監査対象） |
| E logger追加 | — | あり | 不採用候補 |

## 5. U4 分離（staged・条件付き）

- take 観測後に `takeあり＋publishなし` が出た場合のみ、build-result 点を追加検討する。
  現時点では take 点のみを要求し、build-result 点は commit しない
  （最小 N の段階化・§9）。
- `catch` 内部への logger 追加方向には進まない（指示どおり）。

## 6. U1 後回し（順序制約の固定）

- health／pressure の instrument は commit-attempt 確立後。
  現時点では commit 側入力リスト（sealed／generation／health ref／throttle flag／
  fading uuid・R13 確定）の確認に留める。

## 7. R10／T6／T9 不変（混同禁止の維持）

- `getActiveRuntimeSnapshot()`・T6／T9・ab／aa schema を変更しない。
- R14 の対象は rebuild pipeline 境界のみであり、active 観測の強化ではない。

## 8. F／R 将来 vehicle（固定・未実行）

```text
F: fresh → g0/os1/am=-20 → queue → consume? → build? → commit? → publish? → capture
R: fresh → g0/os1/am=-6  → queue → consume? → build? → commit? → publish? → capture
```

- 30s wait／sleepPump(800)／500ms post-settle は不変。
- order 以外の差異なし。F/R 実行は本 Step では行わない。

## 9. 最終設計表＋最小 N

| State | 既存観測 | 最小追加観測 | U分離 |
| --- | --- | --- | --- |
| request | requestCount Δ | なし | — |
| queued | queuedCount Δ（＋duplicate／reject Δ） | なし | U6-pre 確定 |
| worker take | 不可（§3） | **take 点 ×1（production accessor 要）** | U6 境界 |
| build start／result | 不可 | 条件付き（take後残余時のみ） | U4／U5 |
| commit attempt | 不可（drop は lifecycle 差分で事後検出可） | 条件付き（同上） | U1 |
| publish | sequence Δ | 既存 | — |
| active world | R10 accessor | 既存 | — |

- 最小 N＝1（take 点）。全行への追加はしない（指示どおり）。
- take 点の形（counter 追加 vs getter 追加）は次段の設計監査事項とし、
  本 Step ではundeﬁnedのまま残す（R7–R10 パターン踏襲）。

## 10. R14-A/B/C 判定

```text
R14-A（7条件）： 条件2（consume観測）・3（cluster分離）・4（test-only vehicle定義）が
  不成立のため REJECTED。条件1・5・6・7 は充足可能。
R14-B（residual）： take 自体が不可のため例示形に合致せず → 非該当。
R14-C（PRODUCTION ACCESSOR REQUIRED）： ADOPTED
  test-only では private worker state に到達不能。
  R7–R10 と同じく「最小POD／const noexcept／ownership非露出／RT影響なし／
  publish-retire-rebuild副作用なし」の設計監査へ戻す。
  P3-1-D（limiter系）とは分離する。
```

停止位置：設計監査へ戻る。以降の実装・build・F/R実行・比較・P3-1-D なし。
R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
