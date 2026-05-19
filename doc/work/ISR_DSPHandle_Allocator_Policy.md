# ConvoPeq DSPHandle Allocator Policy

## 目的

`slotIndex + generation` 方式の DSPHandle で、slot lifetime / reuse / overflow の未確定点を固定する。

## Handle 形式（確定）

```text
DSPHandle {
  uint32 slotIndex
  uint64 generation
  uint16 retireAuthorityId
}
```

## Allocator 戦略（確定）

### A1. Slot Table

- slot table は non-RT で管理
- RT は handle 解決のみ（allocate/free 禁止）

### A2. Reuse Policy

- retire enqueue 後、**2回の epoch advance 完了まで再利用禁止**（quarantine）
- quarantine 終了後に free-list へ戻す

### A3. Generation 更新

- slot 再利用時に generation を +1
- handle 解決時、generation 不一致なら stale handle として reject

### A4. Overflow Policy

- generation は uint64
- overflow 到達前にメンテナンス停止を要求（運用上非現実的だが規定）

### A5. Fragmentation Policy

- 即時コンパクション禁止（RT影響回避）
- 非RTメンテナンスフェーズで compaction 計画を実行

## Lifetime 終端（確定）

- crossfade 完了
- runtime world 切替完了
- engine shutdown

上記イベントで retire enqueue。destroy は grace 条件成立後のみ。

## 禁止事項

- raw pointer 直接 retire
- RT thread で slot allocate/free
- generation 無視の handle 解決

## 検証ゲート

- [ ] stale handle reject テスト
- [ ] quarantine epoch テスト
- [ ] bug2 系 UAF 再現シナリオの防止検証
- [ ] fragmentation メンテナンス手順の文書化
