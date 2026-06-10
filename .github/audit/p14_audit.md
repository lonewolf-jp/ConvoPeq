# P14 Publication Unit Audit Report

AUDIT_RESULT: PASS
AUDIT_DATE: 2026-06-05
AUDITOR: GitHub Copilot (AI Assistant)
CHECKED_SYMBOLS: publishWorld, coordinator.publishWorld, RuntimePublicationCoordinator::publishWorld, RuntimeStore::publishAndSwap, RuntimeBuilder::buildRuntimePublishWorld, applyRuntimeCommitFromIntent, enqueuePublicationIntentForRuntimeCommit, appendPublicationIntentForCommitProducer, appendPublicationIntentForCommitConsumer, appendPublicationIntentForCommitSlot
FINDINGS: Publication Unit Audit（完了条件必須）。RuntimePublishWorld 全体以外の部分公開インターフェース（publish(generation), publish(dsp) など）の存在を調査。RuntimeStore::publishAndSwap() は RuntimePublicationCoordinator::publishWorld() からのみ呼ばれ、publishWorld() は常に complete RuntimePublishWorld 全体を受け取る。enqueuePublicationIntentForRuntimeCommit は PublicationLog に DSPCore を enqueue するが、これは publish 前の準備であり、部分公開ではない。applyRuntimeCommitFromIntent → commitNewDSP 内で complete world が buildRuntimePublishWorld() で構築され publishWorld される。部分的な world を publish するインターフェース（例: publishGenerationOnly, publishDSPOnly）は存在しない。PublicationIntent 構造体は完全な RuntimeBuildSnapshot を保持しており、部分データのみの Intent は存在しない。ただし PublicationLog 経路自体は Phase1-B で削除予定のレガシーパターンであり、world 全体の publish は commitNewDSP 内で coordinator.publishWorld() 経由で行われている。
SEARCH_COMMANDS: grep -r "publishWorld\|publishAndSwap" src/**, grep -r "buildRuntimePublishWorld" src/**, grep -r "publish(generation)\|publish(dsp)\|publish(" src/**, grep -r "PublicationIntent\|PublicationLog" src/**, codegraph caller analysis for publishWorld
