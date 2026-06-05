# P18 RuntimeWorld Identity Audit（推奨）

AUDIT_RESULT: PASS
AUDIT_DATE: 2026-06-05
AUDITOR: GitHub Copilot (AI Assistant)
CHECKED_SYMBOLS: RuntimeState::generation, RuntimeState::publication::sequenceId, RuntimeState::publication::epoch, RuntimeState::publication::mappedRuntimeGeneration, RuntimeState::topology::runtimeUuid, RuntimeState::topology::fadingRuntimeUuid, RuntimeState::generationSemantic::runtimeGeneration, RuntimeState::projectionFreshness::projectionGeneration, RuntimeState::schemaVersion, RuntimeState::metadata::schemaVersion, RuntimeState::metadata::publicationSequence
FINDINGS: RuntimeWorld Identity Audit（推奨）。RuntimePublishWorld (RuntimeState) の識別子を調査。World は以下の複合キーで一意に識別される: (1) generation (strict monotonic), (2) publication.sequenceId (strict monotonic), (3) publication.epoch (activationEpoch, strict monotonic), (4) topology.runtimeUuid (DSPCore の UUID)。generation と publication.sequenceId / epoch / mappedRuntimeGeneration は相互に一貫性が維持されている（validateSemanticCompleteness で確認）。同一 generation での epoch 単独変更は P4 契約により禁止。RuntimeBuildSnapshot の識別子（snapshot 内の generation/sequence）は世界構築の入力であり、RuntimeWorld 自体の識別子ではない。runtimeUuid は DSPCore 単位で一意であり、同一 DSP が複数 World に出現することは許容される（fading 遷移時）。全体として、RuntimeWorld の識別子体系は一貫しており、重複や不整合は検出されなかった。
SEARCH_COMMANDS: grep -r "generation\|sequenceId\|publication\.epoch\|mappedRuntimeGeneration" src/audioengine/AudioEngine.h, grep -r "runtimeUuid\|fadingRuntimeUuid" src/**, grep -r "validateSemanticCompleteness" src/**, grep -r "projectionGeneration\|projectionFreshness" src/**
