# P8 Authority Source Audit

AUDIT_RESULT: PASS
AUDIT_DATE: 2026-06-05
AUDITOR: GitHub Copilot (AI Assistant)
CHECKED_SYMBOLS: RuntimePublishWorld, RuntimeState, RuntimePublicationCoordinator::publishWorld, coordinator.publishWorld, AudioEngine::publishWorld, RuntimeStore::observe, hasFadingRuntimeInWorld, hasPendingCrossfadeInWorld, shouldUseDryAsOldInWorld, resolveCurrentDSPFromRuntimeWorldOnly, resolveFadingRuntimeDSPFromRuntimeWorldOnly, captureAudioThreadParameterSnapshot
FINDINGS: Semantic decision（意味的决定）は RuntimeWorld のみを参照する。Audio Thread が runtime 状態を読む際は RuntimeStore::observe() 経由で RuntimePublishWorld の current ポインタを取得し、その内容のみに基づいて processing decision を行う。Executor local な決定（処理順序、バッファサイズ等）は Audio Thread ローカルパラメータに基づく。Semantic decision と Execution decision の分離が維持されている。PublicationIntent 経路（Phase1-B で削除予定）には RuntimeBuildSnapshot が含まれるが、これは Semantic decision ではなく build-time データのキャリアであり、authority として使用されていない。
SEARCH_COMMANDS: grep -r "observe\|runtimeWorld\|RuntimePublishWorld" src/core/RuntimePublicationCoordinator.h, grep -r "resolveCurrentDSP\|resolveFadingRuntimeDSP\|hasFadingRuntime\|shouldUseDryAsOld" src/audioengine/**, grep -r "captureAudioThreadParameterSnapshot" src/audioengine/AudioEngine.h
