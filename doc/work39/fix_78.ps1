$f = 'C:\VSC_Project\ConvoPeq\doc\work39\recovery_system_implementation_plan.md'
$c = [System.IO.File]::ReadAllText($f)
$n = 0
$crlf = [char]13 + [char]10

# 7-field
$oldF = '    uint64_t m_lastFifoTickUs_{0};'
$newF = "    uint64_t m_lastFifoTickUs_{0};" + $crlf + "    bool     m_learnerWasActive_{false};  // Learner restart detection (Issue7)"
if ($c.Contains($oldF)) { $c = $c.Replace($oldF, $newF); $n++ }

# 7-EMA
$old7 = "    const bool learnerActive = convo::consumeAtomic(*m_learnerRunningRef," + $crlf + "                                                     std::memory_order_acquire);" + $crlf + "    if (!learnerActive)" + $crlf + "        return;" + $crlf + $crlf + "    // FIFO"
$new7 = "    const bool learnerActive = convo::consumeAtomic(*m_learnerRunningRef," + $crlf + "                                                     std::memory_order_acquire);" + $crlf + $crlf + "    // Learner restart -> EMA reset (Issue7)" + $crlf + "    if (!learnerActive) {" + $crlf + "        m_learnerWasActive_ = false;" + $crlf + "        return;" + $crlf + "    }" + $crlf + "    if (!m_learnerWasActive_) {" + $crlf + "        m_fifoEma_ = -1.0;" + $crlf + "        m_lastFifoEma_ = 0.0;" + $crlf + "        m_learnerFifoHighSinceUs_ = 0;" + $crlf + "        m_learnerWasActive_ = true;" + $crlf + "    }" + $crlf + $crlf + "    // FIFO"
if ($c.Contains($old7)) { $c = $c.Replace($old7, $new7); $n++ }

# 8-RestoreFlush
$old8 = "        noiseShaperLearner->setState(lastKnownGoodNoiseShaper_.state);" + $crlf + "    // Step2"
$new8 = "        noiseShaperLearner->setState(lastKnownGoodNoiseShaper_.state);" + $crlf + "    // Restore: ForceSnapshotPublish + DeferredPublicationFlush (Issue8)" + $crlf + "    if (runtimeOrchestrator_)" + $crlf + "        runtimeOrchestrator_->clearDeferredForShutdown();" + $crlf + "    // Step2"
if ($c.Contains($old8)) { $c = $c.Replace($old8, $new8); $n++ }

[System.IO.File]::WriteAllText($f, $c)
Write-Host $n
