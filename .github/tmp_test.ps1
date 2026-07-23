cd C:\VSC_Project\ConvoPeq
 = @()
if (Select-String -Path src\**\* -Pattern 'struct PublicationIntent|struct PublicationLog' -Quiet) {
     += 'PublicationIntent/PublicationLog struct still exists'
}
if (Select-String -Path src\**\* -Pattern '\bappendPublicationIntentForCommit\s*\(|\bdrainPublicationLogForShutdown\s*\(|\bhasPublicationLogPending\s*\(' -Quiet) {
     += 'Legacy PublicationLog functions still exist'
}
if (.Count -gt 0) {
    foreach ( in ) { Write-Host '::error::'  }
    exit 1
}
Write-Host 'Phase1-B: All PublicationIntent/PublicationLog remnants removed.'
