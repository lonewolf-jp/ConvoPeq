# N1 — Authority Disposition

```text
record type      disposition record.  organises candidates for an Owner decision.
                 it is NOT a re-pin, NOT an authorisation, NOT a repair.
N1 state         OPEN / AUTHORITY DISPOSITION IN PROGRESS
purpose          to put an adoptable authority to Owner judgement.
                 it is NOT to recover the lost historical authority.
subjects modified  0
repository state unchanged; every measurement below is read-only
```

---

## 1. Historical authority

```text
historical authoritative source   ConvoPeq(4).md
status                            destroyed / not currently present
historical authority identity
    size                          5,535,334 B
    SHA-256                       E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
direct availability               NO / NOT PROVEN
```

The waiting premise is withdrawn. Because the historical authority does not exist
as a file, its direct availability is not the precondition for the next step.

---

## 2. Direct availability of the historical artefact

```text
ConvoPeq(4).md  present in repository            NO
ConvoPeq(6).md  present in repository            NO
files matching  ConvoPeq(N).md  anywhere searched  0
   (repository, user profile, %TEMP%)
files of exactly 5,535,334 B inside the repository  0
```

---

## 3. Option A — require the historical `ConvoPeq(4).md`

```text
condition      size    = 5,535,334 B
               SHA-256 = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
availability   NOT AVAILABLE / NOT PROVEN
evidence       the artefact is absent.  no file, in the locations searched,
               carries that identity.
```

### 3.1 A measurement that bears on Option A, offered as neither authority nor ground

While gathering the evidence above, a byte-level relation was measured. It is
recorded because it is decision-relevant, and it is recorded **with its limits
attached**, because the Owner's instruction forbids two readings of it.

```text
HEAD                e565019df6d263370b3cac2ed8e7a14d7c7f8db4
index entry         100644 5907a1c3d053b651f6edc860e022409c89afe0f2  0  ConvoPeq.md
committed blob      5,414,041 B   pure LF, 121,293 lines, 0 CRLF
that blob with one CR inserted per line
                    5,535,334 B   <-- equals the historical authority size
   SHA-256 of that rendering
                    E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
                    <-- equals the historical authority identity, exactly
```

Sweep of every blob in the store between 5.0 and 6.0 MB, each rendered with CRLF
line endings and hashed: **exactly one** reaches the authority size, and it is the
one above.

```text
what this establishes
    the historical authority's byte content is DETERMINATE.
    It is not a lost artefact whose content is in question.  Its content is a
    function of committed state e565019d, once the line-ending normalisation
    already on record is accounted for: the generator writes CRLF, and
    .gitattributes normalises *.md to LF on commit, so the committed form and
    the working-tree form of the same content differ by exactly one CR per line.

what this does NOT establish, and is not offered as
    it is not an authority.
    it is not a re-pin.
    it is not a reconstruction offered for adoption.
    "it reproduces from git" is NOT a ground for adoption, and the Owner's
    instruction forbids treating a git-blob reconstruction as authority.
    That prohibition is unaffected by this measurement.
    Whether the determinacy just measured should bear on the Owner's choice is
    the Owner's to weigh.  This record draws no conclusion from it.
```

---

## 4. Option B — the current `ConvoPeq.md` as a NEW authority

This is a **new authority candidate, distinct from the historical authority.** It
is not asserted to be the historical authority, and it is not the same content.

```text
path              ConvoPeq.md   (working tree)
size              5,550,764 B
SHA-256           C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
Generated stamp   2026-09-27 00:36:31      (line 3 of the file)
file mtime        2026-09-27 00:37:26
HEAD              e565019df6d263370b3cac2ed8e7a14d7c7f8db4
last commit       e565019d  2026-09-27 00:01:32 +0900  lonewolf-jp  "commit"
git status        M ConvoPeq.md        (tracked, modified, not staged)
diff --numstat    363 insertions, 22 deletions against HEAD
line endings      CRLF throughout, 121,634 lines
```

Relation to the historical authority, stated as measurement:

```text
historical authority   5,535,334 B   E5E74200…
this candidate         5,550,764 B   C5873522…
difference            +15,430 B
```

That the two differ is not an obstacle to Option B. The point at issue is not
identity with the past; it is an explicit Owner decision to adopt a new authority.

Adopting it would be a **re-pin / re-authorisation**, which is not reconstruction
of the historical authority and not its restoration. It requires a separate Owner
authorisation, which has not been given.

---

## 5. Option C — an Owner-provided snapshot

```text
availability   NOT PROVIDED
required measurements once provided
    size
    SHA-256
    generation timestamp / snapshot identity
```

No such snapshot has been supplied, and none is assumed.

---

## 6. Reconstruction and substitution remain prohibited

None of the following is treated as an authority in this disposition:

```text
any artefact reconstructed from a git blob
ConvoPeq(6).md
any other generation of ConvoPeq.md
any composite synthesised from the difference between current and past files
any estimated or completed snapshot
```

None of the following is a ground for adoption:

```text
"close enough to the historical authority"
"it reproduces from git"
"most of the source content matches"
```

---

## 7. Evidence summary per candidate

```text
OPTION A   historical ConvoPeq(4).md
           size / SHA-256 as recorded in section 1
           artefact ABSENT (section 2)
           availability   NOT AVAILABLE / NOT PROVEN
           §3.1 records a determinacy measurement, explicitly not as a ground

OPTION B   current ConvoPeq.md
           size 5,550,764 B
           SHA-256 C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
           Generated 2026-09-27 00:36:31
           HEAD e565019d, status M, +363/-22
           availability   AVAILABLE / candidate
           NOT YET AUTHORIZED

OPTION C   Owner-provided snapshot
           availability   NOT PROVIDED
```

---

## 8. Correction to earlier corroboration of N1

Earlier work in this series reported that the historical authority was "not present
in git", on the evidence that **no blob in the store had a size of 5,535,334 B**.
That check was incomplete, and the conclusion drawn from it was wrong in substance.

```text
what was checked     blob sizes as stored (LF-normalised)
what was not checked the CRLF rendering of those blobs
why it mattered      the committed form and the working-tree form of the same
                     content differ by one CR per line, so a size-keyed search
                     over stored blobs cannot bound the set of authorities
```

The same shape as the process defect recorded separately in
`P1-5-IR-P2_PROCESS-DEFECT_UNNUMBERED_Exact-Token-vs-Corpus-Absence_20260927.md`:
**absence of one representation was treated as absence of the thing.** That record's
rule 1 — cover more than one surface form before reporting absence — applies here
to byte representations as well as to search tokens.

Consequently the N1 register's stated basis, "the pinned authority corresponds to
**no state in git**", is contradicted by the measurement in §3.1.

**This record does not declare N1 closed, reclassified, or resolved.** It reports
that the register's stated rationale does not survive the corrected check, and
leaves the determination to the Owner.

---

## 9. Downstream gates remain blocked

```text
re-pin                    NOT AUTHORIZED
source audit              NOT AUTHORIZED
production source audit   NOT AUTHORIZED
implementation            NOT AUTHORIZED
build                     NOT AUTHORIZED
new measurement           NOT AUTHORIZED
CDB                       NOT AUTHORIZED
R1                        NOT AUTHORIZED
RR-1..RR-18               HOLD
CR-3                      PENDING
Phase 2-B                 NOT STARTED
p1_5 §13.4                MEASUREMENT OBSERVATION / UNRESOLVED
```

---

## 10. Owner decision required

```text
DECISION  A  require the historical ConvoPeq(4).md from the Owner
DECISION  B  re-authorise the current ConvoPeq.md as a NEW authority
          (a separate authorisation, not performed here)
DECISION  C  supply a different snapshot for identity verification

RECOMMENDATION
= none.  No preference is expressed among A, B and C.  The choice of
  authority is the Owner's, not the auditor's.
```

---

## 11. State

```text
record type        disposition record.  NOT a re-pin.  NOT an authorisation.
files modified     0
files created      1  (this record)
p1_5 §13.4         MEASUREMENT OBSERVATION / UNRESOLVED      (unchanged)
16× historical     DOCUMENTED                                   (unchanged)
§13.4 causality    NOT DETERMINED                              (unchanged)
N1                 OPEN / AUTHORITY DISPOSITION IN PROGRESS
N2                 DOCUMENTED
N2-OPEN-1          C2 / REMAINS OPEN
authority          no authority is adopted by this record
gate entered       none
```
