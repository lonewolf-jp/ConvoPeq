# N1 — NEW AUTHORITY RE-PIN

```text
record type      re-pin record.  A DECLARATION of authority.
                 it is NOT a repair, NOT a source modification, NOT a commit.
N1 state         RESOLVED — NEW AUTHORITY ADOPTED
subjects modified  0
repository state unchanged; ConvoPeq.md was read and hashed, never written
```

---

## 1. Owner Decision

```text
DECISION = B
```

The current working-tree `ConvoPeq.md` is adopted as the **new** authoritative
source. This is an explicit authority change made by the Owner.

---

## 2. Historical authority — unchanged, still destroyed

```text
historical authoritative source   ConvoPeq(4).md
status                            destroyed / not present
identity
    size                          5,535,334 B
    SHA-256                       E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
```

**This decision does not restore it and does not record it as restored.** The
historical authority remains destroyed and unavailable. The two authorities are
separate generations and are kept separate throughout this record.

---

## 3. New authority adopted

```text
path              ConvoPeq.md
role              NEW authoritative source, sole source-of-truth for
                  subsequent source-level verification
adopted by        explicit Owner decision (Option B)
```

---

## 4. Identity fixed

```text
size              5,550,764 B
SHA-256           C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
Generated stamp   2026-09-27 00:36:31      (line 3 of the file)
file mtime        2026-09-27 00:37:26
HEAD at re-pin    e565019df6d263370b3cac2ed8e7a14d7c7f8db4
index entry       100644 5907a1c3d053b651f6edc860e022409c89afe0f2  0  ConvoPeq.md
git status        M ConvoPeq.md      (tracked, modified, not staged)
line endings      CRLF throughout, 121,634 lines
```

---

## 5. Re-pin execution

```text
executed          2026-09-27 17:02:08 +0900   (2026-09-27 08:02:08 UTC)
scope             declaration only
```

The re-pin consists of this record. `ConvoPeq.md` was **not** edited, regenerated,
relocated, staged, or committed. Its bytes are unchanged by this operation.

---

## 6. Post-re-pin identity re-confirmation

Measured after the record was written:

```text
size            5,550,764 B                MATCH
SHA-256         C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE   MATCH
Generated       2026-09-27 00:36:31        unchanged
git status      M ConvoPeq.md              unchanged
HEAD            e565019d                   unchanged
```

---

## 7. Separation from the historical authority

```text
                  historical              new
path              ConvoPeq(4).md          ConvoPeq.md
size              5,535,334 B             5,550,764 B
SHA-256           E5E74200…F3609          C5873522…7DFE
generation        earlier                 current
status            destroyed               ACTIVE
size difference   +15,430 B
identical         no
```

The two are not the same content, not the same generation, and not the same line
count. None of that is an obstacle: the condition of this adoption is **not**
identity with the historical authority. It is an explicit decision to adopt a new
authority.

The measurement recorded in the disposition record — that the historical
authority's byte identity is determinate from committed state `e565019d` once
line-ending normalisation is accounted for — was **not** used as grounds for this
adoption, and the historical authority was **not** adopted.

---

## 8. Rules binding on subsequent work

```text
the new authority is the SOLE source-of-truth for source-level verification.

  the historical ConvoPeq(4).md  must NOT be used as a source authority
  ConvoPeq(6).md                 must NOT be used as a source authority
  any other generation          must NOT be used as a source authority
  any reconstruction            must NOT be used as a source authority

a statement of the form "a record says so" is NOT source-confirmed evidence.
every source claim must cite, in the new authority:
    file  /  section  /  line or exact source location
    and the text must be verified present in the new authority itself.
```

---

## 9. Scope of the re-pin

```text
permitted and performed
    fixing the new authority identity
    creating this re-pin record
    recording separation from the historical authority
    re-confirming identity afterwards

NOT performed
    no production source change
    no measurement code change
    no build
    no CDB
    no new measurement
    no git commit / checkout / reset / add
    no regeneration of ConvoPeq.md
    no authority reconstruction
```

---

## 10. State

```text
N1                         RESOLVED — NEW AUTHORITY ADOPTED
new authority              ACTIVE
   path                    ConvoPeq.md
   size                    5,550,764 B
   SHA-256                 C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
   Generated               2026-09-27 00:36:31
historical authority       destroyed / unavailable  (NOT recorded as recovered)

p1_5 §13.4                 MEASUREMENT OBSERVATION / UNRESOLVED
source audit               AUTHORIZED / READ-ONLY
implementation             NOT AUTHORIZED

N2                         DOCUMENTED                    (unchanged)
N2-OPEN-1                  C2 / REMAINS OPEN              (unchanged)
R1                         NOT AUTHORIZED                (unchanged)
RR-1..RR-18                HOLD                          (unchanged)
CR-3                       PENDING                        (unchanged)
Phase 2-B                  NOT STARTED                   (unchanged)
RB3                        DEFERRED                      (unchanged)

files modified             0
files created              1  (this record)
```
