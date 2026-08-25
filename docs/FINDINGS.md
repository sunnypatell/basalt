<img src="assets/header-findings.svg" alt="basalt Findings" width="100%" />


Measurements of sm_120 made with basalt, each with the command that reproduces it and the
evidence it rests on. Where a result is uncertain or not claimed, it says so.

All figures below are from an **NVIDIA GeForce RTX 5070 Ti** (sm_120, 70 SMs) with
`ptxas` **V13.3.73**. Published characterisation of this architecture has used other parts,
so a figure differing on a different SKU is interesting rather than contradictory.

<details>
<summary><b>The machine every number was measured on</b></summary>

<br/>

Every measurement here was taken on one machine, a personal desktop build rather than a
datacentre part or a cloud instance. Stated in full because "a 5070 Ti" is not enough to
reproduce a timing, and because the wall-clock figures quoted for the controls depend on the
host as much as on the card.

| Component | Exactly what it is |
| :--- | :--- |
| GPU | Gigabyte GeForce RTX 5070 Ti EAGLE OC, sm_120a, 70 SMs, 16,303 MiB, 2,542 MHz boost |
| Driver | 610.88 |
| CPU | AMD Ryzen 7 9800X3D, 8 cores, 16 threads |
| Memory | 32 GB Patriot 6000 Series DDR5, 2 x 16 GB, running at 6000 MT/s under EXPO |
| Storage | PCIe 4.0 NVMe |
| OS | Windows 11 Pro, build 26200 |
| Python | 3.14.6 |
| Toolchain | `ptxas` / `nvdisasm` / `cuobjdump` V13.3.73, from the pinned CUDA 13.3.1 redistributable |
| Cross-check toolchain | CUDA 13.0.3, for finding 29 |
| Audited libraries | cuBLAS 13.6.0.2, cuFFT 12.3.0.29, cuRAND 10.4.3.29, cuSOLVER 12.2.6.9, cuSPARSE 12.8.2.51, NPP 13.1.2.81, nvJPEG 13.2.1.68 |

Nothing here needs this exact machine. Everything except the four GPU controls runs on any
host with the fetched redistributable and no NVIDIA hardware at all, which is what makes the
instruction database, the assembler and the checker reproducible in CI.

</details>

---

## 1. A stall count of zero is a safe encoding, not zero cycles

The `stall` field is four bits. Value 0 does **not** mean "issue the next instruction
immediately". It is a distinct encoding that waits for outstanding results as well as
elapsed cycles.

Measured over a 128-link dependent `IMAD` chain, patching the control bits of every link
directly and comparing the answer against a reference:

| `stall` | cycles per instruction | result |
| ---: | ---: | :--- |
| **0** | **36.85** | **correct** |
| 1 | 4.88 | wrong |
| 2 | 4.88 | wrong |
| 3 | 5.88 | wrong |
| 4 | 6.88 | correct |
| 8 | 10.88 | correct |
| 15 | 18.02 | correct |

<img src="assets/chart-stall.svg" alt="Cycles per instruction for each stall encoding: 0 costs 36.85 and is correct, 1, 2 and 3 cost 4.88, 4.88 and 5.88 and are silently wrong, and 4, 8 and 15 cost 6.88, 10.88 and 18.02 and are correct.">

So a zero stall costs roughly nine times a correctly scheduled instruction and is always
safe, while 1 to 3 corrupt silently.

**Why it matters beyond the encoding.** This is why `ptxas -O0` emits an entirely zeroed
control word, with no stalls and no scoreboards anywhere, and the code still computes the
right answer about nine times slower. It also means unscheduled output cannot be used to
learn anything about what a dependency requires, and that a checker reading 0 as zero
cycles will report correct programs as broken.

```bash
python -m basalt.cli verify path/to/O0.cubin --latencies src/basalt/data/latency/rtx-5070-ti.json
```

## 2. The checker agrees with the vendor compiler across the whole corpus

Every kernel `ptxas` builds from the generated corpus is compiled and verified.
Its scheduling is the reference, so an error on any of them is basalt's fault.

| Positive control | Result |
| :--- | ---: |
| Kernel and optimisation-level pairs | 1,323 |
| Dependencies checked | 30,421 |
| Errors | **0** |

Every dependency in that count is judged against a rule basalt derived rather
than copied, read barriers included since finding 25.

This is not a formality. Every modelling error basalt has made was found here or
by the smaller version of it, never by reasoning about the architecture:
scoreboards treated as flags rather than counters, a wait required from every
consumer rather than from any instruction, a guard predicate read as an opcode,
a scoreboard ignored because the producer was fixed-latency, `VOTEU` classified
as completing out of order, and a stall requirement treated as a property of the
producer when it belongs to the pair.

```bash
pytest -m slow      # runs the control; it also runs in CI on every push
```

## 3. Understalling corrupts silently, and basalt predicts exactly when

The premise of the project, checked directly rather than argued. For each encodable stall on
a dependent producer, basalt's static verdict is compared against what the silicon computes:

| `stall` | basalt says | hardware computes | agree |
| ---: | :--- | :--- | :--- |
| 0 | clean | correct | yes |
| 1 | hazard | **wrong** | yes |
| 2 | hazard | **wrong** | yes |
| 3 | hazard | **wrong** | yes |
| 4 | clean | correct | yes |
| 5 | clean | correct | yes |
| 6 | clean | correct | yes |
| 7 | clean | correct | yes |

No crash, no fault, no warning at any of the wrong rows. The wrong answer is also
deterministic rather than a race: every repeat produces the same incorrect value.

This is held as a test (`tests/test_gpu.py::TestVerdictsMatchHardware`), so a change that
breaks the agreement fails the suite.

## 4. Required stall, by three independent methods

Three ways of asking the question, which do not always agree because they are not quite the
same question.

- **Chain timing** measures `max(latency, initiation interval)`, so a rate-limited unit reads
  high.
- **What `ptxas` schedules** is an upper bound: the compiler may be cautious.
- **Fault injection** measures the requirement itself, by shortening the gap until the answer
  changes.

| Instruction | Chain timing | `ptxas` leaves | Fault injection | Reading |
| :--- | ---: | ---: | ---: | :--- |
| `IMAD` | 4 | 4 | **4** | all three agree |
| `IADD3` | 4 | 4 | **4** | all three agree |
| `FFMA` | 4 | 4 | **4** | all three agree |
| `FADD` | 4 | 4 | **4** | all three agree |
| `FMUL` | 4 | 4 | **4** | all three agree |
| `LOP3` | 4 | 4 | **4** | all three agree |
| `SHF` | 4 | 4 | **4** | all three agree |
| `I2FP` | 24 (with `F2I`) | 6 | not established | see below |
| `MUFU` | 44 | n/a | scoreboarded | result takes 44, covered by a scoreboard |
| `POPC` | 18 | n/a | scoreboarded | same |
| `DADD` | 64 | 64 | scoreboarded | see below |
| `DFMA` | 64 | 64 | scoreboarded | see below |

```bash
python -m basalt.cli measure -o my-card.json   # chain timing
python -m basalt.cli probe-stalls                             # fault injection
```

Three of these contradict the model basalt shipped with before any of it was measured, and
not by a little:

<img src="assets/chart-latency.svg" alt="fp64 add was assumed 48 cycles and measured 64, POPC was assumed 4 and measured 18, and the I2FP plus F2I conversion round trip was assumed 12 and measured 24.">

**On the fp64 rows.** The 64-cycle figure is corroborated twice over: `ptxas` covers a `DFMA`
dependency by padding with NOPs at stall 15 and accumulating exactly 64 cycles, matching the
independently timed figure to the cycle. It also signals a scoreboard on the same
instruction, which is belt and braces, and it is the scoreboard that makes shortening the
stalls harmless. That is why fault injection reports the pair as scoreboarded rather than
returning a number.

**On `MUFU` and `POPC`.** A deterministic latency and a stall-covered latency are different
things. `MUFU` produces a perfectly linear 44 cycles under timing, but `ptxas` signals a
scoreboard on it and the dependent instruction waits on that scoreboard, so the stall does
not have to carry the dependency. basalt keeps it classified as variable for that reason.

## 5. Tensor cores: throughput and requirement are far apart

Timed with a dependent chain that accumulates through the D operand, so each
`mma.sync` cannot issue until the previous one has written the accumulator back.
`ptxas` emits these back to back with nothing between them, so the slope is the
instruction and nothing else.

| Instruction | Cycles per instruction, one warp | R² |
| :--- | ---: | ---: |
| `HMMA.16816.F32` (f16) | 34.52 | 0.99999 |
| `HMMA.16816.F32.BF16` | 34.44 | 1.00000 |
| `HMMA.1688.F32.TF32` | 34.37 | 1.00000 |
| `IMMA.16832.S8.S8` | 26.86 | 1.00000 |
| `QMMA.16832.F32.E4M3.E4M3` | 34.36 | 1.00000 |
| `QMMA.16832.F32.E5M2.E5M2` | 34.41 | 1.00000 |
| `QMMA.16832.F32.E3M2.E3M2` | 34.48 | 1.00000 |
| `QMMA.16832.F32.E2M1.E2M1` | 34.50 | 1.00000 |

Two things stand out.

**The low-precision format does not change this number.** E4M3, E5M2, E3M2 and
E2M1 all land within a tenth of a cycle of each other and of f16. Whatever FP4
buys on this part, it is not a shorter dependent-accumulate interval.

**Integer is faster than float here**, at 26.9 against 34.4.

**These are not latencies.** `ptxas` schedules the same back-to-back chain with
`stall=11`, and shortening it further still computes the right answer, so the
figure above is the interval at which one warp can push dependent matrix
operations through the unit rather than the time a result takes to appear. The
distinction matters: a checker that treated 34 as the required stall would
reject correct code. It is recorded here as throughput, and the number is not
written into the latency model.

## 6. The instruction database is usable, not just readable

Knowing which bits moved an operand describes an encoding. Being able to write a
value into those bits and get it back is what an assembler needs, and it does not
follow automatically: a table built by observation can be a description that
happens to be wrong.

So every measured field is written through and read back, across five values
rather than one, since a single value can agree by accident.

| Write-through and read-back | Result |
| :--- | ---: |
| Forms checked | 339 |
| Register-bearing operand slots | 866 |
| Slots that behave as measured | **845 (97.6%)** |
| Forms with every register slot controllable | 310 |

The 268 remaining slots hold something other than a register, a branch target,
an immediate, a named special register such as `SR_CLOCKLO`, and are counted
apart rather than reported as failures, because reading a register number back
out of them is the wrong question.

Sixteen forms still have a slot that does not behave, mostly a carry or
predicate input in the last position on extended-precision forms. Those are
listed rather than rounded away.

```bash
python -m basalt.cli validate-isa --show
```

## 7. fp64 is carried by the scoreboard, and still owes a small stall

`ptxas` covers a dependent `DFMA` pair with both mechanisms at once: 64 cycles of
stall, padded out with maximum-stall NOPs, and a scoreboard signalled on the
producer that the consumer waits on. Which one is actually carrying it can be
settled directly.

| What was changed | Result |
| :--- | :--- |
| Nothing | correct |
| Scoreboards and waits kept, stalls across the fp64 stretch reduced to 1 | correct |
| Stalls kept, the `DADD` write barrier and the store's wait removed | **wrong** |
| Everything kept, the `DADD`'s own stall alone cut from 2 to 1 | **wrong** |

So the scoreboard is what carries the dependency. The 64 cycles are the cost of a
dependent chain, which is a real number and a different question from what
correctness requires.

The last row is the part worth keeping. A wait covers the long, variable part of
the result and not the whole of it: the producer still owes a small stall of its
own, and `ptxas` knows exactly how much. Across the corpus it never schedules any
of `DADD`, `DSETP`, `DFMA` or `DMUL` below 2 cycles, however the consumer waits.
basalt mines that minimum per opcode alongside everything else and applies it
wherever a scoreboard is signalled.

Every one of the 219 fp64 instructions in the corpus carries a write scoreboard
and none goes without, so fp64 is modelled as completing out of order rather than
on a fixed schedule.

> [!NOTE]
> **This entry previously concluded the opposite**, that the cycles were
> load-bearing and the scoreboard was belt and braces. The experiment behind it
> reduced every stall in the kernel, including the one on the `LDC.64` that sets
> up the store address, which breaks the kernel on its own and has nothing to do
> with fp64. Confining the change to the fp64 stretch reverses the result. The
> wrong conclusion survived because it agreed with the model already in the code:
> fp64 was classified as fixed latency, so nothing disagreed with it until a
> rescheduled fp64 kernel was run on the GPU and returned 15.84 where the vendor
> returned 20.72.

**A related bug this uncovered.** `DFMA R4, R4, R6, R4` carries no width suffix
anywhere, but fp64 operands occupy register pairs: it writes R4 and R5 and reads
R6 and R7. basalt's operand model widened only on an explicit `.64`, so it saw
half of every fp64 dependency. Silent in the checker, silent in the scheduler,
and found only by running a rescheduled kernel and getting the wrong number.

## 8. The stall field cannot express a long latency

Four bits, so 15 is the largest gap a single instruction can request. Any requirement above
that must be covered by accumulating stalls across several instructions, or by a scoreboard.
`ptxas` does both: for a 64-cycle fp64 dependency it emits

```
DFMA R4, R4, R6, R4     stall=15  wr=1
NOP                     stall=15
NOP                     stall=15
NOP                     stall=15
NOP                     stall= 4          <- 15+15+15+15+4 = 64
DFMA R4, R6, R4, R4     stall=15  wait=0x02
```

Four NOPs whose only purpose is to spend cycles.

## 9. A guard predicate costs two and a half times an ordinary read

`@P1 IMAD` and `SEL R3, R0, R7, P1` both read `P1`. They do not need the same lead.

A guard decides whether the instruction issues at all, so it has to be resolved before
issue. An ordinary source is read later, once the instruction is already going. The gap
between the two is large enough to corrupt a result:

| how the predicate is read | cycles needed from a fixed-latency producer |
| --- | --- |
| as the instruction's guard, `@P1 IMAD` | 13 |
| as a data operand, `SEL ..., P1` | 5 |
| as a general register, for comparison | 5 |

Measured twice, two ways.

**On hardware.** A loop kernel where `ISETP.GT.AND P1, PT, R5, 0x1, PT` feeds
`@P1 IMAD R3, R0, R7, R5`, with the stall on the `ISETP` swept and everything else held:

| stall | result | |
| --- | --- | --- |
| 1 to 5 | 17 or 668, varying between launches | wrong |
| 6 to 8 | 17 or 53 | wrong |
| 9 to 11 | 17 | wrong |
| 12 | 13 | wrong |
| 13 | 2005 | correct |
| 14, 15 | 2005 | correct |

The staircase is the useful part. A single wrong answer could be anything; a value that
climbs monotonically toward the right one as the gap widens, and then stops changing, is a
timing requirement being met.

**Across the corpus.** Every dependent pair in the corpus, split by how the
consumer reads the value, scoreboard-covered pairs excluded because there the stall says
nothing:

| bucket | pairings | samples | median requirement |
| --- | --- | --- | --- |
| guard predicate | 15 | 55 | 13 |
| predicate as data | 7 | 51 | 5 |
| general register | 102 | 785 | 5 |

The discriminating case is `LOP3 -> MOV`, which appears in the corpus both ways: 13 cycles
when `MOV` is guarded by the result, 5 when it reads it as data. Same producer, same
consumer, same distance available, different requirement. So this is a property of issue,
not of the predicate file, and not of the opcode pair.

`ptxas` has clearly always known. It leaves 13 cycles in front of a guard and 5 in front of
a data read, consistently, across every kernel that has both.

### Why this one matters more than its size suggests

Getting it wrong is silent in the dangerous direction. Charging a guard the cheaper
requirement produces a schedule that assembles, loads, launches, returns a plausible
number, and is wrong. Nothing faults. Basalt did exactly this until the sweep above, and
its own checker agreed with its own scheduler the whole time, because both consulted the
same figure.

It also cannot be found by reasoning about the corpus alone. The mined minimum for
`ISETP -> IMAD` is 5, which is correct and useless: it is mined from the unguarded pairs.
Basalt now mines guards under their own key, `ISETP -> @IMAD`, and keeps the two apart in
the per-producer fallback as well. Collapsing them would let one 5 cycle `ISETP -> SEL`
answer for every guard in the program.

Closing this made the loop kernel round-trip, which had been recorded here as a
loop-carried scheduling gap. It was not one. The diagnosis was wrong and the hardware
said so.

## 10. Rescheduling the whole corpus, on the GPU

The scheduler was run over seven hand-written kernels for a long time and passed all
seven. That is a smoke test wearing the clothes of a control.

`scripts/roundtrip_corpus.py` runs it over everything the corpus generates, plus thirteen
hand-written kernels whose control flow is the point rather than their opcodes. For each of
the 330: compile with `ptxas`, discard every control bit it produced, compute new ones,
write them back, and run both versions on the card with identical input. The rescheduled
kernel has to produce the same bytes.

The thirteen exist because the generated corpus is deliberately narrow. One or two
instructions of body per kernel is right for attributing a bit to a form and wrong for
exercising a scheduler: almost nothing in it has a loop, a barrier, a nested branch, or
shared memory that is actually addressable. The hand-written ones have counted loops with
accumulators carried around the back edge, nested loops, a branch inside a loop, barriers
with traffic on both sides, predicated writes, and long unbranched dependent chains.

| Reschedule and re-run | Kernels |
| :--- | ---: |
| Kernels rescheduled and run | 441 |
| Comparable (the vendor runs here, deterministically, and reproducibly) | 314 |
| **Byte-identical to the vendor schedule** | **314** |
| Wrong | 0 |

Every kernel basalt can be compared on now computes exactly what the vendor's schedule
computes, from control bits it worked out itself. The comparable count moves by one or two
from run to run, because a few kernels' reproducibility depends on what last used the card;
the match count moves with it and the failure count stays at zero.

It holds at three optimisation levels, which is three different vendor schedules to
replace rather than one. All three legs below come from a single run on a clean tree at one
commit, which the sweep prints above its own results:

| `ptxas` level | Comparable | Matching | Mismatched |
| :--- | ---: | ---: | ---: |
| `-O1` | 315 | 315 | 0 |
| `-O2` | 313 | 313 | 0 |
| `-O3` | 314 | 314 | 0 |

`-O0` is not offered. It emits a zeroed control word, so there is no schedule there to
replace and nothing the comparison would prove.

The first run of this scored 246. Everything between then and now was found by it:

- a guard predicate needing 13 cycles where the same predicate read as data needs 5
  (finding 9),
- a waited-on scoreboard still leaving a residual gap the producer has to cover, and that
  gap being a distance to the consumer rather than the producer's own stall (finding 7),
- the conversion pipe, `POPC` and `FLO` completing out of order rather than on a fixed
  schedule, like fp64 before them,
- requirements having to be keyed on the full mnemonic on both paths, since `I2F.RP` needs
  1 cycle where every other `I2F` needs 2, and `IMAD.WIDE.U32.X` into `IADD` is scheduled at
  3 where plain `IMAD` into `IADD` is scheduled at 5; collapsing either takes a different
  instruction's requirement and wears this one's name,
- a predicated write not killing the previous definition, because `@!P0 FMUL R7, R7, c`
  leaves R7 holding whatever produced it whenever the guard is false,
- an instruction that writes a predicate and a register reading as though it wrote only the
  predicate, so nothing scoreboarded the returned value of `SHFL.IDX PT, R9, ...` or of any
  atomic, and nothing waited for it,
- a variable-latency unit returning results in the order it was given work, so a wait on
  something issued later covers everything that unit still owes: `ptxas` scoreboards the
  second of two consecutive shuffles and waits only on that,
- a wait carried by a predicated instruction not being something a later consumer can lean
  on, since the instruction carrying it may not execute,
- `IMAD.WIDE` writing a register pair, and taking one as its addend, with nothing in the
  mnemonic to say so,
- a call needing everything outstanding to have landed first, because control leaves for
  code this analysis has not read and the callee may use any register,
- the yield bit not being independent of the stall count, which is the one below,
- and the scheduler refusing to allocate a seventh outstanding load instead of sharing a
  scoreboard, which a counter permits and which rejected 45 kernels outright.

None of those were reachable by reasoning, and none were visible to the checker, because
the checker reads the same latency model the scheduler does. A wrong entry satisfies both
at once. That is the whole argument for running the silicon.

### The last one, and why it was the last one

The final two failures were a signed integer divide and a warp-aggregated 64-bit atomic,
and both had resisted every reading of their dependencies. Neither was a dependency
problem.

**The yield bit is not independent of the stall count.** Across the whole corpus `ptxas`
emits a stall of zero with the bit clear 17,999 times and set never, and a stall of one with
it set 6,192 times and clear twice. basalt wrote the stall and left the bit as it found it,
which produced pairs the vendor emits rarely or not at all. Finding 26 fits the relationship
properly and measures what the bit is worth, which is nothing to the answer.

`nvdisasm` refuses them outright, with `undefined value 0x10 for table TABLES_opex_0`. The
GPU does not refuse them. It runs the kernel and returns an answer, which is the worse of
the two outcomes, because nothing complains until something tries to read the result back.

That is also how the bug hid. Two of basalt's own checks reschedule a kernel and hand it
back to the verifier, and a program `nvdisasm` will not read comes back empty. An empty
program has no hazards, so both checks reported clean and had been reporting clean for as
long as the bug existed. They now compare the instruction count first and fail if the
result did not survive the round trip, because a check that passes on nothing is worse than
no check.

Clearing the bit at any stall from 2 upwards is a throughput choice rather than a
correctness one, and `ptxas` is seen doing it at every value in that range, so the rule
basalt follows produces only pairs the vendor also emits.

One reading of the last two failures was implemented before the encoding bug was found, and
is worth recording as rejected. `ptxas` puts a wait on `HFMA2 R4, -RZ, RZ, 0, 0` in the divide,
which reads nothing and writes a constant, so no dependency in the operand list accounts
for it. The obvious explanation is write-after-read: it overwrites R4 while an outstanding
conversion is still reading it.

Teaching the scheduler to wait before overwriting any register an outstanding
variable-latency instruction reads took the corpus from 301 matching down to 293, and made
eight previously correct kernels non-deterministic. SASS carries a separate read-barrier
field for instructions that collect their sources late and `ptxas` uses that rather than
write scoreboards, so treating every scoreboarded instruction as a late reader is not a
conservative approximation of anything. Recorded so the next attempt does not spend the
same afternoon on it.

### How much of the ISA this actually covers

"Every comparable kernel" is a claim about kernels, and the useful question is how much of
the instruction set they contain between them:

| Opcode coverage | Count |
| :--- | ---: |
| Opcodes in the database | 90 |
| Opcodes the corpus reaches at the levels the round trip runs | **85 (94%)** |

```bash
python scripts/corpus_figures.py            # -O1, -O2 and -O3, the levels that schedule
python scripts/corpus_figures.py --opt 0
```

The denominator moves every time the database is rebuilt, which is why it is computed by
that command rather than counted by hand once. It was 74 of 77 when this section was first
written, and both halves moved.

The five it never reaches are `BPT`, `ENDCOLLECTIVE`, `LDL`, `R2UR` and `WARPSYNC`, and the
reason is specific rather than incidental: **every one of them appears at `-O0` and at no
higher level**, which is measured rather than argued. Recompiling the whole corpus at `-O0`
reaches all five, and recompiling at `-O1`, `-O2` and `-O3` reaches none. For `WARPSYNC` and
`ENDCOLLECTIVE` the mechanism is visible in the PTX, since `shfl.sync` and `bar.sync` lower
to them there and the optimiser folds them away above it. `BPT` and `LDL` joined the group
when the database was rebuilt wider; `-O0` keeping locals in memory rather than registers is
the obvious source of `LDL`, and neither is claimed beyond the measurement. The round trip
deliberately does not run `-O0`, because that level emits a zeroed control word, so there is
no schedule to replace and nothing the comparison would prove.

So they are in the database legitimately and are unreachable by this control by
construction. Nothing is claimed about how basalt schedules them.

`BMSK` used to be in that group for the same reason, reachable only through `bfi.b32` at
`-O0`. Written directly as `bmsk.clamp.b32` it survives `-O3`, so a kernel was added that
does, and it is now covered.

This is the number to attack. Every correction in this document came from widening what
gets run, twice from the corpus growing and once from running the same kernels against more
than one input, so the ceiling here is coverage rather than cleverness.

### What is still excluded

12 kernels that are not runnable by construction, 2 whose vendor output is not
deterministic under 32 threads storing to one address, and any whose result is not
reproducible once something else has used the card.

The first group is the shared and local memory forms. They read shared memory through an
address that has been converted to the global space, which exists to make `ptxas` emit an
`LDS` or an `LDL` and was never meant to execute. Excluding them is not a limitation of the
runner. That last group is why the vendor is
run a second time, after basalt has had the GPU: a kernel reading uninitialised shared
memory is stable until it is not, and its first result is not ground truth. Every `LDSM`
and `MOVMATRIX` kernel read as a basalt failure until that check existed. All three groups
are excluded from the 303 rather than counted as passes.

## 11. Where a branch keeps its destination

A branch cannot be assembled on its own. The field holds a distance rather than an address,
so `BRA \`(.L_x_0)` is a different 128 bits in every kernel it appears in, and an assembler
that reuses a harvested encoding emits a jump to wherever the harvested kernel jumped.

The field resisted probing. Flipping one bit at a time and reading the decoded text back,
which is how every other field here was found, reports 95 bits as moving the target, because
changing the opcode changes what the rest of the word is read as. Searching for a contiguous
run whose value matches the distance finds nothing either, at any width, under any
convention.

It falls out immediately from real kernels instead. The label table gives the destination,
the instruction gives its own address, and the word gives the bits, so a field and a
convention that agree on every sample is the encoding:

| Instruction at | Jumps to | Distance | Bits 16:23 | As signed |
| ---: | ---: | ---: | ---: | ---: |
| `0x0b0` | `0x180` | `+192` | `0x30` | `+48` |
| `0x170` | `0x0e0` | `-160` | `0xd8` | `-40` |
| `0x230` | `0x080` | `-448` | `0x90` | `-112` |
| `0x270` | `0x270` | `-16` | `0xfc` | `-4` |

Every distance is four times the signed byte at bits 16:23, and the sign continues into bits
34:81, which sit far away with ten unrelated bits between them. So:

> **target = address + 16 + 4 × signed(bits[16:23] ++ bits[34:81])**

The split is why a contiguous search finds nothing, and the scale of four is why a search
for the raw distance finds nothing either. All 1,548 branches in the corpus decode to their
label under this rule and none decodes wrongly.

The rule is a measurement, and a measurement written down as a constant is exactly what goes
quietly wrong when a compiler version changes, so it is re-derived from the corpus by a test
rather than trusted. With it, assembling every corpus kernel as a whole program reproduces
12,395 of 12,400 instructions bit-identically at `-O3`, and none to anything else.

## 12. What the correctness costs

A scheduler that reports only whether it was right is hiding the trade it made. basalt's
schedules are correct on every comparable corpus kernel, and here is what they cost:

| Schedule | Issue cycles |
| :--- | ---: |
| `ptxas`, all three levels | 56,441 |
| basalt | 59,125 |
| **ratio** | **1.05x** |

Slower on 111 of the 1,323 kernel and optimisation-level pairs and cheaper on 842, with
every comparable kernel still byte-identical on the GPU at all three.

Cheaper than the vendor is believable rather than suspicious, for a specific reason: basalt
schedules every dependency at the tightest gap `ptxas` was observed to leave for that exact
pairing, never below the producer's own latency, and `ptxas` does not always schedule at its
own minimum. It is balancing
register pressure and memory alongside issue latency; this is optimising one number.

It was not believed on sight. The first time the ratio went under 1.0 the hardware round
trip broke, on an `IMNMX` whose destination the operand model could not see, and the number
only stood once that was fixed and every kernel round-tripped again.

Two decisions still cost cycles, and both were taken deliberately.

**The safe stall encoding where a value leaves a block.** A definition consumed in another
block is covered by putting a zero stall on the block's last instruction, which waits for
outstanding results as well as elapsed cycles and costs about 37 cycles.

It used to be placed at every boundary regardless, which was unconditionally correct and
most of the gap: 732 of them across the corpus, against 74 now. Restricting it to blocks
that actually have something live out took the ratio from 1.39x to 1.29x. Live-out is
computed as the ordinary backwards fixed point rather than guessed, because trading a known
cost for an unknown correctness risk is the wrong way round.

**Stall the assignment placed and nothing needs.** The fixed point only ever adds: it finds
a consumer that is short and spends cycles in the window before it, and a cycle spent for
one pair also separates every other pair spanning that point. So a later pair can be
satisfied by stall placed for an earlier one, leaving the earlier placement larger than
anything requires. Walking that back, one cycle at a time, judged by the same requirement
function that placed them, took 1.29x to 0.93x. Charging for anti-dependencies (finding 23)
took it back to 1.06x, and mining the anti-dependency per pairing rather than
charging a constant took it to 1.05x, which is what that correctness costs. `LDC` alone was half the excess
before any of it, almost all overshoot rather than requirement.

**Not leaning on a wait a predicated instruction carries.** `ptxas` does lean on them and
its output runs; basalt emits its own wait instead, because relying on one was measurably
wrong for `MUFU` feeding a store through a predicated `FMUL` (finding 10). One extra wait
per occurrence.

### Costing this is easy to get wrong

The first attempt at this measurement reported basalt as nearly four times **faster**,
which would have been a lovely thing to write down and completely false. Everything after
the first `EXIT` is padding the assembler emits to fill a cache line and never issues, and
`ptxas` leaves it at a zero stall. Counting that padding at 37 cycles each charged the
vendor several hundred phantom cycles per kernel.

The number is a cost model, not a measurement. It counts what the control bits ask the
scheduler to wait, which is the part basalt decides, and says nothing about memory latency
or occupancy. It is pinned in the test suite from both sides: getting slower is a
regression, and getting much faster without the hardware round trip also moving is a
reason to distrust the costing rather than to celebrate.

## 13. A read barrier covers more reads than its own instruction's

A read barrier means "signal once my operands have been read". What it protects is less
obvious: `ptxas` puts one on the *last* of a run of loads and lets in-order issue plus the
gaps it chose carry the rest. By the time the fourth load has read the address register, the
first three have too, so one barrier covers four reads. Compress those gaps and the
guarantee disappears with them, silently, because nothing in the encoding records that the
barrier was ever standing in for its neighbours.

`k_mma_m16n8k32_s4_s4_s32` at `-O1` is the case that exposed it:

```
      LDG.E R7, desc[UR4][R4.64]          stall 4
      LDG.E R2, desc[UR4][R4.64+0x4]      stall 4
      LDG.E R0, desc[UR4][R4.64+0x40]     stall 4
      LDG.E R3, desc[UR4][R4.64+0x80]     stall 2   read_barrier 0
      MOV   R4, 0x90                                wait 0x01
```

Four loads take their address from `R4`, and the `MOV` that overwrites `R4` waits for the
barrier. basalt kept the barrier, kept the wait, and issued the loads one cycle apart
instead of four. The barrier fired while the earlier loads were still reading, `R4` was
overwritten under them, and they loaded from the wrong address. No fault, wrong answer,
full speed. It is the exact failure this repository exists to catch, produced by basalt.

**How the cause was isolated.** The first attempt was to start from the vendor's cubin and
apply basalt's control words to a growing prefix, which pointed at the fourth load. That
answer was wrong, and wrong in an instructive way: scoreboard numbers are global, so a cubin
holding basalt's write barriers and the vendor's wait masks is broken by construction and
the bisection was measuring its own hybrid. The experiment that works keeps basalt's
schedule whole and changes only stalls, which is always safe:

| Cubin | Matches the vendor |
| :--- | :---: |
| basalt's schedule | no |
| basalt's schedule, every stall raised to the vendor's | **yes** |

That separates the two hypotheses in one run. The bug is a stall, not a barrier, and a
second bisection over which stalls have to be raised lands on the run of loads.

**The rule basalt adopted, and the better one that replaced it.** The first rule was to
decline to make the window tighter than `ptxas` made it: inside the window, the vendor's own
stalls were a floor. That worked, and it was the last thing in the scheduler copied rather
than derived, which meant it did nothing at all for a program the vendor never compiled.

What actually holds the window open is the rate the unit accepts work. Mine the smallest
stall `ptxas` ever leaves between each *consecutive pairing* and it comes out as a table:

| Pairing | Cycles the vendor never goes below | Observations |
| :--- | ---: | ---: |
| `LDG` after `LDG` | 4 | 1,953 |
| `STS` after `STS` | 4 | 161 |
| `IMAD` after `IMAD` | 2 | 95 |
| `LDS` after `LDS` | 1 | 33 |
| `FFMA` after `FFMA` | 1 | 159 |

Four loads under one barrier therefore take at least 4 cycles between them however little the
dependencies ask for, which is exactly the gap this finding is about, and it is now a number
basalt derived rather than one it read off the schedule it was replacing. The round trip is
unchanged at 439 of 439 across all three levels.

The floor applies only inside a read-barrier window. Outside one the ordinary dependency
rules cover everything, and applying an issue rate everywhere would charge `NOP` after `NOP`
15 cycles, which is padding rather than a rate. Where the *barrier* goes is a separate
question, and finding 25 answers that one from measurement too.

The window is bounded by the previous read barrier and by the enclosing basic block, and
the block bound is not a convenience. `s_loop_double` at `-O1` has a read barrier on a
`DFMA` inside a loop body, where the thing it guards against is the *next iteration*
overwriting the operands of this one. The instructions above the label are preamble that
runs once, and pinning their stalls would cost cycles to protect nothing. More generally,
control can arrive at a branch target from anywhere, so whatever gap the fall-through path
happened to have is not a guarantee `ptxas` is relying on either.

**What the window actually needs, and what the rule costs.** The rule above is conservative
by construction, and it is worth knowing by how much. Holding the kernel at the vendor's
schedule and sweeping only the stall on the run of four loads:

| Stall on each load | Result |
| ---: | :--- |
| 0 | correct, and 0 is the long-wait encoding rather than a short gap |
| **1** | **wrong** |
| 2 | correct |
| 3 to 8 | correct |
| *vendor leaves 4* | *correct, two cycles above the minimum* |

Reproducible across repeated runs, and deterministic rather than a race. Two things follow.
The hazard is real: one cycle is not enough and the answer is wrong every time, so the
window is not an artifact of the harness. And the true requirement here is 2 where `ptxas`
leaves 4 and basalt therefore also leaves 4, which is two cycles per load spent to avoid
guessing.

That is deliberately not turned into a model. One pattern of four `LDG.E` in one kernel is a
data point, not a latency, and this is exactly the shape of evidence that produced the wrong
answer in finding 7. The rule stays "do not make the window tighter than the vendor made
it", and the number above records what it costs rather than justifying a shortcut.

> [!NOTE]
> Nothing in this finding is inherited any more. It established that a read barrier stands in
> for its neighbours; finding 25 turns that into the rule basalt places them by, and the
> issue rate above replaced the last thing the scheduler copied from the vendor's word.

## 14. A modifier is one bit, and it is nowhere near its operand

`IADD R5, R4, -R0` and `IADD R5, R4, R0` differ by one character of text and one bit of
encoding, and that bit is not in the operand's field:

| Operand | Register number | Negate |
| :--- | :--- | :--- |
| source 1 | bits 24:31 | **bit 72** |
| source 2 | bits 32:39 | **bit 63** |

The prober groups both into the same slot, because flipping either one changes that
operand's text. That grouping is correct and too coarse to write through: an assembler
handed `R5` against a form harvested as `-R0` sees a nine-bit field, no way to say which
part is the sign, and refuses.

Telling them apart costs nothing extra. The prober already records the operand text with
each bit clear and set, so a bit whose flip adds or removes a leading `-` is the negate, and
a bit whose flip changes `R4` to `R5` is the value. Across the database that separates 276
negate fields, 135 absolute, 104 invert and 11 bitwise-not from the values beside them, none
of them guessed.

Two details matter more than they look.

**The polarity is read off the form, not assumed.** Whether bit 72 set means negated is not
something to take on faith from the order the prober happened to record a pair in. The
reference text says whether its own encoding is negated, so the bit that has to change is
simply the opposite of whatever the reference holds.

**An unreadable bit cancels the split.** A bracket operand can lose a bit and lose only the
part it belonged to; the offset stays writable when a bank bit is unattributed. A value
cannot, because its bits are one number: writing a register number into the readable
fraction of a field encodes a different register, in silence. So if any bit in a field could
not be read, the field goes back to being whole and the assembler refuses it.

The effect on the corpus, at `ptxas -O3`, is 54 more instructions assembled bit-identically
and 54 fewer refusals:

| | Exact | Refused | Wrong |
| :--- | ---: | ---: | ---: |
| Before | 8,374 | 210 | 0 |
| After | **8,428** | 156 | **0** |

## 15. A mnemonic that depends on its operand's value hides the operand

Differential probing rests on one assumption: change a bit, and whatever moves in the
printed text is what that bit controls. `IMAD.SHL.U32` breaks it.

That mnemonic is not a separate instruction. It is what the disassembler calls `IMAD` when
the multiplier is a power of two, because then the multiply is a shift. Writing values
straight into the immediate field of one shows it plainly:

| Immediate written | Decoded as |
| :--- | :--- |
| `0x10` | `IMAD.SHL.U32 R22, R2, 0x10, RZ` |
| `0x20` | `IMAD.SHL.U32 R22, R2, 0x20, RZ` |
| `0x10000000` | `IMAD.SHL.U32 R22, R2, 0x10000000, RZ` |
| `0x30` | **`IMAD.U32`** `R22, R2, 0x30, RZ` |
| `0x5` | **`IMAD.U32`** `R22, R2, 0x5, RZ` |

Same bits, same field, different name, and the name depends on the value. So every single
flip of that immediate changes the suffix, the probe records all thirty-two bits as suffix
bits rather than operand bits, and the operand ends up with no field at all:

| Form | Bits attributed to the immediate |
| :--- | ---: |
| `IMAD R8, R2, 0x5, RZ` | 30 |
| `IMAD.U32 R12, R11, 0x10000, RZ` | 31 |
| `IMAD.SHL.U32 R22, R2, 0x10, RZ` | **0** |

The assembler then refused every `IMAD.SHL.U32` in the corpus, correctly, for want of
anywhere to put its operand. It was the largest single group of refusals there was.

**Recovering it without assuming it.** The tempting fix is to copy the field from a sibling
form, and that is exactly the kind of reasoning this repository refuses. What is done
instead: a bit that changed a suffix *and* moved exactly one operand is a candidate, and a
candidate is only accepted if writing several distinct values through the candidate bits
reproduces those values exactly and leaves every other operand untouched. A bit that really
does control a modifier fails that check and stays where it was. Only immediates are
recovered this way, because a register that moves a suffix is a genuinely different encoding
rather than a rendering of the same one.

| | Exact | Refused | Wrong |
| :--- | ---: | ---: | ---: |
| Before finding 14 | 8,374 | 210 | 0 |
| After finding 14 | 8,428 | 156 | 0 |
| After this | 8,479 | 105 | 0 |
| After signed immediates | 8,491 | 93 | 0 |
| After merging the candidates | **8,514** | 70 | **0** |

The last two rows are the same idea applied twice more. A minus on a *number* is part of
the number rather than a bit somewhere else in the word, and treating `-0x1` like `-R0`
refused every subtract-by-add in the corpus. And the recovery above only worked where the
hidden bits were the whole field: in `IMAD R8, R2, 0x5, RZ` they are bits 32 and 34, exactly
the two that are set, because clearing either leaves a power of two. Two bits of a 32-bit
field cannot reproduce a written value on their own, so the candidates have to be merged
with the bits already attributed to that operand before being checked.

## 16. When basalt says a schedule is unsafe, is it?

The positive control says basalt stays quiet on correct code. The negative control says it
complains about one deliberately broken instruction. Neither asks the question a user
actually has, which is whether a hazard basalt reports is a hazard the silicon agrees with.

`scripts/agreement_sweep.py` asks it across the corpus. Take the vendor's own working
schedule, shorten one stall on a real dependency, and collect two independent verdicts: what
basalt says statically, and what the GPU computes against eight input patterns.

| | GPU agrees with the reference | GPU computes something else |
| :--- | ---: | ---: |
| **basalt: clean** | agreed safe | **missed** |
| **basalt: hazard** | over-strict | agreed broken |

The top-right cell is the one that matters. A miss is a schedule basalt called safe that
computes a wrong answer, which is the entire failure this repository exists to prevent.

| Shortened-dependency study | Kernels |
| :--- | ---: |
| Kernels, one dependency shortened in each | 233 |
| Agreed broken | 79 |
| Over-strict | 90 |
| **Missed** | **0** |
| Unstable, excluded | 64 |

Which kernels land in the excluded column moves between runs, because several read memory
this harness never initialises and are therefore stable only until something else has used
the card. The share is larger than it was: the corpus now covers immediate-source and fp64
arithmetic, which read more uninitialised input than the register forms did. The missed
count does not move.

**Kernels with a loop are left out of this sweep, and not for tidiness.** This is the one
tool that breaks a kernel on purpose, and a loop keeps its trip count in a register:
shorten the stall in front of that register and the bound becomes whatever was stale there,
so the kernel never returns. `s_nested_loops` does exactly that. On a card that is also
driving a display, a kernel that never returns is a driver reset and a black screen, which
is not a price worth paying for seven more rows. The round trip still covers every one of
them, because it breaks nothing: it asks whether basalt's schedule computes what the
vendor's schedule computes, and both terminate.

**It did not start at zero.** The first run of this sweep reported **34 misses**, all of the
same shape: `IADD -> STG` shortened from 5 cycles to 1, computing a different answer, with
basalt reporting a warning rather than an error. The cause was that severity was decided by
the *producer's* generic latency confidence rather than by the evidence behind the
requirement that was actually used. `IADD` has no measured latency entry, so a shortfall
against a number mined from 44 of the vendor's own schedules came out as a suspicion.
Severity now follows the requirement's own provenance: measured or mined from at least three
vendor schedules is an error, an assumed producer latency is still a warning.

**What that cost.** Precision. Before the change, 57 shortened dependencies were called clean
and the GPU tolerated them; after it, they are flagged. Those became over-strict verdicts
rather than misses, which is the right direction on an architecture with no interlock, and
it is a real cost rather than a free win.

**Over-strict is not the same as wrong.** A schedule can be tighter than anything the vendor
emits and still return the right answer, because a stale read only changes the result when
the stale value and the fresh one differ. Running eight patterns instead of one moves
verdicts out of over-strict and into agreed broken, which are cases where basalt was right
and a single pattern had not been enough to show it. The rest are unproven either way, and
they are counted separately rather than folded into an accuracy figure.

## 17. Going looking for a wrong word, rather than waiting for one

"Nothing assembles to the wrong bytes" was measured against instructions `ptxas` chose to
emit. That is the right control and it is a narrow one: the failure this repository exists
to catch does not need the vendor's cooperation, and a corpus can only ever ask about text
the compiler happened to produce.

`scripts/fuzz_assembler.py` goes looking instead. It mutates the operand text of every form
in the database, assembles the result, and hands it straight back to the decoder, holding
one property:

> assemble(text) must disassemble to text

The interesting part is telling a real defect from a difference in spelling, because most
mismatches are the second. `P7` and `PT` are one register. `URZ` prints as `UR63`. A
negative immediate prints as its two's complement. A discarded result is not printed at all,
so `VOTE.ANY RZ, PT, P0` reads back as `VOTE.ANY PT, P0`. And a suffix can follow an operand's
value, which is finding 15 again. None of those are wrong words, and the test that separates
them from one is whether what came back assembles to the *same* word.

**It found four defects in the first thirty seconds**, none of which the corpus had ever
reached:

| What broke | Why |
| :--- | :--- |
| `PLOP3.LUT`'s predicate slot turned `P1` into `UP0` | five bits holding a uniform-register flag, a number and a negate; splitting the negate off left four bits that are not a number |
| `IADD R4, R0, -0x1` refused | a minus on a *number* is part of the number, not a bit elsewhere in the word |
| an immediate too wide for its field was trimmed | the value was masked at the call site before anything could refuse it |
| `DFMA`/`FSEL` took an integer for a float field | splitting a modifier out of the field bypassed the check on what the field holds |

The first is the one that matters: it wrote a word naming a different register, which is
exactly the failure mode being hunted. It survived the corpus because the corpus only ever
puts `PT` in that slot.

After the fixes, at 120 mutations per form across five seeds:

| Assembler fuzzing | Result |
| :--- | ---: |
| Mutations | 219,000 |
| Assembled and round-tripped | all of them |
| Same word, printed differently | counted separately |
| **Wrong** | **0** |

The seed is fixed so a failure reproduces exactly, and three seeds run in CI on every push.

## 18. The sign of a float immediate is part of the number

`FSEL R5, R0, -2.875, P0` and `FSEL R5, R0, 2.875, P0` differ by one bit, and basalt wrote
the wrong one:

```
vendor  000fe20000000000c038000000057808
basalt  000fe200000000004038000000057808
                        ^ bit 63
```

The cause is in the sub-field classifier from finding 14. That splits a modifier out of an
operand field by watching what a bit does to the printed text, and a bit whose flip adds a
leading `-` is the negate. For `-R0` that is exactly right: the sign of a register lives in
a bit of its own, nowhere near the register number. For `-2.875` it is exactly wrong. The
sign of a float is IEEE bit 31, inside the value, and calling it a modifier left the value
31 bits wide. The assembler then wrote 31 bits of a 32-bit float and dropped the sign.

The rule is one line: a leading `-` on a *literal* belongs to the literal. It already held
for integers, where `-0x1` is two's complement in its own field, and the guard simply never
covered floats because no corpus kernel had produced a negative float immediate.

**What surfaced it is the point.** No reasoning found this. The corpus was widened to cover
immediate-source arithmetic, which had never been harvested at all, and the count of words
that assemble to something other than the vendor's bytes went from 0 to 2. That number is
pinned at zero precisely so that a change like this cannot land quietly.

## 19. A register wearing a modifier is not a literal

One mnemonic covers several encodings, and the database holds one entry per operand *shape*
so it can tell them apart. The shape is what the operands are, ignoring their values:
`FADD Rd, Ra, Rb` and `FADD Rd, Ra, imm` differ in bits outside every operand field, so a
form harvested as one describes the other only by accident.

`|R0|` was read as a literal. It is not; it is a register carrying an absolute-value bit,
the same way `-R0` carries a negate. So these two collapsed into one bucket:

| Text | Read as | Actually |
| :--- | :--- | :--- |
| `FADD R4, -RZ, \|R0\|` | reg, reg, immediate | reg, reg, **reg** with a modifier |
| `FADD R7, R2, -24` | reg, reg, immediate | reg, reg, immediate |

Probing is one nvdisasm run per shape, so a bucket gets one representative. Whichever of the
two arrived first won it, and the other had **no encoding in the database at all** and had
to be refused. The same collision hid `c[0x3][R5]` behind `c[0x0][0x380]`: a register index
and a literal index behind identical brackets.

Fixing it is a matter of stripping a modifier before asking what an operand is, in both the
harvester and the assembler, which had disagreed with each other about `|R0|` as well. On
the same corpus:

| | Exact | Refused | Wrong |
| :--- | ---: | ---: | ---: |
| Before | 9,837 | 17 | **2** |
| After | **9,846** | 10 | **0** |

The ten that remain are bits the prober could not attribute to anything, spread across five
`RET.REL.NODEC`, three `LDC` base fields, one `FADD` and one `IADD3`. Those refuse rather
than guess, which is the designed answer.

## 20. Kernels that never compiled, and the count that hid them

`ptxas` rejecting a snippet is an ordinary result here. The corpus is deliberately broad and
tries forms the architecture may not have, so a rejection is recorded as a negative rather
than raised, and the harvest prints how many there were and carries on. That is the right
design, and it hid ten separate gaps for as long as they had existed:

| Kernels | What was wrong | What it cost |
| :--- | :--- | :--- |
| all 22 half precision | `ld.global.f16`, a load type PTX does not have | `HADD2`, `HMUL2`, `HFMA2`, `HMNMX2` and the f16 conversions, absent from the database, the latency model and the mined stall table alike |
| `popc.b64`, `clz.b64` | a 64-bit destination for a count that is 32 bits wide | both 64-bit forms |
| 12 bitwise atomics | `.and`, `.or`, `.xor` and `.exch` given `.s32`/`.u32`/`.u64`, when PTX types them as bit operations and takes only `.b32` or `.b64` | every bitwise `ATOMG` |
| `div.f32`, `div.f64` | float division has no default rounding mode in PTX and has to name one | both |
| 3 widening `cvt` | a rounding modifier on an exact conversion, which PTX rejects | f16 to f32, f16 to f64, f32 to f64 |
| `mma.m16n8k16.f16.f16.f16` | `.f16` accumulate given `.f32` registers, when it packs two halves into a `.b32` | the only f16-accumulating tensor form |
| 3 `ldmatrix.m16n16.b8` | one register per matrix, when a 16x16 tile of bytes is two, and `.x4` does not exist for the shape | `LDSM.8.MT1616` |
| 3 sparse `mma.sp` | B sized for the dense shape rather than the full k, and `.kind::f8f6f4` carried over from the dense forms, which the sparse ones reject | `HMMA.SP`, `IMMA.SP` and `QMMA.SP` |
| 1-bit `mma` | no `.and.popc` or `.xor.popc`, which the 1-bit form requires | see below |
| shared atomics | `_atomic`'s `space` parameter was never passed anything but its default | every `ATOMS` |
| reductions | `red`, an atomic that returns nothing, was never generated at all | `REDG` and `REDUX` |

The half-precision case is the one worth dwelling on. The type table had carried `b16` as
f16's container since it was written, in a column that nothing read; `_load` interpolated the
arithmetic type straight into the instruction instead. So the corpus claimed a whole
precision class and covered none of it.

**A corpus bug and a form the architecture does not have look identical from outside.** Both
are one more rejected snippet. The difference is in the reason, which was being recorded and
never read. So the reasons are now sorted: a parse error, a type mismatch or an arguments
mismatch means the PTX is wrong, and anything else is a genuine negative. The harvest counts
the first kind separately and a test fails if there are any.

That test found `popc`, `clz` and `mma` on its first run, none of which had anything to do
with the half-precision bug that prompted it. Reading the rest of the reasons rather than
counting them found the atomics, the divisions, the conversions and the matrix shapes.

**What the 1-bit form turned out to be.** `mma.b1` compiles for sm_120a and there is no
1-bit tensor instruction behind it. `ptxas` calls an internal helper that decomposes the
operation into `LOP3` masks, `MOVM.U4TO8.M832` and a chain of `IMMA.16832.U8.U8`. So the
capability is present at the PTX level and emulated underneath, which is only visible
because the kernel now builds.

**An async copy is a third dependency mechanism.** `cp.async` lowers to `LDGSTS`, and it
signals no scoreboard at all. Completion is tracked by a separate pair, `LDGDEPBAR` marking
the group and `DEPBAR.LE SB0, 0x0` waiting for the outstanding copies to drain, and only
then is the shared read safe:

```
LDGSTS.E [R7], desc[UR4][R2.64]     no write barrier at all
LDGDEPBAR                           signals SB0
DEPBAR.LE SB0, 0x0                  waits for the copies to drain
BAR.SYNC.DEFER_BLOCKING 0x0
LDS R9, [R7]                        reads what the copy wrote
```

basalt models the stall count and the scoreboard, and neither connects the copy to the read.
Given a fresh schedule those kernels return different bytes on the card at every optimisation
level, which is how this was found rather than argued: the round trip went from every
comparable kernel matching to three that did not, the moment the corpus first emitted one.
The corpus no longer emits `cp.async` and the roadmap says why, because a kernel the
scheduler is known to get wrong would either break that control or need an excuse carved
into it.

**Six rejections remain and all of them are real.** Five are the `mxf8f6f4` block-scaled
family with a `ue4m3` scale, which is refused at every register count and scale vector while
the same forms with `ue8m0` build; one is sparse `e2m1` at `m16n8k128`, which demands
`.kind::f8f6f4` and then refuses the type. Those are answers about sm_120a rather than
defects in the corpus.

| | Mnemonics | Opcodes | Instructions reproduced |
| :--- | ---: | ---: | ---: |
| Before | 276 | 77 | 9,846 of 9,856 |
| After | **345** | **90** | **12,395 of 12,400** |

## 21. A corpus of two-instruction kernels overestimates what the compiler requires

basalt mines its stall requirements from the vendor's own scheduling: the tightest gap
`ptxas` was ever seen to leave for a pairing is a lower bound on what that pairing needs.
The premise was written down alongside a caveat, that the compiler might simply be
conservative and basalt would then be merely strict, and one assurance:

> If the compiler were ever wrong in the other direction the positive control would already
> be failing.

It was not failing because it was not looking. The control checked `-O3` only, and only the
generated corpus, never the hand-written kernels with loops and barriers in them. Widening it
to every kernel at every optimisation level, and adding five workloads shaped like code
somebody would run, made it fail immediately:

| Pairing | Mined floor | What the vendor actually did | Observations behind the floor |
| :--- | ---: | ---: | ---: |
| `FADD -> *` | 5 | **4** | 24 |
| `ULEA -> *` | 9 | **7** | 54 |
| `CS2R -> *` | 16 | **13** | 3 |

**Why a narrow corpus reads high.** A kernel with two instructions of body gives `ptxas`
nothing to fill a gap with. It leaves whatever spacing is convenient, and that spacing is an
upper bound on the requirement wearing the clothes of a lower one. Give it a tiled matrix
multiply and it packs independent work into the same gaps, and the floor it cannot go below
is the one that shows.

Re-mining with those workloads present moved `FADD` from 5 to 4 across 90 observations
instead of 24, and `ULEA` from 9 to 7 across 93. The `FADD` number now agrees with fault
injection, which had said 4 all along in finding 4: the mined table had been quietly
contradicting the measured one, and the checker was believing the wrong half.

**Three observations is not evidence.** `CS2R` had exactly three, said 16, and the
instruction's measured latency is 4. Beside it in the same table, `CS2R.32` had three
observations and said 7, and `FADD.RM`, `.RP` and `.RZ` each had three and said 5 where
plain `FADD` had ninety and said 4. The threshold for calling mined evidence trustworthy was
three, and at three the numbers disagree with each other.

Raising it to eight costs less than it looks. It demotes a requirement to a warning rather
than deleting it, and the negative control is where the cost would show: nothing moved into
the missed column, which stayed at zero.

| | Errors on vendor output | Kernel/level pairs checked | Dependencies |
| :--- | ---: | ---: | ---: |
| Before | not measured at `-O1`, or on any shape kernel | 423 | 7,325 |
| After | **0** | **1,323** | **30,421** |

## 22. A loop-carried dependency is not a distance

`DFMA` to `DFMA` mines at 64 cycles across 194 observations, and in `s_loop_double` at `-O1`
the vendor leaves 18. Both are correct, because they are not the same situation.

```
#8   DFMA R8, R6, R8, R4     stall=3  writes SB2   waits on SB1, SB2, SB5
#9   UISETP.GE.AND ...       stall=9
#10  BRA.U !UP0              stall=5   <- back edge to #8
```

The `DFMA` waits on scoreboard 2, which is the barrier its own previous iteration signalled.
That is what carries the dependency around the back edge, and 18 cycles is all the spacing it
needs on top. In a straight line there is no previous iteration to have signalled anything,
so `ptxas` covers the same dependency by padding with `NOP`s to the full 64 (finding 4). The
mined figure is the straight-line habit, and applying it to the loop reported the vendor's own
working kernel as understalled.

The checker already declined to compare a gap that crossed a block boundary, for exactly this
reason: it is a minimum over paths rather than a distance. An instruction reaching *itself* is
that same case and was not recognised as one, because it can only do so around a back edge.
Saying so is one condition, and it is the difference between a control that passes and a
control that is right.

## 23. An operand read is not instantaneous, and nothing was modelling that

`ULEA UR5, UR12, UR4, 0x18` reads `UR4`. The next instruction, `UMOV UR4, URZ`,
overwrites it. `ptxas` leaves three cycles between them, and basalt's scheduler left one,
which is the entire difference between a tiled matrix multiply that computes the right
answer and one that does not.

Shortening that gap on the card and nothing else:

| `stall` on the reader | result |
| ---: | :--- |
| **0** | correct |
| 1 | **wrong** |
| 2 | **wrong** |
| 3 | correct |
| 4 to 7 | correct |

The same shape as finding 1, and for the same reason: zero is the long safe encoding, and
the first values above it are the ones that corrupt. Three cycles is the requirement *for
this pairing*, and it is a requirement basalt had no concept of.

**It is a pairing, not a constant, and reading it as a constant was the first mistake made
with it.** basalt charged three cycles for every anti-dependency, which passed everything,
because being too generous is the safe direction. Checking that against the vendor says
otherwise: across the corpus `ptxas` leaves one or two cycles here 386 times, on hundreds of
distinct pairings, and it is not wrong 386 times.

So the gap is mined per pairing, exactly as the read-after-write requirement is, and the
numbers spread the way every other latency in this project does:

| Pairing | Cycles the vendor never goes below | Observations |
| :--- | ---: | ---: |
| `FFMA` ~> `FFMA` | 1 | 116 |
| `IMAD.SHL.U32` ~> `SHF` | 1 | 84 |
| `IADD` ~> `IADD` | 1 | 47 |
| `IMAD` ~> `IADD` | 2 | 13 |
| `STS.128` ~> `LDSM` | 19 | 24 |
| `ULEA` ~> `UMOV` | *3, and one observation* | 1 |

Evidence only ever lowers the charge, never raises it. The mined minimum is the smallest gap
the vendor left, and that gap is frequently there for another reason entirely: `DFMA` into
`DFMA` mines at 64, which is the *result* latency of the instruction before it and nothing
to do with the read. Charging that would quadruple the cost of every fp64 kernel to cover a
hazard already covered.

`ULEA` into `UMOV` has one observation, so it is not trusted, so it keeps the constant, which
is the number fault injection measured for it. That is not a coincidence worth leaning on:
it is what the fallback is for.

The checker reports this hazard only where the evidence is trusted. A constant there would
call `ptxas` broken 386 times, and a control that fires on the reference is not a control.
The scheduler stays stricter than the checker on purpose: it charges three where it has no
evidence, and the checker says nothing, because being conservative about what to emit and
being conservative about what to allege are different jobs.

**Every dependency basalt modelled was a read after a write.** A value is produced, and a
later instruction must not read it too early. This is the other direction: a value is read,
and a later instruction must not overwrite it too early. The hardware does not check that
one either, and the corruption is the same kind, silent and repeatable.

It went unnoticed because a corpus of two-instruction kernels almost never overwrites a
register it has just read. Real code does it constantly, because registers are scarce and a
compiler reuses them the moment they are dead.

This is the same gap finding 13 names from the other side. Read barriers exist because an
instruction can consume its operands late, and at this point basalt inherited them rather
than computing them. Here `ptxas` protected the same thing with spacing rather than a
barrier, so there was nothing to inherit and the gap became visible as a wrong answer. Three
cycles is the first measurement of that latency, and it is what finding 25 needed to stop
inheriting the barriers as well.

## 24. An instruction that waits on the scoreboard it signals

`s_tile_matmul` did not round-trip at `-O2` or `-O3`. Substituting one half of the control
word at a time puts it in the scoreboards rather than the spacing: basalt's stalls alone
compute the vendor's answer, and so do its reuse flags.

The thing that named it was asking what shape basalt emits that `ptxas` never does.

> Across the kernel, `ptxas` has no instruction that waits on the scoreboard it signals.
> basalt has three.

The wait is evaluated at issue and the signal follows, so on a counter model that shape looks
harmless. **It is harmless, and reading it as a defect was a mistake worth keeping in the
record**, because the diagnosis it led to was right and the rule it produced was not.

`ptxas` emits that shape 251 times in 37,008 instructions: 206 on a write barrier and 45 on a
read barrier. `LDC.64 R2, c[0x0][0x388]` waits on SB0 and then signals SB0, which is reusing a
number that has just drained rather than waiting on its own result. The observation above was
true of that kernel and false in general, and the sample that produced it was one kernel.

What the shape was, was a *symptom*. Three separate causes were feeding it, all three real.

**A cubin holds more than its entry function.** `mma.b1` lowers to a call into an internal
helper, and 125 of that kernel's 144 instructions sit in bodies with no edge from the entry
block. The dataflow never reaches them, so every wait basalt emitted there came from
inheritance rather than analysis. Those blocks now keep the vendor's words untouched, which
is the honest thing to do with code the analysis has not read.

**The read-barrier inheritance was program-wide.** basalt kept the vendor's read-barrier
numbering at the time, and decided which vendor wait bits to carry over from a single mask of
every read-barrier number used anywhere in the program. So a vendor wait on a *write* barrier
was inherited whenever its number happened to match a read barrier somewhere else. Scoping
that to where a barrier is actually live fixed it; finding 25 later removed the inheritance
altogether, which is the better answer and needed this one first to be reachable.

**The allocator could not see the waits.** Scoreboards are allocated in one pass and the
waits are computed by a dataflow in the next, so an instruction is given a number before
anything knows what it will wait on. The fix at the time was a repair loop: allocate, look at
what came back waiting on its own barrier, and allocate again away from it.

| | Round trip |
| :--- | :--- |
| Before | `-O1` clean, one kernel short at `-O2` and `-O3` |
| After | **every comparable kernel, at all three levels** |

### The repair loop did not survive its own premise

Once the vendor was checked properly and found emitting the shape 251 times, the loop had
nothing left to justify it, and the honest test is whether removing it changes an answer.
It does not: 439 of 439 at every optimisation level with the loop gone, on the same eight
input patterns.

It was not free, either. Forbidding a number the instruction happens to wait on pushes the
allocator onto a fresh scoreboard each time, and `s_tile_matmul` was using 57 where it needs
20. That pressure is what forces sharing elsewhere, so a rule invented to avoid an imagined
hazard was manufacturing a real cost.

What did the work, then, was the other two causes: unreachable blocks are no longer given
computed waits, and read barriers are no longer inherited at all (finding 25). Both were
found by chasing the symptom, which is the argument for chasing symptoms and against
promoting one into a rule before checking how often the reference produces it.

## 25. Deriving the read barrier instead of copying it

For most of this project basalt computed the stall counts, the write barriers and the waits,
and copied the read barrier straight out of the vendor's word. That is a real limit rather
than an untidy one. A tool that can only place three of the four fields cannot schedule a
program nobody has compiled, which is the whole point of the assembler, and a round trip
that carries a field across unchanged is not evidence about that field.

The question is what a read barrier is *for*, and the corpus answers it. Across 37,008
instructions of `ptxas` output at three optimisation levels there are **299** read barriers,
which is few enough to characterise exhaustively. **285 of them are on variable-latency
instructions and the other 14 are on stores.** So the rule is not about memory, and not about
any particular opcode: an instruction whose *result* the hardware makes you wait for is also
one whose *operands* it has not finished taking at issue, and a store is the same case with
no result to wait for. Nothing outside those two sets ever carries one.

That says who can need a barrier. When they actually need one takes a second measurement.
**All 299 sit on an instruction whose source register is overwritten somewhere**: 233 later in
the same block, and 66 only on a path through another block, which is the loop-carried case
and the one a per-block pass cannot see at all.

The interesting cell is the other one, the 318 in-block overwrites `ptxas` leaves without a
barrier. Three mechanisms account for every one of them, and they add up exactly:

| How `ptxas` covers it instead | Count |
| :--- | ---: |
| A wait on the reader's own write barrier, which cannot clear before the read | 246 |
| A barrier on a *later* late reader, which covers this one too (finding 13) | 12 |
| Distance alone, averaging 31 cycles | 60 |
| **Total** | **318** |

Nothing is left over, which is the part worth insisting on. The third row is the same
mechanism finding 23 measures from the other side, and the first row is the one that matters
most because it is the largest: a wait placed for the *result* of a load also guarantees that
the load has taken its address.

> [!NOTE]
> Every figure in this finding comes from `python scripts/corpus_figures.py`, which recomputes
> them from a fresh compile. They were hand-counted once and had already drifted twice before
> that script existed, and one of them was not drift but a wrong claim, which is the more
> expensive kind.

```
#3 LDG.E.64 R4, [R2.64]            rb 7    <- reads R2, no barrier
#4 LDG.E.64 R6, [R2.64+0x8]        rb 0    <- reads R2, signals SB0
#5 LDC.64    R2, c[0x0][0x388]     wait 0x01   <- overwrites R2
```

**The rule basalt now uses**, stated in full: walk the graph forwards to a fixed point,
tracking for each register the latest variable-latency instruction or store that read it. An
instruction overwriting that register opens a window, unless it already waits on the
reader's write barrier. Give each window the lowest scoreboard no write barrier is
outstanding across, and put the wait on the overwriter.

The fixed point over the graph is not decoration. `s_tile_matmul` needs read barriers on
three `STS` instructions whose registers are overwritten by *the next iteration of the
loop*, and a pass that walks each block once sees an outstanding read at the end of the body
and no overwrite anywhere. Those three are every read barrier that kernel has.

**How it was checked.** The strongest available test is that removing them breaks something:

| Schedule | `s_tile_matmul` at `-O3` |
| :--- | :--- |
| computed read barriers | matches the vendor |
| no read barriers at all | **wrong answer on the card** |
| vendor's read barriers copied across | matches the vendor |

So the derived barriers are load-bearing rather than decorative, and the whole corpus agrees
at all three optimisation levels with nothing inherited. basalt does not reproduce the
vendor's *choice* of barrier everywhere; it over-approximates, because a scoreboard is a
counter and sharing one over-synchronises rather than corrupting. What it no longer does is
copy an answer it could not derive.

### The defect this exposed on the way

Deriving read barriers needs free scoreboards, and basalt did not have any: it was assigning
a write barrier to every variable-latency instruction with a result, whether or not anything
ever waited on it. Fixing that meant knowing which producer a wait was *for*, which the
dataflow could not say, because it tracked scoreboard numbers rather than the instructions
that signalled them, and one number is reused several times in a kernel.

Re-keying it on the producer exposed a second thing, and this one is a correctness rule
worth stating on its own. A scoreboard is a counter, so a wait placed for one producer
drains every producer sharing that number. Crediting only the producer the wait was written
for therefore calls the others unwaited-for, and dropping *their* barriers takes away
synchronisation that downstream code was leaning on. `s_tile_matmul` returned the wrong
answer on the card until the credit followed the counter rather than the intent.

## 26. The yield bit is a hint, and that is a measurement rather than an assumption

Five of the six control fields decide whether a program is correct. The sixth, one bit at
109, is described everywhere as a hint to the warp scheduler, and basalt set it by a rule
nobody had checked: yield when the stall is exactly 1. That agrees with `ptxas` on 73.2% of
37,008 instructions, which is not a model, it is a coincidence that held for the commonest
stall value.

Two questions, and they need different kinds of evidence.

**Does it affect the answer?** Take the vendor's own schedule, invert every yield bit in it,
and run both on the card against eight input patterns.

| Kernel | Level | Bits inverted | Result |
| :--- | :--- | ---: | :--- |
| `k_fma_rn_f32` | -O1, -O3 | 24, 24 | same |
| `k_sqrt_rn_f32` | -O1, -O3 | 56, 56 | same |
| `k_rcp_rn_f64` | -O1, -O3 | 144, 144 | same |
| `k_atom_shared_exch_b32` | -O1, -O3 | 24, 24 | same |
| `s_loop_double` | -O1, -O3 | 24, 40 | same |
| `s_tile_matmul` | -O1, -O3 | 48, 72 | same |

680 inversions across the fp64, transcendental, tensor, loop, barrier and shared-atomic
kernels, and every one produced the vendor's answer. So on this part the bit does not gate
correctness, and basalt is free to choose it on any grounds it likes. Worth saying plainly,
because "it is only a hint" is the kind of claim that gets repeated without anyone checking,
and a scheduler that writes it is one experiment away from knowing.

**What does `ptxas` actually do with it?** Fit against the same 37,008 instructions:

| Rule | Agreement |
| :--- | ---: |
| yield when the stall is 1 *(what basalt did)* | 73.2% |
| yield whenever the stall is not 0 | 92.1% |
| **yield when the stall is 1 to 11** | **93.7%** |
| yield when the stall is 1 to 8, or 11 | 93.9% |

basalt takes `1 <= stall < 12`. The last row fits marginally better and is worse: it is two
disjoint intervals to buy 0.2%, which is a rule shaped by the residuals rather than by
anything.

The remaining 6.3% is one-directional. Every disagreement is the vendor declining to yield
where the rule says it would, never the reverse, and the obvious explanations do not hold:
the operand reuse cache does not require it (all 212 instructions with a reuse flag also
have the bit set, so reuse *implies* yield rather than forbidding it), and neither "the next
instruction consumes this result" nor "this instruction waits on a scoreboard" improves the
fit. It looks like an occupancy heuristic, and there is no evidence here that would let it
be reconstructed, so it is left as a stated residual rather than guessed at.

## 27. A scoreboard named in an operand, and the third way of waiting

`cp.async` was out of the corpus for most of this project, with a stated reason: it lowers to
`LDGSTS`, the copy signals no scoreboard, and giving one a fresh schedule returned different
bytes on the card at every optimisation level. The reason was right about the symptom and
wrong about the cause, which is a good argument for going back to limitations that were
written down once and left alone.

Here is what `ptxas` emits, with the control fields printed beside it:

```
#12  LDGSTS.E [R7], desc[UR6][R2.64]     stall 4  wait 0x08  wb 7   <- signals nothing
#13  LDGDEPBAR                           stall 1             wb 0   <- signals SB0
#14  DEPBAR.LE SB0, 0x0                  stall 4             wb 7   <- waits for SB0
#15  BAR.SYNC.DEFER_BLOCKING 0x0         stall 6
#16  LDS R9, [R7]                        stall 4             wb 4   <- reads what was copied
```

The copy really does signal nothing, and completion really is tracked by a separate pair of
instructions. But that is a mechanism, not an obstacle: basalt never reorders, so it only has
to leave the pair intact. What actually broke it is on line 14. **`DEPBAR` names its
scoreboard in the operand text.** Every other wait in the architecture is the six-bit mask in
the control word, which basalt rewrites; this one is `SB0` printed in the disassembly and
encoded in the operand field. So renumbering `LDGDEPBAR`'s write barrier leaves `DEPBAR`
waiting on a scoreboard nothing signals any more, and there is nothing in the control word
to notice it by. basalt then went one worse: with nothing waiting on SB0 in the mask, the
pass that reclaims barriers nobody waits for deleted the signal as well.

**The rule.** A scoreboard named in an operand is pinned to the vendor's number, and that
number is unavailable to the allocator from the signaller until the instruction that names
it. Twenty lines, and it generalises past `cp.async` to anything else that waits by name.

With that in place `cp.async` is back in the corpus, in four forms across two cache
modifiers and three widths, and all twelve kernel and optimisation-level pairs match the
vendor's output exactly.

The general lesson is worth more than the fix. basalt rewrites a control word on the
assumption that the control word is where control lives. `DEPBAR` is a counter-example
sitting in the instruction stream, and the way to find the rest of them is to look for
operands that name a piece of scheduling state rather than a register.

## 28. The schedule is a property of the architecture, not of the part

The largest caveat on everything here is that it was measured on one card, and the obvious
way to lift it is to buy more cards. That is the wrong instrument. The question underneath is
not "what does this part do" but "is the required schedule a property of the part", and the
compiler answers that one without any card at all, because on sm_120 the compiler is what
carries the schedule. The hardware checks nothing.

`ptxas` targets six members of this family. `sm_120a` is architecture-specific, `sm_120f`
family-specific, `sm_120` the portable subset, and `sm_121` is a **different chip** with the
same three suffixes. If the requirement varied across them, the emitted control words would
have to vary. Compile the whole corpus for each and compare:

| Target | Kernels built | Identical to `sm_120a` | Different code | Same code, different schedule |
| :--- | ---: | ---: | ---: | ---: |
| `sm_120` | 1,215 | 1,215 | 0 | **0** |
| `sm_120f` | 1,323 | 1,323 | 0 | **0** |
| `sm_121a` | 1,323 | 1,323 | 0 | **0** |
| `sm_121` | 1,215 | 1,215 | 0 | **0** |
| `sm_121f` | 1,323 | 1,323 | 0 | **0** |

Every kernel every target builds comes out **byte-identical, control words included**. The
108 the bare names decline are the tensor forms that need the `a` suffix, and they are
declined at compile time rather than scheduled differently.

This is the negative result the roadmap said was worth having. `ptxas` does not tune its
scheduling per part in this family: it applies one model to all six targets, so the mined
half of basalt's model, 56 of 87 opcodes, is derived from output that is identical for every
one of them. What remains card-specific is the 11 latencies measured on silicon, which exist
as a cross-check on the mined numbers rather than as the source of them.

Stated carefully, because it is easy to overclaim. This does not prove the silicon is
identical. It proves that if two parts in this family had different scheduling requirements,
NVIDIA's own compiler would be emitting an unsafe schedule for one of them. That is a
different sentence and a much stronger one than anything a second card would have bought.

    python scripts/across_the_family.py --opt 1 2 3

## 29. Deriving the instruction set twice, from two compilers

basalt's instruction model comes from differential probing of one `ptxas` build, and one
measurement is not a result. Every bit role it assigns could be a property of that build, and
nothing inside the pipeline would notice. So the model was derived again, independently, from
CUDA 13.0.3 against the 13.3.1 it ships with, and the two compared.

The comparison has to be on the right thing. Two releases disagree on **195 of 343** shared
encodings, which sounds alarming and means nothing: an encoding carries the control bits and
the branch target of whichever kernel the form was harvested from, so `EXIT` reads
`000fc0...` under one and `000fea...` under the other and the difference is entirely
scheduling. What matters is the model: which bits are the opcode, which are modifiers, and
which bits each operand occupies.

| Compared on | Forms differing |
| :--- | ---: |
| The exemplar encoding | 195 |
| The derived bit model | 38 |
| **The operand model** | **0** |

Not one operand field, sub-field or modifier bit differs across 343 shared forms. The 38 are
all the same shape: a bit reads as opcode under one exemplar and as something else under the
other, which is a property of the exemplar rather than of the architecture, since which flips
change the mnemonic depends on the rest of the word.

The stronger test is whether one model can read the other's output, and it can:

| Assembling CUDA 13.0.3 output with the 13.3.1 database | Exact | Refused | Wrong |
| :--- | ---: | ---: | ---: |
| `-O1` | 12,145 of 12,168 | 23 | **0** |
| `-O3` | 12,351 of 12,360 | 9 | **0** |

Two forms exist only in 13.3.1 and one only in 13.0.3, which is a code generator choosing
`IADD3.X` where the other chose `IADD.X` rather than an instruction set changing.

    python scripts/cross_toolchain.py

## 32. Pointing the checker at code basalt did not produce

Every control up to here runs on kernels the corpus compiled. That establishes the checker
agrees with `ptxas` on `ptxas` output, which is necessary and is not the question anyone
holding a cubin actually has. Stage 10 asks the real one.

The subject is every sm_120 kernel NVIDIA ships in CUDA 13.3.1: `cublas`, `cublasLt`,
`cusolver`, `cusolverMg`, `cusparse`, `npp`, `curand` and `nvjpeg`, 2,473 cubins and 844 MB of
device code, scheduled by a compiler basalt has never seen the output of, at instruction mixes
the corpus does not reach.

```bash
python scripts/fetch_toolchain.py --libs
python scripts/audit_shipped.py --libs third_party/cuda/13.3.1/libs --report audit.json
```

**The first run reported 6,593 errors in 250 nvjpeg kernels.** Twenty-six per kernel, in
production code that decodes JPEGs correctly on every Blackwell card sold. Nothing in
NVIDIA's code was wrong. Eight things in basalt's model were, and no other control could have
found any of them, because every one is invisible on a corpus basalt generates itself.

| # | What was wrong | Why the corpus could not show it |
| :-- | :--- | :--- |
| 1 | `F2IP` read `F2I`'s entry and was called variable latency | `F2IP` appears nowhere in 37,008 instructions of corpus output |
| 2 | `R2UR` was called variable latency | all 3,098 corpus instances carry the safe stall encoding, which exempts them |
| 3 | a mined figure could exceed the producer's own latency | four observations said `IADD -> MOV` needs 20 and the vendor ships it at 5 |
| 4 | a guard's could exceed the 13 cycles measured for one | 9 observations said 18, and the corpus never wrote a tighter one to disagree |
| 5 | the scoreboard residue was whatever was mined | `LDG.E.64 => FFMA` mined at 461 cycles from 3 observations |
| 6 | a missing scoreboard was always an error | the vendor covers `LDS` with spacing alone 734,837 times |
| 7 | where it does, nothing checked the distance instead | `DSETP -> FSEL` sits 66 cycles apart 4,322 times and nothing was reading the 66 |
| 8 | `@P0` was treated as reaching `@!P0` | a generated corpus almost never writes complementary guards |

**The corpus-mined requirement table cannot fail on the corpus it came from.** That is the
structural point, and it is why 1,323 kernels of positive control kept passing while the model
carried all eight. The tightest gap `ptxas` was seen to leave *is* the floor, by construction,
for exactly the code it was measured on. Finding 21 caught the narrow version of this: `FADD`
read 5 from a thin corpus and 4 from a wider one, and 4 was what fault injection said. Stage
10 is the same finding three orders of magnitude out.

### The requirement, re-mined from a corpus a thousand times larger

`scripts/mine_shipped.py` mines the same per-pair gaps out of shipped kernels, **holding out
the libraries the audit then reports on**, because a table measured on the code it is checked
against is the flaw being fixed rather than a fix for it. 909 cubins, 24,311 kernels.

| Requirement table | Corpus | With shipped code | New pairings | Read too high |
| :--- | ---: | ---: | ---: | ---: |
| dependent pairs | 340 | **3,957** | 3,617 | 66 |
| per producer | 162 | **512** | 350 | 50 |
| scoreboard residue | 255 | **1,850** | 1,595 | 51 |
| anti-dependency | 256 | **6,177** | 5,921 | 73 |
| issue rate | 1,146 | **30,735** | 29,589 | 337 |

The corroboration is the part worth reading. Where the wider corpus disagrees with the narrow
one, it lands on the number basalt had already measured on silicon by an entirely different
method:

| Pairing | Corpus | Shipped | Independently |
| :--- | ---: | ---: | :--- |
| `ISETP -> @BRA` | 18, from 9 observations | **13**, from 229,567 | 13 is what fault injection measured for a guard (finding 9) |
| `FADD -> FMUL` | 18, from 15 | **4**, from 18,663 | 4 is `FADD`'s measured latency (finding 4) |
| `IADD -> MOV` | 20, from 4 | **5**, from 17,426 | above `IADD`'s measured 4, as it has to be |

A guard predicate needing thirteen cycles was measured on this card by shortening the gap
until the answer changed. Two hundred and twenty-nine thousand independent scheduling
decisions by NVIDIA's compiler, in code written years apart from that experiment, put the
floor at the same thirteen.

### Whether a missing scoreboard is a hazard is measurable

The checker used to treat any variable-latency producer without a `write_barrier` as an error.
Mining separates the two cases cleanly, since a producer that signalled a barrier is recorded
in one table and one that did not in another, so the question has an answer:

| Producer | Covered by a scoreboard | Covered by spacing alone |
| :--- | ---: | ---: |
| `LDG` | 1,363,466 | **0** |
| `LDC` | 532,295 | **0** |
| `LDL` | 144,529 | **0** |
| `S2R` | 135,333 | **0** |
| `LD` | 99,975 | **0** |
| `LDS` | 1,972,507 | **734,837** |
| `SHFL` | 228,185 | **59,347** |
| `I2F` | 77,816 | **15,186** |

<img src="assets/chart-spacing.svg" alt="LDG, LDC, LDL, S2R and LD are covered by a scoreboard in every dependent pair and never by spacing alone. LDS is covered by spacing alone 27 percent of the time, SHFL 21 percent and I2F 16 percent.">

A global load is never once covered without a barrier across 1.3 million dependent pairs, and
a shared load is covered that way in one case in four. basalt's classification is exactly right
for global, constant, local and special-register reads, and saying the same thing about a
shared load is alleging a race in code that demonstrably works. Severity now follows the
evidence rather than the class, and where the vendor does cover a result with spacing the
distance is checked rather than merely noted.

### Then the same mistake, one order of magnitude up

At that point the audit read: zero errors over 250 kernels. It was not evidence, for exactly
the reason the whole finding is about. Widening the held-out set from one library to three,
from 250 kernels to **2,762**, and from 64 thousand instructions to **5.2 million**, took the
error count from 0 to **940**.

Five more corrections came out of it, and again none of them was NVIDIA's:

| What was wrong | The evidence |
| :--- | :--- |
| `DMMA` was variable where every other MMA is fixed | 892 alleged missing waits in one accumulate chain |
| a distance-covered producer was grounded by its own measured latency | the latency is not the pairing, and nothing bounds the pairing from above |
| the guard's 13 cycles were applied to a uniform predicate | measured for a vector predicate; shipped code resolves a uniform one in 11 |
| a load's destination was treated as overwritten at issue | it lands when the memory returns, so the gap to it is not the distance |
| the anti-dependency constant was charged on every pairing | 3 was measured on `ULEA ~> UMOV` alone, and 2,591 trusted pairings go below it |

One latent defect fell out on the way, and it had nothing to do with shipped code: a wait
rebuilt each reaching definition with four of its six fields, so `yielded` and `crossed` reset
every time. A kernel that waits on a scoreboard after a branch, which is most fp64 code, was
being judged against a gap that is a minimum over paths rather than a distance.

### All three tools, on code none of them produced

The checker is the one this stage exists for, and the same libraries answer the same question
about the other two. The assembler is handed each shipped kernel's disassembly and has to
reproduce the vendor's exact 128 bits, and the scheduler is handed each shipped kernel with
every control bit discarded and has to compute new ones.

| Library | Instructions | Assembled exactly | Refused | Wrong |
| :--- | ---: | ---: | ---: | ---: |
| `nvjpeg` | 63,968 | 48,810 | 15,158 | **0** |
| `curand` | 642,960 | 573,761 | 69,199 | **0** |
| `cusolverMg` | 4,530,520 | 3,962,765 | 567,755 | **0** |
| **Total** | **5,237,448** | **4,585,336 (87.5%)** | 652,112 | **0** |

<img src="assets/chart-assembler.svg" alt="59,693 of 59,760 corpus instructions and 4,585,336 of 5,237,448 shipped instructions assembled to the vendor's exact bytes, the rest refused by name, and zero of all 5,297,208 assembled to the wrong bytes.">

```bash
python scripts/assembler_coverage.py --cubins <dir> --show-refusals
python scripts/audit_shipped.py --cubins <dir> --exclude <the mined ones> --reschedule
```

Every refusal names the field it could not place, and the reasons are coverage rather than
correctness: `LDG.E.U8`, `BSSY.RECONVERGENT`, `SEL.64` and `LDCU.128` are forms no kernel
basalt generates has ever emitted, so the prober has never seen one. The count that is pinned
at zero is the third column, and it stays at zero on 5.2 million instructions of machine code
the assembler had never met. One defect turned up getting there and it was the right kind:
`c[0x0][UR4]` indexes its offset by a register where the recorded form holds a number, and the
encoder raised `ValueError` instead of refusing by name. A crash on foreign input is worse
than a wrong verdict, because the caller gets neither.

### The result

| Run | Kernels | Instructions | Dependencies | Errors | Warnings |
| :--- | ---: | ---: | ---: | ---: | ---: |
| First run, one library | 250 | 63,968 | 93,972 | 6,593 | 554 |
| Widened to three | 2,762 | 5,237,448 | 10,218,030 | 940 | 217,855 |
| **After thirteen corrections** | **2,762** | **5,237,448** | **10,218,030** | **0** | **395** |

<img src="assets/chart-audit.svg" alt="The first run reported 6,593 errors over 250 kernels and 93,972 dependencies. Widened to three libraries, 2,762 kernels and 10,218,030 dependencies, it reported 940. After thirteen corrections the same corpus reported 0.">

**2,762 of 2,762 kernels fully analysed**, no indirect branch leaving an edge the dataflow
could not follow, and 0 of 5,237,448 instructions failing to decode. A checker that quietly
skipped a tenth of the code could report the same zero, so the audit prints how much of each
library it actually reached.

### Every control, on one commit, from a clean slate

```bash
python scripts/verify_all.py
```

| Control | Answer |
| :--- | :--- |
| lint, format, types on two platforms | clean |
| both oracles | `ptxas` V13.3.73 |
| ISA rebuild has not drifted | 345 forms committed, 345 rebuilt, compared by encoding |
| measured fields still behave | 845 of 866 register slots controllable |
| assembler at `-O0`, `-O1`, `-O2`, `-O3` | 22,714/22,752, 12,189/12,208, 12,395/12,400, 12,395/12,400 |
| assembler under mutation | every word that assembled decoded back to the text asked for |
| one schedule per family target | the schedule is a property of the architecture, not the part |
| round trip on the card at `-O1`, `-O2`, `-O3` | **439 of 439** comparable kernels match the vendor, each level |
| the checker against the silicon | 233 kernels, one dependency shortened in each, **0 missed** |
| shipped libraries audit | **0 errors** |
| the suite | passed |

**19 controls, 16.0 minutes, all passing.** That is the state the numbers above describe, and
the command that reproduces the set is the same one that produced it.

**What that is and is not.** It is not a clean bill of health for NVIDIA's code and is not
meant to be. The finding is that a checker built and calibrated entirely on one body of code
reported twenty-six hazards per kernel the first time it saw another, that widening the second
body by a factor of eighty found five more model errors after the first eight were fixed, and
that every single one of the thirteen was its own. Anyone auditing machine code against a model
tuned on a corpus should expect the same, and this is the measurement of how much.

The 395 warnings are not noise either. They are the places where the model has an assumed
number and says so, or where shipped code schedules tighter than the vendor's own habit
elsewhere, which is the honest thing for a tool to say where it cannot ground a claim. What
basalt declines to say is part of the result: 945 of the first run's warnings were the gap
beside a waited-on scoreboard, which carries no requirement at all because the wait is the
mechanism, and 217,855 of the wide run's were a read barrier the vendor covers with spacing
instead. Both are gone, because a tool that cannot ground a claim should not make it.

## Where these findings sit against published work

Not every finding here establishes something, and the ones that do not are more useful for it.
The premise underlying [finding 3](#3-understalling-corrupts-silently-and-basalt-predicts-exactly-when),
that sm_120 does not interlock and a short stall count reads a stale register with no fault, was
published before this work, as was the fp64 figure in
[finding 4](#4-required-stall-by-three-independent-methods): 63.57 cycles by Jarmusch, Graddon
and Chandrasekaran in July 2025, and 64.13 by a cycle-level characterisation in May 2026,
against the 63.99 measured here. Three parties, three methods, three parts, one number. Those
findings reproduce independently rather than establish, and are stronger evidence for it.

What no published work carried as of August 2026 is the per-pair requirement the rest of these
findings build, or a checker validated on one against code held out of every table it reads.
The full comparison, with links, is in [the method](METHOD.md#prior-and-concurrent-work-on-sm_120).

## 30. What is deliberately not claimed

Stated so the boundary of the evidence is visible.

- **`I2FP` and `F2I` cannot be separated by timing.** A conversion cannot feed the next link
  of a dependent chain without converting back, so the chain always contains one of each. The
  24-cycle figure is the pair, and it is recorded in a separate `composite` section rather
  than halved and presented as a per-instruction latency. Fault injection cannot separate them
  either, because the round trip is idempotent and the chain reaches a fixed point.
- **Only one card has been measured, and that matters less than it did.** Everything measured
  on silicon is a Gigabyte GeForce RTX 5070 Ti EAGLE OC, 70 SMs, 2542 MHz boost, named exactly
  because "a 5070 Ti" is not enough to reproduce a run. Finding 28 removes most of the sting:
  `ptxas` emits byte-identical schedules for all six targets in this family, `sm_121` included,
  so the mined model is derived from output that does not vary by part. What is still specific
  to this card is the 11 latencies measured on it, and those exist to cross-check the mined
  numbers rather than to be the source of them. basalt records the part alongside every
  measurement so a second card can be compared rather than merged.
- **The scoreboard residual is not checked across a block boundary.** A waited-on
  scoreboard still leaves a gap the producer has to cover (finding 7), and that gap is
  mined one block at a time because a distance that spans a branch depends on which path
  was taken. So a definition that reaches its consumer through an edge is exempted from
  that one rule rather than judged against evidence collected somewhere looser. Reaching
  definitions carry a `crossed` flag for exactly this, and every other rule still applies
  to them. It is a gap in the checker's coverage, not a wrong answer, and the round trip
  covers the same ground from the other side.
- **Five instructions per kernel set will not assemble, and the reason is the prober rather
  than the assembler.** `RET.REL.NODEC R4`, `WARPSYNC.COLLECTIVE R12` and a `BRA` on `!P2` all
  need a register or a predicate field the differential probe never isolated, because flipping
  a bit of those forms moves the printed branch target as well as the register and two operands
  moving at once is not a reading. The obvious fix is to decode the base word at the same
  address as each mutant, since the batch decodes as one blob and `nvdisasm` prints an absolute
  target; that was tried, and the two still differ by a constant, so a `RET`'s printed operand
  is not a plain field read and the probe is not measuring what it looks like it is measuring.
  Reverted rather than kept, because it changed no outcome: 345 forms either way, and the same
  22,714 of 22,752 at `-O0`. The refusals stand and name the field they could not place.
- **Three opcodes the compiler emits have no form in the database.** `ptxas` emits `UIMAD`,
  `UISETP` and `USHF` from basalt's own corpus at `-O1` and above, and the differential probe
  has never isolated a form for any of them, so the database holds none. The consequence is
  coverage rather than correctness: asked for one, the assembler raises `AssemblyError` naming
  the opcode rather than guessing an encoding, and the latency model carries all three, so the
  checker still reasons about them where they appear in someone else's machine code. It is
  recorded here because nothing found it for a long time. The database was only ever compared
  against itself, and pointing `scripts/corpus_figures.py` at both directions at once, what
  the database holds and what the compiler actually emitted, is what surfaced it.
- **Ten opcodes have no evidence behind their latency**, out of the 134 the model now holds.
  The other 124 split into 11 measured on silicon, 91 mined from what the vendor schedules
  across shipped code as well as the corpus, and 22 with no register result at all, whose
  number is never consulted because an instruction that defines nothing is never a producer.
  The ten are `BMMA`, `F2IP`, `ISCADD`, `LDGSTS`, `LOP`, `RED`, `SULD`, `TEX`, `TLD` and
  `VABSDIFF`, none of which appears as a producer anywhere in the corpus or in 24,311 shipped
  kernels. `F2IP` is the interesting one: it is in the model because the audit found 302 of
  them, all in a library held out of the mining set, so its class is evidence and its number
  is not. The model marks every assumed number as assumed, and a hazard derived from one is
  reported as a warning rather than an error, because the difference between a lead and a
  finding is where the number came from.
- **Four input patterns were enough, and that was checked rather than assumed.** A stale read
  only changes the answer when the stale value and the fresh one differ, so the round trip's
  strength is bounded by how much its inputs disagree. Doubling them to eight, chosen to
  disagree with the first four as well as with each other, found nothing new: 439 of 439 at
  every optimisation level, the same two exclusions. It was not wasted, though. The *negative*
  control is not saturated: the extra patterns moved five verdicts out of over-strict and into
  agreed-broken, cases where basalt was right and four patterns had not been enough to show
  it. Both controls keep all eight, at 5.5 seconds on the card against 3.0.
- **A test that cannot detect corruption proves nothing.** An early version of the injection
  probe multiplied by 1.0000001, so a stale read rounded back to the same float and `FMUL`
  appeared to need only one cycle. Every probe now runs a sensitivity control first: a chain
  one link shorter must produce a different answer, established independently of the stall
  sweep. Without that control, "no value was unsafe" is ambiguous between "every value is
  genuinely safe" and "this kernel cannot tell", which are opposite claims.
- **`POPC` and `I2FP` fail that control**, because their chains reach a fixed point: `popc`
  of a `popc` stops changing, and an integer round trip through float is idempotent after the
  first conversion. An earlier run reported `I2FP` as requiring 4 cycles; the control
  retracted it, and it is listed as not established rather than quietly kept.

## 31. Corrections made along the way

Kept because a method is only as trustworthy as its error log.

- Scoreboards are counters, not flags. An early rule treated a second signal on the same
  scoreboard as a hazard; several producers sharing one is ordinary.
- A wait by any intervening instruction satisfies a dependency for everything downstream.
  Requiring each consumer to carry its own wait produced eight false findings in a
  thirty-two instruction kernel.
- A guard predicate is printed before the mnemonic, so reading the first token as the opcode
  silently misparses every predicated instruction.
- A scoreboard covers a dependency whatever the producer's latency class. Checking stalls
  only for fixed-latency producers missed that `ptxas` scoreboards fp64.
- `VOTEU` was classified as completing out of order. `ptxas` emits it with no scoreboard and
  reads the result on the next instruction, which settles it.
- The required stall belongs to the producer/consumer pair, not to the producer. `IMAD` into
  `IMAD` is scheduled at four cycles and `IMAD` into `IADD` at three.
- fp64 operands are register pairs with nothing in the mnemonic to say so, so half of every
  fp64 dependency was invisible.
- A guard predicate was charged the same as a predicate read as data. It needs thirteen
  cycles against five, and the difference silently corrupts (finding 9).
- A predicated write was treated as killing the previous definition of its register. It
  does not: on the path where the guard is false the earlier producer is what the next
  reader sees, and the dependency on it was being dropped.
- Requirements were keyed on the bare opcode. The modifier decides the number, and
  collapsing forms takes the minimum across them, so `I2F` read as 1 cycle on the strength
  of `I2F.RP` while every other form needs 2.
- The scheduler refused to schedule a seventh outstanding load rather than sharing a
  scoreboard. A scoreboard is a counter, so sharing is permitted and over-synchronises
  slightly; refusing rejected 45 of 317 corpus kernels.
- A scoreboard wait was treated as covering a dependency completely. It does not: the
  producer still owes a per-opcode minimum stall, 2 cycles for `DADD`, and one cycle less
  is silently wrong.
- fp64 was classified as fixed latency because an early experiment appeared to show the
  stalls carrying the dependency. The experiment also cut the stall on an unrelated address
  setup (finding 7).
- The `loop` kernel's failure to round-trip was recorded as a loop-carried scheduling gap
  for longer than it should have been. Bisecting the schedule one instruction at a time
  against hardware showed a single guard predicate, in a straight line of code, with
  nothing loop-carried about it.
- `DSETP -> SEL` was required to be six cycles apart on the strength of three observations.
  Widening the corpus to cover immediate-source arithmetic produced kernels where the
  vendor scheduled it at two, and the mined floor came down to match. Re-mining changed
  nothing about how many dependencies are checked, 6,105 either way; it turned four false
  errors into none.
- `F2FP` was reading `F2F`'s entry, the conversion pipe, which signals a scoreboard and
  completes out of order. `ptxas` emits `F2FP` with no scoreboard anywhere and reads the
  packed result five cycles later, so it cannot be. It is on the fixed pipeline, the same
  correction `I2FP` and `VOTEU` needed before it, and the half-precision kernels are what
  finally reached it.
- `HMNMX2` had no latency entry at all, while `HADD2`, `HMUL2` and `HFMA2` beside it did.
  Nothing had noticed because no kernel producing one ever compiled.

- A mined requirement was trusted on three observations. `CS2R` had three, said sixteen
  cycles, and the instruction's measured latency is four; the vendor's own thirteen came out
  as an error. Eight observations now, and nothing moved into the negative control's missed
  column when it changed.
- The anti-dependency requirement was keyed on the bare opcode when the exact form was
  missing. `IMAD.U32` is a shift the vendor schedules at one cycle and `IMAD` is a multiply
  it never schedules under four, so collapsing them called two of `ptxas`'s own kernels
  broken. Exact form only now, which is the same correction `I2F` needed on the other
  direction of dependency.
- The scheduler and the checker were given the anti-dependency rule separately, and the
  checker's was stricter. basalt then produced schedules that failed its own verifier, which
  is the failure mode the two sharing a model exists to prevent. One function now, read by
  both.
- An instruction reaching itself was checked as though the gap were a distance. It can only
  reach itself around a back edge, where the vendor leans on the scoreboard rather than
  padding to the full latency, and `DFMA` was being asked for 64 cycles where 18 was right.

- `F2IP` read `F2I`'s entry, the third instruction to fall into a trap two comments in the
  same file already warned about, and the only one of the three the corpus could not reach.
- A scoreboard was freed the moment an instruction read what it guarded, including by the
  instruction then allocating one, so a load could end up waiting on the barrier it signals.
  The probe had a standing check for that shape and it took tighter schedules to trip it.
- A definition guarded by `@P0` was treated as reaching a use guarded by `@!P0`. Within a
  thread exactly one of them runs.

Each of these was caught by the positive control: the vendor compiler's own output must
verify clean, and every one of them made it fail. The last three came from stage 10, where
the output belongs to someone else.
