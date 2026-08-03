//! `evaluate(grid, spec) -> Verdict` — the single function everything else in
//! this repository is defined against.
//!
//! Correctness and latency are measured in two separate passes, on purpose:
//!
//! * **Pass A** evaluates every input assignment from a cold start. Cold
//!   starts are order-free, so the truth table it produces cannot depend on
//!   which row happened to run first.
//! * **Pass B** walks the assignments in Gray-code order without resetting, so
//!   each step is a genuine single-input transition and the settle time is a
//!   genuine propagation delay. It also re-reads the outputs, which catches
//!   circuits whose answer depends on history — those are latches wearing a
//!   combinational costume, and v1 rejects them.

use crate::block::Block;
use crate::circuit::Circuit;
use crate::grid::{Grid, Pos, CELLS};
use crate::power::MAX_STRENGTH;
use crate::spec::Spec;
use crate::tick::{Settle, Sim, DEFAULT_MAX_GAME_TICKS};

/// One row of the truth table that came out wrong.
///
/// Row-level diagnosis is what makes the replay buffer of §06 useful: a
/// candidate that misses exactly one row is usually one or two blocks from
/// correct, and that is worth training on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowMismatch {
    /// Input assignment bitmask; input `k` is bit `k`.
    pub inputs: u64,
    pub observed: u64,
    pub expected: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintViolation {
    Latency { got: u32, max: u32 },
    Blocks { got: u16, max: u16 },
    Region { got: (u8, u8), max: (u8, u8) },
}

/// Why a grid was rejected before, or independently of, its truth table.
///
/// §03 lists four reasons; the extra three are ones the implementation
/// actually needs. `MaskedCell` catches a half-denoised diffusion sample,
/// `ExcludedBlock` catches v1-out-of-scope components, and `HistoryDependent`
/// catches sequential behaviour in a combinational spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MalformedReason {
    /// Dust in a cell with nothing solid beneath it.
    FloatingDust { at: Pos },
    /// A lever off the input face, a lamp off the output face, or a declared
    /// port whose cell holds the wrong block.
    PortViolation { at: Pos },
    /// A component whose support block is missing — a wall torch on air, a
    /// repeater over a hole. §03 calls this `Overlap`.
    Unsupported { at: Pos },
    /// A block that exists in the vocabulary but is out of scope for v1.
    ExcludedBlock { at: Pos },
    /// A control token (usually `MASK`) inside the grid body.
    MaskedCell { at: Pos },
    /// A torch toggled itself to death.
    Burnout { at: Pos },
    /// The output depends on the order inputs were applied.
    HistoryDependent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Pass {
        latency_rt: u8,
        blocks: u16,
        bbox: (u8, u8, u8),
    },
    Fail {
        mismatched_rows: Vec<RowMismatch>,
        constraint: Option<ConstraintViolation>,
    },
    /// Oscillating. Tracked separately from `Fail` because a rising unstable
    /// rate during self-training means something different from a rising
    /// failure rate — it is an early collapse warning.
    Unstable {
        period_ticks: u8,
    },
    Malformed {
        reason: MalformedReason,
    },
}

impl Verdict {
    pub fn is_pass(&self) -> bool {
        matches!(self, Verdict::Pass { .. })
    }

    /// How far from correct this candidate was, for ranking the replay buffer.
    /// A pass is 0; anything not even simulable is treated as maximally bad.
    pub fn mismatch_count(&self) -> usize {
        match self {
            Verdict::Pass { .. } => 0,
            Verdict::Fail { mismatched_rows, constraint } => {
                mismatched_rows.len() + usize::from(constraint.is_some())
            }
            Verdict::Unstable { .. } | Verdict::Malformed { .. } => usize::MAX,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Verdict::Pass { .. } => "pass",
            Verdict::Fail { .. } => "fail",
            Verdict::Unstable { .. } => "unstable",
            Verdict::Malformed { .. } => "malformed",
        }
    }
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Pass { latency_rt, blocks, bbox } => {
                write!(f, "PASS latency={latency_rt}rt blocks={blocks} bbox={bbox:?}")
            }
            Verdict::Fail { mismatched_rows, constraint } => {
                write!(f, "FAIL {} row(s)", mismatched_rows.len())?;
                if let Some(c) = constraint {
                    write!(f, " + constraint {c:?}")?;
                }
                Ok(())
            }
            Verdict::Unstable { period_ticks } => write!(f, "UNSTABLE period={period_ticks}gt"),
            Verdict::Malformed { reason } => write!(f, "MALFORMED {reason:?}"),
        }
    }
}

/// Cheap structural checks, run before any simulation.
///
/// Every one of these is a class of sample a generator will produce in bulk
/// early in training, and catching them here keeps the malformed rate a
/// measurable quantity rather than a mysterious source of failures.
pub fn check_malformed(grid: &Grid, spec: &Spec) -> Option<MalformedReason> {
    for i in 0..CELLS {
        let p = Pos::from_index(i);
        match grid.at(i) {
            Block::Wire => {
                if !grid.get(p.down()).supports_dust() {
                    return Some(MalformedReason::FloatingDust { at: p });
                }
            }
            Block::Torch { attach } => {
                let support = p.attached(attach);
                if !grid.get(support).is_conductive() {
                    return Some(MalformedReason::Unsupported { at: p });
                }
            }
            Block::Lever { attach } => {
                if !grid.get(p.step(attach)).is_conductive() {
                    return Some(MalformedReason::Unsupported { at: p });
                }
                if !spec.inputs.iter().any(|q| q.pos == p) {
                    return Some(MalformedReason::PortViolation { at: p });
                }
            }
            Block::Repeater { .. } | Block::Comparator { .. } => {
                if !grid.get(p.down()).is_opaque() {
                    return Some(MalformedReason::Unsupported { at: p });
                }
            }
            Block::Lamp => {
                if !spec.outputs.iter().any(|q| q.pos == p) {
                    return Some(MalformedReason::PortViolation { at: p });
                }
            }
            Block::Observer { .. } => {
                return Some(MalformedReason::ExcludedBlock { at: p });
            }
            _ => {}
        }
    }
    // Every declared port must actually exist, with the right block.
    for port in &spec.inputs {
        if !matches!(grid.get(port.pos), Block::Lever { .. }) {
            return Some(MalformedReason::PortViolation { at: port.pos });
        }
    }
    for port in &spec.outputs {
        if grid.get(port.pos) != Block::Lamp {
            return Some(MalformedReason::PortViolation { at: port.pos });
        }
    }
    None
}

fn settle_failure(s: &Settle) -> Option<Verdict> {
    match s {
        Settle::Settled { .. } => None,
        Settle::Oscillating { period_gt } => {
            Some(Verdict::Unstable { period_ticks: (*period_gt).min(255) as u8 })
        }
        // A run that never settles inside the cap is an oscillator whose
        // period is longer than the state-history window. Reporting it as
        // unstable with period 0 keeps it out of the `Fail` bucket, where it
        // would pollute the row-level diagnostics.
        Settle::Timeout { .. } => Some(Verdict::Unstable { period_ticks: 0 }),
        Settle::Burnout { at } => {
            Some(Verdict::Malformed { reason: MalformedReason::Burnout { at: *at } })
        }
    }
}

/// Apply an input assignment to the levers.
fn set_inputs(sim: &mut Sim, spec: &Spec, mask: usize) {
    for (k, port) in spec.inputs.iter().enumerate() {
        sim.set_lever(port.pos, mask >> k & 1 == 1);
    }
}

fn read_outputs(sim: &Sim, spec: &Spec) -> u64 {
    let mut out = 0u64;
    for (j, port) in spec.outputs.iter().enumerate() {
        if sim.lamp_lit(port.pos) {
            out |= 1 << j;
        }
    }
    out
}

/// Evaluate a token sequence. Rejects control tokens before decoding, which a
/// [`Grid`] cannot represent.
pub fn evaluate_tokens(tokens: &[u8], spec: &Spec) -> Verdict {
    for (i, &t) in tokens.iter().enumerate() {
        if t >= crate::block::CONTROL_BASE {
            return Verdict::Malformed {
                reason: MalformedReason::MaskedCell { at: Pos::from_index(i) },
            };
        }
    }
    match Grid::from_tokens(tokens) {
        Ok(g) => evaluate(&g, spec),
        Err(_) => {
            Verdict::Malformed { reason: MalformedReason::MaskedCell { at: Pos::from_index(0) } }
        }
    }
}

pub fn evaluate(grid: &Grid, spec: &Spec) -> Verdict {
    evaluate_with_cap(grid, spec, DEFAULT_MAX_GAME_TICKS)
}

pub fn evaluate_with_cap(grid: &Grid, spec: &Spec, max_game_ticks: u32) -> Verdict {
    if let Some(reason) = check_malformed(grid, spec) {
        return Verdict::Malformed { reason };
    }

    let blocks = grid.material_blocks();
    let bbox = grid.bbox();
    let n_rows = spec.n_rows();
    let mut sim = Sim::new(Circuit::new(grid.clone()));

    // --- pass A: cold start per assignment, order-free -------------------
    let mut observed = vec![0u64; n_rows];
    for (m, slot) in observed.iter_mut().enumerate() {
        sim.reset(false);
        set_inputs(&mut sim, spec, m);
        let s = sim.settle(max_game_ticks);
        if let Some(v) = settle_failure(&s) {
            return v;
        }
        *slot = read_outputs(&sim, spec);
    }

    let mismatched: Vec<RowMismatch> = observed
        .iter()
        .zip(spec.rows.iter())
        .enumerate()
        .filter(|(_, (got, want))| got != want)
        .map(|(m, (got, want))| RowMismatch { inputs: m as u64, observed: *got, expected: *want })
        .collect();
    if !mismatched.is_empty() {
        return Verdict::Fail { mismatched_rows: mismatched, constraint: None };
    }

    // --- pass B: Gray-code walk for latency and history independence -----
    let mut latency_rt = 0u32;
    sim.reset(false);
    set_inputs(&mut sim, spec, 0);
    let s = sim.settle(max_game_ticks);
    if let Some(v) = settle_failure(&s) {
        return v;
    }
    for k in 1..n_rows {
        let m = k ^ (k >> 1);
        set_inputs(&mut sim, spec, m);
        let s = sim.settle(max_game_ticks);
        if let Some(v) = settle_failure(&s) {
            return v;
        }
        if read_outputs(&sim, spec) != observed[m] {
            return Verdict::Malformed { reason: MalformedReason::HistoryDependent };
        }
        latency_rt = latency_rt.max(s.latency_rt());
    }

    // --- constraints ------------------------------------------------------
    let c = &spec.constraints;
    let violation = if let Some(max) = c.max_latency_rt.filter(|&max| latency_rt > max) {
        Some(ConstraintViolation::Latency { got: latency_rt, max })
    } else if let Some(max) = c.max_blocks.filter(|&max| blocks > max) {
        Some(ConstraintViolation::Blocks { got: blocks, max })
    } else if let Some((mx, mz)) = c.max_region.filter(|&(mx, mz)| bbox.0 > mx || bbox.2 > mz) {
        Some(ConstraintViolation::Region { got: (bbox.0, bbox.2), max: (mx, mz) })
    } else {
        None
    };
    if let Some(v) = violation {
        return Verdict::Fail { mismatched_rows: Vec::new(), constraint: Some(v) };
    }

    Verdict::Pass {
        latency_rt: latency_rt.min(MAX_STRENGTH as u32 * 17).min(255) as u8,
        blocks,
        bbox,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{Attach, Dir4};
    use crate::grid::LOGIC_Y;
    use crate::spec::{Constraints, Port};

    fn port(name: &str, x: i32, z: i32) -> Port {
        Port { name: name.into(), pos: Pos::new(x, LOGIC_Y as i32, z) }
    }

    /// lever -> block -> torch -> dust -> repeater -> lamp
    fn not_circuit() -> (Grid, Spec) {
        let y = LOGIC_Y as i32;
        let z = 8;
        let mut g = Grid::with_substrate();
        g.set(Pos::new(0, y, z), Block::Lever { attach: Dir4::East });
        g.set(Pos::new(1, y, z), Block::Solid);
        g.set(Pos::new(2, y, z), Block::Torch { attach: Attach::West });
        for x in 3..14 {
            g.set(Pos::new(x, y, z), Block::Wire);
        }
        g.set(Pos::new(14, y, z), Block::Repeater { facing: Dir4::East, delay: 1 });
        g.set(Pos::new(15, y, z), Block::Lamp);
        let spec = Spec::new(
            vec![port("A", 0, z)],
            vec![port("Q", 15, z)],
            vec![1, 0],
            Constraints::default(),
        )
        .unwrap();
        (g, spec)
    }

    #[test]
    fn a_correct_not_gate_passes() {
        let (g, spec) = not_circuit();
        let v = evaluate(&g, &spec);
        assert!(v.is_pass(), "{v}");
        if let Verdict::Pass { latency_rt, blocks, .. } = v {
            // torch (1rt) + repeater (1rt)
            assert_eq!(latency_rt, 2);
            assert!(blocks > 0);
        }
    }

    #[test]
    fn inverting_the_expected_table_fails_every_row() {
        let (g, mut spec) = not_circuit();
        spec.rows = vec![0, 1];
        match evaluate(&g, &spec) {
            Verdict::Fail { mismatched_rows, .. } => assert_eq!(mismatched_rows.len(), 2),
            other => panic!("expected FAIL, got {other}"),
        }
    }

    #[test]
    fn floating_dust_is_caught_before_simulation() {
        let (mut g, spec) = not_circuit();
        g.set(Pos::new(6, LOGIC_Y as i32 + 2, 3), Block::Wire);
        assert!(matches!(
            evaluate(&g, &spec),
            Verdict::Malformed { reason: MalformedReason::FloatingDust { .. } }
        ));
    }

    #[test]
    fn a_lever_off_the_input_face_is_a_port_violation() {
        let (mut g, spec) = not_circuit();
        g.set(Pos::new(6, LOGIC_Y as i32, 3), Block::Solid);
        g.set(Pos::new(5, LOGIC_Y as i32, 3), Block::Lever { attach: Dir4::East });
        assert!(matches!(
            evaluate(&g, &spec),
            Verdict::Malformed { reason: MalformedReason::PortViolation { .. } }
        ));
    }

    #[test]
    fn a_latency_constraint_can_reject_a_functionally_correct_circuit() {
        let (g, mut spec) = not_circuit();
        spec.constraints = Constraints { max_latency_rt: Some(1), ..Default::default() };
        match evaluate(&g, &spec) {
            Verdict::Fail {
                constraint: Some(ConstraintViolation::Latency { got, max }), ..
            } => {
                assert_eq!((got, max), (2, 1));
            }
            other => panic!("expected a latency violation, got {other}"),
        }
    }

    #[test]
    fn masked_cells_never_reach_the_simulator() {
        let (g, spec) = not_circuit();
        let mut tokens = g.to_tokens();
        tokens[900] = crate::block::TOK_MASK;
        assert!(matches!(
            evaluate_tokens(&tokens, &spec),
            Verdict::Malformed { reason: MalformedReason::MaskedCell { .. } }
        ));
    }
}
