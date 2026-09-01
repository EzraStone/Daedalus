//! The golden suite: hand-built circuits with hand-derived expected verdicts.
//!
//! This is the only thing standing between "the simulator is correct" and
//! "the simulator is self-consistent". Every case here was laid out by hand
//! and its expected verdict worked out from the rules in the module docs, not
//! from running the simulator and writing down what came out.
//!
//! The distinction matters more than it sounds. Three of these cases were
//! wrong on the first pass — dust approaching a block sideways does not power
//! it — and the suite is what caught the author, not the code.
//!
//! Two layers of assertion:
//!
//! * each case declares an expected verdict *class* (pass with a stated
//!   latency, fail on n rows, unstable, malformed with a stated reason);
//! * the exact rendered verdicts are snapshotted to
//!   `tests/golden/verdicts.txt`, so any change in behaviour shows up as a
//!   reviewable diff even when the class is unchanged.
//!
//! Re-bless the snapshot with `REDSIM_BLESS=1 cargo test --test golden`.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use redsim::block::{Attach, CmpMode, Dir4, Dir6};
use redsim::builder::Builder;
use redsim::grid::LOGIC_Y;
use redsim::spec::Constraints;
use redsim::{evaluate, evaluate_tokens, Block, Grid, MalformedReason, Pos, Spec, Verdict};

const SNAPSHOT: &str = include_str!("golden/verdicts.txt");
const SNAPSHOT_PATH: &str = "tests/golden/verdicts.txt";

/// What a case is expected to produce.
#[derive(Clone, Debug)]
enum Expect {
    /// Correct, with a hand-derived latency in redstone ticks.
    Pass(u8),
    /// Correct, latency not asserted (used where the interesting property is
    /// something other than timing).
    PassAny,
    /// Wrong on exactly `n` truth-table rows.
    Fail(usize),
    /// Wrong because a hard constraint was violated, not the truth table.
    FailConstraint,
    Unstable,
    Malformed(&'static str),
}

struct Case {
    name: String,
    grid: Grid,
    spec: Spec,
    expect: Expect,
}

fn reason_name(r: &MalformedReason) -> &'static str {
    match r {
        MalformedReason::FloatingDust { .. } => "floating_dust",
        MalformedReason::PortViolation { .. } => "port_violation",
        MalformedReason::Unsupported { .. } => "unsupported",
        MalformedReason::ExcludedBlock { .. } => "excluded_block",
        MalformedReason::MaskedCell { .. } => "masked_cell",
        MalformedReason::Burnout { .. } => "burnout",
        MalformedReason::HistoryDependent => "history_dependent",
    }
}

fn matches(expect: &Expect, v: &Verdict) -> Result<(), String> {
    match (expect, v) {
        (Expect::PassAny, Verdict::Pass { .. }) => Ok(()),
        (Expect::Pass(want), Verdict::Pass { latency_rt, .. }) => {
            if want == latency_rt {
                Ok(())
            } else {
                Err(format!("expected latency {want}rt, got {latency_rt}rt"))
            }
        }
        (Expect::Fail(n), Verdict::Fail { mismatched_rows, constraint: None }) => {
            if *n == mismatched_rows.len() {
                Ok(())
            } else {
                Err(format!("expected {n} bad row(s), got {}", mismatched_rows.len()))
            }
        }
        (Expect::FailConstraint, Verdict::Fail { constraint: Some(_), .. }) => Ok(()),
        (Expect::Unstable, Verdict::Unstable { .. }) => Ok(()),
        (Expect::Malformed(want), Verdict::Malformed { reason }) => {
            let got = reason_name(reason);
            if *want == got {
                Ok(())
            } else {
                Err(format!("expected malformed/{want}, got malformed/{got}"))
            }
        }
        _ => Err(format!("expected {expect:?}, got {v}")),
    }
}

// ---------------------------------------------------------------------------
// circuit fragments
// ---------------------------------------------------------------------------

/// `A -> dust -> repeater -> lamp`. The repeater costs 1 rt and nothing else
/// delays, so latency is exactly the repeater's delay.
fn buffer(z: i32, delay: u8) -> Case {
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 13, z);
    b.output_delay("Q", z, delay);
    let spec = b.spec(vec![0, 1], Constraints::default());
    Case {
        name: format!("buffer/row{z:02}/d{delay}"),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Pass(delay),
    }
}

/// `A -> dust -> torch -> dust -> repeater -> lamp`. Torch 1 rt, repeater 1 rt.
fn inverter(z: i32) -> Case {
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 3, z);
    b.invert(4, z);
    b.dust_x(6, 13, z);
    b.output("Q", z);
    let spec = b.spec(vec![1, 0], Constraints::default());
    Case { name: format!("not/row{z:02}"), grid: b.grid.clone(), spec, expect: Expect::Pass(2) }
}

/// A buffer whose dust takes a `k`-row detour, so total run length is `12 + k`
/// cells. Strength at the far end is `15 - (11 + k)`; the run dies once that
/// reaches zero, i.e. from `k = 4` onward.
fn detour(k: i32) -> Case {
    let z = 2;
    let zo = z + k;
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 7, z);
    b.dust_z(7, z, zo);
    b.dust_x(8, 13, zo);
    b.output("Q", zo);
    let alive = 15 - (11 + k) >= 1;
    let spec = b.spec(vec![0, 1], Constraints::default());
    Case {
        name: format!("dust_range/detour{k}"),
        grid: b.grid.clone(),
        spec,
        // When the signal dies the lamp is dark for both assignments, so only
        // the `A = 1` row is wrong.
        expect: if alive { Expect::Pass(1) } else { Expect::Fail(1) },
    }
}

/// `n` repeaters of delay `d` in series ahead of the output repeater.
fn repeater_chain(n: usize, d: u8) -> Case {
    let z = 8;
    let mut b = Builder::new();
    b.input("A", z);
    let mut x = 2;
    for _ in 0..n {
        b.dust_x(x, x + 2, z);
        b.repeat(x + 3, z, d);
        x += 4;
    }
    b.dust_x(x, 13, z);
    b.output("Q", z);
    let spec = b.spec(vec![0, 1], Constraints::default());
    Case {
        name: format!("repeater/chain{n}/d{d}"),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Pass(n as u8 * d + 1),
    }
}

/// Two inverted inputs merged onto one dust net: `!A | !B` = NAND.
fn nand2(za: i32, zb: i32, zo: i32) -> Case {
    let mut b = Builder::new();
    b.input("A", za);
    b.dust_x(2, 3, za);
    b.invert(4, za);
    b.dust_x(6, 7, za);

    b.input("B", zb);
    b.dust_x(2, 3, zb);
    b.invert(4, zb);
    b.dust_x(6, 7, zb);

    b.dust_z(7, za, zb);
    b.dust_x(8, 13, zo);
    b.output("Q", zo);
    let spec = b.spec(vec![1, 1, 1, 0], Constraints::default());
    Case {
        name: format!("nand2/{za}_{zb}_{zo}"),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Pass(2),
    }
}

/// Both inputs routed onto one dust net: a dust join is a free OR.
fn or2(za: i32, zb: i32, zo: i32) -> Case {
    let mut b = Builder::new();
    b.input("A", za);
    b.dust_x(2, 7, za);
    b.input("B", zb);
    b.dust_x(2, 7, zb);
    b.dust_z(7, za, zb);
    b.dust_x(8, 13, zo);
    b.output("Q", zo);
    let spec = b.spec(vec![0, 1, 1, 1], Constraints::default());
    Case {
        name: format!("or2/{za}_{zb}_{zo}"),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Pass(1),
    }
}

/// OR followed by an inverter.
fn nor2(za: i32, zb: i32, zo: i32) -> Case {
    let mut b = Builder::new();
    b.input("A", za);
    b.dust_x(2, 7, za);
    b.input("B", zb);
    b.dust_x(2, 7, zb);
    b.dust_z(7, za, zb);
    b.dust_x(8, 8, zo);
    b.invert(9, zo);
    b.dust_x(11, 13, zo);
    b.output("Q", zo);
    let spec = b.spec(vec![1, 0, 0, 0], Constraints::default());
    Case {
        name: format!("nor2/{za}_{zb}_{zo}"),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Pass(2),
    }
}

/// NAND followed by an inverter: `!(!A | !B)` = AND.
fn and2(za: i32, zb: i32, zo: i32) -> Case {
    let mut b = Builder::new();
    b.input("A", za);
    b.dust_x(2, 3, za);
    b.invert(4, za);
    b.dust_x(6, 7, za);

    b.input("B", zb);
    b.dust_x(2, 3, zb);
    b.invert(4, zb);
    b.dust_x(6, 7, zb);

    b.dust_z(7, za, zb);
    b.dust_x(8, 8, zo);
    b.invert(9, zo);
    b.dust_x(11, 13, zo);
    b.output("Q", zo);
    let spec = b.spec(vec![0, 0, 0, 1], Constraints::default());
    Case {
        name: format!("and2/{za}_{zb}_{zo}"),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Pass(3),
    }
}

/// Rows for the three-input gates.
///
/// The spacing is not arbitrary. A raw input arrives at the merge column
/// already down to strength 10, and every row of vertical travel costs another
/// point, so the outermost input has to stay within three rows of the output
/// or the signal dies before the output repeater. Two rows of clearance
/// between neighbouring runs keeps them from merging into one net.
const ROWS3: [(&str, i32); 3] = [("A", 4), ("B", 7), ("C", 10)];
const ROW3_OUT: i32 = 7;

/// Three inputs merged onto one net.
fn or3() -> Case {
    let mut b = Builder::new();
    for (name, z) in ROWS3 {
        b.input(name, z);
        b.dust_x(2, 7, z);
    }
    b.dust_z(7, ROWS3[0].1, ROWS3[2].1);
    b.dust_x(8, 13, ROW3_OUT);
    b.output("Q", ROW3_OUT);
    let spec = b.spec(vec![0, 1, 1, 1, 1, 1, 1, 1], Constraints::default());
    Case { name: "or3".into(), grid: b.grid.clone(), spec, expect: Expect::Pass(1) }
}

fn nor3() -> Case {
    let mut b = Builder::new();
    for (name, z) in ROWS3 {
        b.input(name, z);
        b.dust_x(2, 7, z);
    }
    b.dust_z(7, ROWS3[0].1, ROWS3[2].1);
    b.dust_x(8, 8, ROW3_OUT);
    b.invert(9, ROW3_OUT);
    b.dust_x(11, 13, ROW3_OUT);
    b.output("Q", ROW3_OUT);
    let spec = b.spec(vec![1, 0, 0, 0, 0, 0, 0, 0], Constraints::default());
    Case { name: "nor3".into(), grid: b.grid.clone(), spec, expect: Expect::Pass(2) }
}

fn nand3() -> Case {
    let mut b = Builder::new();
    for (name, z) in ROWS3 {
        b.input(name, z);
        b.dust_x(2, 3, z);
        b.invert(4, z);
        b.dust_x(6, 7, z);
    }
    b.dust_z(7, ROWS3[0].1, ROWS3[2].1);
    b.dust_x(8, 13, ROW3_OUT);
    b.output("Q", ROW3_OUT);
    let spec = b.spec(vec![1, 1, 1, 1, 1, 1, 1, 0], Constraints::default());
    Case { name: "nand3".into(), grid: b.grid.clone(), spec, expect: Expect::Pass(2) }
}

fn and3() -> Case {
    let mut b = Builder::new();
    for (name, z) in ROWS3 {
        b.input(name, z);
        b.dust_x(2, 3, z);
        b.invert(4, z);
        b.dust_x(6, 7, z);
    }
    b.dust_z(7, ROWS3[0].1, ROWS3[2].1);
    b.dust_x(8, 8, ROW3_OUT);
    b.invert(9, ROW3_OUT);
    b.dust_x(11, 13, ROW3_OUT);
    b.output("Q", ROW3_OUT);
    let spec = b.spec(vec![0, 0, 0, 0, 0, 0, 0, 1], Constraints::default());
    Case { name: "and3".into(), grid: b.grid.clone(), spec, expect: Expect::Pass(3) }
}

/// `Q = !A | B` — implication, built as an OR of an inverted input and a raw
/// one. Exercises two different arrival latencies on the same net.
fn imply() -> Case {
    let mut b = Builder::new();
    b.input("A", 4);
    b.dust_x(2, 3, 4);
    b.invert(4, 4);
    b.dust_x(6, 7, 4);

    b.input("B", 10);
    b.dust_x(2, 7, 10);

    b.dust_z(7, 4, 10);
    b.dust_x(8, 13, ROW3_OUT);
    b.output("Q", ROW3_OUT);
    // A is bit 0, B is bit 1: rows are A=0/B=0, A=1/B=0, A=0/B=1, A=1/B=1.
    let spec = b.spec(vec![1, 0, 1, 1], Constraints::default());
    Case { name: "imply".into(), grid: b.grid.clone(), spec, expect: Expect::Pass(2) }
}

/// Two independent single-input circuits sharing one grid.
fn dual_buffer() -> Case {
    let mut b = Builder::new();
    b.input("A", 3);
    b.dust_x(2, 13, 3);
    b.output("P", 3);
    b.input("B", 11);
    b.dust_x(2, 13, 11);
    b.output("Q", 11);
    // Two outputs: bit 0 is P = A, bit 1 is Q = B.
    let spec = b.spec(vec![0b00, 0b01, 0b10, 0b11], Constraints::default());
    Case {
        name: "multi_output/dual_buffer".into(),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Pass(1),
    }
}

/// Dust into a block, block out to dust. The block is only *weakly* powered,
/// so the far dust must stay dark and the `A = 1` row must fail.
fn weak_power_gap() -> Case {
    let z = 8;
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 6, z);
    b.solid(7, z);
    b.dust_x(8, 13, z);
    b.output("Q", z);
    let spec = b.spec(vec![0, 1], Constraints::default());
    Case { name: "power_tier/weak_gap".into(), grid: b.grid.clone(), spec, expect: Expect::Fail(1) }
}

/// The same gap bridged by a torch tower: the cap is *strongly* powered and
/// re-emits at 15. Inverting, so the spec is `Q = !A`.
fn strong_power_tower() -> Case {
    let z = 8;
    let y = LOGIC_Y as i32;
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 5, z);
    b.solid(6, z);
    // Floor torch under a solid cap, mounted on the substrate beside the
    // powered block; the cap re-emits into the dust one layer up.
    b.put(7, z, Block::Torch { attach: Attach::West });
    b.put_at(Pos::new(7, y + 1, z), Block::Solid);
    b.put_at(Pos::new(8, y + 1, z), Block::Wire);
    b.put_at(Pos::new(8, y, z), Block::Solid);
    b.dust_x(9, 13, z);
    b.put_at(Pos::new(9, y, z), Block::Wire);
    b.output("Q", z);
    let spec = b.spec(vec![1, 0], Constraints::default());
    Case {
        name: "power_tier/strong_tower".into(),
        grid: b.grid.clone(),
        spec,
        expect: Expect::PassAny,
    }
}

/// A repeater facing away from the source blocks the signal entirely.
fn repeater_blocks_reverse() -> Case {
    let z = 8;
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 6, z);
    b.put(7, z, Block::Repeater { facing: Dir4::West, delay: 1 });
    b.dust_x(8, 13, z);
    b.output("Q", z);
    let spec = b.spec(vec![0, 1], Constraints::default());
    Case {
        name: "repeater/blocks_reverse".into(),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Fail(1),
    }
}

/// A comparator in compare mode passes its rear signal through unchanged.
fn comparator_compare() -> Case {
    let z = 8;
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 6, z);
    b.compare(7, z, CmpMode::Compare);
    b.dust_x(8, 13, z);
    b.output("Q", z);
    let spec = b.spec(vec![0, 1], Constraints::default());
    // Comparator 1 rt + output repeater 1 rt.
    Case {
        name: "comparator/compare_passthrough".into(),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Pass(2),
    }
}

/// Subtract mode with no side input is also a pass-through.
fn comparator_subtract() -> Case {
    let z = 8;
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 6, z);
    b.compare(7, z, CmpMode::Subtract);
    b.dust_x(8, 13, z);
    b.output("Q", z);
    let spec = b.spec(vec![0, 1], Constraints::default());
    Case {
        name: "comparator/subtract_no_side".into(),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Pass(2),
    }
}

/// Subtract mode with a full-strength side input: the side always wins, so the
/// output is dark for every assignment and the `A = 1` row fails.
fn comparator_subtract_side() -> Case {
    let z = 8;
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 6, z);
    b.compare(7, z, CmpMode::Subtract);
    // B feeds the comparator's north face. It has to come in on a distant row
    // and turn up the x=7 column: run it alongside the rear line and the two
    // nets simply merge, because adjacent dust is one net.
    b.input("B", z - 4);
    b.dust_x(2, 7, z - 4);
    b.dust_z(7, z - 4, z - 1);
    b.dust_x(8, 13, z);
    b.output("Q", z);
    // A is bit 0, B is bit 1. The rear arrives at 11 and the side at 7, so
    // B = 1 drops the output to 4 — not zero, but far too weak to survive the
    // six-cell run to the output repeater. The lamp goes dark either way.
    let spec = b.spec(vec![0, 1, 0, 0], Constraints::default());
    Case {
        name: "comparator/subtract_side_weakens".into(),
        grid: b.grid.clone(),
        spec,
        expect: Expect::PassAny,
    }
}

/// A target block is inert in v1 — behaviourally a solid block — so it forms
/// the same weak-power gap as stone does.
fn target_is_inert() -> Case {
    let z = 8;
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 6, z);
    b.target(7, z);
    b.dust_x(8, 13, z);
    b.output("Q", z);
    let spec = b.spec(vec![0, 1], Constraints::default());
    Case { name: "target/inert".into(), grid: b.grid.clone(), spec, expect: Expect::Fail(1) }
}

/// A ring of `n` inverters. Odd rings oscillate; even rings settle, and an
/// even ring that settles has an output that depends on nothing, so the spec
/// `Q = A` fails on one row.
fn torch_ring(n: usize) -> Case {
    let z = 5;
    let mut b = Builder::new();
    // The ring is deliberately not wired to the output; the circuit under
    // test is a working buffer that happens to share the grid with a clock.
    // An unstable verdict must win over a correct truth table.
    b.input("A", 12);
    b.dust_x(2, 13, 12);
    b.output("Q", 12);

    // A chain of inverters whose output loops back to the first one's support
    // block. The return path runs two rows clear and comes in from the north,
    // so the final cell has a single connection and points into the block.
    let mut x = 2;
    for _ in 0..n {
        b.invert(x, z);
        b.dust_x(x + 2, x + 2, z);
        x += 3;
    }
    let last = x - 1;
    b.dust_z(last, z, z + 2);
    b.dust_x(2, last, z + 2);
    b.dust_z(2, z + 1, z + 2);
    let spec = b.spec(vec![0, 1], Constraints::default());
    Case {
        name: format!("unstable/torch_ring{n}"),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Unstable,
    }
}

/// Dust running head-on into a bare lamp. The last cell has exactly one
/// connection, so the line rule points it at the lamp. Nothing here is
/// delayed, which makes this the one case with 0 rt of latency.
fn bare_lamp_line() -> Case {
    let z = 8;
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 14, z);
    b.bare_output("Q", z);
    let spec = b.spec(vec![0, 1], Constraints::default());
    Case {
        name: "dust_pointing/lamp_head_on".into(),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Pass(0),
    }
}

/// The same lamp, approached by a dust line running *past* it. The final cell
/// has two connections, so it points north-south and the lamp stays dark.
/// This is the trap that caught three hand-built fixtures during development.
fn bare_lamp_sideways() -> Case {
    let z = 8;
    let mut b = Builder::new();
    b.input("A", z - 1);
    b.dust_x(2, 14, z - 1);
    b.dust_z(14, z - 1, z + 1);
    b.bare_output("Q", z);
    let spec = b.spec(vec![0, 1], Constraints::default());
    Case {
        name: "dust_pointing/lamp_sideways".into(),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Fail(1),
    }
}

/// A dust run that climbs one layer and comes back down, exercising both
/// slope rules in a single circuit.
fn dust_ramp() -> Case {
    let z = 8;
    let y = LOGIC_Y as i32;
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 3, z);
    for x in 4..=5 {
        b.solid(x, z);
        b.put_at(Pos::new(x, y + 1, z), Block::Wire);
    }
    b.dust_x(6, 13, z);
    b.output("Q", z);
    let spec = b.spec(vec![0, 1], Constraints::default());
    Case {
        name: "dust/ramp_up_and_down".into(),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Pass(1),
    }
}

/// Two inverters chained through block adjacency rather than dust: the first
/// torch weakly powers the second torch's support block. Real behaviour, and a
/// common source of accidental coupling in generated layouts.
fn torch_adjacent_block_interference() -> Case {
    let z = 8;
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 3, z);
    b.invert(4, z); // solid(4), torch(5)
    b.solid(6, z); // weakly powered by torch(5) whenever it is lit
    b.put(7, z, Block::Torch { attach: Attach::West });
    b.dust_x(8, 13, z);
    b.output("Q", z);
    // Two inversions cancel: Q = A, at 1 rt per torch plus 1 rt for the
    // output repeater.
    let spec = b.spec(vec![0, 1], Constraints::default());
    Case {
        name: "torch/adjacent_block_interference".into(),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Pass(3),
    }
}

/// A repeater locked by a side repeater holds its output, which makes the
/// circuit a latch. Its cold-start truth table looks combinational; the
/// Gray-code pass catches that the answer depends on history.
fn repeater_locking_is_history_dependent() -> Case {
    let z = 8;
    let mut b = Builder::new();
    b.input("A", z);
    b.dust_x(2, 7, z);
    b.put(8, z, Block::Repeater { facing: Dir4::East, delay: 1 });
    b.dust_x(9, 13, z);
    b.output("Q", z);

    // B feeds a side repeater from behind, so it locks rather than drives.
    b.input("B", z - 3);
    b.dust_x(2, 8, z - 3);
    b.dust_z(8, z - 3, z - 2);
    b.put(8, z - 1, Block::Repeater { facing: Dir4::South, delay: 1 });

    let spec = b.spec(vec![0, 1, 0, 1], Constraints::default());
    Case {
        name: "repeater/locking_is_a_latch".into(),
        grid: b.grid.clone(),
        spec,
        expect: Expect::Malformed("history_dependent"),
    }
}

fn malformed_cases() -> Vec<Case> {
    let y = LOGIC_Y as i32;
    let mut out = Vec::new();

    // Dust with nothing under it.
    {
        let mut c = buffer(8, 1);
        c.grid.set(Pos::new(5, y + 2, 3), Block::Wire);
        c.name = "malformed/floating_dust".into();
        c.expect = Expect::Malformed("floating_dust");
        out.push(c);
    }
    // A lever that is not a declared input port.
    {
        let mut c = buffer(8, 1);
        c.grid.set(Pos::new(6, y, 3), Block::Solid);
        c.grid.set(Pos::new(5, y, 3), Block::Lever { attach: Dir4::East });
        c.name = "malformed/stray_lever".into();
        c.expect = Expect::Malformed("port_violation");
        out.push(c);
    }
    // A lamp that is not a declared output port.
    {
        let mut c = buffer(8, 1);
        c.grid.set(Pos::new(5, y, 3), Block::Lamp);
        c.name = "malformed/stray_lamp".into();
        c.expect = Expect::Malformed("port_violation");
        out.push(c);
    }
    // A declared input whose cell was overwritten.
    {
        let mut c = buffer(8, 1);
        c.grid.set(Pos::new(0, y, 8), Block::Air);
        c.name = "malformed/missing_input_port".into();
        c.expect = Expect::Malformed("port_violation");
        out.push(c);
    }
    // A declared output whose cell was overwritten.
    {
        let mut c = buffer(8, 1);
        c.grid.set(Pos::new(15, y, 8), Block::Air);
        c.name = "malformed/missing_output_port".into();
        c.expect = Expect::Malformed("port_violation");
        out.push(c);
    }
    // A wall torch hanging on air.
    {
        let mut c = buffer(8, 1);
        c.grid.set(Pos::new(5, y, 3), Block::Torch { attach: Attach::West });
        c.name = "malformed/unsupported_torch".into();
        c.expect = Expect::Malformed("unsupported");
        out.push(c);
    }
    // A repeater over a hole in the substrate.
    {
        let mut c = buffer(8, 1);
        c.grid.set(Pos::new(5, 0, 3), Block::Air);
        c.grid.set(Pos::new(5, y, 3), Block::Repeater { facing: Dir4::East, delay: 1 });
        c.name = "malformed/unsupported_repeater".into();
        c.expect = Expect::Malformed("unsupported");
        out.push(c);
    }
    // An observer: in the vocabulary, out of scope for v1.
    {
        let mut c = buffer(8, 1);
        c.grid.set(Pos::new(5, y, 3), Block::Observer { facing: Dir6::East });
        c.name = "malformed/observer_excluded".into();
        c.expect = Expect::Malformed("excluded_block");
        out.push(c);
    }
    // Dust directly on the substrate is fine; dust on a lamp is fine; dust on
    // a torch is not.
    {
        let mut c = buffer(8, 1);
        c.grid.set(Pos::new(5, y, 3), Block::Torch { attach: Attach::Floor });
        c.grid.set(Pos::new(5, y + 1, 3), Block::Wire);
        c.name = "malformed/dust_on_torch".into();
        c.expect = Expect::Malformed("floating_dust");
        out.push(c);
    }
    out
}

fn constraint_cases() -> Vec<Case> {
    let mut out = Vec::new();
    {
        let mut c = inverter(8); // costs 2 rt
        c.spec.constraints = Constraints { max_latency_rt: Some(1), ..Default::default() };
        c.name = "constraint/latency_too_tight".into();
        c.expect = Expect::FailConstraint;
        out.push(c);
    }
    {
        let mut c = inverter(8);
        c.spec.constraints = Constraints { max_latency_rt: Some(4), ..Default::default() };
        c.name = "constraint/latency_satisfied".into();
        c.expect = Expect::Pass(2);
        out.push(c);
    }
    {
        let mut c = inverter(8);
        c.spec.constraints = Constraints { max_blocks: Some(3), ..Default::default() };
        c.name = "constraint/footprint_too_tight".into();
        c.expect = Expect::FailConstraint;
        out.push(c);
    }
    {
        let mut c = inverter(8);
        c.spec.constraints = Constraints { max_blocks: Some(200), ..Default::default() };
        c.name = "constraint/footprint_satisfied".into();
        c.expect = Expect::Pass(2);
        out.push(c);
    }
    {
        let mut c = inverter(8);
        c.spec.constraints = Constraints { max_region: Some((4, 4)), ..Default::default() };
        c.name = "constraint/region_too_tight".into();
        c.expect = Expect::FailConstraint;
        out.push(c);
    }
    {
        let mut c = inverter(8);
        c.spec.constraints = Constraints { max_region: Some((16, 16)), ..Default::default() };
        c.name = "constraint/region_satisfied".into();
        c.expect = Expect::Pass(2);
        out.push(c);
    }
    out
}

/// Cases where the circuit is fine and the *spec* is wrong, which is the
/// common shape during self-training: a candidate that nearly works.
fn mislabelled_cases() -> Vec<Case> {
    let mut out = Vec::new();
    {
        let mut c = buffer(8, 1);
        c.spec.rows = vec![1, 0];
        c.name = "fail/buffer_labelled_inverter".into();
        c.expect = Expect::Fail(2);
        out.push(c);
    }
    {
        let mut c = and2(4, 10, 7);
        c.spec.rows = vec![0, 0, 0, 0];
        c.name = "fail/and_labelled_constant_low".into();
        c.expect = Expect::Fail(1);
        out.push(c);
    }
    {
        let mut c = nand2(4, 10, 7);
        c.spec.rows = vec![1, 1, 0, 0];
        c.name = "fail/nand_off_by_one_row".into();
        c.expect = Expect::Fail(1);
        out.push(c);
    }
    {
        let mut c = or2(4, 10, 7);
        c.spec.rows = vec![1, 0, 0, 0];
        c.name = "fail/or_labelled_nor".into();
        c.expect = Expect::Fail(4);
        out.push(c);
    }
    out
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();

    // --- translation invariance: the same circuit on every legal row -----
    //
    // A simulator with an off-by-one in its neighbour arithmetic passes on
    // one row and fails on another, so sweeping the rows is cheap insurance.
    for z in 1..=14 {
        let mut c = buffer(z, 1);
        c.name = format!("invariance/buffer/row{z:02}");
        v.push(c);
    }
    for z in 1..=14 {
        let mut c = inverter(z);
        c.name = format!("invariance/not/row{z:02}");
        v.push(c);
    }

    // --- timing ----------------------------------------------------------
    for d in 1..=4 {
        v.push(buffer(8, d));
    }
    for n in 1..=2 {
        for d in 1..=4 {
            v.push(repeater_chain(n, d));
        }
    }

    // --- dust range ------------------------------------------------------
    for k in 0..=8 {
        v.push(detour(k));
    }
    v.push(bare_lamp_line());
    v.push(bare_lamp_sideways());
    v.push(dust_ramp());

    // --- gates, each on four different row assignments -------------------
    for (za, zb, zo) in [(4, 10, 7), (2, 8, 5), (7, 13, 10), (3, 9, 6)] {
        v.push(and2(za, zb, zo));
        v.push(or2(za, zb, zo));
        v.push(nand2(za, zb, zo));
        v.push(nor2(za, zb, zo));
    }
    v.push(and3());
    v.push(or3());
    v.push(nand3());
    v.push(nor3());
    v.push(imply());
    v.push(dual_buffer());

    // --- power tiers and component semantics -----------------------------
    v.push(weak_power_gap());
    v.push(strong_power_tower());
    v.push(repeater_blocks_reverse());
    v.push(comparator_compare());
    v.push(comparator_subtract());
    v.push(comparator_subtract_side());
    v.push(target_is_inert());
    v.push(torch_adjacent_block_interference());
    v.push(repeater_locking_is_history_dependent());

    // --- instability -----------------------------------------------------
    for n in [1, 3] {
        v.push(torch_ring(n));
    }

    v.extend(malformed_cases());
    v.extend(constraint_cases());
    v.extend(mislabelled_cases());
    v
}

// ---------------------------------------------------------------------------
// the tests
// ---------------------------------------------------------------------------

#[test]
fn golden_suite_matches_hand_derived_verdicts() {
    let cases = cases();
    assert!(cases.len() >= 100, "golden suite has shrunk to {} cases", cases.len());

    let mut rendered = BTreeMap::new();
    let mut failures = Vec::new();
    for c in &cases {
        let v = evaluate(&c.grid, &c.spec);
        if let Err(why) = matches(&c.expect, &v) {
            failures.push(format!("  {}: {why}", c.name));
        }
        rendered.insert(c.name.clone(), format!("{v}"));
    }
    assert!(
        failures.is_empty(),
        "{} of {} golden cases disagree with their hand-derived verdict:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );

    // --- export for the fidelity harness ---------------------------------
    //
    // The golden circuits are built here, in Rust, and the fidelity harness is
    // Python. Three documents promise "100% agreement on the golden set, 104
    // hand-built circuits" and the harness could reach two of them, because
    // these had no way out of the test binary. Writing them where compare.py
    // can read them is what makes that promise checkable rather than
    // aspirational.
    //
    // Gated on an environment variable, like the blessing path above: a test
    // that writes files as a side effect of every run is a test that surprises
    // people.
    if let Some(path) = std::env::var_os("REDSIM_DUMP_GOLDEN") {
        let mut out = String::from("[\n");
        for (i, c) in cases.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            let tokens: Vec<String> =
                c.grid.cells().iter().map(|b| b.to_token().to_string()).collect();
            let input_z: Vec<String> =
                c.spec.inputs.iter().map(|p| p.pos.z.to_string()).collect();
            let output_z: Vec<String> =
                c.spec.outputs.iter().map(|p| p.pos.z.to_string()).collect();
            let rows: Vec<String> = c.spec.rows.iter().map(|r| r.to_string()).collect();
            // A case that is malformed by construction -- floating dust, a
            // port violation -- cannot be placed in a world at all, so it is
            // not a fidelity question and replaying it would score ten
            // guaranteed disagreements against a suite whose target is 100%.
            let malformed = matches!(c.expect, Expect::Malformed(_));
            let _ = write!(
                out,
                concat!(
                    r#"  {{"name":"{}","input_z":[{}],"output_z":[{}],"#,
                    r#""rows":[{}],"malformed":{},"tokens":[{}]}}"#
                ),
                c.name,
                input_z.join(","),
                output_z.join(","),
                rows.join(","),
                malformed,
                tokens.join(",")
            );
        }
        out.push_str("\n]\n");
        // Cargo runs an integration test from the crate directory, so a
        // relative path lands under crates/redsim and its parent may not
        // exist. Create it rather than panicking on a path the caller
        // reasonably expected to work.
        let path = std::path::PathBuf::from(&path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).unwrap();
            }
        }
        std::fs::write(&path, &out).unwrap();
        eprintln!("wrote {} golden cases to {}", cases.len(), path.to_string_lossy());
    }

    // --- snapshot --------------------------------------------------------
    let mut snap = String::new();
    snap.push_str("# redsim golden verdicts -- regenerate with REDSIM_BLESS=1\n");
    for (name, verdict) in &rendered {
        let _ = writeln!(snap, "{name}\t{verdict}");
    }
    if std::env::var_os("REDSIM_BLESS").is_some() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(SNAPSHOT_PATH);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &snap).unwrap();
        eprintln!("blessed {} cases into {}", rendered.len(), path.display());
        return;
    }
    // Git may materialise text fixtures with CRLF on Windows.  The snapshot
    // records verdict content, not a platform-specific line-ending choice.
    let expected = SNAPSHOT.replace("\r\n", "\n");
    if snap != expected {
        let want: Vec<&str> = expected.lines().collect();
        let got: Vec<&str> = snap.lines().collect();
        let mut diff = Vec::new();
        for i in 0..want.len().max(got.len()) {
            let a = want.get(i).copied().unwrap_or("<missing>");
            let b = got.get(i).copied().unwrap_or("<missing>");
            if a != b {
                diff.push(format!("  line {}:\n    was: {a}\n    now: {b}", i + 1));
            }
        }
        panic!(
            "golden snapshot drifted ({} line(s)); re-bless with REDSIM_BLESS=1 if intended:\n{}",
            diff.len(),
            diff.iter().take(20).cloned().collect::<Vec<_>>().join("\n")
        );
    }
}

/// The suite is only meaningful if the cases are actually distinct circuits.
#[test]
fn golden_cases_have_unique_names_and_distinct_grids() {
    let cases = cases();
    let mut names = std::collections::HashSet::new();
    for c in &cases {
        assert!(names.insert(c.name.clone()), "duplicate golden case name {}", c.name);
    }
    let distinct: std::collections::HashSet<Vec<u8>> =
        cases.iter().map(|c| c.grid.to_tokens()).collect();
    // Constraint cases deliberately reuse one circuit, so exact-distinctness
    // is not expected; a healthy suite still has most grids unique.
    assert!(
        distinct.len() * 10 >= cases.len() * 7,
        "only {} distinct grids across {} cases",
        distinct.len(),
        cases.len()
    );
}

/// Verdicts must not depend on how many times the circuit has been run, or on
/// which other circuit ran before it.
#[test]
fn evaluation_is_deterministic() {
    for c in cases() {
        let a = evaluate(&c.grid, &c.spec);
        let b = evaluate(&c.grid, &c.spec);
        assert_eq!(a, b, "{} is not deterministic", c.name);
        let via_tokens = evaluate_tokens(&c.grid.to_tokens(), &c.spec);
        assert_eq!(a, via_tokens, "{} disagrees between grid and token entry points", c.name);
    }
}

/// §03 budgets well under a millisecond per candidate at n <= 6, because §06
/// samples 64 candidates per spec across 20k specs per round. This asserts the
/// order of magnitude rather than a precise number, so it does not become a
/// flaky benchmark.
#[test]
fn evaluation_is_fast_enough_for_the_sampling_loop() {
    let cases: Vec<Case> = cases().into_iter().filter(|c| c.spec.n_inputs() >= 2).collect();
    let start = std::time::Instant::now();
    let reps = 20;
    for _ in 0..reps {
        for c in &cases {
            std::hint::black_box(evaluate(&c.grid, &c.spec));
        }
    }
    let n = cases.len() * reps;
    let per = start.elapsed().as_secs_f64() / n as f64;
    // Debug builds run this roughly 20x slower than release, so the ceiling
    // has to differ. Both numbers are loose enough not to be flaky and tight
    // enough to catch an accidental quadratic.
    let ceiling = if cfg!(debug_assertions) { 8e-3 } else { 5e-4 };
    eprintln!(
        "golden: {:.0} us per evaluation over {} evaluations ({} build)",
        per * 1e6,
        n,
        if cfg!(debug_assertions) { "debug" } else { "release" }
    );
    assert!(per < ceiling, "{:.3} ms per evaluation is too slow", per * 1e3);
}

/// A spec whose ports are not on the fixed faces must be rejected at
/// construction, not silently evaluated.
#[test]
fn specs_cannot_move_their_ports() {
    let mut b = Builder::new();
    b.input("A", 8);
    b.dust_x(2, 13, 8);
    b.output("Q", 8);
    let (ins, outs) = b.ports();
    let mut moved = ins.to_vec();
    moved[0].pos = Pos::new(4, LOGIC_Y as i32, 8);
    assert!(Spec::new(moved, outs.to_vec(), vec![0, 1], Constraints::default()).is_err());
}
