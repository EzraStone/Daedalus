//! The tick loop: latch, update, settle.
//!
//! Time base is `1 redstone tick = 2 game ticks = 0.1 s`. The simulator steps
//! in game ticks; latency is reported in redstone ticks.
//!
//! Update order is deliberately **not** Minecraft's. Real Java processes block
//! updates in a quasi-random, chunk-and-order-dependent sequence, and circuits
//! that depend on it are called "locational". Here every component latches its
//! input from the pre-update power field and they all apply simultaneously.
//! That is a documented divergence, and the fidelity harness exists to find
//! circuits where it matters — those are rejected rather than trusted.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

use crate::block::Block;
use crate::circuit::Circuit;
use crate::grid::Pos;
use crate::power::{self, Levels, States};

pub const GAME_TICKS_PER_RT: u32 = 2;
/// Cap on how long a single input combination may take to settle.
pub const DEFAULT_MAX_GAME_TICKS: u32 = 200;
/// Minecraft burns out a torch that toggles too fast; we reject the design
/// instead of modelling the burnout. A circuit that burns out is a bad design.
pub const BURNOUT_WINDOW_GT: u32 = 60;
pub const BURNOUT_TOGGLE_LIMIT: usize = 8;

/// How a run of the tick loop ended.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Settle {
    /// Reached a fixed point after `game_ticks`.
    Settled { game_ticks: u32 },
    /// Entered a cycle of period `period_gt` game ticks. Distinct from a
    /// wrong answer: an oscillator is a *different* failure and gets its own
    /// metric.
    Oscillating { period_gt: u32 },
    /// Still changing at the cap. Almost always a slow oscillator whose period
    /// exceeds the state-history window.
    Timeout { game_ticks: u32 },
    /// A torch toggled more than [`BURNOUT_TOGGLE_LIMIT`] times inside
    /// [`BURNOUT_WINDOW_GT`].
    Burnout { at: Pos },
}

impl Settle {
    pub fn is_settled(&self) -> bool {
        matches!(self, Settle::Settled { .. })
    }

    /// Settle time in redstone ticks, rounded up.
    pub fn latency_rt(&self) -> u32 {
        match self {
            Settle::Settled { game_ticks } => game_ticks.div_ceil(GAME_TICKS_PER_RT),
            _ => u32::MAX,
        }
    }
}

/// A circuit plus its mutable simulation state.
pub struct Sim {
    pub circuit: Circuit,
    pub states: States,
    levels: Levels,
    /// Game tick at which each stateful component's scheduled update fires,
    /// or `-1` for "nothing scheduled".
    pending: Vec<i64>,
    /// Toggle timestamps per torch, parallel to `circuit.torches`.
    torch_toggles: Vec<VecDeque<u32>>,
    /// Cell index -> slot in `torch_toggles`, or `u32::MAX`.
    torch_slot: Vec<u32>,
    history: Vec<(u64, u32)>,
    /// Reused across ticks so the settle loop allocates nothing.
    fired: Vec<(usize, u8)>,
}

impl Sim {
    pub fn new(circuit: Circuit) -> Sim {
        let states = States::cold(&circuit);
        let n_torch = circuit.torches.len();
        let mut torch_slot = vec![u32::MAX; crate::grid::CELLS];
        for (slot, &i) in circuit.torches.iter().enumerate() {
            torch_slot[i] = slot as u32;
        }
        Sim {
            levels: Levels::new(),
            pending: vec![-1; crate::grid::CELLS],
            torch_toggles: vec![VecDeque::new(); n_torch],
            torch_slot,
            history: Vec::with_capacity(32),
            fired: Vec::with_capacity(16),
            circuit,
            states,
        }
    }

    /// Reset every latched component, optionally keeping lever positions.
    /// Used between truth-table rows so each row is evaluated from a known
    /// start rather than from whatever the previous row left behind.
    pub fn reset(&mut self, keep_levers: bool) {
        self.states.reset(&self.circuit, keep_levers);
        for &i in &self.circuit.stateful {
            self.pending[i] = -1;
        }
        for q in &mut self.torch_toggles {
            q.clear();
        }
        self.history.clear();
    }

    pub fn set_lever(&mut self, p: Pos, on: bool) {
        if p.in_bounds() {
            self.states.lever_on[p.index()] = on;
        }
    }

    pub fn levels(&self) -> &Levels {
        &self.levels
    }

    /// Is the lamp at `p` lit in the current power field?
    pub fn lamp_lit(&self, p: Pos) -> bool {
        self.levels.block_powered(p)
    }

    /// The value a stateful component wants to hold, given the current field.
    fn desired(&self, i: usize) -> u8 {
        let p = Pos::from_index(i);
        let grid = &self.circuit.grid;
        match grid.at(i) {
            Block::Torch { attach } => {
                let support = p.attached(attach);
                // An unsupported torch cannot be switched off by anything, so
                // it stays lit. The malformed check rejects these up front;
                // this branch just keeps the loop total.
                u8::from(!self.levels.block_powered(support))
            }
            Block::Repeater { facing, .. } => {
                if power::repeater_locked(grid, &self.states, &self.levels, p) {
                    u8::from(self.states.rep_out[i])
                } else {
                    let rear =
                        power::input_signal(grid, &self.states, &self.levels, p, facing.opposite());
                    u8::from(rear > 0)
                }
            }
            Block::Comparator { .. } => {
                power::comparator_output(grid, &self.states, &self.levels, p)
            }
            _ => 0,
        }
    }

    fn current(&self, i: usize) -> u8 {
        match self.circuit.grid.at(i) {
            Block::Torch { .. } => u8::from(self.states.torch_lit[i]),
            Block::Repeater { .. } => u8::from(self.states.rep_out[i]),
            Block::Comparator { .. } => self.states.cmp_out[i],
            _ => 0,
        }
    }

    fn delay_gt(&self, i: usize) -> u32 {
        match self.circuit.grid.at(i) {
            Block::Repeater { delay, .. } => delay.clamp(1, 4) as u32 * GAME_TICKS_PER_RT,
            _ => GAME_TICKS_PER_RT,
        }
    }

    fn apply(&mut self, i: usize, v: u8) {
        match self.circuit.grid.at(i) {
            Block::Torch { .. } => self.states.torch_lit[i] = v != 0,
            Block::Repeater { .. } => self.states.rep_out[i] = v != 0,
            Block::Comparator { .. } => self.states.cmp_out[i] = v,
            _ => {}
        }
    }

    /// Hash of everything that defines the future of the simulation: latched
    /// outputs plus outstanding schedules. Two ticks with the same hash are
    /// the same point in the state machine.
    fn state_hash(&self, t: u32) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for &i in &self.circuit.stateful {
            self.current(i).hash(&mut h);
            // Schedules must be hashed as *remaining* delay. Hashing the
            // absolute fire time would make every tick of a pending wait look
            // like a fresh state, and — worse — make two consecutive waiting
            // ticks compare equal, which reads as a period-1 oscillation.
            let remaining = if self.pending[i] < 0 { -1 } else { self.pending[i] - t as i64 };
            remaining.hash(&mut h);
        }
        h.finish()
    }

    fn note_torch_toggle(&mut self, torch_slot: usize, t: u32) -> bool {
        let q = &mut self.torch_toggles[torch_slot];
        q.push_back(t);
        while let Some(&front) = q.front() {
            if t.saturating_sub(front) > BURNOUT_WINDOW_GT {
                q.pop_front();
            } else {
                break;
            }
        }
        q.len() > BURNOUT_TOGGLE_LIMIT
    }

    /// Run until the circuit reaches a fixed point, starts repeating, or hits
    /// the cap.
    pub fn settle(&mut self, max_game_ticks: u32) -> Settle {
        self.history.clear();
        let n_stateful = self.circuit.stateful.len();

        let mut t: u32 = 0;
        loop {
            power::solve_into(&self.circuit, &self.states, &mut self.levels);

            // --- fire everything due at this tick, all reading the same
            // pre-update field ---
            let mut fired = std::mem::take(&mut self.fired);
            fired.clear();
            for k in 0..n_stateful {
                let i = self.circuit.stateful[k];
                if self.pending[i] == t as i64 {
                    fired.push((i, self.desired(i)));
                }
            }
            let mut changed = false;
            for &(i, v) in fired.iter() {
                self.pending[i] = -1;
                if self.current(i) != v {
                    let was_torch = matches!(self.circuit.grid.at(i), Block::Torch { .. });
                    self.apply(i, v);
                    changed = true;
                    if was_torch {
                        let slot = self.torch_slot[i];
                        if slot != u32::MAX && self.note_torch_toggle(slot as usize, t) {
                            self.fired = fired;
                            return Settle::Burnout { at: Pos::from_index(i) };
                        }
                    }
                }
            }
            self.fired = fired;
            if changed {
                power::solve_into(&self.circuit, &self.states, &mut self.levels);
            }

            // --- schedule anything that now disagrees with the field ---
            let mut any_pending = false;
            for k in 0..n_stateful {
                let i = self.circuit.stateful[k];
                if self.pending[i] >= 0 {
                    any_pending = true;
                    continue;
                }
                let want = self.desired(i);
                if want != self.current(i) {
                    self.pending[i] = (t + self.delay_gt(i)) as i64;
                    any_pending = true;
                }
            }

            if !any_pending {
                return Settle::Settled { game_ticks: t };
            }

            // --- cycle detection ---
            let h = self.state_hash(t);
            if let Some(&(_, prev)) = self.history.iter().find(|(hh, _)| *hh == h) {
                return Settle::Oscillating { period_gt: t - prev };
            }
            self.history.push((h, t));

            t += 1;
            if t > max_game_ticks {
                return Settle::Timeout { game_ticks: t };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{Attach, Dir4};
    use crate::grid::{Grid, LOGIC_Y};

    /// `lever -> block -> torch`: the canonical NOT gate.
    fn not_gate() -> (Sim, Pos, Pos) {
        let y = LOGIC_Y as i32;
        let z = 8;
        let mut g = Grid::with_substrate();
        let lever = Pos::new(0, y, z);
        g.set(lever, Block::Lever { attach: Dir4::East });
        g.set(Pos::new(1, y, z), Block::Solid);
        let torch = Pos::new(2, y, z);
        g.set(torch, Block::Torch { attach: Attach::West });
        g.set(Pos::new(3, y, z), Block::Wire);
        let lamp = Pos::new(4, y, z);
        g.set(lamp, Block::Lamp);
        (Sim::new(Circuit::new(g)), lever, lamp)
    }

    #[test]
    fn torch_inverts_with_one_redstone_tick_of_delay() {
        let (mut sim, lever, lamp) = not_gate();

        sim.set_lever(lever, false);
        let s = sim.settle(DEFAULT_MAX_GAME_TICKS);
        assert!(s.is_settled(), "{s:?}");
        assert!(sim.lamp_lit(lamp), "lever off -> torch lit -> lamp on");

        sim.set_lever(lever, true);
        let s = sim.settle(DEFAULT_MAX_GAME_TICKS);
        assert!(s.is_settled(), "{s:?}");
        assert!(!sim.lamp_lit(lamp), "lever on -> torch off -> lamp off");
        assert_eq!(s.latency_rt(), 1, "a torch costs exactly one redstone tick");
    }

    #[test]
    fn torch_feeding_its_own_support_oscillates() {
        // A torch whose output loops back onto its own support block is the
        // classic one-torch clock. It must report Oscillating, never Settled.
        //
        // Note the return path approaches the support block head-on from the
        // west. A dust line that merely runs *past* the block would not power
        // it: dust only weakly powers the blocks it points at, and its
        // pointing is decided by its connections.
        let y = LOGIC_Y as i32;
        let mut g = Grid::with_substrate();
        g.set(Pos::new(5, y, 5), Block::Solid);
        g.set(Pos::new(6, y, 5), Block::Torch { attach: Attach::West });
        // The loop is routed one row clear of the support block so that the
        // only dust adjacent to it is the final head-on cell.
        for p in [
            Pos::new(6, y, 6),
            Pos::new(6, y, 7),
            Pos::new(5, y, 7),
            Pos::new(4, y, 7),
            Pos::new(3, y, 7),
            Pos::new(3, y, 6),
            Pos::new(3, y, 5),
            Pos::new(4, y, 5),
        ] {
            g.set(p, Block::Wire);
        }
        let mut sim = Sim::new(Circuit::new(g));
        let s = sim.settle(DEFAULT_MAX_GAME_TICKS);
        assert!(
            matches!(s, Settle::Oscillating { .. } | Settle::Burnout { .. }),
            "expected an unstable verdict, got {s:?}"
        );
    }

    #[test]
    fn repeater_delay_scales_with_its_setting() {
        for delay in 1..=4u8 {
            let y = LOGIC_Y as i32;
            let z = 8;
            let mut g = Grid::with_substrate();
            let lever = Pos::new(0, y, z);
            g.set(lever, Block::Lever { attach: Dir4::East });
            g.set(Pos::new(1, y, z), Block::Solid);
            g.set(Pos::new(2, y, z), Block::Wire);
            g.set(Pos::new(3, y, z), Block::Repeater { facing: Dir4::East, delay });
            g.set(Pos::new(4, y, z), Block::Wire);
            let lamp = Pos::new(5, y, z);
            g.set(lamp, Block::Lamp);

            let mut sim = Sim::new(Circuit::new(g));
            sim.set_lever(lever, false);
            assert!(sim.settle(DEFAULT_MAX_GAME_TICKS).is_settled());
            assert!(!sim.lamp_lit(lamp));

            sim.set_lever(lever, true);
            let s = sim.settle(DEFAULT_MAX_GAME_TICKS);
            assert!(s.is_settled(), "delay {delay}: {s:?}");
            assert!(sim.lamp_lit(lamp), "delay {delay}: repeater should pass the signal");
            assert_eq!(s.latency_rt(), delay as u32, "delay {delay} should cost {delay} rt");
        }
    }

    /// The power field the `power` protocol op exposes.
    ///
    /// Levels are computed on the way to every verdict and were discarded;
    /// once something reads them, the decay along a run becomes the diagnostic
    /// people actually use, so it has to be exactly one step per block.
    #[test]
    fn dust_loses_one_step_of_strength_per_block() {
        let y = 1;
        let z = 5;
        let mut g = Grid::with_substrate();
        let lever = Pos::new(0, y, z);
        g.set(lever, Block::Lever { attach: Dir4::East });
        g.set(Pos::new(1, y, z), Block::Solid);
        for x in 2..14 {
            g.set(Pos::new(x, y, z), Block::Wire);
        }

        let mut sim = Sim::new(Circuit::new(g));
        sim.set_lever(lever, true);
        assert!(sim.settle(DEFAULT_MAX_GAME_TICKS).is_settled());

        let dust = &sim.levels().dust;
        let first = dust[Pos::new(2, y, z).index()];
        assert_eq!(first, 15, "dust beside a powered block starts full");
        for step in 0..12 {
            let x = 2 + step as i32;
            let want = 15u8.saturating_sub(step);
            assert_eq!(
                dust[Pos::new(x, y, z).index()],
                want,
                "strength at x={x} should be {want}"
            );
        }
    }

    /// A run longer than fifteen blocks goes dark, which is the whole reason
    /// the router inserts repeaters -- and the reason reading the field is
    /// worth doing, since the zero is visible where the failure is not.
    #[test]
    fn dust_runs_out_after_fifteen_blocks() {
        let y = 1;
        let z = 5;
        let mut g = Grid::with_substrate();
        let lever = Pos::new(0, y, z);
        g.set(lever, Block::Lever { attach: Dir4::East });
        g.set(Pos::new(1, y, z), Block::Solid);
        for x in 2..16 {
            g.set(Pos::new(x, y, z), Block::Wire);
        }

        let mut sim = Sim::new(Circuit::new(g));
        sim.set_lever(lever, true);
        assert!(sim.settle(DEFAULT_MAX_GAME_TICKS).is_settled());

        let dust = &sim.levels().dust;
        assert!(dust[Pos::new(15, y, z).index()] < 3, "the far end should be nearly dark");
    }

    /// Nothing that is not dust carries a level, or the field would be
    /// describing something other than signal strength.
    #[test]
    fn only_dust_cells_carry_a_level() {
        let y = 1;
        let mut g = Grid::with_substrate();
        let lever = Pos::new(0, y, 3);
        g.set(lever, Block::Lever { attach: Dir4::East });
        g.set(Pos::new(1, y, 3), Block::Solid);
        g.set(Pos::new(2, y, 3), Block::Wire);
        g.set(Pos::new(3, y, 3), Block::Lamp);

        let mut sim = Sim::new(Circuit::new(g.clone()));
        sim.set_lever(lever, true);
        sim.settle(DEFAULT_MAX_GAME_TICKS);

        for (i, level) in sim.levels().dust.iter().enumerate() {
            if *level > 0 {
                assert_eq!(
                    g.get(Pos::from_index(i)),
                    Block::Wire,
                    "cell {i} carries a level but is not dust"
                );
            }
        }
    }
}
