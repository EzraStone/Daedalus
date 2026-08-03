//! Power propagation: the weak/strong tier system and the dust solver.
//!
//! Within a single game tick, propagation is *instantaneous* and acyclic:
//!
//! 1. Latched component outputs (levers, torches, repeaters, comparators) are
//!    given — they only change at tick boundaries, in [`crate::tick`].
//! 2. Those outputs determine **strong** power on conductive blocks.
//! 3. Dust strength is a multi-source BFS seeded by component outputs and by
//!    strongly powered blocks, decaying by one per step.
//! 4. Dust determines **weak** power on conductive blocks.
//!
//! Step 4 never feeds back into step 3, because *dust reads only strong power
//! from blocks*. That asymmetry is the whole point of the two-tier system and
//! it is what makes the fixed point solvable in one pass instead of by
//! iteration. Getting it wrong is the single largest source of
//! simulator/game divergence, so it is spelled out here rather than buried.

use crate::block::{Block, CmpMode, Dir4, DIR4};
use crate::circuit::Circuit;
use crate::grid::{Grid, Pos, CELLS};

pub const MAX_STRENGTH: u8 = 15;

/// Latched outputs of every stateful component, indexed by flat cell index.
///
/// Kept as parallel dense arrays rather than a map: the grid is only 1536
/// cells, and dense arrays make the per-tick state hash trivially stable.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct States {
    /// Lever positions. Set by the harness, never by the simulator.
    pub lever_on: Vec<bool>,
    /// Whether each torch is currently lit.
    pub torch_lit: Vec<bool>,
    /// Whether each repeater is currently driving its output.
    pub rep_out: Vec<bool>,
    /// Current output strength of each comparator.
    pub cmp_out: Vec<u8>,
}

impl States {
    /// Initial state: torches lit, repeaters off, comparators at zero.
    ///
    /// Torches-lit is the correct cold-start: an unpowered torch is lit, and
    /// starting them all off would make every NOT gate settle one tick late.
    pub fn cold(c: &Circuit) -> States {
        let mut s = States {
            lever_on: vec![false; CELLS],
            torch_lit: vec![false; CELLS],
            rep_out: vec![false; CELLS],
            cmp_out: vec![0; CELLS],
        };
        for &i in &c.torches {
            s.torch_lit[i] = true;
        }
        s
    }

    /// Reset to the cold state without reallocating, preserving lever
    /// positions when asked. The tick loop re-runs a circuit once per
    /// truth-table row, so this is on the hot path.
    pub fn reset(&mut self, c: &Circuit, keep_levers: bool) {
        if !keep_levers {
            self.lever_on.iter_mut().for_each(|v| *v = false);
        }
        self.torch_lit.iter_mut().for_each(|v| *v = false);
        for &i in &c.torches {
            self.torch_lit[i] = true;
        }
        for &i in &c.repeaters {
            self.rep_out[i] = false;
        }
        for &i in &c.comparators {
            self.cmp_out[i] = 0;
        }
    }
}

/// The solved power field for one game tick.
pub struct Levels {
    /// Signal strength of each dust cell, `0..=15`. Zero everywhere else.
    pub dust: Vec<u8>,
    /// Strong power on each conductive block. Only strong power re-emits into
    /// adjacent dust.
    pub strong: Vec<u8>,
    /// Weak power on each conductive block, from dust pointing into it.
    pub weak: Vec<u8>,
    /// Per-dust-cell connection bitmask over [`DIR4`], by `Dir4::index()`.
    pub links: Vec<u8>,
    /// Bucket queue for the dust BFS, one bucket per strength. Carried on the
    /// struct purely so a settle loop does not allocate 16 vectors per game
    /// tick — at roughly 600 solves per candidate that was the single largest
    /// cost in the verifier.
    buckets: Vec<Vec<usize>>,
    scratch: Vec<Pos>,
    /// Conductive cells written during the last solve. Only a handful of
    /// blocks in a circuit are ever powered, so clearing this list beats
    /// re-zeroing all 256 substrate cells every tick.
    dirty: Vec<usize>,
}

/// Raise `arr[i]` to `v`, remembering the cell so it can be cleared cheaply.
#[inline]
fn raise(arr: &mut [u8], dirty: &mut Vec<usize>, i: usize, v: u8) {
    if v == 0 {
        return;
    }
    if arr[i] == 0 {
        dirty.push(i);
    }
    if v > arr[i] {
        arr[i] = v;
    }
}

impl Levels {
    pub fn new() -> Levels {
        Levels {
            dust: vec![0; CELLS],
            strong: vec![0; CELLS],
            weak: vec![0; CELLS],
            links: vec![0; CELLS],
            buckets: vec![Vec::new(); (MAX_STRENGTH + 1) as usize],
            scratch: Vec::with_capacity(8),
            dirty: Vec::with_capacity(64),
        }
    }

    /// Zero only what the last solve actually wrote.
    fn clear(&mut self, c: &Circuit) {
        for &i in &c.wires {
            self.dust[i] = 0;
            self.links[i] = 0;
        }
        for &i in &self.dirty {
            self.strong[i] = 0;
            self.weak[i] = 0;
        }
        self.dirty.clear();
    }

    /// Is this cell powered at all, by either tier?
    #[inline]
    pub fn block_powered(&self, p: Pos) -> bool {
        if !p.in_bounds() {
            return false;
        }
        let i = p.index();
        self.strong[i] > 0 || self.weak[i] > 0
    }

    #[inline]
    pub fn dust_at(&self, p: Pos) -> u8 {
        if p.in_bounds() {
            self.dust[p.index()]
        } else {
            0
        }
    }
}

#[inline]
fn bit(d: Dir4) -> u8 {
    1 << d.index()
}

/// The dust cells reachable from `p` by wire-to-wire connection, including the
/// one-block slopes.
///
/// Up-slope: dust climbs onto an adjacent opaque block only if the cell
/// directly above `p` is transparent — otherwise the run is roofed and cut.
/// Down-slope: dust drops to an adjacent lower dust only if the cell between
/// them is transparent.
pub fn wire_neighbours(grid: &Grid, p: Pos, out: &mut Vec<Pos>) {
    out.clear();
    let roofed = grid.get(p.up()).is_opaque();
    for d in DIR4 {
        let n = p.step(d);
        if grid.get(n) == Block::Wire {
            out.push(n);
            continue;
        }
        if !roofed && grid.get(n).is_opaque() && grid.get(n.up()) == Block::Wire {
            out.push(n.up());
        }
        if !grid.get(n).is_opaque() && grid.get(n.down()) == Block::Wire {
            out.push(n.down());
        }
    }
}

/// Connection bitmask for a dust cell: the directions it visually links to.
///
/// A direction counts if it reaches another dust cell (flat or sloped) or a
/// component that accepts a redstone connection from that side.
pub fn dust_links(grid: &Grid, p: Pos) -> u8 {
    let roofed = grid.get(p.up()).is_opaque();
    let mut mask = 0u8;
    for d in DIR4 {
        let n = p.step(d);
        let nb = grid.get(n);
        let flat = nb == Block::Wire || nb.connects_dust(d);
        let up = !roofed && nb.is_opaque() && grid.get(n.up()) == Block::Wire;
        let down = !nb.is_opaque() && grid.get(n.down()) == Block::Wire;
        if flat || up || down {
            mask |= bit(d);
        }
    }
    mask
}

/// The directions a dust cell *points*, which is what decides which blocks it
/// weakly powers.
///
/// Minecraft renders a dust with exactly one connection as a straight line
/// through the cell, so it points at the opposite side too. A dust with no
/// connections is a dot and points nowhere horizontally.
pub fn dust_points(links: u8) -> u8 {
    match links.count_ones() {
        0 => 0,
        1 => {
            let d = DIR4[links.trailing_zeros() as usize];
            bit(d) | bit(d.opposite())
        }
        _ => links,
    }
}

/// Signal a non-dust component delivers to an adjacent cell lying in direction
/// `toward` from the component. Returns 0 when the component does not reach
/// that way.
fn component_signal(block: Block, at: Pos, toward: Dir4, vertical: i32, st: &States) -> u8 {
    let i = at.index();
    match block {
        Block::Torch { attach } => {
            if !st.torch_lit[i] {
                return 0;
            }
            // A torch reaches every face except the one it hangs on.
            let (ax, ay, az) = attach.delta();
            let (dx, dz) = toward.delta();
            if (ax, ay, az) == (dx, vertical, dz) {
                0
            } else {
                MAX_STRENGTH
            }
        }
        Block::Lever { attach } => {
            if !st.lever_on[i] {
                return 0;
            }
            let (ax, az) = attach.delta();
            let (dx, dz) = toward.delta();
            if (ax, 0, az) == (dx, vertical, dz) {
                0
            } else {
                MAX_STRENGTH
            }
        }
        Block::Repeater { facing, .. } => {
            if st.rep_out[i] && vertical == 0 && facing == toward {
                MAX_STRENGTH
            } else {
                0
            }
        }
        Block::Comparator { facing, .. } => {
            if vertical == 0 && facing == toward {
                st.cmp_out[i]
            } else {
                0
            }
        }
        _ => 0,
    }
}

impl Default for Levels {
    fn default() -> Self {
        Levels::new()
    }
}

/// Strong power a component pushes into the block it faces.
fn apply_strong(c: &Circuit, st: &States, strong: &mut [u8], dirty: &mut Vec<usize>) {
    let grid = &c.grid;
    for &i in &c.emitters {
        let p = Pos::from_index(i);
        match grid.at(i) {
            Block::Lever { attach } => {
                if st.lever_on[i] {
                    let t = p.step(attach);
                    if t.in_bounds() && grid.get(t).is_conductive() {
                        raise(strong, dirty, t.index(), MAX_STRENGTH);
                    }
                }
            }
            Block::Torch { .. } => {
                if st.torch_lit[i] {
                    // Both floor and wall torches strongly power the block
                    // directly above them; that is the torch-tower primitive.
                    let t = p.up();
                    if t.in_bounds() && grid.get(t).is_conductive() {
                        raise(strong, dirty, t.index(), MAX_STRENGTH);
                    }
                }
            }
            Block::Repeater { facing, .. } => {
                if st.rep_out[i] {
                    let t = p.step(facing);
                    if t.in_bounds() && grid.get(t).is_conductive() {
                        raise(strong, dirty, t.index(), MAX_STRENGTH);
                    }
                }
            }
            Block::Comparator { facing, .. } => {
                let v = st.cmp_out[i];
                if v > 0 {
                    let t = p.step(facing);
                    if t.in_bounds() && grid.get(t).is_conductive() {
                        raise(strong, dirty, t.index(), v);
                    }
                }
            }
            _ => {}
        }
    }
}

/// The best signal reaching a dust cell from its six neighbours, ignoring
/// other dust (that part is the BFS).
fn dust_seed(grid: &Grid, st: &States, strong: &[u8], p: Pos) -> u8 {
    let mut best = 0u8;
    // Four horizontal neighbours.
    for d in DIR4 {
        let n = p.step(d);
        if !n.in_bounds() {
            continue;
        }
        let nb = grid.get(n);
        if nb.is_conductive() {
            best = best.max(strong[n.index()]);
        } else {
            best = best.max(component_signal(nb, n, d.opposite(), 0, st));
        }
    }
    // Above and below. A component directly under a dust cell reaches up; a
    // strongly powered block under it does too, which is how a torch tower
    // hands power to the dust on its cap.
    for (n, vert) in [(p.up(), -1i32), (p.down(), 1i32)] {
        if !n.in_bounds() {
            continue;
        }
        let nb = grid.get(n);
        if nb.is_conductive() {
            best = best.max(strong[n.index()]);
        } else {
            for d in DIR4 {
                best = best.max(component_signal(nb, n, d, vert, st));
            }
        }
    }
    best
}

/// Solve the whole power field for one game tick.
pub fn solve(c: &Circuit, st: &States) -> Levels {
    let mut lv = Levels::new();
    solve_into(c, st, &mut lv);
    lv
}

/// Solve into a reused buffer. Same result as [`solve`], no allocation.
pub fn solve_into(c: &Circuit, st: &States, lv: &mut Levels) {
    lv.clear(c);
    let grid = &c.grid;
    let Levels { dust, strong, weak, links, buckets, scratch, dirty } = lv;
    for b in buckets.iter_mut() {
        b.clear();
    }

    apply_strong(c, st, strong, dirty);

    // --- multi-source BFS over the dust network -------------------------
    //
    // Strength is bounded at 15, so a bucket queue indexed by strength is an
    // exact shortest-path solve in a single descending sweep. No iteration to
    // a fixed point, no ordering ambiguity.
    for &i in &c.wires {
        let p = Pos::from_index(i);
        links[i] = dust_links(grid, p);
        let seed = dust_seed(grid, st, strong, p);
        if seed > 0 {
            dust[i] = seed;
            buckets[seed as usize].push(i);
        }
    }

    for s in (1..=MAX_STRENGTH).rev() {
        // `buckets[s]` can only grow from higher strengths, never from lower,
        // so a single pass per level is complete.
        let mut k = 0;
        while k < buckets[s as usize].len() {
            let i = buckets[s as usize][k];
            k += 1;
            if dust[i] != s {
                continue; // superseded by a stronger seed
            }
            wire_neighbours(grid, Pos::from_index(i), scratch);
            for &n in scratch.iter() {
                let ni = n.index();
                if dust[ni] < s - 1 {
                    dust[ni] = s - 1;
                    if s - 1 > 0 {
                        buckets[(s - 1) as usize].push(ni);
                    }
                }
            }
        }
    }

    // --- weak power from dust -------------------------------------------
    for &i in &c.wires {
        let s = dust[i];
        if s == 0 {
            continue;
        }
        let p = Pos::from_index(i);
        // Dust always powers the block it stands on.
        let below = p.down();
        if below.in_bounds() && grid.get(below).is_conductive() {
            raise(weak, dirty, below.index(), s);
        }
        let points = dust_points(links[i]);
        for d in DIR4 {
            if points & bit(d) == 0 {
                continue;
            }
            let t = p.step(d);
            if t.in_bounds() && grid.get(t).is_conductive() {
                raise(weak, dirty, t.index(), s);
            }
        }
    }

    // --- weak power from components --------------------------------------
    // A lit torch or a thrown lever weakly powers every adjacent block, not
    // just the one it is mounted on. This is why two torches on neighbouring
    // blocks interfere, and why circuits need spacing.
    for &i in &c.emitters {
        let p = Pos::from_index(i);
        let b = grid.at(i);
        for d in DIR4 {
            let t = p.step(d);
            if t.in_bounds() && grid.get(t).is_conductive() {
                let v = component_signal(b, p, d, 0, st);
                raise(weak, dirty, t.index(), v);
            }
        }
        for (t, vert) in [(p.up(), 1i32), (p.down(), -1i32)] {
            if t.in_bounds() && grid.get(t).is_conductive() {
                let mut v = 0;
                for d in DIR4 {
                    v = v.max(component_signal(b, p, d, vert, st));
                }
                // The vertical faces are reached by torches and levers only.
                if matches!(b, Block::Torch { .. } | Block::Lever { .. }) {
                    raise(weak, dirty, t.index(), v);
                }
            }
        }
    }
}

/// Signal presented to a component's input face at `from` looking in direction
/// `dir` (the direction from the component toward the source).
///
/// Blocks hand over **strong power only** here. A block that is merely weakly
/// powered — by dust running across its top, say — does not drive a repeater
/// in front of it, and that asymmetry is load-bearing for most compact gates.
pub fn input_signal(grid: &Grid, st: &States, lv: &Levels, from: Pos, dir: Dir4) -> u8 {
    let n = from.step(dir);
    if !n.in_bounds() {
        return 0;
    }
    let nb = grid.get(n);
    match nb {
        Block::Wire => lv.dust[n.index()],
        _ if nb.is_conductive() => lv.strong[n.index()],
        _ => component_signal(nb, n, dir.opposite(), 0, st),
    }
}

/// Signal presented to a comparator's *side* face.
///
/// Comparator sides deliberately ignore conductive blocks: only dust and
/// signal-emitting components register. This is real Minecraft behaviour and
/// is what makes side-by-side comparator chains work.
pub fn side_signal(grid: &Grid, st: &States, lv: &Levels, from: Pos, dir: Dir4) -> u8 {
    let n = from.step(dir);
    if !n.in_bounds() {
        return 0;
    }
    let nb = grid.get(n);
    match nb {
        Block::Wire => lv.dust[n.index()],
        Block::Repeater { .. } | Block::Comparator { .. } => {
            component_signal(nb, n, dir.opposite(), 0, st)
        }
        _ => 0,
    }
}

/// Comparator output for the current power field.
pub fn comparator_output(grid: &Grid, st: &States, lv: &Levels, p: Pos) -> u8 {
    let (facing, mode) = match grid.get(p) {
        Block::Comparator { facing, mode } => (facing, mode),
        _ => return 0,
    };
    let rear = input_signal(grid, st, lv, p, facing.opposite());
    let (sa, sb) = match facing {
        Dir4::North | Dir4::South => (Dir4::West, Dir4::East),
        Dir4::West | Dir4::East => (Dir4::North, Dir4::South),
    };
    let side = side_signal(grid, st, lv, p, sa).max(side_signal(grid, st, lv, p, sb));
    match mode {
        CmpMode::Compare => {
            if rear >= side {
                rear
            } else {
                0
            }
        }
        CmpMode::Subtract => rear.saturating_sub(side),
    }
}

/// Is a repeater currently locked by a side-facing repeater or comparator?
///
/// A locked repeater holds its output regardless of its input. Locking is
/// cheap to implement and common enough in real builds that leaving it out
/// would make the simulator wrong on circuits people actually write.
pub fn repeater_locked(grid: &Grid, st: &States, lv: &Levels, p: Pos) -> bool {
    let facing = match grid.get(p) {
        Block::Repeater { facing, .. } => facing,
        _ => return false,
    };
    let (sa, sb) = match facing {
        Dir4::North | Dir4::South => (Dir4::West, Dir4::East),
        Dir4::West | Dir4::East => (Dir4::North, Dir4::South),
    };
    for d in [sa, sb] {
        let n = p.step(d);
        if !n.in_bounds() {
            continue;
        }
        match grid.get(n) {
            Block::Repeater { facing: f, .. } if f == d.opposite() => {
                if st.rep_out[n.index()] {
                    return true;
                }
            }
            Block::Comparator { facing: f, .. } if f == d.opposite() => {
                if st.cmp_out[n.index()] > 0 {
                    return true;
                }
            }
            _ => {}
        }
    }
    let _ = lv;
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Attach;
    use crate::grid::LOGIC_Y;

    fn solved(g: Grid) -> Levels {
        let c = Circuit::new(g);
        let st = States::cold(&c);
        solve(&c, &st)
    }

    #[test]
    fn dust_decays_one_per_block() {
        let mut g = Grid::with_substrate();
        for x in 0..16 {
            g.set(Pos::new(x, LOGIC_Y as i32, 8), Block::Wire);
        }
        // Drive the run from a lit floor torch at the head of the row.
        g.set(Pos::new(0, LOGIC_Y as i32, 8), Block::Torch { attach: Attach::Floor });
        let lv = solved(g);
        assert_eq!(lv.dust_at(Pos::new(1, LOGIC_Y as i32, 8)), 15);
        assert_eq!(lv.dust_at(Pos::new(2, LOGIC_Y as i32, 8)), 14);
        assert_eq!(lv.dust_at(Pos::new(15, LOGIC_Y as i32, 8)), 1);
    }

    #[test]
    fn dust_dies_after_fifteen_blocks() {
        let mut g = Grid::with_substrate();
        g.set(Pos::new(0, LOGIC_Y as i32, 8), Block::Torch { attach: Attach::Floor });
        for x in 1..16 {
            g.set(Pos::new(x, LOGIC_Y as i32, 8), Block::Wire);
        }
        // Extend into a second row so the run exceeds 15.
        for z in 9..12 {
            g.set(Pos::new(15, LOGIC_Y as i32, z), Block::Wire);
        }
        let lv = solved(g);
        assert_eq!(lv.dust_at(Pos::new(15, LOGIC_Y as i32, 8)), 1);
        assert_eq!(lv.dust_at(Pos::new(15, LOGIC_Y as i32, 9)), 0);
    }

    #[test]
    fn weak_power_does_not_re_emit_into_dust() {
        // dust -> block -> dust. The middle block is only weakly powered, so
        // the far dust must stay dark. This is the rule that separates a real
        // simulator from a plausible one.
        let mut g = Grid::with_substrate();
        let y = LOGIC_Y as i32;
        g.set(Pos::new(0, y, 8), Block::Torch { attach: Attach::Floor });
        g.set(Pos::new(1, y, 8), Block::Wire);
        g.set(Pos::new(2, y, 8), Block::Solid);
        g.set(Pos::new(3, y, 8), Block::Wire);
        let lv = solved(g);
        assert_eq!(lv.dust_at(Pos::new(1, y, 8)), 15);
        assert!(lv.weak[Pos::new(2, y, 8).index()] > 0, "block is weakly powered");
        assert_eq!(lv.strong[Pos::new(2, y, 8).index()], 0, "but not strongly");
        assert_eq!(lv.dust_at(Pos::new(3, y, 8)), 0, "weak power must not re-emit");
    }

    #[test]
    fn strong_power_does_re_emit_into_dust() {
        let mut g = Grid::with_substrate();
        let y = LOGIC_Y as i32;
        // Torch under a block: the block is strongly powered and hands 15 to
        // the dust sitting on top of it.
        g.set(Pos::new(2, y, 8), Block::Torch { attach: Attach::Floor });
        g.set(Pos::new(2, y + 1, 8), Block::Solid);
        g.set(Pos::new(2, y + 2, 8), Block::Wire);
        let lv = solved(g);
        assert_eq!(lv.strong[Pos::new(2, y + 1, 8).index()], 15);
        assert_eq!(lv.dust_at(Pos::new(2, y + 2, 8)), 15);
    }

    #[test]
    fn lone_dust_is_a_dot_and_points_nowhere() {
        let mut g = Grid::with_substrate();
        let y = LOGIC_Y as i32;
        g.set(Pos::new(5, y, 5), Block::Wire);
        assert_eq!(dust_points(dust_links(&g, Pos::new(5, y, 5))), 0);
    }

    #[test]
    fn single_connection_dust_is_a_line() {
        let mut g = Grid::with_substrate();
        let y = LOGIC_Y as i32;
        g.set(Pos::new(5, y, 5), Block::Wire);
        g.set(Pos::new(4, y, 5), Block::Wire);
        let links = dust_links(&g, Pos::new(5, y, 5));
        let points = dust_points(links);
        assert_eq!(links.count_ones(), 1);
        // Points west (its only link) and east (the line rule).
        assert_eq!(points.count_ones(), 2);
        assert!(points & 1 << Dir4::East.index() != 0);
    }

    #[test]
    fn dust_slopes_up_an_open_step() {
        let mut g = Grid::with_substrate();
        let y = LOGIC_Y as i32;
        g.set(Pos::new(4, y, 5), Block::Wire);
        g.set(Pos::new(5, y, 5), Block::Solid);
        g.set(Pos::new(5, y + 1, 5), Block::Wire);
        let mut out = Vec::new();
        wire_neighbours(&g, Pos::new(4, y, 5), &mut out);
        assert!(out.contains(&Pos::new(5, y + 1, 5)), "open step should connect");

        // Roof the source cell and the climb is cut.
        g.set(Pos::new(4, y + 1, 5), Block::Solid);
        wire_neighbours(&g, Pos::new(4, y, 5), &mut out);
        assert!(!out.contains(&Pos::new(5, y + 1, 5)), "roofed step must not connect");
    }
}
