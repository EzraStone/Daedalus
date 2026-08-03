//! A grid with its component index precomputed.
//!
//! Roughly 85% of every grid is air, and the tick loop touches the same cells
//! over and over. Building the index once and iterating it — rather than
//! sweeping all 1536 cells per tick — is the difference between a simulator
//! fast enough to sit inside a sampling loop and one that isn't.

use crate::block::Block;
use crate::grid::{Grid, Pos, CELLS};

/// Immutable view of a grid, indexed by component kind.
pub struct Circuit {
    pub grid: Grid,
    /// Every `redstone_wire` cell.
    pub wires: Vec<usize>,
    /// Every block that can hold weak/strong power.
    pub conductive: Vec<usize>,
    pub torches: Vec<usize>,
    pub levers: Vec<usize>,
    pub repeaters: Vec<usize>,
    pub comparators: Vec<usize>,
    pub lamps: Vec<usize>,
    /// Torches, levers, repeaters and comparators, in raster order — the
    /// components that emit signal on their own.
    pub emitters: Vec<usize>,
    /// Torches, repeaters and comparators: everything with latched state that
    /// the tick loop has to schedule.
    pub stateful: Vec<usize>,
}

impl Circuit {
    pub fn new(grid: Grid) -> Circuit {
        let mut c = Circuit {
            grid,
            wires: Vec::new(),
            conductive: Vec::new(),
            torches: Vec::new(),
            levers: Vec::new(),
            repeaters: Vec::new(),
            comparators: Vec::new(),
            lamps: Vec::new(),
            emitters: Vec::new(),
            stateful: Vec::new(),
        };
        for i in 0..CELLS {
            let b = c.grid.at(i);
            match b {
                Block::Air => continue,
                Block::Wire => c.wires.push(i),
                Block::Torch { .. } => {
                    c.torches.push(i);
                    c.emitters.push(i);
                    c.stateful.push(i);
                }
                Block::Lever { .. } => {
                    c.levers.push(i);
                    c.emitters.push(i);
                }
                Block::Repeater { .. } => {
                    c.repeaters.push(i);
                    c.emitters.push(i);
                    c.stateful.push(i);
                }
                Block::Comparator { .. } => {
                    c.comparators.push(i);
                    c.emitters.push(i);
                    c.stateful.push(i);
                }
                Block::Lamp => c.lamps.push(i),
                _ => {}
            }
            if b.is_conductive() {
                c.conductive.push(i);
            }
        }
        c
    }

    #[inline]
    pub fn get(&self, p: Pos) -> Block {
        self.grid.get(p)
    }

    #[inline]
    pub fn at(&self, i: usize) -> Block {
        self.grid.at(i)
    }

    pub fn into_grid(self) -> Grid {
        self.grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Attach;
    use crate::grid::LOGIC_Y;

    #[test]
    fn index_partitions_the_grid() {
        let mut g = Grid::with_substrate();
        let y = LOGIC_Y as i32;
        g.set(Pos::new(1, y, 1), Block::Wire);
        g.set(Pos::new(2, y, 1), Block::Torch { attach: Attach::Floor });
        g.set(Pos::new(3, y, 1), Block::Lamp);
        let c = Circuit::new(g);
        assert_eq!(c.wires.len(), 1);
        assert_eq!(c.torches.len(), 1);
        assert_eq!(c.lamps.len(), 1);
        // 256 substrate blocks plus the lamp.
        assert_eq!(c.conductive.len(), 257);
        assert_eq!(c.stateful, c.torches);
    }
}
