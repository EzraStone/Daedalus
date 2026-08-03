//! Ergonomic grid construction.
//!
//! The golden suite is ~120 circuits whose expected verdicts are derived by
//! hand. If building each one takes twenty lines of `grid.set`, the suite will
//! be wrong in ways nobody notices. These helpers exist so a golden test reads
//! as a description of the circuit rather than as a list of coordinates.
//!
//! The primitives here mirror the gate library in `daedalus/synth`: torch-based
//! NOT and NOR cells on a single logic layer, with dust routed between them.
//! Keeping the two in step is what lets the Python placer be checked against
//! Rust-side fixtures.

use crate::block::{Attach, Block, Dir4};
use crate::grid::{Grid, Pos, LOGIC_Y, SX};
use crate::spec::{Constraints, Port, Spec};

/// A grid under construction, with a substrate already laid.
pub struct Builder {
    pub grid: Grid,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    y: i32,
}

impl Default for Builder {
    fn default() -> Self {
        Builder::new()
    }
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            grid: Grid::with_substrate(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            y: LOGIC_Y as i32,
        }
    }

    /// Work on a different logic layer. Everything placed afterwards uses it.
    pub fn layer(mut self, y: i32) -> Builder {
        self.y = y;
        self
    }

    pub fn at(&self, x: i32, z: i32) -> Pos {
        Pos::new(x, self.y, z)
    }

    pub fn put(&mut self, x: i32, z: i32, b: Block) -> &mut Builder {
        self.grid.set(self.at(x, z), b);
        self
    }

    pub fn put_at(&mut self, p: Pos, b: Block) -> &mut Builder {
        self.grid.set(p, b);
        self
    }

    /// An input port: a lever on the `x=0` face hanging off a solid block at
    /// `x=1`. The block is what carries the lever's strong power into the
    /// circuit, so dust picks the signal up from `x=2` onward.
    pub fn input(&mut self, name: &str, z: i32) -> &mut Builder {
        let p = self.at(0, z);
        self.grid.set(p, Block::Lever { attach: Dir4::East });
        self.grid.set(self.at(1, z), Block::Solid);
        self.inputs.push(Port { name: name.into(), pos: p });
        self
    }

    /// An output port: a lamp on the `x=15` face, driven by a repeater at
    /// `x=14`.
    ///
    /// The repeater is not decoration. A dust cell only weakly powers the
    /// blocks it *points* at, and whether it points at the lamp depends on how
    /// many other connections it happens to have. Terminating every output net
    /// in a repeater makes the last hop unconditional, at a cost of 1 rt.
    pub fn output(&mut self, name: &str, z: i32) -> &mut Builder {
        let p = self.at(SX as i32 - 1, z);
        self.grid.set(p, Block::Lamp);
        self.grid.set(self.at(SX as i32 - 2, z), Block::Repeater { facing: Dir4::East, delay: 1 });
        self.outputs.push(Port { name: name.into(), pos: p });
        self
    }

    /// As [`Builder::output`], with a chosen repeater delay so a test can dial
    /// the circuit's latency directly.
    pub fn output_delay(&mut self, name: &str, z: i32, delay: u8) -> &mut Builder {
        let p = self.at(SX as i32 - 1, z);
        self.grid.set(p, Block::Lamp);
        self.grid.set(self.at(SX as i32 - 2, z), Block::Repeater { facing: Dir4::East, delay });
        self.outputs.push(Port { name: name.into(), pos: p });
        self
    }

    /// A lamp with no repeater in front, for tests that want to exercise the
    /// dust-pointing rule directly.
    pub fn bare_output(&mut self, name: &str, z: i32) -> &mut Builder {
        let p = self.at(SX as i32 - 1, z);
        self.grid.set(p, Block::Lamp);
        self.outputs.push(Port { name: name.into(), pos: p });
        self
    }

    /// Dust along a row, `x0..=x1` inclusive.
    pub fn dust_x(&mut self, x0: i32, x1: i32, z: i32) -> &mut Builder {
        let (a, b) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        for x in a..=b {
            self.grid.set(self.at(x, z), Block::Wire);
        }
        self
    }

    /// Dust along a column, `z0..=z1` inclusive.
    pub fn dust_z(&mut self, x: i32, z0: i32, z1: i32) -> &mut Builder {
        let (a, b) = if z0 <= z1 { (z0, z1) } else { (z1, z0) };
        for z in a..=b {
            self.grid.set(self.at(x, z), Block::Wire);
        }
        self
    }

    /// An inverting cell: a solid block at `x` with a wall torch on its east
    /// face at `x+1`. Feed it from `x-1`, read it from `x+2`.
    ///
    /// This is the only active primitive v1 needs — NOT and NOR between them
    /// span all of combinational logic, and both are this same cell with a
    /// different number of dust lines running into the block.
    pub fn invert(&mut self, x: i32, z: i32) -> &mut Builder {
        self.grid.set(self.at(x, z), Block::Solid);
        self.grid.set(self.at(x + 1, z), Block::Torch { attach: Attach::West });
        self
    }

    /// A repeater pointing east, used to restore a signal that has run more
    /// than 15 blocks.
    pub fn repeat(&mut self, x: i32, z: i32, delay: u8) -> &mut Builder {
        self.grid.set(self.at(x, z), Block::Repeater { facing: Dir4::East, delay });
        self
    }

    pub fn solid(&mut self, x: i32, z: i32) -> &mut Builder {
        self.grid.set(self.at(x, z), Block::Solid);
        self
    }

    pub fn target(&mut self, x: i32, z: i32) -> &mut Builder {
        self.grid.set(self.at(x, z), Block::Target);
        self
    }

    /// A comparator pointing east.
    pub fn compare(&mut self, x: i32, z: i32, mode: crate::block::CmpMode) -> &mut Builder {
        self.grid.set(self.at(x, z), Block::Comparator { facing: Dir4::East, mode });
        self
    }

    /// A torch tower: a floor torch at `(x, z)` with a solid cap above it, so
    /// the signal comes back out one layer up. The cap is strongly powered,
    /// which is what lets it re-emit into dust.
    pub fn tower(&mut self, x: i32, z: i32) -> &mut Builder {
        self.grid.set(self.at(x, z), Block::Torch { attach: Attach::Floor });
        self.grid.set(Pos::new(x, self.y + 1, z), Block::Solid);
        self
    }

    pub fn ports(&self) -> (&[Port], &[Port]) {
        (&self.inputs, &self.outputs)
    }

    /// Finish, pairing the grid with a spec built from the declared ports and
    /// a caller-supplied truth table.
    pub fn spec(&self, rows: Vec<u64>, constraints: Constraints) -> Spec {
        Spec::new(self.inputs.clone(), self.outputs.clone(), rows, constraints)
            .expect("builder ports are well formed by construction")
    }

    pub fn build(self) -> Grid {
        self.grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verdict::evaluate;

    #[test]
    fn builder_makes_a_working_inverter() {
        let mut b = Builder::new();
        b.input("A", 8);
        b.dust_x(2, 3, 8);
        b.invert(4, 8);
        b.dust_x(6, 13, 8);
        b.output("Q", 8);
        let spec = b.spec(vec![1, 0], Constraints::default());
        let v = evaluate(&b.grid, &spec);
        assert!(v.is_pass(), "{v}");
    }
}
