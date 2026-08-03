//! The dense voxel grid.
//!
//! Redstone is overwhelmingly planar: circuits spread in `x`/`z` and use `y`
//! only for torch towers, dust-over-block runs and the occasional two-level
//! crossing. A cube is the wrong shape, so v1 is a slab.
//!
//! Token order is `y → z → x` (layer-major). Keeping a full planar layer
//! contiguous in the sequence is what lets a local-attention model see a whole
//! slice at once.

use crate::block::{Attach, Block, Dir4};

pub const SX: usize = 16;
pub const SY: usize = 6;
pub const SZ: usize = 16;
pub const CELLS: usize = SX * SY * SZ;

/// Layer 0 is the substrate. Dust lives on `y = 1` and up, which removes an
/// entire class of "floating dust" invalid samples by construction.
pub const SUBSTRATE_Y: usize = 0;
pub const LOGIC_Y: usize = 1;

/// Inputs sit on the `x = 0` face, outputs on `x = SX - 1`.
pub const INPUT_X: usize = 0;
pub const OUTPUT_X: usize = SX - 1;

/// A cell coordinate. Kept as `i32` so neighbour arithmetic can go out of
/// bounds before being checked.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Pos {
    pub const fn new(x: i32, y: i32, z: i32) -> Pos {
        Pos { x, y, z }
    }

    pub const fn in_bounds(self) -> bool {
        self.x >= 0
            && self.y >= 0
            && self.z >= 0
            && (self.x as usize) < SX
            && (self.y as usize) < SY
            && (self.z as usize) < SZ
    }

    /// Flat index in `y → z → x` raster order.
    pub const fn index(self) -> usize {
        (self.y as usize * SZ + self.z as usize) * SX + self.x as usize
    }

    pub const fn from_index(i: usize) -> Pos {
        let x = i % SX;
        let z = (i / SX) % SZ;
        let y = i / (SX * SZ);
        Pos::new(x as i32, y as i32, z as i32)
    }

    pub const fn offset(self, dx: i32, dy: i32, dz: i32) -> Pos {
        Pos::new(self.x + dx, self.y + dy, self.z + dz)
    }

    pub const fn step(self, d: Dir4) -> Pos {
        let (dx, dz) = d.delta();
        Pos::new(self.x + dx, self.y, self.z + dz)
    }

    pub const fn up(self) -> Pos {
        Pos::new(self.x, self.y + 1, self.z)
    }

    pub const fn down(self) -> Pos {
        Pos::new(self.x, self.y - 1, self.z)
    }

    pub const fn attached(self, a: Attach) -> Pos {
        let (dx, dy, dz) = a.delta();
        Pos::new(self.x + dx, self.y + dy, self.z + dz)
    }

    /// The six axis-aligned neighbours, in a fixed order. Determinism of this
    /// order matters: it is what makes the whole simulator reproducible.
    pub fn neighbours6(self) -> [Pos; 6] {
        [
            self.offset(0, 0, -1),
            self.offset(0, 0, 1),
            self.offset(-1, 0, 0),
            self.offset(1, 0, 0),
            self.offset(0, 1, 0),
            self.offset(0, -1, 0),
        ]
    }
}

/// A 16x6x16 dense array of fully resolved block states.
#[derive(Clone, PartialEq, Eq)]
pub struct Grid {
    cells: Vec<Block>,
}

impl Default for Grid {
    fn default() -> Self {
        Grid::new()
    }
}

impl Grid {
    /// An all-air grid. Note this is *not* a legal circuit — the substrate has
    /// to be laid down explicitly, which keeps "there is a floor" an assertion
    /// rather than an assumption.
    pub fn new() -> Grid {
        Grid { cells: vec![Block::Air; CELLS] }
    }

    /// All-air except a solid layer at `y = 0`.
    pub fn with_substrate() -> Grid {
        let mut g = Grid::new();
        for z in 0..SZ {
            for x in 0..SX {
                g.set(Pos::new(x as i32, SUBSTRATE_Y as i32, z as i32), Block::Solid);
            }
        }
        g
    }

    #[inline]
    pub fn get(&self, p: Pos) -> Block {
        if p.in_bounds() {
            self.cells[p.index()]
        } else {
            // Outside the build volume is void, not stone. Treating it as air
            // means a circuit cannot lean on the world border for support.
            Block::Air
        }
    }

    #[inline]
    pub fn set(&mut self, p: Pos, b: Block) {
        if p.in_bounds() {
            self.cells[p.index()] = b;
        }
    }

    #[inline]
    pub fn at(&self, i: usize) -> Block {
        self.cells[i]
    }

    pub fn cells(&self) -> &[Block] {
        &self.cells
    }

    /// Decode a `y → z → x` token sequence of exactly [`CELLS`] entries.
    ///
    /// Control tokens are not legal inside a grid body; they decode to air so
    /// that a half-denoised sample can still be inspected, and the malformed
    /// check in [`crate::verdict`] is what actually rejects them.
    pub fn from_tokens(tokens: &[u8]) -> Result<Grid, GridError> {
        if tokens.len() != CELLS {
            return Err(GridError::WrongLength { got: tokens.len(), want: CELLS });
        }
        let mut g = Grid::new();
        for (i, &t) in tokens.iter().enumerate() {
            if t as usize >= crate::block::VOCAB_SIZE {
                return Err(GridError::BadToken { index: i, token: t });
            }
            g.cells[i] = Block::from_token(t).unwrap_or(Block::Air);
        }
        Ok(g)
    }

    pub fn to_tokens(&self) -> Vec<u8> {
        self.cells.iter().map(|b| b.to_token()).collect()
    }

    /// Count of non-air cells, excluding the substrate layer.
    ///
    /// The substrate is a fixed cost every circuit pays and including it would
    /// swamp the compactness metric with a constant 256.
    pub fn material_blocks(&self) -> u16 {
        let mut n = 0u16;
        for y in 1..SY {
            for z in 0..SZ {
                for x in 0..SX {
                    if self.get(Pos::new(x as i32, y as i32, z as i32)).is_material() {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    /// Bounding box of all non-air cells above the substrate, as `(dx, dy, dz)`
    /// extents. An empty circuit reports `(0, 0, 0)`.
    pub fn bbox(&self) -> (u8, u8, u8) {
        let (mut x0, mut y0, mut z0) = (SX, SY, SZ);
        let (mut x1, mut y1, mut z1) = (0usize, 0usize, 0usize);
        let mut any = false;
        for y in 1..SY {
            for z in 0..SZ {
                for x in 0..SX {
                    if self.get(Pos::new(x as i32, y as i32, z as i32)).is_material() {
                        any = true;
                        x0 = x0.min(x);
                        y0 = y0.min(y);
                        z0 = z0.min(z);
                        x1 = x1.max(x);
                        y1 = y1.max(y);
                        z1 = z1.max(z);
                    }
                }
            }
        }
        if !any {
            return (0, 0, 0);
        }
        ((x1 - x0 + 1) as u8, (y1 - y0 + 1) as u8, (z1 - z0 + 1) as u8)
    }

    /// Positions of every cell holding a given predicate, in raster order.
    pub fn find(&self, pred: impl Fn(Block) -> bool) -> Vec<Pos> {
        (0..CELLS).filter(|&i| pred(self.cells[i])).map(Pos::from_index).collect()
    }
}

impl std::fmt::Debug for Grid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Grid {SX}x{SY}x{SZ}")?;
        for y in 0..SY {
            let empty = (0..SZ * SX).all(|i| {
                let x = i % SX;
                let z = i / SX;
                !self.get(Pos::new(x as i32, y as i32, z as i32)).is_material()
            });
            if empty {
                continue;
            }
            writeln!(f, "-- y={y}")?;
            for z in 0..SZ {
                let row: String = (0..SX)
                    .map(|x| glyph(self.get(Pos::new(x as i32, y as i32, z as i32))))
                    .collect();
                writeln!(f, "  {row}")?;
            }
        }
        Ok(())
    }
}

/// Single-character rendering used by `Debug` and by the ASCII grid format.
pub fn glyph(b: Block) -> char {
    match b {
        Block::Air => '.',
        Block::Solid => '#',
        Block::Wire => 'd',
        Block::Torch { .. } => 't',
        Block::Repeater { .. } => '>',
        Block::Comparator { .. } => 'c',
        Block::Lever { .. } => 'V',
        Block::Lamp => 'L',
        Block::Target => 'T',
        Block::Observer { .. } => 'o',
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GridError {
    WrongLength { got: usize, want: usize },
    BadToken { index: usize, token: u8 },
}

impl std::fmt::Display for GridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GridError::WrongLength { got, want } => {
                write!(f, "grid needs exactly {want} tokens, got {got}")
            }
            GridError::BadToken { index, token } => {
                write!(f, "token {token} at index {index} is outside the vocabulary")
            }
        }
    }
}

impl std::error::Error for GridError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_is_layer_major() {
        // Two cells in the same y-layer and the same z-row must be adjacent in
        // the sequence; that is the property local attention relies on.
        let a = Pos::new(3, 2, 5).index();
        let b = Pos::new(4, 2, 5).index();
        assert_eq!(b, a + 1);
        // A whole layer is contiguous.
        assert_eq!(Pos::new(0, 3, 0).index(), 3 * SX * SZ);
    }

    #[test]
    fn index_roundtrip() {
        for i in 0..CELLS {
            assert_eq!(Pos::from_index(i).index(), i);
        }
    }

    #[test]
    fn out_of_bounds_reads_as_air() {
        let g = Grid::with_substrate();
        assert_eq!(g.get(Pos::new(-1, 0, 0)), Block::Air);
        assert_eq!(g.get(Pos::new(0, 0, 0)), Block::Solid);
    }

    #[test]
    fn substrate_is_not_counted_as_material() {
        let g = Grid::with_substrate();
        assert_eq!(g.material_blocks(), 0);
        assert_eq!(g.bbox(), (0, 0, 0));
    }
}
