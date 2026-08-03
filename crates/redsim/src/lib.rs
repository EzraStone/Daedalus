//! `redsim` — a deterministic model of a documented subset of Java-edition
//! redstone, plus the verdict function the rest of Daedalus is defined against.
//!
//! This is **not** a Minecraft reimplementation. It is a deliberately bounded
//! model with an explicit divergence list, and a fidelity harness (see
//! `harness/`) whose job is to keep that list honest by replaying the same
//! circuits through the real game.
//!
//! # Scope
//!
//! * Combinational logic only. Latches, clocks and pistons are v2.
//! * 16x6x16 build volume, inputs on the `x=0` face, outputs on `x=15`.
//! * Dust, torches, repeaters (including side-locking), comparators, levers,
//!   lamps, solid blocks and target blocks.
//! * No quasi-connectivity. Modelling it correctly triples the simulator and
//!   makes golden tests version-dependent.
//!
//! # Known divergences from Java edition
//!
//! | Area | Java | Here |
//! |---|---|---|
//! | update order | quasi-random, chunk-order dependent | simultaneous latch-then-apply |
//! | torch burnout | torch goes dark and recovers | circuit is rejected |
//! | quasi-connectivity | present | absent |
//! | target block | comparator-readable | inert conductive block |
//!
//! Any circuit whose behaviour depends on one of these is rejected rather than
//! guessed at, which is the only way the agreement number in the README can
//! mean anything.
//!
//! # Example
//!
//! ```
//! use redsim::{Block, Constraints, Dir4, Grid, Pos, Port, Spec, evaluate};
//! use redsim::block::Attach;
//!
//! let mut g = Grid::with_substrate();
//! g.set(Pos::new(0, 1, 8), Block::Lever { attach: Dir4::East });
//! g.set(Pos::new(1, 1, 8), Block::Solid);
//! g.set(Pos::new(2, 1, 8), Block::Torch { attach: Attach::West });
//! for x in 3..15 { g.set(Pos::new(x, 1, 8), Block::Wire); }
//! g.set(Pos::new(15, 1, 8), Block::Lamp);
//!
//! let spec = Spec::new(
//!     vec![Port { name: "A".into(), pos: Pos::new(0, 1, 8) }],
//!     vec![Port { name: "Q".into(), pos: Pos::new(15, 1, 8) }],
//!     vec![1, 0],                       // Q = not A
//!     Constraints::default(),
//! ).unwrap();
//!
//! assert!(evaluate(&g, &spec).is_pass());
//! ```

pub mod block;
pub mod builder;
pub mod circuit;
pub mod grid;
pub mod power;
pub mod spec;
pub mod tick;
pub mod verdict;

pub use block::{Attach, Block, CmpMode, Dir4, Dir6, CONTROL_BASE, VOCAB_SIZE};
pub use builder::Builder;
pub use circuit::Circuit;
pub use grid::{Grid, Pos, CELLS, SX, SY, SZ};
pub use power::{Levels, States};
pub use spec::{Constraints, Port, Spec, SpecError};
pub use tick::{Settle, Sim, DEFAULT_MAX_GAME_TICKS};
pub use verdict::{
    check_malformed, evaluate, evaluate_tokens, evaluate_with_cap, ConstraintViolation,
    MalformedReason, RowMismatch, Verdict,
};

use rayon::prelude::*;

/// Evaluate many candidate grids against one spec, in parallel.
///
/// This is the shape §06 needs: 64 candidates per spec, 20k specs per round.
/// Verifier throughput is the real scaling axis for the whole project — far
/// more than GPU time — so it gets the thread pool.
pub fn evaluate_batch(grids: &[Vec<u8>], spec: &Spec) -> Vec<Verdict> {
    grids.par_iter().map(|g| evaluate_tokens(g, spec)).collect()
}

/// Evaluate independent `(grid, spec)` pairs in parallel.
pub fn evaluate_pairs(pairs: &[(Vec<u8>, Spec)]) -> Vec<Verdict> {
    pairs.par_iter().map(|(g, s)| evaluate_tokens(g, s)).collect()
}
