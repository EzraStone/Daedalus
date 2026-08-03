//! The machine-readable side of a specification.
//!
//! The DSL of §04 lives in Python; by the time a spec reaches the verifier it
//! has been reduced to the only thing the simulator can check — a truth table
//! bound to concrete port positions, plus hard constraints.
//!
//! Natural language is a *view* of a spec, never its source of truth, and it
//! does not appear here at all.

use crate::grid::{Pos, INPUT_X, OUTPUT_X};

/// Maximum inputs a v1 spec may declare. 6 gives a 64-row truth table, which
/// is what §03 budgets per candidate.
pub const MAX_INPUTS: usize = 6;
pub const MAX_OUTPUTS: usize = 6;

/// A named port pinned to a cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Port {
    pub name: String,
    pub pos: Pos,
}

/// Hard constraints. A candidate that violates one is not a pass, however
/// correct its truth table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Constraints {
    pub max_latency_rt: Option<u32>,
    pub max_blocks: Option<u16>,
    /// `(x, z)` extent limit.
    pub max_region: Option<(u8, u8)>,
}

/// A fully resolved specification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    /// `rows[m]` holds the expected output bits for the input assignment whose
    /// bitmask is `m`; input `k` is bit `k`, output `j` is bit `j`.
    /// Length is exactly `1 << inputs.len()`.
    pub rows: Vec<u64>,
    pub constraints: Constraints,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecError {
    NoInputs,
    TooManyInputs(usize),
    TooManyOutputs(usize),
    NoOutputs,
    WrongTableLength { got: usize, want: usize },
    InputNotOnInputFace(Pos),
    OutputNotOnOutputFace(Pos),
    DuplicatePort(Pos),
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecError::NoInputs => write!(f, "a spec needs at least one input"),
            SpecError::TooManyInputs(n) => {
                write!(f, "{n} inputs exceeds the v1 limit of {MAX_INPUTS}")
            }
            SpecError::TooManyOutputs(n) => {
                write!(f, "{n} outputs exceeds the v1 limit of {MAX_OUTPUTS}")
            }
            SpecError::NoOutputs => write!(f, "a spec needs at least one output"),
            SpecError::WrongTableLength { got, want } => {
                write!(f, "truth table has {got} rows, expected {want}")
            }
            SpecError::InputNotOnInputFace(p) => {
                write!(f, "input port at {p:?} is not on the x={INPUT_X} face")
            }
            SpecError::OutputNotOnOutputFace(p) => {
                write!(f, "output port at {p:?} is not on the x={OUTPUT_X} face")
            }
            SpecError::DuplicatePort(p) => write!(f, "two ports share cell {p:?}"),
        }
    }
}

impl std::error::Error for SpecError {}

impl Spec {
    pub fn new(
        inputs: Vec<Port>,
        outputs: Vec<Port>,
        rows: Vec<u64>,
        constraints: Constraints,
    ) -> Result<Spec, SpecError> {
        if inputs.is_empty() {
            return Err(SpecError::NoInputs);
        }
        if inputs.len() > MAX_INPUTS {
            return Err(SpecError::TooManyInputs(inputs.len()));
        }
        if outputs.is_empty() {
            return Err(SpecError::NoOutputs);
        }
        if outputs.len() > MAX_OUTPUTS {
            return Err(SpecError::TooManyOutputs(outputs.len()));
        }
        let want = 1usize << inputs.len();
        if rows.len() != want {
            return Err(SpecError::WrongTableLength { got: rows.len(), want });
        }
        // Fixing port positions removes a large nuisance degree of freedom in
        // v1; free placement is a v2 ablation. Enforcing it here means the
        // generator can never quietly move a port to make a circuit "work".
        let mut seen = Vec::new();
        for p in &inputs {
            if p.pos.x != INPUT_X as i32 {
                return Err(SpecError::InputNotOnInputFace(p.pos));
            }
            if seen.contains(&p.pos) {
                return Err(SpecError::DuplicatePort(p.pos));
            }
            seen.push(p.pos);
        }
        for p in &outputs {
            if p.pos.x != OUTPUT_X as i32 {
                return Err(SpecError::OutputNotOnOutputFace(p.pos));
            }
            if seen.contains(&p.pos) {
                return Err(SpecError::DuplicatePort(p.pos));
            }
            seen.push(p.pos);
        }
        Ok(Spec { inputs, outputs, rows, constraints })
    }

    pub fn n_inputs(&self) -> usize {
        self.inputs.len()
    }

    pub fn n_outputs(&self) -> usize {
        self.outputs.len()
    }

    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    /// Expected value of output `j` under input assignment `m`.
    pub fn expect(&self, m: usize, j: usize) -> bool {
        (self.rows[m] >> j) & 1 == 1
    }

    /// A stable fingerprint of *what the spec means*, independent of port
    /// names and positions.
    ///
    /// This is the deduplication key for the corpus and the basis of the
    /// novelty metric: two specs with the same behaviour hash to the same
    /// value even if one calls its inputs `A B` and the other `sw1 sw2`.
    pub fn semantic_hash(&self) -> u64 {
        // FNV-1a over (n_inputs, n_outputs, table). Chosen over the default
        // hasher because the value is written into dataset files and has to
        // stay identical across Rust releases.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |b: u8| {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        };
        mix(self.n_inputs() as u8);
        mix(self.n_outputs() as u8);
        for r in &self.rows {
            for k in 0..8 {
                mix((r >> (k * 8)) as u8);
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::LOGIC_Y;

    fn port(name: &str, x: i32, z: i32) -> Port {
        Port { name: name.into(), pos: Pos::new(x, LOGIC_Y as i32, z) }
    }

    fn nand() -> Spec {
        // out = !(A & B): 1,1,1,0 for A,B = 00,01,10,11
        Spec::new(
            vec![port("A", 0, 4), port("B", 0, 8)],
            vec![port("Q", 15, 6)],
            vec![1, 1, 1, 0],
            Constraints::default(),
        )
        .unwrap()
    }

    #[test]
    fn ports_must_sit_on_their_faces() {
        let bad = Spec::new(
            vec![port("A", 3, 4)],
            vec![port("Q", 15, 6)],
            vec![0, 1],
            Constraints::default(),
        );
        assert!(matches!(bad, Err(SpecError::InputNotOnInputFace(_))));
    }

    #[test]
    fn table_length_is_checked() {
        let bad = Spec::new(
            vec![port("A", 0, 4), port("B", 0, 8)],
            vec![port("Q", 15, 6)],
            vec![1, 1, 1],
            Constraints::default(),
        );
        assert_eq!(bad, Err(SpecError::WrongTableLength { got: 3, want: 4 }));
    }

    #[test]
    fn semantic_hash_ignores_names_and_positions() {
        let a = nand();
        let b = Spec::new(
            vec![port("sw1", 0, 1), port("sw2", 0, 14)],
            vec![port("light", 15, 2)],
            vec![1, 1, 1, 0],
            Constraints::default(),
        )
        .unwrap();
        assert_eq!(a.semantic_hash(), b.semantic_hash());
    }

    #[test]
    fn semantic_hash_separates_different_functions() {
        let nand = nand();
        let and = Spec::new(
            vec![port("A", 0, 4), port("B", 0, 8)],
            vec![port("Q", 15, 6)],
            vec![0, 0, 0, 1],
            Constraints::default(),
        )
        .unwrap();
        assert_ne!(nand.semantic_hash(), and.semantic_hash());
    }
}
