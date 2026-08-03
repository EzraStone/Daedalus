//! Block-state vocabulary.
//!
//! One token per *fully resolved* block state, not per block type. A repeater
//! facing north with delay 3 is a different token from one facing north with
//! delay 1, because the two behave differently and nothing downstream should
//! have to infer state from context.
//!
//! Two properties are deliberately **excluded** from the vocabulary because
//! they are derived rather than chosen:
//!
//! * `redstone_wire.power` — that is simulator *output*. Putting it in the
//!   vocabulary would let a generator emit a grid asserting a power level that
//!   the physics cannot produce.
//! * `redstone_wire`'s connection shape (`north/south/east/west`) — fully
//!   determined by neighbours; see [`crate::power::dust_links`].
//!
//! The result is exactly 48 tokens, four of which are sequence-control tokens
//! that never appear in a grid.

use std::fmt;

/// Horizontal direction. `North` is `-z`, `South` is `+z`, `West` is `-x`,
/// `East` is `+x`.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum Dir4 {
    North,
    South,
    West,
    East,
}

pub const DIR4: [Dir4; 4] = [Dir4::North, Dir4::South, Dir4::West, Dir4::East];

impl Dir4 {
    /// `(dx, dz)` step for this direction.
    pub const fn delta(self) -> (i32, i32) {
        match self {
            Dir4::North => (0, -1),
            Dir4::South => (0, 1),
            Dir4::West => (-1, 0),
            Dir4::East => (1, 0),
        }
    }

    pub const fn opposite(self) -> Dir4 {
        match self {
            Dir4::North => Dir4::South,
            Dir4::South => Dir4::North,
            Dir4::West => Dir4::East,
            Dir4::East => Dir4::West,
        }
    }

    /// True when the two directions lie on the same axis.
    pub const fn same_axis(self, other: Dir4) -> bool {
        matches!(
            (self, other),
            (Dir4::North, Dir4::North)
                | (Dir4::North, Dir4::South)
                | (Dir4::South, Dir4::North)
                | (Dir4::South, Dir4::South)
                | (Dir4::West, Dir4::West)
                | (Dir4::West, Dir4::East)
                | (Dir4::East, Dir4::West)
                | (Dir4::East, Dir4::East)
        )
    }

    pub const fn index(self) -> usize {
        match self {
            Dir4::North => 0,
            Dir4::South => 1,
            Dir4::West => 2,
            Dir4::East => 3,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Dir4::North => "north",
            Dir4::South => "south",
            Dir4::West => "west",
            Dir4::East => "east",
        }
    }
}

/// Full six-direction enum, used only by observers.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum Dir6 {
    North,
    South,
    West,
    East,
    Up,
    Down,
}

impl Dir6 {
    pub const fn delta(self) -> (i32, i32, i32) {
        match self {
            Dir6::North => (0, 0, -1),
            Dir6::South => (0, 0, 1),
            Dir6::West => (-1, 0, 0),
            Dir6::East => (1, 0, 0),
            Dir6::Up => (0, 1, 0),
            Dir6::Down => (0, -1, 0),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Dir6::North => "north",
            Dir6::South => "south",
            Dir6::West => "west",
            Dir6::East => "east",
            Dir6::Up => "up",
            Dir6::Down => "down",
        }
    }
}

/// Where a wall-mountable component finds its support block, expressed as the
/// direction *from the component to the block it is attached to*.
///
/// `Floor` means the block directly below. There is no ceiling attachment in
/// v1: hanging torches add a whole update-order wrinkle for no logic value.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum Attach {
    Floor,
    North,
    South,
    West,
    East,
}

impl Attach {
    pub const fn delta(self) -> (i32, i32, i32) {
        match self {
            Attach::Floor => (0, -1, 0),
            Attach::North => (0, 0, -1),
            Attach::South => (0, 0, 1),
            Attach::West => (-1, 0, 0),
            Attach::East => (1, 0, 0),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Attach::Floor => "floor",
            Attach::North => "north",
            Attach::South => "south",
            Attach::West => "west",
            Attach::East => "east",
        }
    }
}

/// Comparator mode.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum CmpMode {
    Compare,
    Subtract,
}

/// A fully resolved block state — one variant per vocabulary entry.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum Block {
    Air,
    /// The single canonical opaque block. Stone vs. deepslate is cosmetic and
    /// would double the vocabulary for zero behavioural signal.
    Solid,
    /// `redstone_wire`. Power level is simulator output, never grid input.
    Wire,
    Torch {
        attach: Attach,
    },
    /// `facing` is the direction the **output** points. The input side is the
    /// opposite face; the two remaining faces are the locking side inputs.
    Repeater {
        facing: Dir4,
        /// 1..=4 redstone ticks.
        delay: u8,
    },
    Comparator {
        facing: Dir4,
        mode: CmpMode,
    },
    Lever {
        attach: Dir4,
    },
    Lamp,
    /// Cheap way to teach signal-strength arithmetic: a target block
    /// re-emits whatever weak/strong power it receives.
    Target,
    /// In the vocabulary, excluded from v1 generation — edge-triggered
    /// components make the circuit sequential and the spec DSL combinational.
    Observer {
        facing: Dir6,
    },
}

/// Total vocabulary size, including the four control tokens.
pub const VOCAB_SIZE: usize = 48;

/// First control-token id. Ids `[0, CONTROL_BASE)` are real block states.
pub const CONTROL_BASE: u8 = 44;
pub const TOK_PAD: u8 = 44;
pub const TOK_BOS: u8 = 45;
pub const TOK_EOS: u8 = 46;
/// Absorbing state for masked discrete diffusion.
pub const TOK_MASK: u8 = 47;

impl Block {
    /// Encode to the canonical token id.
    ///
    /// The layout is stable and is what `daedalus.tokenize` mirrors on the
    /// Python side; changing it invalidates every trained checkpoint.
    pub fn to_token(self) -> u8 {
        match self {
            Block::Air => 0,
            Block::Solid => 1,
            Block::Wire => 2,
            Block::Torch { attach } => {
                3 + match attach {
                    Attach::Floor => 0,
                    Attach::North => 1,
                    Attach::South => 2,
                    Attach::West => 3,
                    Attach::East => 4,
                }
            }
            Block::Repeater { facing, delay } => {
                8 + (facing.index() as u8) * 4 + (delay.clamp(1, 4) - 1)
            }
            Block::Comparator { facing, mode } => {
                24 + (facing.index() as u8) * 2
                    + match mode {
                        CmpMode::Compare => 0,
                        CmpMode::Subtract => 1,
                    }
            }
            Block::Lever { attach } => 32 + attach.index() as u8,
            Block::Lamp => 36,
            Block::Target => 37,
            Block::Observer { facing } => {
                38 + match facing {
                    Dir6::North => 0,
                    Dir6::South => 1,
                    Dir6::West => 2,
                    Dir6::East => 3,
                    Dir6::Up => 4,
                    Dir6::Down => 5,
                }
            }
        }
    }

    /// Decode a token id. Returns `None` for the four control tokens and for
    /// ids outside the vocabulary.
    pub fn from_token(t: u8) -> Option<Block> {
        const DIRS: [Dir4; 4] = DIR4;
        Some(match t {
            0 => Block::Air,
            1 => Block::Solid,
            2 => Block::Wire,
            3 => Block::Torch { attach: Attach::Floor },
            4 => Block::Torch { attach: Attach::North },
            5 => Block::Torch { attach: Attach::South },
            6 => Block::Torch { attach: Attach::West },
            7 => Block::Torch { attach: Attach::East },
            8..=23 => {
                let k = t - 8;
                Block::Repeater { facing: DIRS[(k / 4) as usize], delay: (k % 4) + 1 }
            }
            24..=31 => {
                let k = t - 24;
                Block::Comparator {
                    facing: DIRS[(k / 2) as usize],
                    mode: if k.is_multiple_of(2) { CmpMode::Compare } else { CmpMode::Subtract },
                }
            }
            32..=35 => Block::Lever { attach: DIRS[(t - 32) as usize] },
            36 => Block::Lamp,
            37 => Block::Target,
            38 => Block::Observer { facing: Dir6::North },
            39 => Block::Observer { facing: Dir6::South },
            40 => Block::Observer { facing: Dir6::West },
            41 => Block::Observer { facing: Dir6::East },
            42 => Block::Observer { facing: Dir6::Up },
            43 => Block::Observer { facing: Dir6::Down },
            _ => return None,
        })
    }

    /// Does this block occupy its cell as a full opaque cube?
    ///
    /// Opacity drives three separate rules: dust needs an opaque block beneath
    /// it, dust cannot slope up past an opaque block, and only opaque blocks
    /// carry the weak/strong power distinction.
    pub fn is_opaque(self) -> bool {
        matches!(self, Block::Solid | Block::Lamp | Block::Target | Block::Observer { .. })
    }

    /// Can this block hold weak/strong power as a *block* (rather than being a
    /// component with its own output)?
    pub fn is_conductive(self) -> bool {
        matches!(self, Block::Solid | Block::Lamp | Block::Target)
    }

    /// Can dust sit on top of this block?
    pub fn supports_dust(self) -> bool {
        self.is_opaque()
    }

    /// Does redstone dust form a visual connection toward a neighbouring
    /// `self` placed in direction `from_dust`?
    ///
    /// This is the `canConnectRedstone` predicate. Repeaters and comparators
    /// only connect along their own axis; opaque blocks never connect, which
    /// is why a dust line running into a lamp powers it through the *line*
    /// rule rather than through a connection.
    pub fn connects_dust(self, from_dust: Dir4) -> bool {
        match self {
            Block::Wire => true,
            Block::Torch { .. } => true,
            Block::Lever { .. } => true,
            Block::Repeater { facing, .. } => facing.same_axis(from_dust),
            Block::Comparator { facing, .. } => facing.same_axis(from_dust),
            // Observers only emit from their back face; v1 excludes them from
            // generation entirely, so refusing the connection is the
            // conservative choice.
            Block::Observer { .. } => false,
            Block::Air | Block::Solid | Block::Lamp | Block::Target => false,
        }
    }

    /// Blocks that count toward the reported `blocks` cost of a circuit.
    pub fn is_material(self) -> bool {
        !matches!(self, Block::Air)
    }

    /// Canonical Minecraft block-state string, for `.schem` export and for
    /// error messages.
    pub fn state_string(self) -> String {
        match self {
            Block::Air => "minecraft:air".into(),
            Block::Solid => "minecraft:stone".into(),
            Block::Wire => "minecraft:redstone_wire[power=0]".into(),
            Block::Torch { attach: Attach::Floor } => "minecraft:redstone_torch[lit=true]".into(),
            Block::Torch { attach } => {
                // A wall torch's `facing` is the direction it points *away*
                // from its support, i.e. the opposite of our attach direction.
                let f = match attach {
                    Attach::North => "south",
                    Attach::South => "north",
                    Attach::West => "east",
                    Attach::East => "west",
                    Attach::Floor => unreachable!(),
                };
                format!("minecraft:redstone_wall_torch[facing={f},lit=true]")
            }
            Block::Repeater { facing, delay } => format!(
                "minecraft:repeater[facing={},delay={},locked=false,powered=false]",
                facing.opposite().name(),
                delay
            ),
            Block::Comparator { facing, mode } => format!(
                "minecraft:comparator[facing={},mode={},powered=false]",
                facing.opposite().name(),
                match mode {
                    CmpMode::Compare => "compare",
                    CmpMode::Subtract => "subtract",
                }
            ),
            Block::Lever { attach } => format!(
                "minecraft:lever[face=wall,facing={},powered=false]",
                attach.opposite().name()
            ),
            Block::Lamp => "minecraft:redstone_lamp[lit=false]".into(),
            Block::Target => "minecraft:target[power=0]".into(),
            Block::Observer { facing } => {
                format!("minecraft:observer[facing={},powered=false]", facing.name())
            }
        }
    }
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.state_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_roundtrip_is_total_and_injective() {
        let mut seen = [false; VOCAB_SIZE];
        for t in 0..CONTROL_BASE {
            let b = Block::from_token(t).expect("every non-control id decodes");
            let back = b.to_token();
            assert_eq!(t, back, "token {t} round-tripped to {back} via {b:?}");
            assert!(!seen[t as usize], "duplicate encoding for token {t}");
            seen[t as usize] = true;
        }
        for t in CONTROL_BASE..VOCAB_SIZE as u8 {
            assert!(Block::from_token(t).is_none(), "control token {t} decoded to a block");
        }
    }

    #[test]
    fn vocabulary_is_exactly_48() {
        // 1 air + 1 solid + 1 wire + 5 torch + 16 repeater + 8 comparator
        // + 4 lever + 1 lamp + 1 target + 6 observer + 4 control
        assert_eq!(CONTROL_BASE as usize, 1 + 1 + 1 + 5 + 16 + 8 + 4 + 1 + 1 + 6);
        assert_eq!(VOCAB_SIZE, CONTROL_BASE as usize + 4);
    }

    #[test]
    fn repeater_axis_connection() {
        let r = Block::Repeater { facing: Dir4::East, delay: 1 };
        assert!(r.connects_dust(Dir4::West), "dust behind connects");
        assert!(r.connects_dust(Dir4::East), "dust in front connects");
        assert!(!r.connects_dust(Dir4::North), "side dust does not connect");
    }
}
