//! `redsim` — the verifier as a long-lived worker process.
//!
//! Python drives the simulator over a length-prefixed binary protocol on
//! stdin/stdout rather than through a native extension. That is a deliberate
//! trade: PyO3 would shave a few microseconds per call, but it also means a
//! compiler toolchain, an ABI to keep in step, and a build step between
//! "clone the repo" and "run the tests". A pipe has none of that, batches
//! amortise the syscall away, and the same binary is usable from any language.
//!
//! Protocol, all integers little-endian:
//!
//! ```text
//! request  := "RSIM" u8:version u8:op spec u32:n_grids (1536 bytes)*
//! spec     := u8:n_in  (u8 x, u8 y, u8 z)*
//!             u8:n_out (u8 x, u8 y, u8 z)*
//!             u32:n_rows u64:row*
//!             u8:flags u32:max_latency u16:max_blocks u8:region_x u8:region_z
//! response := "RSOK" u32:n_verdicts verdict*
//! verdict  := u8:kind ...                     -- see `write_verdict`
//! ```

use std::io::{self, BufWriter, Read, Write};

use redsim::grid::CELLS;
use redsim::spec::{Constraints, Port};
use redsim::verdict::{ConstraintViolation, MalformedReason};
use redsim::{evaluate_batch, Pos, Spec, Verdict};

const MAGIC_REQ: &[u8; 4] = b"RSIM";
const MAGIC_RESP: &[u8; 4] = b"RSOK";
const PROTOCOL_VERSION: u8 = 2;
const OP_EVALUATE: u8 = 1;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("serve") => {
            if let Err(e) = serve() {
                if e.kind() != io::ErrorKind::UnexpectedEof {
                    eprintln!("redsim: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some("--version") | Some("version") => {
            println!("redsim {}", env!("CARGO_PKG_VERSION"));
        }
        Some("selftest") => selftest(),
        Some("vocab") => dump_vocab(),
        Some(other) => {
            eprintln!("redsim: unknown command {other:?}");
            eprintln!("usage: redsim [serve|selftest|vocab|version]");
            std::process::exit(2);
        }
    }
}

/// Read requests until stdin closes. One process serves an entire training
/// round; the cost of spawning is paid once, not per candidate.
fn serve() -> io::Result<()> {
    let stdin = io::stdin();
    let mut r = stdin.lock();
    let stdout = io::stdout();
    let mut w = BufWriter::new(stdout.lock());

    loop {
        let mut magic = [0u8; 4];
        if !read_exact_or_eof(&mut r, &mut magic)? {
            return Ok(());
        }
        if &magic != MAGIC_REQ {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad request magic"));
        }
        let version = read_u8(&mut r)?;
        if version != PROTOCOL_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("protocol version {version} != {PROTOCOL_VERSION}"),
            ));
        }
        let op = read_u8(&mut r)?;
        if op != OP_EVALUATE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unknown op {op}")));
        }

        let spec = read_spec(&mut r)?;
        let n = read_u32(&mut r)? as usize;
        let mut grids = Vec::with_capacity(n);
        for _ in 0..n {
            let mut buf = vec![0u8; CELLS];
            r.read_exact(&mut buf)?;
            grids.push(buf);
        }

        let verdicts = evaluate_batch(&grids, &spec);

        w.write_all(MAGIC_RESP)?;
        w.write_all(&(verdicts.len() as u32).to_le_bytes())?;
        for v in &verdicts {
            write_verdict(&mut w, v)?;
        }
        w.flush()?;
    }
}

fn read_exact_or_eof<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..])? {
            0 if filled == 0 => return Ok(false),
            0 => return Err(io::ErrorKind::UnexpectedEof.into()),
            n => filled += n,
        }
    }
    Ok(true)
}

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_ports<R: Read>(r: &mut R, prefix: &str) -> io::Result<Vec<Port>> {
    let n = read_u8(r)? as usize;
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let x = read_u8(r)? as i32;
        let y = read_u8(r)? as i32;
        let z = read_u8(r)? as i32;
        out.push(Port { name: format!("{prefix}{k}"), pos: Pos::new(x, y, z) });
    }
    Ok(out)
}

fn read_spec<R: Read>(r: &mut R) -> io::Result<Spec> {
    let inputs = read_ports(r, "in")?;
    let outputs = read_ports(r, "out")?;
    let n_rows = read_u32(r)? as usize;
    let mut rows = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        rows.push(read_u64(r)?);
    }
    let flags = read_u8(r)?;
    let latency = read_u32(r)?;
    let blocks = read_u16(r)?;
    let rx = read_u8(r)?;
    let rz = read_u8(r)?;
    let constraints = Constraints {
        max_latency_rt: (flags & 1 != 0).then_some(latency),
        max_blocks: (flags & 2 != 0).then_some(blocks),
        max_region: (flags & 4 != 0).then_some((rx, rz)),
    };
    Spec::new(inputs, outputs, rows, constraints)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
}

fn malformed_code(reason: &MalformedReason) -> (u8, Pos) {
    match reason {
        MalformedReason::FloatingDust { at } => (0, *at),
        MalformedReason::PortViolation { at } => (1, *at),
        MalformedReason::Unsupported { at } => (2, *at),
        MalformedReason::ExcludedBlock { at } => (3, *at),
        MalformedReason::MaskedCell { at } => (4, *at),
        MalformedReason::Burnout { at } => (5, *at),
        MalformedReason::HistoryDependent => (6, Pos::new(0, 0, 0)),
    }
}

fn write_verdict<W: Write>(w: &mut W, v: &Verdict) -> io::Result<()> {
    match v {
        Verdict::Pass { latency_rt, blocks, bbox } => {
            w.write_all(&[0, *latency_rt])?;
            w.write_all(&blocks.to_le_bytes())?;
            w.write_all(&[bbox.0, bbox.1, bbox.2])?;
        }
        Verdict::Fail { mismatched_rows, constraint } => {
            w.write_all(&[1])?;
            w.write_all(&(mismatched_rows.len() as u16).to_le_bytes())?;
            for m in mismatched_rows {
                w.write_all(&m.inputs.to_le_bytes())?;
                w.write_all(&m.observed.to_le_bytes())?;
                w.write_all(&m.expected.to_le_bytes())?;
            }
            // The code alone says which budget was missed but not by how
            // much, and the numbers are right here. "38 blocks against a
            // budget of 34" is something a caller can act on; "blocks" is not.
            // Scalar budgets leave the second component zero; a region uses
            // both, as (x, z).
            let (code, got, max) = match constraint {
                None => (0u8, (0u32, 0u32), (0u32, 0u32)),
                Some(ConstraintViolation::Latency { got, max }) => {
                    (1u8, (*got, 0), (*max, 0))
                }
                Some(ConstraintViolation::Blocks { got, max }) => {
                    (2u8, (u32::from(*got), 0), (u32::from(*max), 0))
                }
                Some(ConstraintViolation::Region { got, max }) => (
                    3u8,
                    (u32::from(got.0), u32::from(got.1)),
                    (u32::from(max.0), u32::from(max.1)),
                ),
            };
            w.write_all(&[code])?;
            if code != 0 {
                for v in [got.0, got.1, max.0, max.1] {
                    w.write_all(&v.to_le_bytes())?;
                }
            }
        }
        Verdict::Unstable { period_ticks } => w.write_all(&[2, *period_ticks])?,
        Verdict::Malformed { reason } => {
            let (code, at) = malformed_code(reason);
            w.write_all(&[3, code, at.x as u8, at.y as u8, at.z as u8])?;
        }
    }
    Ok(())
}

/// Dump the vocabulary and a set of reference spec hashes as text.
///
/// The Python side re-derives all of this independently, and
/// `tests/test_vocab_parity.py` diffs the two. Token ids are a wire format
/// baked into every serialised corpus and every checkpoint, so "the two
/// implementations agree" has to be a test, not a comment.
fn dump_vocab() {
    use redsim::block::{Block, CONTROL_BASE, VOCAB_SIZE};
    use redsim::grid::glyph;

    println!("# id\tglyph\topaque\tconductive\tstate");
    for t in 0..CONTROL_BASE {
        let b = Block::from_token(t).expect("non-control ids decode");
        println!(
            "{t}\t{}\t{}\t{}\t{}",
            glyph(b),
            u8::from(b.is_opaque()),
            u8::from(b.is_conductive()),
            b.state_string()
        );
    }
    for t in CONTROL_BASE..VOCAB_SIZE as u8 {
        let name = match t {
            redsim::block::TOK_PAD => "PAD",
            redsim::block::TOK_BOS => "BOS",
            redsim::block::TOK_EOS => "EOS",
            _ => "MASK",
        };
        println!("{t}\t-\t0\t0\tcontrol:{name}");
    }

    println!("# geometry\tsx\tsy\tsz\tcells\tsubstrate_y\tlogic_y\tinput_x\toutput_x");
    println!(
        "geometry\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        redsim::grid::SX,
        redsim::grid::SY,
        redsim::grid::SZ,
        redsim::grid::CELLS,
        redsim::grid::SUBSTRATE_Y,
        redsim::grid::LOGIC_Y,
        redsim::grid::INPUT_X,
        redsim::grid::OUTPUT_X
    );

    // Reference semantic hashes. Any drift here means the corpus dedup key
    // has silently changed and every cached dataset is stale.
    println!("# hash\tn_in\tn_out\trows...\tvalue");
    for (n_in, n_out, rows) in [
        (1usize, 1usize, vec![0u64, 1]),
        (1, 1, vec![1, 0]),
        (2, 1, vec![1, 1, 1, 0]),
        (2, 1, vec![0, 0, 0, 1]),
        (2, 1, vec![0, 1, 1, 0]),
        (2, 2, vec![0b00, 0b01, 0b10, 0b11]),
        (3, 1, vec![0, 1, 1, 0, 1, 0, 0, 1]),
    ] {
        let inputs: Vec<Port> = (0..n_in)
            .map(|k| Port { name: format!("in{k}"), pos: Pos::new(0, 1, 2 * k as i32 + 1) })
            .collect();
        let outputs: Vec<Port> = (0..n_out)
            .map(|k| Port { name: format!("out{k}"), pos: Pos::new(15, 1, 2 * k as i32 + 1) })
            .collect();
        let spec = Spec::new(inputs, outputs, rows.clone(), Constraints::default()).unwrap();
        let cells: Vec<String> = rows.iter().map(|r| r.to_string()).collect();
        println!("hash\t{n_in}\t{n_out}\t{}\t{:016x}", cells.join(","), spec.semantic_hash());
    }
}

/// A five-second sanity check that the binary works at all, for anyone who
/// just cloned the repo and wants to know before reading further.
fn selftest() {
    use redsim::builder::Builder;
    use redsim::{evaluate, Constraints};

    // NAND = !(A & B) = !A | !B. Invert each input with a torch, then merge
    // the two dust runs — a dust join is a free OR.
    let mut b = Builder::new();
    b.input("A", 6);
    b.dust_x(2, 3, 6);
    b.invert(4, 6);
    b.dust_x(6, 7, 6);

    b.input("B", 10);
    b.dust_x(2, 3, 10);
    b.invert(4, 10);
    b.dust_x(6, 7, 10);

    b.dust_z(7, 6, 10);
    b.dust_x(8, 13, 8);
    b.output("Q", 8);

    let spec = b.spec(vec![1, 1, 1, 0], Constraints::default());
    let verdict = evaluate(&b.grid, &spec);
    println!("{:?}", b.grid);
    println!("NAND -> {verdict}");
    if !verdict.is_pass() {
        eprintln!("selftest FAILED");
        std::process::exit(1);
    }
    println!("selftest ok");
}
