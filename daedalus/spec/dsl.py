"""The specification DSL: lexer, parser and evaluator.

Grammar (§04)::

    spec  := inputs ident+ outputs ident+ constraint* rule+
    rule  := ident = expr
    expr  := ident | ¬ expr | expr (∧|∨|⊕) expr | ( expr )
           | delay( expr , int )
           | strength( expr , int )
    constraint := latency <= int
           | footprint <= int
           | region <= int x int

Every operator has an ASCII alias, because a DSL you cannot type on a keyboard
is a DSL nobody writes tests in::

    ¬  !  ~  not        ∧  &  and        ∨  |  or        ⊕  ^  xor

Precedence, tightest first: ``¬`` then ``∧`` then ``⊕`` then ``∨``. All binary
operators associate to the left.

Two constructs parse but do not affect the v1 truth table:

``delay(e, n)``
    an annotation that the output should arrive no sooner than ``n`` redstone
    ticks. The verifier measures propagation delay but has no notion of a
    *minimum*, so v1 records the request and ignores it.
``strength(e, n)``
    comparator signal-strength work, which is v2. It evaluates as ``e``.

Both are kept in the grammar rather than deferred so that corpora written now
stay parseable when v2 lands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_INPUTS = 6
MAX_OUTPUTS = 6


class SpecSyntaxError(ValueError):
    """The source text is not a well-formed spec."""

    def __init__(self, message: str, line: int = 0, column: int = 0):
        super().__init__(f"line {line}, col {column}: {message}" if line else message)
        self.message = message
        self.line = line
        self.column = column


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Ref:
    name: str


@dataclass(frozen=True, slots=True)
class Not:
    operand: object


@dataclass(frozen=True, slots=True)
class Binary:
    op: str  # "and" | "or" | "xor"
    left: object
    right: object


@dataclass(frozen=True, slots=True)
class Delay:
    """Timing annotation. Transparent to the truth table."""

    operand: object
    ticks: int


@dataclass(frozen=True, slots=True)
class Strength:
    """Signal-strength annotation. Transparent to the v1 truth table."""

    operand: object
    level: int


Expr = Ref | Not | Binary | Delay | Strength


def evaluate_expr(expr: Expr, env: dict[str, bool]) -> bool:
    """Evaluate an expression under a boolean assignment."""
    if isinstance(expr, Ref):
        try:
            return env[expr.name]
        except KeyError:
            raise SpecSyntaxError(f"unknown identifier {expr.name!r}") from None
    if isinstance(expr, Not):
        return not evaluate_expr(expr.operand, env)
    if isinstance(expr, Binary):
        a = evaluate_expr(expr.left, env)
        b = evaluate_expr(expr.right, env)
        if expr.op == "and":
            return a and b
        if expr.op == "or":
            return a or b
        return a != b
    if isinstance(expr, (Delay, Strength)):
        return evaluate_expr(expr.operand, env)
    raise AssertionError(f"unreachable expression node {expr!r}")


def format_expr(expr: Expr, ascii_only: bool = False) -> str:
    """Render an expression back to source, fully parenthesised."""
    ops = (
        {"and": "&", "or": "|", "xor": "^"}
        if ascii_only
        else {"and": "∧", "or": "∨", "xor": "⊕"}
    )
    neg = "!" if ascii_only else "¬"
    if isinstance(expr, Ref):
        return expr.name
    if isinstance(expr, Not):
        inner = format_expr(expr.operand, ascii_only)
        return f"{neg}{inner}" if isinstance(expr.operand, Ref) else f"{neg}({inner})"
    if isinstance(expr, Binary):
        left = format_expr(expr.left, ascii_only)
        right = format_expr(expr.right, ascii_only)
        return f"({left} {ops[expr.op]} {right})"
    if isinstance(expr, Delay):
        return f"delay({format_expr(expr.operand, ascii_only)}, {expr.ticks})"
    if isinstance(expr, Strength):
        return f"strength({format_expr(expr.operand, ascii_only)}, {expr.level})"
    raise AssertionError(f"unreachable expression node {expr!r}")


def gate_count(expr: Expr) -> int:
    """Number of logic operators in an expression.

    This is the difficulty axis the corpus is stratified on, so it counts what
    a builder would count: operators, not identifiers or annotations.
    """
    if isinstance(expr, Ref):
        return 0
    if isinstance(expr, Not):
        return 1 + gate_count(expr.operand)
    if isinstance(expr, Binary):
        return 1 + gate_count(expr.left) + gate_count(expr.right)
    if isinstance(expr, (Delay, Strength)):
        return gate_count(expr.operand)
    raise AssertionError(f"unreachable expression node {expr!r}")


def referenced(expr: Expr) -> set[str]:
    if isinstance(expr, Ref):
        return {expr.name}
    if isinstance(expr, Not):
        return referenced(expr.operand)
    if isinstance(expr, Binary):
        return referenced(expr.left) | referenced(expr.right)
    if isinstance(expr, (Delay, Strength)):
        return referenced(expr.operand)
    raise AssertionError(f"unreachable expression node {expr!r}")


# --------------------------------------------------------------------------
# constraints
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Constraints:
    """Hard limits. A candidate that violates one is not a pass, however
    correct its truth table."""

    max_latency_rt: int | None = None
    max_blocks: int | None = None
    max_region: tuple[int, int] | None = None

    def describe(self) -> list[str]:
        out = []
        if self.max_latency_rt is not None:
            out.append(f"latency <= {self.max_latency_rt}")
        if self.max_blocks is not None:
            out.append(f"footprint <= {self.max_blocks}")
        if self.max_region is not None:
            out.append(f"region <= {self.max_region[0]} x {self.max_region[1]}")
        return out

    def unsatisfiable(self, spec=None) -> str | None:
        """Why no layout could ever meet these, if that is the case.

        Worth asking before building anything. A budget nothing can satisfy
        otherwise consumes the whole retry allowance and reports the same
        thing a merely unlucky run does, so the answer looks like bad luck
        rather than a spec that cannot be built.

        Pass ``spec`` for the checks that depend on how many ports there are.
        The footprint floor is *not* the substrate: `material_blocks` counts
        from ``y = 1`` up, so the floor a layout stands on is free.
        """
        from .. import vocab as V

        if self.max_region is not None:
            width, depth = self.max_region
            # Levers sit on the input face and lamps on the output face, so
            # every layout spans the full width of the build volume. A width
            # budget under that is not tight, it is impossible.
            if width < V.SX:
                return (
                    f"region width {width} is below {V.SX}: ports are pinned to "
                    f"x=0 and x={V.SX - 1}, so every layout spans the full width"
                )
            if depth < 1:
                return f"region depth {depth} leaves no room for a circuit"
        if self.max_latency_rt is not None and self.max_latency_rt < 0:
            return "latency budget is negative"
        if self.max_blocks is not None:
            if self.max_blocks < 1:
                return f"footprint {self.max_blocks} leaves no room for a circuit"
            if spec is not None:
                # Every input needs a lever and the block it hangs on; every
                # output needs a lamp and the repeater driving it. That is the
                # floor before a single wire is routed.
                floor = 2 * spec.n_inputs + 2 * spec.n_outputs
                if self.max_blocks < floor:
                    return (
                        f"footprint {self.max_blocks} is below the {floor} blocks "
                        f"{spec.n_inputs} input and {spec.n_outputs} output ports "
                        "need before any wiring"
                    )
        return None


# --------------------------------------------------------------------------
# lexer
# --------------------------------------------------------------------------

_UNICODE_OPS = {"¬": "!", "∧": "&", "∨": "|", "⊕": "^", "→": "->"}

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>[ \t]+)
  | (?P<comment>\#[^\n]*)
  | (?P<newline>\n)
  | (?P<le><=)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<int>\d+)
  | (?P<punct>[()=,!~&|^])
    """,
    re.VERBOSE,
)

#: Words the grammar reserves. Using one as a port name is a mistake worth
#: catching at parse time rather than as a baffling truth table later.
#:
#: The ``x`` in ``region <= 8 x 8`` is deliberately *not* here. It is only a
#: separator in that one position, and reserving it globally would outlaw the
#: single most natural name for a boolean input.
KEYWORDS = frozenset(
    {
        "inputs",
        "outputs",
        "latency",
        "footprint",
        "region",
        "delay",
        "strength",
        "not",
        "and",
        "or",
        "xor",
    }
)


@dataclass(frozen=True, slots=True)
class Tok:
    kind: str
    text: str
    line: int
    col: int


def lex(source: str) -> list[Tok]:
    for uni, ascii_ in _UNICODE_OPS.items():
        source = source.replace(uni, ascii_)
    toks: list[Tok] = []
    line, col_base = 1, 0
    pos = 0
    while pos < len(source):
        m = _TOKEN_RE.match(source, pos)
        if not m:
            raise SpecSyntaxError(
                f"unexpected character {source[pos]!r}", line, pos - col_base + 1
            )
        kind = m.lastgroup
        text = m.group()
        col = pos - col_base + 1
        pos = m.end()
        if kind == "newline":
            toks.append(Tok("newline", "\n", line, col))
            line += 1
            col_base = pos
        elif kind in ("ws", "comment"):
            continue
        elif kind == "le":
            toks.append(Tok("le", text, line, col))
        elif kind == "ident":
            toks.append(Tok("ident", text, line, col))
        elif kind == "int":
            toks.append(Tok("int", text, line, col))
        else:
            toks.append(Tok(text, text, line, col))
    toks.append(Tok("eof", "", line, 1))
    return toks


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ParsedSpec:
    """The direct result of parsing, before canonicalisation."""

    inputs: list[str]
    outputs: list[str]
    rules: dict[str, Expr]
    constraints: Constraints = field(default_factory=Constraints)

    def gate_count(self) -> int:
        return sum(gate_count(e) for e in self.rules.values())


class _Parser:
    def __init__(self, toks: list[Tok]):
        self.toks = toks
        self.i = 0

    # -- token helpers -----------------------------------------------------

    def peek(self) -> Tok:
        return self.toks[self.i]

    def next(self) -> Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def skip_newlines(self) -> None:
        while self.peek().kind == "newline":
            self.i += 1

    def at_word(self, word: str) -> bool:
        t = self.peek()
        return t.kind == "ident" and t.text == word

    def expect(self, kind: str, what: str) -> Tok:
        t = self.peek()
        if t.kind != kind:
            raise SpecSyntaxError(f"expected {what}, found {t.text or 'end of input'!r}", t.line, t.col)
        return self.next()

    # -- productions -------------------------------------------------------

    def parse(self) -> ParsedSpec:
        self.skip_newlines()
        inputs = self.parse_name_list("inputs")
        self.skip_newlines()
        outputs = self.parse_name_list("outputs")

        constraints = Constraints()
        rules: dict[str, Expr] = {}
        while True:
            self.skip_newlines()
            t = self.peek()
            if t.kind == "eof":
                break
            if t.kind == "ident" and t.text in ("latency", "footprint", "region"):
                constraints = self.parse_constraint(constraints)
            elif t.kind == "ident":
                name, expr = self.parse_rule()
                if name in rules:
                    raise SpecSyntaxError(f"output {name!r} is assigned twice", t.line, t.col)
                rules[name] = expr
            else:
                raise SpecSyntaxError(f"unexpected {t.text!r}", t.line, t.col)

        return _validate(ParsedSpec(inputs, outputs, rules, constraints))

    def parse_name_list(self, keyword: str) -> list[str]:
        t = self.peek()
        if not self.at_word(keyword):
            raise SpecSyntaxError(f"expected {keyword!r} declaration", t.line, t.col)
        self.next()
        names: list[str] = []
        while self.peek().kind == "ident" and self.peek().text not in KEYWORDS:
            tok = self.next()
            if tok.text in names:
                raise SpecSyntaxError(f"duplicate {keyword[:-1]} {tok.text!r}", tok.line, tok.col)
            names.append(tok.text)
        if not names:
            raise SpecSyntaxError(f"{keyword!r} declares no names", t.line, t.col)
        return names

    def parse_constraint(self, current: Constraints) -> Constraints:
        kw = self.next()
        self.expect("le", "'<='")
        first = int(self.expect("int", "an integer").text)
        if kw.text == "latency":
            return Constraints(first, current.max_blocks, current.max_region)
        if kw.text == "footprint":
            return Constraints(current.max_latency_rt, first, current.max_region)
        # region <= W x D
        if not self.at_word("x"):
            t = self.peek()
            raise SpecSyntaxError("expected 'x' between region extents", t.line, t.col)
        self.next()
        second = int(self.expect("int", "an integer").text)
        return Constraints(current.max_latency_rt, current.max_blocks, (first, second))

    def parse_rule(self) -> tuple[str, Expr]:
        name = self.expect("ident", "an output name").text
        self.expect("=", "'='")
        return name, self.parse_or()

    # Precedence chain: or < xor < and < unary.
    def parse_or(self) -> Expr:
        left = self.parse_xor()
        while self.peek().kind == "|" or self.at_word("or"):
            self.next()
            left = Binary("or", left, self.parse_xor())
        return left

    def parse_xor(self) -> Expr:
        left = self.parse_and()
        while self.peek().kind == "^" or self.at_word("xor"):
            self.next()
            left = Binary("xor", left, self.parse_and())
        return left

    def parse_and(self) -> Expr:
        left = self.parse_unary()
        while self.peek().kind == "&" or self.at_word("and"):
            self.next()
            left = Binary("and", left, self.parse_unary())
        return left

    def parse_unary(self) -> Expr:
        t = self.peek()
        if t.kind in ("!", "~") or self.at_word("not"):
            self.next()
            return Not(self.parse_unary())
        return self.parse_atom()

    def parse_atom(self) -> Expr:
        t = self.next()
        if t.kind == "(":
            inner = self.parse_or()
            self.expect(")", "')'")
            return inner
        if t.kind == "ident":
            if t.text in ("delay", "strength"):
                self.expect("(", "'(' after " + t.text)
                inner = self.parse_or()
                self.expect(",", "',' in " + t.text)
                n = int(self.expect("int", "an integer").text)
                self.expect(")", "')'")
                return Delay(inner, n) if t.text == "delay" else Strength(inner, n)
            if t.text in KEYWORDS:
                raise SpecSyntaxError(f"{t.text!r} is a reserved word", t.line, t.col)
            return Ref(t.text)
        raise SpecSyntaxError(f"expected an expression, found {t.text or 'end of input'!r}", t.line, t.col)


def _validate(spec: ParsedSpec) -> ParsedSpec:
    if not spec.rules:
        raise SpecSyntaxError("a spec needs at least one rule")
    if len(spec.inputs) > MAX_INPUTS:
        raise SpecSyntaxError(f"{len(spec.inputs)} inputs exceeds the v1 limit of {MAX_INPUTS}")
    if len(spec.outputs) > MAX_OUTPUTS:
        raise SpecSyntaxError(f"{len(spec.outputs)} outputs exceeds the v1 limit of {MAX_OUTPUTS}")

    overlap = set(spec.inputs) & set(spec.outputs)
    if overlap:
        raise SpecSyntaxError(f"{sorted(overlap)} is declared as both an input and an output")

    missing = [o for o in spec.outputs if o not in spec.rules]
    if missing:
        raise SpecSyntaxError(f"no rule assigns output(s) {missing}")
    extra = [name for name in spec.rules if name not in spec.outputs]
    if extra:
        raise SpecSyntaxError(f"rule(s) {extra} assign names that are not declared outputs")

    known = set(spec.inputs)
    for name, expr in spec.rules.items():
        unknown = sorted(referenced(expr) - known)
        if unknown:
            raise SpecSyntaxError(f"rule for {name!r} references undeclared name(s) {unknown}")

    # An input nothing depends on makes the truth table twice as large for no
    # reason, and gives the placer a port it has to route nowhere.
    used: set[str] = set()
    for expr in spec.rules.values():
        used |= referenced(expr)
    unused = [i for i in spec.inputs if i not in used]
    if unused:
        raise SpecSyntaxError(f"input(s) {unused} are declared but never used")
    return spec


def parse(source: str) -> ParsedSpec:
    """Parse spec source text. Raises :class:`SpecSyntaxError` on any problem."""
    return _Parser(lex(source)).parse()
