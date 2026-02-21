import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


STANDARD_AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
RCSB_IDEAL_SDF_URL = "https://files.rcsb.org/ligands/download/{ccd_id}_ideal.sdf"


def rewrite(s: str, rules: dict[str, str], steps: int) -> str:
    for _ in range(steps):
        s = "".join(rules[ch] for ch in s)
    return s

def rot_yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def rot_pitch(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def turtle_3d(
    commands: str,
    step: float = 1.0,
    yaw=np.deg2rad(30),
    pitch=np.deg2rad(30),
    max_points: int | None = None,
):
    """
    Turtle interpreter for commands in {"1","2","3"}.

    Forward step length is constant (`step`). Stops early at max_points.
    """
    pos = np.zeros(3)
    R = np.eye(3)
    forward = np.array([1.0, 0.0, 0.0])
    pts = [tuple(pos)]

    for ch in commands:
        if ch == "1":
            pos = pos + (R @ forward) * step
            pts.append(tuple(pos))
            if max_points is not None and len(pts) >= max_points:
                break
        elif ch == "2":
            R = R @ rot_yaw(yaw)
        elif ch == "3":
            R = R @ rot_pitch(pitch)
        else:
            continue

    return pts


def generate_random_genome(length: int, seed: int) -> str:
    if length <= 0:
        raise ValueError("length must be > 0")
    rng = np.random.default_rng(seed)
    return "".join(str(x) for x in rng.integers(1, 4, size=length))


def build_amino_programs(points_per_codon: int = 30) -> dict[str, str]:
    """
    Legacy random baseline (kept for comparison experiments).
    Build one deterministic "amino-acid program" for each codon (111..333).
    Each program emits exactly points_per_codon forward moves ("1").
    """
    if points_per_codon <= 0:
        raise ValueError("points_per_codon must be > 0")

    codons = [a + b + c for a in "123" for b in "123" for c in "123"]
    programs: dict[str, str] = {}

    for codon in codons:
        idx = ((int(codon[0]) - 1) * 9) + ((int(codon[1]) - 1) * 3) + (int(codon[2]) - 1)
        rng = np.random.default_rng(10_000 + idx)
        codon_digits = [int(d) for d in codon]

        cmd: list[str] = []
        for i in range(points_per_codon):
            cmd.append("1")
            motif = codon_digits[i % 3]
            pick = (idx + i + motif + int(rng.integers(0, 3))) % 6
            if pick in (0, 1):
                cmd.append("2")
            elif pick in (2, 3):
                cmd.append("3")
            elif pick == 4:
                cmd.extend(["2", "2"])
            else:
                cmd.extend(["3", "3"])

        programs[codon] = "".join(cmd)

    return programs


def all_codons() -> list[str]:
    return [a + b + c for a in "123" for b in "123" for c in "123"]


def parse_v2000_sdf_atoms(sdf_text: str) -> list[dict[str, float | str]]:
    lines = sdf_text.splitlines()
    if len(lines) < 4:
        raise ValueError("SDF text is too short")

    counts = lines[3]
    try:
        atom_count = int(counts[0:3])
    except ValueError:
        atom_count = int(counts.split()[0])

    atoms: list[dict[str, float | str]] = []
    start = 4
    end = start + atom_count
    if len(lines) < end:
        raise ValueError("SDF text does not contain declared atom block")

    for line in lines[start:end]:
        x = float(line[0:10])
        y = float(line[10:20])
        z = float(line[20:30])
        element = line[31:34].strip()
        atoms.append({"element": element, "x": x, "y": y, "z": z})
    return atoms


def fetch_rcsb_amino_acid_shapes(
    *,
    amino_acids: list[str] | None = None,
    include_hydrogens: bool = True,
    timeout: float = 20.0,
) -> dict[str, list[dict[str, float | str]]]:
    """
    Download idealized atom coordinates for amino acids from the RCSB ligand endpoint.
    """
    aa_codes = amino_acids if amino_acids is not None else STANDARD_AMINO_ACIDS
    shapes: dict[str, list[dict[str, float | str]]] = {}

    for aa in aa_codes:
        url = RCSB_IDEAL_SDF_URL.format(ccd_id=aa)
        try:
            with urlopen(url, timeout=timeout) as resp:
                text = resp.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError(f"Failed to download {aa} from {url}: {exc}") from exc

        atoms = parse_v2000_sdf_atoms(text)
        if not include_hydrogens:
            atoms = [a for a in atoms if str(a["element"]).upper() != "H"]
        if not atoms:
            raise RuntimeError(f"No atoms parsed for amino acid {aa}")
        shapes[aa] = atoms

    return shapes


def load_or_build_amino_shapes(
    *,
    cache_dir: str | Path = ".algogen_cache",
    include_hydrogens: bool = True,
    force_refresh: bool = False,
) -> tuple[dict[str, list[dict[str, float | str]]], Path, str]:
    """
    Load amino-acid shapes from local cache; fetch from RCSB if cache is absent/invalid.
    """
    cache_dir = Path(cache_dir)
    atom_mode = "all" if include_hydrogens else "heavy"
    cache_path = cache_dir / f"amino_acid_shapes_{atom_mode}_atoms.json"

    if not force_refresh and cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes")
            if isinstance(shapes, dict) and all(aa in shapes for aa in STANDARD_AMINO_ACIDS):
                return shapes, cache_path, "loaded"
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    shapes = fetch_rcsb_amino_acid_shapes(include_hydrogens=include_hydrogens)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "source": "RCSB Ligand ideal SDF",
        "source_url_template": RCSB_IDEAL_SDF_URL,
        "include_hydrogens": include_hydrogens,
        "shapes": shapes,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return shapes, cache_path, "fetched"


def nearest_neighbor_order(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    if n <= 2:
        return points

    remaining = set(range(1, n))
    order = [0]
    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda i: np.linalg.norm(points[i] - points[last]))
        order.append(nxt)
        remaining.remove(nxt)
    return points[np.array(order, dtype=int)]


def amino_atoms_to_turtle_commands(
    atoms: list[dict[str, float | str]],
    *,
    turn_angle_deg: float = 30.0,
) -> str:
    if turn_angle_deg <= 0 or turn_angle_deg >= 360:
        raise ValueError("turn_angle_deg must be in (0, 360)")

    pts = np.array([[float(a["x"]), float(a["y"]), float(a["z"])] for a in atoms], dtype=float)
    pts = pts - pts.mean(axis=0, keepdims=True)
    pts = nearest_neighbor_order(pts)

    turn_angle = np.deg2rad(turn_angle_deg)
    turns_per_rev = max(1, int(round((2.0 * np.pi) / turn_angle)))
    yaw_mats = [np.linalg.matrix_power(rot_yaw(turn_angle), k) for k in range(turns_per_rev)]
    pitch_mats = [np.linalg.matrix_power(rot_pitch(turn_angle), k) for k in range(turns_per_rev)]

    R = np.eye(3)
    forward = np.array([1.0, 0.0, 0.0])
    commands: list[str] = []

    for i in range(len(pts) - 1):
        vec = pts[i + 1] - pts[i]
        dist = float(np.linalg.norm(vec))
        if dist < 1e-9:
            continue
        target_dir = vec / dist

        best_yaw = 0
        best_pitch = 0
        best_score = -np.inf
        for yaw_steps in range(turns_per_rev):
            Ry = R @ yaw_mats[yaw_steps]
            for pitch_steps in range(turns_per_rev):
                cand_R = Ry @ pitch_mats[pitch_steps]
                score = float(np.dot(cand_R @ forward, target_dir))
                if score > best_score:
                    best_score = score
                    best_yaw = yaw_steps
                    best_pitch = pitch_steps

        if best_yaw:
            commands.append("2" * best_yaw)
        if best_pitch:
            commands.append("3" * best_pitch)
        R = R @ yaw_mats[best_yaw] @ pitch_mats[best_pitch]
        commands.append("1")

    return "".join(commands)


def build_real_amino_programs(
    amino_shapes: dict[str, list[dict[str, float | str]]],
    *,
    turn_angle_deg: float = 30.0,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Build codon programs from real amino-acid atom geometries.
    Returns:
      - codon -> turtle commands
      - codon -> amino-acid code
    """
    amino_commands: dict[str, str] = {}
    for aa in STANDARD_AMINO_ACIDS:
        atoms = amino_shapes[aa]
        amino_commands[aa] = amino_atoms_to_turtle_commands(atoms, turn_angle_deg=turn_angle_deg)

    codons = all_codons()
    codon_to_aa: dict[str, str] = {}
    codon_programs: dict[str, str] = {}
    for i, codon in enumerate(codons):
        aa = STANDARD_AMINO_ACIDS[i % len(STANDARD_AMINO_ACIDS)]
        codon_to_aa[codon] = aa
        codon_programs[codon] = amino_commands[aa]

    return codon_programs, codon_to_aa


def load_or_build_real_amino_programs(
    *,
    cache_dir: str | Path = ".algogen_cache",
    include_hydrogens: bool = True,
    turn_angle_deg: float = 30.0,
    force_rebuild: bool = False,
    force_refresh_shapes: bool = False,
) -> tuple[dict[str, str], dict[str, str], Path, str, Path, str]:
    """
    Cache-aware real amino-acid command builder based on RCSB ideal SDF coordinates.
    """
    cache_dir = Path(cache_dir)
    atom_mode = "all" if include_hydrogens else "heavy"
    tag = f"{atom_mode}_angle{turn_angle_deg:.2f}".replace(".", "p")
    prog_cache_path = cache_dir / f"amino_programs_real_{tag}.json"

    if not force_rebuild and prog_cache_path.exists():
        try:
            with prog_cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            programs = data.get("programs")
            codon_to_aa = data.get("codon_to_amino")
            if isinstance(programs, dict) and isinstance(codon_to_aa, dict):
                if all(c in programs for c in all_codons()):
                    shapes_path = cache_dir / f"amino_acid_shapes_{atom_mode}_atoms.json"
                    return programs, codon_to_aa, prog_cache_path, "loaded", shapes_path, "unknown"
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    amino_shapes, shapes_cache_path, shapes_status = load_or_build_amino_shapes(
        cache_dir=cache_dir,
        include_hydrogens=include_hydrogens,
        force_refresh=force_refresh_shapes,
    )
    programs, codon_to_aa = build_real_amino_programs(
        amino_shapes,
        turn_angle_deg=turn_angle_deg,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "source": "RCSB ideal SDF -> turtle approximation",
        "include_hydrogens": include_hydrogens,
        "turn_angle_deg": turn_angle_deg,
        "codon_to_amino": codon_to_aa,
        "programs": programs,
    }
    with prog_cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return programs, codon_to_aa, prog_cache_path, "built", shapes_cache_path, shapes_status


def load_or_build_amino_programs(
    points_per_codon: int = 30,
    *,
    cache_dir: str | Path = ".algogen_cache",
    force_rebuild: bool = False,
) -> tuple[dict[str, str], Path, str]:
    """
    Legacy random-baseline cache loader (not used in the real-shape pipeline).
    Load amino programs from disk cache if available, otherwise build and cache them.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"amino_programs_ppc{points_per_codon}.json"

    if not force_rebuild and cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            programs = data.get("programs")
            cached_ppc = data.get("points_per_codon")
            if isinstance(programs, dict) and cached_ppc == points_per_codon:
                return {str(k): str(v) for k, v in programs.items()}, cache_path, "loaded"
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    programs = build_amino_programs(points_per_codon=points_per_codon)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "points_per_codon": points_per_codon,
        "programs": programs,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return programs, cache_path, "built"


def expand_genome_to_commands(genome: str, amino_programs: dict[str, str]) -> tuple[str, int]:
    usable_len = len(genome) - (len(genome) % 3)
    codons = [genome[i:i + 3] for i in range(0, usable_len, 3)]
    expanded = "".join(amino_programs[codon] for codon in codons)
    return expanded, len(codons)


def sequence_to_jump_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    digest = hashlib.sha256(sequence.encode("ascii")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float((1 << 64) - 1)


def find_codon_periodic(genome: str, codon: str, start_index: int) -> int:
    n = len(genome)
    m = len(codon)
    if n == 0 or m == 0 or m > n:
        return -1

    start = start_index % n
    for k in range(n):
        idx = (start + k) % n
        if all(genome[(idx + j) % n] == codon[j] for j in range(m)):
            return idx
    return -1


def periodic_slice(genome: str, start_index: int, end_index: int) -> str:
    n = len(genome)
    if n == 0:
        return ""
    start = start_index % n
    end = end_index % n
    if start <= end:
        return genome[start:end]
    return genome[start:] + genome[:end]


def expand_genome_start_stop(
    genome: str,
    amino_programs: dict[str, str],
    *,
    start_codon: str = "111",
    stop_codon: str = "333",
    target_points: int = 10_000,
    max_genes: int = 10_000,
) -> tuple[str, list[dict[str, float | int]]]:
    """
    Periodically scan genome for START/STOP coding regions.
    Start with first START from index 0, then jump by a sequence-derived fraction.
    """
    if len(start_codon) != 3 or len(stop_codon) != 3:
        raise ValueError("start_codon and stop_codon must both have length 3")
    if target_points <= 1:
        raise ValueError("target_points must be > 1")
    if max_genes <= 0:
        raise ValueError("max_genes must be > 0")

    n = len(genome)
    if n < 3:
        return "", []

    commands: list[str] = []
    metadata: list[dict[str, float | int]] = []
    forward_count = 0

    search_index = find_codon_periodic(genome, start_codon, 0)
    if search_index == -1:
        return "", metadata

    for _ in range(max_genes):
        start_idx = find_codon_periodic(genome, start_codon, search_index)
        if start_idx == -1:
            break

        after_start = (start_idx + len(start_codon)) % n
        stop_idx = find_codon_periodic(genome, stop_codon, after_start)
        if stop_idx == -1:
            break

        coding_seq = periodic_slice(genome, after_start, stop_idx)
        usable_len = len(coding_seq) - (len(coding_seq) % 3)
        codon_count = usable_len // 3

        gene_commands = ""
        if codon_count > 0:
            codons = [coding_seq[i:i + 3] for i in range(0, usable_len, 3)]
            gene_commands = "".join(amino_programs[c] for c in codons)
            commands.append(gene_commands)
            forward_count += gene_commands.count("1")

        jump_fraction = sequence_to_jump_fraction(coding_seq)
        search_index = int(jump_fraction * n) % n
        metadata.append({
            "start_idx": start_idx,
            "stop_idx": stop_idx,
            "codons": codon_count,
            "jump_fraction": jump_fraction,
            "next_search_idx": search_index,
        })

        if forward_count >= target_points - 1:
            break

    return "".join(commands), metadata


def mutate_genome(genome: str, mutation_count: int, seed: int) -> tuple[str, list[dict[str, int | str]]]:
    """
    Apply `mutation_count` point mutations at unique positions.
    Each mutation flips one digit to one of the other two values in {"1","2","3"}.
    """
    if mutation_count < 0:
        raise ValueError("mutation_count must be >= 0")
    if mutation_count == 0 or not genome:
        return genome, []

    n = len(genome)
    actual_count = min(mutation_count, n)
    rng = np.random.default_rng(seed)
    indices = sorted(int(i) for i in rng.choice(n, size=actual_count, replace=False))

    chars = list(genome)
    mutations: list[dict[str, int | str]] = []
    for idx in indices:
        old = chars[idx]
        candidates = [d for d in "123" if d != old]
        new = candidates[int(rng.integers(0, len(candidates)))]
        chars[idx] = new
        mutations.append({"index": idx, "old": old, "new": new})

    return "".join(chars), mutations


def run_pipeline_for_genome(
    genome: str,
    amino_programs: dict[str, str],
    *,
    step: float,
    yaw: float,
    pitch: float,
    start_codon: str,
    stop_codon: str,
    target_points: int,
    max_genes: int,
) -> tuple[str, list[dict[str, float | int]], list[tuple[float, float, float]]]:
    expanded, genes = expand_genome_start_stop(
        genome,
        amino_programs,
        start_codon=start_codon,
        stop_codon=stop_codon,
        target_points=target_points,
        max_genes=max_genes,
    )
    points = turtle_3d(
        expanded,
        step=step,
        yaw=yaw,
        pitch=pitch,
        max_points=target_points,
    )
    return expanded, genes, points


def compare_runs(
    base_genes: list[dict[str, float | int]],
    mut_genes: list[dict[str, float | int]],
    base_points: list[tuple[float, float, float]],
    mut_points: list[tuple[float, float, float]],
) -> dict[str, float | int | None]:
    overlap_genes = min(len(base_genes), len(mut_genes))
    changed_genes = 0
    first_changed_gene: int | None = None

    for i in range(overlap_genes):
        a = base_genes[i]
        b = mut_genes[i]
        same = (
            a["start_idx"] == b["start_idx"]
            and a["stop_idx"] == b["stop_idx"]
            and a["codons"] == b["codons"]
            and a["next_search_idx"] == b["next_search_idx"]
            and abs(float(a["jump_fraction"]) - float(b["jump_fraction"])) < 1e-12
        )
        if not same:
            changed_genes += 1
            if first_changed_gene is None:
                first_changed_gene = i + 1

    a_pts = np.asarray(base_points, dtype=float)
    b_pts = np.asarray(mut_points, dtype=float)
    overlap_points = min(len(a_pts), len(b_pts))
    mean_displacement = 0.0
    max_displacement = 0.0
    if overlap_points > 0:
        d = np.linalg.norm(a_pts[:overlap_points] - b_pts[:overlap_points], axis=1)
        mean_displacement = float(d.mean())
        max_displacement = float(d.max())

    return {
        "base_genes": len(base_genes),
        "mutated_genes": len(mut_genes),
        "overlap_genes": overlap_genes,
        "changed_overlap_genes": changed_genes,
        "first_changed_gene": first_changed_gene,
        "base_points": len(base_points),
        "mutated_points": len(mut_points),
        "overlap_points": overlap_points,
        "mean_point_displacement": mean_displacement,
        "max_point_displacement": max_displacement,
    }


def sequential_spheres(points: np.ndarray, r1_mode: str = "half_next", eps: float = 0.0) -> np.ndarray:
    """
    Ordered greedy sphere growth:
      - Set r1 by r1_mode
      - For i>0: r[i] = max(0, min_j<i (dist(i,j) - r[j]) - eps)

    points: (N,3) float array
    r1_mode:
      - "half_next": r1 = 0.5 * ||p2 - p1|| (or 0 if N<2)
      - "zero": r1 = 0
      - "fixed:<value>": e.g. "fixed:1.25"
    eps: optional small safety gap so spheres don't *numerically* intersect
    """
    pts = np.asarray(points, dtype=float)
    N = pts.shape[0]
    r = np.zeros(N, dtype=float)
    if N == 0:
        return r

    # Initialize r1
    if r1_mode == "half_next":
        r[0] = 0.5 * np.linalg.norm(pts[1] - pts[0]) if N >= 2 else 0.0
    elif r1_mode == "zero":
        r[0] = 0.0
    elif r1_mode.startswith("fixed:"):
        r[0] = float(r1_mode.split(":", 1)[1])
    else:
        raise ValueError(f"Unknown r1_mode: {r1_mode}")

    # Sequentially grow
    for i in range(1, N):
        # Compute allowable radius against all previous spheres
        dif = pts[:i] - pts[i]
        dists = np.sqrt(np.sum(dif * dif, axis=1))
        limits = dists - r[:i]
        ri = np.min(limits) - eps
        r[i] = max(0.0, ri)

    return r

def show_spheres_pyvista(points, radii, *,
                         sphere_radius=1.0,
                         sphere_theta=16,
                         sphere_phi=16,
                         opacity=0.35,
                         show_centers=False,
                         centers_size=5.0,
                         color_by="order",   # "radius" or "order"
                         cmap="viridis",
                         notebook=False):
    """
    Fast GPU-ish visualization via glyphing: one sphere mesh, many instances.

    points: (N,3) array
    radii: (N,) array

    sphere_theta/phi control sphere tessellation (quality vs speed).
    """
    pts = np.asarray(points, dtype=float)
    r = np.asarray(radii, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if r.ndim != 1 or r.shape[0] != pts.shape[0]:
        raise ValueError("radii must be (N,) matching points")

    # Build a PolyData with per-point scalar = radius (for scaling)
    cloud = pv.PolyData(pts)
    cloud["scale"] = r

    # Single unit sphere, then glyph it
    base_sphere = pv.Sphere(radius=sphere_radius, theta_resolution=sphere_theta, phi_resolution=sphere_phi)
    spheres = cloud.glyph(geom=base_sphere, scale="scale", orient=False)

    pl = pv.Plotter(notebook=notebook)
    pl.set_background("white")

    # Coloring
    if color_by == "radius":
        spheres["val"] = np.repeat(r, base_sphere.n_points)  # scalar per glyph point
        pl.add_mesh(spheres, scalars="val", cmap=cmap, opacity=opacity, smooth_shading=True)
    elif color_by == "order":
        order = np.arange(len(r), dtype=float)
        spheres["val"] = np.repeat(order, base_sphere.n_points)
        pl.add_mesh(spheres, scalars="val", cmap=cmap, opacity=opacity, smooth_shading=True)
    else:
        pl.add_mesh(spheres, color="lightsteelblue", opacity=opacity, smooth_shading=True)

    # Optional centers (super cheap)
    if show_centers:
        pl.add_points(cloud, color="black", point_size=centers_size, render_points_as_spheres=True)

    pl.add_axes()
    pl.show_grid()
    pl.show()    


def show_single_amino_acid_pyvista(
    amino_acid: str,
    *,
    cache_dir: str | Path = ".algogen_cache",
    include_hydrogens: bool = True,
    turn_angle_deg: float = 30.0,
    step: float = 1.0,
    notebook: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Plot one amino acid as:
      - real atom coordinates (RCSB ideal SDF)
      - turtle-reconstructed polyline from the same amino acid

    Returns (atom_points_centered, turtle_points_aligned, command_string).
    """
    aa = amino_acid.upper()
    if aa not in STANDARD_AMINO_ACIDS:
        raise ValueError(f"Unknown amino acid '{amino_acid}'. Use one of: {STANDARD_AMINO_ACIDS}")

    shapes, _, _ = load_or_build_amino_shapes(
        cache_dir=cache_dir,
        include_hydrogens=include_hydrogens,
        force_refresh=False,
    )
    atoms = shapes[aa]
    atom_points = np.array([[float(a["x"]), float(a["y"]), float(a["z"])] for a in atoms], dtype=float)
    atom_points_centered = atom_points - atom_points.mean(axis=0, keepdims=True)

    commands = amino_atoms_to_turtle_commands(atoms, turn_angle_deg=turn_angle_deg)
    turtle_points = np.asarray(
        turtle_3d(
            commands,
            step=step,
            yaw=np.deg2rad(turn_angle_deg),
            pitch=np.deg2rad(turn_angle_deg),
        ),
        dtype=float,
    )
    turtle_points_centered = turtle_points - turtle_points.mean(axis=0, keepdims=True)

    atom_extent = np.max(np.linalg.norm(atom_points_centered, axis=1)) if len(atom_points_centered) else 0.0
    turtle_extent = np.max(np.linalg.norm(turtle_points_centered, axis=1)) if len(turtle_points_centered) else 0.0
    scale = (atom_extent / turtle_extent) if turtle_extent > 1e-12 else 1.0
    turtle_points_aligned = turtle_points_centered * scale

    pl = pv.Plotter(notebook=notebook)
    pl.set_background("white")

    element_colors = {
        "C": "dimgray",
        "N": "royalblue",
        "O": "tomato",
        "S": "goldenrod",
        "H": "lightgray",
    }

    elements = np.array([str(a["element"]).upper() for a in atoms], dtype=object)
    for elem in sorted(set(elements.tolist())):
        idx = np.where(elements == elem)[0]
        if idx.size == 0:
            continue
        cloud = pv.PolyData(atom_points_centered[idx])
        pl.add_points(
            cloud,
            color=element_colors.get(elem, "seagreen"),
            point_size=12.0 if elem != "H" else 8.0,
            render_points_as_spheres=True,
            label=f"{elem} atoms",
        )

    if len(turtle_points_aligned) >= 2:
        path = pv.lines_from_points(turtle_points_aligned, close=False)
        pl.add_mesh(path, color="crimson", line_width=4, label="Turtle path")
    pl.add_points(
        turtle_points_aligned,
        color="crimson",
        point_size=6,
        render_points_as_spheres=True,
        opacity=0.6,
    )

    pl.add_axes()
    pl.show_grid()
    pl.add_legend()
    pl.add_title(
        f"{aa} | atoms={len(atom_points_centered)} | turtle_pts={len(turtle_points_aligned)} | angle={turn_angle_deg}deg"
    )
    pl.show()
    return atom_points_centered, turtle_points_aligned, commands

# Example usage:
# points = np.array([...])  # (N,3)
# radii = sequential_spheres(points, r1_mode="half_next", eps=1e-9)

genome_seed = 10
genome_length = 100000  # choose any length; use multiples of 3 to consume full genome
step = 4.0
turn_angle_deg = 30.0
include_hydrogens = True
start_codon = "111"
stop_codon = "333"
target_points = 50000
max_genes = 1000
amino_cache_dir = ".algogen_cache"
force_refresh_amino_shapes = False
force_rebuild_amino_program_cache = False
run_mutation_experiment = False
mutation_count = 5
mutation_seed = 42
preview_amino_acid: str | None = None  # e.g. "TRP"

genome = generate_random_genome(genome_length, genome_seed)
(
    amino_programs,
    codon_to_amino,
    amino_program_cache_path,
    amino_program_cache_status,
    amino_shape_cache_path,
    amino_shape_cache_status,
) = load_or_build_real_amino_programs(
    cache_dir=amino_cache_dir,
    include_hydrogens=include_hydrogens,
    turn_angle_deg=turn_angle_deg,
    force_rebuild=force_rebuild_amino_program_cache,
    force_refresh_shapes=force_refresh_amino_shapes,
)
yaw = np.deg2rad(turn_angle_deg)
pitch = np.deg2rad(turn_angle_deg)
expanded, genes, points = run_pipeline_for_genome(
    genome,
    amino_programs,
    step=step,
    yaw=yaw,
    pitch=pitch,
    start_codon=start_codon,
    stop_codon=stop_codon,
    target_points=target_points,
    max_genes=max_genes,
)
print(
    f"seed={genome_seed}, genome_len={genome_length}, genes={len(genes)}, "
    f"start={start_codon}, stop={stop_codon}, step={step}, "
    f"turn_angle_deg={turn_angle_deg}, points={len(points)}"
)
print(
    f"amino_shape_cache={amino_shape_cache_status} path={amino_shape_cache_path}"
)
print(
    f"amino_program_cache={amino_program_cache_status} path={amino_program_cache_path}"
)
print(
    "codon_mapping_sample="
    + ", ".join(f"{c}->{codon_to_amino[c]}" for c in all_codons()[:9])
)
if genes:
    for i, gene in enumerate(genes, start=1):
        print(
            f"gene_{i}: start={gene['start_idx']}, stop={gene['stop_idx']}, "
            f"codons={gene['codons']}, jump={gene['jump_fraction']:.4f}, "
            f"next_search_idx={gene['next_search_idx']}"
        )
else:
    print("No START/STOP genes found.")

if run_mutation_experiment:
    mutated_genome, mutations = mutate_genome(
        genome, mutation_count=mutation_count, seed=mutation_seed
    )
    _, mut_genes, mut_points = run_pipeline_for_genome(
        mutated_genome,
        amino_programs,
        step=step,
        yaw=yaw,
        pitch=pitch,
        start_codon=start_codon,
        stop_codon=stop_codon,
        target_points=target_points,
        max_genes=max_genes,
    )
    cmp_stats = compare_runs(genes, mut_genes, points, mut_points)
    print(
        f"mutation_experiment: count={len(mutations)} seed={mutation_seed} "
        f"mutated_points={len(mut_points)} mutated_genes={len(mut_genes)}"
    )
    for m in mutations:
        print(f"mutation: index={m['index']} old={m['old']} new={m['new']}")
    print(
        f"mutation_effect: overlap_genes={cmp_stats['overlap_genes']} "
        f"changed_overlap_genes={cmp_stats['changed_overlap_genes']} "
        f"first_changed_gene={cmp_stats['first_changed_gene']}"
    )
    print(
        f"mutation_effect_points: overlap={cmp_stats['overlap_points']} "
        f"mean_displacement={cmp_stats['mean_point_displacement']:.6f} "
        f"max_displacement={cmp_stats['max_point_displacement']:.6f}"
    )

if preview_amino_acid:
    show_single_amino_acid_pyvista(
        preview_amino_acid,
        cache_dir=amino_cache_dir,
        include_hydrogens=include_hydrogens,
        turn_angle_deg=turn_angle_deg,
        step=step,
    )

show_spheres_pyvista(points, 2*np.ones(len(points)), sphere_theta=16, sphere_phi=16, opacity=1)    
