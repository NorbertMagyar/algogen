import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import hashlib


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
    yaw=np.deg2rad(20),
    pitch=np.deg2rad(15),
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

def plot_spheres_matplotlib(points, radii, 
                           show_centers=True,
                           sphere_resolution=16,
                           alpha=0.35,
                           cmap="viridis"):
    """
    Visualize spheres with matplotlib 3D.

    points: (N,3) array
    radii: (N,) array
    sphere_resolution: sphere mesh density (8–24 is reasonable)
    alpha: sphere transparency
    cmap: color map by radius
    """

    pts = np.asarray(points)
    r = np.asarray(radii)
    N = len(pts)

    # Setup figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Normalize colors by radius
    norm = (r - r.min()) / (r.max() - r.min() + 1e-12)
    colors = plt.get_cmap(cmap)(norm)

    # Precompute sphere mesh
    u = np.linspace(0, 2*np.pi, sphere_resolution)
    v = np.linspace(0, np.pi, sphere_resolution)
    uu, vv = np.meshgrid(u, v)

    for i in range(N):
        if r[i] <= 0:
            continue

        x0, y0, z0 = pts[i]

        xs = x0 + r[i] * np.cos(uu) * np.sin(vv)
        ys = y0 + r[i] * np.sin(uu) * np.sin(vv)
        zs = z0 + r[i] * np.cos(vv)

        ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.4)


    # Optional centers
    if show_centers:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c="black", s=5, alpha=0.6)

    # Equal aspect ratio (important!)
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max()
    center = pts.mean(axis=0)

    ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
    ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
    ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)

    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Sequential Sphere Growth")
    plt.tight_layout()
    plt.show()

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

# Example usage:
# points = np.array([...])  # (N,3)
# radii = sequential_spheres(points, r1_mode="half_next", eps=1e-9)

genome_seed = 40
genome_length = 100000  # choose any length; use multiples of 3 to consume full genome
points_per_codon = 30
step = 6.0
start_codon = "111"
stop_codon = "333"
target_points = 50_000
max_genes = 1000

genome = generate_random_genome(genome_length, genome_seed)
amino_programs = build_amino_programs(points_per_codon=points_per_codon)
expanded, genes = expand_genome_start_stop(
    genome,
    amino_programs,
    start_codon=start_codon,
    stop_codon=stop_codon,
    target_points=target_points,
    max_genes=max_genes,
)
points = turtle_3d(expanded, step=step, max_points=target_points)
print(
    f"seed={genome_seed}, genome_len={genome_length}, genes={len(genes)}, "
    f"start={start_codon}, stop={stop_codon}, step={step}, points={len(points)}"
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
# plot_spheres_matplotlib(points, radii,
#                         sphere_resolution=14,
#                         alpha=0.4)
show_spheres_pyvista(points, 2*np.ones(len(points)), sphere_theta=12, sphere_phi=12, opacity=0.2)    
