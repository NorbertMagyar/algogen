# algogen

`algogen` generates 3D turtle trajectories from a synthetic genome made of digits `1`, `2`, and `3`.

The current workflow is:
- Generate a reproducible random genome from a fixed `seed` and `length`.
- Find coding regions between `START` and `STOP` codons (defaults: `111` and `333`).
- Translate each codon in the coding region into a longer command sequence ("amino program").
- Convert command sequences into 3D points with a turtle interpreter.
- Jump to the next search location using a deterministic `0..1` value derived from the prior coding sequence.
- Repeat periodically over the genome until the target number of points is reached.

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`
- `pyvista`

Install dependencies:

```bash
pip install numpy matplotlib pyvista
```

## Run

From the repository root:

```bash
python algogen.py
```

If your system uses `python3`:

```bash
python3 algogen.py
```

## Main Parameters

Edit these values near the bottom of `algogen.py`:

- `genome_seed`: random seed for reproducible genomes
- `genome_length`: total genome size
- `points_per_codon`: number of forward moves generated per translated codon
- `step`: fixed turtle forward step length
- `start_codon`: coding region start marker (default `111`)
- `stop_codon`: coding region stop marker (default `333`)
- `target_points`: generation cap for total points
- `max_genes`: safety cap for number of genes processed

## Output

The script prints:
- a run summary (seed, genome length, number of genes, point count)
- per-gene jump details (`gene_1`, `gene_2`, ...)

It then opens a PyVista window to visualize generated spheres/points.
