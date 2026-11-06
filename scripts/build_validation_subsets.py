"""Build fixed, disjoint validation subsets without loading the full dataset.

Run with the starter pack's Python environment. Output Parquet files are local
data; the JSON manifest is intended for version control. Existing outputs are
never overwritten. Use --verify to check existing files against the source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pyarrow.parquet as pq

sys.dont_write_bytecode = True


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate(table):
    if table.num_rows != 20_000:
        raise ValueError('Expected one complete 20,000-row sequence')
    ids = table['seq_ix'].to_numpy()
    if not np.all(ids == ids[0]):
        raise ValueError('Multiple sequence IDs in one row group')
    if not np.array_equal(table['step_in_seq'].to_numpy(), np.arange(20_000)):
        raise ValueError('Sequence steps are not 0..19,999')
    need = table['need_prediction'].to_numpy().astype(bool)
    if not np.array_equal(need, np.arange(20_000) >= 99):
        raise ValueError('Invalid warm-up flags')
    scored = table['is_scored'].to_numpy().astype(bool)
    if scored[:99].any():
        raise ValueError('Scoring mask selects warm-up rows')
    targets = np.column_stack([table[c].to_numpy() for c in ('t0', 't1')])
    if not np.isfinite(targets).all():
        raise ValueError('Nonfinite targets')
    weights = np.abs(np.clip(targets[need & scored], -2, 2)).astype(np.float64).sum(axis=0)
    return int(ids[0]), int((need & scored).sum()), weights


def verify(root, manifest):
    source_path = root / manifest['source']['path']
    if source_path.stat().st_size != manifest['source']['bytes'] or sha256(source_path) != manifest['source']['sha256']:
        raise ValueError('Source file differs from the recorded source')
    source = pq.ParquetFile(source_path)
    all_ids = set()
    all_groups = set()
    for name, entry in manifest['subsets'].items():
        path = root / entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError(f'{name}: subset checksum differs')
        subset = pq.ParquetFile(path)
        if not subset.schema_arrow.equals(source.schema_arrow, check_metadata=True):
            raise ValueError(f'{name}: schema differs')
        if subset.num_row_groups != entry['sequences'] or subset.metadata.num_rows != entry['rows']:
            raise ValueError(f'{name}: incorrect dimensions')
        scored_total = 0
        weight_total = np.zeros(2, np.float64)
        for index, record in enumerate(entry['selection']):
            original = record['source_row_group']
            table = subset.read_row_group(index)
            seq, scored, weights = validate(table)
            if seq != record['seq_ix'] or seq in all_ids or original in all_groups:
                raise ValueError(f'{name}: identity mismatch or overlap')
            if not table.equals(source.read_row_group(original)):
                raise ValueError(f'{name}: rows differ from source')
            all_ids.add(seq)
            all_groups.add(original)
            scored_total += scored
            weight_total += weights
        if scored_total != entry['scored_rows'] or not np.allclose(weight_total, entry['target_weight_sums'], rtol=0, atol=1e-8):
            raise ValueError(f'{name}: statistics differ')
        print(f'Verified {name}: {entry["sequences"]} sequences, {entry["rows"]:,} rows, {scored_total:,} scored rows', flush=True)
    print('Verified: exact source rows, schemas, sequence order, warm-up, masks, hashes, and no overlap.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--dev-sequences', type=int, default=64)
    parser.add_argument('--confirmation-sequences', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / 'experiments/splits/validation_v1.json'
    if args.verify:
        verify(root, json.loads(manifest_path.read_text()))
        return
    if manifest_path.exists():
        raise FileExistsError(f'{manifest_path} exists; use --verify. Do not silently change fixed splits.')
    source_path = root / 'datasets/valid.parquet'
    source = pq.ParquetFile(source_path)
    if min(args.dev_sequences, args.confirmation_sequences) < 1:
        raise ValueError('Each subset must have at least one sequence')
    total = args.dev_sequences + args.confirmation_sequences
    if total >= source.num_row_groups:
        raise ValueError('Leave validation sequences outside both subsets')
    groups = list(range(source.num_row_groups))
    random.Random(args.seed).shuffle(groups)
    selections = {
        'development': sorted(groups[:args.dev_sequences]),
        'confirmation': sorted(groups[args.dev_sequences:total]),
    }
    paths = {
        'development': root / f'datasets/subsets/valid_dev{args.dev_sequences}.parquet',
        'confirmation': root / f'datasets/subsets/valid_confirm{args.confirmation_sequences}.parquet',
    }
    for path in paths.values():
        if path.exists() or path.with_suffix('.parquet.partial').exists():
            raise FileExistsError(path)
    print('Hashing the original validation file once...', flush=True)
    manifest = {
        'version': 1,
        'seed': args.seed,
        'selection_method': 'Python random.Random(seed).shuffle of source row-group indices; disjoint slices; each output sorted by source row group',
        'source': {'path': 'datasets/valid.parquet', 'bytes': source_path.stat().st_size,
                   'sha256': sha256(source_path), 'rows': source.metadata.num_rows,
                   'sequences': source.num_row_groups},
        'unselected_source_row_groups': sorted(groups[total:]),
        'subsets': {},
    }
    seen_ids = set()
    for name, selected in selections.items():
        output = paths[name]
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.parquet.partial')
        records = []
        scored_total = 0
        weight_total = np.zeros(2, np.float64)
        print(f'Writing {name}: {len(selected)} complete sequences...', flush=True)
        with pq.ParquetWriter(temporary, source.schema_arrow, compression='snappy') as writer:
            for original in selected:
                table = source.read_row_group(original)
                seq, scored, weights = validate(table)
                if seq in seen_ids:
                    raise ValueError('Duplicate sequence ID across selected row groups')
                seen_ids.add(seq)
                writer.write_table(table, row_group_size=20_000)
                records.append({'source_row_group': original, 'seq_ix': seq})
                scored_total += scored
                weight_total += weights
        temporary.replace(output)
        manifest['subsets'][name] = {
            'path': output.relative_to(root).as_posix(), 'sequences': len(selected),
            'rows': len(selected) * 20_000, 'required_prediction_rows': len(selected) * 19_901,
            'scored_rows': scored_total, 'target_weight_sums': weight_total.tolist(),
            'bytes': output.stat().st_size, 'sha256': sha256(output), 'selection': records,
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(f'Manifest: {manifest_path}', flush=True)
    verify(root, manifest)


if __name__ == '__main__':
    main()
