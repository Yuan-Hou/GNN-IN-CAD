"""Clean disposable files and package the self-contained report, independent of Git."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

REPORT_DIR = Path(__file__).resolve().parents[1]
DISPOSABLE_DIRS = {'__pycache__', '.ipynb_checkpoints', '.pytest_cache'}
EXCLUDED_DIRS = DISPOSABLE_DIRS | {'.git', '.vscode', '.venv', 'env', 'node_modules'}


def cleanup(root=REPORT_DIR):
    """Remove only known disposable editor/Python caches; preserve data and runs."""
    removed = []
    for path in sorted(root.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_symlink():
            continue
        if path.is_dir() and path.name in DISPOSABLE_DIRS:
            shutil.rmtree(path)
            removed.append(str(path.relative_to(root)))
        elif path.is_file() and (path.name == '.DS_Store' or path.suffix == '.pyc'):
            path.unlink()
            removed.append(str(path.relative_to(root)))
    return removed


def submission_files(root=REPORT_DIR):
    completed = {}
    runs = root / 'data' / 'model_runs'
    for result_path in runs.glob('*/result.json'):
        run = result_path.parent
        result = json.loads(result_path.read_text())
        final = f"epoch_{result['final_epoch']:03d}"
        required = result.get('files', {f'{final}/{name}': None for name in ('checkpoint.pt', 'train_metrics.json', 'validation_predictions.pt')})
        for name in required:
            if not (run / name).is_file():
                raise FileNotFoundError(f'Completed model run is missing {run / name}')
        completed[run.relative_to(root)] = final
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f'Submission must contain real local files, not symlinks: {rel}')
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if rel.parts[:2] == ('data', 'MFCADDataset'):
            continue  # Include the original ZIP, not its extracted contents.
        if path.name.startswith('.') or path.suffix in {'.tmp', '.pyc', '.log', '.zip'}:
            if rel != Path('data/MFCADDataset.zip'):
                continue
        if rel.parts[0] == 'working' and path.suffix == '.html':
            continue  # Large standalone inspection exports are reproducible.
        if rel.parts[:2] == ('data', 'model_runs'):
            run_rel = Path(*rel.parts[:3])
            if run_rel not in completed:
                continue  # Never submit partial training runs.
            remainder = rel.parts[3:]
            if remainder and remainder[0].startswith('epoch_') and remainder[0] != completed[run_rel]:
                continue
        yield path


def package_submission(output=None, root=REPORT_DIR):
    output = Path(output or root.parent / 'dist' / 'report_submission.zip').resolve()
    if output.is_relative_to(root.resolve()):
        raise ValueError('Write the submission ZIP outside the report folder.')
    files = list(submission_files(root))
    relative = {p.relative_to(root).as_posix() for p in files}
    for required in ('main.ipynb', 'README.md', 'requirement.txt', 'data/MFCADDataset.zip', 'data/label_mapping.json', 'data/class_order.json'):
        if required not in relative:
            raise FileNotFoundError(f'Required submission file missing: {required}')
    # Catch missing notebook Markdown assets before publishing the ZIP.
    import re
    for path in files:
        if path.suffix != '.ipynb':
            continue
        notebook = json.loads(path.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] != 'markdown':
                continue
            for target in re.findall(r'\]\(([^)]+)\)', ''.join(cell['source'])):
                if '://' in target or target.startswith('#'):
                    continue
                destination = (path.parent / target.split('#')[0]).resolve()
                if not destination.is_relative_to(root.resolve()) or destination.relative_to(root.resolve()).as_posix() not in relative:
                    raise ValueError(f'Notebook link is outside the package or excluded: {path.name}: {target}')
    manifest = {'format_version': 1, 'files': {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            rel = path.relative_to(root).as_posix()
            data = path.read_bytes()
            archive.writestr('report/' + rel, data)
            manifest['files'][rel] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        archive.writestr('report/SUBMISSION_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'Corrupt ZIP member: {bad}')
    temporary.replace(output)
    print(f'Packaged {len(files)} files: {output} ({output.stat().st_size / 1024**2:.1f} MiB)')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', action='store_true', help='Remove disposable caches before packaging.')
    parser.add_argument('--output', type=Path, help='ZIP destination outside the report folder.')
    args = parser.parse_args()
    if args.clean:
        print(f'Removed {len(cleanup())} disposable files/directories.')
    package_submission(args.output)
