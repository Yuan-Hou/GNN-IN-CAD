"""Build a teammate ZIP without datasets, checkpoints, caches or Git history."""

import json
from pathlib import Path
import zipfile


root = Path(__file__).resolve().parents[1]
output = root / "dist" / "project-vm.zip"
output.parent.mkdir(exist_ok=True)
paths = [root / name for name in (
    "README.md", "VM_SETUP.md", "LICENSE", "pyproject.toml", "requirements.txt",
    "environment-vm.yml", "environment-occwl.yml", "project_plan_contract_groupID07.pdf",
    "data/download_mfcad.sh",
)]
paths += list((root / "src").rglob("*.py"))
paths += list((root / "scripts").glob("*.py"))
paths += list((root / "scripts").glob("*.sh"))
paths += [p for p in (root / "notebooks").glob("*.ipynb") if not p.name.startswith("baseline_before_")]
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(paths)):
        name = "project/" + path.relative_to(root).as_posix()
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    cell.update(execution_count=None, outputs=[])
            archive.writestr(name, json.dumps(notebook, indent=1) + "\n")
        else:
            archive.write(path, name)
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
print(output)
