import json
from pathlib import Path

HERE = Path(__file__).parent
EXACT_CONTACT = 3.62335
SPRINT1_S3125_RELATIVE = 0.0038

evaluation = json.loads((HERE / "evaluation.json").read_text())
metadata = json.loads((HERE / "fit" / "metadata.json").read_text())
contact = evaluation["contact_densities"][0]
result = {
    "contact_density": contact,
    "exact_contact_density": EXACT_CONTACT,
    "relative_contact_error": contact / EXACT_CONTACT - 1,
    "sprint1_relative_contact_error_same_recipe": SPRINT1_S3125_RELATIVE,
    "raw_electron_counts": evaluation["raw_electron_counts"],
    "dipole": evaluation["dipole"],
    "hellmann_feynman_forces": evaluation["hellmann_feynman_forces"],
    "nuclear_potential_energy": evaluation["nuclear_potential_energy"],
    "fit_seconds": metadata["history"][-1]["time"],
    "reload_error": metadata["wavefunction"]["reload_error"],
}
(HERE / "result.json").write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
