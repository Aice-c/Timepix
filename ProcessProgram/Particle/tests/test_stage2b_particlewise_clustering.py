import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ProcessProgram.Particle.stage2b_particlewise_clustering import (
    FEATURE_COLUMNS,
    build_particle_scaled_features,
    select_gmm_components,
)


def test_particle_scaling_is_fitted_independently():
    df = pd.DataFrame(
        {
            "particle": ["A", "A", "A", "B", "B", "B"],
            "Npix": [1, 2, 100, 10, 20, 1000],
            "S_total_ToT": [10, 20, 1000, 100, 200, 10000],
            "Pmax": [1.0, 0.5, 0.1, 1.0, 0.5, 0.1],
            "Rg": [0.0, 0.5, 5.0, 0.0, 1.0, 10.0],
            "E_pca": [1.0, 1.5, 3.0, 1.0, 2.0, 5.0],
            "Fbox": [1.0, 0.8, 0.2, 1.0, 0.6, 0.1],
        }
    )

    scaled, params = build_particle_scaled_features(df)

    assert list(FEATURE_COLUMNS) == ["Npix", "S_total_ToT", "Pmax", "Rg", "E_pca", "Fbox"]
    assert set(params) == {"A", "B"}
    for particle in ["A", "B"]:
        sub = scaled.loc[scaled["particle"] == particle]
        for name in FEATURE_COLUMNS:
            values = sub[f"scaled_{name}"].to_numpy()
            assert np.isfinite(values).all()
            assert abs(np.median(values)) < 1e-9


def test_gmm_selection_reports_requested_component_counts():
    rng = np.random.default_rng(42)
    x1 = rng.normal(loc=-2.0, scale=0.2, size=(20, 2))
    x2 = rng.normal(loc=2.0, scale=0.2, size=(20, 2))
    x = np.vstack([x1, x2])

    summary = select_gmm_components(x, random_state=42, component_counts=(1, 2, 3))

    assert summary["n_components"].tolist() == [1, 2, 3]
    assert summary["bic"].notna().all()
    assert summary["aic"].notna().all()
    assert summary.loc[summary["n_components"] == 2, "bic"].item() < summary.loc[
        summary["n_components"] == 1, "bic"
    ].item()


if __name__ == "__main__":
    test_particle_scaling_is_fitted_independently()
    test_gmm_selection_reports_requested_component_counts()
    print("stage2b particlewise clustering tests passed")
