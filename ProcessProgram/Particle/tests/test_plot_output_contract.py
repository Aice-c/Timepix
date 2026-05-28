import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _call_keywords(path: Path, function_name: str, call_name: str) -> list[dict[str, object]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls: list[dict[str, object]] = []
    in_target = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            in_target = True
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == call_name:
                    keywords: dict[str, object] = {}
                    for kw in child.keywords:
                        if kw.arg == "density":
                            keywords[kw.arg] = ast.literal_eval(kw.value)
                    calls.append(keywords)
            break
    assert in_target, f"Function not found: {function_name}"
    return calls


class PlotOutputContractTest(unittest.TestCase):
    def test_feature_histograms_use_candidate_count_not_density(self) -> None:
        scripts = [
            (ROOT / "plot_stage1_cleaning_diagnostics.py", "plot_feature_histograms"),
            (ROOT / "plot_stage1_angle_diagnostics.py", "plot_particle_angle_feature_histograms"),
            (ROOT / "plot_stage1_angle_diagnostics.py", "plot_angle_particle_feature_histograms"),
        ]
        for path, function_name in scripts:
            with self.subTest(path=path.name, function=function_name):
                hist_calls = _call_keywords(path, function_name, "hist")
                self.assertGreater(len(hist_calls), 0)
                for keywords in hist_calls:
                    self.assertNotEqual(keywords.get("density"), True)

    def test_plot_filenames_separate_histograms_from_hexbin_counts(self) -> None:
        for path in [
            ROOT / "plot_stage1_cleaning_diagnostics.py",
            ROOT / "plot_stage1_angle_diagnostics.py",
        ]:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("marginal_features", text)
                self.assertNotIn("marginal_distributions", text)
                self.assertNotIn("active_total_density", text)
                self.assertIn("feature_histograms_count", text)
                self.assertIn("active_total_hexbin_count", text)

    def test_co60_detail_is_not_a_mixed_plot(self) -> None:
        text = (ROOT / "plot_co60_cleaning_detail.py").read_text(encoding="utf-8")
        self.assertNotIn("co60_active_total_and_aspect_detail", text)
        self.assertIn("co60_active_total_hexbin_count", text)
        self.assertIn("co60_bbox_aspect_histogram_count", text)


if __name__ == "__main__":
    unittest.main()
