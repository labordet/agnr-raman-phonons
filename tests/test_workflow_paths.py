"""The recommended command must never direct outputs into source data."""

from pathlib import Path
import tempfile
import unittest

from paper_reproduction.reproduce_all import data_directory, output_directory


class WorkflowPathTests(unittest.TestCase):
    def test_data_layout_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing"):
                data_directory(Path(directory))

    def test_protected_input_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "archive"
            (root / "data/raw").mkdir(parents=True)
            for name in ("data", "metadata", "figures", "tables", "outputs"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "outside"):
                    output_directory(root, root / name / "new_results")
            accepted = output_directory(root, root.parent / "external_results")
            self.assertTrue(accepted.is_dir())

    def test_redirected_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "archive"
            (root / "data").mkdir(parents=True)
            link = base / "redirected"
            try:
                link.symlink_to(root / "data", target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Directory symlinks are unavailable")
            with self.assertRaisesRegex(ValueError, "redirected"):
                output_directory(root, link / "new_results")


if __name__ == "__main__":
    unittest.main()
