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
            root = Path(directory).resolve()
            (root / "data/raw").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "outputs"):
                output_directory(root, root / "data/raw/new_results")
            accepted = output_directory(root, root / "outputs/check")
            self.assertTrue(accepted.is_dir())


if __name__ == "__main__":
    unittest.main()
