"""Tests of training_utils.repository_root.

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_utils import repository_root  # noqa: E402


class RepositoryRootTests(unittest.TestCase):
    def test_it_is_the_folder_of_the_pipeline_modules(self):
        root = repository_root()
        self.assertTrue(os.path.isabs(root))
        for module in ("training_utils.py", "checkpoint_utils.py", "evaluation_utils.py", "training_config.py"):
            self.assertTrue(os.path.isfile(os.path.join(root, module)), module)

    def test_it_does_not_depend_on_the_working_directory(self):
        # a notebook opened from inside a result folder must still write into the repository root
        expected = repository_root()
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            try:
                os.chdir(elsewhere)
                self.assertEqual(repository_root(), expected)
            finally:
                os.chdir(original)   # leave the folder before it is deleted (Windows cannot delete the cwd)


if __name__ == "__main__":
    unittest.main()
