import unittest
from pathlib import Path

from trflow.config import REPO
from trflow.output import _portable_paths


class PortablePathTests(unittest.TestCase):
    def test_repository_paths_become_relative_recursively(self):
        value = {
            "file": str(REPO / "example" / "example_input.json"),
            "nested": [str(REPO / "outputs" / "sample")],
        }
        self.assertEqual(
            _portable_paths(value),
            {
                "file": "example/example_input.json",
                "nested": ["outputs/sample"],
            },
        )

    def test_external_absolute_path_is_preserved(self):
        external = Path("/opt/trflow/python")
        self.assertEqual(_portable_paths(str(external)), str(external))


if __name__ == "__main__":
    unittest.main()
