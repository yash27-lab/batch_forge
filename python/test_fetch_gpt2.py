"""Network-free regression tests for GPT-2 asset installation."""
import os
import tempfile
import unittest
from unittest.mock import patch

import fetch_gpt2


class DownloadTests(unittest.TestCase):
    def test_complete_download_replaces_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = os.path.join(directory, "model")
            def transfer(url, temporary):
                with open(temporary, "wb") as stream:
                    stream.write(b"complete")
            with patch("fetch_gpt2.urllib.request.urlretrieve", side_effect=transfer):
                fetch_gpt2.download_file("https://example.test/model", destination)
            with open(destination, "rb") as stream:
                self.assertEqual(stream.read(), b"complete")
            self.assertEqual(os.listdir(directory), ["model"])

    def test_failure_preserves_previous_asset_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = os.path.join(directory, "model")
            with open(destination, "wb") as stream:
                stream.write(b"previous")
            def transfer(url, temporary):
                with open(temporary, "wb") as stream:
                    stream.write(b"partial")
                raise OSError("interrupted")
            with patch("fetch_gpt2.urllib.request.urlretrieve", side_effect=transfer):
                with self.assertRaises(OSError):
                    fetch_gpt2.download_file("https://example.test/model", destination)
            with open(destination, "rb") as stream:
                self.assertEqual(stream.read(), b"previous")
            self.assertEqual(os.listdir(directory), ["model"])


if __name__ == "__main__":
    unittest.main()
