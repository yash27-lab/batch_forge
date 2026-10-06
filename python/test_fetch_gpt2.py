"""Network-free regression tests for GPT-2 asset installation."""
import argparse
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import fetch_gpt2


class Response(io.BytesIO):
    def __init__(self, payload=b"complete", length=None):
        super().__init__(payload)
        self.headers = {} if length is None else {"Content-Length": str(length)}


class DownloadTests(unittest.TestCase):
    def test_complete_download_replaces_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = os.path.join(directory, "model")
            with patch("fetch_gpt2.urllib.request.urlopen", return_value=Response(length=8)) as opener:
                fetch_gpt2.download_file("https://example.test/model", destination, 12.0)
                opener.assert_called_once_with("https://example.test/model", timeout=12.0)
            with open(destination, "rb") as stream:
                self.assertEqual(stream.read(), b"complete")
            self.assertEqual(os.listdir(directory), ["model"])

    def test_failure_preserves_previous_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = os.path.join(directory, "model")
            with open(destination, "wb") as stream:
                stream.write(b"previous")
            with patch("fetch_gpt2.urllib.request.urlopen", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    fetch_gpt2.download_file("https://example.test/model", destination)
            with open(destination, "rb") as stream:
                self.assertEqual(stream.read(), b"previous")
            self.assertEqual(os.listdir(directory), ["model"])

    def test_empty_and_truncated_downloads_never_become_final_assets(self):
        for payload, length in [(b"", None), (b"short", 100)]:
            with tempfile.TemporaryDirectory() as directory:
                destination = os.path.join(directory, "model")
                with patch("fetch_gpt2.urllib.request.urlopen", return_value=Response(payload, length)):
                    with self.assertRaises(OSError):
                        fetch_gpt2.download_file("https://example.test/model", destination)
                self.assertEqual(os.listdir(directory), [])

    def test_timeout_validation(self):
        for value in ["0", "-1", "NaN", "inf", "invalid"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                fetch_gpt2.positive_timeout(value)
        self.assertEqual(fetch_gpt2.positive_timeout("5"), 5.0)

    def test_existing_assets_are_skipped_unless_forced(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in fetch_gpt2.FILES:
                with open(os.path.join(directory, name), "wb") as stream:
                    stream.write(b"previous")
            with patch("builtins.print"), patch("fetch_gpt2.download_file") as transfer:
                self.assertEqual(fetch_gpt2.main(["--out", directory]), 0)
                transfer.assert_not_called()
                self.assertEqual(fetch_gpt2.main(["--out", directory, "--force", "--timeout", "3"]), 0)
                self.assertEqual(transfer.call_count, len(fetch_gpt2.FILES))

    def test_main_reports_network_failure_as_unsuccessful(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("builtins.print"), patch("fetch_gpt2.download_file", side_effect=OSError("network")):
                self.assertEqual(fetch_gpt2.main(["--out", directory]), 1)


if __name__ == "__main__":
    unittest.main()
