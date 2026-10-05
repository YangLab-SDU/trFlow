"""Startup must reject an occupied port before a prediction worker exists."""
import argparse
from http.server import BaseHTTPRequestHandler
import tempfile
import unittest
from unittest import mock

from trflow.web import LocalHTTPServer, run


class ServerStartupTests(unittest.TestCase):
    def test_duplicate_port_never_constructs_a_job_manager(self):
        server = LocalHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        try:
            with tempfile.TemporaryDirectory() as directory:
                args = argparse.Namespace(port=server.server_port, data_dir=directory,
                                          python="not-used", env=None,
                                          no_browser=True, no_import_existing=True)
                with mock.patch("trflow.web.JobManager") as manager:
                    with self.assertRaises(OSError):
                        run(args)
                    manager.assert_not_called()
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
