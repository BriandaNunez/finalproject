#!/usr/bin/env python

"""Tests for `finalproject_itesm_mlops` package."""


import unittest
from click.testing import CliRunner

from finalproject_itesm_mlops import finalproject_itesm_mlops
from finalproject_itesm_mlops import cli


class TestFinalproject_itesm_mlops(unittest.TestCase):
    """Tests for `finalproject_itesm_mlops` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'finalproject_itesm_mlops.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
