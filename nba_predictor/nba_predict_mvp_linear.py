#!/usr/bin/env python3
"""Predict NBA MVP rankings with linear regression. Run with --help for options."""
import sys

from nba_predictor.nba_mvp_common import main


if __name__ == '__main__':
    try:
        sys.exit(main('linear'))
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f'ERROR: {exc}')
