#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.ml
#  Copyright (C) 2015-2020 Comet ML INC
#  This file can not be copied and/or distributed without the express
#  permission of Comet ML Inc.
# *******************************************************

""" Upload an offline experiment archive to the backend
"""

import argparse
import logging
import sys

from comet_ml.offline import main_upload

LOGGER = logging.getLogger("comet_ml")
ADDITIONAL_ARGS = False


def get_parser_arguments(parser):
    parser.add_argument(
        "archives", nargs="+", help="the offline experiment archives to upload"
    )
    parser.add_argument(
        "--force-reupload",
        help="force reupload offline experiments that were already uploaded",
        action="store_const",
        const=True,
        default=False,
    )


def upload(args, rest=None):
    main_upload(args.archives, args.force_reupload)


def main(args):
    # Called via `comet upload EXP.zip`
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)

    upload(parsed_args)


if __name__ == "__main__":
    # Called via `python -m comet_ml.scripts.comet_upload EXP.zip`
    # Called via `comet upload EXP.zip`
    main(sys.argv[1:])
