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

"""
Author: Boris Feld

This module contains logging configuration for Comet

"""
import json
import logging
import os
import re
import sys
from copy import copy
from datetime import datetime

import requests

from .config import get_config
from .json_encoder import NestedEncoder
from .logging_messages import *  # noqa ignore=F405
from .logging_messages import (
    COMET_DISABLED_AUTO_LOGGING_MSG,
    FILE_MSG_FORMAT,
    MSG_FORMAT,
)
from .utils import get_user, makedirs

LOGGER = logging.getLogger(__name__)


class TracebackLessFormatter(logging.Formatter):
    def format(self, record):

        if getattr(record, "show_traceback", False) is False:

            # Make a copy of the record to avoid altering it
            new_record = copy(record)

            # And delete exception information so no traceback could be formatted
            # and displayed
            new_record.exc_info = None
            new_record.exc_text = None
        else:
            new_record = record

        return super(TracebackLessFormatter, self).format(new_record)


def shorten_record_name(record_name):
    """ Return the first part of the record (which can be None, comet or
    comet.connection)
    """
    if record_name is None:
        return record_name

    return record_name.split(".", 1)[0]


class HTTPHandler(logging.Handler):
    def __init__(self, url, api_key, experiment_key):
        super(HTTPHandler, self).__init__()
        self.url = url
        self.api_key = api_key
        self.experiment_key = experiment_key
        self.session = requests.Session()

    def mapLogRecord(self, record):
        return record.__dict__

    def emit(self, record):
        """
        Emit a record.

        Send the record to the Web server as JSON body
        """
        try:
            payload = {
                "apiKey": self.api_key,
                "record": self.mapLogRecord(record),
                "experimentKey": self.experiment_key,
                "levelname": record.levelname,
                "sender": record.name,
                "shortSender": shorten_record_name(record.name),
            }
            body = json.dumps(payload, cls=NestedEncoder)

            response = self.session.post(
                self.url,
                data=body,
                headers={"Content-Type": "application/json;charset=utf-8"},
                timeout=10,
            )
            response.raise_for_status()
        except Exception:
            self.handleError(record)

    def handleError(self, record):
        # Hide errors to avoid bad interaction with console logging
        pass


def expand_log_file_path(log_file_path):
    """
    Expand patterns in the file logging path.

    Allowed patterns:
        * {datetime}
        * {pid}
        * {project}
        * {user}
    """

    def make_valid(s):
        # type: (str) -> str
        s = str(s).strip().replace(" ", "_")
        return re.sub(r"(?u)[^-\w.]", "", s)

    user = make_valid(get_user())

    patterns = {
        "datetime": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "pid": os.getpid(),
        "project": get_config("comet.project_name") or "general",
        "user": user,
    }
    if log_file_path is not None:
        try:
            return log_file_path.format(**patterns)
        except KeyError:
            LOGGER.info(
                "Invalid logging file pattern: '%s'; ignoring" % log_file_path,
                exc_info=True,
            )
            return log_file_path


def setup(config):
    root = logging.getLogger("comet_ml")
    root.setLevel(logging.DEBUG)

    # Don't send comet-ml to the application logger
    root.propagate = False

    # Add handler for console, basic INFO:
    console = logging.StreamHandler()
    logging_console = config["comet.logging.console"]

    if logging_console and logging_console.upper() in [
        "DEBUG",
        "ERROR",
        "INFO",
        "CRITICAL",
        "FATAL",
        "WARN",
        "WARNING",
    ]:
        console.setLevel(logging_console.upper())
        console.setFormatter(logging.Formatter(MSG_FORMAT))
    else:
        console.setLevel(logging.INFO)
        console.setFormatter(TracebackLessFormatter(MSG_FORMAT))
    root.addHandler(console)

    # The std* logger might conflicts with the logging if a log record is
    # emitted for each WS message as it would results in an infinite loop. To
    # avoid this issue, all log records after the creation of a message should
    # be at a level lower than info as the console handler is set to info
    # level.

    # Add an additional file handler
    log_file_path = expand_log_file_path(config["comet.logging.file"])
    log_file_level = config["comet.logging.file_level"]
    log_file_overwrite = config["comet.logging.file_overwrite"]
    if log_file_path is not None:
        # Create logfile path, if possible:
        try:
            makedirs(os.path.dirname(log_file_path), exist_ok=True)
        except Exception:
            LOGGER.error(
                "can't create path to log file %r", log_file_path, exc_info=True
            )

        try:
            # Overwrite file if comet.logging.file_overwrite:
            if log_file_overwrite:
                file_handler = logging.FileHandler(log_file_path, "w+")
            else:
                file_handler = logging.FileHandler(log_file_path)
        except Exception:
            LOGGER.error(
                "can't open log file %r; file logging is disabled",
                log_file_path,
                exc_info=True,
            )
            return

        if log_file_level is not None:
            log_file_level = log_file_level.upper()
            file_handler.setLevel(log_file_level)
        else:
            file_handler.setLevel(logging.DEBUG)

        file_handler.setFormatter(logging.Formatter(FILE_MSG_FORMAT))
        root.addHandler(file_handler)


def setup_http_handler(url, api_key, experiment_key):
    root = logging.getLogger("comet_ml")

    http_handler = HTTPHandler(url, api_key, experiment_key)
    http_handler.setLevel(logging.INFO)

    # Remove any other previous HTTPHandlers:
    root.handlers[:] = [
        handler
        for handler in list(root.handlers)
        if not isinstance(handler, HTTPHandler)
    ]

    # Add new HTTPHandler. Only the current experiment should
    # have its logs forwarded to the backend.
    root.addHandler(http_handler)

    return http_handler


ALREADY_IMPORTED_MODULES = set()


def check_module(module):
    """
    Check to see if a module has already been loaded.
    This is an error, unless comet.disable_auto_logging == 1
    """
    if get_config("comet.disable_auto_logging"):
        LOGGER.debug(COMET_DISABLED_AUTO_LOGGING_MSG % module)
    elif module in sys.modules:
        ALREADY_IMPORTED_MODULES.add(module)


def is_module_already_imported(module):
    return module in ALREADY_IMPORTED_MODULES


def _reset_already_imported_modules():
    # Modify the set in place to be sure to keep the same reference every
    # where
    ALREADY_IMPORTED_MODULES.clear()
