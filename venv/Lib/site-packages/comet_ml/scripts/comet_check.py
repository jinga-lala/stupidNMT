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

import argparse
import logging
import sys

from comet_ml._typing import Any
from comet_ml.config import (
    ConfigDictEnv,
    ConfigEnvFileEnv,
    ConfigIniEnv,
    ConfigOSEnv,
    get_config,
)
from comet_ml.connection import (
    _comet_version,
    get_http_proxy,
    get_http_session,
    get_root_url,
    get_run_id_url,
    sanitize_url,
    url_join,
    urlparse,
    urlunparse,
)
from comet_ml.utils import local_timestamp

import websocket

LOGGER = logging.getLogger("comet_ml")
ADDITIONAL_ARGS = False


def activate_debug():
    import http.client

    http.client.HTTPConnection.debuglevel = 5

    websocket.enableTrace(True)


def check_ws_connection(websocket_url, debug):
    # type: (str, bool) -> bool

    try:
        http_proxy_host, http_proxy_port, http_proxy_auth, proxy_type = get_http_proxy()
        ws = websocket.create_connection(
            websocket_url,
            http_proxy_host=http_proxy_host,
            http_proxy_port=http_proxy_port,
            http_proxy_auth=http_proxy_auth,
            proxy_type=proxy_type,
        )

        if debug:
            print("Sending test message")
            print(ws.send("test"))

            print("Receiving a message")
            print(ws.recv())

        ws.close()

        return True
    except Exception:
        LOGGER.error("Error checking websocket connectivity", exc_info=True)
        return False


def get_default_ws_url(clientlib_address):
    parsed_address = urlparse(clientlib_address)

    if parsed_address.scheme == "http":
        ws_scheme = "ws"
    elif parsed_address.scheme == "https":
        ws_scheme = "wss"
    else:
        raise NotImplementedError()

    ws_host = parsed_address.netloc
    ws_path = "ws/logger-ws"

    return urlunparse((ws_scheme, ws_host, ws_path, None, None, None))


def check_server_connection(server_address):
    url = get_run_id_url(server_address)
    try:
        with get_http_session(server_address) as session:
            payload = {
                "apiKey": "XXX",
                "local_timestamp": local_timestamp(),
                "experimentKey": "YYY",
                "offline": False,
                "projectName": None,
                "teamName": None,
                "libVersion": _comet_version(),
            }
            headers = {"Content-Type": "application/json;charset=utf-8"}
            session.post(url, data=payload, headers=headers)
            return True
    except Exception:
        LOGGER.error("Error checking server connectivity", exc_info=True)
        return False


def check_rest_api_connection(rest_api_url):
    url = url_join(rest_api_url, "account-details")
    try:
        with get_http_session(rest_api_url) as session:
            session.get(url)
            return True
    except Exception:
        LOGGER.error("Error checking rest api connectivity", exc_info=True)
        return False


def check_optimizer_connection(optimizer_url):
    url = url_join(optimizer_url, "spec")
    try:
        with get_http_session(optimizer_url) as session:
            params = {"algorithmName": "bayes"}
            session.get(url, params=params)
            return True
    except Exception:
        LOGGER.error("Error checking optimizer connectivity", exc_info=True)
        return False


def check_predictor_connection(predictor_url):
    url = url_join(predictor_url, "isAlive/ping")
    try:
        with get_http_session(predictor_url) as session:
            session.get(url)
            return True
    except Exception:
        LOGGER.error("Error checking predictor connectivity", exc_info=True)
        return False


def config_source(env):
    # type: (Any) -> str
    if isinstance(env, ConfigOSEnv):
        return "environment variable"
    elif isinstance(env, ConfigEnvFileEnv):
        return "environment file %r" % env.path
    elif isinstance(env, ConfigIniEnv):
        return "INI file %r" % env.path
    elif isinstance(env, ConfigDictEnv):
        return "backend overriden values"
    else:
        LOGGER.debug("Unknown env class %r", env)
        return None


def check(args, rest=None):
    # Called via `comet upload EXP.zip`
    if args.debug:
        activate_debug()

    config = get_config()

    LOGGER.info("Comet Check")
    LOGGER.info("=" * 80)
    print("")

    LOGGER.info("Checking connectivity to server...")
    print("")

    # Clientlib
    server_address = sanitize_url(config["comet.url_override"])
    server_address_config_origin = config_source(
        config.get_config_origin("comet.url_override")
    )
    LOGGER.info("Configured server address %r", server_address)
    if server_address_config_origin:
        LOGGER.info("Server address was configured in %s", server_address_config_origin)
    else:
        LOGGER.info("Server address is the default one")
    print("")
    server_connected = check_server_connection(server_address)
    print("")
    if server_connected:
        LOGGER.info("Server connection is ok")
    else:
        LOGGER.warning("Server connection is not ok")

    # Rest API
    LOGGER.info("=" * 80)
    LOGGER.info("Checking connectivity to Rest API...")
    LOGGER.info("=" * 80)

    root_url = sanitize_url(get_root_url(config["comet.url_override"]))
    rest_api_url = url_join(root_url, *["api/rest/", "v2" + "/"])
    LOGGER.info("Configured Rest API address %r", rest_api_url)
    if server_address_config_origin:
        LOGGER.info(
            "Rest API address was configured in %s", server_address_config_origin
        )
    else:
        LOGGER.info("Rest API address is the default one")
    print("")
    rest_api_connected = check_rest_api_connection(rest_api_url)
    print("")
    if rest_api_connected:
        LOGGER.info("REST API connection is ok")
    else:
        LOGGER.warning("REST API connection is not ok")

    # Websocket
    LOGGER.info("=" * 80)
    LOGGER.info("Checking connectivity to Websocket Server")
    LOGGER.info("=" * 80)

    websocket_url = config["comet.ws_url_override"]
    if websocket_url is None:
        websocket_url = get_default_ws_url(server_address)
        LOGGER.warning(
            "No WS address configured on client side, fallbacking on default WS address %r, if that's incorrect set the WS url through the `comet.ws_url_override` config key",
            websocket_url,
        )
        websocket_url_config_origin = None
    else:
        websocket_url = websocket_url
        websocket_url_config_origin = config_source(
            config.get_config_origin("comet.ws_url_override")
        )
    LOGGER.info(
        "Configured WS address %r", websocket_url,
    )
    if websocket_url_config_origin:
        LOGGER.info("WS address was configured in %s", websocket_url_config_origin)
    print("")
    ws_connected = check_ws_connection(websocket_url, args.debug)
    print("")
    if ws_connected:
        LOGGER.info("Websocket connection is ok")
    else:
        LOGGER.warning("Websocket connection is not ok")

    # Optimizer
    LOGGER.info("=" * 80)
    LOGGER.info("Checking connectivity to Optimizer Server")
    LOGGER.info("=" * 80)

    optimizer_url = sanitize_url(config["comet.optimizer_url"])
    optimizer_url_config_origin = config_source(
        config.get_config_origin("comet.optimizer_url")
    )
    LOGGER.info(
        "Configured Optimizer address %r", optimizer_url,
    )
    if optimizer_url_config_origin:
        LOGGER.info(
            "Optimizer address was configured in %s", optimizer_url_config_origin
        )
    else:
        LOGGER.info("Optimizer address is the default one")
    print("")
    optimizer_connected = check_optimizer_connection(optimizer_url)
    print("")
    if optimizer_connected:
        LOGGER.info("Optimizer connection is ok")
    else:
        LOGGER.warning("Optimizer connection is not ok")

    # Predictor
    LOGGER.info("=" * 80)
    LOGGER.info("Checking connectivity to Predictor Server")
    LOGGER.info("=" * 80)

    predictor_url = sanitize_url(config["comet.predictor_url"])
    predictor_url_config_origin = config_source(
        config.get_config_origin("comet.predictor_url")
    )
    LOGGER.info(
        "Configured Predictor address %r", predictor_url,
    )
    if predictor_url_config_origin:
        LOGGER.info(
            "Predictor address was configured in %s", predictor_url_config_origin
        )
    else:
        LOGGER.info("Predictor address is the default one")
    print("")
    predictor_connected = check_predictor_connection(predictor_url)
    print("")
    if predictor_connected:
        LOGGER.info("Predictor connection is ok")
    else:
        LOGGER.warning("Predictor connection is not ok")

    print("")
    print("")

    LOGGER.info("Summary")
    LOGGER.info("-" * 80)
    LOGGER.info("Server connectivity\t\t\t%s", server_connected)
    LOGGER.info("Rest API connectivity\t\t%r", rest_api_connected)
    LOGGER.info("WS server connectivity\t\t%r", ws_connected)
    LOGGER.info("Optimizer server connectivity\t%r", optimizer_connected)
    LOGGER.info("Predictor server connectivity\t%r", predictor_connected)


def get_parser_arguments(parser):
    parser.add_argument(
        "--debug",
        help="Activate the raw HTTP logs and be more verbose in general",
        action="store_const",
        const=True,
        default=False,
    )


def main(args):
    # Called via `comet check`
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)

    check(parsed_args)


if __name__ == "__main__":
    # Called via python -m comet_ml.scripts.comet_check
    main(sys.argv[1:])
