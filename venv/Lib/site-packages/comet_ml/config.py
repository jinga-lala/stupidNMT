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

from __future__ import print_function

import io
import logging
import os
import os.path

import six
from everett.manager import (
    NO_VALUE,
    ConfigDictEnv,
    ConfigEnvFileEnv,
    ConfigManager,
    ConfigOSEnv,
    ListOf,
    listify,
    parse_bool,
)

from ._typing import Any, Dict, List, Optional
from .utils import log_once_at_level

try:  # everett version 1.0.1 or greater
    from everett.ext.inifile import ConfigIniEnv
except ImportError:  # everett version 0.9 for Python 2
    from everett.manager import ConfigIniEnv as UpstreamConfigIniEnv

    class ConfigIniEnv(UpstreamConfigIniEnv):
        """ Backport of everett ConfigIniEnv in Python 3 that save the path when found
        """

        def __init__(self, possible_paths):
            super(ConfigIniEnv, self).__init__(possible_paths)

            if self.cfg:
                self.path = os.path.abspath(os.path.expanduser(possible_paths.strip()))


LOGGER = logging.getLogger(__name__)

DEBUG = False

# Global experiment placeholder. Should be set by the latest call of Experiment.init()
experiment = None

DEFAULT_UPLOAD_SIZE_LIMIT = 100 * 1024 * 1024  # 100 MebiBytes

DEFAULT_ASSET_UPLOAD_SIZE_LIMIT = 2500 * 1024 * 1024  # 2.5GB

DEFAULT_STREAMER_MSG_TIMEOUT = 5 * 60

ADDITIONAL_STREAMER_UPLOAD_TIMEOUT = 10 * 60


def get_global_experiment():
    global experiment
    return experiment


def set_global_experiment(new_experiment):
    global experiment
    experiment = new_experiment


def parse_str_or_identity(_type):
    def parse(value):
        if not isinstance(value, str):
            return value

        return _type(value.strip())

    return parse


class ParseListOf(ListOf):
    """
    Superclass to apply subparser to list items.
    """

    def __init__(self, _type, _parser):
        super(ParseListOf, self).__init__(_type)
        self._type = _type
        self._parser = _parser

    def __call__(self, value):
        f = self._parser(self._type)
        if not isinstance(value, list):
            value = super(ParseListOf, self).__call__(value)
        return [f(v) for v in value]


PARSER_MAP = {
    str: parse_str_or_identity(str),
    int: parse_str_or_identity(int),
    float: parse_str_or_identity(float),
    bool: parse_str_or_identity(parse_bool),
    list: ParseListOf(str, parse_str_or_identity),
    "int_list": ParseListOf(int, parse_str_or_identity(int)),
}


# Vendor generate_uppercase_key for Python 2
def generate_uppercase_key(key, namespace=None):
    """Given a key and a namespace, generates a final uppercase key."""
    if namespace:
        namespace = [part for part in listify(namespace) if part]
        key = "_".join(namespace + [key])

    key = key.upper()
    return key


class Config(object):
    def __init__(self, config_map):
        self.config_map = config_map
        self.override = {}  # type: Dict[str, Any]
        self.backend_override = ConfigDictEnv({})
        config_override = os.environ.get("COMET_INI")
        if config_override is not None:
            log_once_at_level(
                logging.WARNING, "COMET_INI is deprecated; use COMET_CONFIG"
            )
        else:
            config_override = os.environ.get("COMET_CONFIG")
        self.manager = ConfigManager(
            [  # User-defined overrides
                ConfigOSEnv(),
                ConfigEnvFileEnv(".env"),
                ConfigIniEnv(config_override),
                ConfigIniEnv("./.comet.config"),
                ConfigIniEnv("~/.comet.config"),
                # Comet-defined overrides
                self.backend_override,
            ],
            doc=(
                "See https://comet.ml/docs/python-sdk/getting-started/ for more "
                + "information on configuration."
            ),
        )

    def __setitem__(self, name, value):
        self.override[name] = value

    def _set_backend_override(self, cfg, namespace):
        # Reset the existing overrides
        self.backend_override.cfg = {}

        for key, value in cfg.items():
            namespaced_key = "_".join(namespace.split("_") + [key])
            full_key = generate_uppercase_key(namespaced_key)
            self.backend_override.cfg[full_key] = value

    def keys(self):
        return self.config_map.keys()

    def get_raw(self, user_value, config_name, default=None, not_set_value=None):
        # type: (Any, str, Optional[Any], Optional[Any]) -> Any
        """
        Returns the correct config value based on the following priority list:
        * User_value if set and not None
        * The override value from the Backend
        * The configured value
        * The default value passed in argument if not None
        * The configured value default
        """

        # 1. User value
        if user_value is not not_set_value:
            return user_value

        # 2. Override
        if config_name in self.override:
            override_value = self.override[config_name]

            if override_value is not None:
                return override_value

        # 3. Configured value
        config_type = self.config_map[config_name].get("type", str)
        parser = PARSER_MAP[config_type]

        # Value
        splitted = config_name.split(".")

        config_value = self.manager(
            splitted[-1], namespace=splitted[:-1], parser=parser, raise_error=False
        )

        if config_value != NO_VALUE:
            return config_value

        else:
            # 4. Provided default
            if default is not None:
                return default

            # 5. Config default
            config_default = parser(self.config_map[config_name].get("default", None))
            return config_default

    def get_string(self, user_value, config_name, default=None, not_set_value=None):
        # type: (Any, str, Optional[str], Any) -> str
        """
        Returns the correct config value based on the following priority list:
        * User_value if set and not None
        * The override value from the Backend
        * The configured value
        * The default value passed in argument if not None
        * The configured value default

        In addition make sure the returned value is a string
        """

        value = self.get_raw(
            user_value=user_value,
            config_name=config_name,
            default=default,
            not_set_value=not_set_value,
        )

        return value

    def get_bool(self, user_value, config_name, default=None, not_set_value=None):
        # type: (Any, str, Optional[bool], Any) -> bool
        """
        Returns the correct config value based on the following priority list:
        * User_value if set and not None
        * The override value from the Backend
        * The configured value
        * The default value passed in argument if not None
        * The configured value default

        In addition make sure the returned value is a bool
        """

        value = self.get_raw(
            user_value=user_value,
            config_name=config_name,
            default=default,
            not_set_value=not_set_value,
        )

        return value

    def get_int(self, user_value, config_name, default=None, not_set_value=None):
        # type: (Any, str, Optional[int], int) -> bool
        """
        Returns the correct config value based on the following priority list:
        * User_value if set and not None
        * The override value from the Backend
        * The configured value
        * The default value passed in argument if not None
        * The configured value default

        In addition make sure the returned value is an int
        """

        value = self.get_raw(
            user_value=user_value,
            config_name=config_name,
            default=default,
            not_set_value=not_set_value,
        )

        return value

    def get_int_list(self, user_value, config_name, default=None, not_set_value=None):
        # type: (Any, str, Optional[int], int) -> List[int]
        """
        Returns the correct config value based on the following priority list:
        * User_value if set and not None
        * The override value from the Backend
        * The configured value
        * The default value passed in argument if not None
        * The configured value default

        In addition make sure the returned value is a list of int
        """

        value = self.get_raw(
            user_value=user_value,
            config_name=config_name,
            default=default,
            not_set_value=not_set_value,
        )

        return value

    def get_string_list(
        self, user_value, config_name, default=None, not_set_value=None
    ):
        # type: (Any, str, Optional[int], int) -> List[str]
        """
        Returns the correct config value based on the following priority list:
        * User_value if set and not None
        * The override value from the Backend
        * The configured value
        * The default value passed in argument if not None
        * The configured value default

        In addition make sure the returned value is a list of str
        """

        value = self.get_raw(
            user_value=user_value,
            config_name=config_name,
            default=default,
            not_set_value=not_set_value,
        )

        return value

    def get_deprecated_raw(
        self,
        old_user_value,
        old_config_name,
        new_user_value,
        new_config_name,
        new_not_set_value=None,
    ):
        # type: (Any, str, Any, str, Any) -> Any
        """
        Returns the correct value for deprecated config values:
        * New user value
        * Old user value
        * New config value
        * Old config value
        * New config default

        Note: The old config default is not used and should be set to None
        """
        old_config_value = self.get_raw(None, old_config_name, default=NO_VALUE)

        if new_user_value is not new_not_set_value:
            if old_user_value:
                LOGGER.warning(
                    "Deprecated config key %r was set, but ignored as new config key %r is set",
                    old_config_name,
                    new_config_name,
                )
            elif old_config_value:
                LOGGER.warning(
                    "Deprecated config key %r was set in %r, but ignored as new config key %r is set",
                    old_config_name,
                    self.get_config_origin(old_config_name),
                    new_config_name,
                )
            return new_user_value

        # Deprecated parameter default value must be None
        if old_user_value is not None:
            LOGGER.warning(
                "Config key %r is deprecated, please use %r instead",
                old_config_name,
                new_config_name,
            )
            return old_user_value

        new_config_value = self.get_raw(None, new_config_name, default=NO_VALUE)
        if new_config_value is not NO_VALUE:
            return new_config_value

        old_config_value = self.get_raw(None, old_config_name, default=NO_VALUE)
        if old_config_value is not NO_VALUE:
            LOGGER.warning(
                "Config key %r is deprecated (was set in %r), please use %r instead",
                old_config_name,
                self.get_config_origin(old_config_name),
                new_config_name,
            )
            return old_config_value

        config_type = self.config_map[new_config_name].get("type", str)
        parser = PARSER_MAP[config_type]
        return parser(self.config_map[new_config_name].get("default", None))

    def get_deprecated_bool(
        self,
        old_user_value,
        old_config_name,
        new_user_value,
        new_config_name,
        new_not_set_value=None,
    ):
        # type: (Any, str, Any, str, bool) -> bool
        """
        Returns the correct value for deprecated config values:
        * New user value
        * Old user value
        * New config value
        * Old config value
        * New config default

        Note: The old config default is not used and should be set to None
        """
        value = self.get_deprecated_raw(
            old_user_value,
            old_config_name,
            new_user_value,
            new_config_name,
            new_not_set_value=new_not_set_value,
        )

        return value

    def get_subsections(self):
        """
        Return the subsection config names.
        """
        sections = set()
        for key in self.keys():
            parts = key.split(".", 2)
            if len(parts) == 3:
                sections.add(parts[1])
        return sections

    def __getitem__(self, name):
        # type: (str) -> Any
        # Config
        config_type = self.config_map[name].get("type", str)
        parser = PARSER_MAP[config_type]
        config_default = self.config_map[name].get("default", None)

        if name in self.override:
            return self.override[name]

        # Value
        splitted = name.split(".")

        value = self.manager(
            splitted[-1], namespace=splitted[:-1], parser=parser, raise_error=False
        )

        if value == NO_VALUE:
            return parser(config_default)

        return value

    def display(self, display_all=False):
        """
        Show the Comet config variables and values.
        """
        n = 1
        print("=" * 65)
        print("Comet config variables and values, in order of preference:")
        print("    %d) Operating System Variable" % n)
        n += 1
        for path in ["./.env", "~/.comet.config", "./.comet.config"]:
            path = os.path.abspath(os.path.expanduser(path))
            if os.path.exists(path):
                print("    %d) %s" % (n, path))
                n += 1
        print("=" * 65)
        print("Settings:\n")
        last_section = None
        for section, setting in sorted(
            [key.rsplit(".", 1) for key in self.config_map.keys()]
        ):
            key = "%s.%s" % (section, setting)
            value = self[key]
            if "." in section:
                section = section.replace(".", "_")
            if value is None:
                value = "..."
            default_value = self.config_map[key].get("default", None)
            if value == default_value or value == "...":
                if display_all:
                    if section != last_section:
                        if last_section is not None:
                            print()  # break between sections
                        print("[%s]" % section)
                        last_section = section
                    print("%s = %s" % (setting, value))
            else:
                if section != last_section:
                    if last_section is not None:
                        print("")  # break between sections
                    print("[%s]" % section)
                    last_section = section
                print("%s = %s" % (setting, value))
        print("=" * 65)

    def save(self, directory="./", save_all=False, **kwargs):
        """
        Save the settings to .comet.config (default) or
        other path/filename. Defaults are commented out.

        Args:
            directory: the path to save the .comet.config config settings.
            save_all: save unset variables with defaults too
            kwargs: key=value pairs to save
        """
        directory = os.path.expanduser(directory)
        filename = os.path.abspath(os.path.join(directory, ".comet.config"))
        print('Saving config to "%s"...' % filename, end="")
        with io.open(filename, "w", encoding="utf-8") as ini_file:
            ini_file.write(six.u("# Config file for Comet.ml\n"))
            ini_file.write(
                six.u(
                    "# For help see https://www.comet.ml/docs/python-sdk/getting-started/\n"
                )
            )
            last_section = None
            for section, setting in sorted(
                [key.rsplit(".", 1) for key in self.config_map.keys()]
            ):
                key = "%s.%s" % (section, setting)
                key_arg = "%s_%s" % (section, setting)
                if key_arg in kwargs:
                    value = kwargs[key_arg]
                    del kwargs[key_arg]
                elif key_arg.upper() in kwargs:
                    value = kwargs[key_arg.upper()]
                    del kwargs[key_arg.upper()]
                else:
                    value = self[key]
                if len(kwargs) != 0:
                    raise ValueError(
                        "'%s' is not a valid config key" % list(kwargs.keys())[0]
                    )
                if "." in section:
                    section = section.replace(".", "_")
                if value is None:
                    value = "..."
                default_value = self.config_map[key].get("default", None)
                LOGGER.debug("default value for %s is %s", key, default_value)
                if value == default_value or value == "...":
                    # It is a default value
                    # Only save it, if save_all is True:
                    if save_all:
                        if section != last_section:
                            if section is not None:
                                ini_file.write(six.u("\n"))  # break between sections
                            ini_file.write(six.u("[%s]\n" % section))
                            last_section = section
                        if isinstance(value, list):
                            value = ",".join(value)
                        ini_file.write(six.u("# %s = %s\n" % (setting, value)))
                else:
                    # Not a default value; write it out:
                    if section != last_section:
                        if section is not None:
                            ini_file.write(six.u("\n"))  # break between sections
                        ini_file.write(six.u("[%s]\n" % section))
                        last_section = section
                    if isinstance(value, list):
                        value = ",".join([str(v) for v in value])
                    ini_file.write(six.u("%s = %s\n" % (setting, value)))
        print(" done!")

    def get_config_origin(self, name):
        # type: (str) -> Optional[str]
        splitted = name.split(".")

        for env in self.manager.envs:
            value = env.get(splitted[-1], namespace=splitted[:-1])

            if value != NO_VALUE:
                return env

        return None


CONFIG_MAP = {
    "comet.disable_auto_logging": {"type": int, "default": 0},
    "comet.api_key": {"type": str},
    "comet.rest_api_key": {"type": str},
    "comet.offline_directory": {"type": str},
    "comet.git_directory": {"type": str},
    "comet.offline_sampling_size": {"type": int, "default": 15000},
    "comet.url_override": {"type": str, "default": "https://www.comet.ml/clientlib/"},
    "comet.optimizer_url": {"type": str, "default": "https://optimizer.comet.ml"},
    "comet.ws_url_override": {"type": str, "default": None},
    "comet.predictor_url": {"type": str, "default": "https://predictor.comet.ml/"},
    "comet.experiment_key": {"type": str},
    "comet.project_name": {"type": str},
    "comet.workspace": {"type": str},
    "comet.display_summary_level": {"type": int, "default": 1},
    # Logging
    "comet.logging.file": {"type": str},
    "comet.logging.file_level": {"type": str, "default": "INFO"},
    "comet.logging.file_overwrite": {"type": bool, "default": False},
    "comet.logging.console": {"type": str},
    "comet.logging.metrics_ignore": {
        "type": list,
        "default": "keras:batch_size,keras:batch_batch",
    },
    "comet.logging.parameters_ignore": {
        "type": list,
        "default": "keras:verbose,keras:do_validation,keras:validation_steps",
    },
    "comet.logging.others_ignore": {"type": list, "default": ""},
    "comet.logging.env_blacklist": {
        "type": list,
        "default": "api_key,apikey,authorization,passwd,password,secret,token,comet",
    },
    # Timeout, unit is seconds
    "comet.timeout.cleaning": {"type": int, "default": DEFAULT_STREAMER_MSG_TIMEOUT},
    "comet.timeout.upload": {
        "type": int,
        "default": ADDITIONAL_STREAMER_UPLOAD_TIMEOUT,
    },
    "comet.timeout.http": {"type": int, "default": 10},
    "comet.timeout.api": {"type": int, "default": 10},
    "comet.timeout.file_upload": {"type": int, "default": 900},
    "comet.timeout.file_download": {"type": int, "default": 600},
    "comet.timeout.predictor": {"type": int, "default": 60},
    # HTTP Allow header
    "comet.allow_header.name": {"type": str},
    "comet.allow_header.value": {"type": str},
    # Backend minimal rest V2 version
    "comet.rest_v2_minimal_backend_version": {"type": str, "default": "1.2.78"},
    # Feature flags
    "comet.override_feature.sdk_http_logging": {
        "type": bool
    },  # Leave feature toggle default to None
    "comet.override_feature.use_http_messages": {
        "type": bool
    },  # Leave feature toggle default to None
    "comet.override_feature.sdk_log_env_variables": {
        "type": bool
    },  # Leave feature toggle default to None
    # Experiment log controls:
    "comet.auto_log.cli_arguments": {"type": bool},
    "comet.auto_log.code": {"type": bool},
    "comet.auto_log.disable": {"type": bool},
    "comet.auto_log.env_cpu": {"type": bool},
    "comet.auto_log.env_details": {"type": bool},
    "comet.auto_log.env_gpu": {"type": bool},
    "comet.auto_log.env_host": {"type": bool},
    "comet.auto_log.git_metadata": {"type": bool},
    "comet.auto_log.git_patch": {"type": bool},
    "comet.auto_log.graph": {"type": bool},
    "comet.auto_log.metrics": {"type": bool},
    "comet.auto_log.figures": {"type": bool, "default": True},
    "comet.auto_log.output_logger": {"type": str},
    "comet.auto_log.parameters": {"type": bool},
    "comet.auto_log.histogram_tensorboard": {"type": bool, "default": False},
    "comet.auto_log.histogram_epoch_rate": {"type": int, "default": 1},
    "comet.auto_log.histogram_weights": {"type": bool, "default": False},
    "comet.auto_log.histogram_gradients": {"type": bool, "default": False},
    "comet.auto_log.histogram_activations": {"type": bool, "default": False},
    "comet.keras.histogram_name_prefix": {
        "type": str,
        "default": "{layer_num:0{max_digits}d}",
    },
    "comet.keras.histogram_activation_index_list": {"type": "int_list", "default": "0"},
    "comet.keras.histogram_activation_layer_list": {"type": list, "default": "-1"},
    "comet.keras.histogram_batch_size": {"type": int, "default": 1000},
    "comet.keras.histogram_gradient_index_list": {"type": "int_list", "default": "0"},
    "comet.keras.histogram_gradient_layer_list": {"type": list, "default": "-1"},
    "comet.auto_log.metric_step_rate": {"type": int, "default": 10},
    "comet.auto_log.co2": {"type": bool},
    "comet.auto_log.tfma": {"type": bool, "default": False},
    # Internals:
    "comet.internal.reporting": {"type": bool, "default": True},
    # Deprecated:
    "comet.display_summary": {"type": bool, "default": None},
    "comet.auto_log.weights": {"type": bool, "default": None},
}


def get_config(setting=None):
    # type: (Any) -> Config
    """
    Get a config or setting from the current config
    (os.environment or .env file).

    Note: this is not cached, so every time we call it, it
    re-reads the file. This makes these values always up to date
    at the expense of re-getting the data.
    """
    cfg = Config(CONFIG_MAP)
    if setting is None:
        return cfg
    else:
        return cfg[setting]


def get_api_key(api_key, config):
    if api_key is None:
        return config["comet.api_key"]
    else:
        return api_key


def get_display_summary_level(display_summary_level, config):
    if display_summary_level is None:
        return config["comet.display_summary_level"]
    else:
        try:
            return int(display_summary_level)
        except Exception:
            LOGGER.warning(
                "invalid display_summary_level %r; ignoring", display_summary_level
            )
            return 1


def get_ws_url(ws_server_from_backend, config):
    """ Allow users to override the WS url from the backend using the usual
    config mechanism
    """
    ws_server_from_config = config["comet.ws_url_override"]
    if ws_server_from_config is None:
        return ws_server_from_backend
    else:
        return ws_server_from_config


def get_previous_experiment(previous_experiment, config):
    if previous_experiment is None:
        return config["comet.experiment_key"]
    else:
        return previous_experiment


def save(directory="~/", save_all=False, **settings):
    """
    An easy way to create a config file.

    Args:
        directory: str (optional), location to save the
            .comet.config file. Typical values are "~/" (home)
            and "./" (current directory). Default is "~/"
        save_all: bool (optional). If True, will create
            entries for all items that are configurable
            with their defaults. Default is False
        settings: any valid setting and value

    Valid settings include:

    * api_key
    * disable_auto_logging
    * experiment_key
    * offline_directory
    * workspace
    * project_name
    * logging_console
    * logging_file
    * logging_file_level
    * logging_file_overwrite
    * timeout_cleaning
    * timeout_upload

    Examples:

    >>> import comet_ml
    >>> comet_ml.config.save(api_key="...")
    >>> comet_ml.config.save(api_key="...", directory="./")
    """
    cfg = get_config()
    subsections = cfg.get_subsections()
    for setting in settings:
        # Correct the keys:
        key = None
        for prefix in subsections:
            if setting.startswith(prefix + "_"):
                key = ("comet.%s." % prefix) + setting[len(prefix) + 1 :]
                break
        if key is None:
            key = "comet." + setting
        cfg[key] = settings[setting]
    cfg.save(directory, save_all=save_all)
