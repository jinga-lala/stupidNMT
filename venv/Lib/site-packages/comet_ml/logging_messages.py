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
#  This file can not be copied and/or distributed without
#  the express permission of Comet ML Inc.
# *******************************************************

# Don't import comet_ml items here to avoid import cycles

MSG_FORMAT = "COMET %(levelname)s: %(message)s"

FILE_MSG_FORMAT = "[%(process)d-%(processName)s:%(thread)d] %(relativeCreated)d COMET %(levelname)s [%(filename)s:%(lineno)d]: %(message)s"

GO_TO_DOCS_MSG = " \nFor more details, please refer to: https://www.comet.ml/docs/python-sdk/warnings-errors/"

INTERNET_CONNECTION_ERROR = (
    "Failed to establish a connection to Comet server. Please check your internet connection. "
    "Your experiment would not be logged" + GO_TO_DOCS_MSG
)

INVALID_WORKSPACE_NAME = "Workspace %s doesn't exist."

INVALID_PROJECT_NAME = "project_name argument can't be empty."

INVALID_API_KEY = (
    "The given API key %s is invalid, please check it against the dashboard. "
    "Your experiment would not be logged" + GO_TO_DOCS_MSG
)

INVALID_REST_API_KEY = (
    "The given REST API key %s is invalid, please check it against the dashboard. "
)

METRIC_ARRAY_WARNING = (
    "Cannot safely convert %r object to a scalar value, using its string"
    " representation for logging."
)

PARSING_ERR_MSG = """We failed to parse your parameter configuration file.

Type casting will be disabled for this run, please fix your configuration file.
"""

CASTING_ERROR_MESSAGE = """Couldn't cast parameter %r, returning raw value instead.
Please report it to comet.ml and use `.raw(%r)` instead of `[%r]` in the meantime."""

LOG_ASSET_FOLDER_ERROR = (
    "We failed to read directory %s for uploading.\n"
    "Please double-check the file path, permissions, and that it is a directory"
)

LOG_ASSET_FOLDER_EMPTY = (
    "Directory %s is empty; no files were uploaded.\n"
    "Please double-check the directory path and the recursive parameter"
)

UPLOAD_FILE_OS_ERROR = (
    "We failed to read file %s for uploading.\n"
    "Please double-check the file path and permissions"
)

UPLOAD_ASSET_TOO_BIG = "Asset %s is bigger than the upload limit, %s > %s"

LOG_IMAGE_TOO_BIG = "Image %s is bigger than the upload limit, %s > %s"

LOG_FIGURE_TOO_BIG = "Figure number %d is bigger than the upload limit, %s > %s"

LOG_AUDIO_TOO_BIG = "Audio %s is bigger than the upload limit, %s > %s"

NATIVE_STD_WRAPPER_NOT_AVAILABLE = (
    "Native output logging mode is not available, fallbacking on basic output logging"
)

UNKOWN_STD_WRAPPER_SPEC = (
    "Unknown output logging mode: %s, fallbacking on basic output logging"
)

EXPERIMENT_LIVE = "Experiment is live on comet.ml %s\n"

OFFLINE_EXPERIMENT_END = "To upload this offline experiment, run:\n    comet upload %s"

OFFLINE_SENDER_STARTS = "Starting the upload of the experiment"

OFFLINE_SENDER_ENDS = "The offline experiment has been uploaded on comet.ml %s\n"

ADD_TAGS_ERROR = "Failed to add tag(s) %s to the experiment\n"

ADD_SYMLINK_ERROR = "Failed to create symlink to project:%s for experiment\n"

OFFLINE_EXPERIMENT_TEMPORARY_DIRECTORY = (
    "The offline directory %r was not usable.\n"
    "Reason: %r\n"
    "Saving the experiment in directory %s instead."
)

OFFLINE_EXPERIMENT_INVALID_WS_MSG = "An invalid message has been detected"

OFFLINE_EXPERIMENT_INVALID_UPLOAD_MSG = "An invalid upload message has been detected"

EXPERIMENT_INVALID_STEP = "Passed step value %r is not a number, ignoring it"

EXPERIMENT_INVALID_EPOCH = "Passed epoch value %r is not a number, ignoring it"

STREAMER_WAIT_FOR_FINISH_FAILED = (
    "Comet failed to send all the data back (%d messages and %d uploads)"
)

REGISTER_RPC_FAILED = "Failed to register callback named %r"

WS_SSL_ERROR_MSG = (
    "There's seem to be an issue with your system's SSL certificate bundle."
    "This is likely a system-wide issue that is not related to Comet."
    "Please see more information here:"
    "https://www.comet.ml/docs/python-sdk/warnings-errors/"
)

OFFLINE_EXPERIMENT_ALREADY_UPLOADED = (
    "Experiment %r was already uploaded, you can re-upload it by using the"
    " --force-reupload flag"
)

SEND_NOTIFICATION_FAILED = (
    "Error sending a notification, make sure you have opted-in for notifications"
)

GET_CALLBACK_FAILURE = "Failure to add Comet SDK callback to %s, the run will likely miss parameters and/out metrics"

WS_ON_OPEN_MSG = "WS Socket connection open"

WS_ON_CLOSE_MSG = "WS connection closed %r"

COMET_DISABLED_AUTO_LOGGING_MSG = "COMET_DISABLE_AUTO_LOGGING is 1; ignoring '%s'"

LOG_DATASET_ERROR = "Failed to create dataset hash"

OS_PACKAGE_MSG_SENDING_ERROR = "Error sending os_packages message, got %d %r"

MODEL_GRAPH_MSG_SENDING_ERROR = "Error sending model graph message, got %d %r"

CLOUD_DETAILS_MSG_SENDING_ERROR = "Error sending cloud details messages, got %d %r"

OPTIMIZER_KWARGS_CONSTRUCTOR_DEPRACETED = "Passing Experiment through Optimizer constructor is deprecated; pass them to Optimizer.get_experiments or Optimizer.next"

BACKEND_VERSION_CHECK_ERROR = "Failed to check backend version at URL: %r"

INVALID_CONFIG_MINIMAL_BACKEND_VERSION = "Invalid configured `comet.rest_v2_minimal_backend_version` value %r, skipping backend version check"

MLFLOW_NESTED_RUN_UNSUPPORTED = "MLFlow Nested Runs are not tracked in Comet.ml SDK."

MLFLOW_OFFLINE_EXPERIMENT_FALLBACK = "No Comet API Key was found, creating an OfflineExperiment. Set up your API Key to get the full Comet experience https://www.comet.ml/docs/python-sdk/advanced/#python-configuration"

MLFLOW_RESUMED_RUN = "Resumed MLFlow run are tracked as new Comet.ml Experiments"

ONLINE_EXPERIMENT_THROTTLED = "Experiment has been throttled, some data will be missing"

TF_KERAS_FALLBACK_FAILED = (
    "Keras is not available; attempted to fallback to tensorflow.keras but still failed"
)

CONFUSION_MATRIX_ERROR = "Error creating confusion matrix: %s; ignoring"

CONFUSION_MATRIX_GENERAL_ERROR = "Error logging confusion matrix; ignoring"

CONFUSION_MATRIX_ERROR_WRONG_LENGTH = (
    "y_true and y_predicted should have the same lengths (%d != %d)"
)

CONFUSION_MATRIX_ERROR_RESULTING_LENGTH = (
    "Resulting len(y_true) != len(y_predicted) (%s != %s)"
)

CONFUSION_MATRIX_ERROR_MUST_GIVE_BOTH = (
    "if you give y_true OR y_predicted you must give both"
)
CODECARBON_NOT_INSTALLED = "codecarbon package not installed; skipping co2 tracking"

CODECARBON_START_FAILED = "Failed to set-up CO2 logger"

CODECARBON_STOP_FAILED = "Failed to shutdown the CO2Tracker properly"

CODECARBON_DIR_CREATION_FAILED = "Failed to create the temporary CO2 tracker directory %r; skipping CO2 tracking; error %r"

CONFUSION_MATRIX_INDEX_TO_EXAMPLE_ERROR = (
    "%r failed for index %s; example not generated"
)
CONFUSION_MATRIX_EXAMPLE_NONE = "%r returned None for index %s; example not generated"

CONFUSION_MATRIX_EXAMPLE_DICT_INVALID_FORMAT = "%r returned an invalid dict for index %s, must match {'sample': ..., 'assetId': ...}"

CONFUSION_MATRIX_EXAMPLE_INVALID_TYPE = "%r returned invalid type %r at index %s, must return an int, string, URL, or {'sample': string, 'assetId': string}"

MISSING_PANDAS_LOG_DATAFRAME = "pandas is required to log a dataframe; ignoring"

MISSING_PANDAS_PROFILING = "pandas_profiling is required to log profile; ignoring"

NOT_PANDAS_DATAFRAME = "dataframe must be a pandas DataFrame; ignoring"

DATAFRAME_PROFILE_ERROR = "unable to profile dataframe; ignoring"

LOG_CODE_CALLER_NOT_FOUND = "unable to find caller source code; ignoring"

LOG_CODE_CALLER_JUPYTER = (
    "unable to find caller source code in a jupyter notebook; ignoring"
)

LOG_CODE_FILE_NAME_FOLDER_MUTUALLY_EXCLUSIVE = (
    "Experiment.log_code either accept file_name or folder, not both; ignoring"
)
