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

"""comet-ml"""
from __future__ import print_function

import logging
import traceback

from ._logging import (
    ADD_SYMLINK_ERROR,
    ADD_TAGS_ERROR,
    EXPERIMENT_LIVE,
    INTERNET_CONNECTION_ERROR,
    INVALID_API_KEY,
    REGISTER_RPC_FAILED,
    SEND_NOTIFICATION_FAILED,
    setup,
    setup_http_handler,
)
from ._reporting import EXPERIMENT_CREATED, EXPERIMENT_CREATION_FAILED
from ._typing import Any, Dict, Optional
from .api import API, APIExperiment
from .comet import Streamer, format_url, generate_guid, is_valid_experiment_key
from .config import (  # noqa
    get_api_key,
    get_config,
    get_global_experiment,
    get_previous_experiment,
    get_ws_url,
)
from .confusion_matrix import ConfusionMatrix
from .connection import (
    INITIAL_BEAT_DURATION,
    RestServerConnection,
    WebSocketConnection,
    get_backend_address,
    get_comet_api_client,
    get_rest_api_client,
    log_url,
)
from .exceptions import BadCallbackArguments, InvalidAPIKey
from .experiment import BaseExperiment
from .feature_toggles import HTTP_LOGGING, FeatureToggles
from .json_encoder import NestedEncoder
from .loggers.fastai_logger import patch as fastai_patch
from .loggers.keras_logger import patch as keras_patch
from .loggers.lightgbm_logger import patch as lgbm_patch
from .loggers.mlflow_logger import patch as mlflow_patch
from .loggers.pytorch_logger import patch as pytorch_patch
from .loggers.shap_logger import patch as shap_patch
from .loggers.sklearn_logger import patch as sklearn_patch
from .loggers.tensorboard_logger import patch as tb_patch
from .loggers.tensorflow_logger import patch as tf_patch
from .loggers.tfma_logger import patch as tfma_patch
from .loggers.xgboost_logger import patch as xg_patch
from .logging_messages import ONLINE_EXPERIMENT_THROTTLED
from .monkey_patching import CometModuleFinder
from .offline import OfflineExperiment
from .optimizer import Optimizer
from .rpc import create_remote_call, get_remote_action_definition
from .utils import Embedding, Histogram, get_comet_version, merge_url, valid_ui_tabs

__author__ = "Gideon<Gideon@comet.ml>"
__all__ = [
    "API",
    "APIExperiment",
    "ConfusionMatrix",
    "Embedding",
    "ExistingExperiment",
    "Experiment",
    "get_comet_api_client",
    "Histogram",
    "OfflineExperiment",
    "Optimizer",
]
__version__ = get_comet_version()

LOGGER = logging.getLogger(__name__)

if not get_config("comet.disable_auto_logging"):
    # Activate the monkey patching
    MODULE_FINDER = CometModuleFinder()
    keras_patch(MODULE_FINDER)
    sklearn_patch(MODULE_FINDER)
    tf_patch(MODULE_FINDER)
    tb_patch(MODULE_FINDER)
    pytorch_patch(MODULE_FINDER)
    fastai_patch(MODULE_FINDER)
    mlflow_patch(MODULE_FINDER)
    xg_patch(MODULE_FINDER)
    tfma_patch(MODULE_FINDER)
    shap_patch(MODULE_FINDER)
    lgbm_patch(MODULE_FINDER)
    MODULE_FINDER.start()

# Configure the logging
setup(get_config())


def start():
    """
    If you are not using an Experiment in your first loaded Python file, you
    must import `comet_ml` and call `comet_ml.start` before any other imports
    to ensure that comet.ml is initialized correctly.
    """


class Experiment(BaseExperiment):
    """
    Experiment is a unit of measurable research that defines a single run with some data/parameters/code/results.

    Creating an Experiment object in your code will report a new experiment to your Comet.ml project. Your Experiment
    will automatically track and collect many things and will also allow you to manually report anything.

    You can create multiple objects in one script (such as when looping over multiple hyper parameters).

    """

    def __init__(
        self,
        api_key=None,  # type: Optional[str]
        project_name=None,  # type: Optional[str]
        workspace=None,  # type: Optional[str]
        log_code=True,  # type: Optional[bool]
        log_graph=True,  # type: Optional[bool]
        auto_param_logging=True,  # type: Optional[bool]
        auto_metric_logging=True,  # type: Optional[bool]
        parse_args=True,  # type: Optional[bool]
        auto_output_logging="default",  # type: Optional[str]
        log_env_details=True,  # type: Optional[bool]
        log_git_metadata=True,  # type: Optional[bool]
        log_git_patch=True,  # type: Optional[bool]
        disabled=False,  # type: Optional[bool]
        log_env_gpu=True,  # type: Optional[bool]
        log_env_host=True,  # type: Optional[bool]
        display_summary=None,  # type: Optional[bool]
        log_env_cpu=True,  # type: Optional[bool]
        display_summary_level=1,  # type: Optional[int]
        optimizer_data=None,  # type: Optional[Dict[str, Any]]
        auto_weight_logging=None,  # type: Optional[bool]
        auto_log_co2=True,  # type: Optional[bool]
        auto_metric_step_rate=10,  # type: Optional[int]
        auto_histogram_tensorboard_logging=False,  # type: Optional[bool]
        auto_histogram_epoch_rate=1,  # type: Optional[int]
        auto_histogram_weight_logging=False,  # type: Optional[bool]
        auto_histogram_gradient_logging=False,  # type: Optional[bool]
        auto_histogram_activation_logging=False,  # type: Optional[bool]
    ):
        """
        Creates a new experiment on the Comet.ml frontend.
        Args:
            api_key: Your API key obtained from comet.ml
            project_name: Optional. Send your experiment to a specific project. Otherwise will be sent to `Uncategorized Experiments`.
                             If project name does not already exists Comet.ml will create a new project.
            workspace: Optional. Attach an experiment to a project that belongs to this workspace
            log_code: Default(True) - allows you to enable/disable code logging
            log_graph: Default(True) - allows you to enable/disable automatic computation graph logging.
            auto_param_logging: Default(True) - allows you to enable/disable hyper parameters logging
            auto_metric_logging: Default(True) - allows you to enable/disable metrics logging
            auto_metric_step_rate: Default(10) - controls how often batch metrics are logged
            auto_histogram_tensorboard_logging: Default(False) - allows you to enable/disable automatic tensorboard histogram logging
            auto_histogram_epoch_rate: Default(1) - controls how often histograms are logged
            auto_histogram_weight_logging: Default(False) - allows you to enable/disable histogram logging for biases and weights
            auto_histogram_gradient_logging: Default(False) - allows you to enable/disable automatic histogram logging of gradients
            auto_histogram_activation_logging: Default(False) - allows you to enable/disable automatic histogram logging of activations
            auto_output_logging: Default("default") - allows you to select
                which output logging mode to use. You can pass `"native"`
                which will log all output even when it originated from a C
                native library. You can also pass `"simple"` which will work
                only for output made by Python code. If you want to disable
                automatic output logging, you can pass `False`. The default is
                `"default"` which will detect your environment and deactivate
                the output logging for IPython and Jupyter environment and
                sets `"native"` in the other cases.
            auto_log_co2: Default(True) - automatically tracks the CO2 emission of
                this experiment if `codecarbon` package is installed in the environment
            parse_args: Default(True) - allows you to enable/disable automatic parsing of CLI arguments
            log_env_details: Default(True) - log various environment
                information in order to identify where the script is running
            log_env_gpu: Default(True) - allow you to enable/disable the
                automatic collection of gpu details and metrics (utilization, memory usage etc..).
                `log_env_details` must also be true.
            log_env_cpu: Default(True) - allow you to enable/disable the
                automatic collection of cpu details and metrics (utilization, memory usage etc..).
                `log_env_details` must also be true.
            log_env_host: Default(True) - allow you to enable/disable the
                automatic collection of host information (ip, hostname, python version, user etc...).
                `log_env_details` must also be true.
            log_git_metadata: Default(True) - allow you to enable/disable the
                automatic collection of git details
            log_git_patch: Default(True) - allow you to enable/disable the
                automatic collection of git patch
            display_summary_level: Default(1) - control the summary detail that is
                displayed on the console at end of experiment. If 0, the summary
                notification is still sent. Valid values are 0 to 2.
            disabled: Default(False) - allows you to disable all network
                communication with the Comet.ml backend. It is useful when you
                want to test to make sure everything is working, without actually
                logging anything.
        """
        self.config = get_config()

        self.api_key = get_api_key(api_key, self.config)

        if self.api_key is None:
            raise ValueError(
                "Comet.ml requires an API key. Please provide as the "
                "first argument to Experiment(api_key) or as an environment"
                " variable named COMET_API_KEY "
            )

        super(Experiment, self).__init__(
            project_name=project_name,
            workspace=workspace,
            log_code=log_code,
            log_graph=log_graph,
            auto_param_logging=auto_param_logging,
            auto_metric_logging=auto_metric_logging,
            parse_args=parse_args,
            auto_output_logging=auto_output_logging,
            log_env_details=log_env_details,
            log_git_metadata=log_git_metadata,
            log_git_patch=log_git_patch,
            disabled=disabled,
            log_env_gpu=log_env_gpu,
            log_env_host=log_env_host,
            display_summary=display_summary,  # deprecated
            display_summary_level=display_summary_level,
            log_env_cpu=log_env_cpu,
            optimizer_data=optimizer_data,
            auto_weight_logging=auto_weight_logging,  # deprecated
            auto_log_co2=auto_log_co2,
            auto_metric_step_rate=auto_metric_step_rate,
            auto_histogram_tensorboard_logging=auto_histogram_tensorboard_logging,
            auto_histogram_epoch_rate=auto_histogram_epoch_rate,
            auto_histogram_weight_logging=auto_histogram_weight_logging,
            auto_histogram_gradient_logging=auto_histogram_gradient_logging,
            auto_histogram_activation_logging=auto_histogram_activation_logging,
        )

        self.ws_connection = None
        self.connection = None  # type: Optional[RestServerConnection]
        self.rest_api_client = None

        if self.disabled is not True:
            self._start()

            if self.alive is True:
                self._report(event_name=EXPERIMENT_CREATED)

                LOGGER.info(EXPERIMENT_LIVE, self._get_experiment_url())

    def _setup_http_handler(self):
        if not self.feature_toggles[HTTP_LOGGING]:
            LOGGER.debug("Do not setup http logger, disabled by feature toggle")
            return

        self.http_handler = setup_http_handler(
            log_url(get_backend_address()), self.api_key, self.id
        )

    def _setup_streamer(self):
        """
        Do the necessary work to create mandatory objects, like the streamer
        and feature flags
        """

        server_address = get_backend_address()

        self.connection = RestServerConnection(
            self.api_key, self.id, server_address, self.config["comet.timeout.http"]
        )
        self.rest_api_client = get_rest_api_client("v2", api_key=self.api_key)

        try:
            results = self._authenticate()
            if results:
                ws_server_from_backend, initial_offset = results
            else:
                ws_server_from_backend, initial_offset = None, None
        except ValueError:
            tb = traceback.format_exc()
            LOGGER.error(INTERNET_CONNECTION_ERROR, exc_info=True)
            self._report(event_name=EXPERIMENT_CREATION_FAILED, err_msg=tb)
            return False

        # Authentication failed somehow
        if ws_server_from_backend is None:
            return False

        ws_server = get_ws_url(ws_server_from_backend, self.config)

        # Generate the full ws url based on ws server
        full_ws_url = format_url(ws_server, apiKey=self.api_key, runId=self.run_id)

        # Setup the HTTP handler
        self._setup_http_handler()

        # Initiate the streamer
        self._initialize_streamer(full_ws_url, initial_offset)

        return True

    def _authenticate(self):
        """
        Do the handshake with the Backend to authenticate the api key and get
        various parameters and settings
        """
        # Get an id for this run
        run_id_results = self.connection.get_run_id(self.project_name, self.workspace)

        (
            self.run_id,
            ws_server,
            self.project_id,
            self.is_github,
            self.focus_link,
            self.upload_limit,
            self.asset_upload_limit,
            feature_toggles,
            initial_offset,
            self.upload_web_asset_url_prefix,
            self.upload_web_image_url_prefix,
            self.upload_api_asset_url_prefix,
            self.upload_api_image_url_prefix,
        ) = run_id_results

        self.feature_toggles = FeatureToggles(feature_toggles, self.config)

        return (ws_server, initial_offset)

    def _initialize_streamer(self, full_ws_url, initial_offset):
        """
        Initialize the streamer with the websocket url received during the
        handshake.
        """
        # Initiate the streamer
        self.ws_connection = WebSocketConnection(full_ws_url, self.connection)
        self.ws_connection.start()
        self.ws_connection.wait_for_connection()
        self.streamer = Streamer(
            self.ws_connection,
            INITIAL_BEAT_DURATION,
            self.connection,
            initial_offset,
            self.id,
            self.api_key,
            self.run_id,
            self.project_id,
            self.rest_api_client,
            self._on_pending_rpcs_callback,
            self.config["comet.timeout.cleaning"],
            self.config["comet.timeout.upload"],
        )

        # Start streamer thread.
        self.streamer.start()

    def _mark_as_started(self):
        try:
            self.connection.update_experiment_status(
                self.run_id, self.project_id, self.alive
            )
        except Exception:
            LOGGER.error("Failed to report experiment status", exc_info=True)

    def _mark_as_ended(self):
        if self.alive:
            try:
                self.connection.update_experiment_status(
                    self.run_id, self.project_id, False
                )
            except Exception:
                LOGGER.error("Failed to report experiment status", exc_info=True)

    def _report(self, *args, **kwargs):
        self.connection.report(*args, **kwargs)

    def _on_end(self, wait=True):
        """ Called when the Experiment is replaced by another one or at the
        end of the script
        """
        successful_clean = super(Experiment, self)._on_end(wait=wait)

        if not successful_clean:
            LOGGER.warning("Failed to log run in comet.ml")
        else:
            if self.alive:
                LOGGER.info(EXPERIMENT_LIVE, self._get_experiment_url())

            # If we didn't drain the streamer, don't close the websocket connection
            if self.ws_connection is not None and wait is True:
                self.ws_connection.close()
                LOGGER.debug("Waiting for WS connection to close")
                if wait is True:
                    ws_cleaned = self.ws_connection.wait_for_finish()
                    if ws_cleaned is True:
                        LOGGER.debug("Websocket connection clean successfully")
                    else:
                        LOGGER.debug("Websocket connection DIDN'T clean successfully")
                        successful_clean = False
                        self.ws_connection.force_close()

            if self.connection is not None:
                self.connection.close()

        # Display throttling message
        try:
            if self._check_experiment_throttled():
                LOGGER.warning(ONLINE_EXPERIMENT_THROTTLED)
        except Exception:
            LOGGER.debug("Failed to check experiment metadata", exc_info=True)

        return successful_clean

    def _check_experiment_throttled(self):
        experiment_metadata = self.rest_api_client.get_experiment_metadata(self.id)
        return experiment_metadata.get("throttle", False)

    @property
    def url(self):
        """
        Get the url of the experiment.

        Example:

        ```python
        >>> api_experiment.url
        "https://www.comet.ml/username/34637643746374637463476"
        ```
        """
        return self._get_experiment_url()

    def _get_experiment_url(self, tab=None):
        if self.focus_link:
            if tab:
                if tab in valid_ui_tabs():
                    return merge_url(
                        self.focus_link + self.id,
                        {"experiment-tab": valid_ui_tabs(tab)},
                    )
                else:
                    LOGGER.info("tab must be one of: %r", valid_ui_tabs(preferred=True))
            return self.focus_link + self.id

        return ""

    def _on_pending_rpcs_callback(self):
        """ Called by streamer when we have pending rpcs
        """
        LOGGER.debug("Checking pending rpcs")
        calls = self.connection.get_pending_rpcs()["remoteProcedureCalls"]

        LOGGER.debug("Got pending rpcs: %r", calls)
        for raw_call in calls:
            call = create_remote_call(raw_call)
            if call is None:
                continue
            self._add_pending_call(call)

    def _send_rpc_callback_result(
        self, call_id, remote_call_result, start_time, end_time
    ):
        # Send the result to the backend
        self.connection.send_rpc_result(
            call_id, remote_call_result, start_time, end_time
        )

    def create_symlink(self, project_name):
        """
        creates a symlink for this experiment in another project.
        The experiment will now be displayed in the project provided and the original project.

        Args:
            project_name: String. represents the project name. Project must exists.
        """
        try:
            if self.alive:
                self.connection.send_new_symlink(project_name)
        except Exception:
            LOGGER.warning(ADD_SYMLINK_ERROR, project_name, exc_info=True)

    def add_tag(self, tag):
        """
        Add a tag to the experiment. Tags will be shown in the dashboard.
        Args:
            tag: String. A tag to add to the experiment.
        """
        try:
            if self.alive:
                self.connection.add_tags([tag])

            super(Experiment, self).add_tag(tag)
        except Exception:
            LOGGER.warning(ADD_TAGS_ERROR, tag, exc_info=True)

    def add_tags(self, tags):
        """
        Add several tags to the experiment. Tags will be shown in the
        dashboard.
        Args:
            tag: List<String>. Tags list to add to the experiment.
        """
        try:
            if self.alive:
                self.connection.add_tags(tags)

            # If we successfully send them to the backend, save them locally
            super(Experiment, self).add_tags(tags)
        except Exception:
            LOGGER.warning(ADD_TAGS_ERROR, tags, exc_info=True)

    def register_callback(self, remote_action):
        """
        Register the remote_action passed as argument to be a RPC.
        Args:
            remote_action: Callable.
        """
        super(Experiment, self).register_callback(remote_action)

        try:
            remote_action_definition = get_remote_action_definition(remote_action)
        except BadCallbackArguments as exc:
            # Don't keep bad callbacks registered
            self.unregister_callback(remote_action)
            LOGGER.warning(str(exc), exc_info=True)
            return

        try:
            self._register_callback_remotely(remote_action_definition)
        except Exception:
            # Don't keep bad callbacks registered
            self.unregister_callback(remote_action)
            LOGGER.warning(
                REGISTER_RPC_FAILED, remote_action_definition["functionName"]
            )

    def _register_callback_remotely(self, remote_action_definition):
        self.connection.register_rpc(remote_action_definition)

    def send_notification(self, title, status=None, additional_data=None):
        # type: (str, Optional[str], Optional[Dict[str, Any]]) -> None
        """
        Send yourself a notification through email when an experiment
        ends.

        Args:
            title: str - the email subject.
            status: str - the final status of the experiment. Typically,
                something like "finished", "completed" or "aborted".
            additional_data: dict - a dictionary of key/values to notify.

        Note:
            In order to receive the notification, you need to have turned
            on Notifications in your Settings in the Comet user interface.

        If you wish to have the `additional_data` saved with the
        experiment, you should also call `Experiment.log_other()` with
        this data as well.

        This method uses the email address associated with your account.
        """
        try:
            name = self.others.get("Name")

            if additional_data is None:
                additional_data = {}

            self.connection.send_notification(
                title,
                status,
                name,
                self._get_experiment_url(),
                additional_data,
                custom_encoder=NestedEncoder,
            )

        except Exception:
            LOGGER.error(SEND_NOTIFICATION_FAILED, exc_info=True)

    def log_embedding(
        self,
        vectors,
        labels,
        image_data=None,
        image_size=None,
        image_preprocess_function=None,
        image_transparent_color=None,
        image_background_color_function=None,
        title="Comet Embedding",
        template_filename="template_projector_config.json",
        group=None,
    ):
        """
        Log a multi-dimensional dataset and metadata for viewing with
        Comet's Embedding Projector (experimental).

        Args:
            vectors: the tensors to visualize in 3D
            labels: labels for each tensor
            image_data: (optional) list of arrays or Images
            image_size: (optional, required if image_data is given) the size of each image
            image_preprocess_function: (optional) if image_data is an
                array, apply this function to each element first
            image_transparent_color: a (red, green, blue) tuple
            image_background_color_function: a function that takes an
                index, and returns a (red, green, blue) color tuple
            title: (optional) name of tensor
            template_filename: (optional) name of template JSON file

        See also: `Experiment._log_embedding_list()` and `comet_ml.Embedding`

        Example:

        ```python
        from comet_ml import Experiment

        import numpy as np
        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        def label_to_color(index):
            label = y_test[index]
            if label == 0:
                return (255, 0, 0)
            elif label == 1:
                return (0, 255, 0)
            elif label == 2:
                return (0, 0, 255)
            elif label == 3:
                return (255, 255, 0)
            elif label == 4:
                return (0, 255, 255)
            elif label == 5:
                return (128, 128, 0)
            elif label == 6:
                return (0, 128, 128)
            elif label == 7:
                return (128, 0, 128)
            elif label == 8:
                return (255, 0, 255)
            elif label == 9:
                return (255, 255, 255)

        experiment = Experiment(project_name="projector-embedding")

        experiment.log_embedding(
            vectors=x_test,
            labels=y_test,
            image_data=x_test,
            image_preprocess_function=lambda matrix: np.round(matrix/255,0) * 2,
            image_transparent_color=(0, 0, 0),
            image_size=(28, 28),
            image_background_color_function=label_to_color,
        )
        ```
        """
        if not self.alive:
            return None

        LOGGER.warning(
            "Logging embedding is experimental - the API and logged data are subject to change"
        )

        embedding = self._create_embedding(
            vectors,
            labels,
            image_data,
            image_size,
            image_preprocess_function,
            image_transparent_color,
            image_background_color_function,
            title,
        )

        if group is not None:
            self._embedding_groups[group].append(embedding)
            return embedding
        else:
            # Log the template:
            template = {"embeddings": [embedding.to_json()]}
            return self._log_asset_data(
                template, template_filename, asset_type="embeddings"
            )


class ExistingExperiment(Experiment):
    """Existing Experiment allows you to report information to an
    experiment that already exists on comet.ml and is not currently
    running. This is useful when your training and testing happen on
    different scripts.

    For example:

    train.py:
    ```
    exp = Experiment(api_key="my-key")
    score = train_model()
    exp.log_metric("train accuracy", score)
    ```

    Now obtain the experiment key from comet.ml. If it's not visible
    on your experiment table you can click `Customize` and add it as a
    column.


    test.py:
    ```
    exp = ExistingExperiment(api_key="my-key",
             previous_experiment="your experiment key from comet.ml")
    score = test_model()
    exp.log_metric("test accuracy", score)
    ```

    Alternatively, you can pass the api_key via an environment
    variable named `COMET_API_KEY` and the previous experiment id via
    an environment variable named `COMET_EXPERIMENT_KEY` and omit them
    from the ExistingExperiment constructor:

    ```
    exp = ExistingExperiment()
    score = test_model()
    exp.log_metric("test accuracy", score)
    ```

    """

    def __init__(self, api_key=None, previous_experiment=None, **kwargs):
        """
        Append to an existing experiment on the Comet.ml frontend.
        Args:
            api_key: Your API key obtained from comet.ml
            previous_experiment: Optional. Your experiment key from comet.ml, could be set through
                configuration as well.
            project_name: Optional. Send your experiment to a specific project. Otherwise will be sent to `Uncategorized Experiments`.
                             If project name does not already exists Comet.ml will create a new project.
            workspace: Optional. Attach an experiment to a project that belongs to this workspace
            log_code: Default(False) - allows you to enable/disable code logging
            log_graph: Default(False) - allows you to enable/disable automatic computation graph logging.
            auto_param_logging: Default(True) - allows you to enable/disable hyper parameters logging
            auto_metric_logging: Default(True) - allows you to enable/disable metrics logging
            auto_metric_step_rate: Default(10) - controls how often batch metrics are logged
            auto_histogram_tensorboard_logging: Default(False) - allows you to enable/disable automatic tensorboard histogram logging
            auto_histogram_epoch_rate: Default(1) - controls how often histograms are logged
            auto_histogram_weight_logging: Default(False) - allows you to enable/disable automatic histogram logging of biases and weights
            auto_histogram_gradient_logging: Default(False) - allows you to enable/disable automatic histogram logging of gradients
            auto_histogram_activation_logging: Default(False) - allows you to enable/disable automatic histogram logging of activations
            auto_output_logging: Default("default") - allows you to select
                which output logging mode to use. You can pass `"native"`
                which will log all output even when it originated from a C
                native library. You can also pass `"simple"` which will work
                only for output made by Python code. If you want to disable
                automatic output logging, you can pass `False`. The default is
                `"default"` which will detect your environment and deactivate
                the output logging for IPython and Jupyter environment and
                sets `"native"` in the other cases.
            auto_log_co2: Default(True) - automatically tracks the CO2 emission of
                this experiment if `codecarbon` package is installed in the environment
            parse_args: Default(False) - allows you to enable/disable automatic parsing of CLI arguments
            log_env_details: Default(False) - log various environment
                information in order to identify where the script is running
            log_env_gpu: Default(False) - allow you to enable/disable the
                automatic collection of gpu details and metrics (utilization, memory usage etc..).
                `log_env_details` must also be true.
            log_env_cpu: Default(False) - allow you to enable/disable the
                automatic collection of cpu details and metrics (utilization, memory usage etc..).
                `log_env_details` must also be true.
            log_env_host: Default(False) - allow you to enable/disable the
                automatic collection of host information (ip, hostname, python version, user etc...).
                `log_env_details` must also be true.
            log_git_metadata: Default(False) - allow you to enable/disable the
                automatic collection of git details
            log_git_patch: Default(False) - allow you to enable/disable the
                automatic collection of git patch
            display_summary_level: Default(1) - control the summary detail that is
                displayed on the console at end of experiment. If 0, the summary
                notification is still sent. Valid values are 0 to 2.
            disabled: Default(False) - allows you to disable all network
                communication with the Comet.ml backend. It is useful when you
                just needs to works on your machine-learning scripts and need
                to relaunch them several times at a time.

        Note: ExistingExperiment does not alter nor destroy previously
        logged information. To override or add to previous information
        you will have to set the appropriate following parameters to True:

        * log_code
        * log_graph
        * parse_args
        * log_env_details
        * log_git_metadata
        * log_git_patch
        * log_env_gpu
        * log_env_cpu
        * log_env_host

        For example, to continue to collect GPU information in an
        `ExistingExperiment` you will need to override these parameters:

        ```python
        >>> experiment = ExistingExperiment(
        ...                 log_env_details=True,
        ...                 log_env_gpu=True)
        ```
        """
        # Validate the previous experiment id
        self.config = get_config()

        self.previous_experiment = get_previous_experiment(
            previous_experiment, self.config
        )

        if not is_valid_experiment_key(self.previous_experiment):
            raise ValueError("Invalid experiment key: %s" % self.previous_experiment)

        # TODO: Document the parameter
        self.step_copy = kwargs.pop("step_copy", None)

        ## Defaults for ExistingExperiment:
        ## For now, don't destroy previous Experiment information by default:

        for (key, config_name, default) in [
            ("log_code", "comet.auto_log.code", False),
            ("log_graph", "comet.auto_log.graph", False),
            ("parse_args", "comet.auto_log.cli_arguments", False),
            ("log_env_details", "comet.auto_log.env_details", False),
            ("log_git_metadata", "comet.auto_log.git_metadata", False),
            ("log_git_patch", "comet.auto_log.git_patch", False),
            ("log_env_gpu", "comet.auto_log.env_gpu", False),
            ("log_env_cpu", "comet.auto_log.env_cpu", False),
            ("log_env_host", "comet.auto_log.env_host", False),
        ]:
            if key not in kwargs or kwargs[key] is None:
                kwargs[key] = self.config.get_bool(
                    None, config_name, default, not_set_value=None
                )

        super(ExistingExperiment, self).__init__(api_key, **kwargs)

    def _get_experiment_key(self):
        if self.step_copy is None:
            return self.previous_experiment
        else:
            return generate_guid()

    def _authenticate(self):
        """
        Do the handshake with the Backend to authenticate the api key and get
        various parameters and settings
        """
        # Get an id for this run
        try:
            if self.step_copy is None:
                run_id_response = self.connection.get_old_run_id(
                    self.previous_experiment
                )
            else:
                run_id_response = self.connection.copy_run(
                    self.previous_experiment, self.step_copy
                )

            (
                self.run_id,
                full_ws_url,
                self.project_id,
                self.is_github,
                self.focus_link,
                self.upload_limit,
                self.asset_upload_limit,
                feature_toggles,
                initial_offset,
                self.upload_web_asset_url_prefix,
                self.upload_web_image_url_prefix,
                self.upload_api_asset_url_prefix,
                self.upload_api_image_url_prefix,
            ) = run_id_response

            self.feature_toggles = FeatureToggles(feature_toggles, self.config)

            return (full_ws_url, initial_offset)

        except InvalidAPIKey as e:
            LOGGER.error(INVALID_API_KEY, e.api_key, exc_info=True)
            return

        except ValueError:
            LOGGER.error(INTERNET_CONNECTION_ERROR, exc_info=True)
            return

    def send_notification(self, *args, **kwargs):
        """
        With an `Experiment`, this method will send your a notification
        through email when an experiment ends. However, with an
        `ExistingExperiment` this method does nothing.
        """
        pass

    def get_name(self):
        """
        This functionality is not available in ExistingExperiment.
        Use APIExperiment() instead.
        """
        raise NotImplementedError(
            "get_name() is not available in ExistingExperiment; use APIExperiment instead"
        )
