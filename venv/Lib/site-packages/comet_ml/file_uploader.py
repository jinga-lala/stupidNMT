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

""" This module handles syncing git repos with the backend. Used for pull
request features."""

import io
import json
import logging
import os
import shutil
import tempfile
import threading
import zipfile

import six
from requests import Response

from ._reporting import FILE_UPLOADED_FAILED
from ._typing import (
    IO,
    Any,
    Dict,
    Optional,
    TemporaryFilePath,
    Union,
    UserText,
    ValidFilePath,
)
from .config import Config, get_config
from .connection import Reporting, get_http_session
from .exceptions import FileIsTooBig
from .logging_messages import (
    LOG_AUDIO_TOO_BIG,
    LOG_FIGURE_TOO_BIG,
    LOG_IMAGE_TOO_BIG,
    UPLOAD_ASSET_TOO_BIG,
    UPLOAD_FILE_OS_ERROR,
)
from .messages import UploadFileMessage, UploadInMemoryMessage
from .utils import (
    data_to_fp,
    encode_metadata,
    get_file_extension,
    image_data_to_file_like_object,
    write_file_like_to_tmp_file,
    write_numpy_array_as_wav,
)

LOGGER = logging.getLogger(__name__)

try:
    import numpy
except ImportError:
    LOGGER.warning("numpy not installed; some functionality will be unavailable")
    pass


def compress_git_patch(git_patch):
    # Create a zip
    zip_dir = tempfile.mkdtemp()

    zip_path = os.path.join(zip_dir, "patch.zip")
    archive = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    archive.writestr("git_diff.patch", git_patch)
    archive.close()

    return archive, zip_path


def _send_file(url, files, params, experiment_id, config):
    # type: (str, Any, Dict[str, str], str, Config) -> Response
    LOGGER.debug("Uploading files %r to %s with params %s", files, url, params)

    with get_http_session() as session:
        r = session.post(
            url,
            params=params,
            files=files,
            timeout=config["comet.timeout.file_upload"],
            headers={"X-COMET-DEBUG-EXPERIMENT-KEY": experiment_id},
        )  # type: Response

    LOGGER.debug("Uploading file to %s done", url)

    if r.status_code != 200:
        raise ValueError(
            "POSTing file failed (%s) on url %r: %r" % (r.status_code, url, r.content)
        )

    return r


def send_file(
    post_endpoint,
    api_key,
    experiment_id,
    project_id,
    file_path,
    config,
    additional_params=None,
    metadata=None,
):
    # type: (str, str, str, str, str, Config, Optional[Dict[str, str]], Optional[Dict[Any, Any]]) -> Response
    params = {"experimentId": experiment_id, "projectId": project_id, "apiKey": api_key}

    if additional_params is not None:
        params.update(additional_params)

    with open(file_path, "rb") as _file:
        files = {"file": _file}  # type: Dict[str, Any]

        if metadata is not None:
            encoded_metadata = encode_metadata(metadata)
            if encoded_metadata:
                files["metadata"] = encoded_metadata

        return _send_file(
            post_endpoint,
            params=params,
            files=files,
            experiment_id=experiment_id,
            config=config,
        )


def send_file_like(
    post_endpoint,
    api_key,
    experiment_id,
    project_id,
    file_like,
    config,
    additional_params=None,
    metadata=None,
):
    # type: (str, str, str, str, Any, Config, Optional[Dict[str, str]], Optional[Dict[Any, Any]]) -> Response
    params = {"experimentId": experiment_id, "projectId": project_id, "apiKey": api_key}

    if additional_params is not None:
        params.update(additional_params)

    files = {"file": file_like}
    if metadata is not None:
        encoded_metadata = encode_metadata(metadata)
        if encoded_metadata:
            files["metadata"] = encoded_metadata

    return _send_file(
        post_endpoint,
        params=params,
        files=files,
        experiment_id=experiment_id,
        config=config,
    )


def upload_file(
    project_id,
    experiment_id,
    file_path,
    upload_endpoint,
    api_key,
    additional_params=None,
    metadata=None,
    clean=True,
):
    config = get_config()

    try:
        response = send_file(
            upload_endpoint,
            api_key,
            experiment_id,
            project_id,
            file_path,
            additional_params=additional_params,
            metadata=metadata,
            config=config,
        )

        if clean is True:
            # Cleanup file
            try:
                os.remove(file_path)
            except OSError:
                pass

        LOGGER.debug(
            "File successfully uploaded to (%s): %s",
            response.status_code,
            upload_endpoint,
        )
    except Exception as e:
        LOGGER.error("File could not be uploaded", exc_info=True)
        Reporting.report(
            event_name=FILE_UPLOADED_FAILED,
            experiment_key=experiment_id,
            project_id=project_id,
            api_key=api_key,
            err_msg=str(e),
            config=config,
        )


def upload_file_like(
    project_id,
    experiment_id,
    file_like,
    upload_endpoint,
    api_key,
    additional_params=None,
    metadata=None,
):
    config = get_config()

    try:
        response = send_file_like(
            upload_endpoint,
            api_key,
            experiment_id,
            project_id,
            file_like,
            config,
            additional_params=additional_params,
            metadata=metadata,
        )

        LOGGER.debug(
            "File-like successfully uploaded to (%s): %s",
            response.status_code,
            upload_endpoint,
        )
    except Exception as e:
        LOGGER.error("File-like could not be uploaded", exc_info=True)
        Reporting.report(
            event_name=FILE_UPLOADED_FAILED,
            experiment_key=experiment_id,
            project_id=project_id,
            api_key=api_key,
            err_msg=str(e),
            config=config,
        )


def upload_file_thread(*args, **kwargs):
    p = threading.Thread(target=upload_file, args=args, kwargs=kwargs)
    p.daemon = True
    p.start()
    return p


def upload_file_like_thread(*args, **kwargs):
    p = threading.Thread(target=upload_file_like, args=args, kwargs=kwargs)
    p.daemon = True
    p.start()
    return p


def is_valid_file_path(file_path):
    # type: (Any) -> bool
    """ Check if the given argument is corresponding to a valid file path,
    ready for reading
    """
    try:
        if os.path.isfile(file_path):
            return True
        else:
            return False
    # We can receive lots of things as arguments
    except (TypeError, ValueError):
        return False


def is_user_text(input):
    # type: (Any) -> bool
    return isinstance(input, (six.string_types, bytes))


# Requests accepts either a file-object (IO, StringIO and BytesIO), a file path, string.
# We also accepts specific inputs for each logging method


def check_max_file_size(file_path, max_upload_size, too_big_msg):
    """ Check if a file identified by its file path is bigger than the maximum
    allowed upload size. Raises FileIsTooBig if the file is greater than the
    upload limit.
    """

    # Check the file size before reading it
    try:
        file_size = os.path.getsize(file_path)
        if file_size > max_upload_size:
            raise FileIsTooBig(file_path, file_size, max_upload_size)

    except OSError:
        LOGGER.error(too_big_msg, file_path, exc_info=True)
        raise


def save_matplotlib_figure(figure=None):
    # type: (Optional[Any]) -> str
    """ Try saving either the current global pyplot figure or the given one
    and return None in case of error.
    """
    # Get the right figure to upload
    if figure is None:
        import matplotlib.pyplot

        # Get current global figure
        figure = matplotlib.pyplot.gcf()

    if hasattr(figure, "gcf"):
        # The pyplot module was passed as figure
        figure = figure.gcf()

    if figure.get_axes():
        # Save the file to a tempfile but don't delete it, the file uploader
        # thread will take care of it
        tmpfile = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
        figure.savefig(tmpfile, format="svg")
        tmpfile.flush()

        return tmpfile.name
    else:
        # TODO DISPLAY BETTER ERROR MSG
        msg = (
            "Refuse to upload empty figure, please call log_figure before calling show"
        )
        LOGGER.warning(msg)
        raise TypeError(msg)


class BaseUploadProcessor(object):

    TOO_BIG_MSG = ""
    UPLOAD_TYPE = ""

    def __init__(
        self,
        user_input,  # type: Any
        upload_limit,  # type: int
        url_params,  # type: Dict[str, Optional[Any]]
        metadata,  # type: Optional[Dict[str, str]]
        copy_to_tmp,  # type: bool
        error_message_identifier,  # type: Any
        tmp_dir,  # type: str
    ):
        # type: (...) -> None
        self.user_input = user_input
        self.url_params = url_params
        self.metadata = self._validate_metadata(metadata)
        self.upload_limit = upload_limit
        self.error_message_identifier = error_message_identifier
        self.tmp_dir = tmp_dir

        self.copy_to_tmp = copy_to_tmp

        LOGGER.debug("%r created with %r", self, self.__dict__)

    def process(self):
        if isinstance(self.user_input, ValidFilePath) or is_valid_file_path(
            self.user_input
        ):
            return self.process_upload_by_filepath(self.user_input)
        elif hasattr(self.user_input, "read"):  # Support Python 2 legacy StringIO
            return self.process_io_object(self.user_input)
        elif is_user_text(self.user_input):
            return self.process_user_text(self.user_input)
        else:
            return self.process_upload_to_be_converted(self.user_input)

    # Dispatched user input method, one method per supported type in general. By
    # default those methods raise an exception, implement them for supported
    # input type per upload type

    def process_upload_by_filepath(self, upload_filepath):
        # type: (ValidFilePath) -> Optional[UploadFileMessage]
        raise TypeError("Unsupported upload input %r" % type(upload_filepath))

    def process_upload_to_be_converted(self, user_input):
        # type: (Any) -> Union[None, UploadInMemoryMessage, UploadFileMessage]
        raise TypeError("Unsupported upload input %r" % type(user_input))

    def process_io_object(self, io_object):
        # type: (IO) -> Union[None, UploadInMemoryMessage, UploadFileMessage]
        raise TypeError("Unsupported upload input %r" % type(io_object))

    def process_user_text(self, user_text):
        # type: (UserText) -> Union[None, UploadInMemoryMessage, UploadFileMessage]
        raise TypeError("Unsupported upload input %r" % user_text)

    # Low-level common code, once we have either an IO object or a filepath to upload

    def _process_upload_by_filepath(self, user_filepath):
        # type: (ValidFilePath) -> Optional[UploadFileMessage]
        try:
            check_max_file_size(user_filepath, self.upload_limit, self.TOO_BIG_MSG)
        except FileIsTooBig as exc:
            if self.error_message_identifier is None:
                error_message_identifier = exc.file_path
            else:
                error_message_identifier = self.error_message_identifier

            LOGGER.error(
                self.TOO_BIG_MSG, error_message_identifier, exc.file_size, exc.max_size
            )
            return None
        except Exception:
            LOGGER.debug("Error while checking the file size", exc_info=True)
            return None

        upload_filepath = self._handle_in_memory_file_upload(user_filepath)

        # If we failed to copy the file, abort
        if not upload_filepath:
            return None

        LOGGER.debug(
            "File upload message %r, type %r, params %r",
            upload_filepath,
            self.UPLOAD_TYPE,
            self.url_params,
        )

        # Clean only temporary files
        if isinstance(upload_filepath, TemporaryFilePath):
            clean = True
        else:
            clean = False

        if self.copy_to_tmp and not isinstance(upload_filepath, TemporaryFilePath):
            LOGGER.warning(
                "File %s should have been copied to a temporary location but was not",
                upload_filepath,
            )

        upload_message = UploadFileMessage(
            upload_filepath,
            self.UPLOAD_TYPE,
            self.url_params,
            self.metadata,
            clean=clean,
        )

        return upload_message

    def _handle_in_memory_file_upload(self, upload_filepath):
        # type: (ValidFilePath) -> Union[None, ValidFilePath, TemporaryFilePath]
        # If we cannot remove the uploaded file or need the file content will
        # be frozen to the time the upload call is made, pass copy_to_tmp with
        # True value
        if self.copy_to_tmp is True and not isinstance(
            upload_filepath, TemporaryFilePath
        ):
            tmpfile = tempfile.NamedTemporaryFile(delete=False)
            LOGGER.debug(
                "Copying %s to %s because of copy_to_tmp", upload_filepath, tmpfile.name
            )
            try:
                shutil.copyfile(upload_filepath, tmpfile.name)
            except (OSError, IOError):
                LOGGER.error(UPLOAD_FILE_OS_ERROR, upload_filepath, exc_info=True)
                return None
            upload_filepath = TemporaryFilePath(tmpfile.name)

        return upload_filepath

    def _process_upload_io(self, io_object):
        # type: (IO) -> Union[None, UploadInMemoryMessage, UploadFileMessage]
        if self.copy_to_tmp:
            LOGGER.debug("Saving IO to tmp_file because of copy_to_tmp")
            # Convert the file-like to a temporary file on disk
            file_path = write_file_like_to_tmp_file(io_object, self.tmp_dir)
            self.copy_to_tmp = False

            # TODO it would be easier to use the same field name for a file or a figure upload
            if "fileName" in self.url_params and self.url_params["fileName"] is None:
                self.url_params["fileName"] = os.path.basename(file_path)

            if "figName" in self.url_params and self.url_params["figName"] is None:
                self.url_params["figName"] = os.path.basename(file_path)

            return self._process_upload_by_filepath(TemporaryFilePath(file_path))

        LOGGER.debug(
            "File-like upload message %r, type %r, params %r",
            io_object,
            self.UPLOAD_TYPE,
            self.url_params,
        )

        return UploadInMemoryMessage(
            io_object, self.UPLOAD_TYPE, self.url_params, self.metadata
        )

    def _process_upload_text(self, user_text):
        # type: (UserText) -> Union[None, UploadInMemoryMessage, UploadFileMessage]
        if self.copy_to_tmp:
            # TODO: Be more efficient here
            io_object = data_to_fp(user_text)

            if not io_object:
                # We couldn't convert to an io_object
                return None

            file_path = write_file_like_to_tmp_file(io_object, self.tmp_dir)

            return self._process_upload_by_filepath(TemporaryFilePath(file_path))

        LOGGER.debug(
            "Text upload message %r, type %r, params %r",
            user_text,
            self.UPLOAD_TYPE,
            self.url_params,
        )

        return UploadInMemoryMessage(
            user_text, self.UPLOAD_TYPE, self.url_params, self.metadata
        )

    @staticmethod
    def _validate_metadata(user_metadata):
        # type: (Any) -> Dict[Any, Any]

        if user_metadata is None:
            return {}

        if type(user_metadata) is not dict:
            LOGGER.warning("Invalid metadata, expecting dict type", exc_info=True)
            return {}

        result = user_metadata  # type: Dict[Any, Any]

        return result


class AssetUploadProcessor(BaseUploadProcessor):

    TOO_BIG_MSG = UPLOAD_ASSET_TOO_BIG

    def __init__(
        self,
        user_input,  # type: Any
        upload_type,  # type: str
        url_params,  # type: Dict[str, Optional[Any]]
        metadata,  # type: Optional[Dict[str, str]]
        upload_limit,  # type: int
        copy_to_tmp,  # type: bool
        error_message_identifier,  # type: Any
        tmp_dir,  # type: str
    ):
        # type: (...) -> None
        self.UPLOAD_TYPE = upload_type

        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

        super(AssetUploadProcessor, self).__init__(
            user_input,
            upload_limit,
            url_params,
            metadata,
            copy_to_tmp,
            error_message_identifier,
            tmp_dir,
        )

    def process_upload_by_filepath(self, upload_filepath):
        # type: (ValidFilePath) -> Optional[UploadFileMessage]

        if self.url_params["fileName"] is None:
            self.url_params["fileName"] = os.path.basename(upload_filepath)

        self.url_params["extension"] = get_file_extension(upload_filepath)

        return self._process_upload_by_filepath(upload_filepath)

    def process_io_object(self, io_object):
        # type: (IO) -> Union[None, UploadInMemoryMessage, UploadFileMessage]
        extension = get_file_extension(self.url_params["fileName"])
        if extension is not None:
            self.url_params["extension"] = extension

        return self._process_upload_io(io_object)

    def process_user_text(self, user_text):
        # type: (UserText) -> Union[None, UploadInMemoryMessage, UploadFileMessage]
        LOGGER.error(UPLOAD_FILE_OS_ERROR, user_text)
        return None


class FigureUploadProcessor(BaseUploadProcessor):

    TOO_BIG_MSG = LOG_FIGURE_TOO_BIG
    UPLOAD_TYPE = "visualization"

    def __init__(
        self,
        user_input,  # type: Any
        upload_limit,  # type: int
        url_params,  # type: Dict[str, Optional[Any]]
        metadata,  # type: Optional[Dict[str, str]]
        copy_to_tmp,  # type: bool
        error_message_identifier,  # type: Any
        tmp_dir,  # type: str
        upload_type=None,  # type: Optional[str]
    ):
        super(FigureUploadProcessor, self).__init__(
            user_input,
            upload_limit,
            url_params,
            metadata,
            copy_to_tmp,
            error_message_identifier,
            tmp_dir,
        )
        if upload_type is not None:
            self.UPLOAD_TYPE = upload_type

    def process_upload_to_be_converted(self, user_input):
        # type: (Any) -> Optional[UploadFileMessage]
        try:
            filename = save_matplotlib_figure(user_input)
        except Exception:
            LOGGER.warning("Failing to save the matplotlib figure", exc_info=True)
            # An error occurred
            return None

        return self._process_upload_by_filepath(TemporaryFilePath(filename))


class ImageUploadProcessor(BaseUploadProcessor):

    TOO_BIG_MSG = LOG_IMAGE_TOO_BIG
    UPLOAD_TYPE = "visualization"

    def __init__(
        self,
        user_input,  # type: Any
        name,  # type: Optional[str]
        overwrite,  # type: bool
        image_format,
        image_scale,
        image_shape,
        image_colormap,
        image_minmax,
        image_channels,
        upload_limit,  # type: int
        url_params,  # type: Dict[str, Optional[Any]]
        metadata,  # type: Optional[Dict[str, str]]
        copy_to_tmp,  # type: bool
        error_message_identifier,  # type: Any
        tmp_dir,
    ):
        # type: (...) -> None
        self.name = name
        self.image_format = image_format
        self.image_scale = image_scale
        self.image_shape = image_shape
        self.image_colormap = image_colormap
        self.image_minmax = image_minmax
        self.image_channels = image_channels
        super(ImageUploadProcessor, self).__init__(
            user_input,
            upload_limit,
            url_params,
            metadata,
            copy_to_tmp,
            error_message_identifier,
            tmp_dir,
        )

    def process_upload_by_filepath(self, upload_filepath):
        # type: (ValidFilePath) -> Optional[UploadFileMessage]

        if self.url_params["figName"] is None:
            self.url_params["figName"] = os.path.basename(upload_filepath)

        self.url_params["extension"] = get_file_extension(upload_filepath)

        return self._process_upload_by_filepath(upload_filepath)

    def process_upload_to_be_converted(self, user_input):
        # type: (Any) -> Union[None, UploadInMemoryMessage, UploadFileMessage]
        try:
            image_object = image_data_to_file_like_object(
                user_input,
                self.name,
                self.image_format,
                self.image_scale,
                self.image_shape,
                self.image_colormap,
                self.image_minmax,
                self.image_channels,
            )
        except Exception:
            LOGGER.error(
                "Could not convert image_data into an image; ignored", exc_info=True
            )
            return None

        if not image_object:
            LOGGER.error(
                "Could not convert image_data into an image; ignored", exc_info=True
            )
            return None

        return self._process_upload_io(image_object)

    def process_io_object(self, io_object):
        # type: (IO) -> Union[None, UploadInMemoryMessage, UploadFileMessage]
        extension = get_file_extension(self.name)
        if extension is not None:
            self.url_params["extension"] = extension

        return self._process_upload_io(io_object)

    def process_user_text(self, user_text):
        # type: (UserText) -> None
        LOGGER.error(UPLOAD_FILE_OS_ERROR, user_text)
        return None


class AudioUploadProcessor(BaseUploadProcessor):

    TOO_BIG_MSG = LOG_AUDIO_TOO_BIG
    UPLOAD_TYPE = "audio"

    def __init__(
        self,
        user_input,  # type: Any
        sample_rate,  # type: Optional[int]
        overwrite,  # type: bool
        upload_limit,  # type: int
        url_params,  # type: Dict[str, Optional[Any]]
        metadata,  # type: Optional[Dict[str, str]]
        copy_to_tmp,  # type: bool
        error_message_identifier,  # type: Any
        tmp_dir,  # type: str
    ):
        # type: (...) -> None
        self.sample_rate = sample_rate

        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

        super(AudioUploadProcessor, self).__init__(
            user_input,
            upload_limit,
            url_params,
            metadata,
            copy_to_tmp,
            error_message_identifier,
            tmp_dir,
        )

    def process_upload_by_filepath(self, upload_filepath):
        # type: (ValidFilePath) -> Optional[UploadFileMessage]
        if self.url_params["fileName"] is None:
            self.url_params["fileName"] = os.path.basename(upload_filepath)

        self.url_params["extension"] = get_file_extension(upload_filepath)

        # The file has not been sampled
        self.url_params["sampleRate"] = None

        return self._process_upload_by_filepath(upload_filepath)

    def process_upload_to_be_converted(self, user_input):
        # type: (Any) -> Union[None, UploadInMemoryMessage, UploadFileMessage]

        try:
            if not isinstance(user_input, numpy.ndarray):
                raise TypeError("Unsupported audio_data type %r" % type(user_input))
        except NameError:
            # Numpy is not available
            raise TypeError("Numpy is needed when passing a numpy array to log_audio")

        extension = get_file_extension(self.url_params["fileName"])
        if extension is not None:
            self.url_params["extension"] = extension

        if self.sample_rate is None:
            raise TypeError("sample_rate cannot be None when logging a numpy array")

        if not self.sample_rate:
            raise TypeError("sample_rate cannot be 0 when logging a numpy array")

        # Send the sampling rate to the backend
        self.url_params["sampleRate"] = self.sample_rate

        # And save it in the metadata too
        self.metadata["sample_rate"] = self.sample_rate

        # Write to a file directly to avoid temporary IO copy when we know it
        # will ends up on the file-system anyway
        if self.copy_to_tmp:
            tmpfile = tempfile.NamedTemporaryFile(delete=False)

            write_numpy_array_as_wav(user_input, self.sample_rate, tmpfile)

            return self._process_upload_by_filepath(TemporaryFilePath(tmpfile.name))
        else:
            io_object = io.BytesIO()

            write_numpy_array_as_wav(user_input, self.sample_rate, io_object)

            return self._process_upload_io(io_object)

    def process_user_text(self, user_text):
        # type: (UserText) -> None
        LOGGER.error(UPLOAD_FILE_OS_ERROR, user_text)
        return None


class AssetDataUploadProcessor(BaseUploadProcessor):

    TOO_BIG_MSG = UPLOAD_ASSET_TOO_BIG

    def __init__(
        self,
        user_input,  # type: Any
        upload_type,  # type: str
        url_params,  # type: Dict[str, Optional[Any]]
        metadata,  # type: Optional[Dict[str, str]]
        upload_limit,  # type: int
        copy_to_tmp,  # type: bool
        error_message_identifier,  # type: Any
        tmp_dir,  # type: str
    ):
        # type: (...) -> None
        self.UPLOAD_TYPE = upload_type
        super(AssetDataUploadProcessor, self).__init__(
            user_input,
            upload_limit,
            url_params,
            metadata,
            copy_to_tmp,
            error_message_identifier,
            tmp_dir,
        )

    def process_upload_to_be_converted(self, user_input):
        # type: (Any) -> Union[None, UploadInMemoryMessage, UploadFileMessage]
        # We have an object which is neither an IO object, neither a str or bytes
        try:
            converted = json.dumps(user_input)
        except Exception:
            LOGGER.error("Failed to log asset data as JSON", exc_info=True)
            return None

        extension = get_file_extension(self.url_params["fileName"])
        if extension is not None:
            self.url_params["extension"] = extension

        return self._process_upload_text(converted)

    def process_user_text(self, user_text):
        # type: (UserText) -> Union[None, UploadInMemoryMessage, UploadFileMessage]
        extension = get_file_extension(self.url_params["fileName"])
        if extension is not None:
            self.url_params["extension"] = extension

        return self._process_upload_text(user_text)


class GitPatchUploadProcessor(BaseUploadProcessor):

    TOO_BIG_MSG = UPLOAD_ASSET_TOO_BIG
    UPLOAD_TYPE = "git-patch"

    def process_upload_by_filepath(self, upload_filepath):
        # type: (ValidFilePath) -> Optional[UploadFileMessage]
        return self._process_upload_by_filepath(upload_filepath)
