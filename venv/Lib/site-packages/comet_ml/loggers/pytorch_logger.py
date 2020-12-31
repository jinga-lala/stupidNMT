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

import logging

import wrapt

from .._logging import check_module
from .._typing import Any
from ..experiment import BaseExperiment

LOGGER = logging.getLogger(__name__)


def _get_loss(loss_backward, experiment):
    # type: (Any, BaseExperiment) -> Any
    """ Returns the right loss tensor based on the loss tensor object where
    backward has been called. The right loss might be the unscaled one when
    using APEX or Pytorch 1.6 AMP
    """
    return experiment._localstorage.pytorch["amp_loss_mapping"].get(
        id(loss_backward), loss_backward
    )


def _add_amp_loss_mapping(scaled_loss, unscaled_loss, experiment):
    # type: (int, Any, BaseExperiment) -> None
    # First argument is the scaled_loss id
    experiment._localstorage.pytorch["amp_loss_mapping"][scaled_loss] = unscaled_loss


def _clean_amp_loss_mapping(scaled_loss, experiment):
    # type: (int, BaseExperiment) -> None
    # First argument is the scaled_loss id
    experiment._localstorage.pytorch["amp_loss_mapping"].pop(scaled_loss, None)


def tensor_backward(experiment, original, result, *args, **kwargs):
    # args[0] is self, the Tensor (loss):
    try:
        if experiment.curr_step is None:
            experiment.curr_step = 0
        else:
            experiment.curr_step += 1
        if experiment.log_graph:
            model = experiment._storage["torch"]["model"]
            if experiment.curr_step == 0 and model is not None:
                experiment._set_model_graph(model, framework="pytorch")
        if experiment.auto_metric_logging:
            arg_loss = args[0]

            ## Throttle report to every 10 batch updates:
            if experiment.curr_step % 10 == 0:
                loss = _get_loss(arg_loss, experiment)

                if len(loss.data.shape) == 0:
                    metric = loss.data.item()
                    experiment._log_metric(
                        "loss", metric, step=experiment.curr_step, framework="pytorch",
                    )
                else:
                    experiment._log_metric(
                        "loss",
                        loss.data.mean().item(),
                        step=experiment.curr_step,
                        framework="pytorch",
                    )

            # Clean the reference to the original unscaled loss to avoid leaking
            # memory
            _clean_amp_loss_mapping(id(arg_loss), experiment)
    except Exception:
        LOGGER.info("Failed to run Tensor.backward logger", exc_info=True)
    return result


def model_constructor(experiment, original, *args, **kwargs):
    ## Assume the first one is the model:
    try:
        model = experiment._storage["torch"]["model"]
        if model is None:
            experiment._storage["torch"]["model"] = args[1]
    except Exception:
        LOGGER.info("Failed to run Module.__init__ logger", exc_info=True)


class CallableWrapper(wrapt.ObjectProxy):
    def __init__(self, wrapped, original_loss, experiment):
        super(CallableWrapper, self).__init__(wrapped)
        self.original_loss = original_loss
        self.experiment = experiment
        self.scaled_loss = None

    def __enter__(self, *args, **kwargs):
        return_value = self.__wrapped__.__enter__(*args, **kwargs)

        try:
            self.scaled_loss = id(return_value)
            _add_amp_loss_mapping(self.scaled_loss, self.original_loss, self.experiment)
        except Exception:
            LOGGER.debug("Error in Apex amp.scale_loss __enter__", exc_info=True)

        return return_value

    def __exit__(self, *args, **kwargs):
        try:
            if self.scaled_loss:
                _clean_amp_loss_mapping(self.scaled_loss, self.experiment)
        except Exception:
            LOGGER.debug("Error in Apex amp.scale_loss __exit__", exc_info=True)

        return self.__wrapped__.__exit__(*args, **kwargs)


def scale_loss_hook(experiment, original, return_value, original_loss, *args, **kwargs):
    return CallableWrapper(return_value, original_loss, experiment)


def amp_scale_loss_hook(
    experiment, original, return_value, scaler, original_loss, *args, **kwargs
):
    # Save the original loss mapped to the scaled one so we can find it back after
    _add_amp_loss_mapping(id(return_value), original_loss, experiment)


def patch(module_finder):
    ## For testing:
    check_module("torch")

    ## For each backpropagation of the gradient:
    module_finder.register_after("torch.tensor", "Tensor.backward", tensor_backward)
    ## For each model constructor:
    module_finder.register_after(
        "torch.nn.modules.module", "Module.__init__", model_constructor
    )

    module_finder.register_after("apex.amp.handle", "scale_loss", scale_loss_hook)

    module_finder.register_after(
        "torch.cuda.amp.grad_scaler", "GradScaler.scale", amp_scale_loss_hook
    )


check_module("torch")
