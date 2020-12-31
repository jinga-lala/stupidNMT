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


def _in_jupyter_environment():
    # type: () -> bool
    """
    Check to see if code is running in a Jupyter environment,
    including jupyter notebook, lab, or console.
    """
    try:
        import IPython
    except Exception:
        return False

    ipy = IPython.get_ipython()
    if ipy is None or not hasattr(ipy, "kernel"):
        return False
    else:
        return True


def _in_ipython_environment():
    # type: () -> bool
    """
    Check to see if code is running in an IPython environment.
    """
    try:
        import IPython
    except Exception:
        return False

    ipy = IPython.get_ipython()
    if ipy is None:
        return False
    else:
        return True


def display_or_open_browser(url, clear=False, wait=True, new=0, autoraise=True):
    # type: (str, bool, bool, int, bool) -> None
    if _in_jupyter_environment():
        from IPython.display import display, IFrame, clear_output

        if clear:
            clear_output(wait=wait)
        display(IFrame(src=url, width="100%", height="800px"))
    else:
        import webbrowser

        webbrowser.open(url, new=new, autoraise=autoraise)
