# -*- coding: utf-8 -*
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Olivier Cornelis
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Terminal color text utilities.
"""

# Import
import logging
from tqdm import tqdm
from termcolor import colored


def print_white(string, *args):
    string = str(string)
    if args:
        string = string + "".join([str(a) for a in args])
    tqdm.write(colored(string, "white"))
    log = logging.getLogger("benchmarcotte")
    log.info(string)


def print_cyan(string, *args):
    string = str(string)
    if args:
        string = string + "".join([str(a) for a in args])
    tqdm.write(colored(string, "cyan"))
    log = logging.getLogger("benchmarcotte")
    log.info(string)


def print_yellow(string, *args):
    string = str(string)
    if args:
        string = string + "".join([str(a) for a in args])
    tqdm.write(colored(string, "yellow"))
    log = logging.getLogger("benchmarcotte")
    log.info(string)


def print_red(string, *args):
    string = str(string)
    if args:
        string = string + "".join([str(a) for a in args])
    tqdm.write(colored(string, "red"))
    log = logging.getLogger("benchmarcotte")
    log.info(string)


def print_green(string, *args):
    string = str(string)
    if args:
        string = string + "".join([str(a) for a in args])
    tqdm.write(colored(string, "green"))
    log = logging.getLogger("benchmarcotte")
    log.info(string)


def print_error(string, *args):
    string = str(string)
    if args:
        string = string + "".join([str(a) for a in args])
    tqdm.write(colored(string, "cyan", "on_red"))
    log = logging.getLogger("benchmarcotte")
    log.error(string)


def print_tip(string, *args):
    string = str(string)
    if args:
        string = string + "".join([str(a) for a in args])
    tqdm.write(colored(string, "magenta"))
    log = logging.getLogger("benchmarcotte")
    log.info(string)
