import logging
import socket
import sys
import traceback

logger = logging.getLogger(__name__)


def _handle_unhandled_exception(*args):
    logger.error("Uncaught exception", exc_info=args)


_glog_level_to_abbr = {
    "DEBUG": "V",  # V is for VERBOSE in glog
    "INFO": "I",
    "WARNING": "W",
    "ERROR": "E",
    "CRITICAL": "C",
}


class _Formatter(logging.Formatter):
    def __init__(self, suffix):
        self.suffix = suffix

    def format(self, record):
        message = record.getMessage()
        asctime = self.formatTime(record, "%m%d %H:%M:%S")

        lines = message.strip().split("\n")
        if record.exc_info:
            exc_info = self.formatException(record.exc_info)
            lines.extend(exc_info.strip().split("\n"))
        if record.stack_info:
            stack_info = self.formatStack(record.stack_info)
            lines.extend(stack_info.strip().split("\n"))

        shortlevel = _glog_level_to_abbr.get(record.levelname, record.levelname[0])

        prefix = (
            f"{shortlevel}{asctime}.{int(record.msecs*1000):06d} "
            f"{record.pathname}:"
            f"{record.lineno}]{self.suffix}"
        )
        return "\n".join(f"{prefix} {l}" for l in lines)


def initialize_logging(process_name=None):
    suffix = "" if process_name is None else f" {process_name}:"
    sh = logging.StreamHandler()
    sh.setFormatter(_Formatter(suffix))
    sh.setLevel(logging.INFO)
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(sh)

    sys.excepthook = _handle_unhandled_exception


def gethostname():
    """Get the hostname of the machine."""
    hostname = socket.gethostname()
    hostname = hostname.replace(".facebook.com", "")
    return hostname
