import logging
import socket
import sys

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

def fix_exception_lines(tb_lines):
    formatted_lines = []
    for line in tb_lines:
        # Replace the standard file and line format with the custom format
        if line.startswith('  File'):
            # Extract the filename and line number
            parts = line.split(',')
            file_info = parts[0].strip()[6:-1]  # Remove '  File "' and '"'
            line_info = parts[1].strip()[5:]   # Remove 'line '
            new_line = f'  File {file_info}:{line_info}'
            if len(parts) > 2:
                new_line += ', ' + ','.join(parts[2:]).strip()
            formatted_lines.append(new_line)
        else:
            formatted_lines.append(line.strip())
    return formatted_lines

class _Formatter(logging.Formatter):
    def __init__(self, suffix):
        self.suffix = suffix

    def format(self, record):
        message = record.getMessage()
        asctime = self.formatTime(record, "%m%d %H:%M:%S")

        lines = message.strip().split("\n")
        if record.exc_info:
            exc_info = fix_exception_lines(self.formatException(record.exc_info).split('\n'))
            lines.extend(exc_info)
        if record.stack_info:
            stack_info = self.formatStack(record.stack_info)
            lines.extend(stack_info.strip().split("\n"))

        shortlevel = _glog_level_to_abbr.get(record.levelname, record.levelname[0])

        prefix = (
            f"{shortlevel}{asctime}.{int(record.msecs*1000):06d} "
            f"{record.pathname}:"
            f"{record.lineno}]{self.suffix}"
        )
        return "\n".join(f"{prefix} {line}" for line in lines)


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
