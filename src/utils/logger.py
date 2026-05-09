import logging
from rich.logging import RichHandler


def setup_logger(name="app", debug=False):

    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(name)s - %(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

    return logging.getLogger(name)