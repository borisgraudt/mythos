import logging


def setup_logging(level: int = logging.INFO, rich: bool = True) -> None:
    try:
        if rich:
            from rich.logging import RichHandler
            logging.basicConfig(
                level=level,
                format="%(message)s",
                handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
            )
            return
    except ImportError:
        pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
