import logging
from logging import Logger

# from pythonjsonlogger import jsonlogger
import datetime


import os
import shutil
import gc
import re
import zipfile
from copy import deepcopy

from typing import Optional, List, Literal, Callable, Dict, Union, Tuple, Any

from functools import wraps

from time import perf_counter, sleep
import time
import json


import pandas as pd
import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid
import numpy as np


from io import BytesIO
from tqdm.auto import tqdm

from IPython.display import display_html
from IPython.display import HTML


from urllib.parse import unquote

from math import ceil

################################################

LOG_PATH = "/home/shailja/logs"

# Color mappings
COLOUR_MAPPING = {
    "BLACK": "\033[30m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "END": "\033[0m",
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}


class ColumnNotFound(Exception):
    def __init__(self, columns):
        self.message = f"Avialable columns are: {columns}"
        super().__init__(self.message)


############################################# Logger #######################################

def create_suppression_filter(suppressed_loggers):
    def filter_func(record):
        return record.name not in suppressed_loggers

    return filter_func


def add_logging_level(
    level_name: str, level_num: int, method_name: Optional[str] = None
):
    if not method_name:
        method_name = level_name.lower()

    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def log_to_root(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name.upper())
    setattr(logging, level_name.upper(), level_num)
    setattr(logging.getLoggerClass(), method_name, log_for_level)
    setattr(logging, method_name, log_to_root)


class CustomFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": COLOUR_MAPPING["CYAN"],
        "INFO": COLOUR_MAPPING["GREEN"],
        "VERBOSE": COLOUR_MAPPING["WHITE"],
        "WARNING": COLOUR_MAPPING["YELLOW"],
        "ERROR": COLOUR_MAPPING["RED"] + COLOUR_MAPPING["BOLD"],
        "CRITICAL": COLOUR_MAPPING["RED"]
        + COLOUR_MAPPING["BOLD"]
        + COLOUR_MAPPING["UNDERLINE"],
    }

    def format(self, record):
        formatted_record = deepcopy(record)

        level_name = formatted_record.levelname
        color = self.COLORS.get(level_name, "")

        # Adjust name based on whether it's in the main module or not
        formatted_record.name = (
            formatted_record.name
            if formatted_record.funcName == "<module>"
            else f"{formatted_record.name}.{formatted_record.funcName}"
        )
        

        # Create the log message without color first
        custom_format = (
            "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s"
        )
        formatter = logging.Formatter(custom_format, datefmt="%Y-%m-%d %H:%M:%S")
        log_message = formatter.format(formatted_record)

        # Then apply color to the entire message
        colored_message = f"{color}{log_message}{COLOUR_MAPPING['END']}"

        return colored_message


class JSONFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            t = time.localtime(record.created)
            s = time.strftime("%Y-%m-%d %H:%M:%S", t)
        return s


def create_logger(
    name: str,
    level: Literal[
        "notset",
        "debug",
        "info",
        "verbose",
        "warning",
        "error",
        "critical",
    ] = "info",
    log_file: Optional[str] = None,
    consolidate_file_loggers: bool = True,
    use_json: bool = True,
    filters: Optional[List[Callable[[logging.LogRecord], bool]]] = None,
    custom_levels: Optional[Dict[str, int]] = None,
    suppress_loggers: Optional[List[str]] = None,
) -> Logger:
    """
    Create a logger with the specified name and level, including color formatting for console
    and JSON formatting for file output, with optional custom filters.

    Args:
        name: Name of the logger.
        level: Logging level. Defaults to "info".
        log_file: Name of the log file. Defaults to None.
        consolidate_file_loggers: Whether to consolidate file loggers. Defaults to True.
        use_json: Whether to use JSON formatting for file logging. Defaults to True.
        filters: List of custom filter functions to apply to the logger. Defaults to None.
        custom_levels: Dictionary of custom log level names and their corresponding integer values.


    allowed_colours = ["black", "red", "green", "yellow", "blue", "cyan", "white",
                    "bold_black", "bold_red", "bold_green", "bold_yellow", "bold_blue",
                     "bold_cyan", "bold_white",
                    ]



    Returns:
        Configured logger object.
    """
    verbose_level: int = logging.INFO + 3

    level_to_int_map: Dict[str, int] = {
        "notset": logging.NOTSET,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "verbose": verbose_level,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    ## Adding custom log level verbose
    add_logging_level("verbose", verbose_level)
    level_to_int_map["verbose"] = verbose_level

    # Add custom levels
    if custom_levels:
        for level_name, level_value in custom_levels.items():
            add_logging_level(level_name, level_value)
            level_to_int_map[level_name.lower()] = level_value
            CustomFormatter.COLORS[level_name.upper()] = COLOUR_MAPPING[
                "BLUE"
            ]  # Default color for custom levels

    logger: Logger = logging.getLogger(name)
    level_int: int = (
        level_to_int_map[level.lower()] if isinstance(level, str) else level
    )
    logger.setLevel(level_int)

    # Add suppression filter if suppress_loggers is provided
    if suppress_loggers:
        suppression_filter = create_suppression_filter(suppress_loggers)
        logger.addFilter(suppression_filter)

    custom_formatter = CustomFormatter(
        "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # JSON formatter for file output
    #     json_formatter: jsonlogger.JsonFormatter = jsonlogger.JsonFormatter(
    #         "%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    #     )

    json_formatter = JSONFormatter()

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with color formatting
    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setLevel(level_int)
    console_handler.setFormatter(custom_formatter)
    logger.addHandler(console_handler)

    # File logging setup
    if log_file:
        today: str = datetime.datetime.now().strftime("%Y_%m_%d")
        curr_time: str = datetime.datetime.now().strftime("%H_%M_%S")
        log_dir: str = os.path.join(LOG_PATH, today, log_file)
        os.makedirs(log_dir, exist_ok=True)
        log_file_path: str = os.path.join(
            log_dir, f"{curr_time}.json" if use_json else f"{curr_time}.log"
        )
        file_handler: logging.FileHandler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level_int)
        file_handler.setFormatter(
            json_formatter
            if use_json
            else logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        if consolidate_file_loggers:
            logging.getLogger().addHandler(file_handler)
        else:
            logger.addHandler(file_handler)

    # Add custom filters
    if filters:
        for filter_func in filters:
            logger.addFilter(filter_func)

    return logger

############################################## Decorators #####################################################

def log_function(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):

        # Determine if this is a method call, First argument is always Class itself (excluding staticmethods)
        is_method = args and hasattr(args[0].__class__, func.__name__)
        if is_method:
            class_name = args[0].__class__.__name__
            if func.__name__ == "__init__":
                # For __init__, use just the class name
                function_name = class_name
            else:
                function_name = f"{class_name}.{func.__name__}"
            self = args[0]
        else:
            function_name = func.__name__
            self = None

        # If a logger is provided in args or kwargs, use it; otherwise, use the decorator's logger
        func_logger = kwargs.get("logger") or (
            args[0] if args and isinstance(args[0], logging.Logger) else None
        )
        log_level = logging.INFO if func_logger is None else func_logger.level

        # Create a new logger with the appropriate level
        new_logger = create_logger(
            function_name, level=logging.getLevelName(log_level).lower()
        )

        # Replace the logger in kwargs with the new logger
        kwargs["logger"] = new_logger

        if is_method:
            self.logger = new_logger

        # Log the function call
        new_logger.debug(f"Calling function: {function_name}")

        # Call the original function
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            raise

        # Log the function completion
        new_logger.debug(f"Function {function_name} completed")

        return result

    return wrapper


logger = create_logger('utils', 'info')

def timeit(func: Callable):

    logger = create_logger("timeit", "info")

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        execution_time = end_time - start_time
        logger.info(
            f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute."
        )
        return result

    return wrapper


############################################## Jupyter Kernel ##################################################

def restart_kernel() -> None:

    display_html("<script>Jupyter.notebook.kernel.restart()</script>", raw=True)


def shutdown_kernel() -> None:

    from IPython.core.getipython import get_ipython

    get_ipython().kernel.do_shutdown(True)


def get_kernel_env() -> None:

    raise NotImplementedError

def get_kernel_env():
    raise NotImplementedError


def get_kernel_environment():
    """
    Detects the environment in which the code is being run (Jupyter Notebook, Jupyter Lab, or terminal).

    Returns:
        str: A string indicating the environment ('Jupyter Notebook', 'Jupyter Lab', 'Terminal', or 'Unknown').
    """
    try:
        # Check if the code is running in an IPython environment
        from IPython import get_ipython
        
        ipython_env = get_ipython()
        
        if ipython_env:
            # Check if running in a Jupyter Notebook
            if 'IPKernelApp' in ipython_env.config:
                return "Jupyter Notebook"
            elif 'TerminalInteractiveShell' in ipython_env.config:
                return "Terminal"
            else:
                return "Unknown IPython environment"
        else:
            return "Terminal"
    except ImportError:
        # If IPython is not installed, it's a terminal environment
        return "Terminal"

#######################################################################################


def run_untill(target_datetime: set) -> None:
    """

    target_datetime: date and time till when you want to hold the code after that it will end, (2024, 3, 5, 15, 30) YYYY, M, D, HH, MM

    """

    # Define the specific date and time you want to wait until
    target_datetime = datetime.datetime(target_datetime)  # March 5, 2024, 3:30 PM

    logger.info(f"Will wait till: {target_datetime}")

    while True:
        current_datetime = datetime.datetime.now()
        if current_datetime >= target_datetime:
            break
        else:
            sleep(60)  # Sleep for 60 seconds before checking again

    logger.info("Target date and time reached. Exiting loop.")


############################################# Files ####################################################


def get_file_extension(file_path: str, ) -> str:

    to_remove_compression = [".gzip", ".zip"]

    logger.debug(f"Removing compression extensions: {to_remove_compression}")
    for to_remove in to_remove_compression:

        file_path = file_path.replace(to_remove, "")

    return file_path.split(".")[-1]


def get_all_files(
    rootdir: str,
    extensions: Union[str, List[str]] = ".csv",
    ignore: Optional[List[str]] = None,
    contains: Optional[List[str]] = None,
    depth: int = -1,
) -> List[str]:
    """
    Recursively searches for files in the specified directory and its subdirectories up to a certain depth.
    Args:
    - rootdir (str): Path of the directory to search in.
    - extensions (list, optional): List of file extensions to include. Defaults to [".csv"].
    - ignore (list, optional): List of keywords. Files containing any of these keywords in their names will be ignored. Defaults to [].
    - contains (list, optional): List of keywords. Only files containing any of these keywords in their names will be included. Defaults to [].
    - depth (int, optional): Depth up to which to search for files inside directories. -1 means search till the last depth. Defaults to -1.
    - logger (logging.Logger, optional): Logger object. If not provided, a new logger will be created.
    Returns:
    - list: List of paths to all found files.


    Note: ignore is given priority over contains
    """

    logger.debug(f"Starting file search in directory: {rootdir}")

    logger.debug(f"Original File Path: {rootdir}")

    rootdir = unquote(rootdir)

    logger.debug(f"Parsed File Path: {rootdir}")

    extensions = list(extensions) if isinstance(extensions, str) else extensions

    ignore = [] if ignore is None else ignore

    contains = [] if contains is None else contains

    all_files = []
    rootdir = os.path.normpath(rootdir)
    for root, subFolders, files in os.walk(rootdir):
        current_depth = root.count(os.sep) - rootdir.count(os.sep)
        if depth != -1 and current_depth > depth:
            logger.debug(f"Skipping directory {root} due to depth limit")
            continue  # Skip if the current depth exceeds the specified depth

        logger.debug(f"Searching in directory: {root}")
        for file in files:
            if ignore and any(keyword in file for keyword in ignore):
                logger.debug(f"Ignoring file: {file}")
                continue
            if contains and not any(keyword in file for keyword in contains):
                logger.debug(
                    f"Skipping file {file} as it doesn't contain required keywords"
                )
                continue
            if file.endswith(tuple(extensions)):
                full_path = os.path.join(root, file)
                all_files.append(full_path)
                logger.debug(f"Found matching file: {full_path}")

    logger.info(f"File search completed. Total files found: {len(all_files)}")
    return all_files


def organize_files_by_extension(folder_path: str, depth: int = 1, current_depth: int = 0):
    """
    Organizes files within a given folder by creating subfolders based on file extensions and moving the files into them.

    Args:
        folder_path (str): The path to the folder that needs to be organized.
        depth (int, optional): The depth to which the function will recursively organize files in subfolders. 
                               The default is 1 (organize only the specified folder).
        current_depth (int, optional): Tracks the current recursion depth. This is managed internally by the function.
                                       The default is 0.

    Returns:
        None

    Behavior:
        - The function lists all items in the provided folder.
        - For each item, if it's a directory and the current depth is less than the specified depth, 
          the function will recursively organize the files within that subfolder.
        - For each file, the function identifies the file extension and creates a subfolder 
          (named after the extension) if it doesn't already exist.
        - The file is then moved into the corresponding subfolder.
        - If a file has no extension, it is moved into a folder named 'no_extension'.
        - Logging statements provide detailed information about the process.

    Example Usage:
        organize_files_by_extension('/path/to/your/folder', depth=2)
    """
    if current_depth >= depth:
        return

    # Get a list of all files and folders in the given folder
    items = os.listdir(folder_path)

    for item in items:
        item_path = os.path.join(folder_path, item)

        logger.info(f"Current file is: {item_path}")

        if os.path.isdir(item_path):
            logger.debug(f"{item_path} is a folder, at depth: {current_depth}.")
            # Recursively organize the subfolder
            organize_files_by_extension(item_path, depth, current_depth + 1)
        else:
            # Get the file extension and make a folder for that extension
            file_extension = os.path.splitext(item)[1][1:]  # Get extension without the dot
            if not file_extension:
                file_extension = "no_extension"  # Handle files with no extension

            destination_folder = os.path.join(folder_path, file_extension)

            if (not os.path.exists(destination_folder)) & (logger.level != 10):
                logger.debug(f"Creating folder at: {destination_folder}")
                os.makedirs(destination_folder)

            # Move the file to the appropriate folder
            if logger.level != 10:
                logger.debug(f"Saving file at {os.path.join(destination_folder, item)}")
                shutil.move(item_path, os.path.join(destination_folder, item))


############################################# Dataframe ########################################

def read_parquet_with_sample(
    file_path: str,
    sample_size: Optional[int] = None,
    columns: Optional[List[str]] = None,
    row_groups: Optional[Any] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a Parquet file and return its contents as a pandas DataFrame.

    This function attempts to read the Parquet file using pyarrow. If that fails,
    it falls back to using pandas to read the file.

    Parameters:
    -----------
    file_path : str
        The path to the Parquet file to be read.
    sample_size : int, optional
        If provided, only this many rows will be read from the file.
    get_metadata : bool, default False
        If True, return the file's metadata instead of its contents.
    columns : List[str], optional
        A list of column names to read. If None, all columns are read.
    row_groups : Any, optional
        only the given row group will be read
    logger : Logger, optional
        A logger object for logging messages.

    Returns:
    --------
    pd.DataFrame or pyarrow.parquet.FileMetaData
        If get_metadata is False, returns a pandas DataFrame containing the Parquet file's data.
        If get_metadata is True, returns the Parquet file's metadata.

    Raises:
    -------
    ArrowInvalid
        If the file is not a valid Parquet file.
    ColumnNotFound
        If any of the specified columns are not found in the Parquet file.

    Notes:
    ------
    - If pyarrow fails to read the file, the function will attempt to read it using pandas.
    - If sample_size is provided, only the first batch of that size will be returned.
    """
    try:
        parquet_file = pq.ParquetFile(file_path)
        logger.debug(f"Read metadata of file: {file_path}")
    except ArrowInvalid:
        if get_file_extension(file_path) != "parquet":
            logger.error("Incorrect File Format, only parquet files are accepted.")
            raise
        raise
    except Exception as e:
        logger.warning(
            "Not able to read file by pyarrow, trying to read it using pandas"
        )
        return pd.read_parquet(file_path)

    if columns:
        not_found_cols = set(columns) - set(parquet_file.schema.names)
        if not_found_cols:
            raise ColumnNotFound(parquet_file.schema.names)

    # if parquet_file.metadata.num_rows <= sample_size:
    #     logger.warning(f"sample_size is greater than total rows in df, hence returning whole dataframe.")
    #     return parquet_file.read(columns=columns, use_threads=True, **kwargs).to_pandas()

    if (sample_size) & (parquet_file.metadata.num_rows >= sample_size):
        logger.debug(f"Reading sample")
        return next(
            parquet_file.iter_batches(
                batch_size=sample_size, columns=columns, row_groups=row_groups
            )
        ).to_pandas()

    return parquet_file.read(columns=columns, use_threads=True, **kwargs).to_pandas()


def read_parquet_with_filters(
    file_path: str,
    columns: Optional[List[str]] = None,
    filters: Union[List[Tuple], List[List[Tuple]]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a Parquet file and return its contents as a pandas DataFrame.

    This function attempts to read the Parquet file using pyarrow. If that fails,
    it falls back to using pandas to read the file.

    Parameters:
    -----------
    file_path : str
        The path to the Parquet file to be read.
    filters : List[Tuple], optional
        Filter syntax: [[(column, op, val), …],…] where op is [==, =, >, >=, <, <=, !=, in, not in]
    columns : List[str], optional
        A list of column names to read. If None, all columns are read.
    logger : Logger, optional
        A logger object for logging messages.

    Returns:
    --------
    pd.DataFrame or pyarrow.parquet.FileMetaData

    Raises:
    -------
    ArrowInvalid
        If the file is not a valid Parquet file.
    ColumnNotFound
        If any of the specified columns are not found in the Parquet file.

    Notes:
    ------
    - If pyarrow fails to read the file, the function will attempt to read it using pandas.
    - If sample_size is provided, only the first batch of that size will be returned.
    """
    try:
        metadata = pq.read_metadata(file_path)
        logger.debug(f"Read metadata of file: {file_path}")
    except ArrowInvalid:
        if get_file_extension(file_path) != "parquet":
            logger.error("Incorrect File Format, only parquet files are accepted.")
            raise
        raise
    except Exception as e:
        logger.warning(
            "Not able to read file by pyarrow, trying to read it using pandas"
        )
        return pd.read_parquet(file_path, filters=filters, columns=columns)

    if columns:
        not_found_cols = set(columns) - set(metadata.schema.names)
        if not_found_cols:
            raise ColumnNotFound(metadata.schema.names)

    return pq.read_table(
        file_path, columns=columns, use_threads=True, filters=filters, **kwargs
    ).to_pandas()


def get_columns(file_path: str, get_metadata: bool = False) -> list:
    file_extension = get_file_extension(file_path)

    column_readers = {
        "csv": lambda: pd.read_csv(file_path, nrows=0).columns.tolist(),
        "txt": lambda: pd.read_csv(file_path, nrows=0).columns.tolist(),
        "xlsx": lambda: pd.read_excel(file_path, nrows=0).columns.tolist(),
        "parquet": lambda: pq.read_schema(file_path).names,
    }

    if get_metadata:

        return pq.read_metadata(file_path)

    reader = column_readers.get(file_extension)
    if reader:

        reader = column_readers.get(file_extension)
        return reader()
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def rename_columns(df, ) -> pd.DataFrame:
    # Define a function to clean individual column names
    def clean_column_name(col):
        # Replace spaces and dots with underscores, remove consecutive underscores,
        # and strip leading/trailing underscores in one pass
        return re.sub(r"_+", "_", re.sub(r"[\s.]+", "_", col.lower())).strip("_")

    logger.debug(f"Original Columns: {df.columns.tolist()}")

    # Rename the columns using a dictionary comprehension
    df.rename(columns={col: clean_column_name(col) for col in df.columns}, inplace=True)

    logger.debug(f"Renamed Columns: {df.columns.tolist()}")

    return df

def read_excel_pass(
    excel_path: str,
    password: str,
    sheet_name: str = "Sheet1",
    **kwargs,
) -> pd.DataFrame:
    """
    Read the password proted excel file:

    excel_path: path of file

    password: password of a file

    sheet_name: Sheet name which you want to read, default `Sheet1`

    """

    logger.debug(f"Original Excel Path: {excel_path}")

    excel_path = unquote(excel_path)

    logger.debug(f"Parsed Excel Path: {excel_path}")

    if password is None:
        return pd.read_excel(excel_path, sheet_name=sheet_name, **kwargs)

    import msoffcrypto
    from msoffcrypto.exceptions import DecryptionError

    decrypted_workbook = BytesIO()

    try:
        with open(excel_path, "rb") as file:
            office_file = msoffcrypto.OfficeFile(file)
            office_file.load_key(password=password)
            office_file.decrypt(decrypted_workbook)
            logger.info(f"Decrypted the excel file")
    except DecryptionError as e:
        logger.warning("File is not encrypted. Reading directly.")
        return pd.read_excel(excel_path, sheet_name=sheet_name, **kwargs)
    except FileNotFoundError:
        logger.error(f"No such File: {excel_path}")
        raise
    except Exception as e:
        logger.error(f"Error while decrypting the excel file {excel_path}: {e}")
        raise

    try:
        df = pd.read_excel(decrypted_workbook, sheet_name=sheet_name or "Sheet1")
        return df
    except Exception as e:
        logger.error(f"Error while reading decrypted excel file: {e}")
        raise


def read_dataframe(
    file_path: str,
    chunk_size: Optional[int] = None,
    password: Optional[str] = None,
    sheet_name: str = "Sheet1",
    rename_cols: bool = False,
    sample_size: Optional[int] = None,
    # get_metadata: bool = False,
    columns: Optional[List[str]] = None,
    filters: Union[List[Tuple], List[List[Tuple]]] = None,
    # row_groups: Optional[int] = None,
    **kwargs,
) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
    """
    Read a dataframe from a file.

    This function can read various file formats including CSV, Excel, and potentially others
    depending on the file extension. It supports reading large files in chunks and password-protected files.

    Parameters:
    -----------
    file_path : str
        The path to the file to be read.
    logger : Logger, optional
        A logger object for logging messages.
    chunk_size : int, optional
        If specified, return an iterable object to read the file in chunks.
    password : str, optional
        Password for encrypted Excel files.
    sheet_name : str, default "Sheet1"
        Name of the sheet to read for Excel files.
    rename_cols : bool, default False
        Whether to rename the columns or not
    **kwargs
        Additional keyword arguments to pass to the pandas reading function.

    Returns:
    --------
    Union[pd.DataFrame, pd.io.parsers.TextFileReader]
        A pandas DataFrame if chunk_size is None, otherwise a TextFileReader object for iterating over chunks.

    """
    ...

    logger.debug(f"Original File Path: {file_path}")

    file_path = unquote(file_path)

    logger.debug(f"Parsed File Path: {file_path}")

    file_extension = get_file_extension(file_path)

    def read_excel():

        if password:

            return read_excel_pass(
                file_path,
                password,
                sheet_name=sheet_name,
                nrows=sample_size,
                **kwargs,
            )
        else:
            return pd.read_excel(
                file_path, sheet_name=sheet_name, nrows=sample_size, **kwargs
            )

    readers = {
        "xlsx": read_excel,
        "xls": read_excel,
        "xlsm": read_excel,
        "csv": lambda: (
            pd.read_csv(file_path, nrows=sample_size, **kwargs)
            if not chunk_size
            else pd.concat(
                pd.read_csv(file_path, chunksize=chunk_size), ignore_index=True
            )
        ),
        "txt": lambda: (
            pd.read_csv(file_path, nrows=sample_size, **kwargs)
            if not chunk_size
            else pd.concat(
                pd.read_csv(file_path, chunksize=chunk_size), ignore_index=True
            )
        ),
        "gzip": lambda: (
            pd.read_csv(file_path, compression="gzip", **kwargs)
            if file_path.endswith(".csv.gzip")
            else pd.read_parquet(file_path, **kwargs)
        ),
        "parquet": lambda: (
            read_parquet_with_sample(
                file_path,
                sample_size=sample_size,
                # get_metadata=get_metadata,
                columns=columns,
                # row_groups=row_groups,
                **kwargs,
            )
            if sample_size
            else read_parquet_with_filters(
                file_path, columns, filters=filters, **kwargs
            )
        ),
    }

    try:
        logger.info(f"Reading: {file_path}")
        logger.debug(f"File extension: {file_extension}")
        reader = readers.get(file_extension)

        if reader:

            df = reader()

            if rename_cols:
                return rename_columns(df)

            return df
        # elif file_extension == 'parquet':

        #     df = read_parquet_with_sample(file_path, sample_size=sample_size, columns=columns, row_groups=row_groups, logger=logger) if sample_size else read_parquet_with_filters(file_path, columns, filters=filters, logger=logger)

        else:
            logger.warning(
                f"File extension '{file_extension}' not supported. Supported extensions: {list(readers.keys())}"
            )
            return None

    except FileNotFoundError:
        logger.error(f"No such File: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise


def save_dataframe(
    df: pd.DataFrame,
    file_path: str,
    nrow_groups: int = None,
    **kwargs,
) -> None:

    # file_extension = file_path.split(".")[-1].lower()
    file_extension = get_file_extension(file_path)

    row_group_size = df.shape[0] // nrow_groups if nrow_groups else None
    logger.debug(f"Saving df wiith row_group_size: {row_group_size}")

    savers = {
        #         'xlsx': lambda: df.to_excel(file_path, engine="openpyxl", index=False, **kwargs),
        "xlsx": lambda: df.to_excel(
            file_path, index=False, engine="openpyxl", **kwargs
        ),
        "csv": lambda: df.to_csv(file_path, index=False, **kwargs),
        "txt": lambda: df.to_csv(file_path, index=False, sep="\t", **kwargs),
        "gzip": lambda: (
            df.to_csv(file_path, index=False, compression="gzip", **kwargs)
            if file_path.endswith(".csv.gzip")
            else df.to_parquet(file_path, index=False, compression="gzip", **kwargs)
        ),
        "parquet": lambda: df.to_parquet(
            file_path,
            index=False,
            row_group_size=row_group_size,
            compression="gzip",
            **kwargs,
        ),
    }

    if logger.level == 10:

        logger.debug(f"Not Saving File to: {file_path}")

        return

    try:
        savers[file_extension]()
        logger.info(f"DataFrame successfully saved to {file_path}")
    except KeyError:
        logger.error(
            f"File extension '{file_extension}' not supported. Supported extensions: {list(savers.keys())}"
        )
        raise
    except Exception as e:
        logger.error(f"Error saving DataFrame to {file_path}: {str(e)}")
        raise

def save_dataframes_as_excel_sheets(
    dfs: List[pd.DataFrame],
    path: str,
    sheet_names: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """Saves a list of dataframes as separate sheets in an excel file.

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        The list of dataframes to save.
    path : str
        The file path to save the excel file.
    sheet_names : List[str], optional
        The names of the sheets in the excel file. By default, None. If None, the sheet names are set as Sheet1, Sheet2, ...
    **kwargs : dict
        Additional keyword arguments to pass to the save

    Returns
    -------
    str
        The file path where the excel file is saved.
    """
    if sheet_names is None:
        sheet_names = [f"Sheet{i+1}" for i in range(len(dfs))]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for df, sheet_name in zip(dfs, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name, **kwargs)
    # return path



def get_concated_df(
    dfs: List[pd.DataFrame],
    remove_duplicates: List[Optional[str]] = [None],
    chunk_size: int = 0,
    apply_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Concatenate multiple dataframes from various file sources.

    Args:
        dfs (list): List of file paths to concatenate.
        remove_duplicates (list, optional): Columns to use for removing duplicates. Defaults to [None].
        chunk_size (int, optional): Size of chunks to use when reading large CSV files. Defaults to 0.
        apply_filter (function, optional): Function to apply filters to each dataframe. Defaults to None.
        **kwargs: Additional keyword arguments for pandas read functions.

    Returns:
        pandas.DataFrame: Concatenated dataframe.
    """

    df = (
        pd.DataFrame(columns=remove_duplicates)
        if remove_duplicates[0]
        else pd.DataFrame()
    )

    pbar = tqdm(dfs, desc="Processing files")

    for file in pbar:

        a = read_dataframe(file,  chunk_size=chunk_size, **kwargs)

        if remove_duplicates[0]:
            for col in remove_duplicates:
                logger.debug(
                    f"original shape before removing duplicates of column: {col} is {a.shape[0]}"
                )
                a = a[
                    ~a[col].isin(df[col].unique())
                ]  # only keeping those which aren't already present in concated_df

                logger.debug(
                    f"Final shape after removing duplicates of column: {col} is {a.shape[0]}"
                )

        if apply_filter:

            logger.debug(f"Shape before filter: {a.shape[0]}")
            a = apply_filter(a)
            logger.debug(f"Shape after filter: {a.shape[0]}")

        df = pd.concat([df, a], ignore_index=True)
        pbar.set_postfix({"Total Shape": f"{df.shape[0]:_}"}, refresh=True)

    logger.info(f"Total shape of concated df: {df.shape[0]:_}")

    return df



def save_zip(
    df: pd.DataFrame,
    zip_name: str,
    inside_csv_name: str,
    save_location: str = ".",
    mode: Literal["w", "a"] = "w",
) -> None:
    """

    df: dataframes which you want to save in a zip

    inside_csv_name: csv name, with which we want to save it inside the zip (in csv format only)

    save_location: path where you want to save the zip

    mode: `w` for write, `a` for append, default = `w`

    Returns: Nothing

    """

    if logger.level == 10:

        logger.debug(f"Not Saving File to: {save_location}")

        return

    try:

        with zipfile.ZipFile(
            (os.path.join(save_location, zip_name)),
            mode=mode,
            compression=zipfile.ZIP_DEFLATED,
        ) as z:
            z.writestr(inside_csv_name, df.to_csv(index=False).encode())
    except Exception as e:
        logger.error(f"Error while saving zip file: {e}")
        raise


def _get_matched_id(llist, item) -> int:
    for i, l in enumerate(llist):
        if item in l:
            return i
    return -1



def zip_to_dfs(
    zip_path: str,
    files_keyword: List[str] = None,
    file_name: bool = False,
    apply_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    **kwargs,
) -> Union[Tuple[pd.DataFrame, ...], None]:
    """
    zip_path: path of the zip file, which you want to extract in RAM
    verbose: will display the files name inside zip
    files_keyword: keyword present in a file name which is inside a zip file (order must be same in which it will return the dataframe)
    Returns the files present inside the zip file
    """

    to_return = []
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        files_in_zip = zip_ref.namelist()

        logger.info(f"Order of the files returned will be {files_in_zip}")

        # Loop through the files inside the zip archive
        for i, file_keyword in enumerate(files_keyword):
            logger.debug(f"Searching for file with keyword: {file_keyword}")
            matched_id = _get_matched_id(files_in_zip, file_keyword)
            if matched_id == -1:
                logger.warning(f"{file_keyword} isn't given in the files_keyword")
                return

            file_name = files_in_zip[matched_id]
            logger.info(f"Processing file: {file_name}")

            with zip_ref.open(file_name) as file:
                # Read the CSV content into a BytesIO object
                csv_data = BytesIO(file.read())
                df = pd.read_csv(csv_data, **kwargs)

                logger.debug(f"File {file_name} Read, with shape: {df.shape}")

                if apply_filter:
                    logger.debug("Applying filter to DataFrame")
                    df = apply_filter(df)
                    logger.debug(f"DataFrame shape after filtering: {df.shape}")

                to_return.append(df)
                logger.debug("DataFrame appended to return list")

                del df
                gc.collect()
                logger.debug("DataFrame deleted and garbage collected")

    logger.debug("Zip file processing completed")

    if len(to_return) == 1 and file_name:
        logger.debug(f"Returning single DataFrame with file name: {file_name}")
        return to_return[0], file_name
    elif len(to_return) == 1:
        logger.debug("Returning single DataFrame")
        return to_return[0]

    logger.debug(f"Returning tuple of {len(to_return)} DataFrames")
    return tuple(to_return)

