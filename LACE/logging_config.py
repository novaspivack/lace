# =========== START of logging_config.py ===========
from __future__ import annotations
import sys
import multiprocessing as mp
import setproctitle
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from typing import Tuple, Optional, Union, TypeVar
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
plt.ioff()
import logging
import os
from datetime import datetime
import numpy as np
import warnings
import cProfile
import pstats



warnings.filterwarnings('ignore', category=UserWarning)
_current_log_file: Optional[str] = None
# Type aliases for improved type hints
NodeIndex = int
GridArray = npt.NDArray[np.float64]
NeighborIndices = npt.NDArray[np.int64]
Coordinates = Tuple[float, ...]
StateVarType = TypeVar('StateVarType', bound=Union[bool, int, float])
_global_grid_boundary: str = 'bounded'
_CREATING_DEFAULT_LIBRARY = False

class LogSettings:
    class Logging:
        LOG_LEVEL: str = "DEBUG"

    class Performance:
        ENABLE_PERIODIC_REPORTING: bool = False
        ENABLE_DEEP_PROFILING: bool = False
        REPORTING_INTERVAL: int = 10
        MIN_REPORTING_INTERVAL: int = 1
        MAX_REPORTING_INTERVAL: int = 1000
        ENABLE_DETAILED_LOGGING: bool = False

# --- Custom Log Level ---
DETAIL_LEVEL_NUM = 15  # Between DEBUG (10) and INFO (20)
logging.addLevelName(DETAIL_LEVEL_NUM, "DETAIL")

def detail(self, message, *args, **kws):
    """Logs a message with level DETAIL on this logger."""
    if self.isEnabledFor(DETAIL_LEVEL_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(DETAIL_LEVEL_NUM, message, args, **kws)

# Add the 'detail' method to the Logger class
logging.Logger.detail = detail  # type: ignore [attr-defined] 
# --- End Custom Log Level ---


class FindFontFilter(logging.Filter):
    def filter(self, record):
        return "findfont" not in record.getMessage() and not record.name.startswith('matplotlib')

class NumbaLogFilter(logging.Filter):
    """
    Filters Numba logs. Allows WARNING and above always.
    Allows INFO/DEBUG only if the root logger level is DEBUG or DETAIL.
    """
    def filter(self, record):
        # Allow all non-numba logs
        if not record.name.startswith('numba'):
            return True

        # Allow Numba WARNING, ERROR, CRITICAL always
        if record.levelno >= logging.WARNING:
            return True

        # Check the root logger's effective level for Numba INFO/DEBUG
        root_level = logging.getLogger().getEffectiveLevel()
        # Allow Numba INFO/DEBUG only if root is set to DEBUG or DETAIL
        if root_level <= DETAIL_LEVEL_NUM: # DETAIL_LEVEL_NUM is 15, DEBUG is 10
             return True
        else:
             # Suppress Numba INFO/DEBUG if root level is INFO or higher
             # logger.debug(f"Suppressing Numba log (Level: {record.levelname}) because root level is {logging.getLevelName(root_level)}") # Optional debug
             return False

def setup_logging(log_dir: str, report_dir: str, profile_dir: str) -> logging.Logger: # Added report_dir, profile_dir
    """Setup logging with main log file and conditionally created report/profile files.
       Creates an inactive RenderingPipeline logger.
       (Round 16: Added NumbaLogFilter)
       (Round 17: Removed NumbaLogFilter, configure main Numba logger level)
       (Round 33: Create inactive RenderingPipeline logger)"""
    try:
        # Check if this is a worker process
        if mp.current_process().name != 'MainProcess':
            # Worker logging is handled by _worker_initializer_func
            logger = logging.getLogger(__name__)
            if not logger.handlers: logger.addHandler(logging.NullHandler())
            return logger

        # Get the root logger
        root_logger = logging.getLogger()
        if root_logger.handlers: return logging.getLogger(__name__) # Already initialized

        logger = logging.getLogger(__name__) # Get the main logger for the application

        timestamp_24hr = datetime.now().strftime("%Y%m%d_%H%M%S") # 24-hour format

        # --- Main Logger Setup ---
        log_level_str = LogSettings.Logging.LOG_LEVEL.upper()
        file_log_level = getattr(logging, log_level_str, logging.INFO)
        if log_level_str == "DETAIL":
            file_log_level = DETAIL_LEVEL_NUM
            logger.info(f"Detected DETAIL log level setting. File log level set to {DETAIL_LEVEL_NUM}.")

        console_log_level = logging.INFO # Keep console less verbose
        main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

        # --- Create Handlers ---
        main_log_filename = f'LACE_{timestamp_24hr}.log'
        main_file_handler = logging.FileHandler(os.path.join(log_dir, main_log_filename))
        main_file_handler.setFormatter(main_formatter)
        main_file_handler.setLevel(file_log_level) # Set handler level

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(main_formatter)
        console_handler.setLevel(console_log_level) # Set handler level

        # Set root level to the *lowest* of all handlers
        root_logger.setLevel(min(file_log_level, console_log_level))
        root_logger.addHandler(main_file_handler)
        root_logger.addHandler(console_handler)

        # --- Conditionally Setup Periodic Report Logger (Unchanged) ---
        periodic_report_logger = logging.getLogger("periodic_report")
        periodic_report_logger.propagate = False
        periodic_report_file_path = "Disabled" # Default
        logger.info(f"setup_logging: Checking GlobalSettings.Performance.ENABLE_PERIODIC_REPORTING = {LogSettings.Performance.ENABLE_PERIODIC_REPORTING}")
        if LogSettings.Performance.ENABLE_PERIODIC_REPORTING:
            logger.info("setup_logging: Periodic reporting ENABLED, attempting to add FileHandler.")
            periodic_report_logger.setLevel(logging.INFO)
            periodic_formatter = logging.Formatter('%(asctime)s - %(message)s')
            periodic_filename = f'periodic_report_{timestamp_24hr}.log'
            periodic_file_handler = logging.FileHandler(os.path.join(report_dir, periodic_filename))
            periodic_file_handler.setFormatter(periodic_formatter)
            for handler in periodic_report_logger.handlers[:]: periodic_report_logger.removeHandler(handler) # Clear existing
            periodic_report_logger.addHandler(periodic_file_handler)
            periodic_report_file_path = os.path.join(report_dir, periodic_filename)
            logger.info(f"setup_logging: Periodic report logger handlers after add: {periodic_report_logger.handlers}")
        else:
            logger.info("setup_logging: Periodic reporting DISABLED, adding NullHandler.")
            for handler in periodic_report_logger.handlers[:]: periodic_report_logger.removeHandler(handler) # Clear existing
            periodic_report_logger.addHandler(logging.NullHandler())

        # --- Conditionally Setup Deep Profile Logger (Unchanged) ---
        deep_profile_logger = logging.getLogger("deep_profile")
        deep_profile_logger.propagate = False
        deep_profile_file_path = "Disabled" # Default
        logger.info(f"setup_logging: Checking GlobalSettings.Performance.ENABLE_DEEP_PROFILING = {LogSettings.Performance.ENABLE_DEEP_PROFILING}")
        if LogSettings.Performance.ENABLE_DEEP_PROFILING:
            logger.info("setup_logging: Deep profiling ENABLED, attempting to add FileHandler.")
            deep_profile_logger.setLevel(logging.DEBUG)
            profile_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            profile_filename = f'deep_profile_{timestamp_24hr}.log'
            profile_file_handler = logging.FileHandler(os.path.join(profile_dir, profile_filename))
            profile_file_handler.setFormatter(profile_formatter)
            for handler in deep_profile_logger.handlers[:]: deep_profile_logger.removeHandler(handler) # Clear existing
            deep_profile_logger.addHandler(profile_file_handler)
            deep_profile_file_path = os.path.join(profile_dir, profile_filename)
            logger.info(f"setup_logging: Deep profile logger handlers after add: {deep_profile_logger.handlers}")
        else:
            logger.info("setup_logging: Deep profiling DISABLED, adding NullHandler.")
            for handler in deep_profile_logger.handlers[:]: deep_profile_logger.removeHandler(handler) # Clear existing
            deep_profile_logger.addHandler(logging.NullHandler())

        # --- ADDED: Setup Separate Rendering Pipeline Logger (Inactive by default) ---
        rendering_logger = logging.getLogger("RenderingPipeline")
        rendering_logger.propagate = False # Do not pass messages to root logger
        # Remove any handlers that might exist from previous runs (e.g., during development)
        for handler in rendering_logger.handlers[:]:
            rendering_logger.removeHandler(handler)
        # Add NullHandler to prevent "No handlers found" warnings if used before activation
        rendering_logger.addHandler(logging.NullHandler())
        # Set level high initially to ensure it's inactive
        rendering_logger.setLevel(logging.CRITICAL + 1)
        logger.info("Initialized separate 'RenderingPipeline' logger (inactive by default).")
        # --- END ADDED ---

        # --- Configure matplotlib and numba loggers ---
        # Numba logger setup (remains the same)
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)
        for handler in numba_logger.handlers[:]: numba_logger.removeHandler(handler)
        if not numba_logger.hasHandlers(): numba_logger.addHandler(logging.NullHandler())
        numba_logger.propagate = False
        logger.info("Configured MAIN process 'numba' logger with level WARNING.")

        # Matplotlib setup (remains the same)
        matplotlib_logger = logging.getLogger('matplotlib')
        matplotlib_logger.setLevel(logging.WARNING)
        for handler in matplotlib_logger.handlers[:]: matplotlib_logger.removeHandler(handler)
        find_font_filter = FindFontFilter()
        stream_h = logging.StreamHandler(sys.stdout)
        stream_h.addFilter(find_font_filter)
        stream_h.setLevel(logging.WARNING)
        matplotlib_logger.addHandler(stream_h)
        matplotlib_logger.propagate = False
        logger.info("Configured 'matplotlib' logger with level WARNING and FindFontFilter.")
        # ---

        file_log_level_name = logging.getLevelName(file_log_level)
        console_log_level_name = logging.getLevelName(console_log_level)
        root_log_level_name = logging.getLevelName(root_logger.level)
        logger.info(f"Logging initialized. Root Level: {root_log_level_name}, File Handler Level: {file_log_level_name}, Console Handler Level: {console_log_level_name}")
        logger.info(f"Log file: {os.path.join(log_dir, main_log_filename)}")
        logger.info(f"Periodic Report file: {periodic_report_file_path}")
        logger.info(f"Deep Profile file: {deep_profile_file_path}")
        return logger # Return the initialized main logger

    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise
      

APP_DIR = "LACE"
# --- MODIFIED: Added 'reports' ---
SUBDIRS = {
    'logs': 'logs',
    'saves': 'saves',
    'rules_backups': 'rules_backups',
    'data': 'data',
    'cache': 'cache',
    'profiles': 'profiles',
    'config': 'config',
    'reports': 'reports' # ADDED
}
# --- END MODIFIED ---

def setup_directories() -> Tuple[dict, str]:
    """Sets up the necessary directories for the application."""
    try:
        base_path = os.path.join(os.getcwd(), APP_DIR)

        # Check if Resources directory exists
        resources_path = os.path.join(base_path, "Resources")
        if not os.path.exists(resources_path):
            # Create Resources directory
            os.makedirs(resources_path, exist_ok=True)

        # Define subdirectories with new structure
        # --- MODIFIED: Added 'reports' and updated config paths ---
        SUBDIRS = {
            'logs': 'logs',
            'saves': 'saves',
            'data': 'data',
            'cache': 'cache',
            'profiles': 'profiles',
            'config': 'config',
            'config_colors': 'config/colors',
            'config_rules': 'config/rules',
            'config_rules_backups': 'config/rules_backups',
            'config_presets': 'config/presets',
            'reports': 'reports' # ADDED
        }
        # --- END MODIFIED ---

        # Create subdirectories inside Resources
        paths = {}
        for key, subdir in SUBDIRS.items():
            path = os.path.join(resources_path, subdir)
            os.makedirs(path, exist_ok=True)
            paths[key] = path

        return paths, base_path
    except Exception as e:
        print(f"Fatal error in directory setup: {str(e)}")
        raise SystemExit(1)


# Initialize directory structure and logger
APP_PATHS, BASE_PATH = setup_directories() # This needs to happen BEFORE ColorManager
logger = setup_logging(APP_PATHS['logs'], APP_PATHS['reports'], APP_PATHS['profiles'])


# =========== END of logging_config.py ===========