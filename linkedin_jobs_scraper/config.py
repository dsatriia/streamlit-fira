import os
import logging


class Config:
    os.environ["LI_AT_COOKIE"] = "AQEDATL5DxIAUUHwAAABfX592GAAAAF9oopcYFYAtBvpaCQmA1JoOYCcQq42fC-dH5VC98o3D7-jcp8kRoL_uHsTdPJs5mX2oUlbC9wcrr_nT2FxDtNUVnhNdqMdXqMdt12Ur6IXxWQmtns7Yhf9MshR"
    LI_AT_COOKIE = os.environ['LI_AT_COOKIE'] if 'LI_AT_COOKIE' in os.environ else None
    print("debug baru")
    print(os.environ['LI_AT_COOKIE'])
    LOGGER_NAMESPACE = 'li:scraper'

    _level = logging.INFO

    if 'LOG_LEVEL' in os.environ:
        _level_env = os.environ['LOG_LEVEL'].upper().strip()

        if _level_env == 'DEBUG':
            _level = logging.DEBUG
        elif _level_env == 'INFO':
            _level = logging.INFO
        elif _level_env == 'WARN' or _level_env == 'WARNING':
            _level = logging.WARN
        elif _level_env == 'ERROR':
            _level = logging.ERROR
        elif _level_env == 'FATAL':
            _level = logging.FATAL

    LOGGER_LEVEL = _level
