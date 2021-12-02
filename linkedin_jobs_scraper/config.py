import os
import logging


class Config:
    os.environ["LI_AT_COOKIE"] = "AQEDATL5DxIFTuPsAAABfXs13I0AAAF9n0JgjVYAU4X6cGNAQ9I5jPwzbgTtL5rabjZxqEp6DuGvFo687mMp62zhM5EXlKKr1f6nPdBify_xjaDQ6YeEb48G-tCqH3WvNTv4BSZ2da9R_yT65Zpw8UPw"
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
