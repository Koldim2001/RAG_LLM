import logging
import os
import time

logger_profile = logging.getLogger("profile")


def check_and_set_env_var(var_name, value_new):
    """
    Проверяет, установлена ли переменная окружения `var_name`. Если не установлена, 
    присваивает ей значение `value_new`
    """
    value = os.getenv(var_name)
    if value is None:
        os.environ[var_name] = str(value_new)
        print(f"Значение {value_new} сохранено в переменную окружения {var_name}.")
    else:
        print(f"Переменная {var_name} уже установлена: {value}")


def profile_time(func):
    def exec_and_print_status(*args, **kwargs):
        t_start = time.time()
        out = func(*args, **kwargs)
        t_end = time.time()
        dt_msecs = (t_end - t_start) * 1000

        self = args[0]
        logger_profile.debug(
            f"{self.__class__.__name__}.{func.__name__}, time spent {dt_msecs:.2f} msecs"
        )
        return out

    return exec_and_print_status