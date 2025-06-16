import os

test_dir = os.path.dirname(__file__)
test_output_dir = os.path.join(os.path.dirname(test_dir), "test_output")


def get_test_result_output_dir(func) -> str:
    """
    Returns a unique output directory for the given test function.

    For running profiling and timing tests, each function tested should have their own output directory under a common
    "test_output" directory.
    :param func: Function argument to get the name from
    :return: the absolute path of the output directory
    """
    if isinstance(func, str):
        func_name = func
    elif hasattr(func, "__name__"):
        func_name = func.__name__
    else:
        raise RuntimeError(f"Provided function argument should be a function or a string.")

    out_dir = os.path.join(test_output_dir, func_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return os.path.abspath(out_dir)
