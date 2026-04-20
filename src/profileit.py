import cProfile
import time


def profileit(output_file=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            with pr:
                result = func(*args, **kwargs)  # Call the original function

            # Save the profiling data to a .prof file if output_file is provided
            if output_file:
                pr.dump_stats(output_file + "." + str(time.time_ns()))
            else:
                print("No output file specified. Profiling data not saved.")

            return result

        return wrapper

    return decorator
