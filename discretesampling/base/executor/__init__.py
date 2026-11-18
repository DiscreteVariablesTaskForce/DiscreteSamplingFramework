from .executor import Executor  # noqa

try:
    from .executor_MPI import Executor_MPI  # noqa
except (ImportError, RuntimeError) as _mpi_import_error:
    # mpi4py or smccomponents is missing (ImportError), or mpi4py is installed
    # but finds no MPI library to load (RuntimeError, from mpi4py 4): either
    # way everything but MPI still works, e.g. on a cluster node set up for
    # serial runs only.
    _MPI_IMPORT_ERROR = _mpi_import_error

    class Executor_MPI:  # noqa: N801 -- stands in for the real class's name
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Executor_MPI needs mpi4py and smccomponents, which could not be "
                f"imported: {_MPI_IMPORT_ERROR}") from _MPI_IMPORT_ERROR
