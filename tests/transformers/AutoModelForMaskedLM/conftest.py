import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--mxq-path",
        action="store",
        default=None,
        help="Override default mxq_path for pipeline loading.",
    )


@pytest.fixture(scope="module")
def mxq_path(request):
    return request.config.getoption("--mxq-path")
