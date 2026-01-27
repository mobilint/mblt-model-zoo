import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--mxq-path",
        action="store",
        default=None,
        help="Override default mxq_path for pipeline loading.",
    )
    parser.addoption(
        "--revision",
        action="store",
        default=None,
        help="Override model revision (e.g., W8).",
    )
    parser.addoption(
        "--embedding-weight",
        action="store",
        default=None,
        help="Path to custom embedding weights.",
    )


@pytest.fixture(scope="module")
def mxq_path(request):
    return request.config.getoption("--mxq-path")


@pytest.fixture(scope="module")
def revision(request):
    return request.config.getoption("--revision")


@pytest.fixture(scope="module")
def embedding_weight(request):
    return request.config.getoption("--embedding-weight")
