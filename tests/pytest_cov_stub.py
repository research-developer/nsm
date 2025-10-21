def pytest_addoption(parser):
    """Provide no-op handlers for coverage flags when pytest-cov is unavailable."""

    parser.addoption("--cov", action="store", default=None, help="stub option")
    parser.addoption(
        "--cov-report", action="append", default=[], help="stub option"
    )
