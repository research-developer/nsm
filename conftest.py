def pytest_addoption(parser):
    """Global stub for coverage options when pytest-cov is unavailable."""

    try:
        parser.addoption("--cov", action="store", default=None, help="stub option")
    except ValueError:
        pass

    try:
        parser.addoption(
            "--cov-report", action="append", default=[], help="stub option"
        )
    except ValueError:
        pass
