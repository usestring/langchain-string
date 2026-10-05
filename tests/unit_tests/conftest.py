from pytest_socket import disable_socket


def pytest_runtest_setup() -> None:
    # asyncio's event loop needs a Unix socketpair; every network socket is refused.
    disable_socket(allow_unix_socket=True)
