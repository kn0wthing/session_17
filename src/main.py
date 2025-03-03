"""
Main module for the project.
"""


def hello_world() -> str:
    """Return a hello world message.

    Returns:
        str: A greeting message
    """
    return "Hello, World!"


def main() -> None:
    """Main function."""
    message = hello_world()
    print(message)


if __name__ == "__main__":
    main() 