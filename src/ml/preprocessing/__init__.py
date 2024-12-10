from typing import Callable


def preprocessing_factory(name: str, **kwargs) -> Callable:
    """Factory function for preprocessing functions."""
    match name:
        case _:
            raise ValueError(f"Unknown preprocessing function {name}.")
