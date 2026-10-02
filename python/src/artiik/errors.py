"""Exceptions raised by artiik."""


class ArtiikError(Exception):
    """Base class for every error artiik raises."""


class FormatError(ArtiikError, ValueError):
    """Provider data doesn't have the expected shape, or can't be sent in the target format.

    The message starts with the path to the offending value, for example
    ``messages[3].content[1].tool_use_id``.
    """


class ValidationError(ArtiikError, ValueError):
    """A conversation isn't a valid request for its provider.

    ``problems`` lists every issue found, each starting with the path to the
    offending entry, for example ``messages[4]``.
    """

    def __init__(self, problems: tuple[str, ...]) -> None:
        super().__init__(
            "the conversation can't be sent as it is:\n" + "\n".join(f"- {p}" for p in problems)
        )
        self.problems = problems


class BudgetError(ArtiikError):
    """The request can't fit the token budget, even after the guard dropped what it could.

    ``needed`` is the estimated size of the smallest request the guard could
    build, and ``budget`` the limit it had to fit.
    """

    def __init__(self, needed: int, budget: int) -> None:
        super().__init__(
            f"the request needs about {needed} tokens but the budget is {budget}, even after "
            "dropping every older turn and step; raise the budget, or shorten the current turn"
        )
        self.needed = needed
        self.budget = budget
