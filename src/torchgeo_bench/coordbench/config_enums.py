#TODO refactor elsewhere!
from enum import Enum


class TemporalResolution(str, Enum):
    """A benchmark's temporal granularity; compares and formats as its plain string value."""

    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    FOUR_WEEKS = "4_weeks"
    THIRTEEN_WEEKS = "13_weeks"
    MONTHLY = "monthly"
    SEASONALLY = "seasonally"
    YEARLY = "yearly"