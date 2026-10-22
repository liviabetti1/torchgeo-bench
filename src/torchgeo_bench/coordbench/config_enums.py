#TODO refactor elsewhere!
from enum import Enum


class TemporalResolution(str, Enum):
    """A benchmark's temporal granularity; compares and formats as its plain string value."""

    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly" #=2_weeks

    FOUR_WEEKS = "4_weeks" # usually, will use this vs. month 
    THIRTEEN_WEEKS = "13_weeks" # usually, will use this vs. season 

    MONTHLY = "monthly" # in case acually monthly (although number of days in month differs)
    SEASONALLY = "seasonally" # same here
    YEARLY = "yearly"