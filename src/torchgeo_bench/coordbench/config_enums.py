#TODO refactor elsewhere!
from enum import Enum


class TemporalResolution(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    FOUR_WEEKS = "4_weeks"
    THIRTEEN_WEEKS = "13_weeks"
    MONTHLY = "monthly"
    SEASONALLY = "seasonally"
    YEARLY = "yearly"
    TWO_YEARS = "2_years"
    THREE_YEARS = "3_years"
    FOUR_YEARS = "4_years"
