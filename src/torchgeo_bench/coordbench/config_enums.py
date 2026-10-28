#TODO refactor elsewhere!
from enum import Enum


class TemporalResolution(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    FOUR_WEEKS = "4_weeks"
    THIRTEEN_WEEKS = "13_weeks"
    DAYS_30 = "days_30"
    DAYS_365 = "days_365"
    MONTHLY = "monthly"
    YEARLY = "yearly"
