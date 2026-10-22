#TODO refactor elsewhere!
from enum import Enum


class TemporalResolution(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"