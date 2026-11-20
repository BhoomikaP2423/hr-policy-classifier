"""Category definitions and departmental routing rules."""

CATEGORIES = [
    "Leave & Attendance",
    "Compensation & Benefits",
    "Code of Conduct & Ethics",
    "Remote Work & IT Security",
    "Onboarding & Offboarding",
]

DEPARTMENT_ROUTING = {
    "Leave & Attendance": "HR Operations & Timekeeping Desk",
    "Compensation & Benefits": "Payroll, Rewards & Benefits Team",
    "Code of Conduct & Ethics": "Legal, Compliance & Ethics Committee",
    "Remote Work & IT Security": "Workplace IT & InfoSec Governance",
    "Onboarding & Offboarding": "Talent Acquisition & People Experience",
}

LABEL2ID = {label: idx for idx, label in enumerate(CATEGORIES)}
ID2LABEL = {idx: label for idx, label in enumerate(CATEGORIES)}
