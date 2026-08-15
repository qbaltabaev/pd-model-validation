# Synthetic underwriting data

The example dataset represents front-book credit-card applications observed at the underwriting decision date. Every row is one synthetic application. Features are assumed to be available at application time; `pd90_12m` is the subsequently observed performance outcome and must never be used as a model input.

## Outcome and sample

| Column | Definition |
|---|---|
| `application_id` | Synthetic row identifier with no source-system meaning |
| `sample` | Chronological `reference` pool used for development/holdout splitting, or later `current` out-of-time sample |
| `application_date` | Credit-card application date |
| `pd90_12m` | 1 if the account reached at least 90 days past due within 12 months after application; otherwise 0 |

## Application-time features

| Column | Definition |
|---|---|
| `annual_income` | Applicant's annual income |
| `employment_length_years` | Time in employment or self-employment |
| `employment_status` | Employment category |
| `housing_status` | Rent, own, or mortgage housing category |
| `requested_credit_limit` | Requested limit for the new credit card |
| `number_of_credit_cards` | Number of existing credit cards |
| `credit_card_balance` | Total outstanding credit-card balance |
| `credit_card_utilization` | Existing card balance divided by available card limit |
| `mortgage_payment` | Monthly mortgage payment; zero when absent |
| `total_loan_balance` | Total outstanding balance across loan products |
| `auto_loan_payment` | Monthly auto-loan payment; zero when absent |
| `deposit_balance` | Available deposit balance at application |
| `delinquencies_last_3_months` | Delinquency count in the previous 3 months |
| `delinquencies_last_6_months` | Delinquency count in the previous 6 months |
| `delinquencies_last_12_months` | Delinquency count in the previous 12 months |
| `recent_credit_inquiries` | Recent credit inquiry count |
| `months_since_oldest_credit_line` | Age of the oldest observed credit line |
| `debt_to_income_ratio` | Estimated monthly debt service divided by monthly income |
| `bankruptcy_flag` | 1 when a prior bankruptcy is present; otherwise 0 |

The 160 rows are generated from a fixed seed and include 120 reference and 40 current applications. The executable model uses the earliest 90 reference rows for development, the next 30 for holdout validation, and all 40 current rows for out-of-time assessment. Monetary values are synthetic unitless amounts. A few income and deposit values are deliberately missing to exercise preprocessing. The data contains no people, organizations, locations, currencies, or operational identifiers and must not be used for real credit decisions.
