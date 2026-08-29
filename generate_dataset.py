

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 5000


def generate_credit_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(19, 70, n_samples)
    annual_income = rng.gamma(shape=6, scale=8000, size=n_samples) + 12000
    employment_length_years = np.clip(rng.normal(7, 5, n_samples), 0, 40)

    existing_debt = np.clip(
        annual_income * rng.uniform(0.02, 0.55, n_samples), 0, None
    )
    debt_to_income_ratio = existing_debt / annual_income

    num_credit_lines = rng.integers(0, 12, n_samples)
    num_late_payments_2yrs = rng.poisson(lam=np.clip(1.2 - annual_income / 100000, 0.05, None))
    num_late_payments_2yrs = np.clip(num_late_payments_2yrs, 0, 12)

    credit_utilization = np.clip(rng.beta(2, 3, n_samples), 0, 1)
    savings_balance = np.clip(rng.gamma(2, annual_income / 20, n_samples), 0, None)
    checking_balance = np.clip(rng.normal(annual_income / 15, annual_income / 20, n_samples), 0, None)

    loan_amount = np.clip(rng.gamma(3, 3000, n_samples), 500, 60000)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n_samples)

    has_bankruptcy = rng.binomial(1, 0.05, n_samples)
    home_ownership = rng.choice(["own", "mortgage", "rent"], n_samples, p=[0.25, 0.35, 0.40])
    purpose = rng.choice(
        ["car", "education", "home_improvement", "medical", "business", "debt_consolidation"],
        n_samples,
    )

    credit_history_years = np.clip(age - rng.integers(18, 26, n_samples), 0, None)

    # --- Underwriting-style latent "risk score" (higher = safer) ---
    risk_score = (
        1.15
        - 3.0 * debt_to_income_ratio
        - 0.35 * num_late_payments_2yrs
        - 2.2 * credit_utilization
        + 0.05 * credit_history_years
        + 0.00002 * annual_income
        + 0.02 * employment_length_years
        + 0.00001 * savings_balance
        - 1.6 * has_bankruptcy
        - 0.00003 * loan_amount
        + rng.normal(0, 0.6, n_samples)  # noise so the problem isn't trivially separable
    )

    prob_good = 1 / (1 + np.exp(-risk_score))
    creditworthy = rng.binomial(1, prob_good)

    df = pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income.round(2),
            "employment_length_years": employment_length_years.round(1),
            "existing_debt": existing_debt.round(2),
            "debt_to_income_ratio": debt_to_income_ratio.round(3),
            "num_credit_lines": num_credit_lines,
            "num_late_payments_2yrs": num_late_payments_2yrs,
            "credit_utilization": credit_utilization.round(3),
            "savings_balance": savings_balance.round(2),
            "checking_balance": checking_balance.round(2),
            "loan_amount": loan_amount.round(2),
            "loan_term_months": loan_term_months,
            "credit_history_years": credit_history_years,
            "has_bankruptcy": has_bankruptcy,
            "home_ownership": home_ownership,
            "purpose": purpose,
            "creditworthy": creditworthy,  # target: 1 = good credit risk, 0 = bad
        }
    )
    return df


if __name__ == "__main__":
    df = generate_credit_dataset()
    out_path = "data/credit_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["creditworthy"].value_counts(normalize=True).rename("class balance"))
