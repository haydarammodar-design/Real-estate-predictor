import unittest

import pandas as pd
from pydantic import ValidationError

from src.app import DevelopmentFeasibilityInput, FinancialInput, PredictionInput, _development_feasibility, financial_analysis
from src.data_processor import COMPARABLE_FEATURES, add_prior_year_comparable_features


class DevelopmentFeasibilityTests(unittest.TestCase):
    def test_feasibility_uses_both_ces_and_floor_area_ratio(self):
        result = _development_feasibility(
            DevelopmentFeasibilityInput(
                land_area_sqm=500,
                ces=0.4,
                floor_area_ratio=1.2,
                floor_count=3,
                saleable_ratio_pct=82,
                construction_cost_per_gross_sqm=2200,
                land_purchase_price_eur=450000,
                land_acquisition_fee_pct=8,
                additional_costs_eur=150000,
                sale_price_per_saleable_sqm=6500,
                planning_source="PLUi test zone",
            )
        )
        self.assertEqual(result["surfaces"]["buildable_footprint_sqm"], 200.0)
        self.assertEqual(result["surfaces"]["gross_floor_area_sqm"], 600.0)
        self.assertEqual(result["surfaces"]["saleable_area_sqm"], 492.0)
        self.assertEqual(result["returns"]["developer_margin_eur"], 1242000.0)

    def test_financial_roi_includes_purchase_costs(self):
        result = financial_analysis(
            FinancialInput(
                purchase_price=300000,
                down_payment_pct=20,
                loan_rate=3.5,
                loan_term_years=20,
                monthly_rent=1500,
                monthly_expenses=200,
            )
        )
        self.assertEqual(result["total_investment"], 84000.0)
        self.assertAlmostEqual(result["cash_on_cash_roi"], -1.31, places=2)


class ComparableFeatureTests(unittest.TestCase):
    def test_current_year_prices_do_not_affect_prior_year_features(self):
        source = pd.DataFrame(
            {
                "date_mutation": ["2023-06-01", "2024-06-01", "2024-07-01"],
                "latitude": [48.8566, 48.8566, 48.8567],
                "longitude": [2.3522, 2.3522, 2.3523],
                "price": [200000, 250000, 9000000],
                "area_sqm": [50, 50, 50],
                "property_type": ["apartment", "apartment", "apartment"],
            }
        )
        baseline = add_prior_year_comparable_features(source, [2024])
        changed = source.copy()
        changed.loc[2, "price"] = 1
        comparison = add_prior_year_comparable_features(changed, [2024])
        pd.testing.assert_frame_equal(
            baseline.loc[1:, COMPARABLE_FEATURES].reset_index(drop=True),
            comparison.loc[1:, COMPARABLE_FEATURES].reset_index(drop=True),
        )
        self.assertGreater(baseline.loc[1, "comp_500m_sale_count"], 0)


class ApiInputValidationTests(unittest.TestCase):
    def test_property_prediction_rejects_coordinates_outside_france(self):
        with self.assertRaises(ValidationError):
            PredictionInput(
                area_sqm=60,
                rooms=2,
                latitude=40.7128,
                longitude=-74.0060,
                department="75",
                property_type="apartment",
            )


if __name__ == "__main__":
    unittest.main()
