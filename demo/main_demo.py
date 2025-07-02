# main_demo.py

import pandas as pd
import numpy as np
import json
import nltk  # Import nltk

# Import our modular components
from data_generator import generate_all_data
from data_analyzer import analyze_data


def run_demo():
    """
    Orchestrates the entire demo:
    1. Ensures NLTK data is downloaded.
    2. Generates 'Before' and 'After' datasets.
    3. Runs AI-powered analysis on the datasets.
    4. Prints a summarized report mimicking the app's dashboard.
    """
    print("--- Starting AI-Powered Employee-to-Customer Loop Demo ---")
    print(
        "This demo simulates your app's ability to connect internal employee experience"
    )
    print("to external customer satisfaction and project business ROI.")

    # --- Ensure NLTK VADER Lexicon is Available ---
    print("\nChecking for necessary AI data (NLTK VADER lexicon)...")
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
        print("VADER lexicon found. Proceeding.")
    except nltk.downloader.DownloadError:
        print(
            "VADER lexicon not found. Attempting to download now (requires internet connection)..."
        )
        try:
            nltk.download("vader_lexicon", quiet=True)
            print("VADER lexicon downloaded successfully!")
        except Exception as e:
            print(
                f"ERROR: Could not download VADER lexicon. Please ensure you have an internet connection."
            )
            print(
                f"You might need to run: python -c \"import nltk; nltk.download('vader_lexicon')\" manually."
            )
            print(f"Details: {e}")
            return  # Exit if critical data isn't available

    # --- 1. Generate Data ---
    # Using default numbers from data_generator (750 employees, 8000 customer reviews per phase)
    # You can customize these if needed:
    # df_emp_b, df_cust_b, df_emp_a, df_cust_a = generate_all_data(
    #     num_employees=500, num_customer_reviews=5000, random_seed=42
    # )
    df_employees_before, df_customers_before, df_employees_after, df_customers_after = (
        generate_all_data()
    )

    # --- 2. Analyze Data with AI ---
    print("\n--- AI Analyzing Data: Please Wait ---")
    analysis_results = analyze_data(
        df_employees_before, df_customers_before, df_employees_after, df_customers_after
    )
    print("\n--- AI Analysis Complete! Generating Report ---")

    # --- 3. Present Results (Mimicking App Dashboard) ---
    print("\n" + "=" * 70)
    print("                          AI ANALYTICS DASHBOARD")
    print("                         Employee-to-Customer Loop")
    print("=" * 70)

    print("\n--- Phase 1: Before Intervention (Initial State) ---")
    print(
        f"  Employee Job Satisfaction (Avg): {analysis_results['before']['employee_job_satisfaction_avg']:.2f} / 5.0"
    )
    print(
        f"  Employee Confidence in Product (Avg): {analysis_results['before']['employee_confidence_in_product_avg']:.2f} / 5.0"
    )
    print(
        f"  AI-Identified Employee Feedback Sentiment: {analysis_results['before']['employee_suggestions_sentiment']}"
    )
    if analysis_results["before"]["employee_suggestions_neg_themes"]:
        print(
            f"  Top Employee Pain Points (AI-Identified): {', '.join(analysis_results['before']['employee_suggestions_neg_themes'])}"
        )
    print(
        f"  Employee Management Sentiment (AI-Derived): {analysis_results['before']['employee_management_sentiment']}"
    )
    if analysis_results["before"]["employee_management_neg_themes"]:
        print(
            f"  Key Management Criticisms (AI-Found): {', '.join(analysis_results['before']['employee_management_neg_themes'])}"
        )
    print(
        f"  Top Systemic Detrimental Areas (AI-Aggregated): {', '.join([f'{k} ({v:.1f}%)' for k,v in analysis_results['before']['top_detrimental_areas'].items()])}"
    )

    print(
        f"\n  Customer Star Rating (Avg): {analysis_results['before']['customer_star_rating_avg']:.2f} / 5.0"
    )
    print(
        f"  Customer Net Promoter Score (NPS): {analysis_results['before']['customer_nps']:.1f}"
    )
    print(
        f"  AI-Identified Customer Review Sentiment: {analysis_results['before']['customer_overall_sentiment']}"
    )
    if analysis_results["before"]["customer_neg_themes"]:
        print(
            f"  Key Customer Dissatisfaction Themes (AI-Extracted): {', '.join(analysis_results['before']['customer_neg_themes'])}"
        )

    print("\n" + "-" * 70)
    print("\n--- Phase 2: After Intervention (Improved State) ---")
    print(
        f"  Employee Job Satisfaction (Avg): {analysis_results['after']['employee_job_satisfaction_avg']:.2f} / 5.0"
    )
    print(
        f"  Employee Confidence in Product (Avg): {analysis_results['after']['employee_confidence_in_product_avg']:.2f} / 5.0"
    )
    print(
        f"  AI-Identified Employee Feedback Sentiment: {analysis_results['after']['employee_suggestions_sentiment']}"
    )
    if analysis_results["after"]["employee_suggestions_pos_themes"]:
        print(
            f"  Top Employee Positive Feedback (AI-Identified): {', '.join(analysis_results['after']['employee_suggestions_pos_themes'])}"
        )
    print(
        f"  Employee Management Sentiment (AI-Derived): {analysis_results['after']['employee_management_sentiment']}"
    )
    if analysis_results["after"]["employee_management_pos_themes"]:
        print(
            f"  Key Management Praises (AI-Found): {', '.join(analysis_results['after']['employee_management_pos_themes'])}"
        )
    print(
        f"  Top Systemic Detrimental Areas (AI-Aggregated): {', '.join([f'{k} ({v:.1f}%)' for k,v in analysis_results['after']['top_detrimental_areas'].items()]) if analysis_results['after']['top_detrimental_areas'] else 'None (significant improvement!)'}"
    )

    print(
        f"\n  Customer Star Rating (Avg): {analysis_results['after']['customer_star_rating_avg']:.2f} / 5.0"
    )
    print(
        f"  Customer Net Promoter Score (NPS): {analysis_results['after']['customer_nps']:.1f}"
    )
    print(
        f"  AI-Identified Customer Review Sentiment: {analysis_results['after']['customer_overall_sentiment']}"
    )
    if analysis_results["after"]["customer_pos_themes"]:
        print(
            f"  Key Customer Satisfaction Themes (AI-Extracted): {', '.join(analysis_results['after']['customer_pos_themes'])}"
        )

    print("\n" + "=" * 70)
    print("                    AI-DRIVEN INSIGHTS & PROJECTIONS")
    print("=" * 70)

    print("\n--- AI's Core Correlation ---")
    print(analysis_results["ai_insights"]["core_correlation"])
    print(analysis_results["ai_insights"]["employee_to_customer_link"])

    print("\n--- Key Quantitative Shifts (AI-Identified) ---")
    for k, v in analysis_results["ai_insights"]["key_shifts"].items():
        print(f"  - {k}: {v}")

    print("\n--- Machine Learning Model Predictions ---")
    print(f"  Model Used: {analysis_results['ml_predictions']['model_used']}")
    print(
        f"  AI Predicted NPS (Before Intervention): {analysis_results['ml_predictions']['predicted_nps_before_intervention']:.1f}"
    )
    print(
        f"  AI Predicted NPS (After Intervention): {analysis_results['ml_predictions']['predicted_nps_after_intervention']:.1f}"
    )
    print(
        f"  AI Predicted NPS Lift (due to Employee Changes): +{analysis_results['ml_predictions']['predicted_nps_lift_by_model']:.1f} points"
    )
    print(
        f"  (Model coefficients: Job Sat Impact={analysis_results['ml_predictions']['model_coefficients']['job_satisfaction_impact']:.2f}, Confidence Impact={analysis_results['ml_predictions']['model_coefficients']['confidence_impact']:.2f})"
    )

    print("\n--- Projected ROI & Business Impact ---")
    print(
        f"  • Estimated Customer Retention Increase: +{analysis_results['roi_projection']['customer_retention_increase_pct']:.1f}%"
    )
    print(
        f"  • Observed NPS Increase: +{analysis_results['roi_projection']['nps_increase_points']:.1f} points"
    )
    print(
        f"  • Projected NPS Lift (from ML Model): +{analysis_results['roi_projection']['ml_predicted_nps_lift']:.1f} points"
    )
    print(
        f"  • Estimated Organic Sign-ups Increase per Quarter (due to improved NPS): +{analysis_results['roi_projection']['estimated_organic_signups_per_qtr']}"
    )
    print("\n" + "=" * 70)
    print(
        "Demo Complete. Your app provides actionable insights for tangible business growth!"
    )
    print("=" * 70)


if __name__ == "__main__":
    # --- Instructions for Setup ---
    print("To set up this demo on a new machine:")
    print("1. Ensure you have Python (3.8+) installed.")
    print("2. Navigate to the project directory in your terminal.")
    print("3. Install required libraries using: pip install -r requirements.txt")
    print("4. Then, simply run: python main_demo.py")
    print("----------------------------")

    run_demo()
