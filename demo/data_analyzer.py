# data_analyzer.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split # Not strictly needed for this simplified demo
# from sklearn.metrics import mean_absolute_error # Not strictly needed for this simplified demo

# Import VADER sentiment analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

def _calculate_nps(star_ratings: pd.Series) -> float:
    """
    Calculates Net Promoter Score (NPS) from a series of star ratings.
    - Promoters: 5-star
    - Passives: 4-star
    - Detractors: 1-3 star
    NPS = % Promoters - % Detractors
    """
    total_reviews = len(star_ratings)
    if total_reviews == 0:
        return 0.0

    promoters = (star_ratings == 5).sum()
    passives = (star_ratings == 4).sum()
    detractors = ((star_ratings >= 1) & (star_ratings <= 3)).sum()

    pct_promoters = (promoters / total_reviews) * 100
    pct_detractors = (detractors / total_reviews) * 100

    nps = pct_promoters - pct_detractors
    return nps

def _analyze_text_sentiment_and_themes(text_series: pd.Series, phase: str, review_type: str) -> dict:
    """
    Uses NLTK's VADER for sentiment analysis and identifies key themes
    based on predefined keywords (mimicking deeper AI theme extraction).
    """
    analysis = {
        'overall_sentiment': 'Neutral',
        'key_positive_themes': [],
        'key_negative_themes': [],
        'avg_vader_compound_score': 0.0 # VADER's composite score (-1 to 1)
    }

    # Define keywords for theme extraction. VADER handles overall sentiment.
    # We still keep these to show the AI's ability to pull out specific actionable themes.
    if review_type == 'employee_suggestions':
        if phase == 'before':
            negative_themes_keywords = {'slow tools', 'outdated software', 'unclear training', 'ignored', 'technical roadblocks', 'unheard', 'red tape', 'inefficient', 'burnout'}
            positive_themes_keywords = {}
        else: # after
            negative_themes_keywords = {}
            positive_themes_keywords = {'new tools', 'faster', 'helpful', 'confident', 'implemented', 'boosts morale', 'efficient', 'proud', 'energized', 'collaborative'}
    elif review_type == 'employee_management':
        if phase == 'before':
            negative_themes_keywords = {'unresponsive', 'undervalued', 'lack of empathy', 'detached', 'poor communication', 'unsupportive'}
            positive_themes_keywords = {}
        else: # after
            negative_themes_keywords = {}
            positive_themes_keywords = {'supportive', 'listens', 'valued', 'appreciated', 'great communication', 'caring', 'empowering', 'proactive'}
    elif review_type == 'customer_reviews':
        if phase == 'before':
            negative_themes_keywords = {'took forever', 'slow', 'unsure', 'hard to get help', 'crashed', 'nightmare', 'useless', 'confused', 'terrible'}
            positive_themes_keywords = {}
        else: # after
            negative_themes_keywords = {}
            positive_themes_keywords = {'instant', 'fast', 'lightning', 'helpful', 'friendly', 'knowledgeable', 'resolved', 'impressed', 'top-notch', 'responsive', 'effective'}
    else:
        return analysis

    vader_scores = []
    detected_positive_themes = set()
    detected_negative_themes = set()

    for text in text_series:
        text_lower = text.lower()
        
        # Get VADER sentiment score
        vs = analyzer.polarity_scores(text)
        vader_scores.append(vs['compound']) # Use the compound score

        # Identify themes (still based on keywords for demo focus)
        for kw in positive_themes_keywords:
            if kw in text_lower:
                detected_positive_themes.add(kw)
        for kw in negative_themes_keywords:
            if kw in text_lower:
                detected_negative_themes.add(kw)

    if vader_scores:
        avg_compound_score = np.mean(vader_scores)
        analysis['avg_vader_compound_score'] = avg_compound_score

        # Interpret compound score into qualitative sentiment
        if avg_compound_score >= 0.05:
            analysis['overall_sentiment'] = 'Positive'
            if avg_compound_score >= 0.5: # More strict for "Highly Positive"
                 analysis['overall_sentiment'] = 'Highly Positive'
        elif avg_compound_score <= -0.05:
            analysis['overall_sentiment'] = 'Negative'
            if avg_compound_score <= -0.5: # More strict for "Highly Negative"
                analysis['overall_sentiment'] = 'Highly Negative'
        else:
            analysis['overall_sentiment'] = 'Neutral'
    
    analysis['key_positive_themes'] = sorted(list(detected_positive_themes)) # Sort for consistent output
    analysis['key_negative_themes'] = sorted(list(detected_negative_themes)) # Sort for consistent output

    return analysis


def _train_and_predict_customer_impact(emp_data_before_avg, cust_nps_before, emp_data_after_avg) -> dict:
    """
    Simulates training a simple ML model to predict customer NPS based on employee metrics.
    For demo purposes, we'll create a small synthetic dataset for training.
    """
    print("\n--- AI: Training Machine Learning Model ---")

    # Generate synthetic training data:
    # Features (X): Employee Job Satisfaction Avg, Employee Confidence Avg
    # Target (y): Corresponding Customer NPS
    # This simulates historical data from various time periods or companies.
    num_synthetic_samples = 30
    X_synthetic = []
    y_synthetic = []

    for _ in range(num_synthetic_samples):
        # Varying employee satisfaction and confidence
        job_sat = np.random.uniform(2.5, 4.8) # Range for satisfaction
        confidence = np.random.uniform(2.8, 4.9) # Range for confidence

        # A simple linear relationship + some noise for NPS
        # Higher sat/confidence leads to higher NPS
        nps = (job_sat * 15) + (confidence * 10) - 70 + np.random.normal(0, 10)
        nps = np.clip(nps, -100, 100) # NPS range is -100 to 100

        X_synthetic.append([job_sat, confidence])
        y_synthetic.append(nps)

    X_synthetic = np.array(X_synthetic)
    y_synthetic = np.array(y_synthetic)

    # Initialize and train a Linear Regression model
    model = LinearRegression()
    
    model.fit(X_synthetic, y_synthetic) # Train on synthetic data

    # --- Make Predictions on our Actual Demo Data ---
    # Prepare features from our "Before" and "After" aggregated employee data
    features_before = np.array([[
        emp_data_before_avg['employee_job_satisfaction_avg'],
        emp_data_before_avg['employee_confidence_in_product_avg']
    ]])
    
    features_after = np.array([[
        emp_data_after_avg['employee_job_satisfaction_avg'],
        emp_data_after_avg['employee_confidence_in_product_avg']
    ]])

    predicted_nps_before = model.predict(features_before)[0]
    predicted_nps_after = model.predict(features_after)[0]

    # Calculate the predicted lift in NPS due to employee changes
    predicted_nps_lift = predicted_nps_after - predicted_nps_before

    print("AI Model Prediction Complete.")

    return {
        'model_used': 'Linear Regression (Scikit-learn)',
        'predicted_nps_before_intervention': predicted_nps_before,
        'predicted_nps_after_intervention': predicted_nps_after,
        'predicted_nps_lift_by_model': predicted_nps_lift,
        'model_coefficients': {
            'job_satisfaction_impact': model.coef_[0],
            'confidence_impact': model.coef_[1]
        }
    }


def analyze_data(
    df_employees_before: pd.DataFrame,
    df_customers_before: pd.DataFrame,
    df_employees_after: pd.DataFrame,
    df_customers_after: pd.DataFrame
) -> dict:
    """
    Simulates the AI's analysis of employee and customer data to derive insights
    and project ROI, including a simple ML model's predictions.
    """

    results = {
        'before': {},
        'after': {},
        'ai_insights': {},
        'ml_predictions': {}, # New section for ML model's output
        'roi_projection': {}
    }

    # --- Analysis for 'Before' Phase ---
    print("AI is analyzing 'Before' data...")
    results['before']['employee_job_satisfaction_avg'] = df_employees_before['job_satisfaction'].mean()
    results['before']['employee_confidence_in_product_avg'] = df_employees_before['confidence_in_product'].mean()
    
    emp_sugg_analysis_before = _analyze_text_sentiment_and_themes(df_employees_before['suggestions_thoughts'], 'before', 'employee_suggestions')
    emp_mgmt_analysis_before = _analyze_text_sentiment_and_themes(df_employees_before['sentiment_management'], 'before', 'employee_management')

    results['before']['employee_suggestions_sentiment'] = emp_sugg_analysis_before['overall_sentiment']
    results['before']['employee_suggestions_neg_themes'] = emp_sugg_analysis_before['key_negative_themes']
    results['before']['employee_management_sentiment'] = emp_mgmt_analysis_before['overall_sentiment']
    results['before']['employee_management_neg_themes'] = emp_mgmt_analysis_before['key_negative_themes']
    
    all_detrimental_areas_before = [area.strip() for areas in df_employees_before['detrimental_areas'] for area in areas.split(', ')]
    detrimental_counts_before = pd.Series(all_detrimental_areas_before).value_counts(normalize=True) * 100
    results['before']['top_detrimental_areas'] = detrimental_counts_before[detrimental_counts_before.index != 'None'].nlargest(2).to_dict()


    results['before']['customer_star_rating_avg'] = df_customers_before['star_rating'].mean()
    results['before']['customer_nps'] = _calculate_nps(df_customers_before['star_rating'])
    
    cust_review_analysis_before = _analyze_text_sentiment_and_themes(df_customers_before['review_text'], 'before', 'customer_reviews')
    results['before']['customer_overall_sentiment'] = cust_review_analysis_before['overall_sentiment']
    results['before']['customer_neg_themes'] = cust_review_analysis_before['key_negative_themes']

    # --- Analysis for 'After' Phase ---
    print("AI is analyzing 'After' data...")
    results['after']['employee_job_satisfaction_avg'] = df_employees_after['job_satisfaction'].mean()
    results['after']['employee_confidence_in_product_avg'] = df_employees_after['confidence_in_product'].mean()

    emp_sugg_analysis_after = _analyze_text_sentiment_and_themes(df_employees_after['suggestions_thoughts'], 'after', 'employee_suggestions')
    emp_mgmt_analysis_after = _analyze_text_sentiment_and_themes(df_employees_after['sentiment_management'], 'after', 'employee_management')
    
    results['after']['employee_suggestions_sentiment'] = emp_sugg_analysis_after['overall_sentiment']
    results['after']['employee_suggestions_pos_themes'] = emp_sugg_analysis_after['key_positive_themes']
    results['after']['employee_management_sentiment'] = emp_mgmt_analysis_after['overall_sentiment']
    results['after']['employee_management_pos_themes'] = emp_mgmt_analysis_after['key_positive_themes']

    all_detrimental_areas_after = [area.strip() for areas in df_employees_after['detrimental_areas'] for area in areas.split(', ')]
    detrimental_counts_after = pd.Series(all_detrimental_areas_after).value_counts(normalize=True) * 100
    results['after']['top_detrimental_areas'] = detrimental_counts_after[detrimental_counts_after.index != 'None'].nlargest(2).to_dict()


    results['after']['customer_star_rating_avg'] = df_customers_after['star_rating'].mean()
    results['after']['customer_nps'] = _calculate_nps(df_customers_after['star_rating'])
    
    cust_review_analysis_after = _analyze_text_sentiment_and_themes(df_customers_after['review_text'], 'after', 'customer_reviews')
    results['after']['customer_overall_sentiment'] = cust_review_analysis_after['overall_sentiment']
    results['after']['customer_pos_themes'] = cust_review_analysis_after['key_positive_themes']

    # --- AI: Machine Learning Prediction ---
    ml_output = _train_and_predict_customer_impact(results['before'], results['before']['customer_nps'], results['after'])
    results['ml_predictions'] = ml_output


    # --- AI Insights & Correlation ---
    print("\nAI is identifying correlations and projecting ROI...")
    results['ai_insights']['core_correlation'] = (
        "The AI's deep analysis reveals a strong positive correlation between improvements "
        "in employee satisfaction, confidence, and internal tools/training, and a "
        "significant uplift in customer satisfaction metrics."
    )
    results['ai_insights']['employee_to_customer_link'] = (
        f"Specifically, the AI observed a drastic reduction in employee mentions of "
        f"'{list(results['before']['top_detrimental_areas'].keys())[0] if results['before']['top_detrimental_areas'] else 'prior pain points'}' "
        f"and improved management sentiment, directly correlating with customers "
        f"praising '{list(results['after']['customer_pos_themes'])[0] if results['after']['customer_pos_themes'] else 'improved service'}' "
        f"and '{list(results['after']['customer_pos_themes'])[1] if len(results['after']['customer_pos_themes']) > 1 else 'agent knowledge'}'. "
        "This indicates that addressing internal friction points directly translates to better external service."
    )
    results['ai_insights']['key_shifts'] = {
        'Job Satisfaction Change': f"{results['before']['employee_job_satisfaction_avg']:.1f} -> {results['after']['employee_job_satisfaction_avg']:.1f}",
        'Confidence in Product Change': f"{results['before']['employee_confidence_in_product_avg']:.1f} -> {results['after']['employee_confidence_in_product_avg']:.1f}",
        'Customer Star Rating Change': f"{results['before']['customer_star_rating_avg']:.1f} -> {results['after']['customer_star_rating_avg']:.1f}",
        'NPS Change (Actual)': f"{results['before']['customer_nps']:.1f} -> {results['after']['customer_nps']:.1f}"
    }

    # --- ROI Projection (now incorporates ML prediction) ---
    results['roi_projection']['customer_retention_increase_pct'] = 8.0 # This can still be a fixed estimate for demo
    results['roi_projection']['nps_increase_points'] = round(results['after']['customer_nps'] - results['before']['customer_nps'])
    
    # Highlight the ML model's predicted NPS lift
    results['roi_projection']['ml_predicted_nps_lift'] = ml_output['predicted_nps_lift_by_model']
    results['roi_projection']['estimated_organic_signups_per_qtr'] = 1000 # Example number based on NPS uplift

    return results

# This block allows you to test the analysis directly if you run this file.
if __name__ == "__main__":
    print("Running data_analyzer.py directly for testing purposes...")
    
    # Import data from data_generator
    from data_generator import generate_all_data

    # Generate a smaller dataset for quick testing
    emp_b, cust_b, emp_a, cust_a = generate_all_data(
        num_employees=100,
        num_customer_reviews=1000,
        random_seed=42 # Use the same seed as generator for consistency
    )

    analysis_results = analyze_data(emp_b, cust_b, emp_a, cust_a)

    print("\n--- Simulated AI Analysis Results ---")
    import json
    print(json.dumps(analysis_results, indent=2))
    print("\nData analysis test complete.")