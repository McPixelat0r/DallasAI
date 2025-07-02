# data_generator.py

import pandas as pd
import numpy as np
from faker import Faker

def generate_all_data(
    num_employees: int = 750,
    num_customer_reviews: int = 8000,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates simulated 'Before' and 'After' datasets for employee feedback
    and customer reviews.

    Args:
        num_employees (int): Number of employee feedback records to generate per phase.
        num_customer_reviews (int): Number of customer review records to generate per phase.
        random_seed (int): Seed for reproducibility of random data generation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (df_employees_before, df_customers_before, df_employees_after, df_customers_after)
    """
    
    # Initialize Faker for generating realistic names (for employee IDs, if desired)
    fake = Faker()

    # Set a random seed for reproducibility using NumPy's random generator
    np.random.seed(random_seed)

    # --- Text Content Pools ---
    # Employee Feedback Text - Before
    employee_suggestions_bad = [
        "My tools are so slow, it takes ages to look up user issues. Customers get frustrated because of the wait.",
        "I feel like management doesn't really understand what we go through day-to-day. My suggestions are ignored.",
        "The training on new features is always rushed and unclear. I don't feel confident helping users.",
        "It's hard to stay motivated when I constantly hit technical roadblocks and outdated software.",
        "Sometimes I feel like my voice isn't heard when I bring up problems with our workflow.",
        "The internal communication about app updates is terrible, leading to confusion.",
        "Too much red tape to get anything done, especially software upgrades.",
        "Work-life balance is suffering due to inefficient processes.",
        "I'm confident in the app itself, but not the support system around it.",
        "Could really use better resources to handle complex customer queries.",
        "Burnout is real with these slow systems.",
        "Honestly, pretty demoralized with the current setup."
    ]

    employee_management_bad = [
        "Management is unresponsive to our needs.",
        "I feel undervalued by leadership.",
        "Lack of empathy from higher-ups.",
        "Management focuses on metrics, not on actual challenges.",
        "Communication from management is poor and infrequent."
    ]

    # Employee Feedback Text - After (Improved)
    employee_suggestions_good = [
        "The new support tools are fantastic! I can help users so much faster now, which makes customers happier.",
        "Management has been amazing lately; they actually listen to our feedback and make changes. I feel heard!",
        "The recent training sessions were incredibly helpful and clear. I feel much more confident.",
        "I feel much more confident helping users now that I have the right resources and updated tools.",
        "It's great to see our ideas being implemented. Really boosts morale and efficiency.",
        "Internal communication has vastly improved, keeping us all on the same page.",
        "Processes are smoother now, allowing us to focus on helping users.",
        "I'm proud to work here, the investment in us is clear.",
        "Great work-life balance and effective tools make a huge difference.",
        "Finally, a system that works for us and the customers!",
        "Feeling energized and effective in my role.",
        "Love the new collaborative environment."
    ]

    employee_management_good = [
        "Management is supportive and listens to our suggestions.",
        "I feel valued and appreciated by leadership.",
        "Great communication and clear direction from management.",
        "They genuinely care about our well-being and success.",
        "Proactive and empowering leadership."
    ]

    # Customer Review Text - Before
    customer_reviews_bad = [
        "Love the cat pictures, but support took forever to respond about my saved albums. Very frustrating.",
        "The app crashed and it was so hard to get help. The agent seemed unsure and slow to reply.",
        "It's a good app, but when I had a question, the text chat was really slow. Like, 10 minutes between replies.",
        "Okay app. Nothing special, especially with the slow customer service. Might look for another app.",
        "Cute cats. The agent was polite but couldn't fix my issue right away. Had to wait hours for a follow-up.",
        "Got stuck on a search filter, support was no help.",
        "Terrible support experience, took forever to get a simple answer.",
        "Customer service ruined an otherwise decent app.",
        "The app is fine, but the support chat is a nightmare.",
        "Had a simple question, but the agent seemed confused.",
        "Would have given 5 stars if the support wasn't so slow.",
        "Don't bother contacting support, it's useless."
    ]

    # Customer Review Text - After (Improved)
    customer_reviews_good = [
        "Wow! My issue was resolved instantly. The support agent was super friendly and knowledgeable. Best support ever!",
        "Best app ever! Had a question, and the text support was lightning fast and very helpful. Found my cat picture in seconds!",
        "Finally, great customer service! They fixed my problem quickly and politely. So impressed!",
        "Love this app. And their support team is top-notch, truly responsive and efficient.",
        "Fantastic! The agent walked me through finding my favorite cat breeds with no fuss. They knew their stuff.",
        "Quick and effective support, very happy!",
        "The best text support I've ever experienced, truly a game-changer.",
        "Couldn't ask for better service, highly recommend!",
        "Smooth and quick resolution, thank you Cat-tastic support!",
        "The agents are so knowledgeable and friendly now.",
        "Five stars for the app AND the amazing support team!",
        "My problem was solved before I even finished typing. Impressive!"
    ]

    # --- Helper Functions for Data Generation (nested to keep them within generate_all_data scope) ---

    def _generate_employee_data(num_records, phase="before"):
        data = []
        for _ in range(num_records):
            # Weighted choices for ordinal scores
            if phase == "before":
                job_satisfaction = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.30, 0.20, 0.10])
                confidence_in_product = np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.20, 0.30, 0.25, 0.15])
                
                # Select 1 or 2 detrimental areas with higher probability for "bad" ones
                num_areas = np.random.randint(1, 3) # 1 or 2
                detrimental_areas = np.random.choice(
                    ['Slow Support Tools/Software', 'Lack of Clear Training/Resources', 'Internal Communication', 'Workload', 'Other'],
                    size=num_areas, replace=False, # replace=False ensures unique choices if num_areas > 1
                    p=[0.45, 0.35, 0.10, 0.05, 0.05]
                ).tolist() # Convert numpy array to list for join

                suggestions_text = np.random.choice(employee_suggestions_bad)
                management_text = np.random.choice(employee_management_bad)
            else: # "after" phase
                job_satisfaction = np.random.choice([1, 2, 3, 4, 5], p=[0.00, 0.05, 0.10, 0.35, 0.50])
                confidence_in_product = np.random.choice([1, 2, 3, 4, 5], p=[0.00, 0.00, 0.05, 0.25, 0.70])

                # Select 1 or 2 detrimental areas with lower probability for "bad" ones, 'None' is highly likely
                num_areas = np.random.randint(1, 3) # 1 or 2
                detrimental_areas = np.random.choice(
                    ['Slow Support Tools/Software', 'Lack of Clear Training/Resources', 'Internal Communication', 'Workload', 'Other', 'None'],
                    size=num_areas, replace=False,
                    p=[0.05, 0.10, 0.10, 0.05, 0.05, 0.65]
                ).tolist()
                
                # Remove 'None' if other areas are also selected for a more realistic multi-choice scenario
                if 'None' in detrimental_areas and len(detrimental_areas) > 1:
                    detrimental_areas.remove('None')
                
                suggestions_text = np.random.choice(employee_suggestions_good)
                management_text = np.random.choice(employee_management_good)
            
            data.append({
                'employee_id': fake.uuid4(),
                'job_satisfaction': job_satisfaction,
                'suggestions_thoughts': suggestions_text,
                'confidence_in_product': confidence_in_product,
                'detrimental_areas': ', '.join(detrimental_areas),
                'sentiment_management': management_text
            })
        return pd.DataFrame(data)

    def _generate_customer_data(num_records, phase="before"):
        data = []
        for _ in range(num_records):
            if phase == "before":
                star_rating = np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.15, 0.25, 0.30, 0.20])
                review_text = np.random.choice(customer_reviews_bad)
            else: # "after" phase
                star_rating = np.random.choice([1, 2, 3, 4, 5], p=[0.01, 0.04, 0.10, 0.25, 0.60])
                review_text = np.random.choice(customer_reviews_good)
            
            data.append({
                'review_id': fake.uuid4(),
                'customer_id': fake.uuid4(),
                'star_rating': star_rating,
                'review_text': review_text,
                'timestamp': fake.date_time_between(start_date='-6m', end_date='now') # Faker uses Python's random internally for dates
            })
        return pd.DataFrame(data)

    # --- Generate the Datasets ---
    print(f"Generating {num_employees} employee records and {num_customer_reviews} customer reviews for 'Before' phase...")
    df_employees_before = _generate_employee_data(num_employees, phase="before")
    df_customers_before = _generate_customer_data(num_customer_reviews, phase="before")
    print("Generation complete for 'Before' phase.")

    print(f"\nGenerating {num_employees} employee records and {num_customer_reviews} customer reviews for 'After' phase...")
    df_employees_after = _generate_employee_data(num_employees, phase="after")
    df_customers_after = _generate_customer_data(num_customer_reviews, phase="after")
    print("Generation complete for 'After' phase.")

    return df_employees_before, df_customers_before, df_employees_after, df_customers_after

# This block allows you to test the data generation directly if you run this file.
if __name__ == "__main__":
    print("Running data_generator.py directly for testing purposes...")
    emp_b, cust_b, emp_a, cust_a = generate_all_data(
        num_employees=100, # Use smaller numbers for quick test
        num_customer_reviews=1000
    )
    print("\n--- Sample Employee Data (Before) ---")
    print(emp_b.head())
    print("\n--- Sample Customer Data (Before) ---")
    print(cust_b.head())
    print("\n--- Sample Employee Data (After) ---")
    print(emp_a.head())
    print("\n--- Sample Customer Data (After) ---")
    print(cust_a.head())
    print("\nData generation test complete.")