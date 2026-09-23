# import library
from cleantext import clean
import pandas as pd
import re
import os

# csv file with job listings

NUMBER_OF_ROWS = 4

file_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(file_dir, "..", "data", "Jobs_NYC_Postings_20260608.csv")


df = pd.read_csv(file_path, nrows=NUMBER_OF_ROWS)

#removes duplicates
df_cleaned = df.drop_duplicates()

#full cleaning pipeline for job description column
def irrelevant_section_cutting(job_description):
    #if the scraped description is not a string type, return nothing
    if not isinstance(job_description, str):
        return ""

    #gets rid of the whitespace around the descriptions
    formatted_text = job_description.strip()
    lowercased_text = formatted_text.lower()

    #list of phrases that would denote a section that would not be a part of the job description
  
    non_job_description_headers = ["about us", "benefits", "Explore Careers", "Job Security", "why you should work for us", 
                                   "to request reasonable accommodation to participate in the job application or interview process", 
                                   "contact", "to apply", "applications", "inclusive equal opportunity", "request accommodations", 
                                   "work from home policy", "our mission", "additional information", 
                                   "detailed information", "note", "legal work status", "preferred skills", "hours"]

    #where the text should be cut off
    current_cutoff_point = len(lowercased_text)

    #don't allow the cutoff point to go below this character count
    forbidden_cutoff_len = 10

    #loops through all the phrases in case one is detected
    for header in non_job_description_headers:

        #ensures headers aren't accidentally just a part of another word and case does not matter
        found = re.search(r'\b' + re.escape(header) + r'\b', lowercased_text.lower())

        #if a different header is found
        if found:
            if found.start() >= forbidden_cutoff_len:
                if found.start() < current_cutoff_point:
                    current_cutoff_point = found.start()
    #text that covers job description only, but potentially has weird character encodings
    relevant_content = lowercased_text[:current_cutoff_point]

    cleaned_text = clean(text=relevant_content,
          fix_unicode=True,
          to_ascii=True)

    return cleaned_text.strip()

#removes sections of the job description most likely not a part of the job description based on boilerplates/common terms
df_cleaned['cleaned descriptions'] = df_cleaned['Job Description'].apply(irrelevant_section_cutting)

print(df_cleaned['cleaned descriptions'])


# --------------------------------------------------------------------
# Outlier detection (Pedrocia)
# Flags rows rather than dropping them, so the team can decide
# downstream whether flagged rows should be excluded from the
#clustering step or kept as legitimate senior-role signal.
# ---------------------------------------------------------------------

def flag_placeholder_positions(df):
    placeholder_position_value = 9999
    #Flag rows where the number of Positions looks like a placeholder rather than a real count
    df = df.copy()
    df['positions_outlier_flag'] = df['# Of Positions'] >= placeholder_position_value
    return df

def flag_zero_salary(df):
    #Flag rows where Salary Range is From 0.
    df = df.copy()
    df['zero_salary_flag'] = df['Salary Range From'] == 0
    return df

def flag_salary_outliers(df):
    #IQR-based outlier flag, computed separately per Salary Frequency to avoid mixing Annual/Hourly/Daily.
    df = df.copy()
    df['salary_outlier_flag'] = False
    #
    salary_range_min = 0.25
    salary_range_max = 0.75
    scaling_factor = 1.5

    for freq, group in df.groupby('Salary Frequency'):
        q1, q3 = group['Salary Range From'].quantile([salary_range_min, salary_range_max])
        iqr = q3 - q1
        lower, upper = q1 - scaling_factor * iqr, q3 + scaling_factor * iqr
        mask = (group['Salary Range From'] < lower) | (group['Salary Range From'] > upper)
        df.loc[group.index[mask], 'salary_outlier_flag'] = True
    return df

df_cleaned = flag_placeholder_positions(df_cleaned)
df_cleaned = flag_zero_salary(df_cleaned)
df_cleaned = flag_salary_outliers(df_cleaned)

print(df_cleaned[['Job ID', 'positions_outlier_flag', 'zero_salary_flag', 'salary_outlier_flag']])




