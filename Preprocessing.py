import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def data_preprocess(df):
    """
    Preprocess the whole DataFrame
    """
    # Read the data
    df = df.fillna('unknown')
    
    # List of data that need to be preprocessed
    interests = [
    "V-pop", "K-pop", "EDM", "art", "politics", "vlogging", "baseball",
    "football", "volleyball", "table tennis", "basketball", "skiing",
    "swimming", "skateboarding", "board games", "DIY", "cooking", "painting",
    "bowling", "ballet", "dancing", "singing", "yoga", "jogging", "anime",
    "food tour", "travel", "study", "gym", "korean food", "coffee", "tea",
    "badminton", "hiphop", "investment", "alcoholic", "karate", "taekwondo",
    "rap", "makeup", "movie", "netflix", "Esport"
    ]
    
    communication_style = ["Video call", "Phone call", "Text message", "In-person (face-to-face)"]
    
    looking_for = ["Study supporter", "Chit Chatting", "Share knowledge"]
    
    subjects = [
        "Calculus", "Linear Algebra", "Chemistry", "Physics", "Programming",
        "Artificial Intelligence", "Biology", "Practical laboratory",
        "Environment", "Health Care", "Electronic"
    ]
    
    study_locations = ["café", "home", "library", "bookstore", "online"]
    
    study_times = ["Early Morning", "Midday", "Evening", "Night", "Midnight", "Overnight"]
    
    personalities = [
        "INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"
    ]
    
    major = ['DS', 'ICT', 'Pharmacy', 'CH', 'CS', 'MST', 'BIT', 'MAT', 'FST',
        'AMS', 'EER', 'SIC', 'MET', 'AE', 'AES', 'ATE', 'SST']
    
    # Data Encoder
    interest_enc = MultiLabelBinarizer()
    communication_style_enc = MultiLabelBinarizer()
    looking_for_enc = MultiLabelBinarizer()
    subjects_enc = MultiLabelBinarizer()
    study_locations_enc = MultiLabelBinarizer()
    study_times_enc = MultiLabelBinarizer()
    personalities_enc = MultiLabelBinarizer()
    major_enc = MultiLabelBinarizer()
    
    interest_enc.fit([interests])
    communication_style_enc.fit([communication_style])
    looking_for_enc.fit([looking_for])
    subjects_enc.fit([subjects])
    study_locations_enc.fit([study_locations])
    study_times_enc.fit([study_times])
    personalities_enc.fit([personalities])
    major_enc.fit([major])
    
    interests_encoded = pd.DataFrame(interest_enc.transform(df['Interests'].str.split(', ')), columns=interest_enc.classes_)
    communication_style_encoded = pd.DataFrame(communication_style_enc.transform(df['Communication_Style'].str.split(', ')), columns=communication_style_enc.classes_)
    looking_for_encoded = pd.DataFrame(looking_for_enc.transform(df['Looking_for'].str.split(', ')), columns=looking_for_enc.classes_)
    subjects_encoded = pd.DataFrame(subjects_enc.transform(df['Favorite_Subject'].str.split(', ')), columns=subjects_enc.classes_)
    study_locations_encoded = pd.DataFrame(study_locations_enc.transform(df['Study_Location'].str.split(', ')), columns=study_locations_enc.classes_)
    study_times_encoded = pd.DataFrame(study_times_enc.transform(df['Study_Time'].str.split(', ')), columns=study_times_enc.classes_)
    personalities_encoded = pd.DataFrame(personalities_enc.transform(df['Personality'].str.split(', ')), columns=personalities_enc.classes_)
    major_encoded = pd.DataFrame(major_enc.transform(df['Major'].str.split(', ')), columns=major_enc.classes_)
    new_df = pd.concat([major_encoded, personalities_encoded, interests_encoded, communication_style_encoded, looking_for_encoded, subjects_encoded, study_locations_encoded, study_times_encoded], axis=1)

    return new_df

def transfer_data(data):
    """
    Args:
        data of 1 student (pd.DataFrame)
        
    Returns:
        transfered as input data for model (DataFrame)
    """
    
    # List of data that need to be preprocessed
    interests = [
    "V-pop", "K-pop", "EDM", "art", "politics", "vlogging", "baseball",
    "football", "volleyball", "table tennis", "basketball", "skiing",
    "swimming", "skateboarding", "board games", "DIY", "cooking", "painting",
    "bowling", "ballet", "dancing", "singing", "yoga", "jogging", "anime",
    "food tour", "travel", "study", "gym", "korean food", "coffee", "tea",
    "badminton", "hiphop", "investment", "alcoholic", "karate", "taekwondo",
    "rap", "makeup", "movie", "netflix", "Esport"
    ]
    
    communication_style = ["Video call", "Phone call", "Text message", "In-person (face-to-face)"]
    
    looking_for = ["Study supporter", "Chit Chatting", "Share knowledge"]
    
    subjects = [
        "Calculus", "Linear Algebra", "Chemistry", "Physics", "Programming",
        "Artificial Intelligence", "Biology", "Practical laboratory",
        "Environment", "Health Care", "Electronic"
    ]
    
    study_locations = ["café", "home", "library", "bookstore", "online"]
    
    study_times = ["Early Morning", "Midday", "Evening", "Night", "Midnight", "Overnight"]
    
    personalities = [
        "INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"
    ]
    
    major = ['DS', 'ICT', 'Pharmacy', 'CH', 'CS', 'MST', 'BIT', 'MAT', 'FST',
        'AMS', 'EER', 'SIC', 'MET', 'AE', 'AES', 'ATE', 'SST']
    
    data.fillna('unknown', inplace=True)
    
    # Data Encoder
    interest_enc = MultiLabelBinarizer()
    communication_style_enc = MultiLabelBinarizer()
    looking_for_enc = MultiLabelBinarizer()
    subjects_enc = MultiLabelBinarizer()
    study_locations_enc = MultiLabelBinarizer()
    study_times_enc = MultiLabelBinarizer()
    personalities_enc = MultiLabelBinarizer()
    major_enc = MultiLabelBinarizer()
    
    interest_enc.fit([interests])
    communication_style_enc.fit([communication_style])
    looking_for_enc.fit([looking_for])
    subjects_enc.fit([subjects])
    study_locations_enc.fit([study_locations])
    study_times_enc.fit([study_times])
    personalities_enc.fit([personalities])
    major_enc.fit([major])
    
    interests_encoded = pd.DataFrame(interest_enc.transform([str(data['Interests']).split(', ')]), columns=interest_enc.classes_)
    communication_style_encoded = pd.DataFrame(communication_style_enc.transform([str(data['Communication_Style']).split(', ')]), columns=communication_style_enc.classes_)
    looking_for_encoded = pd.DataFrame(looking_for_enc.transform([str(data['Looking_for']).split(', ')]), columns=looking_for_enc.classes_)
    subjects_encoded = pd.DataFrame(subjects_enc.transform([str(data['Favorite_Subject']).split(', ')]), columns=subjects_enc.classes_)
    study_locations_encoded = pd.DataFrame(study_locations_enc.transform([str(data['Study_Location']).split(', ')]), columns=study_locations_enc.classes_)
    study_times_encoded = pd.DataFrame(study_times_enc.transform([str(data['Study_Time']).split(', ')]), columns=study_times_enc.classes_)
    personalities_encoded = pd.DataFrame(personalities_enc.transform([str(data['Personality']).split(', ')]), columns=personalities_enc.classes_)
    major_encoded = pd.DataFrame(major_enc.transform([str(data['Major']).split(', ')]), columns=major_enc.classes_)
    new_df = pd.concat([major_encoded, personalities_encoded, interests_encoded, communication_style_encoded, looking_for_encoded, subjects_encoded, study_locations_encoded, study_times_encoded], axis=1)

    return new_df

if __name__ == "__main__":
    path = r"D:\B3\Group Project\USTHConnect\dataset.csv"
    df = pd.read_csv(path)
    data = transfer_data(df.iloc[0])
    print(data)