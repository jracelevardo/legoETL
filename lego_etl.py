import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Text, select
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import textwrap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL connection parameters
db_url = 'postgresql://postgres:littlegiant_28@localhost:5432/postgres'

# Create a SQLAlchemy engine
engine = create_engine(db_url)
metadata = MetaData()

# Define the table schema
lego_sets = Table('lego_sets', metadata,
    Column('set_id', String, primary_key=True),
    Column('name', Text),
    Column('year', Integer),
    Column('theme', Text),
    Column('subtheme', Text),
    Column('themeGroup', Text),
    Column('category', Text),
    Column('pieces', Integer),
)

# Create the table in the database (if not exists)
metadata.create_all(engine)

# Path to the CSV file
file_path = r'C:\Users\User\Documents\3rd Year Stuff\Second Sem\Data Science\LEGO+Sets\lego_sets.csv'

# Step 1: Extract
df = pd.read_csv(file_path)

# Step 2: Transform & Clean Data
# Replace NaN with None to handle PostgreSQL insertion
df = df.where(pd.notnull(df), None)

# Data Cleaning Functions
def clean_year(value):
    try:
        year = int(value)
        if year < 1970 or year > 2022:  # Year range validation
            return None  # Return None for out-of-range values
        return year
    except (ValueError, TypeError):
        return None

def clean_numeric(value):
    try:
        num = int(value)
        if num < 0 or num > 2147483647:  # PostgreSQL INTEGER range
            return None  # Return None for out-of-range values
        return num
    except (ValueError, TypeError):
        return None

# Apply data cleaning functions to relevant columns
df['year'] = df['year'].apply(clean_year)
df['pieces'] = df['pieces'].apply(clean_numeric)


# Step 3: Load
# Insert data into the table, excluding rows where 'pieces' is None
try:
    with engine.connect() as connection:
        transaction = connection.begin()  # Start a transaction
        for index, row in df.iterrows():
            if pd.notna(row['pieces']):  # Exclude rows where 'pieces' is NaN
                try:
                    insert_statement = lego_sets.insert().values(
                        set_id=row['set_id'],
                        name=row['name'],
                        year=row['year'],
                        theme=row['theme'],
                        subtheme=row['subtheme'],
                        themeGroup=row['themeGroup'],
                        category=row['category'],
                        pieces=row['pieces']
                    )
                    connection.execute(insert_statement)
                    logger.info(f"Inserted row {index}")
                except Exception as insert_error:
                    logger.error(f"Error inserting row {index}: {insert_error}")
                    transaction.rollback()  # Rollback transaction on error
                    raise  # Re-raise the exception to halt execution
        transaction.commit()  # Commit transaction after successful insertions
    logger.info(f"ETL process completed successfully.")
except Exception as e:
    logger.error(f"Error during ETL process: {str(e)}")


# Function to fetch data from the database
def fetch_data_from_db():
    try:
        with engine.connect() as connection:
            # Select all rows from the lego_sets table
            query = lego_sets.select()
            result = connection.execute(query)
            data = result.fetchall()
            return data
    except Exception as e:
        logger.error(f"Error fetching data from database: {str(e)}")
        return None 

# Fetch data from the database
loaded_data = fetch_data_from_db()

# Display the fetched data with row and column names and numbers
if loaded_data:
    # Get column names from the SQLAlchemy Table object
    column_names = lego_sets.columns.keys()
    
    # Display header with column names
    header = f"{'Row':<5} | " + " | ".join([f"{name:<20}" for name in column_names])
    print(header)
    print("-" * len(header))
    
    # Display each row with row number
    for idx, row in enumerate(loaded_data):
        row_data = f"{idx+1:<5} | " + " | ".join([f"{item}" for item in row])
        print(row_data)
    print()  # Add a line space after the fetched database is shown
else:
    print("Failed to fetch data from the database.")
    
# Finding Patterns: Exploratory Data Analysis / Association Rule Mining
# Compute top 10 themes by number of sets
top_themes = df['theme'].value_counts().head(10)

# Log data description excluding specified columns and rows where 'pieces' is null
columns_to_exclude = ['minifigs', 'agerange_min', 'US_retailPrice', 'bricksetURL', 'thumbnailURL', 'imageURL']
data_description = df.drop(columns=columns_to_exclude).dropna(subset=['pieces']).describe(include='all')

# Function to format description with aligned columns
def format_description(description):
    description_str = description.to_string()
    formatted_lines = []
    for line in description_str.split('\n'):
        parts = line.split()
        if len(parts) == 0:
            continue
        formatted_line = f"{parts[0]:<15}" + " ".join([f"{part:<20}" for part in parts[1:]])
        formatted_lines.append(formatted_line)
    return "\n".join(formatted_lines)

# Log the formatted data description
logger.info("Data Description:")
logger.info(format_description(data_description))


# Log top 10 themes by number of sets
logger.info("Top 10 Themes by Number of Sets:")
for idx, (theme, count) in enumerate(top_themes.items(), start=1):
    logger.info(f"{idx}. Theme: {theme:<20} Number of Sets: {count}")
print()  # Add a line space after the top 10 themes


# Distribution of sets by year
plt.figure(figsize=(10, 6))
sns.histplot(df['year'].dropna(), bins=30, kde=True)
plt.title('Distribution of LEGO Sets by Year')
plt.xlabel('Year')
plt.ylabel('Number of Sets')
plt.show()
logger.info('Displayed plot: Distribution of LEGO Sets by Year')

# Themes with the most sets
plt.figure(figsize=(10, 6))
sns.barplot(x=top_themes.values, y=top_themes.index, palette="viridis")
plt.title('Top 10 Themes by Number of Sets')
plt.xlabel('Number of Sets')
plt.ylabel('Theme')
plt.show()
logger.info('Displayed plot: Top 10 Themes by Number of Sets')

# Trends in set complexity (number of pieces) over the years
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='year', y='pieces', estimator='mean', ci=None, color="green")
plt.title('Trends in Set Complexity Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Number of Pieces')
plt.show()
logger.info('Displayed plot: Trends in Set Complexity Over the Years')

# Number of Sets Over Time for Selected Names
selected_names = ['Star Wars', 'City', 'Technic']
plt.figure(figsize=(12, 8))
for name in selected_names:
    subset = df[df['name'].str.contains(name, case=False)]
    sns.lineplot(data=subset, x='year', y='pieces', label=name)
plt.title('Number of Sets Over Time for Selected Names')
plt.xlabel('Year')
plt.ylabel('Number of Sets')
plt.legend()
plt.show()

# Most popular theme in each decade
# Function to determine decade based on year
def get_decade(year):
    return int(year // 10 * 10)
# Apply the function to create a 'decade' column
df['decade'] = df['year'].apply(get_decade)
# Group by decade and theme to find the most popular theme in each decade
decade_theme_counts = df.groupby(['decade', 'theme']).size().reset_index(name='count')
idx = decade_theme_counts.groupby(['decade'])['count'].transform(max) == decade_theme_counts['count']
top_theme_per_decade = decade_theme_counts[idx]
# Plotting the results
plt.figure(figsize=(12, 8))
sns.barplot(data=top_theme_per_decade, x='decade', y='count', hue='theme', dodge=False, palette='viridis')
plt.title('Most Popular Theme in Each Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Sets')
plt.xticks(rotation=45)
plt.legend(title='Theme', loc='upper right')
plt.tight_layout()
plt.show()

print()  # Add a line space after the EDA results

# Step 5: Apriori Analysis
# Encode categorical variables into boolean format
df_encoded = pd.get_dummies(df[['theme', 'subtheme', 'category']], drop_first=True)

# Apply Apriori algorithm
itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# Display frequent itemsets
logger.info("Frequent Itemsets:")
for idx, row in itemsets.iterrows():
    logger.info(f"Itemset: {list(row['itemsets'])}, Support: {row['support']:.2f}")

# Generate association rules
rules = association_rules(itemsets, metric="lift", min_threshold=1.0)

# Display association rules
logger.info("Association Rules:")
for idx, row in rules.iterrows():
    logger.info(f"Rule: {list(row['antecedents'])} -> {list(row['consequents'])}, Lift: {row['lift']:.2f}")

logger.info("EDA process and Apriori analysis completed successfully.")
print()  # Add a line space after the Apriori and EDA results


