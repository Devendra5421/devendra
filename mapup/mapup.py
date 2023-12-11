import pandas as pd
import numpy as np
dataset_1 = pd.read_csv("dataset-1.csv")
dataset_2 = pd.read_csv("dataset-2.csv")
dataset_3 = pd.read_csv("dataset-3.csv")
print(dataset_1.head(5))

#Task_1 Question 1: Car Matrix Generation

def generate_car_matrix(dataset_1):
    car_matrix = pd.pivot_table(dataset_1, values="car", index="id_1", columns="id_2")
    for i in range(len(car_matrix)):
        car_matrix.iloc[i, i] = 0
    return car_matrix

# Generate the car matrix
car_matrix = generate_car_matrix(dataset_1.copy())
# Handling the null values  in car matrix
car_matrix = car_matrix.fillna(0)
print(car_matrix)

#QUestion-2 Car Type Count Calculation
# Create a DataFrame from the provided data

def get_type_count(dataset_1):
  
    def map_car_type(car_value):
        if car_value <= 15:
            return "low"
        elif car_value <= 25:
            return "medium"
        else:
            return "high"

    dataset_1["car_type"] = dataset_1["car"].apply(map_car_type)

  # Calculate the count of occurrences for each car type
    type_counts = dataset_1["car_type"].value_counts().to_dict()

  # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

# Get the type count
type_counts = get_type_count(dataset_1.copy())

# Print the type count
print(type_counts)

#Question 3: Bus Count Index Retrieval
def get_bus_indexes(dataset_1):
  
  # Calculate the mean of the bus column
  mean_bus = dataset_1["bus"].mean()

  # Filter the DataFrame based on the condition
  filtered_df = dataset_1[dataset_1["bus"] > 2 * mean_bus]

  # Get the indices of the filtered DataFrame
  bus_indexes = list(filtered_df.index)

  # Sort the indices in ascending order
  bus_indexes.sort()

  return bus_indexes

# Get the bus indexes
bus_indexes = get_bus_indexes(dataset_1.copy())

# Print the bus indexes
print(bus_indexes)

#Question 4: Route Filtering

def filter_routes(dataset_1):

  # Calculate the average truck count for each route
    avg_truck_count = dataset_1.groupby('route')['truck'].mean()

  # Filter routes with an average truck count greater than 7
    filtered_routes = avg_truck_count[avg_truck_count > 7].index.tolist()

  # Sort the filtered routes
    filtered_routes.sort()

    return filtered_routes

# Get the filtered routes
filtered_routes = filter_routes(dataset_1.copy())

# Print the filtered routes
print(filtered_routes)

#Question 5: Matrix Value Modification

import pandas as pd

# Load the dataset
file_path = 'dataset-1.csv'
data = pd.read_csv(file_path)

# Generate the car matrix from the previous question

def generate_car_matrix(df):
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')
    for i in range(len(car_matrix)):
        car_matrix.iloc[i, i] = 0
    return car_matrix

car_matrix = generate_car_matrix(data)

# Define the function to modify the matrix

def multiply_matrix(df):
    # Apply the logic to the DataFrame
    df = df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    # Round the values to 1 decimal place
    df = df.round(1)
    return df

# Call the function and print the head of the modified DataFrame
modified_matrix = multiply_matrix(car_matrix)
print(modified_matrix)

#Question 6: Time Check

def has_incorrect_timestamps(dataset_2):

    def is_incomplete(data):
        start_time = pd.Timestamp(f"{data['startDay']} {data['startTime']}")
        end_time = pd.Timestamp(f"{data['endDay']} {data['endTime']}")

    # Check if the duration covers 24 hours
        duration = (end_time - start_time) / pd.Timedelta(hours=1)
        if duration < 24:
            return True

    # Check if timestamps span all 7 days of the week
        days_covered = (end_time - start_time).days
        if days_covered < 6:
            return True

    return False

  # Check for incorrect timestamps for each (id, id_2) pair
    return dataset_1.groupby(["id", "id_2"]).apply(is_incomplete)

# Check for incorrect timestamps
incorrect_timestamps = has_incorrect_timestamps(dataset_1)

print(incorrect_timestamps)

  #Task_2
  #Qs-1
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

# Load the dataset
file_path = 'dataset-3.csv'
df = pd.read_csv(file_path, encoding='UTF-8-SIG')

# Create a pivot table with id_start as rows, id_end as columns, and distance as values
# Fill missing values with an arbitrary large number to represent no direct path
pivot_df = df.pivot(index='id_start', columns='id_end', values='distance').fillna(1000000)

# Ensure the matrix is symmetric by filling NaNs with transposed values
pivot_df = pivot_df.combine_first(pivot_df.T)

# Set the diagonal to 0, as the distance from any node to itself is 0
np.fill_diagonal(pivot_df.values, 0)

# Convert the DataFrame to a sparse matrix and calculate the shortest paths
sparse_matrix = csr_matrix(pivot_df)
distance_matrix, predecessors = shortest_path(csgraph=sparse_matrix, directed=False, return_predecessors=True)

# Convert the resulting distance matrix back to a DataFrame
result_df = pd.DataFrame(distance_matrix, index=pivot_df.index, columns=pivot_df.columns)

# Show the head of the resulting DataFrame
print(result_df.head())

#Question-2

def unroll_distance_matrix(distance_df):
    # Unroll the distance matrix into a long DataFrame with id_start, id_end, and distance
    distance_long_df = distance_df.stack().reset_index()
    distance_long_df.columns = ['id_start', 'id_end', 'distance']
    # Remove rows where id_start is equal to id_end
    distance_long_df = distance_long_df[distance_long_df['id_start'] != distance_long_df['id_end']]
    return distance_long_df

# Apply the function to the result_df
unrolled_df = unroll_distance_matrix(result_df)

# Show the head of the unrolled DataFrame
print(unrolled_df.head())

#Question-3

def find_ids_within_ten_percentage_threshold(df, reference_id):
    # Calculate the average distance for the reference ID
    avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    # Define the 10% threshold
    threshold = 0.1 * avg_distance

    # Find IDs within the 10% threshold of the average distance
    ids_within_threshold = df[(df['distance'] >= avg_distance - threshold) &
                              (df['distance'] <= avg_distance + threshold)]['id_start'].unique()
    # Sort the list of IDs
    ids_within_threshold.sort()
    return ids_within_threshold.tolist()

# Test the function with a reference ID (assuming 1001400 is a valid ID in the DataFrame)
reference_id = 1001400
ids_list = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)

# Show the sorted list of IDs
print(ids_list)

#Question-4

def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient

    return df

# Apply the function to the unrolled DataFrame
toll_rates_df = calculate_toll_rate(unrolled_df)

# Show the head of the resulting DataFrame with toll rate
print(toll_rates_df.head())

#Question-5
from datetime import datetime, timedelta

# Assuming toll_rates_df is the DataFrame from the previous step
# Let's load the DataFrame first
toll_rates_df = pd.read_csv('dataset-3.csv')

def calculate_time_based_toll_rates(df):
    # Define time ranges and discount factors
    time_discounts_weekday = [
        (datetime.strptime('00:00:00', '%H:%M:%S').time(), datetime.strptime('10:00:00', '%H:%M:%S').time(), 0.8),
        (datetime.strptime('10:00:00', '%H:%M:%S').time(), datetime.strptime('18:00:00', '%H:%M:%S').time(), 1.2),
        (datetime.strptime('18:00:00', '%H:%M:%S').time(), datetime.strptime('23:59:59', '%H:%M:%S').time(), 0.8),
    ]
    time_discount_weekend = 0.7

    # Create a new DataFrame to store time-based toll rates
    time_based_df = pd.DataFrame()

    # Iterate over each unique (id_start, id_end) pair
    for (id_start, id_end), group in df.groupby(['id_start', 'id_end']):
        # Iterate over each day of the week
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            # Apply weekday or weekend discount factors
            if day in ['Saturday', 'Sunday']:
                for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                    group[vehicle_type] *= time_discount_weekend
            else:
                for start_time, end_time, discount in time_discounts_weekday:
                    # Apply discount factor to the vehicle type rates based on the time range
                    for vehicle_type in [ 'car', 'rv', 'bus', 'truck']:
                        # Assuming the discount is to be applied to the entire day for simplicity
                        group[vehicle_type] *= discount
            # Add the time range and day information to the group
            group['start_day'] = day
            group['end_day'] = day
            group['start_time'] = time_discounts_weekday[0][0]
            group['end_time'] = time_discounts_weekday[-1][1]
            # Append the modified group to the time-based DataFrame
            time_based_df = time_based_df.append(group)

    # Reset index of the resulting DataFrame
    time_based_df.reset_index(drop=True, inplace=True)
    return time_based_df

# Apply the function to the DataFrame
time_based_toll_rates_df = calculate_time_based_toll_rates(toll_rates_df)

# Show the head of the resulting DataFrame
print(time_based_toll_rates_df.head())