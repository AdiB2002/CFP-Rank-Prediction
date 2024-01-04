import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# Function to read a CSV file and select specific columns
def read_csv_and_select_columns(file_path, column_names):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Select only the specified columns
        df_selected = df[column_names]

        return df_selected
    except Exception as e:
        return str(e)

selected_columns = ['Team', 'W-L', 'CFP']
df_2023_ranks = read_csv_and_select_columns('2023Ranks.csv', selected_columns)
df_2022_ranks = read_csv_and_select_columns('2022Ranks.csv', selected_columns)

selected_columns = ['School', 'Pct', 'Conf Pct', 'Off', 'Def', 'SRS', 'SOS']
df_2023_stats = read_csv_and_select_columns('2023Stats.csv', selected_columns)
df_2022_stats = read_csv_and_select_columns('2022Stats.csv', selected_columns)
df_2023_stats = df_2023_stats.rename(columns={'School': 'Team'})
df_2022_stats = df_2022_stats.rename(columns={'School': 'Team'})

# Mapping of different team names across data sources 
diff_map = {"Appalachian St":"Appalachian State", "Arizona St":"Arizona State", "Arkansas St":"Arkansas State", "Ball St":"Ball State", "Boise St":"Boise State", "C Michigan":"Central Michigan", "Coastal Car":"Coastal Carolina", "Colorado St":"Colorado State", "E Michigan":"Eastern Michigan", "FL Atlantic":"Florida Atlantic", "Florida Intl":"Florida International", "Florida St":"Florida State", "Fresno St":"Fresno State", "Ga Southern":"Georgia Southern", "Georgia St":"Georgia State", "Iowa St":"Iowa State", "Kansas St":"Kansas State", "Kent":"Kent State", "Miami FL":"Miami (FL)", "Miami OH":"Miami (OH)", "Michigan St":"Michigan State", "MTSU":"Middle Tennessee State", "Mississippi St":"Mississippi State", "N Illinois":"Northern Illinois", "NC State":"North Carolina State", "New Mexico St":"New Mexico State", "Ohio St":"Ohio State", "Oklahoma St":"Oklahoma State", "Oregon St":"Oregon State", "Penn St":"Penn State", "Mississippi":"Ole Miss", "Pittsburgh":"Pitt", "San Diego St":"San Diego State", "San Jose St":"San Jose State", "Southern Miss":"Southern Mississippi", "TCU":"Texas Christian", "Texas St":"Texas State", "ULM":"Louisiana-Monroe", "UNLV":"Nevada-Las Vegas", "UT San Antonio":"UTSA", "Utah St":"Utah State", "W Michigan":"Western Michigan", "Washington St":"Washington State", "Jacksonville St":"Jacksonville State", "Sam Houston St":"Sam Houston", 'WKU':'Western Kentucky'}

# Replace values in 'Team' column based on diff_map
df_2023_ranks['Team'] = df_2023_ranks['Team'].replace(diff_map)
df_2022_ranks['Team'] = df_2022_ranks['Team'].replace(diff_map)

# Merge DataFrames on 'Team' column
df1 = pd.merge(df_2023_ranks, df_2023_stats, on='Team', how='inner')
df2 = pd.merge(df_2022_ranks, df_2022_stats, on='Team', how='inner')
df = pd.concat([df1, df2], ignore_index=True)

# Convert categorical columns to numerical using LabelEncoder
label_encoders = {}
for column in ['Team']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split 'W-L' column into 'W' and 'L', convert to integers
df[['W', 'L']] = df['W-L'].str.split('-', expand=True).astype(int)

# Drop the original 'W-L' column 
df.drop('W-L', axis=1, inplace=True)

# Convert values in 'CFP' columns to integers
df['CFP'] = pd.to_numeric(df['CFP'], errors='coerce').fillna(26).astype(int)

# Replace NaN values in column Conf Pct with values from column Pct
df['Conf Pct'] = df.apply(lambda row: row['Pct'] if pd.isnull(row['Conf Pct']) else row['Conf Pct'], axis=1)

# Split dataset into features and target
X = df.drop('CFP', axis=1).values
y = df['CFP'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a PyTorch dataset
class MyDataset(Dataset):
    def __init__(self, features, target):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

# Create DataLoader
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model
input_size = X.shape[1]
hidden_size = 10
output_size = 1
model = NeuralNetwork(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

# Evaluation on test data
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()

print(f'Test Loss: {test_loss / len(test_loader)}')
