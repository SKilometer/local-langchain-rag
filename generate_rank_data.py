from tqdm import tqdm
import pandas as pd
import ast
import os

# 读取带有（原始问题和排好序的重写问题以及得分）的文件，生成训练数据（query,score,question_original）
def generate_training_data(file_path):
    df = pd.read_csv(file_path)
    training_data = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        original_question = row['Original Question']
        rewritten_questions = ast.literal_eval(row['Sorted Rewritten Questions and Scores'])

        pairs = [(rewritten_question, score, original_question) for rewritten_question, score in rewritten_questions]
        training_data.extend(pairs)
    return training_data


input_csv_path = './sorted/sorted_liver_questions.csv'
output_csv_path = './training_data/liver_training_data.csv'  # shape: (327,3)

output_directory = os.path.dirname(output_csv_path)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

training_data = generate_training_data(input_csv_path)
output_df = pd.DataFrame(training_data, columns=['Rewritten Query', 'Score', 'Original Question'])
output_df.to_csv(output_csv_path, index=False)
print(output_df.head())
print("success save...")
