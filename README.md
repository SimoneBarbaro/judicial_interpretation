# Judicial Interpretation
Replication code for my project in sequencing legal DNA. The code is also available at https://github.com/SimoneBarbaro/judicial_interpretation
## Usage

Before starting the programs, the requirements in requirements.txt must be satisfied. 

Then, mallet must be loaded into the project directory. You can run mallet_prepare.sh or otherwise download and unzip mallet as done in that file.

To launch the program:

python main.py [--data_path "path_to_data"] [--is_new_data true/false] [--result_dir "path_to_where_to_store_results"]

The arguments are optional, by default it will reproduce the results in the paper as long as you put the original dataset sc_opinions_meta_00_11_case_level.csv into the data/ folder. 

To produce results from the new dataset, --data_path parameter must point to the directory where the dataset is located, then --is_new_data parameter must be added to the con with values true.
