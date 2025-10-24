from pathlib import Path

working_dir = Path('working_directory')
working_dir.mkdir(parents=True , exist_ok= True)

input_dir = working_dir / 'input'
input_dir.mkdir(parents=True, exist_ok=True)

import shutil
import os

source_path = "./Data/How_to_invest_money.txt"
destination_path = './working_directory/input/How_to_invest_money.txt'

shutil.copy(source_path,destination_path)

if os.path.exists(destination_path):
    print(f"파일이 {destination_path}에 성공적으로 복사되었습니다.")
else :
    print("파일 복사 실패")