import os
import shutil
import numpy as np

PATH_TO_FOLDERS = '/Users/mm/dev/super_resolution/eeg_fNIRs/shin_2017/data/raw/nirs'
MAT_FILE_NAME = 'probeInfo.mat'
MAT_FILE_PATH = os.path.join('./', MAT_FILE_NAME)
#SUBJECTS = np.arange(1,2).tolist()
SUBJECTS = [2]

if __name__ == "__main__":
    for subject in SUBJECTS:
        folder_path = os.path.join(PATH_TO_FOLDERS, f'VP{subject:03d}')

        for subdir, dirs, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith('.hdr'):
                    file_path = os.path.join(subdir, file_name)
                    with open(file_path, 'r') as file:
                        lines = file.readlines()

                    updated_lines = []
                    for line in lines:
                        if line.strip().startswith('NIRStar'):
                            updated_lines.append('NIRStar="15.0"\n')
                        else:
                            updated_lines.append(line)

                    with open(file_path, 'w') as file:
                        file.writelines(updated_lines)

                    target_file_path = os.path.join(subdir, MAT_FILE_NAME)
                    if not os.path.exists(target_file_path):
                        shutil.copy(MAT_FILE_PATH, target_file_path)
