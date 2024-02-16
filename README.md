Install conda environment through the requirments.txt file or through https://mne.tools/stable/install/manual_install.html

To set up data:
  1) Download raws:
       https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS
  2) Extract raws to data/raw/eeg/<VP__>  and data/raw/nirs/<VP__> respectively
  3) Run src/format_data/convert_nirs_data.py


main.py is a terminal version run --help on the file to get a list of options to run

main_pipeline_search.ipynb is a jupiter notebook version for decoding
