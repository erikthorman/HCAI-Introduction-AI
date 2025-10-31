#Instruktioner:

1. Make sure you are currently in your prefered environment.
2. Install and update dependencies by copying and pasting the following code into your terminal:

pip install -U pip
pip install docling docling-core tqdm pandas
pip install -r requirements.txt


3. To format your PDF files into Markdown or Json. Run script in terminal with following code(make sure to update the searchpath):
python batch_docling_extract.py \
  --in_dir "/searchpath/to/input" 
  --out_dir "/searchpath/to/output" 
  --format both --recursive


4. Run analysis_stine.py 
5. Run birst_analysis.py


