#Instruktioner:
#Börja med att skapa en virtuell miljö och installera dependencies genom att klistra in följande kodstycke i terminalen:


#För Mac IOS
#installera bibliotek
pip install -r requirements.txt

#För Windows

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install docling docling-core tqdm pandas
pip install -r requirements.txt
