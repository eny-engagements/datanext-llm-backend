# AI Powered Data Management Accelerator POC
Streamlit app developed to demonstrate Generative AI based Data Management capabilities. This should serve as a reference for building data management agents for the EY Data NeXt platform.

Please set up the prerequisites by downloading binaries from the following links:
- **Graphviz**: [graphviz-12.2.1 EXE Installer](https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/12.2.1/windows_10_cmake_Release_graphviz-install-12.2.1-win64.exe)
- **Open Source Embedding Model**: [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main) *Note*: Please place it in a folder named "all-mpnet-base-v2" in your working directory.
- **Drivers and Clients**: Please download the ODBC and clients for the various RDBMS that **Discover.ai** will connect to.

Please install the POC accelerators on your EY machine by installing the following commands:
```powershell
py -m venv pocenv
./pocenv/Scripts/Activate
pip install -r requirements.txt
```

Run the accelerators on your local machine by executing the following commands:
```powershell
./pocenv/Scripts/Activate
py -m streamlit run app.py
```